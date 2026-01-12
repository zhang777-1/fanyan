import os
import sys
sys.path.append(os.path.join(os.getcwd(), "fundiff"))
import json
import time
from tqdm import tqdm

from geoelectric_dataset import load_geoelectric_data, log_normalize_data, log_denormalize_data, pad_or_trim
from einops import repeat

import ml_collections
        
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

import jax
from jax import random
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)

from model import Encoder, Decoder
from model_utils import create_train_step, create_encoder_step, create_eval_step
from data_utils import generate_dataset, BaseDataset, BatchParser


def train_and_evaluate(config: ml_collections.ConfigDict):
    print(f"当前正在使用的模型: {config.model.model_name}")
    # -------------------
    # 0) Initialize wandb (only on main process)
    # -------------------
    if jax.process_index() == 0:
        wandb.init(
            project=config.wandb.project,
            tags=[config.wandb.tag] if hasattr(config.wandb, 'tag') else [],
            config=config.to_dict(),
            name=f"{config.model.model_name}_{config.dataset.num_samples}_samples"
        )
    
    # Initialize model
    encoder = Encoder(**config.model.encoder)
    decoder = Decoder(**config.model.decoder)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    # print(f"Number of devices: {num_devices}")
    # print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    train_step = create_train_step(encoder, decoder, mesh)
    eval_step = create_eval_step(encoder, decoder, mesh)
     
    # 读取数据
    train_data = pd.read_json('./train_data.json')
    num_samples = min(10000, len(train_data))
    train_data_sampled = train_data.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # 提取数据 - 使用正确的列名
    try:
        train_rho = np.array([np.array(train_data_sampled['rho'][i]) for i in range(len(train_data_sampled))])
        train_phase = np.array([np.array(train_data_sampled['phase'][i]) for i in range(len(train_data_sampled))])
        train_res = np.array([np.array(train_data_sampled['res'][i]) for i in range(len(train_data_sampled))])
    except KeyError as e:
        print(f"致命错误：训练数据 JSON 文件中缺少列名 {e}。请检查 train_data.json。")
        raise
    
    # 【检查原始数据范围】
    print("--- 原始目标电阻率 (Ω·m) 统计 ---")
    print(f"原始 train_res 最小值: {np.min(train_res)}")
    print(f"原始 train_res 最大值: {np.max(train_res)}")
    # 修复 nan 问题：检查并替换小于等于零的值（Log10 变换前必须保证数据 > 0）
    train_res[train_res <= 0] = 1e-6
    # train_res = np.log10(train_res)  # 注释掉，因为数据已经是log10值
    print(f"DEBUG: 目标变量 train_res 已经是Log10尺度，直接使用，形状: {train_res.shape}")
    print(f"Log10值范围: [{np.min(train_res):.4f}, {np.max(train_res):.4f}]")
    print(f"对应的原始电阻率范围: [10^{np.min(train_res):.4f}={10**np.min(train_res):.2f} Ω·m, "
          f"10^{np.max(train_res):.4f}={10**np.max(train_res):.2f} Ω·m]")

    # 标准化视电阻率数据 (Z-score标准化)
    rho_mean = np.mean(train_rho)
    rho_dev = np.std(train_rho)
    train_rho_N = (train_rho - rho_mean) / rho_dev

    # 标准化相位数据 (Z-score标准化)
    phase_mean = np.mean(train_phase)
    phase_dev = np.std(train_phase)
    train_phase_N = (train_phase - phase_mean) / phase_dev

    # 调整数据维度：先将数据pad/trim到目标长度，然后reshape
    input_size = config.dataset.num_sensors  # 使用配置中的传感器数量 (64)
    output_size = 50  # 输出序列长度为50
    
    # 使用 pad_or_trim 将每个样本调整到目标长度
    train_rho_N = pad_or_trim(train_rho_N, input_size)
    train_phase_N = pad_or_trim(train_phase_N, input_size)
    train_res = pad_or_trim(train_res, output_size)  # 输出调整为50
    
    # 调整维度为 (num_samples, input_size, 1)
    train_rho_N = train_rho_N.reshape(-1, input_size, 1)
    train_phase_N = train_phase_N.reshape(-1, input_size, 1)
    x_train_normalized = np.concatenate([train_rho_N, train_phase_N], axis=2)

    # ========== y不进行归一化，直接使用原始数据 ==========
    # 调整 y 的维度为 (num_samples, output_size, 1) = (num_samples, 50, 1)
    y_train = train_res.reshape(-1, output_size, 1)  # 直接使用原始数据，不归一化
    
    print(f"\n{'='*50}")
    print("输入输出形状信息")
    print(f"{'='*50}")
    print(f"输入 x_train_normalized 形状: {x_train_normalized.shape}")
    print(f"  - 说明: (样本数, 序列长度={input_size}, 通道数=2)")
    print(f"  - 通道: [rho(标准化后), phase(标准化后)]")
    print(f"输出 y_train 形状: {y_train.shape}")
    print(f"  - 说明: (样本数, 序列长度={output_size}, 通道数=1)")
    print(f"  - 通道: [res(原始数据, log10尺度, 未归一化)]")
    print(f"y数据范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"{'='*50}\n")
    
    # ========== 使用 train_test_split 划分训练集验证集 (80/20 划分) ==========
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train_normalized, y_train, test_size=0.2, random_state=42
    )

    print(f"训练集: {len(x_train_split)} 样本, 验证集: {len(x_val)} 样本")


    # 生成输出坐标（50个点，匹配输出维度）
    coords = np.linspace(0, 1, output_size)[:, None]  # 输出50个点

     # 创建训练集batch
    batch_coords = repeat(coords, "b d -> n b d", n=jax.device_count())
    batch = (batch_coords, y_train_split, y_train_split)
    batch = jax.tree.map(jnp.array, batch)
    batch = multihost_utils.host_local_array_to_global_array(batch, mesh, P("batch"))

    # 创建验证集batch
    batch_coords_val = repeat(coords, "b d -> n b d", n=jax.device_count())
    batch_val = (batch_coords_val, y_val, y_val)
    batch_val = jax.tree.map(jnp.array, batch_val)
    batch_val = multihost_utils.host_local_array_to_global_array(batch_val, mesh, P("batch"))

    batch = multihost_utils.host_local_array_to_global_array(
        batch, mesh, P("batch")
        )

    # Create checkpoint manager
    job_name = f"{config.model.model_name}"
    job_name += f"_{config.dataset.num_samples}_samples"

    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Training loop
    rng_key = jax.random.PRNGKey(1234)
    final_train_loss = None
    final_val_loss = None
    
    for step in range(config.training.max_steps):
        start_time = time.time()

        state, loss = train_step(state, batch)

        # Logging
        if step % config.logging.log_interval == 0:
            # Log metrics
            train_loss = loss.item()  # 保存训练损失值
            end_time = time.time()
            val_loss = eval_step(state, batch_val)# 然后计算验证损失
            val_loss = val_loss.item()
            
            # 保存最终损失值
            final_train_loss = train_loss
            final_val_loss = val_loss

            log_dict = {"train_loss": train_loss, "val_loss": val_loss, "lr": lr(step)}
            if jax.process_index() == 0:
                print("step: {}, train_loss: {:.3e}, val_loss: {:.3e}, time: {:.3e}".format(
            step, train_loss, val_loss, end_time - start_time))
                # Log to wandb
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr(step),
                    "step": step,
                    "time_per_step": end_time - start_time
                }, step=step)

        # Save checkpoint
        if step % config.saving.save_interval == 0:
            if jax.process_index() == 0:
                train_loss = loss.item()
                save_checkpoint(ckpt_mngr, state)
                # 构建检查点文件路径（orbax通常以step命名目录）
                ckpt_file_path = os.path.join(ckpt_path, str(step))
                print(f"Saving checkpoint at step {step}, train_loss: {train_loss:.3e}")
                print(f"   检查点文件: {ckpt_file_path}")
                # Log checkpoint save to wandb
                wandb.log({
                    "checkpoint_saved": True,
                    "checkpoint_step": step
                }, step=step)
            else:
                save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break


    # -------------------
    # 保存测试数据（y不再归一化）
    # -------------------
    if jax.process_index() == 0:
        print("\n" + "="*50)
        print("保存测试数据")
        print("="*50)
    
        # 保存验证集作为测试集
        test_data_path = os.path.join(os.getcwd(), job_name, "test_data.npz")
        np.savez(test_data_path,
                 x_test=x_val,      # 验证集作为测试集
                 y_test=y_val)      # y_test 是原始数据，未归一化
        print(f"✅ 测试集保存到: {test_data_path}")
        print(f"   测试集形状: x_test{x_val.shape}, y_test{y_val.shape}")
        print(f"   y_test 数据范围: [{y_val.min():.3f}, {y_val.max():.3f}]")
    
        print("="*50)
    
    # Save final checkpoint
    print("\n" + "="*50)
    print("训练完成，保存最终模型...")
    print("="*50)
    
    if jax.process_index() == 0:
        if final_train_loss is not None:
            print(f"最终训练损失: {final_train_loss:.3e}")
        if final_val_loss is not None:
            print(f"最终验证损失: {final_val_loss:.3e}")
        print(f"模型检查点保存路径: {ckpt_path}")
        
        # Log final metrics to wandb
        if final_train_loss is not None:
            wandb.log({"final_train_loss": final_train_loss})
        if final_val_loss is not None:
            wandb.log({"final_val_loss": final_val_loss})
    
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()
    
    if jax.process_index() == 0:
        # 构建最终检查点文件路径（使用state.step获取当前步数）
        final_step = int(state.step) if hasattr(state, 'step') else (config.training.max_steps - 1)
        final_ckpt_file = os.path.join(ckpt_path, str(final_step))
        print(" 最终模型已保存完成！")
        print(f"   检查点目录: {ckpt_path}")
        print(f"   检查点文件: {final_ckpt_file}")
        print(f"   可通过 restore_checkpoint 函数加载模型进行推理")
        print("="*50)
        
        # Finish wandb run
        wandb.finish()



