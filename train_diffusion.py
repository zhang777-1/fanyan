import os
import json
import time
import copy

from einops import repeat

import ml_collections
# 设置 matplotlib 使用非 GUI 后端，避免 tkinter 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wandb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import jax
import jax.numpy as jnp
from jax import random, jit, lax
from functools import partial

from jax.experimental import mesh_utils, multihost_utils
try:
    from jax.shard_map import shard_map
except ImportError:
    # 兼容旧版本 JAX
    from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    create_diffusion_state,
    compute_total_params,
)
from function_diffusion.utils.train_utils import (
    create_train_diffusion_step,
    get_diffusion_batch,
    sample_ode,
    create_end_to_end_eval_step,
    create_autoencoder_eval_step,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
    restore_fae_state
)

from model import DiT, Encoder, Decoder, DiffusionWrapper, ModelParamsAdapter
from model_utils import create_encoder_step, create_eval_step
from data_utils import generate_dataset
from geoelectric_dataset import log_normalize_data, log_denormalize_data, pad_or_trim


def _plot_inversion_result(y_true, y_pred, step, save_path, y_min=None, y_max=None, depth_range=(0, 1200), resistivity_range=(0, 200)):
    """
    绘制反演结果可视化：真实曲线 vs 预测曲线
    使用和 MT1D_CNN_v1.py 相同的绘图风格
    """
    # y_true, y_pred shape: (batch, seq_len, 1)
    # 取第一个样本来画
    y_true = np.array(y_true[0, :, 0]) if y_true.ndim == 3 else np.array(y_true[0, :])
    y_pred = np.array(y_pred[0, :, 0]) if y_pred.ndim == 3 else np.array(y_pred[0, :])
    
    # y 不再归一化，直接使用原始数据（不需要反归一化）
    
    # 创建深度坐标
    num_points = len(y_true)
    depth_points = np.linspace(depth_range[0], depth_range[1], num_points)
    
    plt.figure(figsize=(10, 6))

    # 绘制曲线：使用和 MT1D_CNN_v1.py 相同的样式
    plt.plot(depth_points, y_true, linestyle='-', linewidth=2, color='blue', label='True $\\rho$ (Ω·m)')
    plt.plot(depth_points, y_pred, linestyle='--', linewidth=2, color='red', label='Predicted $\\rho$ (Ω·m)')

    # 使用对数 Y 轴（和 MT1D_CNN_v1.py 相同）
    plt.yscale('log')
    
    # 动态设置Y轴范围（和 MT1D_CNN_v1.py 相同）
    all_values = np.concatenate([y_true, y_pred])
    y_min_plot = max(0.01, np.min(all_values) * 0.8)  # 使用0.01作为最小边界，乘以0.8留出20%边距
    y_max_plot = min(10000, np.max(all_values) * 1.2)  # 使用10000作为最大边界，乘以1.2留出20%边距
    plt.ylim(y_min_plot, y_max_plot)

    # 设置坐标轴范围和标签
    plt.xlim(depth_range)
    
    # 设置自定义刻度（和 MT1D_CNN_v1.py 相同）
    plt.xticks(np.arange(depth_range[0], depth_range[1] + 1, 400))
    
    # 设置标签和标题
    plt.xlabel('Depth (m)', fontsize=10)
    plt.ylabel('Resistivity (Ω·m)', fontsize=10)
    
    # 计算RMSE用于标题
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    plt.title(f'Inversion Result (Step {step})\nValidation RMSE: {rmse:.4f}', fontsize=12)
    
    plt.grid(True, which='both', alpha=0.5)
    plt.legend(fontsize=10)

    # 保存文件（和 MT1D_CNN_v1.py 相同）
    img_path = os.path.join(save_path, f"inversion_step_{step}.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved inversion plot: {img_path}")


def plot_loss_curve(steps, train_losses, test_losses=None, test_steps=None, save_path=None):
    """
    绘制loss曲线并保存
    Args:
        steps: 训练步数列表
        train_losses: 训练loss列表
        test_losses: 测试loss列表（可选）
        test_steps: 测试loss对应的步数列表（可选）
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    
    if test_losses is not None and test_steps is not None:
        plt.plot(test_steps, test_losses, 'r-', linewidth=2, label='Test Loss', alpha=0.7)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.yscale('log')  # 使用对数刻度，因为loss通常变化很大
    
    # 保存图像
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "evaluation_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loss_plot_path = os.path.join(save_path, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Loss曲线已更新: {loss_plot_path}")


def plot_inversion_result(y_true, y_pred, step, save_path, y_min=None, y_max=None, depth_range=(0, 1200), resistivity_range=(0, 200)):
    """
    绘制反演结果可视化：真实曲线 vs 预测曲线
    使用和 MT1D_CNN_v1.py 相同的绘图风格
    """
    # 数据验证和预处理
    print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')
    try:
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 检查数据有效性
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            print(f"警告：步骤 {step} 的数据包含NaN值，跳过绘图")
            return
            
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            print(f"警告：步骤 {step} 的数据包含无穷值，跳过绘图")
            return
        
        # 处理数据形状：确保是1D数组
        if y_true.ndim == 3:
            y_true = y_true[0, :, 0]  # 取第一个样本的第一个通道
        elif y_true.ndim == 2:
            y_true = y_true[0, :]  # 取第一个样本
        elif y_true.ndim == 1:
            pass  # 已经是1D
        else:
            print(f"警告：y_true的形状 {y_true.shape} 不支持，跳过绘图")
            return
            
        if y_pred.ndim == 3:
            y_pred = y_pred[0, :, 0]  # 取第一个样本的第一个通道
        elif y_pred.ndim == 2:
            y_pred = y_pred[0, :]  # 取第一个样本
        elif y_pred.ndim == 1:
            pass  # 已经是1D
        else:
            print(f"警告：y_pred的形状 {y_pred.shape} 不支持，跳过绘图")
            return
        
        # 检查数据长度是否一致
        if len(y_true) != len(y_pred):
            print(f"警告：y_true长度({len(y_true)})与y_pred长度({len(y_pred)})不一致")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # y 不再归一化，直接使用原始数据（不需要反归一化）
        
        # 检查反归一化后的数据范围
        y_true_range = (np.min(y_true), np.max(y_true))
        y_pred_range = (np.min(y_pred), np.max(y_pred))
        print(f"步骤 {step} - 真实值范围: {y_true_range}, 预测值范围: {y_pred_range}")
        
        # 确保数据都是正数（对数刻度要求）
        y_true = np.maximum(y_true, 1e-6)
        y_pred = np.maximum(y_pred, 1e-6)
        
        # 创建深度坐标
        num_points = len(y_true)
        depth_points = np.linspace(depth_range[0], depth_range[1], num_points)
        
        plt.figure(figsize=(10, 6))

        # 绘制曲线：使用和 MT1D_CNN_v1.py 相同的样式
        plt.plot(depth_points, y_true, linestyle='-', linewidth=2, color='blue', label='True $\\rho$ (Ω·m)')
        plt.plot(depth_points, y_pred, linestyle='--', linewidth=2, color='red', label='Predicted $\\rho$ (Ω·m)')

        # 使用对数 Y 轴（和 MT1D_CNN_v1.py 相同）
        plt.yscale('log')
        
        # 动态设置Y轴范围（和 MT1D_CNN_v1.py 相同）
        all_values = np.concatenate([y_true, y_pred])
        y_min_plot = max(0.01, np.min(all_values) * 0.8)  # 使用0.01作为最小边界，乘以0.8留出20%边距
        y_max_plot = min(10000, np.max(all_values) * 1.2)  # 使用10000作为最大边界，乘以1.2留出20%边距
        plt.ylim(y_min_plot, y_max_plot)

        # 设置坐标轴范围和标签
        plt.xlim(depth_range)
        
        # 设置自定义刻度（和 MT1D_CNN_v1.py 相同）
        plt.xticks(np.arange(depth_range[0], depth_range[1] + 1, 400))
        
        # 设置标签和标题
        plt.xlabel('Depth (m)', fontsize=10)
        plt.ylabel('Resistivity (Ω·m)', fontsize=10)
        
        # 计算RMSE用于标题
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        plt.title(f'Inversion Result (Step {step})\nValidation RMSE: {rmse:.4f}', fontsize=12)
        
        plt.grid(True, which='both', alpha=0.5)
        plt.legend(fontsize=10)

        # 保存文件（和 MT1D_CNN_v1.py 相同）
        img_path = os.path.join(save_path, f"inversion_step_{step}.png")
        plt.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved inversion plot: {img_path}")
            
    except Exception as e:
        print(f"绘图过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def train_and_evaluate(config: ml_collections.ConfigDict):
    # -------------------
    # 0) Initialize wandb (only on main process)
    # -------------------
    if jax.process_index() == 0:
        wandb.init(
            project=config.wandb.project,
            tags=[config.wandb.tag] if hasattr(config.wandb, 'tag') else [],
            config=config.to_dict(),
            name=f"{config.diffusion.model_name}_{config.dataset.num_samples}_samples"
        )
    
# -------------------
    # 1) Initialize autoencoder (load checkpoint if available)
    # -------------------
    encoder = Encoder(**config.autoencoder.encoder)
    decoder = Decoder(**config.autoencoder.decoder)
    
    # 构造检查点路径
    fae_job = f"{config.autoencoder.model_name}_{config.dataset.num_samples}_samples"
    fae_ckpt_path = os.path.join(os.getcwd(), fae_job, "ckpt")

    print("\n" + "="*50)
    print("加载自编码器模型 (强制单通道模式)")
    print("="*50)
    print(f"检查点路径: {fae_ckpt_path}")

    dummy_y = jnp.zeros((1, 50, 1)) 
    
    # 计算 Latent 形状 (用于解码器初始化)
    patch_size = config.autoencoder.encoder.patch_size
    # 兼容 int 或 tuple 格式
    p_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
    latent_len = 50 // p_size
    dummy_latent = jnp.zeros((1, latent_len, config.autoencoder.encoder.emb_dim))

    print("正在初始化 FAE 参数结构 (Target: 1 channel)...")
    rng = random.PRNGKey(0)
    rng, key_enc, key_dec = random.split(rng, 3)
    
    enc_variables = encoder.init(key_enc, dummy_y)
    dec_variables = decoder.init(key_dec, dummy_latent, dummy_y)
    
    initial_params = (enc_variables['params'], dec_variables['params'])
    
    _, tx_fae = create_optimizer(config) 
    
    from flax.training import train_state
    # 创建 State 空壳
    fae_state = train_state.TrainState.create(
        apply_fn=decoder.apply,
        params=initial_params, # 这里放入了正确的 1 通道参数空壳
        tx=tx_fae
    )

    # 创建 Checkpoint Manager
    ckpt_mngr = create_checkpoint_manager(config.saving, fae_ckpt_path)
    
    step = ckpt_mngr.latest_step()

    if step is not None:
        raw_restored = ckpt_mngr.restore(step)
        
        if 'params' in raw_restored:
            params_dict = raw_restored['params']
        elif 'model' in raw_restored and 'params' in raw_restored['model']:
            params_dict = raw_restored['model']['params']
        elif 'state' in raw_restored and 'params' in raw_restored['state']:
            params_dict = raw_restored['state']['params']
        else:
            print(f"警告: 检查点结构未知，尝试直接读取 keys: {raw_restored.keys()}")
            params_dict = raw_restored.get('params', raw_restored)

        fae_state = fae_state.replace(
            params=params_dict,
            step=step
        )
        
        print(f"成功加载自编码器检查点！加载步数: {step}")
        
    else:
        err_msg = (
            f"致命错误：在 {fae_ckpt_path} 未找到 FAE 检查点！\n"
            "Diffusion 模型依赖预训练的自编码器。\n"
            "请先运行 'python train_autoencoder.py'。"
        )
        raise RuntimeError(err_msg)
    
    print("="*50 + "\n")

     # -------------------
    # 2) Initialize diffusion model (Wrapper: CondEncoder + DiT)
    # -------------------
    # 条件编码器配置
    use_conditioning = True
    print("\n" + "="*50)
    print("初始化扩散模型 (Diffusion Model)")
    print("="*50)

    # 1. 配置 CondEncoder
    cond_config = ml_collections.ConfigDict(config.autoencoder.encoder)
    with cond_config.unlocked():
        cond_config.grid_size = (64,)
        cond_config.patch_size = (8,)
        cond_config.emb_dim = config.diffusion.emb_dim
        if 'input_dim' in cond_config: del cond_config.input_dim
    cond_encoder = Encoder(**cond_config)

    # 2. 配置 DiT
    diffusion_config = dict(config.diffusion)
    dit_supported_params = ['grid_size', 'emb_dim', 'depth', 'num_heads', 'mlp_ratio', 'out_dim']
    filtered_config = {k: v for k, v in diffusion_config.items() if k in dit_supported_params}
    dit_core = DiT(model_name=config.diffusion.model_name, **filtered_config)

    # 3. 组装 Wrapper
    raw_model = DiffusionWrapper(dit=dit_core, cond_encoder=cond_encoder)
    model = ModelParamsAdapter(raw_model)
    print("✅ 已启用 ModelParamsAdapter 以修复参数格式兼容性问题。")
    
    p_size = config.autoencoder.encoder.patch_size
    if isinstance(p_size, tuple): p_size = p_size[0]
    latent_len = 50 // p_size
    
    # ⚠️ 注意：DiT 输入的是 Latent，CondEncoder 输入的是原始观测数据
    dummy_x = jnp.zeros((1, latent_len, config.diffusion.emb_dim)) # Latent Z
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_c = jnp.zeros((1, 64, 2))  # <--- 原始条件输入 (必须是 2 通道)

    # 5. ✅ 核心修正：完整初始化
    print("正在初始化参数...")
    rng = random.PRNGKey(config.seed)
    rng, key_diff = random.split(rng)
    
    # 关键点：必须传入 c=dummy_c，否则 cond_encoder 不会创建参数！
    variables = model.init(key_diff, dummy_x, dummy_t, c=dummy_c)
    
    # 6. 创建 TrainState
    lr, tx = create_optimizer(config)
    from flax.training import train_state
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

    # 7. 简单的 Checkpoint 加载 (只为断点续训，不做复杂合并)
    job_name = f"{config.diffusion.model_name}_{config.dataset.num_samples}_samples"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)
    
    step = ckpt_mngr.latest_step()
    
    if step is not None:
        print(f"发现检查点 (Step {step})，正在恢复...")
        # 既然是从头训练，如果真的发现了检查点，那就直接覆盖，不用考虑新旧结构兼容
        restored = ckpt_mngr.restore(step)
        
        # 简单的解包逻辑
        if 'params' in restored:
            loaded_params = restored['params']
        elif 'model' in restored:
            loaded_params = restored['model'].get('params', restored['model'])
        elif 'state' in restored:
            loaded_params = restored['state'].get('params', restored['state'])
        else:
            loaded_params = restored
            
        state = state.replace(params=loaded_params, step=step)
        print("✅ 状态恢复成功。")
    else:
        print("未发现检查点，从头开始训练 (Fresh Start)。")

    # 打印参数确认
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params / 1e6:.2f} M")
    print("="*50 + "\n")

    # -------------------
    # 3) Device / sharding
    # -------------------
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}, local: {num_local_devices}")

    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    
    if jax.device_count() > 1:
        print("多设备环境：正在执行 host_local_array_to_global_array...")
        state = multihost_utils.host_local_array_to_global_array(state, mesh, P())
        fae_state = multihost_utils.host_local_array_to_global_array(fae_state, mesh, P())
    else:
        print("单设备环境：跳过 host_local_array_to_global_array (直接使用 Host 数据)。")

    print("\n🔍 [最终检查] 准备进入训练循环，检查 state.params...")
    if 'cond_encoder' not in state.params:
        print("❌ 致命错误：cond_encoder 在分片/准备阶段丢失！")
        # 这里可以再尝试一次紧急修复，或者直接报错停止
        raise RuntimeError("参数丢失，无法继续训练。")
    else:
        print("✅ [最终检查] cond_encoder 依然存在。Ready to train!")
    print("="*50 + "\n")

    print("\n🔍 正在检查 state.params 完整性...")
    

    # train / encoder steps
    train_step = create_train_diffusion_step(model, mesh, use_conditioning=use_conditioning)
    encoder_step = create_encoder_step(encoder, mesh)
    
    # 创建测试集loss计算函数（使用JIT编译以提高效率）
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P("batch"), P()),
        out_specs=P(),
        check_rep=False
    )
    def compute_test_loss(fae_state, diffusion_state, test_batch, rng):
        # 编码测试数据
        encoder_params, _ = fae_state.params
        coords_test, x_test, y_test = test_batch

        if x_test.shape[-1] == 2:
            x_test = jnp.mean(x_test, axis=-1, keepdims=True)

        z_u_test = encoder.apply(encoder_params, x_test)

        # 生成diffusion batch
        diff_batch_test, _ = get_diffusion_batch(rng, z1=z_u_test, c=None, use_conditioning=use_conditioning)

        if len(diff_batch_test) == 4:
            x, t, c, y = diff_batch_test
            pred = model.apply(diffusion_state.params, x, t, c)
            
        elif len(diff_batch_test) == 3:
            x, t, y = diff_batch_test
            pred = model.apply(diffusion_state.params, x, t)
        else:
            raise ValueError(f"Unexpected batch length: {len(diff_batch_test)}")
        
        eps = 1e-8
        batch_size, seq_len, channels = y.shape
        real_data_length = 50
        mask = jnp.arange(seq_len) < real_data_length
        mask = mask.astype(jnp.float32)
        mask = mask[None, :, None]
        mask = jnp.broadcast_to(mask, (batch_size, seq_len, channels))
        
        valid_count = jnp.sum(mask) + eps
        squared_error = (y - pred) ** 2
        masked_squared_error = squared_error * mask
        test_loss = jnp.sum(masked_squared_error) / valid_count
        test_loss = lax.pmean(test_loss, "batch")
        return test_loss

    # 创建保存路径用于存储评估图像
    save_path = os.path.join(os.getcwd(), "evaluation_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # -------------------
    # 4) Dataset and coords - KEEP CONSISTENT WITH FAE TRAINING
    # -------------------
    
    # 读取数据（与 train_autoencoder.py 保持一致）
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
    output_size = 50  # 输出序列长度为50（与 train_autoencoder.py 一致）
    
    # 使用 pad_or_trim 将每个样本调整到目标长度
    train_rho_N = pad_or_trim(train_rho_N, input_size)
    train_phase_N = pad_or_trim(train_phase_N, input_size)
    train_res = pad_or_trim(train_res, output_size)  # 输出调整为50
    
    # 调整维度为 (num_samples, input_size, 1)
    train_rho_N = train_rho_N.reshape(-1, input_size, 1)
    train_phase_N = train_phase_N.reshape(-1, input_size, 1)
    x_train_normalized = np.concatenate([train_rho_N, train_phase_N], axis=2)

    # ========== y不进行归一化，直接使用原始数据（与 train_autoencoder.py 一致）==========
    # 调整 y 的维度为 (num_samples, output_size, 1) = (num_samples, 50, 1)
    y_train = train_res.reshape(-1, output_size, 1)  # 直接使用原始数据，不归一化
    
    print(f"\n{'='*50}")
    print("扩散模型输入输出形状信息")
    print(f"{'='*50}")
    print(f"输入 x_train_normalized 形状: {x_train_normalized.shape}")
    print(f"  - 说明: (样本数, 序列长度={input_size}, 通道数=2)")
    print(f"  - 通道: [rho(标准化后), phase(标准化后)]")
    print(f"输出 y_train 形状: {y_train.shape}")
    print(f"  - 说明: (样本数, 序列长度={output_size}, 通道数=1)")
    print(f"  - 通道: [res(原始数据, log10尺度, 未归一化)]")
    print(f"y数据范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"{'='*50}\n")

    # 合并输入（保持与自编码器训练一致）
    condition_data = x_train_normalized  # 包含标准化后的 rho 和 phase

    # IMPORTANT: coords must match the shape used when training the autoencoder.
    # 生成输出坐标（50个点，匹配输出维度，与 train_autoencoder.py 一致）
    coords = np.linspace(0, 1, output_size)[:, None]  # shape (output_size, 1) = (50, 1)

    # Repeat coords across devices: shape (n_devices, num_sensors, 1)
    batch_coords = repeat(coords, "b d -> n b d", n=jax.device_count())

    batch = (batch_coords, condition_data, y_train) 
    batch = jax.tree.map(jnp.array, batch)
    batch = multihost_utils.host_local_array_to_global_array(batch, mesh, P("batch"))

    # If checkpoint dir doesn't exist, create and save config
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        # save config
        config_dict = config.to_dict()
        with open(os.path.join(os.getcwd(), job_name, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    # Ensure ckpt manager exists
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # -------------------
    # 5) Prepare test set (keep shapes consistent)
    # -------------------
    test_data_path = os.path.join(os.getcwd(), fae_job, "test_data.npz")
    print(f"Looking for test data at: {test_data_path}")
    if os.path.exists(test_data_path):
        print("✅ 加载自编码器的测试集...")
        test_data = np.load(test_data_path)
        x_test = test_data['x_test']
        y_test = test_data['y_test']
    
        print(f"   测试集形状: x_test{x_test.shape}, y_test{y_test.shape}")
        print(f"   y_test 数据范围: [{y_test.min():.3f}, {y_test.max():.3f}] (原始数据，未归一化)")
    else:
        # 如果没有测试数据文件，从训练数据中分割一部分作为测试集
        print("⚠️ 未找到测试数据文件，使用 train_test_split 从训练数据中分割20%作为测试集...")
        
        # 使用 train_test_split 划分数据 (80/20 划分)
        x_train_normalized, x_test, y_train, y_test = train_test_split(
            x_train_normalized, y_train, test_size=0.2, random_state=42
        )
        
        # 更新训练数据
        condition_data = x_train_normalized
        y_train = y_train
        
        print(f"   训练集: x_train{x_train_normalized.shape}, y_train{y_train.shape}")
        print(f"   测试集: x_test{x_test.shape}, y_test{y_test.shape}")

    # 使用测试数据（x_test 和 y_test 都是原始/标准化后的数据，未归一化）
    condition_data_test = x_test  # 包含标准化后的 rho 和 phase

    batch_coords_test = repeat(coords, "b d -> n b d", n=jax.device_count())
    test_batch = (batch_coords_test, condition_data_test, y_test)
    test_batch = jax.tree.map(jnp.array, test_batch)
    test_batch = multihost_utils.host_local_array_to_global_array(test_batch, mesh, P("batch"))

    # -------------------
    # 6) End-to-end eval step & autoencoder eval step
    # -------------------
    end_to_end_eval_step = create_end_to_end_eval_step(encoder, decoder, model, mesh, use_conditioning=use_conditioning)
    # 自编码器评估使用与 evaluate_autoencoder.py 一致的 eval_step（返回 MSE）
    autoencoder_mse_eval_step = create_eval_step(encoder, decoder, mesh)

    # -------------------
    # 7) Training loop
    # -------------------
    rng = random.PRNGKey(config.training.seed if 'seed' in config.training else 0)
    
    # 初始化loss列表用于绘制曲线
    train_loss_history = []
    test_loss_history = []
    train_step_history = []
    test_step_history = []
    
    for step in range(config.training.max_steps):
        start_time = time.time()
        rng, _ = random.split(rng)

        batch_coords, x_obs, y_true = batch
        z_target = encoder_step(fae_state.params[0], (batch_coords, y_true, y_true))
        c_for_model = x_obs
    
        
        diff_batch, rng = get_diffusion_batch(
            rng,
            z1=z_target,          
            c=c_for_model,   
            use_conditioning=use_conditioning 
        )
        state, loss = train_step(state, diff_batch)

        # Logging
        if step % config.logging.log_interval == 0:
            loss_val = float(loss)
            end_time = time.time()
            if jax.process_index() == 0:
                print(f"step: {step}, loss: {loss_val:.3e}, time: {end_time - start_time:.3f}")
                # 收集训练loss值用于绘图
                train_loss_history.append(loss_val)
                train_step_history.append(step)
                # 更新loss图
                if len(train_loss_history) > 0:
                    plot_loss_curve(
                        train_step_history, 
                        train_loss_history, 
                        test_losses=test_loss_history if len(test_loss_history) > 0 else None,
                        test_steps=test_step_history if len(test_step_history) > 0 else None,
                        save_path=save_path
                    )
                # Log to wandb
                wandb.log({
                    "train_loss": loss_val,
                    "learning_rate": lr(step),
                    "step": step,
                    "time_per_step": end_time - start_time
                }, step=step)

        # Periodic end-to-end evaluation (使用配置文件中的评估间隔)
        if step % config.logging.eval_interval == 0 and step > 0:
            try:
                # 现在返回两个值：rmse, normalized_rmse
                rmse_val, normalized_rmse_val, y_pred_val, y_true_val = end_to_end_eval_step(
                    fae_state, state, test_batch
                )

                # print(f'pred_res = {y_pred_val}')
                # print(f'true_res = {y_true_val}')

                if jax.process_index() == 0:  # 只在主进程画图
                    plot_inversion_result(
                        y_true_val, y_pred_val,
                        step,
                        save_path
                        # y_min 和 y_max 不再需要，因为 y 不归一化
                    )
                rmse_val = float(rmse_val) if rmse_val is not None else None
                normalized_rmse_val = float(normalized_rmse_val) if normalized_rmse_val is not None else None
            except Exception as e:
                rmse_val, normalized_rmse_val = None, None
                print("End-to-end eval failed:", e)
                import traceback
                traceback.print_exc()
            
            if jax.process_index() == 0:
                if rmse_val is not None and normalized_rmse_val is not None:
                    print(f"step: {step}, diffusion_loss: {float(loss):.3e}, end_to_end_rmse: {rmse_val:.3e}, normalized_rmse: {normalized_rmse_val:.3f}")
                    # Log evaluation metrics to wandb
                    wandb.log({
                        "eval/end_to_end_rmse": rmse_val,
                        "eval/normalized_rmse": normalized_rmse_val,
                        "eval/inversion_plot": wandb.Image(os.path.join(save_path, f"inversion_step_{step}.png"))
                    }, step=step)
                else:
                    print(f"step: {step}, diffusion_loss: {float(loss):.3e}, end_to_end_rmse: N/A")

        # Save checkpoint at intervals
        if step % config.saving.save_interval == 0:
            if jax.process_index() == 0:
                loss_val = float(loss)
                save_checkpoint(ckpt_mngr, state)
                # 构建检查点文件路径（orbax通常以step命名目录）
                ckpt_file_path = os.path.join(ckpt_path, str(step))
                print(f"💾 Saving checkpoint at step {step}, diffusion_loss: {loss_val:.3e}")
                print(f"   检查点文件: {ckpt_file_path}")
            else:
                save_checkpoint(ckpt_mngr, state)

    # Save final checkpoint
    print("\n" + "="*50)
    print("训练完成，保存最终模型...")
    print("="*50)
    
    if jax.process_index() == 0:
        print(f"模型检查点保存路径: {ckpt_path}")
    
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()
    
    if jax.process_index() == 0:
        # 构建最终检查点文件路径（使用state.step获取当前步数）
        final_step = int(state.step) if hasattr(state, 'step') else (config.training.max_steps - 1)
        final_ckpt_file = os.path.join(ckpt_path, str(final_step))
        print("✅ 最终模型已保存完成！")
        print(f"   检查点目录: {ckpt_path}")
        print(f"   检查点文件: {final_ckpt_file}")
        print(f"   可通过 restore_checkpoint 函数加载模型进行推理")
        print("="*50)
    
    # 计算最终测试集loss
    print("\n" + "="*50)
    print("计算最终测试集loss...")
    print("="*50)
    try:
        rng_test, _ = random.split(rng)
        test_loss = compute_test_loss(fae_state, state, test_batch, rng_test)
        test_loss_val = float(test_loss)
        
        if jax.process_index() == 0:
            # 收集最终测试loss值用于绘图
            test_loss_history.append(test_loss_val)
            test_step_history.append(final_step)
            print(f"✅ 最终测试集loss: {test_loss_val:.3e}")
            # 更新包含测试loss的loss图
            if len(test_loss_history) > 0:
                plot_loss_curve(
                    train_step_history, 
                    train_loss_history, 
                    test_losses=test_loss_history,
                    test_steps=test_step_history,
                    save_path=save_path
                )
                # Log final test loss to wandb
                wandb.log({
                    "test_loss": test_loss_val,
                    "training_loss_curve": wandb.Image(os.path.join(save_path, "training_loss.png"))
                }, step=final_step)
    except Exception as e:
        test_loss_val = None
        if jax.process_index() == 0:
            print(f"⚠️ 计算测试集loss失败: {e}")
            import traceback
            traceback.print_exc()
    print("="*50)

    # -------------------
    # 8) Unified evaluation (use test set, shapes kept consistent)
    # -------------------
    print("\n" + "="*50)
    print("开始统一模型评估")
    print("="*50)

    print("1. 评估自编码器重建性能...")
    try:
        # 与 evaluate_autoencoder.py 一致：先计算 MSE，再转 RMSE / NRMSE
        ae_mse = autoencoder_mse_eval_step(fae_state, test_batch)
        ae_mse = float(ae_mse)
        autoencoder_rmse = np.sqrt(ae_mse)
        autoencoder_normalized_rmse = autoencoder_rmse / 2.106
    except Exception as e:
        autoencoder_rmse, autoencoder_normalized_rmse = None, None
        print("Autoencoder eval failed:", e)

    print("2. 评估扩散模型生成性能...")
    try:
        # 现在返回四个值：rmse, normalized_rmse
        diffusion_rmse, diffusion_normalized_rmse, _, _ = end_to_end_eval_step(fae_state, state, test_batch)
        diffusion_rmse = float(diffusion_rmse) if diffusion_rmse is not None else None
        diffusion_normalized_rmse = float(diffusion_normalized_rmse) if diffusion_normalized_rmse is not None else None
    except Exception as e:
        diffusion_rmse, diffusion_normalized_rmse = None, None
        print("End-to-end diffusion eval failed:", e)

    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)

    if jax.process_index() == 0:
        if autoencoder_rmse is not None and autoencoder_normalized_rmse is not None:
            print(f"自编码器 - RMSE: {autoencoder_rmse:.6f}, NRMSE: {autoencoder_normalized_rmse:.6f} ({autoencoder_normalized_rmse*100:.1f}%)")
        else:
            print("自编码器测试: 评估失败")
        
        if diffusion_rmse is not None and diffusion_normalized_rmse is not None:
            print(f"扩散模型端到端 - RMSE: {diffusion_rmse:.6f}, NRMSE: {diffusion_normalized_rmse:.6f} ({diffusion_normalized_rmse*100:.1f}%)")
        else:
            print("扩散模型端到端: 评估失败")

        if autoencoder_normalized_rmse is not None and diffusion_normalized_rmse is not None:
            print(f"性能对比: 扩散模型比自编码器 {'更好' if diffusion_normalized_rmse < autoencoder_normalized_rmse else '稍差'}")
            # Log final evaluation metrics to wandb
            wandb.log({
                "final/autoencoder_rmse": autoencoder_rmse,
                "final/autoencoder_normalized_rmse": autoencoder_normalized_rmse,
                "final/diffusion_rmse": diffusion_rmse,
                "final/diffusion_normalized_rmse": diffusion_normalized_rmse,
                "final/diffusion_better": diffusion_normalized_rmse < autoencoder_normalized_rmse
            })

    # Finish wandb run
    if jax.process_index() == 0:
        wandb.finish()

    print("所有模型训练和评估完成！")
    print("="*50)

   
