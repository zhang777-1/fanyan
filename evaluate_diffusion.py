import os
import sys
# 确保项目路径正确
sys.path.append(os.path.join(os.getcwd(), "fundiff"))
import numpy as np
import matplotlib.pyplot as plt
from einops import repeat
import ml_collections
import json
import pandas as pd

import jax
from jax import random, vmap, jit, lax
from functools import partial
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import orbax.checkpoint as ocp

# 引入项目模块
from geoelectric_dataset import pad_or_trim
from model import Encoder, Decoder, DiT, DiffusionWrapper, ModelParamsAdapter
from function_diffusion.utils.model_utils import create_optimizer, create_diffusion_state
from function_diffusion.utils.checkpoint_utils import create_checkpoint_manager, restore_checkpoint
from function_diffusion.utils.train_utils import sample_ode
from configs.diffusion import get_config


def plot_resistivity_curve(y_true, y_pred, output_size, sample_index, rmse_value, save_name, plot_raw_rho=True):
    actual_res = y_true[sample_index, :]
    predicted_res = y_pred[sample_index, :]

    if plot_raw_rho:
        actual_res = np.power(10, actual_res)
        predicted_res = np.power(10, predicted_res)
        rho_unit = 'Ω·m'
        title_scale = 'Raw Resistivity'
    else:
        rho_unit = 'Log10(Ω·m)'
        title_scale = 'Log10 Resistivity'

    depth_points = np.linspace(0, 1200, output_size)

    plt.figure(figsize=(10, 6))

    plt.plot(depth_points, actual_res, linestyle='-', linewidth=2, color='blue', label=f'True $\\rho$ ({rho_unit})')
    plt.plot(depth_points, predicted_res, linestyle='--', linewidth=2, color='red', label=f'Predicted $\\rho$ ({rho_unit})')

    if plot_raw_rho:
        plt.yscale('log')
        all_values = np.concatenate([actual_res, predicted_res])
        y_min = max(0.01, np.min(all_values) * 0.8)
        y_max = min(10000, np.max(all_values) * 1.2)
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(0, 5)

    plt.title(f'Sample {sample_index} {title_scale} Curve\nValidation RMSE : {rmse_value:.4f}', fontsize=12)
    plt.xlabel('Depth (m)', fontsize=10)
    plt.ylabel(f'Resistivity ({rho_unit})', fontsize=10)

    plt.xticks(np.arange(0, 1201, 400))
    plt.xlim(0, 1200)

    plt.legend(fontsize=10)
    plt.grid(True, which='both', alpha=0.5)

    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close()

    return save_name


def evaluate_diffusion(config_path=None, autoencoder_job_name=None, diffusion_job_name=None, num_samples=None, num_steps=10):
    # -------------------
    # 配置加载
    # -------------------
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ml_collections.ConfigDict(config_dict)
    else:
        config = get_config(autoencoder_diffusion="fae,dit")
        if num_samples:
            config.dataset.num_samples = num_samples
    
    if not autoencoder_job_name:
        autoencoder_job_name = f"{config.autoencoder.model_name}_{config.dataset.num_samples}_samples"
    if not diffusion_job_name:
        diffusion_job_name = f"{config.diffusion.model_name}_{config.dataset.num_samples}_samples"
    
    fae_ckpt_path = os.path.join(os.getcwd(), autoencoder_job_name, "ckpt")
    diffusion_ckpt_path = os.path.join(os.getcwd(), diffusion_job_name, "ckpt")
    
    if not os.path.exists(fae_ckpt_path):
        print(f"❌ FAE Checkpoint not found at {fae_ckpt_path}")
        return
    if not os.path.exists(diffusion_ckpt_path):
        print(f"❌ Diffusion Checkpoint not found at {diffusion_ckpt_path}")
        return

    print("="*50)
    print("🚀 开始初始化模型 (严格匹配训练结构)")
    print("="*50)

    # -------------------
    # 始化 FAE 
    # -------------------
    print("🔧 初始化 FAE...")
    # 强制将 input_dim 设为 1，以匹配旧的 FAE Checkpoint
    fae_enc_config = ml_collections.ConfigDict(config.autoencoder.encoder)
    with fae_enc_config.unlocked():
        if 'input_dim' in fae_enc_config:
            del fae_enc_config.input_dim
    
    fae_dec_config = ml_collections.ConfigDict(config.autoencoder.decoder)
    with fae_dec_config.unlocked():
        fae_dec_config.out_dim = 1

    encoder = Encoder(**fae_enc_config)
    decoder = Decoder(**fae_dec_config)

    # FAE 状态恢复 logic
    dummy_y = jnp.zeros((1, 50, 1))
    patch_size = config.autoencoder.encoder.patch_size
    p_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
    latent_len = 50 // p_size
    dummy_latent = jnp.zeros((1, latent_len, config.autoencoder.encoder.emb_dim))

    rng = random.PRNGKey(0)
    rng, key_enc, key_dec = random.split(rng, 3)
    enc_variables = encoder.init(key_enc, dummy_y)
    dec_variables = decoder.init(key_dec, dummy_latent, dummy_y)
    
    # 创建 FAE State
    _, tx_fae = create_optimizer(config)
    from flax.training import train_state
    fae_state = train_state.TrainState.create(
        apply_fn=decoder.apply,
        params=(enc_variables['params'], dec_variables['params']),
        tx=tx_fae
    )
    
    # 加载 FAE Checkpoint
    ckpt_mngr_fae = create_checkpoint_manager(config.saving, fae_ckpt_path)
    step_fae = ckpt_mngr_fae.latest_step()
    if step_fae is None:
        raise RuntimeError("FAE Checkpoint load failed")
    
    raw_fae = ckpt_mngr_fae.restore(step_fae, args=ocp.args.StandardRestore(item=None))
    
    # 处理 FAE 参数解包
    if 'params' in raw_fae:
        fae_params = raw_fae['params']
    elif 'model' in raw_fae:
        fae_params = raw_fae['model']['params']
    elif 'state' in raw_fae:
        fae_params = raw_fae['state']['params']
    else:
        fae_params = raw_fae
        
    fae_state = fae_state.replace(params=fae_params)
    print("✅ FAE 加载成功")

    # -------------------
    # C. 初始化 Diffusion (Padding=64, Latent=25)
    # -------------------
    print("🔧 初始化 Diffusion Model...")
    use_conditioning = True

    # 配置 CondEncoder (输入 64)
    cond_config = ml_collections.ConfigDict(config.autoencoder.encoder)
    with cond_config.unlocked():
        cond_config.grid_size = (64,)  # ⚠️ 必须是 64，匹配训练时的 input_size
        cond_config.patch_size = (8,)
        cond_config.emb_dim = config.diffusion.emb_dim
        if 'input_dim' in cond_config: del cond_config.input_dim
    cond_encoder = Encoder(**cond_config)

    # 配置 DiT (Latent Grid 25)
    diffusion_config = dict(config.diffusion)
    dit_supported_params = ['grid_size', 'emb_dim', 'depth', 'num_heads', 'mlp_ratio', 'out_dim']
    filtered_config = {k: v for k, v in diffusion_config.items() if k in dit_supported_params}
    filtered_config['grid_size'] = (25,) # ⚠️ 强制修正 Latent 长度 (50 // 2)
    dit_core = DiT(model_name=config.diffusion.model_name, **filtered_config)

    # 组装 Wrapper
    raw_model = DiffusionWrapper(dit=dit_core, cond_encoder=cond_encoder)
    model = ModelParamsAdapter(raw_model)

    # Init + Load 
    print("⚡ 正在初始化并合并参数 (Init + Merge)...")
    rng = random.PRNGKey(42)
    rng, key_diff = random.split(rng)
    
    dummy_x = jnp.zeros((1, 25, 256)) # Latent
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_c = jnp.zeros((1, 64, 2))   # Condition (Padding 到 64)

    init_variables = model.init(key_diff, dummy_x, dummy_t, c=dummy_c)
    
    ckpt_mngr_diff = create_checkpoint_manager(config.saving, diffusion_ckpt_path)
    latest_step_diff = ckpt_mngr_diff.latest_step()
    if latest_step_diff is None: raise RuntimeError("Diffusion Checkpoint not found")
    
    print(f"📥 加载 Diffusion Step: {latest_step_diff}")
    raw_diff = ckpt_mngr_diff.restore(latest_step_diff, args=ocp.args.StandardRestore(item=None))

    # 提取 Checkpoint 里的 params
    loaded_params = None
    if 'params' in raw_diff:
        loaded_params = raw_diff['params']
    elif 'model' in raw_diff:
        loaded_params = raw_diff['model'].get('params', raw_diff['model'])
    elif 'state' in raw_diff:
        loaded_params = raw_diff['state'].get('params', raw_diff['state'])
    else:
        loaded_params = raw_diff

    if isinstance(loaded_params, dict) and 'model' in loaded_params and 'dit' not in loaded_params:
        loaded_params = loaded_params['model']

    # 如果 init_variables 顶层就有 params，我们覆盖它
    final_variables = init_variables.copy()
    if 'params' in final_variables:
        # 递归或直接覆盖？通常这里只要覆盖 params 字典即可，flax 会处理
        final_variables['params'] = loaded_params
    else:
        # 如果结构不同，视情况而定，但通常 train_state.params 对应 variables['params']
        final_variables = loaded_params 

    # 创建 Diffusion State
    lr, tx = create_optimizer(config)
    diffusion_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=final_variables['params'] if 'params' in final_variables else final_variables,
        tx=tx
    )
    print("✅ Diffusion 模型加载完成")

    # -------------------
    # D. 并行设置
    # -------------------
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    fae_state = multihost_utils.host_local_array_to_global_array(fae_state, mesh, P())
    diffusion_state = multihost_utils.host_local_array_to_global_array(diffusion_state, mesh, P())

    # -------------------
    # E. 数据加载与预处理 (复刻 train_diffusion.py 逻辑)
    # -------------------
    print("📚 正在准备测试数据...")
    train_data = pd.read_json('./train_data.json')
    num_samples_eval = min(10000, len(train_data))
    train_data_sampled = train_data.sample(n=num_samples_eval, random_state=42).reset_index(drop=True)

    try:
        train_rho = np.array([np.array(x) for x in train_data_sampled['rho']])
        train_phase = np.array([np.array(x) for x in train_data_sampled['phase']])
        train_res = np.array([np.array(x) for x in train_data_sampled['res']])
    except KeyError:
        raise ValueError("Data format error")

    train_res[train_res <= 0] = 1e-6
    
    # 统计量 (应与训练时一致，这里简单重新计算)
    rho_mean, rho_dev = np.mean(train_rho), np.std(train_rho)
    phase_mean, phase_dev = np.mean(train_phase), np.std(train_phase)
    
    train_rho_N = (train_rho - rho_mean) / rho_dev
    train_phase_N = (train_phase - phase_mean) / phase_dev
    
    # 🔥 关键：输入 Pad 到 64，输出保持 50
    input_size = config.dataset.num_sensors # 64
    output_size = 50
    
    train_rho_N = pad_or_trim(train_rho_N, input_size)
    train_phase_N = pad_or_trim(train_phase_N, input_size)
    y_test_data = pad_or_trim(train_res, output_size) # Ground Truth
    
    x_test_data = np.concatenate([
        train_rho_N.reshape(-1, input_size, 1), 
        train_phase_N.reshape(-1, input_size, 1)
    ], axis=2)
    y_test_data = y_test_data.reshape(-1, output_size, 1)

    # 构造 Batch
    coords = np.linspace(0, 1, output_size)[:, None]
    num_test_samples = x_test_data.shape[0]
    batch_coords = jnp.tile(coords[None, :, :], (num_test_samples, 1, 1))
    
    # 简单切分，确保能被 device 整除
    n_dev = jax.device_count()
    limit = (len(x_test_data) // n_dev) * n_dev
    x_test_data = x_test_data[:limit]
    y_test_data = y_test_data[:limit]
    
    batch_test = (batch_coords, x_test_data, y_test_data)
    batch_test = jax.tree.map(jnp.array, batch_test)
    batch_test = multihost_utils.host_local_array_to_global_array(batch_test, mesh, P("batch"))

    # -------------------
    # F. 推理函数 (JIT)
    # -------------------
    class ProxyState:
        def __init__(self, params, apply_fn):
            self.params = params
            self.apply_fn = apply_fn

    # 2. 定义扩散模型单步推理函数
    def diffusion_step_fn(params, x, t, c):
        if use_conditioning:
            return model.apply(params, x, t, c)
        else:
            return model.apply(params, x, t)

    # 3. 定义快速编解码函数 (让逻辑更清晰)
    def fast_encode(params, x):
        return encoder.apply(params, x)

    def fast_decode(params, z, coords):
        return decoder.apply(params, z, coords)

    # 4. 主推理步骤 (Shard Map)
    @partial(shard_map, mesh=mesh, in_specs=(P(), P(), P("batch")), out_specs=P(), check_rep=False)
    def custom_eval_step(fae_state, diffusion_state, batch):
        coords, x_condition, y_true_target = batch
        
        enc_params, dec_params = fae_state.params
        
        x_for_fae = x_condition[:, :50, 0:1] 
        z_condition = fast_encode(enc_params, x_for_fae)
        
        # --- 扩散生成 (Diffusion) ---
        latent_shape = z_condition.shape
        key = random.PRNGKey(42)
        z0 = random.normal(key, latent_shape)
        
        proxy_state = ProxyState(diffusion_state.params, diffusion_step_fn)
        
        z_generated, _ = sample_ode(
            proxy_state, 
            z0, 
            c=x_condition, 
            num_steps=num_steps, 
            use_conditioning=use_conditioning
        )
        
        # --- 快速解码 (Fast Decode) ---
        batch_size = z_generated.shape[0]
        
        if coords.ndim == 3 and coords.shape[2] == 2:
            if coords.shape[0] == batch_size:
                coords_processed = coords[:, :, 0:1]
            else:
                coords_processed = jnp.broadcast_to(coords[0:1, :, 0:1], (batch_size, coords.shape[1], 1))
        elif coords.ndim == 3 and coords.shape[2] == 1:
            if coords.shape[0] == batch_size:
                coords_processed = coords
            else:
                coords_processed = jnp.broadcast_to(coords[0:1, :, :], (batch_size, coords.shape[1], 1))
        elif coords.ndim == 2:
            coords_processed = jnp.broadcast_to(coords[None, :, :], (batch_size, coords.shape[0], coords.shape[1]))
        elif coords.ndim == 1:
            coords_processed = jnp.broadcast_to(coords[None, :, None], (batch_size, coords.shape[0], 1))
        else:
            num_coords = 50
            coords_processed = jnp.broadcast_to(jnp.linspace(0, 1, num_coords)[None, :, None], (batch_size, num_coords, 1))
        
        # 调用快速解码
        y_pred = fast_decode(dec_params, z_generated, coords_processed)
        
        # --- 误差计算 ---
        squared_error = (y_pred - y_true_target) ** 2
        rmse = jnp.sqrt(jnp.mean(squared_error))
        normalized_rmse = rmse / 2.106
        
        return rmse, normalized_rmse, y_pred, y_true_target

    # -------------------
    # G. 执行推理与绘图
    # -------------------
    print("🚀 开始推理...")
    try:
        rmse_val, normalized_rmse_val, y_pred_val, y_true_val = custom_eval_step(
            fae_state, diffusion_state, batch_test
        )
        
        rmse_val = float(np.mean(rmse_val))
        print(f"推理完成! Mean RMSE: {rmse_val:.6f}")
        
        # 数据转换用于绘图
        y_pred_np = np.array(y_pred_val).squeeze()
        y_true_np = np.array(y_true_val).squeeze()
        
        if y_pred_np.ndim == 1: y_pred_np = y_pred_np[None, :]
        if y_true_np.ndim == 1: y_true_np = y_true_np[None, :]

        save_dir = os.path.join(os.getcwd(), "evaluation_plots")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        # 只画第一个样本
        save_name = os.path.join(save_dir, "diffusion_sample_0.png")
        plot_resistivity_curve(
            y_true=y_true_np,
            y_pred=y_pred_np,
            output_size=50,
            sample_index=0,
            rmse_value=rmse_val,
            save_name=save_name,
            plot_raw_rho=True
        )
        print(f"📊 结果已保存: {save_name}")

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import glob
    # 简单的自动查找逻辑
    potential_fae = glob.glob("FAE_*_samples")
    potential_diff = glob.glob("DiT_*_samples")
    
    fae = potential_fae[0] if potential_fae else None
    diff = potential_diff[0] if potential_diff else None
    
    if fae and diff:
        evaluate_diffusion(autoencoder_job_name=fae, diffusion_job_name=diff)
    else:
        evaluate_diffusion()