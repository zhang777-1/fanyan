import sys
sys.path.append('.')
import jax
import jax.numpy as jnp

# 1. 检查导入的配置
print("="*60)
print("1. 检查当前配置")
print("="*60)

try:
    from configs.models import MODEL_CONFIGS
    print(f"FAE num_latents: {MODEL_CONFIGS['FAE'].encoder.num_latents}")
    print(f"DiT grid_size: {MODEL_CONFIGS['DiT'].grid_size}")
except Exception as e:
    print(f"导入配置失败: {e}")

# 2. 检查实际Encoder输出
print("\\n" + "="*60)
print("2. 测试Encoder输出")
print("="*60)

try:
    from model import Encoder
    import ml_collections
    
    # 使用当前配置
    encoder_config = MODEL_CONFIGS['FAE'].encoder.to_dict()
    print(f"Encoder配置: {encoder_config}")
    
    encoder = Encoder(**encoder_config)
    
    # 测试输入
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 64, 1))  # batch=2, seq_len=64, channels=1
    
    # 初始化参数
    params = encoder.init(rng, x)
    print("\\nEncoder参数结构:")
    
    def print_params(pytree, prefix=""):
        if isinstance(pytree, dict):
            for key, value in pytree.items():
                if hasattr(value, 'shape'):
                    print(f"{prefix}{key}: shape={value.shape}")
                    # 特别注意位置编码
                    if 'pos_emb' in str(key):
                        print(f"{prefix}  ⚠️ 找到位置编码!")
                elif isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_params(value, prefix + "  ")
    
    print_params(params)
    
    # 前向传播
    output = encoder.apply(params, x)
    print(f"\\nEncoder输出形状: {output.shape}")
    
except Exception as e:
    print(f"测试Encoder失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查检查点实际内容
print("\\n" + "="*60)
print("3. 检查检查点内容")
print("="*60)

import os
fae_ckpt_path = "./FAE_10000_samples/ckpt"
if os.path.exists(fae_ckpt_path):
    try:
        import orbax.checkpoint as ocp
        mngr = ocp.CheckpointManager(fae_ckpt_path, ocp.PyTreeCheckpointer())
        latest_step = mngr.latest_step()
        
        if latest_step:
            print(f"检查点步数: {latest_step}")
            
            # 尝试加载看看结构
            restored = mngr.restore(latest_step)
            
            print("\\n检查点参数结构:")
            def print_restored(obj, prefix="", depth=0):
                if depth > 3:  # 防止递归太深
                    return
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if hasattr(value, 'shape'):
                            print(f"{prefix}{key}: shape={value.shape}")
                            if value.shape == (1, 12, 256) or value.shape == (1, 8, 256):
                                print(f"{prefix}  ⭐ 关键形状!")
                        elif isinstance(value, dict):
                            print(f"{prefix}{key}:")
                            print_restored(value, prefix + "  ", depth+1)
                elif hasattr(obj, 'shape'):
                    print(f"{prefix}shape: {obj.shape}")
            
            print_restored(restored)
            
    except Exception as e:
        print(f"检查检查点失败: {e}")
else:
    print(f"检查点目录不存在: {fae_ckpt_path}")

print("\\n" + "="*60)
print("诊断完成")
print("="*60)
