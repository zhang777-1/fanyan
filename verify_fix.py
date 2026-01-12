import sys
sys.path.append('.')
from configs.models import MODEL_CONFIGS
from configs.diffusion import get_config

print("="*60)
print("配置验证 - 匹配现有自编码器检查点")
print("="*60)

# 检查自编码器配置
print("1. 自编码器配置:")
fae_config = MODEL_CONFIGS['FAE']
print(f"   num_latents: {fae_config.encoder.num_latents} (应该是12)")
print(f"   emb_dim: {fae_config.encoder.emb_dim} (应该是256)")

# 检查DiT配置
print("\\n2. 扩散模型配置:")
dit_config = MODEL_CONFIGS['DiT']
print(f"   grid_size: {dit_config.grid_size} (应该是(12,))")

# 检查主配置
print("\\n3. 训练主配置:")
config = get_config(autoencoder_diffusion="fae,dit")
print(f"   z_dim: {config.z_dim} (应该是[1, 12, 256])")

# 验证
all_good = True
if fae_config.encoder.num_latents != 12:
    print("❌ 自编码器 num_latents 应该是12")
    all_good = False
    
if dit_config.grid_size != (12,):
    print("❌ DiT grid_size 应该是(12,)")
    all_good = False
    
if config.z_dim != [1, 12, 256]:
    print("❌ z_dim 应该是[1, 12, 256]")
    all_good = False

print("="*60)
if all_good:
    print("✅ 所有配置正确！")
    print("   可以加载现有的自编码器检查点")
    print("   不需要重新训练自编码器")
else:
    print("❌ 配置仍有问题，请检查修改")
