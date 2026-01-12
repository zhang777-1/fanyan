from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, normal, xavier_uniform

import flax.linen as nn
from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x



class PatchEmbed1D(nn.Module):
    # 1. 类型提示修改：允许 int 或 tuple
    patch_size: Union[int, Tuple[int]] = (4,)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, Length, Channels)
        # 例如 FAE: (B, 50, 1), DiT: (B, 64, 2)
        
        # -------------------------------------------------------
        # 2. 健壮性处理：统一 patch_size 格式
        # -------------------------------------------------------
        # nn.Conv 的 kernel_size 需要传入一个元组，例如 (8,)
        if isinstance(self.patch_size, int):
            # 如果传入的是整数 8，转为元组 (8,)
            p_size = (self.patch_size,)
        else:
            # 如果传入的已经是元组 (8,)，直接使用
            p_size = self.patch_size

        x = nn.Conv(
            features=self.emb_dim,
            kernel_size=p_size,    #
            strides=p_size,       
            kernel_init=self.kernel_init,
            padding='VALID',       # 1D Patch Embedding 通常不补零
            name="proj",
        )(x)

        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=self.layer_norm_eps)(x)
            
        return x




class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


pos_emb_init = get_1d_sincos_pos_embed


class Encoder(nn.Module):
    emb_dim: int
    patch_size: Tuple
    depth: int
    num_heads: int
    mlp_ratio: int
    num_latents: int = 256
    grid_size: Tuple = (100,)
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, l, c = x.shape
        x = PatchEmbed1D(self.patch_size, self.emb_dim)(x)

        # 1) 保留一个静态的位置编码参数，用于兼容旧的 checkpoint 结构
        #    形状仍然是 (1, 8, 256)（由 grid_size / patch_size 决定），但不再参与实际计算。
        _ = self.variable(
            "pos_emb",
            "enc_emb",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            self.grid_size[0] // self.patch_size[0],
        )

        # 2) 实际使用基于当前特征序列长度的动态位置编码，避免长度不匹配
        seq_len = x.shape[1]
        pos_emb_dyn = get_1d_sincos_pos_embed(self.emb_dim, seq_len)  # (1, seq_len, emb_dim)
        x = x + pos_emb_dyn

        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, L,  D) --> (B, L', D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (L', D)
                             )

        latents = repeat(latents, 'l d -> b l d', b=x.shape[0])  # (B, L', D)
        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)

        latents = nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)
        return latents


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, coords):
        b, n, c = x.shape  # x: (batch, latent_tokens, latent_dim)

        # ========== 修复坐标形状 ==========
        
        # 更健壮的形状处理
        if coords.ndim == 3:
            # 检查是否是 (batch, 1, 96) 需要转置
            if coords.shape[1] == 1 and coords.shape[2] == 96:
                # 转置坐标形状: (batch, 1, 96) -> (batch, 96, 1)
                coords_processed = jnp.transpose(coords, (0, 2, 1))
            else:
                # 其他3D情况直接使用
                coords_processed = coords
        elif coords.ndim == 2:
            if coords.shape[0] == 1 and coords.shape[1] == 96:
                # (1, 96) -> (batch, 96, 1)
                coords_processed = jnp.transpose(coords, (1, 0))  # (96, 1)
                coords_processed = jnp.broadcast_to(coords_processed[None, :, :], (b, 96, 1))
            else:
                # (num_coords, coord_dim) - 广播到batch维度
                coords_processed = jnp.broadcast_to(coords[None, :, :], (b, coords.shape[0], coords.shape[1]))
        elif coords.ndim == 1:
            if coords.shape[0] == 96:
                # (96,) -> (batch, 96, 1)
                coords_processed = jnp.broadcast_to(coords[None, :, None], (b, 96, 1))
            else:
                # 其他1D情况
                coords_processed = jnp.broadcast_to(coords[None, :, None], (b, coords.shape[0], 1))
        else:
            # 默认创建96个坐标点
            coords_processed = jnp.broadcast_to(jnp.linspace(0, 1, 96)[None, :, None], (b, 96, 1))

        # ========== 修复结束 ==========

        # 投影隐变量和坐标到相同维度
        x_proj = nn.Dense(self.dec_emb_dim)(x)  # (batch, n, dec_emb_dim)
        coords_proj = nn.Dense(self.dec_emb_dim)(coords_processed)  # (batch, num_coords, dec_emb_dim)

        # 使用交叉注意力：坐标作为query，隐变量作为key-value
        for _ in range(self.dec_depth):
            coords_proj = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps
            )(coords_proj, x_proj)  # query=coords, key_value=latents

        # 最终输出处理
        output = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords_proj)
        output = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
            layer_norm_eps=self.layer_norm_eps
        )(output)  # (batch, num_coords, out_dim)

        return output


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    emb_dim: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


def modulate(x, shift, scale):
    b, l, d = x.shape
    
    # 确保 shift 和 scale 是 (batch, emb_dim)
    if shift.ndim == 1:
        # (emb_dim,) -> (batch, emb_dim)
        shift = jnp.broadcast_to(shift[None, :], (b, d))
        scale = jnp.broadcast_to(scale[None, :], (b, d))
    elif shift.ndim == 2 and shift.shape[0] == b and shift.shape[1] == d:
        # 已经正确
        pass
    else:
        # 其他情况，尝试 reshape
        shift = shift.reshape(b, d)
        scale = scale.reshape(b, d)
    
    # 扩展为 (batch, 1, emb_dim)
    shift = shift[:, None, :]
    scale = scale[:, None, :]
    
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    emb_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # 获取输入形状
        b, l, d = x.shape  # batch, seq_len, emb_dim
        
        # 首先确保c的形状正确
        # c 应该是 (batch, emb_dim)
        if c.ndim == 1:
            # 如果是扁平化的，尝试reshape
            if c.shape[0] == b * d:
                c = c.reshape(b, d)
            elif c.shape[0] == d:
                # 如果是(emb_dim,)，广播到batch
                c = jnp.broadcast_to(c[None, :], (b, d))
            else:
                raise ValueError(f"Unexpected condition shape: {c.shape}")
        
        # Calculate adaLn modulation parameters.
        c = nn.gelu(c)
        c = nn.Dense(6 * self.emb_dim, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)

        # 确保所有调制参数形状正确
        # shift_msa, scale_msa, gate_msa: (batch, emb_dim)
        # 我们需要将它们转换为 (batch, 1, emb_dim) 以广播到序列
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads
        )(x_modulated, x_modulated)
        
        # 正确广播gate_msa
        gate_msa = gate_msa[:, None, :]  # (batch, 1, emb_dim)
        x = x + (gate_msa * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_hidden_dim = int(self.emb_dim * self.mlp_ratio)
        mlp_x = MlpBlock(mlp_hidden_dim, self.emb_dim)(x_modulated2)
        
        # 正确广播gate_mlp
        gate_mlp = gate_mlp[:, None, :]  # (batch, 1, emb_dim)
        x = x + (gate_mlp * mlp_x)
        
        return x


pos_emb_init = get_1d_sincos_pos_embed

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Support conditional input.
    """
    model_name: Optional[str]
    grid_size: tuple
    emb_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    out_dim: int
    cond_dim: Optional[int] = None  

    @nn.compact
    def __call__(self, x, t, c=None):
        print(f"[DiT] x shape: {x.shape}, t shape: {t.shape}, c shape: {c.shape if c is not None else 'None'}")
        b, l, d = x.shape
        
        # 输入投影到 emb_dim
        if d != self.emb_dim:
            x = nn.Dense(self.emb_dim, name="input_projection")(x)
            d = self.emb_dim  # 更新维度
        
        # 位置编码
        pos_emb = get_1d_sincos_pos_embed(self.emb_dim, l)
        pos_emb = jnp.broadcast_to(pos_emb, (b, l, self.emb_dim))
        x = x + pos_emb

        # 时间嵌入 - 这是关键修复
        # t 应该是标量时间步，形状为 (batch,)
        # 但我们需要为每个DiTBlock生成条件向量
        t_emb = TimestepEmbedder(self.emb_dim)(t)  # (B, emb_dim)
        
        # 条件处理
        if c is not None:
            
            # 获取条件的特征维度
            if c.ndim == 2:
                cond_features = c.shape[1]
                c_processed = c
            elif c.ndim == 3:
                # 对序列维度取平均
                c_processed = jnp.mean(c, axis=1)  # (batch, features)
                cond_features = c_processed.shape[1]
            else:
                raise ValueError(f"Unexpected condition shape: {c.shape}")
            
            
            # 投影到emb_dim（如果需要）
            if cond_features != self.emb_dim:
                c_proj = nn.Dense(self.emb_dim, name="cond_projection")(c_processed)
            else:
                c_proj = c_processed
            
            # 将条件和时间嵌入融合
            t_emb = t_emb + c_proj
        
        
        # 所以t_emb将作为每个DiTBlock的条件参数c
        for _ in range(self.depth):
            x = DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)(x, t_emb)
        # ========== 修复结束 ==========
        
        # 7. 输出层
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class DiffusionWrapper(nn.Module):
    """
    包含一个条件编码器 (处理观测数据 x) 和一个扩散模型 (DiT)。
    """
    dit: nn.Module          # 核心扩散模型
    cond_encoder: nn.Module # 条件编码器 (处理 x)

    @nn.compact
    def __call__(self, x, t, c):
        
        # 1. 编码条件
        # 输入: (B, 64, 2) -> 输出: (B, 8, 256) (假设 patch=8)
        c_emb = self.cond_encoder(c)
        
        pred = self.dit(x, t, c=c_emb)
        
        return pred

class SafeDiffusionWrapper(nn.Module):
    dit: nn.Module
    cond_encoder: nn.Module

    @nn.compact
    def __call__(self, x, t, c):
        # 1. 编码条件
        # 输入: (B, 64, 2) -> 输出: (B, 8, 256)
        c_emb = self.cond_encoder(c)
        
        # 2. 扩散模型预测
        pred = self.dit(x, t, c=c_emb)
        
        return pred

class ModelParamsAdapter:
    def __init__(self, model):
        self.model = model
        self.cached_pos_emb = None  # 用于缓存位置编码
    
    def init(self, *args, **kwargs):
        variables = self.model.init(*args, **kwargs)
        if 'pos_emb' in variables:
            self.cached_pos_emb = variables['pos_emb']
            print("[Adapter] 成功捕获并缓存 pos_emb 集合。")
        return variables
    
    def apply(self, variables, *args, **kwargs):
        inputs = {}
        
        # 处理 params
        if isinstance(variables, dict) and 'params' not in variables:
            inputs['params'] = variables
        else:
            inputs.update(variables) # 假设已经是完整字典
            
        if self.cached_pos_emb is not None and 'pos_emb' not in inputs:
            inputs['pos_emb'] = self.cached_pos_emb
            
        return self.model.apply(inputs, *args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self.model, name)
