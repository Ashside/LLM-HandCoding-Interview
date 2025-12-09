import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# 由于 RoPE 计算中会频繁使用 cos/sin 表，因此预先计算好以提升效率。
# 这里的实现还支持 YaRN 风格的长度外推缩放。
def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Precompute RoPE cos/sin tables (optionally with YaRN-style scaling).

    Args:
        dim:         rotary dimension (must be even).
        end:         max sequence length to support (number of positions).
        rope_base:   RoPE base, e.g. 1e4 / 1e6.
        rope_scaling:
            Optional dict for length extrapolation (YaRN-style), with keys:
                - "original_max_position_embeddings" (int)
                - "factor" (float)
                - "beta_fast" (float)
                - "beta_slow" (float)
        device:      device for returned tensors.
        dtype:       dtype for returned tensors.

    Returns:
        freqs_cos: [end, dim]
        freqs_sin: [end, dim]
    """
    assert dim % 2 == 0, "RoPE dimension must be even."
    half_dim = dim // 2

    if device is None:
        device = torch.device("cpu")

    # Base inverse frequencies: [half_dim]
    # Note: 使用 half_dim 分母是主流实现
    freqs = torch.arange(0, half_dim, device=device, dtype=dtype)
    freqs = 1.0 / (rope_base ** (freqs / half_dim))

    # Optional YaRN-style scaling for extrapolation
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4.0)
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)

        if end > orig_max:
            # 找到需要进行缩放的分界维度 corr_dim
            # 2π / freq = 波长
            # 这里的 freqs 是 [half_dim]，对应不同波长
            corr_dim = next(
                (i for i in range(half_dim) if 2 * math.pi / freqs[i].item() > orig_max),
                half_dim,
            )

            # 构建从 beta_slow 到 beta_fast 的线性插值
            idx = torch.arange(half_dim, device=device, dtype=dtype)
            # 避免除以零
            power = idx / max(half_dim - 1, 1)
            # 线性插值
            beta = beta_slow + (beta_fast - beta_slow) * power

            # YaRN 标准公式 λ = (β·α - β + 1) / (β·α)
            # 高频部分（idx < corr_dim）使用 YaRN 缩放，低频部分使用 1/factor
            scale = torch.where(
                idx < corr_dim, # 高频部分
                (beta * factor - beta + 1.0) / (beta * factor), # YaRN 缩放
                torch.full_like(beta, 1.0 / factor), # 低频部分
            )
            freqs = freqs * scale # 应用缩放

    # Positions: [end], 从 0 到 end-1
    t = torch.arange(end, device=device, dtype=dtype)

    # [end, half_dim]
    freqs = torch.einsum("i,j->ij", t, freqs).to(dtype)

    # Duplicate for even/odd dims: [end, dim]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    freqs_cos = torch.cat([cos, cos], dim=-1)
    freqs_sin = torch.cat([sin, sin], dim=-1)
    # freqs_cos, freqs_sin: [end, dim]
    return freqs_cos, freqs_sin


def rotate_half(x: Tensor) -> Tensor:
    """
    Helper for RoPE: rotate last dimension by 90 degrees in complex plane.
    [x1, x2] -> [-x2, x1]
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Apply precomputed RoPE cos/sin to query/key.

    Typical shapes:
        q, k:        [batch, n_heads, seq_len, dim]
        cos, sin:    [max_seq_len, dim]
        position_ids: [batch, seq_len] (optional)

    Args:
        q, k:         query/key tensors.
        cos, sin:     RoPE tables from precompute_freqs_cis.
        position_ids: if provided, select positions per batch.
        unsqueeze_dim:
            which dim to unsqueeze for broadcasting to [b, n_heads, seq, dim].

    Returns:
        q_rotated, k_rotated: same shape as q, k.
    """
    if position_ids is not None:
        # position_ids: [b, seq] -> cos_sel: [b, seq, dim]
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        # assume cos/sin already sliced to [seq_len, dim]
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated
