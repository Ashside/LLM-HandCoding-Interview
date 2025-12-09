import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Tuple, Optional

from RoPE import apply_rotary_pos_emb
from RoPE import precompute_freqs_cis  # 一般会在外部调用，传进来 cos, sin


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Repeat key/value heads for Grouped-Query / Multi-Query Attention.

    Input:
        x:     [batch, seq_len, num_kv_heads, head_dim]
        n_rep: number of query heads each kv head should serve

    Output:
        [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        bs, slen, num_kv_heads, n_rep, head_dim
    ).reshape(
        bs, slen, num_kv_heads * n_rep, head_dim
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with RoPE + KV cache.

    Q has `num_attention_heads`, K/V have `num_kv_heads` (< num_attention_heads).
    Each KV head is shared by `n_rep = num_attention_heads // num_kv_heads` query heads.

    Args:
        hidden_size:         model dimension (d_model).
        num_attention_heads: total number of query heads.
        num_kv_heads:        number of key/value heads.
        dropout_prob:        dropout probability on attention weights.
        bias:                whether to use bias in projection layers.

    Shapes:
        x:           [batch, seq_len, hidden_size]
        freqs_cis:   (cos, sin), each [max_seq_len, head_dim]
        position_ids:[batch, seq_len] or None
        past_kv:     (k, v), each [batch, past_len, num_kv_heads, head_dim]
        attn_bias:   broadcastable to [batch, num_attention_heads, L_q, L_k]

    Returns:
        attn_output: [batch, seq_len, hidden_size]
        attn_weights:[batch, num_attention_heads, L_q, L_k]
        present_kv:  (k, v) or None
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        dropout_prob: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert num_attention_heads % num_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_size // num_attention_heads
        self.n_rep = num_attention_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Q: per-head; K/V: grouped heads
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        x:           [b, L, d_model]
        freqs_cis:   (cos, sin), each [max_seq_len, head_dim]
        position_ids:[b, L] or None（None 时通常表示 0..L-1）
        """
        bsz, seq_len, _ = x.size()
        cos, sin = freqs_cis  # [max_seq_len, head_dim]

        # project to Q, K, V
        # Q: [b, L, h_q, d_head]
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim)
        # K/V: [b, L, h_kv, d_head]
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # RoPE expects [b, h, L, d]
        q = q.transpose(1, 2)  # [b, h_q, L, d]
        k = k.transpose(1, 2)  # [b, h_kv, L, d]

        # apply RoPE on current token positions
        # 注意：past_kv 里假定已经是旋转后的 K/V，这里只对当前 step 的 q/k 做 RoPE
        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            position_ids=position_ids,  # [b, L] or None
            unsqueeze_dim=1,            # cos/sin -> [b, 1, L, d]
        )

        # back to [b, L, h, d]
        q = q.transpose(1, 2)  # [b, L, h_q, d]
        k = k.transpose(1, 2)  # [b, L, h_kv, d]

        # append cached KV if provided (cached already rotated)
        if past_kv is not None:
            past_k, past_v = past_kv  # [b, L_past, h_kv, d]
            k = torch.cat([past_k, k], dim=1)  # [b, L_total, h_kv, d]
            v = torch.cat([past_v, v], dim=1)  # [b, L_total, h_kv, d]

        if use_cache:
            present_kv: Optional[Tuple[Tensor, Tensor]] = (k, v)
        else:
            present_kv = None

        # scale Q
        q = q * self.scale  # [b, L_q, h_q, d]

        # repeat KV heads to match Q heads: [b, L_k, h_q, d]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # transpose to [b, h_q, L, d]
        q = q.transpose(1, 2)  # [b, h_q, L_q, d]
        k = k.transpose(1, 2)  # [b, h_q, L_k, d]
        v = v.transpose(1, 2)  # [b, h_q, L_k, d]

        # attention scores: [b, h_q, L_q, L_k]
        attn_scores = einsum("bhqd,bhkd->bhqk", q, k)

        if attn_bias is not None:
            # allow [b, q, k] or [b, 1, q, k] or [b, h, q, k]
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [b, 1, q, k]
            attn_scores = attn_scores + attn_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, h_q, L_q, d_head]
        attn_output = einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # merge heads: [b, L_q, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, present_kv
