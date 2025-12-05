import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Tuple, Optional


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Repeat key/value heads for Grouped-Query / Gated Attention.

    Input:
        x:     [batch, seq_len, num_kv_heads, head_dim]
        n_rep: number of query heads each kv head should serve

    Output:
        [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bsz, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (b, L, h_kv, d) -> (b, L, h_kv, n_rep, d) -> (b, L, h_kv * n_rep, d)
    return x[:, :, :, None, :].expand(
        bsz, seq_len, num_kv_heads, n_rep, head_dim
    ).reshape(
        bsz, seq_len, num_kv_heads * n_rep, head_dim
    )


class GatedAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with optional gating.

    - Q has `num_attention_heads` heads.
    - K/V have `num_kv_heads` heads (num_kv_heads <= num_attention_heads).
    - Each KV head is shared by `n_rep = num_attention_heads // num_kv_heads` Q heads.
    - Optional gating on the attention output:
        * element-wise gating: one gate per (head, feature) -> shape [b, L, h, d]
        * head-wise gating:    one gate per head        -> shape [b, L, h, 1]

    Args:
        hidden_size:         model dimension (d_model).
        num_attention_heads: total number of query heads.
        num_kv_heads:        number of key/value heads.
        element_wise_gating: whether to enable element-wise gating.
        head_wise_gating:    whether to enable head-wise gating.
        dropout_prob:        dropout probability on attention weights.
        bias:                whether to use bias in projection layers.

    Shapes:
        x:          [batch, seq_len, hidden_size]
        attn_bias:  broadcastable to [batch, num_attention_heads, seq_len, seq_len]
                    (e.g., 0 for keep, large negative for masked).

    Returns:
        attn_output:  [batch, seq_len, hidden_size]
        attn_weights: [batch, num_attention_heads, seq_len, seq_len]
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        element_wise_gating: bool = False,
        head_wise_gating: bool = False,
        dropout_prob: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert num_attention_heads % num_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"
        assert not (element_wise_gating and head_wise_gating), (
            "Only one type of gating can be enabled at a time."
        )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.element_wise_gating = element_wise_gating
        self.head_wise_gating = head_wise_gating

        self.head_dim = hidden_size // num_attention_heads
        self.n_rep = num_attention_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        # gating projection
        if element_wise_gating:
            # one gate per (head, feature)
            self.gate_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        elif head_wise_gating:
            # one gate per head
            self.gate_proj = nn.Linear(hidden_size, num_attention_heads, bias=bias)
        else:
            self.gate_proj = None

        # Q: per-head; K/V: grouped heads
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.size()

        # Project to Q, K, V
        # Q: [b, L, h_q, d_head]
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim)
        # KV: [b, L, h_kv, d_head]
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Optional gating scores from input x
        if self.gate_proj is not None:
            gate_scores = self.gate_proj(x)  # [b, L, h_q * d_head] or [b, L, h_q]

        # Scale Q
        q = q * self.scale

        # Repeat KV heads to match Q heads: [b, L, h_q, d_head]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # [b, h_q, L, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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

        # [b, L_q, h_q, d_head]
        attn_output = attn_output.transpose(1, 2)

        # Apply gating if enabled
        if self.gate_proj is not None:
            if self.element_wise_gating:
                # [b, L, h_q * d_head] -> [b, L, h_q, d_head]
                gate_scores = gate_scores.reshape(bsz, seq_len, self.num_attention_heads, self.head_dim)
                gate_values = torch.sigmoid(gate_scores)
            elif self.head_wise_gating:
                # [b, L, h_q] -> [b, L, h_q, 1]
                gate_scores = gate_scores.reshape(bsz, seq_len, self.num_attention_heads, 1)
                gate_values = torch.sigmoid(gate_scores)
            attn_output = attn_output * gate_values  # [b, L, h_q, d_head]

        # merge heads: [b, L, hidden_size]
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
