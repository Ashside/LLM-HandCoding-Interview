import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Tuple, Optional


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
    # (b, L, h_kv, d) -> (b, L, h_kv, n_rep, d) -> (b, L, h_kv * n_rep, d)
    return x[:, :, :, None, :].expand(
        bs, slen, num_kv_heads, n_rep, head_dim
        ).reshape(
        bs, slen, num_kv_heads * n_rep, head_dim
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Q has `num_attention_heads`, K/V have `num_kv_heads` (< num_attention_heads).
    Each KV head is shared by `n_rep = num_attention_heads // num_kv_heads` query heads.

    Args:
        hidden_size:         model dimension (d_model).
        num_attention_heads: total number of query heads.
        num_kv_heads:        number of key/value heads.
        dropout_prob:        dropout probability on attention weights.
        bias:                whether to use bias in projection layers.

    Shapes:
        x:          [batch, seq_len, hidden_size]
        attn_bias:  broadcastable to [batch, num_attention_heads, seq_len, seq_len]
                    e.g. 0 for keep, large negative value for masked.

    Returns:
        attn_output:  [batch, seq_len, hidden_size]
        attn_weights: [batch, num_attention_heads, seq_len, seq_len]
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
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.size()

        # project to Q, K, V
        # Q: [b, L, h_q, d_head]
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_attention_heads, self.head_dim)
        # KV: [b, L, h_kv, d_head]
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Scale Q
        q = q * self.scale

        # Repeat KV heads to match Q heads: [b, L, h_q, d_head]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Transpose to [b, h_q, L, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores: [b, h_q, L_q, L_k]
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

        # Merge heads: [b, L_q, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 4
    hidden_size = 16
    num_attention_heads = 4
    num_kv_heads = 2

    gqa = GroupedQueryAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        dropout_prob=0.1,
        bias=True,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    attn_output, attn_weights = gqa(x)

    print("Attention output shape:", attn_output.shape)  # Expected: [2, 4, 16]
    print("Attention weights shape:", attn_weights.shape)  # Expected: [2, 4, 4, 4]
    