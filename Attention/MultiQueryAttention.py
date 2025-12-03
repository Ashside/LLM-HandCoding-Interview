import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Optional, Tuple


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) module.

    Q is projected per-head, while K/V are shared across heads:
        - Q: [b, h, q_len, d_head]
        - K: [b, 1, k_len, d_head] (shared across heads)
        - V: [b, 1, k_len, d_head] (shared across heads)

    Args:
        hidden_size:   model dimension (d_model).
        num_heads:     number of query heads.
        dropout_prob:  dropout probability on attention weights.
        bias:          whether to use bias in projection layers.

    Shapes:
        query:     [batch, q_len, hidden_size]
        key:       [batch, k_len, hidden_size]
        value:     [batch, k_len, hidden_size]
        attn_bias: broadcastable to [batch, num_heads, q_len, k_len]
                   (e.g., 0 for keep, large negative value for masked).

    Returns:
        attn_output:  [batch, q_len, hidden_size]
        attn_weights: [batch, num_heads, q_len, k_len]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Q has per-head parameters, K/V are shared across heads (MQA)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, q_len, _ = query.size()
        k_len = key.size(1)

        # Q: [b, h, q_len, d_head]
        q = self.q_proj(query).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * self.scale

        # K/V (shared across heads):
        # [b, k_len, d_head] -> [b, 1, k_len, d_head]
        k = self.k_proj(key).reshape(bsz, 1, k_len, self.head_dim)
        v = self.v_proj(value).reshape(bsz, 1, k_len, self.head_dim)

        # attention scores: [b, h, q_len, k_len]
        attn_scores = einsum("bhqd,bgkd->bhqk", q, k)

        if attn_bias is not None:
            # allow [b, q, k] or [b, 1, q, k] or [b, h, q, k]
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [b, 1, q, k]
            attn_scores = attn_scores + attn_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, h, q_len, d_head]
        attn_output = einsum("bhqk,bgkd->bhqd", attn_weights, v)

        # merge heads: [b, q_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

if __name__ == "__main__":
	batch_size = 2
	q_len = 5
	k_len = 7
	hidden_size = 16
	num_heads = 4

	mqa = MultiQueryAttention(hidden_size, num_heads, dropout_prob=0.1)

	query = torch.randn(batch_size, q_len, hidden_size)
	key = torch.randn(batch_size, k_len, hidden_size)
	value = torch.randn(batch_size, k_len, hidden_size)

	attn_output, attn_weights = mqa(query, key, value)

	print("Attention output shape:", attn_output.shape)  # Expected: [2, 5, 16]
	print("Attention weights shape:", attn_weights.shape)  # Expected: [2, 4, 5, 7]