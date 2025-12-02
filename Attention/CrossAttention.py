import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    单头交叉注意力机制。

    Args:
        hidden_size: Dimension of the input and output features (d_model).
        dropout_prob: Dropout probability applied to attention weights.

    Shapes:
        query:     [batch, q_len, hidden_size]
        key:       [batch, k_len, hidden_size]
        value:     [batch, k_len, hidden_size]
        attn_bias: broadcastable to [batch, q_len, k_len],
                   typically 0.0 for visible positions and a large negative
                   value (e.g. -1e4) for masked positions.

    Returns:
        attn_output:  [batch, q_len, hidden_size]
        attn_weights: [batch, q_len, k_len]
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        # In many LLM implementations, projections are bias-free.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # [b, q_len, d]
        q = self.q_proj(query) * self.scale
        # [b, k_len, d]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # attention scores: [b, q_len, k_len]
        attn_scores = einsum("bqd,bkd->bqk", q, k)

        if attn_bias is not None:
            # attn_bias is typically 0 for keep and -1e4 / -inf for masked
            attn_scores = attn_scores + attn_bias

        # [b, q_len, k_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, q_len, d]
        attn_output = einsum("bqk,bkd->bqd", attn_weights, v)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
# Example usage:

if __name__ == "__main__":
    batch_size = 2
    q_len = 5
    k_len = 7
    hidden_size = 16

    cross_attn = CrossAttention(hidden_size, dropout_prob=0.1)

    query = torch.randn(batch_size, q_len, hidden_size)
    key = torch.randn(batch_size, k_len, hidden_size)
    value = torch.randn(batch_size, k_len, hidden_size)
    attn_bias = torch.zeros(batch_size, q_len, k_len)  # No masking

    attn_output, attn_weights = cross_attn(query, key, value, attn_bias)

    print("Attention Output Shape:", attn_output.shape)  # [2, 5, 16]
    print("Attention Weights Shape:", attn_weights.shape)  # [2, 5, 7]