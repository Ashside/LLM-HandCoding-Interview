import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Tuple


class CausalAttention(nn.Module):
    """
    Multi-head causal self-attention used in decoder-only LLM blocks.

    Args:
        hidden_size:  Model dimension (d_model).
        num_heads:    Number of attention heads.
        dropout_prob: Dropout probability applied to attention weights.

    Shapes:
        x:           [batch, seq_len, hidden_size]

    Returns:
        output:      [batch, seq_len, hidden_size]
        attn_probs:  [batch, num_heads, seq_len, seq_len]
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # bias-free projections are common in LLM implementations
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.size()

        # [b, L, d_model] -> [b, h, L, d_head]
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        q = q * self.scale  # [b, h, L, d_head]
        attn_scores = einsum("bhqd,bhkd->bhqk", q, k)  # [b, h, L, L]

        # build causal mask: [1, 1, L, L], 0 for keep, 1 for masked
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        # invert to "True = masked"
        causal_mask = ~causal_mask  # [L, L]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # apply mask: large negative value for masked positions
        attn_scores = attn_scores.masked_fill(causal_mask, -1e4)

        # softmax over key dimension
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # [b, h, L, d_head]
        attn_output = einsum("bhqk,bhkd->bhqd", attn_probs, v)
        # merge heads: [b, L, d_model]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs
