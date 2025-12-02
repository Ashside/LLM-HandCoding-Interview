import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Multi-head self-attention used in Transformer/LLM blocks.

    Args:
        hidden_size: Dimension of the model (d_model).
        num_heads:   Number of attention heads.
        dropout_prob: Dropout probability applied on attention weights.

    Shapes:
        x:          [batch, seq_len, hidden_size]
        attn_bias:  broadcastable to [batch, num_heads, seq_len, seq_len],
                    typically 0.0 for visible positions and a large negative
                    value (e.g. -1e4 or -inf) for masked positions.

    Returns:
        attn_output:  [batch, seq_len, hidden_size]
        attn_weights: [batch, num_heads, seq_len, seq_len]
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        head_dim = hidden_size // num_heads
        assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.head_dim = head_dim

        # scale factor for QK^T
        self.scale = head_dim ** -0.5

        # In LLMs, projections are often bias-free for numerical stability & simplicity
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.size()

        # project to Q, K, V
        # [b, L, d_model] -> [b, h, L, d_head]
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        q = q * self.scale  # [b, h, L, d_head]

        # attention scores: [b, h, L, L]
        attn_scores = einsum("bhqd,bhkd->bhqk", q, k)

        if attn_bias is not None:
            # common cases:
            #  - [b, 1, L, L] (shared across heads)
            #  - [b, L, L]    (no head dimension)
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [b, 1, L, L]
            attn_scores = attn_scores + attn_bias

        # [b, h, L, L]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, h, L, d_head]
        attn_output = einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # merge heads: [b, L, d_model]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    hidden_size = 32
    num_heads = 4

    self_attn = SelfAttention(hidden_size, num_heads, dropout_prob=0.1)

    x = torch.randn(batch_size, seq_len, hidden_size)
    attn_output, attn_weights = self_attn(x)

    print("Self-Attention Output Shape:", attn_output.shape)  # [2, 8, 32]
    print("Self-Attention Weights Shape:", attn_weights.shape)  # [2, 4, 8, 8]
    