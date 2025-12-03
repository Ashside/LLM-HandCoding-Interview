import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor,einsum
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    Allows for different Query, Key, and Value inputs, enabling both Self-Attention and Cross-Attention.

    Args:
        hidden_size: Dimension of the model (d_model).
        num_heads:   Number of attention heads.
        dropout_prob: Dropout probability applied on attention weights.
        bias:        Whether to use bias in linear projections.

    Shapes:
        query:      [batch, seq_len_q, hidden_size]
        key:        [batch, seq_len_k, hidden_size]
        value:      [batch, seq_len_k, hidden_size]
        attn_mask:  [batch, num_heads, seq_len_q, seq_len_k] or broadcastable.
                    Typically 0.0 for visible positions and -inf for masked positions.

    Returns:
        attn_output:  [batch, seq_len_q, hidden_size]
        attn_weights: [batch, num_heads, seq_len_q, seq_len_k]
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Project and reshape to [batch, seq_len, num_heads, head_dim]
        # Then transpose to [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores: [batch, num_heads, seq_len_q, seq_len_k]
        attn_scores = einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: [batch, num_heads, seq_len_q, head_dim]
        attn_output = einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Transpose back and reshape: [batch, seq_len_q, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)

        output = self.out_proj(attn_output)

        return output, attn_weights

if __name__ == "__main__":
	# Example usage
	batch_size = 2
	seq_len = 4
	hidden_size = 8
	num_heads = 2

	mha = MultiHeadAttention(hidden_size, num_heads, dropout_prob=0.1)
	x = torch.randn(batch_size, seq_len, hidden_size)

	output, attn_weights = mha(x, x, x)
	print("Output shape:", output.shape)  # Expected: [2, 4, 8]
	print("Attention weights shape:", attn_weights.shape)  # Expected: [2, 2, 4, 4]