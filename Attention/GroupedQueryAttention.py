import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Tuple, Optional


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 将 K 和 V 的头在维度上进行复制扩展（repeat），使其数量与 Q 头一致。
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # e.g., (bs, slen, num_key_value_heads, head_dim) -> (bs, slen, num_key_value_heads, n_rep, head_dim)  
        # 然后再reshape成 (bs, slen, num_key_value_heads * n_rep, head_dim)
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads,num_kv_heads,dropout_prob=0.1,bias = False):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert num_attention_heads % num_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.n_rep = num_attention_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_attention_heads*self.head_dim,
                                bias=bias)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_kv_heads*self.head_dim,
                                bias=bias)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_kv_heads*self.head_dim,
                                bias=bias)
        self.out_proj = nn.Linear(self.hidden_size,
                                  self.hidden_size,
                                  bias=bias)    
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self,
                x:Tensor,
                attn_bias:Optional[Tensor]=None
                ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.size()
        xq,xk,xv = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        xq = xq.reshape(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim
        )
        xk = xk.reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )
        xv = xv.reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )
        xq = xq * self.scale
        xk = repeat_kv(xk, self.n_rep)  # [b, h_q, L, d_head]
        xv = repeat_kv(xv, self.n_rep)  # [b, h_q, L, d_head]
        # Transpose to get the shape [batch_size, num_heads, seq_len, head_dim]
        xq = xq.transpose(1,2) # [b, h_q, L, d_head]
        xk = xk.transpose(1,2) # [b, h_kv, L, d_head]
        xv = xv.transpose(1,2) # [b, h_kv, L, d_head]
        

        attn_scores = einsum("bhqd,bhkd -> bhqk",xq,xk)
        if attn_bias is not None:
            # allow [b, q, k] or [b, 1, q, k] or [b, h, q, k]
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [b, 1, q, k]
            attn_scores = attn_scores + attn_bias
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum("bhqk,bhkd -> bhqd", attn_weights, xv)  # [b, h_q, L, d_head]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size,seq_len,self.hidden_size)
        return attn_output,attn_weights


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_attention_heads = 8
    num_kv_heads = 4

    model = GroupedQueryAttention(hidden_size, num_attention_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output, attn_weights = model(x)

    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)