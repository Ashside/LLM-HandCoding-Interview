import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """
    SwiGLU MLP block used in many modern LLMs.

    Given input x in R^{..., d_model}, computes:
        y = down_proj( SiLU(x W_g) ⊙ (x W_u) )

    The intermediate size is set to floor(8/3 * hidden_size), then
    rounded up to a multiple of 64 for better kernel efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        bias: bool = False,
        multiple_of: int = 64,
    ) -> None:
        super().__init__()

        # 8/3 * d_model, then round to multiple_of
        intermediate_size = hidden_size * 8 // 3
        if multiple_of is not None and multiple_of > 0:
            intermediate_size = (intermediate_size + multiple_of - 1) // multiple_of * multiple_of

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # [b, L, d] -> [b, L, d_ff]
        gate = self.gate_proj(x)
        gate = F.silu(gate)

        up = self.up_proj(x)

        # SwiGLU: SiLU(W_g x) * (W_u x)
        hidden = gate * up

        # project back to hidden_size
        out = self.down_proj(hidden)
        return out
