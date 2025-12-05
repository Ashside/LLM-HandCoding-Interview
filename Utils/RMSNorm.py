import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in modern LLMs: LLaMA, Mistral, DeepSeek, Falcon, GPT-J...
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # compute RMSNorm using float32 for numerical stability
        float_x = x.to(torch.float32)
        rms = float_x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = float_x * torch.rsqrt(rms + self.eps)
        return x_normed.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x)
