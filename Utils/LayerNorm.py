import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    A numerically stable LayerNorm implementation commonly used in large language models.
    Computes mean/var in float32 regardless of input precision.
    Equivalent to torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine=True)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute mean / var in float32 for stability
        float_x = x.to(torch.float32)

        mean = float_x.mean(dim=-1, keepdim=True)
        var = float_x.var(dim=-1, keepdim=True, unbiased=False)

        y = (float_x - mean) / torch.sqrt(var + self.eps)

        # cast back to original dtype
        y = y.to(x.dtype)

        return y * self.weight + self.bias
