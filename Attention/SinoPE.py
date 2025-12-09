import math
import torch
from torch import Tensor


def sinusoidal_position_embedding(
    dim: int,
    max_len: int = 512,
    *,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    return_batch_dim: bool = False
) -> Tensor:
    """
    Generate standard sinusoidal positional embeddings (absolute PE).

    Args:
        dim:     embedding dimension (must be even).
        max_len: maximum sequence length.
        device:  target device.
        dtype:   target dtype.
        return_batch_dim:
            If True, return shape [1, max_len, dim] instead of [max_len, dim].

    Returns:
        Tensor of shape:
            - [max_len, dim]
            - or [1, max_len, dim] if return_batch_dim=True
    """
    assert dim % 2 == 0, "Sinusoidal PE dimension must be even."

    if device is None:
        device = torch.device("cpu")

    half_dim = dim // 2
    positions = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)  # (max_len, 1)

    # Standard formula: exp(i * (-log(10000) / half_dim))
    div_term = torch.exp(
        torch.arange(half_dim, device=device, dtype=dtype)
        * -(math.log(10000.0) / half_dim)
    )  # (half_dim,)

    pe = torch.zeros(max_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    if return_batch_dim:
        return pe.unsqueeze(0)  # shape [1, max_len, dim]

    return pe
