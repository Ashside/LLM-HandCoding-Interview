from typing import Iterable
import torch
import math

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """
    Gradient clipping by global L2 norm.
    Equivalent to torch.nn.utils.clip_grad_norm_(params, max_l2_norm).

    Args:
        parameters: model.parameters()
        max_l2_norm: maximum allowed L2 norm of gradients
        eps: numerical stability term
    """

    # Convert generator to list to allow multiple loops
    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return 0.0

    # Compute total L2 norm in float32 for stability
    total_norm = torch.sqrt(
        sum((p.grad.detach().float().pow(2).sum() for p in params))
    ).item()

    # Compute clipping coefficient
    coef = max_l2_norm / (total_norm + eps)

    # Apply global clipping
    if coef < 1.0:
        for p in params:
            p.grad.mul_(coef)

    return total_norm
