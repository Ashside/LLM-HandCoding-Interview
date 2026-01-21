import torch
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid function.
    """
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(x))
    )