import torch
def attention_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax specifically used for attention.
    Automatically handles very negative masked values (e.g., -inf or -1e4).
    """
    # subtract max
    x = x - x.max(dim=dim, keepdim=True).values

    # clamp to avoid exp underflow -> nan
    x = torch.clamp(x, min=-50.0)

    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
