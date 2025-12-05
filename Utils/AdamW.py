import math
from typing import Optional, Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    A minimal AdamW implementation, suitable for LLM training code.

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate.
        betas: coefficients used for computing running averages of gradient and its square.
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled weight decay (L2).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    # p = p - lr * weight_decay * p
                    # in-place operation to avoid extra memory allocation
                    p.add_(p, alpha=-lr * weight_decay)

                # Update biased first and second moment estimates
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute step size
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update
                # p = p - step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                # p += -step_size * exp_avg / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
