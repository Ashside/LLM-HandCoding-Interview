import math

def cosine_lr_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Cosine decay learning rate schedule with linear warmup.

    Args:
        step: current step (0-indexed)
        max_lr: peak LR achieved after warmup
        min_lr: final LR after cosine decay
        warmup_steps: number of warmup iterations
        total_steps: number of iterations of cosine schedule

    Returns:
        lr at current step
    """
    # Safety guards
    warmup_steps = max(0, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    # 1. warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 2. cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_lr + (max_lr - min_lr) * cosine_decay

    return lr