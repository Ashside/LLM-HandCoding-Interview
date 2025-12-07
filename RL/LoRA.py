import torch
from torch import nn
from typing import Optional, Dict


class LoRALayer(nn.Module):
    """
    A basic LoRA module: y = B(A(x)) * (alpha / r)

    Args:
        in_features:  input dimension of the base Linear
        out_features: output dimension of the base Linear
        rank:         LoRA rank (r)
        alpha:        LoRA scaling factor (often = rank)
        bias:         whether to use bias in A/B (usually False in LLMs)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: Optional[float] = None,
        bias: bool = False,
        mean: float = 0.0,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        self.A = nn.Linear(in_features, rank, bias=bias)
        self.B = nn.Linear(rank, out_features, bias=bias)

        # init: A ~ N(mean, std), B = 0 -> LoRA 初始为 0
        nn.init.normal_(self.A.weight, mean=mean, std=std)
        nn.init.zeros_(self.B.weight)
        if bias:
            nn.init.zeros_(self.A.bias)
            nn.init.zeros_(self.B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling
def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: Optional[float] = None,
    bias: bool = False,
) -> None:
    """
    Attach LoRA modules to all nn.Linear in the model.
    This will *add* LoRA(x) to the original Linear forward, and freeze
    the original Linear weights by default.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "lora"):
                # 已经打过 LoRA 了，跳过
                continue

            lora = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
                bias=bias,
            )
            # 将 LoRA 模块挂在 Linear 上
            module.lora = lora

            # 冻结原始权重，只训练 LoRA（符合大模型微调习惯）
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

            # 保存原始 forward
            module._orig_forward = module.forward

            def forward_lora(x, base_layer=module._orig_forward, lora_layer=module.lora):
                return base_layer(x) + lora_layer(x)

            module.forward = forward_lora
def remove_lora(model: nn.Module) -> None:
    """
    Remove LoRA hooks and restore original Linear.forward.
    Does NOT undo the training effect on LoRA params themselves;
    it just stops using them in forward.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            # 恢复 forward
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
                delattr(module, "_orig_forward")

            # 删除 LoRA 模块
            delattr(module, "lora")

            # 你可以选择是否解冻原始权重
            module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
def save_lora_state_dict(model: nn.Module, path: str) -> None:
    """
    Save only LoRA parameters into a separate state dict.
    Keys will look like: "{module_name}.lora.A.weight", etc.
    """
    save_state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            for k, v in module.lora.state_dict().items():
                save_state[f"{name}.lora.{k}"] = v
    torch.save(save_state, path)


def load_lora_state_dict(model: nn.Module, path: str, device: Optional[torch.device] = None) -> None:
    """
    Load LoRA parameters into the model.
    Assumes apply_lora() has already been called with matching rank/alpha/etc.
    """
    if device is None:
        # fallback: use first parameter's device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    state_dict = torch.load(path, map_location=device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            prefix = f"{name}.lora."
            # 取出该模块对应的所有 LoRA 参数
            lora_state = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            module.lora.load_state_dict(lora_state, strict=True)
