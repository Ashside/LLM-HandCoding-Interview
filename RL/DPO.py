import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 0) 工具：从 logits 取 token logprob + mask 汇总
# ----------------------------
def token_logprobs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    logits:     [N, T, V]
    target_ids: [N, T]
    return:     [N, T]
    """
    logp_all = F.log_softmax(logits, dim=-1)
    return logp_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [N, T]
    mask: [N, T] float 0/1
    return: [N]
    """
    return (x * mask).sum(dim=-1)


# ----------------------------
# 1) 最简 Policy LM：只需要 logits
# ----------------------------
class PolicyLM(nn.Module):
    def __init__(self, backbone, hidden_size: int, vocab_size: int):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)  # [N, T, H]（占位）
        logits = self.lm_head(h)                      # [N, T, V]
        return logits


# ----------------------------
# 2) 计算 policy 对 response 的序列 logprob（prompt 部分 mask 掉）
# ----------------------------
def seq_logp_for_response(model, prompt_ids, response_ids):
    """
    prompt_ids:   [B, P]
    response_ids: [B, R]
    return:       [B] 序列 logprob（对 response token logprob 求和）
    """
    B, P = prompt_ids.shape
    _, R = response_ids.shape

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, T]
    attention_mask = torch.ones_like(input_ids)

    # mask：只统计 response 区间（对齐 shift 后 target token）
    resp_mask = torch.zeros_like(input_ids, dtype=torch.float)
    resp_mask[:, P:] = 1.0

    logits = model(input_ids, attention_mask)                 # [B, T, V]

    # shift
    logits_shift = logits[:, :-1, :]                          # [B, T-1, V]
    ids_shift    = input_ids[:, 1:]                           # [B, T-1]
    mask_shift   = resp_mask[:, 1:]                           # [B, T-1]

    token_logp = token_logprobs_from_logits(logits_shift, ids_shift)  # [B, T-1]
    logp_seq = masked_sum(token_logp, mask_shift)                     # [B]
    return logp_seq


# ----------------------------
# 3) DPO loss（核心）
# ----------------------------
def dpo_loss(
    policy_model: nn.Module,
    batch: dict,
    beta: float = 0.1,
    label_smoothing: float = 0.0,   # 可选：工业里有时会加
):
    """
    batch:
      prompt_ids:         [B, P]
      chosen_ids:         [B, R]
      rejected_ids:       [B, R]
      ref_chosen_logp:    [B]   # ref policy 的 logp(chosen)
      ref_rejected_logp:  [B]   # ref policy 的 logp(rejected)

    DPO 核心：
      pi: policy, ref: reference
      Δpi  = log pi(y_w|x) - log pi(y_l|x)
      Δref = log ref(y_w|x) - log ref(y_l|x)
      logits = beta * (Δpi - Δref)
      loss = -log σ(logits)
    """
    prompt_ids   = batch["prompt_ids"]
    chosen_ids   = batch["chosen_ids"]
    rejected_ids = batch["rejected_ids"]

    ref_chosen_logp   = batch["ref_chosen_logp"]
    ref_rejected_logp = batch["ref_rejected_logp"]

    # policy 的序列 logp
    pi_chosen_logp   = seq_logp_for_response(policy_model, prompt_ids, chosen_ids)     # [B]
    pi_rejected_logp = seq_logp_for_response(policy_model, prompt_ids, rejected_ids)   # [B]

    # Δpi / Δref
    delta_pi  = pi_chosen_logp - pi_rejected_logp
    delta_ref = ref_chosen_logp - ref_rejected_logp

    # DPO logits
    logits = beta * (delta_pi - delta_ref)  # [B]

    # 标准 DPO：-log(sigmoid(logits))
    # 若加 label smoothing：把“偏好 chosen”从 1 调成 (1-ε)，等价于混合两种方向的 log-sigmoid
    if label_smoothing > 0.0:
        eps = label_smoothing
        loss_pos = -F.logsigmoid(logits)        # y=1
        loss_neg = -F.logsigmoid(-logits)       # y=0
        loss = (1 - eps) * loss_pos + eps * loss_neg
        loss = loss.mean()
    else:
        loss = (-F.logsigmoid(logits)).mean()

    # 常用监控：偏好准确率（logits>0 代表 policy 相比 ref 更偏 chosen）
    with torch.no_grad():
        pref_acc = (logits > 0).float().mean()
        margin = (delta_pi - delta_ref).mean()

    info = {
        "loss": loss.detach(),
        "pref_acc": pref_acc.detach(),
        "mean_logit": logits.mean().detach(),
        "mean_margin": margin.detach(),
        "pi_chosen_logp": pi_chosen_logp.mean().detach(),
        "pi_rejected_logp": pi_rejected_logp.mean().detach(),
    }
    return loss, info


# ----------------------------
# 4) 最简更新步
# ----------------------------
def dpo_update_step(policy_model, optimizer, batch):
    policy_model.train()
    loss, info = dpo_loss(
        policy_model,
        batch=batch,
        beta=0.1,
        label_smoothing=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()
    return info
