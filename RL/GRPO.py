import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 0) 工具：从 logits 取 token logprob
# ----------------------------
def token_logprobs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    logits:     [N, T, V]
    target_ids: [N, T]
    return:     [N, T]
    """
    logp_all = F.log_softmax(logits, dim=-1)
    token_logp = logp_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp

def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [N, T]
    mask: [N, T] (float 0/1)
    return: [N]
    """
    return (x * mask).sum(dim=-1)

# ----------------------------
# 1) 最简：Policy（只需要 LM logits）
# ----------------------------
class PolicyLM(nn.Module):
    """
    假设 backbone 输出 hidden_states: [N, T, H]
    lm_head -> logits: [N, T, V]
    """
    def __init__(self, backbone, hidden_size: int, vocab_size: int):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)  # [N, T, H]（占位）
        logits = self.lm_head(h)                      # [N, T, V]
        return logits

# ----------------------------
# 2) GRPO loss（核心）
# ----------------------------
def grpo_llm_loss(
    model: nn.Module,
    batch: dict,
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,
    use_kl_penalty: bool = True,
    group_whiten: bool = True,
):
    """
    batch:
      prompt_ids:    [B, P]
      response_ids:  [B, K, R]
      old_logp_seq:  [B, K]
      reward:        [B, K]
      ref_logp_seq:  [B, K]

    思想：
      - 对每个 prompt 的 K 条 response，计算新策略 logp_seq_new[b,k]
      - 组内优势：adv[b,k] = (reward[b,k] - mean_k) / (std_k + eps)   (或减均值即可)
      - PPO clip：ratio = exp(logp_new - logp_old)
      - loss_pi = -E[min(ratio*adv, clip(ratio)*adv)]
      - KL（可选）：E[logp_new - logp_ref] * kl_coef
    """
    prompt_ids   = batch["prompt_ids"]      # [B, P]
    response_ids = batch["response_ids"]    # [B, K, R]
    old_logp_seq = batch["old_logp_seq"]    # [B, K]
    reward       = batch["reward"]          # [B, K]
    ref_logp_seq = batch["ref_logp_seq"]    # [B, K]

    B, P = prompt_ids.shape
    _, K, R = response_ids.shape

    # ---- flatten group：把 (B,K) 展成 N=B*K，便于一次 forward ----
    N = B * K
    prompt_flat = prompt_ids.unsqueeze(1).expand(B, K, P).reshape(N, P)     # [N, P]
    resp_flat   = response_ids.reshape(N, R)                                 # [N, R]

    input_ids = torch.cat([prompt_flat, resp_flat], dim=1)                   # [N, T]
    attention_mask = torch.ones_like(input_ids)

    # response mask：只对 response 区间统计 logprob
    resp_mask = torch.zeros_like(input_ids, dtype=torch.float)
    resp_mask[:, P:] = 1.0                                                   # [N, T]

    # ---- forward logits ----
    logits = model(input_ids, attention_mask)                                 # [N, T, V]

    # ---- shift 计算 token logprob，并对 response 做 masked sum 得到 logp_seq_new ----
    logits_shift = logits[:, :-1, :]                                          # [N, T-1, V]
    ids_shift    = input_ids[:, 1:]                                           # [N, T-1]
    mask_shift   = resp_mask[:, 1:]                                           # [N, T-1]

    token_logp = token_logprobs_from_logits(logits_shift, ids_shift)          # [N, T-1]
    logp_seq_new_flat = masked_sum(token_logp, mask_shift)                    # [N]
    logp_seq_new = logp_seq_new_flat.view(B, K)                               # [B, K]

    # ---- group advantage（GRPO 核心：relative / whiten within group）----
    if group_whiten:
        mean = reward.mean(dim=1, keepdim=True)                               # [B,1]
        std  = reward.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        adv = (reward - mean) / std                                           # [B,K]
    else:
        adv = reward - reward.mean(dim=1, keepdim=True)                       # [B,K]
    adv = adv.detach()

    # ---- PPO-style clipped objective ----
    ratio = torch.exp(logp_seq_new - old_logp_seq)                            # [B,K]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # ---- KL penalty (to reference policy) ----
    if use_kl_penalty:
        # 序列层面近似 KL：E[logp_new - logp_ref]
        kl = torch.mean(logp_seq_new - ref_logp_seq)                          # scalar
        kl_penalty = kl_coef * kl
    else:
        kl = torch.tensor(0.0, device=logp_seq_new.device)
        kl_penalty = torch.tensor(0.0, device=logp_seq_new.device)

    loss = policy_loss + kl_penalty

    # ---- 监控指标 ----
    approx_kl = torch.mean(old_logp_seq - logp_seq_new).detach()
    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).detach()

    info = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "mean_reward": reward.mean().detach(),
        "mean_adv": adv.mean().detach(),
        "mean_logp_new": logp_seq_new.mean().detach(),
    }
    return loss, info

# ----------------------------
# 3) 最简更新步
# ----------------------------
def grpo_update_step(model, optimizer, batch):
    model.train()
    loss, info = grpo_llm_loss(
        model,
        batch=batch,
        clip_eps=0.2,
        kl_coef=0.1,
        use_kl_penalty=True,
        group_whiten=True,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return info
