import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 0) 工具：从 logits 取 token logprob
# ----------------------------
def token_logprobs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    logits:     [B, T, V]
    target_ids: [B, T]
    return:     [B, T]  每个位置 target token 的 logprob
    """
    logp_all = F.log_softmax(logits, dim=-1)                     # [B, T, V]
    token_logp = logp_all.gather(-1, target_ids.unsqueeze(-1))   # [B, T, 1]
    return token_logp.squeeze(-1)                                # [B, T]


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [B, T]
    mask: [B, T]  1 表示有效 token（通常只对 response 部分计入）
    return: [B]
    """
    return (x * mask).sum(dim=-1)


# ----------------------------
# 1) 最简：Policy +（可选）Value head
# ----------------------------
class PolicyWithValue(nn.Module):
    """
    伪代码结构：假设你有一个大模型 backbone（例如 transformer），输出 hidden_states/logits。
    - policy logits 用 LM head
    - value 用一个线性头从 hidden state 预测每个 token 的 V(s_t)，再做 mask 聚合得到 V(seq)
    面试时：你也可以直接去掉 value，只做 policy + KL penalty（但 PPO 正统一般会有 value）。
    """
    def __init__(self, backbone, hidden_size: int, vocab_size: int):
        super().__init__()
        self.backbone = backbone          # 假设: forward(input_ids, attention_mask) -> hidden_states [B,T,H]
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.v_head  = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)   # [B, T, H]（占位）
        logits = self.lm_head(h)                        # [B, T, V]
        values = self.v_head(h).squeeze(-1)             # [B, T]
        return logits, values


# ----------------------------
# 2) PPO(clip) for LLM：核心 loss
# ----------------------------
def ppo_llm_loss(
    model: nn.Module,
    batch: dict,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    kl_coef: float = 0.1,
    ent_coef: float = 0.0,       # LLM RLHF 通常不显式加 entropy（有 KL 约束即可），这里保留接口
    whiten_adv: bool = True,
):
    """
    batch:
      prompt_ids:     [B, P]
      response_ids:   [B, R]
      old_logp_seq:   [B]        # 旧策略对整段 response 的 logprob 求和（已 mask）
      reward:         [B]        # 标量奖励（可包含 RM + KL shaping 后结果；这里假设已给出）
      ref_logp_seq:   [B]        # 参考策略对整段 response 的 logprob 求和（已 mask）

    实现选择：
      - 将 (prompt, response) 拼接成 input_ids
      - 计算新策略 logp_seq（对 response token logp 求和）
      - ratio = exp(logp_new - logp_old)
      - advantage：这里用最简 A = reward - V(seq)（实际通常用 GAE / per-token reward）
      - KL penalty：近似 KL ≈ (logp_new - logp_ref)（在序列层面）
    """
    prompt_ids   = batch["prompt_ids"]        # [B, P]
    response_ids = batch["response_ids"]      # [B, R]
    old_logp_seq = batch["old_logp_seq"]      # [B]
    reward       = batch["reward"]            # [B]
    ref_logp_seq = batch["ref_logp_seq"]      # [B]

    B, P = prompt_ids.shape
    _, R = response_ids.shape

    # ---- 拼接输入：input = [prompt, response] ----
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)     # [B, P+R]

    # attention_mask：这里最简假设全 1；真实场景需考虑 padding
    attention_mask = torch.ones_like(input_ids)

    # response mask：只对 response 部分的 token 计算 logprob / value 聚合
    resp_mask = torch.zeros_like(input_ids, dtype=torch.float)
    resp_mask[:, P:] = 1.0                                       # [B, P+R]

    # ---- forward：得到 logits & values ----
    logits, values_tok = model(input_ids, attention_mask)        # logits [B, T, V], values_tok [B, T]
    T = logits.size(1)

    # ---- 计算 response token logprob ----
    # teacher forcing：logits[t] 预测 token_ids[t]（常见做法是 shift）
    # 这里写最“看起来能跑”的 shift 版本：
    logits_shift = logits[:, :-1, :]             # [B, T-1, V]
    ids_shift    = input_ids[:, 1:]              # [B, T-1]
    mask_shift   = resp_mask[:, 1:]              # [B, T-1] 对齐 target token

    token_logp = token_logprobs_from_logits(logits_shift, ids_shift)   # [B, T-1]
    logp_seq_new = masked_sum(token_logp, mask_shift)                  # [B] 只汇总 response 区间

    # ---- value：序列级 V(seq)（最简：对 response token 的 value 做平均 or sum）
    # 这里用 masked mean 更稳定一些
    values_shift = values_tok[:, 1:]                                # [B, T-1]
    denom = mask_shift.sum(dim=-1).clamp_min(1.0)                   # [B]
    v_seq = (values_shift * mask_shift).sum(dim=-1) / denom         # [B]

    # ---- advantage（最简形态）----
    adv = (reward - v_seq).detach()                                 # [B]
    if whiten_adv:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # ---- PPO clipped policy loss ----
    ratio = torch.exp(logp_seq_new - old_logp_seq)                  # [B]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # ---- value loss（拟合 reward / return；这里只用 reward 当 target）----
    value_loss = 0.5 * F.mse_loss(v_seq, reward)

    # ---- KL penalty（序列层面近似）----
    # 常见近似：KL ≈ E[logp_new - logp_ref]，这里直接用序列求和后的差
    kl = torch.mean(logp_seq_new - ref_logp_seq)                    # scalar
    kl_penalty = kl_coef * kl

    # ---- entropy（可选，通常 RLHF 不强依赖）----
    # 如果要写“token-level entropy”，需要对 logits_shift 做 softmax 并计算分布熵，再 mask 平均；
    # 面试中往往不写也可以。
    entropy_bonus = torch.tensor(0.0, device=logits.device)

    # ---- total loss ----
    loss = policy_loss + vf_coef * value_loss + kl_penalty - ent_coef * entropy_bonus

    # ---- 常用监控指标 ----
    approx_kl = torch.mean(old_logp_seq - logp_seq_new).detach()
    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).detach()

    info = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "mean_reward": reward.mean().detach(),
        "mean_logp_new": logp_seq_new.mean().detach(),
    }
    return loss, info


# ----------------------------
# 3) 最简更新步
# ----------------------------
def ppo_llm_update_step(model, optimizer, batch):
    model.train()
    loss, info = ppo_llm_loss(
        model,
        batch=batch,
        clip_eps=0.2,
        vf_coef=0.5,
        kl_coef=0.1,
        ent_coef=0.0,
        whiten_adv=True,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return info
