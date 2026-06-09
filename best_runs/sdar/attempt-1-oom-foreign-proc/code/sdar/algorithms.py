"""SDAR + five baseline algorithms (arXiv 2605.15155, Algorithms 2–6 + SDAR).

All training functions share the same signature so the training loop in
sdar/train.py can dispatch to them by name.

Algorithm implementations:
  - GRPO     (Algorithm 2): clipped surrogate with group-relative advantages
  - OPSD     (Algorithm 3): standalone on-policy self-distillation
  - Skill-SD (Algorithm 4): skill-conditioned self-distillation
  - GRPO+OPSD(Algorithm 5): naive sum without sigmoid gate (unstable)
  - RLSD     (Algorithm 6): RL with self-distillation
  - SDAR     (main paper): GRPO + lambda * gated OPSD

Invariants:
  BETA   = 10.0  (sigmoid sharpness)
  LAMBDA = 0.1   (distillation coefficient)
  gate   = sigmoid(BETA * delta_t).detach()  [stop-gradient!]
  loss   = grpo_loss + LAMBDA * opsd_loss
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from .gating import make_gate
from .utils import compute_token_logp, compute_group_advantages

# ── Paper constants ────────────────────────────────────────────────────────────
BETA: float = 10.0    # sigmoid sharpness (β in paper)
LAMBDA: float = 0.1   # distillation coefficient (λ in paper)
EPS: float = 0.2      # PPO clip ratio (ε in paper)


# ══════════════════════════════════════════════════════════════════════════════
# Loss primitives
# ══════════════════════════════════════════════════════════════════════════════

def grpo_loss_fn(
    logp_new: List[torch.Tensor],
    logp_old: List[torch.Tensor],
    advantages: List[float],
    eps: float = EPS,
) -> torch.Tensor:
    """GRPO clipped surrogate loss (Algorithm 2).

    For each rollout i and each token t:
      ratio_t   = exp(logp_new_t - logp_old_t)
      loss_t    = -min(ratio_t * A_i, clamp(ratio_t, 1-ε, 1+ε) * A_i)

    Returns mean over all rollouts and tokens.
    """
    total = torch.tensor(0.0)
    n_terms = 0

    for i, (lp_new, lp_old, a_i) in enumerate(zip(logp_new, logp_old, advantages)):
        if lp_new is None or lp_old is None:
            continue
        device = lp_new.device
        total = total.to(device)

        lp_old_i = lp_old.detach().to(device)
        ratio = torch.exp(lp_new - lp_old_i)
        a_tensor = torch.tensor(a_i, dtype=lp_new.dtype, device=device)

        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        token_loss = -torch.min(ratio * a_tensor, clipped * a_tensor)
        total = total + token_loss.mean()
        n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, requires_grad=True)
    return total / n_terms


def opsd_loss_fn(
    logp_student: List[torch.Tensor],
    logp_teacher: List[torch.Tensor],
    beta: float = BETA,
    gate_strategy: str = "gap",
    student_logits: Optional[List[Optional[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, float]:
    """OPSD loss: gated token-level reverse KL (Section 2.1).

    Computes: L_OPSD = -mean_t [ g_t * log π_student(y_t) ]
    where g_t = sigmoid(β·Δ_t).detach()
    and Δ_t = log π_teacher(y_t) − log π_student(y_t)

    Returns:
        opsd_loss: scalar tensor with gradient
        gate_active_ratio: fraction of tokens where gate > 0.5 (monitoring metric)
    """
    total = torch.tensor(0.0)
    total_gate_active = 0.0
    n_terms = 0

    for i, (lp_s, lp_t) in enumerate(zip(logp_student, logp_teacher)):
        if lp_s is None or lp_t is None:
            continue
        device = lp_s.device
        total = total.to(device)

        logits_i = student_logits[i] if student_logits else None

        # Compute gate (stop-gradient applied inside make_gate)
        gate = make_gate(
            gate_strategy,
            lp_s,
            lp_t.detach().to(device),
            logits=logits_i,
            beta=beta,
        )  # gate is detached

        # OPSD loss: gradient flows only through lp_s
        loss_i = -(gate * lp_s).mean()
        total = total + loss_i
        n_terms += 1

        total_gate_active += (gate > 0.5).float().mean().item()

    if n_terms == 0:
        return torch.tensor(0.0, requires_grad=True), 0.0

    gate_ratio = total_gate_active / n_terms
    return total / n_terms, gate_ratio


def opsd_loss_ungated(
    logp_student: List[torch.Tensor],
    logp_teacher: List[torch.Tensor],
) -> torch.Tensor:
    """Ungated OPSD for GRPO+OPSD baseline (Algorithm 5).

    Simply uses the teacher-student KL without the sigmoid gate.
    This is the "naive sum" that causes instability in the paper.
    """
    total = torch.tensor(0.0)
    n_terms = 0

    for lp_s, lp_t in zip(logp_student, logp_teacher):
        if lp_s is None or lp_t is None:
            continue
        device = lp_s.device
        total = total.to(device)

        # Reverse KL: -log π_student weighted uniformly (no gate)
        loss_i = -lp_s.mean()
        total = total + loss_i
        n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, requires_grad=True)
    return total / n_terms


# ══════════════════════════════════════════════════════════════════════════════
# Full training-step functions
# ══════════════════════════════════════════════════════════════════════════════

def _get_teacher_logp(
    model,
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    device: str,
) -> List[torch.Tensor]:
    """Compute teacher log probs (no grad — teacher forward is detached)."""
    return compute_token_logp(
        model, teacher_prompt_ids, gen_ids, device, use_grad=False
    )


def sdar_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    eps: float = EPS,
    beta: float = BETA,
    lam: float = LAMBDA,
    gate_strategy: str = "gap",
    device: str = "cuda",
    grad_clip: float = 1.0,
) -> Dict:
    """SDAR training step: GRPO + λ·gated OPSD (Section 2.1).

    Invariants:
      - g_t = sigmoid(β·Δ_t).detach()                    [stop-gradient]
      - Δ_t = logp_teacher(y_t) − logp_student(y_t)
      - loss = L_GRPO + λ·L_OPSD
      - Teacher = SAME MODEL WEIGHTS with skill context appended to prompt
    """
    advantages = compute_group_advantages(rewards, group_size)

    # Teacher forward (no grad)
    logp_teacher = _get_teacher_logp(model, teacher_prompt_ids, gen_ids, device)

    # Student/old forward (no grad — for GRPO ratio denominator)
    logp_old = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=False)

    # Student forward (with grad — for both GRPO and OPSD)
    logp_new = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=True)

    # Losses
    l_grpo = grpo_loss_fn(logp_new, logp_old, advantages, eps=eps)
    l_opsd, gate_ratio = opsd_loss_fn(
        logp_new, logp_teacher, beta=beta, gate_strategy=gate_strategy
    )
    loss = l_grpo + lam * l_opsd

    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "grpo_loss": l_grpo.item(),
        "opsd_loss": l_opsd.item(),
        "gate_active_ratio": gate_ratio,
        "mean_reward": sum(rewards) / len(rewards),
        "teacher_student_gap": _mean_ts_gap(logp_old, logp_teacher),
    }


def grpo_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    eps: float = EPS,
    device: str = "cuda",
    grad_clip: float = 1.0,
    **kwargs,  # absorb extra keyword args for uniform interface
) -> Dict:
    """GRPO training step (Algorithm 2).

    Clipped PPO surrogate with group-relative advantages.
    ratio = exp(logp_new - logp_old)
    loss  = -E[min(ratio*A, clamp(ratio, 1-ε, 1+ε)*A)]
    """
    advantages = compute_group_advantages(rewards, group_size)

    logp_old = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=False)
    logp_new = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=True)

    loss = grpo_loss_fn(logp_new, logp_old, advantages, eps=eps)

    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "grpo_loss": loss.item(),
        "opsd_loss": 0.0,
        "gate_active_ratio": 0.0,
        "mean_reward": sum(rewards) / len(rewards),
        "teacher_student_gap": 0.0,
    }


def opsd_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    beta: float = BETA,
    gate_strategy: str = "gap",
    device: str = "cuda",
    grad_clip: float = 1.0,
    **kwargs,
) -> Dict:
    """OPSD training step (Algorithm 3): standalone self-distillation.

    No GRPO component. Only the gated KL loss.
    Teacher = same model + skills (as in SDAR).
    """
    logp_teacher = _get_teacher_logp(model, teacher_prompt_ids, gen_ids, device)
    logp_student = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=True)

    loss, gate_ratio = opsd_loss_fn(logp_student, logp_teacher, beta=beta, gate_strategy=gate_strategy)

    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "grpo_loss": 0.0,
        "opsd_loss": loss.item(),
        "gate_active_ratio": gate_ratio,
        "mean_reward": sum(rewards) / len(rewards),
        "teacher_student_gap": 0.0,
    }


def skill_sd_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    device: str = "cuda",
    grad_clip: float = 1.0,
    **kwargs,
) -> Dict:
    """Skill-SD training step (Algorithm 4): skill-conditioned self-distillation.

    Similar to OPSD but skill context changes per step (no GRPO component).
    The difference from OPSD is that skills are changed/updated at each step.
    """
    # Implementation is identical to OPSD; distinction is in how skills are
    # sampled/updated between steps (handled in train.py)
    return opsd_step(
        model, optimizer,
        student_prompt_ids, teacher_prompt_ids, gen_ids, rewards,
        group_size=group_size, device=device, grad_clip=grad_clip,
    )


def grpo_opsd_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    eps: float = EPS,
    lam: float = LAMBDA,
    device: str = "cuda",
    grad_clip: float = 1.0,
    **kwargs,
) -> Dict:
    """GRPO+OPSD naive sum (Algorithm 5): GRPO + λ·OPSD WITHOUT sigmoid gate.

    This is the ablation that the paper shows leads to catastrophic instability.
    The gate is removed (replaced by uniform weighting), causing high variance.
    """
    advantages = compute_group_advantages(rewards, group_size)

    logp_teacher = _get_teacher_logp(model, teacher_prompt_ids, gen_ids, device)
    logp_old = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=False)
    logp_new = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=True)

    l_grpo = grpo_loss_fn(logp_new, logp_old, advantages, eps=eps)
    # Ungated OPSD (the instability-inducing variant)
    l_opsd_ungated = opsd_loss_ungated(logp_new, logp_teacher)
    loss = l_grpo + lam * l_opsd_ungated

    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "grpo_loss": l_grpo.item(),
        "opsd_loss": l_opsd_ungated.item(),
        "gate_active_ratio": 0.0,  # no gate in this baseline
        "mean_reward": sum(rewards) / len(rewards),
        "teacher_student_gap": _mean_ts_gap(logp_old, logp_teacher),
    }


def rlsd_step(
    model,
    optimizer,
    student_prompt_ids: List[List[int]],
    teacher_prompt_ids: List[List[int]],
    gen_ids: List[List[int]],
    rewards: List[float],
    group_size: int = 8,
    eps: float = EPS,
    beta: float = BETA,
    lam: float = LAMBDA,
    kl_coef: float = 0.0,
    device: str = "cuda",
    grad_clip: float = 1.0,
    **kwargs,
) -> Dict:
    """RLSD training step (Algorithm 6): RL with self-distillation.

    Combines policy gradient with self-distillation:
    loss = RL_loss + λ·KL(student || teacher)

    Per Appendix A: uses REINFORCE-style gradient (not GRPO clipping).
    """
    advantages = compute_group_advantages(rewards, group_size)

    logp_teacher = _get_teacher_logp(model, teacher_prompt_ids, gen_ids, device)
    logp_student = compute_token_logp(model, student_prompt_ids, gen_ids, device, use_grad=True)

    # REINFORCE loss (no clipping — distinguishes from GRPO)
    rl_loss = torch.tensor(0.0)
    n = 0
    for lp_s, a_i in zip(logp_student, advantages):
        if lp_s is None:
            continue
        device_s = lp_s.device
        rl_loss = rl_loss.to(device_s)
        rl_loss = rl_loss + (-lp_s.mean() * a_i)
        n += 1
    if n > 0:
        rl_loss = rl_loss / n

    # KL distillation toward teacher (no gate)
    kl_loss = torch.tensor(0.0)
    n_kl = 0
    for lp_s, lp_t in zip(logp_student, logp_teacher):
        if lp_s is None or lp_t is None:
            continue
        device_s = lp_s.device
        kl_loss = kl_loss.to(device_s)
        # Forward KL approximation: -lp_t log π_student ≈ -(lp_t * lp_s).mean()
        # Actually using reverse KL: KL(π_s || π_t) ≈ lp_s - lp_t
        kl = (lp_s - lp_t.detach()).mean()
        kl_loss = kl_loss + kl
        n_kl += 1
    if n_kl > 0:
        kl_loss = kl_loss / n_kl

    loss = rl_loss + lam * kl_loss

    optimizer.zero_grad()
    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "grpo_loss": rl_loss.item(),
        "opsd_loss": kl_loss.item(),
        "gate_active_ratio": 0.0,
        "mean_reward": sum(rewards) / len(rewards),
        "teacher_student_gap": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch table
# ══════════════════════════════════════════════════════════════════════════════

ALGORITHM_FNS = {
    "sdar": sdar_step,
    "grpo": grpo_step,
    "opsd": opsd_step,
    "skill_sd": skill_sd_step,
    "grpo_opsd": grpo_opsd_step,
    "rlsd": rlsd_step,
}


def get_step_fn(algorithm: str):
    """Return the training-step function for the named algorithm."""
    algo = algorithm.lower().replace("-", "_")
    if algo not in ALGORITHM_FNS:
        raise ValueError(f"Unknown algorithm: {algo!r}. "
                         f"Available: {list(ALGORITHM_FNS.keys())}")
    return ALGORITHM_FNS[algo]


def algorithm_uses_teacher(algorithm: str) -> bool:
    """Return True if this algorithm requires teacher log-probs."""
    return algorithm.lower() in ("sdar", "opsd", "skill_sd", "grpo_opsd", "rlsd")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mean_ts_gap(
    logp_old: List[Optional[torch.Tensor]],
    logp_teacher: List[Optional[torch.Tensor]],
) -> float:
    """Mean teacher-student log-prob gap across rollouts."""
    diffs = []
    for lp_s, lp_t in zip(logp_old, logp_teacher):
        if lp_s is not None and lp_t is not None:
            diff = (lp_t.detach() - lp_s.detach()).mean().item()
            diffs.append(diff)
    return sum(diffs) / len(diffs) if diffs else 0.0
