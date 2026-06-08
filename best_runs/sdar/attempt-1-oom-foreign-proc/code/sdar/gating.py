"""Gating strategies for SDAR (Section 2.3).

Three strategies:
  - Gap  (default): s_t = Δ_t = log π_teacher(y_t) − log π_student(y_t)
  - Entropy:        s_t = H_t = −Σ_v π_student(v|·) log π_student(v|·)
  - Soft-OR:        g_t = σ(β·H_t) + σ(β·Δ_t) − σ(β·H_t)·σ(β·Δ_t)

The gate is ALWAYS detached (.detach()) so gradients flow only through
the student log-probabilities in the OPSD loss — the paper's stop-gradient
invariant.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

# Paper default
BETA = 10.0


def gap_gate(
    logp_student: torch.Tensor,
    logp_teacher: torch.Tensor,
    beta: float = BETA,
) -> torch.Tensor:
    """Gap (default) gating: g_t = σ(β · (logp_teacher - logp_student)).detach()

    Args:
        logp_student: per-token log probs from student, shape [gl]
        logp_teacher: per-token log probs from teacher (same tokens), shape [gl]
        beta: sharpness parameter (paper default = 10.0)

    Returns:
        gate: detached tensor of shape [gl], values in (0, 1)
    """
    delta = logp_teacher.detach() - logp_student.detach()  # stop-gradient on delta inputs
    gate = torch.sigmoid(beta * delta)
    return gate.detach()  # CRITICAL stop-gradient: gradients must NOT flow through gate


def entropy_gate(
    logits: torch.Tensor,
    beta: float = BETA,
) -> torch.Tensor:
    """Entropy gating: g_t = σ(β · H_t).detach()

    H_t = −Σ_v π_student(v) log π_student(v): per-token entropy of the
    student distribution at each generation step.

    Args:
        logits: raw logits from the student at each position, shape [gl, vocab]
        beta: sharpness parameter

    Returns:
        gate: detached tensor of shape [gl]
    """
    probs = F.softmax(logits.float(), dim=-1)
    # Entropy: -sum(p log p)
    entropy = -(probs * torch.clamp(torch.log(probs + 1e-12), min=-100)).sum(-1)  # [gl]
    gate = torch.sigmoid(beta * entropy)
    return gate.detach()


def soft_or_gate(
    logp_student: torch.Tensor,
    logp_teacher: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    beta: float = BETA,
) -> torch.Tensor:
    """Soft-OR gating combining Gap and Entropy.

    g_t = σ(β·H_t) + σ(β·Δ_t) − σ(β·H_t)·σ(β·Δ_t)

    This is the probabilistic OR of two independent binary gates.

    Args:
        logp_student: per-token student log probs, shape [gl]
        logp_teacher: per-token teacher log probs (same tokens), shape [gl]
        logits: raw student logits, shape [gl, vocab] (needed for entropy)
        beta: sharpness

    Returns:
        gate: detached tensor of shape [gl]
    """
    if logits is None:
        raise ValueError("soft_or_gate requires student logits for entropy computation")

    g_entropy = entropy_gate(logits, beta=beta)
    g_gap = gap_gate(logp_student, logp_teacher, beta=beta)

    # Probabilistic OR: P(A or B) = P(A) + P(B) - P(A)*P(B)
    gate = g_entropy + g_gap - g_entropy * g_gap
    return gate.detach()


def make_gate(
    strategy: str,
    logp_student: torch.Tensor,
    logp_teacher: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    beta: float = BETA,
) -> torch.Tensor:
    """Dispatch to the appropriate gating function.

    Args:
        strategy: one of "gap", "entropy", "soft_or"
        logp_student: shape [gl]
        logp_teacher: shape [gl]
        logits: shape [gl, vocab] — required for "entropy" and "soft_or"
        beta: sharpness

    Returns:
        gate: detached [gl] tensor
    """
    strategy = strategy.lower()
    if strategy == "gap":
        return gap_gate(logp_student, logp_teacher, beta=beta)
    elif strategy == "entropy":
        if logits is None:
            raise ValueError("entropy gate requires logits")
        return entropy_gate(logits, beta=beta)
    elif strategy in ("soft_or", "soft-or"):
        return soft_or_gate(logp_student, logp_teacher, logits=logits, beta=beta)
    else:
        raise ValueError(f"Unknown gating strategy: {strategy!r}. "
                         f"Choose from: gap, entropy, soft_or")
