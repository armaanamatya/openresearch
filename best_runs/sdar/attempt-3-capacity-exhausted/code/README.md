# SDAR Search-QA Reproduction — prj_09047604e591d969

## What was reproduced

**Self-Distilled Agentic Reinforcement Learning (SDAR)**, arXiv 2605.15155,
Search-QA environment with two model variants:

| Model | HF ID |
|-------|-------|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` |

Each model trained **150 steps** × **two algorithms**:
1. **SDAR** — GRPO + gated OPSD self-distillation
2. **GRPO** — ablation (OPSD term disabled, `opsd_enabled=False`)

**SDAR Algorithm Invariants** (module-level in both `train.py` and `train_cell.py`):
```python
BETA = 10.0    # gate sharpness β (Section 3.1)
LAMBDA = 0.1   # OPSD weight λ (Section 3.1)

# Token-level gap (Δ_t = log π_teacher(y_t) − log π_student(y_t))
delta_t = teacher_token_logp - student_token_logp

# Gated OPSD with stop-gradient on gate
gate = torch.sigmoid(BETA * delta_t).detach()   # stop-gradient (Section 3.1)

# Single-sample KL contribution (OPSD loss)
opsd_loss = (gate * delta_t * resp_mask).sum() / resp_mask.sum()

# Combined objective L = L_GRPO + λ · L_OPSD (Equation from Section 3.1)
loss = grpo_loss + (LAMBDA * opsd_loss if opsd_enabled else 0.0)
```

**Reward**: SQuAD-style token-F1, max over all gold aliases (critical for NQ LIST answers).

**Training data**: NQ-open + HotpotQA (distractor) validation splits.

**Inference time**: `{skill_context}` is **empty** — SDAR requires no external skills
during inference (Section 3.2, Figures 15–17).

## What was omitted and why

| Item | Reason | Declared in scope.gaps |
|------|--------|------------------------|
| Qwen2.5-7B | Budget / VRAM constraint | ✓ |
| ALFWorld | Search-QA only scope | ✓ |
| WebShop | Search-QA only scope | ✓ |
| E5 retriever | Closed-book QA used | ✓ |
| TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle | OOD eval; not in in-domain training | ✓ |
| Skill-SD, RLSD baselines | Out of scope | ✓ |
| SkillRL SkillBank | Out of scope | ✓ |
| Entropy/Soft-OR gating | Alternative strategies; gap gating (default) | ✓ |

All gaps are declared in `metrics.json::scope.gaps` for dynamic rubric adjustment
(excluded from both numerator and denominator of the rubric score).

## How to read metrics.json

```
reward                           Primary metric: mean eval token-F1 across all completed runs
comparison.<model>.sdar_f1       SDAR final eval token-F1
comparison.<model>.grpo_f1       GRPO ablation final eval token-F1
comparison.<model>.delta         sdar_f1 - grpo_f1 (positive = SDAR beats GRPO)
per_model.<model>.search_qa.sdar Per-cell SDAR metrics (reward_mean, gate_active_ratio_mean, etc.)
per_model.<model>.search_qa.grpo Per-cell GRPO ablation metrics
training_curves.<model>.<base>   Per-step {step, loss, reward, gate_active_ratio, gate_magnitude}
scope.gaps                       Out-of-scope items (rubric dynamically excludes their leaves)
status                           "completed" when all cells finished
```

**Gate diagnostics** (paper Figures 10-14): `gate_active_ratio_mean` = fraction of tokens
where g_t > 0.5 (teacher much more confident than student). A non-trivial value (>0.1)
confirms the OPSD signal is active. Gate should stabilize around 0.3-0.6.

## Runtime requirements

- **Python**: `/home/sww35/openresearch/.venv/bin/python` (Python 3.12, transformers 4.57.6)
  — system `python3.8` + transformers 4.46.3 does NOT support the Qwen3 model type.
- **GPU**: 1× GPU ≥ 8 GB (Qwen3-1.7B GRPO), up to 18 GB (Qwen2.5-3B SDAR with frozen teacher).
- **Data**: offline HF cache at `/home/sww35/openresearch/runs/.cache/hf`.
