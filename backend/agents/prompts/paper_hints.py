"""Paper-specific extras applied via the ``--paper-hint <id>`` CLI flag.

Each entry in ``PAPER_HINTS`` is a :class:`PaperHint` carrying three independent
layers that compose with operator-set configuration:

  - ``guidance``: appended to ``REPROLAB_BASELINE_EXTRA_GUIDANCE`` so the agent
    sees the paper-specific algorithmic invariants in its baseline-implementation
    prompt (existing hook in ``backend/agents/baseline_implementation.py``).
  - ``default_scope``: a :class:`ScopeSpec` providing the rubric-default models /
    datasets / seeds. The operator's ``--scope-spec`` narrows or expands it via
    :meth:`ScopeSpec.merge_with_paper_default`.
  - ``invariants``: deterministic regex-based code checks the rubric scorer
    applies alongside the LLM leaf grader (hard gate on ``must_not_match``,
    soft signal on ``must_match``). Also callable by the agent as an advisory
    self-check via the ``assert_invariant`` primitive.

Add new entries here as paper-specific patterns emerge. Keep ``guidance`` to a
short paragraph (≤ 500 chars) so the prompt stays compact.
"""

from __future__ import annotations

import re

from backend.agents.schemas import (
    DatasetSlice,
    InvariantSpec,
    PaperHint,
    ScopeSpec,
)

# arXiv ids occasionally arrive with a trailing version suffix (e.g.
# "2605.15155v2"). Normalise to the bare id so lookup is version-agnostic.
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


PAPER_HINTS: dict[str, PaperHint] = {
    "2605.15155": PaperHint(
        guidance=(
            "SDAR (Self-Distilled Agentic Reinforcement Learning) algorithmic "
            "invariants the rubric inspects for: the on-policy self-distillation "
            "gate is g_t = sigmoid(beta * delta_t) with stop_gradient on the "
            "gate (NOT differentiable through it); the self-distillation weight "
            "is lambda = 0.1 and the gate sharpness is beta = 10; the GRPO loss "
            "is ADDED to the gated self-distillation loss (both terms required). "
            "Use the real pretrained Qwen weights from HuggingFace, not "
            "surrogates."
        ),
        default_scope=ScopeSpec(
            models=[
                "Qwen3-1.7B-Instruct",
                "Qwen2.5-3B-Instruct",
                "Qwen2.5-7B-Instruct",
            ],
            datasets=[
                DatasetSlice(name="ALFWorld"),
                DatasetSlice(name="WebShop"),
                DatasetSlice(name="Search-QA"),
            ],
            seeds=[42, 43, 44],
        ),
        invariants=[
            InvariantSpec(
                name="sigmoid_gate_on_advantage",
                rationale=(
                    "SDAR's OPSD gate must be g_t = sigmoid(beta * delta_t). "
                    "Without the sigmoid * beta form, the gate weight does not "
                    "depend on the advantage difference and the on-policy "
                    "self-distillation mechanism collapses."
                ),
                must_match=[
                    r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*",
                ],
            ),
            InvariantSpec(
                name="stop_gradient_on_gate",
                rationale=(
                    "Gate must not propagate gradients (paper §3.2). Use "
                    "``.detach()`` on the gate value or compute it under "
                    "``torch.no_grad()``."
                ),
                must_match=[
                    r"\.detach\(\)",
                    r"with\s+torch\.no_grad\s*\(\s*\)",
                ],
            ),
            InvariantSpec(
                name="lambda_self_distill_weight_0p1",
                rationale=(
                    "Paper sets the OPSD self-distillation weight lambda to "
                    "0.1 (paper §3.2, eq. 5)."
                ),
                must_match=[
                    r"(?:lambda|opsd_weight|distill_weight|self_distill_weight|sd_weight)\s*[:=]\s*0\.1\b",
                ],
            ),
            InvariantSpec(
                name="beta_gate_sharpness_10",
                rationale=(
                    "Paper sets the gate sharpness beta to 10 (paper §3.2)."
                ),
                must_match=[
                    r"\bbeta\s*[:=]\s*10(?:\.0)?\b",
                ],
            ),
            InvariantSpec(
                name="grpo_loss_added_to_distill_loss",
                rationale=(
                    "SDAR loss = GRPO loss + gated self-distillation loss. "
                    "Both terms must appear in the loss computation; a "
                    "GRPO-only or distill-only loss fails algorithmic fidelity."
                ),
                must_match=[
                    r"\b(?:GRPO|grpo)(?:_loss)?\b",
                    r"\b(?:opsd|self[_-]?distill|sd)(?:_loss)?\b",
                ],
            ),
            InvariantSpec(
                name="real_qwen_weights_not_surrogate",
                rationale=(
                    "Rubric verifies real HuggingFace Qwen weights; a TinyLM "
                    "or surrogate model fails the architecture-fidelity leaves "
                    "and produces meaningless reward signal."
                ),
                must_match=[
                    r"from_pretrained\s*\(\s*['\"]Qwen/Qwen",
                ],
                must_not_match=[
                    r"class\s+TinyLM\b",
                    r"#\s*surrogate\s+model",
                    r"#\s*stub\s+model",
                ],
            ),
        ],
        blocked_resources=[
            # The SDAR paper's own implementation — the PaperBench blacklist entry
            # (mirrors third_party/paperbench/ftrl/blacklist.txt). The arXiv run
            # loads no bundle, so this paper-hint entry is what guards it. trl and
            # other framework deps are deliberately NOT listed.
            "https://github.com/BartekCupial/finetuning-RL-as-CL",
        ],
    ),
    # Adam (Kingma & Ba, 2014) — five experiment families. Four are cheap
    # (MNIST-MLP, MNIST logistic regression, IMDB BoW logistic, CIFAR-10 CNN);
    # the long pole is the VAE bias-correction sweep (~21 configs = beta2 x lr x
    # optimizer, each x 20 epochs). The 2026-06-08 run packed all five into one
    # monolithic train.py with the sweep last and wrote metrics.json only at the
    # end — a wall-clock timeout fired mid-sweep and zeroed the four finished
    # families. This hint caps + cell-structures the sweep so each config is
    # independently bounded and partial results survive a timeout.
    "1412.6980": PaperHint(
        guidance=(
            "Adam (1412.6980) timeout-survival structure: the VAE bias-correction "
            "sweep is the long pole (~21 configs = beta2 x lr x optimizer, each x 20 "
            "epochs). CAP it to a smallest-config-first subset (NOT the full grid) and "
            "structure it as `cells.json` cells (one cell per config in train_cell.py), "
            "never a monolithic in-process loop, so the harness bounds each config and "
            "its metrics land incrementally. The four quick families (MNIST-MLP, MNIST "
            "logistic-regression, IMDB, CIFAR10) are cheap: write metrics.json "
            "ATOMICALLY as each completes (never only at the end) so a timeout truncates "
            "the tail, not finished work."
        ),
        default_scope=ScopeSpec(
            datasets=[
                DatasetSlice(name="MNIST"),
                DatasetSlice(name="IMDB"),
                DatasetSlice(name="CIFAR-10"),
            ],
            seeds=[1],
        ),
    ),
}


def _normalize_paper_id(paper_id: str) -> str:
    """Strip a trailing arXiv version suffix so lookup is version-agnostic.

    ``"2605.15155v2"`` → ``"2605.15155"``; bare ids pass through unchanged.
    """
    return _VERSION_SUFFIX.sub("", paper_id.strip())


def lookup_paper_hint(paper_id: str | None) -> PaperHint | None:
    """Return the :class:`PaperHint` for ``paper_id``, or ``None`` if absent.

    ``paper_id`` is normalised (whitespace stripped, arXiv version suffix
    removed) before lookup. Empty / None input returns ``None``. Callers
    should treat ``None`` as "no built-in hint for this paper" — never an
    error.
    """
    if not paper_id:
        return None
    return PAPER_HINTS.get(_normalize_paper_id(paper_id))
