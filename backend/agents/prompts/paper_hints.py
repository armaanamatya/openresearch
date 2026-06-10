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
    "1412.6806": PaperHint(
        guidance=(
            "All-CNN (1412.6806) — the rubric's dominant lever is the CIFAR-10 "
            "model-comparison table: models A/B/C, each as base / strided-CNN / "
            "ConvPool-CNN / All-CNN variants (12 combos; headline: All-CNN-C "
            "~9.1% test error without augmentation, and the strided/all-conv "
            "variants matching or beating their pooling counterparts).\n"
            "LEARNING-RATE PROTOCOL (the #1 prior-run killer — single global "
            "lr=0.05 dead-trained ConvPool/base variants on three attempts): "
            "the paper selects gamma PER MODEL from {0.25, 0.1, 0.05, 0.01}. "
            "Run a SHORT lr probe per (model, variant) — a few epochs, pick "
            "the best by val accuracy — THEN launch the full 350-epoch run at "
            "each model's own lr. Never share one lr across architectures. "
            "Recipe: SGD momentum 0.9, weight decay 0.001, lr x0.1 at epochs "
            "[200, 250, 300], 350 epochs total, dropout 20% on input + 50% "
            "after pooling (or its replacement), global-contrast-normalization "
            "+ ZCA whitening.\n"
            "STRUCTURE: emit cells.json with ONE cell per (model, variant, "
            "dataset) and EXPLICIT model_key/env/baseline axes per cell (e.g. "
            "model_key='c_allcnn', env='cifar10_noaug', baseline='allcnn') "
            "plus that cell's chosen lr; train_cell.py writes a FLAT per-cell "
            "metrics.json with test_error_pct. Aggregate per_model entries "
            "ATOMICALLY as cells finish so a timeout truncates the tail, not "
            "finished work. CIFAR-100 (All-CNN, with aug) comes only AFTER all "
            "CIFAR-10 cells land. The ImageNet experiment is OUT OF SCOPE on "
            "this budget: write it MECHANICALLY into metrics.json as "
            "scope.gaps=[{'item': 'ImageNet', 'reason': 'out of compute scope "
            "(operator-bounded)'}] — a declared gap is excluded from scoring; "
            "an undeclared one scores 0. Never fake or silently omit it.\n"
            "CHEAP RUBRIC EVIDENCE the prior attempt left on the table: "
            "(a) MEASURE per-model parameter counts (count_parameters at model "
            "build) and record them under per_model[*].param_count — the paper's "
            "tables compare them; (b) after the best All-CNN model trains, "
            "produce the paper's Section-4 visualization: guided-backprop / "
            "deconv ReLU-masking saliency for a few CIFAR images (~50 lines of "
            "hooks), saved as fig_relu_masking.png + a JSON sidecar; (c) if a "
            "PRIOR-ATTEMPT MEASURED EVIDENCE block is present in this prompt, "
            "treat it as ground truth: restore configs that hit paper-grade "
            "errors verbatim, and only probe lr for cells with no working "
            "config in ANY attempt."
        ),
        default_scope=ScopeSpec(
            datasets=[
                DatasetSlice(name="CIFAR-10"),
                DatasetSlice(name="CIFAR-100"),
            ],
            seeds=[1],
        ),
    ),
    "1412.6980": PaperHint(
        guidance=(
            "Adam (1412.6980) — PRIORITY #1: reproduce ALL SIX experiment families and "
            "aggregate EACH into metrics.json `per_model[<experiment>]` with MEASURED SCALAR "
            "values. A metrics.json that contains only one experiment scores ~0 on BOTH "
            "Result-match and Eval-protocol — breadth across experiments is the dominant "
            "lever. The six families: (1) MNIST logistic-regression, (2) IMDB BoW, (3) MNIST "
            "MLP, (4) CIFAR-10 CNN, (5) VAE bias-correction, (6) VAE LR sweep (Fig 4). Do the "
            "FOUR CHEAP families (1-4) FIRST and write their per_model[...] entries (with real "
            "accuracy/NLL scalars) BEFORE touching any VAE work; write metrics.json ATOMICALLY "
            "as each family completes so a timeout truncates the tail, not finished work.\n"
            "The VAE LR sweep (6) is the LONG POLE — do it LAST and CAP it to a "
            "smallest-config-first subset (NOT the full ~21-config grid); structure the VAE "
            "sweep as `cells.json` cells (one per config), never a monolithic loop. The "
            "bias-corrected VAE config can DIVERGE at high LR (NaN reconstruction -> CUDA "
            "`input_val >= 0 && <= 1` assert) — clamp/sigmoid the decoder output to [0,1] (or "
            "use BCE-with-logits) and guard NaN so a diverging config records a (bad) ELBO "
            "scalar instead of CRASHING the cell.\n"
            "ADDITIVE convergence evidence — ONLY after the per_model scalars for all six "
            "families are in (never instead of them): the paper's headline claims are about "
            "CONVERGENCE SPEED, so also emit a per-epoch `history` block "
            "(history.<experiment>.<optimizer> = {epoch:[...], <metric>:[...]}) on a COMMON "
            "x-axis with identical init across optimizers (fair_comparison.snapshot_init_state "
            "once -> restore_init_state before each optimizer -> record init_fingerprint per "
            "optimizer in provenance.json); write the VAE sweep results under metrics.json "
            "`vae_lr_sweep`; emit cumulative `regret` as a time-series ARRAY (not a scalar); "
            "render fig_4 as loss-vs-log10(alpha) with a LOG x-axis (axis sidecar JSON). "
            "CIFAR-10 uses global-contrast-normalization + ZCA whitening; the MLP/MNIST panel "
            "includes AdaDelta + SFO baselines. Call "
            "rubric_guard.assert_metrics_schema(structured_evidence=...) at the end of "
            "train.py so a missing curve/sweep/series is repaired — but get the six per_model "
            "scalars landing FIRST."
        ),
        default_scope=ScopeSpec(
            datasets=[
                DatasetSlice(name="MNIST"),
                DatasetSlice(name="IMDB"),
                DatasetSlice(name="CIFAR-10"),
            ],
            seeds=[1],
        ),
        structured_evidence={
            "history_methods": [
                "adam", "sgd_nesterov", "adagrad", "adamax", "rmsprop", "adadelta",
            ],
            "sweeps": ["vae_lr_sweep"],
            "series": ["regret"],
        },
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
