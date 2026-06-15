"""Paper-specific extras applied via the ``--paper-hint <id>`` CLI flag.

Each entry in ``PAPER_HINTS`` is a :class:`PaperHint` carrying three independent
layers that compose with operator-set configuration:

  - ``guidance``: appended to ``OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`` so the agent
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
                # Qwen3 post-trained models drop the "-Instruct" suffix (HF repo
                # Qwen/Qwen3-1.7B; "Qwen/Qwen3-1.7B-Instruct" 401s — does not exist).
                # Only Qwen2.5 uses the "-Instruct" suffix. A wrong id here poisons
                # the guard's enforced canonical path AND the implementer guidance.
                "Qwen3-1.7B",
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
            "FIDELITY CHECK (the prior-run ordering INVERTED): if your measured "
            "All-CNN-C is WORSE than ConvPool-CNN-C or the base CNN (a prior run "
            "got All-CNN-C 14.8% vs ConvPool 11.3% vs base 12.2% — the paper's "
            "ordering REVERSED), that is an IMPLEMENTATION bug — augmentation "
            "order (see below), the strided-conv replacement geometry (the "
            "stride-2 3x3 conv must preserve the pooling layer's receptive "
            "field), or dropout placement — NOT an honest negative. Debug the "
            "recipe until the all-conv variants match-or-beat their pooling "
            "counterparts as the paper claims, before accepting the result.\n"
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
            "AUGMENTATION ORDER (critical — a prior run's AUGMENTED runs scored "
            "WORSE than no-aug, the exact signature of this bug): apply "
            "augmentation (4px reflect-pad then random 32x32 crop; horizontal "
            "flip) to the RAW image FIRST, THEN GCN + ZCA-whiten the augmented "
            "image. NEVER augment an already-whitened tensor — ZCA encodes the "
            "train-set pixel covariance, so cropping/flipping a whitened tensor "
            "corrupts those statistics and makes augmentation HURT instead of "
            "help. The paper's 9.08% All-CNN-C with augmentation requires "
            "correct-order augmentation.\n"
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
            "config in ANY attempt; "
            "(d) emit provenance.json recording, per cell, the SEARCHED lr set "
            "+ each probed rate's result + the SELECTED lr, plus the final "
            "optimizer hyperparameters (momentum, weight_decay, the "
            "[200,250,300] decay schedule) — the eval-protocol and searched-set "
            "rubric leaves credit the recorded search, not just the winning lr."
        ),
        default_scope=ScopeSpec(
            datasets=[
                DatasetSlice(name="CIFAR-10"),
                DatasetSlice(name="CIFAR-100"),
            ],
            seeds=[1],
        ),
    ),
    "1512.03385": PaperHint(
        guidance=(
            "ResNet / Deep Residual Learning (1512.03385) — the rubric's dominant "
            "lever is the CIFAR-10 DEGRADATION CONTRAST (Section 4.2, Table 6): "
            "train BOTH plain and residual nets at MULTIPLE depths and show "
            "residual learning SOLVES degradation — plain nets get WORSE as they "
            "deepen (plain-56 test error > plain-44 > plain-32 > plain-20), while "
            "ResNets get BETTER (resnet-110 6.43% < resnet-56 6.97% < resnet-44 "
            "7.17% < resnet-32 7.51% < resnet-20 8.75%). The CONTRAST is the "
            "claim — a grid of only ResNets, or only one depth, cannot show it.\n"
            "ARCHITECTURE (CIFAR ResNet, 6n+2 weighted layers): first 3x3 conv -> "
            "16 filters; then 3 stages of 2n stacked 3x3-conv blocks on feature "
            "maps 32/16/8 with filters 16/32/64 (stride-2 at stage boundaries); "
            "global-average-pool -> 10-way FC -> softmax. n in {3,5,7,9,18} -> "
            "depth {20,32,44,56,110}. Shortcuts are IDENTITY (option A, "
            "parameter-free; zero-pad the extra channels at dimension increases) "
            "— NOT 1x1-conv projection. He/MSRA init. NO dropout, NO ZCA.\n"
            "RECIPE: SGD momentum 0.9, weight_decay 1e-4, batch 128, lr 0.1 "
            "divided by 10 at 32k and 48k iterations, terminate at 64k iterations "
            "(~164 epochs). Preprocess: per-pixel MEAN subtraction ONLY (NOT "
            "GCN/ZCA — different from All-CNN). Augment: 4-pixel zero-pad then "
            "random 32x32 crop + horizontal flip on TRAIN; test on the single "
            "original 32x32 view. ResNet-110 ONLY: warm up at lr 0.01 until train "
            "error < 80% (~400 iters), THEN set lr 0.1 and resume the schedule — "
            "at depth 110 a plain 0.1 start diverges.\n"
            "STRUCTURE: emit cells.json with ONE cell per (arch, depth) and "
            "EXPLICIT model_key/env/baseline axes (e.g. model_key='resnet_56', "
            "env='cifar10', baseline='resnet'; model_key='plain_56', env='cifar10', "
            "baseline='plain'); train_cell.py writes a FLAT per-cell metrics.json "
            "with test_error_pct. Aggregate per_model ATOMICALLY as cells finish so "
            "a timeout truncates the tail, not finished work. Minimum faithful "
            "grid: plain {20,32,44,56} + resnet {20,32,44,56,110} (9 cells) — the "
            "plain side is REQUIRED to show degradation; resnet-110 is the headline.\n"
            "OUT OF SCOPE on this budget: the paper's ImageNet ResNets "
            "(18/34/50/101/152) take days — implement the CIFAR nets only; if the "
            "rubric has ImageNet leaves leave them honestly ungraded (do NOT fake "
            "or train ImageNet). CHEAP RUBRIC EVIDENCE: (a) record per-depth "
            "param_count under per_model[*].param_count (resnet-110 ~1.7M params); "
            "(b) emit provenance.json with the lr schedule, batch, weight_decay, "
            "and per-cell final test_error_pct + epochs_run; (c) if a PRIOR-ATTEMPT "
            "MEASURED EVIDENCE block is present, restore paper-grade configs verbatim."
        ),
        default_scope=ScopeSpec(
            datasets=[DatasetSlice(name="CIFAR-10")],
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
            "FAILURE ISOLATION (non-negotiable): a CUDA device-side assert poisons the whole "
            "process, so families must NEVER share one. Keep the cells.json + train_cell.py "
            "route on EVERY repair pass (restore rlm_state/last_cells.json if a rewrite lost "
            "it); when any families do run sequentially in one train.py, wrap EACH family in "
            "try/except, validate index ranges up front (max(label) < num_classes, token ids "
            "< vocab_size), and write metrics.json incrementally with per-family "
            "status='complete' as each finishes — one family's crash must cost ONE family, "
            "never the measured work of the others (2026-06-11: a monolithic repair lost a "
            "92.6%-measured logreg family to an assert in the next family).\n"
            "PER-OPTIMIZER LEARNING RATE (the Result-match lever — ~75% of the score gap, do "
            "NOT skip): the paper compares each optimizer AT ITS OWN BEST learning rate (it "
            "grid-searches alpha per method and reports each at its best); a single SHARED lr "
            "makes Adam plateau and INVERTS the headline claim. On 2026-06-14 mnist_mlp MEASURED "
            "adam final_train_loss 0.0087 while sgd_momentum reached 1.6e-6 — a pure lr artifact "
            "that scored the 'Adam converges fastest' leaf 0.0.\n"
            "PROTOCOL (tune-then-run, BOUNDED — do NOT cross-product the full grid): "
            "(a) PREFER the paper's REPORTED per-optimizer learning rate wherever the paper "
            "states it (most faithful, ZERO sweep cost); the canonical defaults are Adam "
            "alpha=1e-3 beta1=0.9 beta2=0.999 eps=1e-8; AdaGrad 1e-2; RMSProp 1e-3; "
            "SGD-Nesterov/momentum 1e-2 (momentum 0.9); AdaDelta rho=0.95; Adamax 2e-3. "
            "(b) ONLY where the paper does not give a value, run a short tuning pass over a "
            "3-point lr grid around the default ({0.3x, 1x, 3x}) and SELECT by the metric the "
            "CLAIM is graded on at the paper's FULL epoch count — NOT a 2-3 epoch proxy, which "
            "can reward the early-fast/late-slow behaviour a claim must REVERSE (e.g. CIFAR "
            "AdaGrad leads early but loses by epoch 45). (c) Then run the FULL comparison ONCE "
            "per optimizer at its SELECTED lr — the final grid stays the SAME SIZE as a "
            "single-lr run, only a cheap tuning phase is prepended (no wall-clock blow-up). "
            "EMIT THIS AS A STAGED SEARCH the harness enforces (do NOT hand-wire the phases or "
            "cross-product the grid yourself): in cells.json add a top-level `search` array, one "
            "entry per (family,optimizer) = {\"group\": <id>, \"select_metric\": "
            "\"final_train_loss\", \"select_objective\": \"min\", \"candidates\": [short-epoch "
            "cells over the lr grid], \"promote\": {the ONE full cell with full epochs}, "
            "\"param_from_winner\": [\"lr\"]}. The harness runs the candidates, picks each winner "
            "by select_metric, budget-checks the remaining wall-clock, and runs ONE full cell per "
            "group at the tuned lr. "
            "Record the selected per-optimizer lr in provenance.json so the grader confirms the "
            "comparison was fair. A faithful Adam reaches training loss within ~2x of the best "
            "baseline, not 100-1000x above it; if it STILL does after a FAIR per-optimizer tune, "
            "that is an honest finding — record it truthfully, never fabricate agreement.\n"
            "ADDITIVE convergence evidence — ONLY after the per_model scalars for all six "
            "families are in (never instead of them): the paper's headline claims are about "
            "CONVERGENCE SPEED, so also emit a per-epoch `history` block "
            "(history.<experiment>.<optimizer> = {epoch:[...], <metric>:[...]}) on a COMMON "
            "x-axis with identical init across optimizers (fair_comparison.snapshot_init_state "
            "once -> restore_init_state before each optimizer -> record init_fingerprint per "
            "optimizer in provenance.json); write the VAE sweep results under metrics.json "
            "`vae_lr_sweep`; emit cumulative `regret` as a time-series ARRAY (not a scalar); "
            "render fig_4 as loss-vs-log10(alpha) with a LOG x-axis (axis sidecar JSON). "
            "CIFAR-10 preprocessing MUST apply global-contrast-normalization (per-image) THEN "
            "ZCA whitening (fit the ZCA matrix on the TRAIN set covariance — "
            "U,S,_=svd(cov); ZCA=U@diag(1/sqrt(S+eps))@U.T — apply to train+test) and record "
            "the flag in provenance.json: the 2026-06-13 run scored the data-fidelity leaf 0.4 "
            "for 'no whitening transform found'. The MLP/MNIST panel includes AdaDelta + SFO "
            "baselines. Call "
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
    "2511.14582": PaperHint(
        guidance=(
            "OmniZip (2511.14582) — TRAINING-FREE inference paper: the claim is that "
            "audio-guided token compression accelerates Qwen2.5-Omni audio-video "
            "inference (~2.5-3.4x prefill speedup, ~1.4x lower peak memory) while "
            "keeping accuracy. PRIORITY ORDER: (1) uncompressed Qwen2.5-Omni-7B "
            "baseline + OmniZip at the 45%-retained setting (rho_a=0.3, rho_v=0.6) on "
            "WorldSense and ShortVid-Bench bounded subsets; (2) random-pruning control "
            "at the SAME retention; (3) Qwen2.5-Omni-3B mirror; (4) DyCoke-style "
            "competitor only if time remains.\n"
            "ALGORITHM (the rubric inspects these): segment the interleaved audio-video "
            "sequence into fixed time windows (~50 audio + 288 video tokens per "
            "window); score per-window AUDIO retention via dominant-audio-token "
            "selection (information-density / event-boundary prior); allocate "
            "per-window VIDEO pruning INVERSELY to audio retention, bounded by "
            "rho_max=0.75 / rho_min=0.35 (k=5; group size G=3 here), holding the TOTAL "
            "pruning rate constant; preserve audio-anchor video tokens via cross-modal "
            "similarity; merge surviving video tokens with the interleaved "
            "spatio-temporal scheme. Frames cap 128; identical caps for every method.\n"
            "MEASUREMENT (paired, or the speedup claim is unfalsifiable): the SAME "
            "bounded subset (>=200 samples, fixed seed, identical prompts/decoding) for "
            "baseline and every compression method; per cell record accuracy, mean "
            "prefill + end-to-end wall-clock per sample (warmup excluded), and "
            "torch.cuda.max_memory_allocated; report speedup as baseline_time / "
            "method_time computed from YOUR OWN measured baseline, never paper "
            "constants. Use FlashAttention if installable else SDPA — for ALL methods "
            "equally, recorded in provenance. Do NOT let any method silently fall back "
            "to CPU or fp32.\n"
            "HOSTING: the 7B does NOT fit one 24 GB card in bf16 — each cell's slot "
            "exposes >=2 GPUs in CUDA_VISIBLE_DEVICES: load the model ONCE with "
            "torch_dtype=bfloat16 and device_map='auto' (sharded hosting across the "
            "visible cards; FSDP-style full sharding also acceptable), and disable the "
            "talker / audio-output head (text answers only) to save ~2 GB. Set "
            "est_vram_gb ~32 for 7B cells and ~14 for 3B cells in cells.json.\n"
            "STRUCTURE: cells.json with ONE cell per (model, benchmark, method) and "
            "EXPLICIT model_key/env/baseline axes (e.g. model_key='qwen2_5_omni_7b', "
            "env='worldsense', baseline='omnizip_r45'; uncompressed = "
            "baseline='uncompressed'); each cell writes a FLAT metrics.json and the "
            "aggregate fills per_model ATOMICALLY as cells finish so a timeout "
            "truncates the tail, never finished work. Declared out-of-scope (write "
            "MECHANICALLY into metrics.json scope.gaps, never fake or silently omit): "
            "AVUT and VideoMME full runs, and the FastV-7B cell (OOMs a 48 GB card "
            "even in the paper; run FastV on the 3B instead if attempted).\n"
            "LESSONS FROM ATTEMPT 1 (2026-06-11, scored 0.656 — these are the exact "
            "points it lost): (1) AGGREGATION IS MANDATORY — after EVERY "
            "run_experiment, rebuild the canonical code/metrics.json "
            "per_model[<model>][worldsense][<arm>] from ALL completed cell outputs; "
            "attempt 1 measured a full 200-sample table but never aggregated it "
            "(an unhashable-dict TypeError in gap dedup — stringify gap items), so "
            "the grader read stale files and result-match leaves scored 0. "
            "(2) SHARED CACHE — pass cache_dir='/home/sww35/openresearch/runs/.cache/hf/hub' "
            "to every from_pretrained/snapshot_download; Qwen2.5-Omni-7B AND -3B are "
            "ALREADY cached there; never write per-cell hf_cache dirs (attempt 1 "
            "duplicated 24 GB and filled the disk). (3) LOADER — use the PROVEN "
            "attempt-1 recipe: the FULL Qwen2_5OmniForConditionalGeneration class with "
            "try: enable_audio_output=False on from_pretrained, except TypeError: "
            "construct then model.disable_talker() — measured 38% accuracy @ "
            "8.7s/sample. Do NOT use the thinker-only class: attempt 2 did and got 0% "
            "accuracy with 4x slower generation (different multimodal generate "
            "plumbing). Never let any torch.load path run (torch is env-pinned < 2.6); "
            "never materialize full-sequence lm_head logits when measuring prefill — "
            "model(**inputs, num_logits_to_keep=1) (a full-logits forward OOMs the "
            "2x24GB slot); generation max_new_tokens<=32 via the processor's chat "
            "template; DEBUG GATE before any grid: 3 samples with raw decoded output + "
            "parsed letter + gold printed, then assert accuracy>=20% on 10 samples. "
            "(4) ShortVid-Bench IS "
            "AVAILABLE: HF dataset 'TencentARC/ShortVid-Bench' (1000 audio+video MCQs) "
            "— attempt 1 guessed a wrong repo id and falsely declared it unavailable. "
            "(5) WorldSense recipe: hf_hub_download the QA json + ONLY the "
            "worldsense_videos_*.zip files covering the <=200-sample subset; assert "
            "each clip exists; print/heartbeat at least every 60s during downloads (a "
            "quiet download was watchdog-killed). (6) CHEAP EVIDENCE attempt 1 left on "
            "the table — do ALL of these: 3B mirror cells (Table 3: 3.27x @35%); "
            "per-domain WorldSense scores across the 8 domains (save per-sample "
            "{id, domain, correct, prefill_ms} records); pruning-step overhead timing "
            "(paper claims <40 ms — time compress_sequence alone, record mean ms); "
            "Table-5 ablations (dynamic-video-pruning-only and audio-anchor-only arms "
            "on the same subset); DyCoke-TTM stage-1 comparison arm; flash_attention_2 "
            "if a prebuilt wheel matches the pinned torch, else record attn_impl=sdpa "
            "identically for every arm."
        ),
        default_scope=ScopeSpec(
            models=[
                "Qwen2.5-Omni-7B",
                "Qwen2.5-Omni-3B",
            ],
            datasets=[
                DatasetSlice(name="WorldSense"),
                DatasetSlice(name="ShortVid-Bench"),
            ],
            seeds=[1],
        ),
        invariants=[
            InvariantSpec(
                name="real_qwen_omni_weights_not_surrogate",
                rationale=(
                    "The speedup/accuracy claim is only meaningful on the real "
                    "Qwen2.5-Omni checkpoints from HuggingFace; a surrogate or "
                    "random-init model cannot validate token compression."
                ),
                must_match=[
                    r"Qwen2\.5-Omni",
                ],
            ),
            InvariantSpec(
                name="sharded_multi_gpu_hosting",
                rationale=(
                    "The 7B model exceeds one 24 GB card in bf16; the cell must "
                    "shard across its visible GPUs (device_map / accelerate / "
                    "FSDP) instead of OOMing or downcasting."
                ),
                must_match=[
                    r"device_map|accelerator\.prepare|FSDP|fully_sharded",
                ],
            ),
            InvariantSpec(
                name="window_pruning_ratio_bounds",
                rationale=(
                    "OmniZip's dynamic allocation is bounded by rho_max=0.75 and "
                    "rho_min=0.35 (paper Sec. 4.1); without the bounds the "
                    "constant-total-pruning property collapses."
                ),
                must_match=[
                    r"0\.75",
                    r"0\.35",
                ],
            ),
        ],
        blocked_resources=[
            "github.com/KD-TAO/OmniZip",
        ],
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
