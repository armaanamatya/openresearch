# Beating the 0.8308 Adam run — fidelity-improvement design (2026-06-09)

**Status:** design only. The live Adam/All-CNN runs are NOT to be disturbed. Implementation
pairs with the live run's verdict — if iteration 2 lands under 0.8308, this is the rerun plan.

**Bar to beat:** `origin/main:best_runs/adam/final_report.json` → **overall 0.8308**, verdict
`reproduced`, 20 leaves, `meets_target=True`. Models: planner/verifier/grader claude-oauth,
executor claude-sonnet-4-6.

---

## 1. What produced the 0.83 — and the decisive finding

The 0.83 run reproduced all six of the Adam paper's experiment families (MNIST logreg, IMDB
BoW, MLP/MNIST, CIFAR-10 CNN, VAE bias-correction, VAE Fig-4 sweep) with correct architectures
and optimizer math — that area (**Method & code fidelity**) scored **0.985 at the dominant
weight (0.339)**. It is essentially maxed; there is almost no headroom there.

**The decisive gap:** its `code/metrics.json` is **entirely final scalars** —
`mnist_final_accuracy`, `cifar10_cnn_adam_nll`, `vae_adam_bc_elbo`, … — **zero per-step /
per-epoch convergence curves** (verified: 0 curve-like arrays in the file). The report's
`paper_claims` array is **empty** (no claim↔observed binding).

The Adam paper's headline claims are about **convergence SPEED / training cost** (reach a target
loss in fewer iterations), not final test accuracy. Judged on final scalars the optimizer
*ordering inverts the paper*:

| experiment | final-scalar reality in the 0.83 run | paper's actual claim |
|---|---|---|
| CIFAR-10 CNN | SGD-Nesterov 0.745 **>** Adam 0.695 | Adam converges *faster* (not higher) |
| IMDB BoW | all ~0.82–0.83; SGD-Nesterov wins NLL | Adam's advantage is in convergence |
| MNIST LR | Adam≈SGD on final acc | Adam's curve drops faster |

So the grader — correctly, given the evidence it was handed — repeatedly wrote "convergence
behavior isn't directly verifiable," "advantage not demonstrated," "convergence-speed curves
are not directly shown." **The reproduction answered the wrong axis.**

---

## 2. Weak-spot map, ranked by leverage (score × weight headroom)

Areas (weight) sorted by score:

| area | score | weight | headroom→0.92 | overall gain |
|---|---|---|---|---|
| **Result match vs paper targets** | 0.613 | 0.169 | +0.307 | **+0.052** |
| **Evaluation protocol & metric correctness** | 0.772 | 0.144 | +0.148 | **+0.021** |
| Artifact completeness & provenance | 0.775 | 0.068 | +0.145 | +0.010 |
| Experiment execution & reproducibility | 0.881 | 0.153 | +0.069 | +0.011 |
| Data & preprocessing fidelity | 0.880 | 0.127 | +0.070 | +0.009 |
| Method & code fidelity | 0.985 | 0.339 | ~0 | ~0 |

**Realistic ceiling ≈ 0.83 + 0.10 ≈ 0.93** (matches the scoring-fairness memo's "Adam re-run
target 0.92"). The Result-match + Eval-protocol pair is ~70% of the available lift, and **both
are driven by the same missing-convergence-curves root cause.**

### Weak leaves (grader justifications), clustered by root cause

- **C1 — convergence/training-cost trajectory not surfaced** (the lever):
  - `cec0919d` 0.40 — IMDB: "large-margin advantage of Adam/AdaGrad over SGD is not demonstrated" (final scalars near-identical).
  - `cc3e019a` 0.60 — CIFAR CNN: "45-epoch convergence behavior isn't directly verifiable from the summary metrics."
  - `ba40317d` 0.80 — MNIST LR: "convergence-speed curves are not directly shown."
  - `32bc979d` 0.55 (partial) — VAE: "early-epoch instability claim is not evidenced."
- **C2 — fair-comparison protocol (identical init + dense grid across methods)**:
  - `aa97209f` 0.60 — "dense momentum grid and identical initialization across optimizers are not evidenced."
- **C3 — data-preprocessing fidelity (paper-specific)**:
  - `6047d3dc` 0.60 — CIFAR "per-channel mean/std as a 'ZCA approximation' rather than the paper's actual whitening."
- **C4 — provenance/evidence surfacing**:
  - `0f5c1035` 0.70 — "minibatch sizes and full per-experiment hyperparameter records are not clearly surfaced."
  - `a9817c23` 0.75 — "45-epoch training length is not confirmed in the evidence."
  - `92509d69` 0.80 — "log-scale axis cannot be directly confirmed."
- **C5 — result-match claim binding** (report `paper_claims=0`): the grader had no structured
  claim↔observed table, so Result-match is judged loosely against prose.
- **C6 — experimental-regime selection (paper-specific)**: VAE no-bias instability only manifests
  at the high-α/LR regime (`32bc979d`).

---

## 3. The solution — modular, dynamic, paper-agnostic

Principle: **make the reproduction emit the evidence the paper's claims are actually about**,
in a *structured* form the grader reads, *gated on the claim type* so it generalizes to any
paper — not hardcoded to Adam. Three new copyable stdlib helpers + two existing tree capabilities
wired in + paper-specific bits in the hint surface. Every component flag-gated, fail-soft,
0-regress, registered in `baseline_implementation._HARNESS_CODE_HELPERS`.

### Module A — `convergence_evidence.py` (NEW, generic, the lever) → fixes C1, lifts Result-match + Eval-protocol
- **Structured curves, not log.** Schema-enforced `history` block in `metrics.json`:
  `history[experiment][method] = {"step":[…], "train_loss":[…], "val_metric":[…]}`. The
  current convergence-trajectory prompt fix backfired precisely because the agent dumped curves
  to *stdout* and dropped structured keys (live iter-1: eval-protocol area crashed 0.772→0.21,
  four 0.0 leaves). `rubric_guard.assert_metrics_schema` REQUIRES the `history` block when the
  paper's claim-type is `convergence|training_cost|sample_efficiency`.
- **Deterministic derived evidence.** From the curves compute, per method:
  `iterations_to_threshold` (steps to reach a common target loss), `auc_loss`, and
  `final` — the comparison the grader needs ("Adam reaches L=0.4 in 1.8k steps vs SGD 4.1k").
- **Figure + axis metadata.** Render the convergence figure with the paper's axis convention
  (log-scale where the paper uses it) and record `{"x":"iterations","y":"train_loss","yscale":"log"}`
  so `92509d69`-type "axis cannot be confirmed" leaves resolve.
- **Verify binding.** `verify_against_rubric` is handed the derived evidence explicitly.
- **Dynamic gate:** engages only when claim-type detection (from the extracted ReproSpec / claim
  text) flags a convergence-style claim. Non-convergence papers: no-op.

### Module B — `fair_comparison.py` (NEW, generic) → fixes C2, de-noises C1
- One initial-weights snapshot per architecture, **reused across every compared method**
  (seed + `state_dict` SHA recorded). Removes init noise that makes final-scalar orderings
  meaningless and directly answers `aa97209f` ("identical initialization across optimizers").
- Records the swept grid (LR × momentum) as structured provenance.

### Module C — `provenance_manifest.py` (extend the scoring-fairness D2 manifest) → fixes C4
- Per-experiment structured record: hyperparameters (α,β1,β2,ε,LR,minibatch), epochs-trained,
  init-snapshot hash, preprocessing description, figure axis metadata → `provenance.json`,
  fed to verify. Resolves `0f5c1035` / `a9817c23` / `92509d69`. The scoring-fairness spec already
  implemented a D2 manifest (flag-gated) — extend its field set, don't rebuild.

### Engine D — wire the existing two-axis Extractor→ReproSpec (U11–U18, already in tree) → fixes C5
- The anti-circular Extractor already extracts the paper's claimed targets into a frozen
  ReproSpec; emit a structured **claim↔observed table** with graded-magnitude credit (task U12,
  currently pending) into the report's `paper_claims` and surface it to the grader. This is the
  generic Result-match engine for *any* paper — no Adam-specific logic.

### Paper-specific (Adam) → `PAPER_HINTS` + invariants, NOT harness code → fixes C3, C6
- C3: CIFAR-10 GCN + **real ZCA whitening** (not per-channel approx) as an Adam invariant +
  preprocessing-provenance string.
- C6: VAE bias-correction at the **high-α/LR regime** where the no-bias instability manifests +
  emit the early-epoch ELBO trajectory.
- IMDB BoW exact setup (10k vocab + dropout) so Adam's convergence advantage is visible.

---

## 4. Why this is precise / robust / modular / scalable / dynamic

- **Precise:** every module maps to named leaves with grader justifications (§2).
- **Robust:** flag-gated default-OFF, fail-soft (a missing curve never crashes training),
  schema-enforced so the agent can't silently regress (the C1 schema guard is the exact backstop
  the live iter-1 regression lacked), 0-regress requirement.
- **Modular:** A/B/C are independent copyable helpers; D and the paper-hints are separate
  surfaces. Any one ships alone.
- **Scalable:** helpers register in `_HARNESS_CODE_HELPERS` and copy into any paper's sandbox
  like `rubric_guard.py`; nothing is Adam-specific in the harness layer.
- **Dynamic:** claim-type detection drives *which* modules engage — a convergence paper gets A;
  a result-table paper leans on D; a multi-method paper gets B. Adam just happens to trip all of
  them.

## 5. Build order (highest leverage first), no live-run disturbance

1. **Module A** (convergence_evidence + schema guard) — ~70% of the lift; also the exact fix for
   the live run's iter-1 regression. TDD, flag `OPENRESEARCH_CONVERGENCE_EVIDENCE`.
2. **Engine D wiring** (claim↔observed table, U12 graded credit) — Result-match binding.
3. **Module C** (extend D2 provenance manifest) — cheap, closes the provenance area.
4. **Module B** (fair_comparison init snapshot) — de-noises orderings.
5. **Adam PAPER_HINTS** (ZCA, VAE regime, IMDB) — paper-specific last.
6. Rerun Adam with all flags on; target ≥ 0.92. Only then package into `best_runs/adam`
   (replacing the 0.83) + push, per the operator directive (score must exceed 0.8308).

Expected: C1+D ≈ +0.07 (Result-match 0.61→~0.90, Eval-protocol 0.77→~0.92); +C +B +hints ≈
+0.03 → **~0.92–0.93 overall.**
