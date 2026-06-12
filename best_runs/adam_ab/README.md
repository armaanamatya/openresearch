# Adam with/without-BES A/B — pair `adam-ab-20260611` (CONFOUNDED — directional only)

Second BES measurement (paper: *Adam: A Method for Stochastic Optimization*,
arXiv 1412.6980), run the same day as the clean
[`allcnn_ab`](../allcnn_ab/README.md) pair. **Read this one as directional,
not controlled** — the confounds below are material and both favor the
control.

## Result

| | control | BES | Δ (bes − control) |
|---|---:|---:|---:|
| **rubric score** | 0.716 | 0.5327 | **−0.1833** |
| verdict | reproduced | failed¹ | |
| iterations | 6 | 7 | |
| wall clock | 14.6 h | 20.0 h | +5.5 h |
| LLM cost | $2.92 | $0² | |

¹ Verdict reconciliation against the floored 0.8308 target (the paper's
all-time best); the BES arm's best in-run verify was 0.5686.
² The BES arm ran fully on the OAuth subscription (no API spend recorded).

## Why this pair is confounded

1. **The control was heavily operator-steered; the BES arm ran autonomous.**
   The control received four mid-run interventions through the steering
   channel — a device-side-assert diagnosis, the in-place `per_model` reshape
   (its 0.0→0.618 jump), the floor re-imposition, and a prioritized weak-leaf
   plan (0.624→0.716). The BES arm received one late steering message ~2 h
   before its ceiling. This asymmetry alone plausibly exceeds the measured Δ:
   the steering playbook has since been automated as `leaf_triage.py`
   precisely because it was worth that much.
2. **Different flag sets.** The control's relaunch dropped three rail flags
   (`SEED_BEST_ATTEMPT`, `TARGET_BEST_FLOOR`, `SCOPE_INCLUSION_EXCLUDE`) and
   predates the A/B stamps (it pairs as `unstamped`); the BES arm carried the
   full rail set + pinned rubric.
3. **Different effective budgets.** The control burned its first 8 h on a
   grid pass a primitive-timeout interrupted, then recovered; the BES arm
   spent its full 20 h but lost its last verify window to in-flight endgame
   repairs that never got re-scored (its banked best, 0.5686, shipped as
   0.5327 because the finalize rescore had moved scope — the deliberate
   floor-skip condition).

## What the pair still says

- **The pool discriminated strongly on Adam** — fidelity-first won 0.643 vs
  0.546, a 10× wider spread than All-CNN's pool (0.557 vs 0.549), consistent
  with Adam's historically high implementation variance. Selection signal is
  real; it didn't convert into a final-score win under the confounds above.
- Combined with [`allcnn_ab`](../allcnn_ab/README.md) (+0.085 BES, clean
  conditions), the campaign evidence is **mixed: 1 clean win, 1 confounded
  loss** — exactly why the repo policy requires ≥3 paired runs before
  flipping any default. The third data point should be a clean Adam pair
  (both arms autonomous, identical flags, with `leaf_triage` active in both).

## Pool record (static SELECT scores)

| candidate | angle | score | |
|---|---|---:|---|
| `rlm_impl#0` | parity | 0.5464 | |
| `rlm_impl#1` | fidelity-first | 0.6430 | ← selected |

Reproduce: `.venv/bin/python scripts/ab_compare.py --paper 1412.6980 --pair-id adam-ab-20260611`
