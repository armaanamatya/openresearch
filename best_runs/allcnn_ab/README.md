# All-CNN with/without-BES A/B — pair `allcnn-ab-20260611`

First controlled measurement of **BES competing candidates** (Bidirectional
Evolutionary Search v1 — best-of-N implementation selection) on the RLM
reproduction path. Two arms reproduced *Striving for Simplicity: The All
Convolutional Net* (arXiv 1412.6806) under identical conditions, differing in
exactly one variable: the BES candidate pool.

## Result

| | control | BES | Δ (bes − control) |
|---|---:|---:|---:|
| **rubric score** | 0.6526 | **0.7378** | **+0.0852** |
| compute-adjusted | 0.6304 | 0.6925 | +0.0621 |
| verdict | reproduced | partial¹ | |
| iterations | 10 | 10 | |
| wall clock | 13.4 h | 13.3 h | −4 min |
| LLM cost | $4.95 | $3.27 | **−$1.69** |

**BES won on every axis.** The arm that paid ~75 minutes up front for its
candidate pool finished *cheaper and no slower* — the better-selected
implementation needed fewer repair cycles downstream. The BES final (0.7378)
landed 0.0018 below the paper's all-time best reproduction (0.7395).

¹ Verdict reconciliation quirk against the floored target (both arms carried
target 0.7395 = the best ancestor); scores are the comparison, verdicts are
cosmetic here.

## The candidate pool

`implement_baseline` ran twice with angle-diversified prompts; each candidate
was snapshotted and statically graded by the leaf scorer (code-only, no GPU);
the winner's tree became `code/`.

| candidate | prompt angle | static SELECT score | |
|---|---|---:|---|
| `rlm_impl#0` | parity (no extra guidance) | 0.5488 | |
| `rlm_impl#1` | fidelity-first | 0.5567 | ← selected |

Full pool record: [`bes/bes_candidates.json`](bes/bes_candidates.json). The
fidelity-first candidate also won the Adam pool launched the same day
(0.643 vs 0.546 — a 10× wider spread, consistent with Adam's higher
implementation variance).

## Top leaf-level moves (bes − control)

| leaf | control | bes | Δ |
|---|---:|---:|---:|
| `d442e584…` | 0.0 | 0.4 | +0.4 |
| `985ca8ba…` | 0.0 | 0.4 | +0.4 |
| `b7325d4f…` | 0.4 | 0.7 | +0.3 |
| `ad46e5ab…` | 0.4 | 0.7 | +0.3 |
| `85042ac0…` | 0.4 | 0.7 | +0.3 |
| `2fb0fff0…` | 0.2 | 0.0 | −0.2 |

Per-leaf justifications: [`control/rubric_evaluation.json`](control/rubric_evaluation.json)
/ [`bes/rubric_evaluation.json`](bes/rubric_evaluation.json).

## Experiment design (what made the pair clean)

- **Single variable.** Identical flag sets except `REPROLAB_BES_ENABLED=1` +
  `REPROLAB_BES_CANDIDATES_PER_CLUSTER=2` on the BES arm. Both arms:
  fidelity-evidence, preflight+execution smoke, theory-leaf exclusion,
  inclusion-scope, dead-loss early-stop, prior-attempt evidence, seeded best
  attempt, floored target.
- **Pinned rubric.** `REPROLAB_REUSE_RUBRIC=1` with the same pre-seeded
  `generated_rubric.json` in both arms (byte-identical, md5-verified) — no
  per-run LLM rubric drift in the delta.
- **Identical history.** Each arm ran as an independent lineage
  (`batch_reproduce --project-id-suffix`) seeded with a skinny clone of the
  canonical project's attempts, including the 0.7395 best ancestor — so the
  anti-regression rails (champions, evidence, floor) saw the same past.
- **Disjoint hardware, same commit.** Control on GPUs 2+5, BES on GPUs 6+7,
  both launched 2026-06-11 03:01/03:02 UTC from `c5c6fb7`.

## Honest caveats

- **n = 1 pair.** Run-to-run agent stochasticity is roughly ±0.05; +0.085 is a
  strong directional read, not proof. Repo policy: ≥3 paired runs before
  flipping any default.
- A foreign 16 GB job shared GPU 7 with the BES arm for part of the run
  (slower cells on that card) — a wall-clock headwind *against* BES that it
  absorbed anyway.
- The Adam pair (same day) is annotated separately: its control predated the
  A/B stamps and lost three rail flags to a relaunch, so it reads as
  directional only.

## Reproduce the comparison

```bash
.venv/bin/python scripts/ab_compare.py --pair-id allcnn-ab-20260611
```

Machine-readable: [`ab_report.json`](ab_report.json). Harness documentation:
`CLAUDE.md` → “BES competing candidates (both paths) + A/B harness”.
