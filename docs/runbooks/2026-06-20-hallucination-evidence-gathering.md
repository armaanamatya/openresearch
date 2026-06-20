# Workstream C — Hallucination Evidence-Gathering Runbook

> **Status:** operator-gated · created 2026-06-20 · pairs with the diagnosis
> `docs/audits/2026-06-20-workstream-c-hallucination-diagnosis.md` (this branch).
>
> **Decision (2026-06-20):** *generate fresh runs first*. The hallucination
> premise is **not validatable** from the current on-disk corpus (8 run dirs,
> **all `failed`/`partial`** — none reached a shippable graded headline, which is
> exactly where report-level fabrication would surface). Before designing or
> flipping any gate, gather a small set of runs that actually reach a graded
> headline, then re-run the diagnosis against them.

## Why this is gated, not automated

Per the cross-cutting constraint: **ZERO paid GPU / no live API spend** may be
incurred autonomously. Every command below **costs money** (GPU node-hours +
LLM tokens) and must be launched by the operator. The harness will not run them.

## What we already know (from the diagnosis — do not re-derive)

The harness is **honest at the report level** on every observed run
(`degraded_no_metrics` zeroing + the verdict gate `report.py::_apply_evidence_gate`,
default ON, both held). Strong-form "fabricated shipped results" is **NOT**
substantiated by local data. Three concrete defects were found regardless:

| # | Defect | Location | Class |
|---|---|---|---|
| 1 | `run_experiment` reports `success=true` while every per-model cell errored (checks only subprocess exit, not cell/per-model status) | `backend/agents/rlm/primitives.py:3982` (`success=all(r.succeeded …)`) | C2 — harness gap |
| 2 | Executor sub-agent fabricated an unsupported `from_pretrained(dtype=…)` kwarg | proven by the error in `experiment_runs.jsonl`; gate caught it | C3 — no API-kwarg lint |
| 3 | `OPENRESEARCH_EVIDENCE_GATE` has **two consumers with opposite defaults** — leaf veto OFF (`evidence_gate.py:48`), verdict gate ON (`report.py:1496`) | config defect | C1 — flag default mismatch |

The fresh-run goal is to determine the **DECIDE** question with evidence:
**model-quality issue** (model fabricates in the *shipped report* → fix = provider/
model routing) **vs harness-trust issue** (model fabricates but a gate should
veto it → fix = evidence gates). These have **opposite fixes**.

## Step 1 — Launch fresh runs (operator, COSTS MONEY)

Use a **reliable root** (gpt-5 — `claude-oauth` degenerates per
`OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD`) and a **cost-bounded** paper so the
runs actually reach a graded headline. Target **≥3 completed runs**, ideally a
mix (1 strong paper expected to pass, 1 marginal, 1 different domain) so we can
observe a *passing* report — the only place report-level fabrication shows.

```bash
# Reliable + cost-bounded. SDAR smallest-two scope (the canonical stress paper):
.venv/bin/python -m backend.cli reproduce 2605.15155 \
    --model gpt-5 --sandbox runpod \
    --models executor=gpt-5,grader=gpt-5,verifier=gpt-5 \
    --max-usd 8 --project-id hallu_evidence_sdar_01

# A second, cheaper/different-domain paper (pick one the team expects to pass):
.venv/bin/python -m backend.cli reproduce <arxiv-id> \
    --model gpt-5 --sandbox runpod --max-usd 6 \
    --project-id hallu_evidence_paper2_01

# Optional third for variance.
```

Notes:
- Keep `OPENRESEARCH_ACCELERATOR=off` unless you specifically want to characterize
  the navigation tier (a separate hallucinator).
- Do **not** flip any of the C fix flags during evidence-gathering — we need the
  *baseline* (current-default) behaviour to see what leaks through.
- RunPod COMMUNITY (`OPENRESEARCH_RUNPOD_CLOUD_TYPE=COMMUNITY`) for cheapest GPU.

## Step 2 — Capture artifacts per run

For each `runs/<project_id>/`, the evidence set is:
- `final_report.json` / `final_report.md` — the **shipped headline** (the thing a
  fabrication would live in).
- `rubric_evaluation.json` + `rubric_tree.json` — per-leaf grades + justifications.
- `experiment_runs.jsonl` — every `run_experiment` (success bool, per-model status, errors).
- `code/metrics.json` (+ `code/outputs/<run>/metrics.json`) — the on-disk measured truth.
- `cost_ledger.jsonl`, `dashboard_events.jsonl` — provenance + `rubric_score` events.

Preserve them: `touch runs/<project_id>/.preserved` so retention/GC won't reap them.

## Step 3 — Re-run the diagnosis against fresh artifacts

For each run, check the falsifiable questions:
1. Does any **credited leaf** (`rubric_evaluation.json` score > 0) cite a
   `per_model[model][dataset]` cell that has **no successful row** in
   `experiment_runs.jsonl` / `code/metrics.json`? → report-level fabrication that
   the evidence gate should have vetoed (**harness-trust**).
2. Does `final_report.json` headline a metric with **no matching measured value**
   on disk? → fabrication in the shipped report (**harness-trust** if a gate
   should catch it; **model-quality** only if the model invents numbers the gate
   structurally cannot see).
3. Did `run_experiment` log `success=true` on an all-errored grid (defect #1)? →
   harness gap, fix #1.
4. Did a sub-agent fabricate API/lib usage (defect #2)? → model-quality at the
   sub-agent tier; the fix is a lint/guard, not a different provider.

Write the verdict (model-quality vs harness-trust, **per example + overall**) as
an update to the diagnosis doc.

## Step 4 — Only then: implement fixes (separate PR)

Prefer enabling/tightening an existing gate over new machinery. Ranked closures:
1. **Defect #1 (smallest, fail-closed):** mirror the cells-route status rule
   (`cell_matrix.py:822-827`) into the monolithic `run_experiment` success calc
   so an all-errored grid cannot report `success=true`. Default-OFF flag until A/B.
2. **Defect #3:** make `OPENRESEARCH_EVIDENCE_GATE`'s leaf-veto default match its
   verdict-gate twin (calibrate grader σ first — see below).
3. **Defect #2:** a default-OFF deterministic AST lint over generated `code/` that
   flags unknown `from_pretrained`/known-API kwargs (sketch in the diagnosis doc).

## Validation policy (per repo rules)

- **≥3 paired A/B runs** before flipping any default (`scripts/ab_compare.py`,
  `OPENRESEARCH_AB_ARM`/`_AB_PAIR_ID`; keep `OPENRESEARCH_BES_ADAPTIVE` OFF on arms).
- If any change touches the **grader**, run `scripts/calibrate_grader.py` and
  confirm σ stays ≤ 0.02 (the gate from the grader-fidelity work).
- Tests are **unit/replay over the saved on-disk artifacts** from Step 2 — **ZERO
  paid GPU**. The fresh runs are gathered ONCE (here); fixes are then validated
  against those frozen artifacts.

## Acceptance for this workstream

1. ≥3 completed runs reaching a graded headline, artifacts `.preserved`.
2. Diagnosis doc updated with the per-example + overall model-quality-vs-harness-
   trust verdict, each backed by `file:line` + run-artifact evidence.
3. A prioritized fix list, each fix mapped to an existing/new gate with an A/B
   plan — implementation in a **separate PR**, after the evidence review.
