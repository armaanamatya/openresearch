# Validation-first SDAR A/B runbook (2026-05-30)

> **Why this exists.** The 2026-05-30 strategic review's verdict: *stop building, validate the
> reliability core in a real run.* Phases 8 (context map) and 9 (negative lessons) and the tightened
> evidence gate are all **unproven on a real SDAR+GPU run** — `3156` green unit tests prove designed
> behavior in isolation, not survival under real chaos. This runbook is the validation the code work
> can't do for itself: it costs RunPod + OpenAI credits and must be launched by the operator.
>
> **Sequencing (from the review):** prove the core stable FIRST, add Phase 8 SECOND, add Phase 9 only
> AFTER a clean run confirms evidence-truth. Do not flip a flag to default until its arm wins its A/B.

## Standing config (all arms)

- **Root model:** `--model claude-oauth` (Claude subscription). **Sub-agents:** `claude-sonnet-4-6` (never Opus).
- **Sandbox:** `--sandbox runpod` (COMMUNITY ≈ $0.34/hr). You have `REPROLAB_RUNPOD_API_KEY`.
- **Env hygiene:** prefix with `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` if a stale shell export might
  shadow `.env` (the 2026-05-28 `401`/`credit balance too low` trap). The boot-time validator warns on a
  shadow, but the prefix is the belt-and-suspenders.
- **Evidence gate:** `REPROLAB_EVIDENCE_GATE=1` (default). Leave ON for every arm — validating it is a goal.
- **Scope:** smallest-two (Qwen3-1.7B + Qwen2.5-3B) to keep wall-clock/cost bounded, via
  `REPROLAB_BASELINE_EXTRA_GUIDANCE` (below). This is the canonical baseline from the 2026-05-23 handoff.

```bash
export REPROLAB_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two SMALLEST model variants the paper tests — Qwen3-1.7B-Instruct and Qwen2.5-3B-Instruct. SKIP Qwen2.5-7B entirely. Use the real pretrained weights from HuggingFace (no surrogate) and the real ALFWorld + Search-QA + WebShop datasets, but evaluate on a small representative slice (e.g. 32 tasks per env) to keep wall-clock practical on a single 24–48GB GPU. Report results for both 1.7B and 3B."
```

## Arm A — core-only baseline (Phases 8 & 9 OFF)

Proves the reliability core (read-idle timeout, typed failures, stall detector, evidence gate, gpt-5-mini
route if enabled) survives a real run before any research prototype is layered on.

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
REPROLAB_CONTEXT_MAP=off REPROLAB_NEGATIVE_LESSONS=0 \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --vram-gb 38 --max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20 \
  --project-id sdar_armA_core_$(date +%s)
```

**Gate to pass before Arm B:** run reaches a terminal report; no `sub_rlm_stalled` wedge; the final verdict
is *consistent with the evidence* (see "Evidence-truth check" below). If the run wedges or the verdict is
hollow, fix the core first — do NOT proceed to Phase 8/9.

## Arm B — + Phase 8 context map (`REPROLAB_CONTEXT_MAP=on`)

Same command, flip one flag:

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
REPROLAB_CONTEXT_MAP=on REPROLAB_NEGATIVE_LESSONS=0 \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --vram-gb 38 --max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20 \
  --project-id sdar_armB_ctxmap_$(date +%s)
```

**A/B win condition (≥3 paired runs A vs B):** lower iteration count AND/OR fewer `rlm_query`+`llm_query`
calls, with **`rubric.overall_score` not regressing**. Expect a *smaller* delta than PEEK's headline (a
per-slice `primitive_cache` already exists; the map's unique value is cross-slice aggregation). If the delta
is within noise, keep it OFF — it's not worth the inventory token.

## Arm C — + Phase 9 negative lessons (`REPROLAB_NEGATIVE_LESSONS=1`)

**Only after Arm A produced a clean, evidence-true run.** Phase 9 mines per-paper failure lessons; a lesson
needs `occurrences >= 2` to promote (except `dockerfile_invalid`), so its effect only shows on the **2nd+**
run of the same arxiv id. Run it ≥2× to let a lesson promote, then compare against an Arm-A pair.

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
REPROLAB_CONTEXT_MAP=off REPROLAB_NEGATIVE_LESSONS=1 \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --vram-gb 38 --max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20 \
  --project-id sdar_armC_lessons_$(date +%s)
```

**A/B win condition:** a class-tagged lesson cuts recurrence of *its* `failure_class` on the next run
without lowering `rubric.overall_score`. Inspect `runs/_lessons/2605.15155.json` between runs to confirm a
lesson promoted `candidate -> active`.

## Metrics harvester (run after each arm)

```bash
# Usage: ./harvest.sh runs/<project_id>
python3 - "$1" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
fr = json.loads((d/"final_report.json").read_text()) if (d/"final_report.json").exists() else {}
ev = [json.loads(l) for l in (d/"dashboard_events.jsonl").read_text().splitlines() if l.strip()] if (d/"dashboard_events.jsonl").exists() else []
def ev_type(e): return e.get("event") or e.get("type") or ""
prim = [e for e in ev if ev_type(e) == "primitive_call"]
def name(e): return (e.get("data") or e).get("primitive") or (e.get("data") or e).get("name") or ""
counts = {}
for e in prim: counts[name(e)] = counts.get(name(e), 0) + 1
stalls = [e for e in ev if "stall" in ev_type(e).lower() or (ev_type(e)=="run_warning" and "stall" in json.dumps(e).lower())]
exp = [json.loads(l) for l in (d/"experiment_runs.jsonl").read_text().splitlines() if l.strip()] if (d/"experiment_runs.jsonl").exists() else []
succ = [r for r in exp if r.get("success") is True and isinstance(r.get("metrics"),dict) and r.get("metrics")]
print(f"run:              {d.name}")
print(f"verdict:          {fr.get('verdict')}")
print(f"rubric.overall:   {(fr.get('rubric') or {}).get('overall_score')}")
print(f"iterations:       {fr.get('iterations')}")
print(f"rlm_query calls:  {counts.get('rlm_query', 0)}")
print(f"llm_query calls:  {counts.get('llm_query', 0)}")
print(f"sub_rlm_stalled:  {len(stalls)}")
print(f"experiment rows:  {len(exp)}  (clean success+metrics: {len(succ)})")
print(f"started/completed:{fr.get('started_at')} -> {fr.get('completed_at')}")
PY
```

Fill one row per run:

| arm | run id | verdict | rubric.overall | iters | rlm_query | llm_query | sub_rlm_stalled | clean-exp rows | wall-clock |
|-----|--------|---------|----------------|-------|-----------|-----------|-----------------|----------------|------------|
| A   |        |         |                |       |           |           |                 |                |            |
| B   |        |         |                |       |           |           |                 |                |            |
| C   |        |         |                |       |           |           |                 |                |            |

## Evidence-truth check (the point of the tightened gate)

For each run, confirm the verdict the gate produced is *consistent with the evidence on disk*:

```bash
.venv/bin/python scripts/replay_run.py runs/<project_id>
```

- A `reproduced`/`partial` verdict **must** have ≥1 `experiment_runs.jsonl` row with `success==true` and
  non-empty metrics (the harvester's "clean-exp rows" > 0). If verdict is success-ish but clean-exp rows == 0,
  the gate failed to fire — that's a P0 regression in `report.py::_apply_evidence_gate`.
- A `failed` verdict with clean-exp rows > 0 means the gate **over-fired** — investigate (the
  `positive_control_real_evidence` replay case guards against this in CI, but a real run is the real test).

## Decision

- Flip `REPROLAB_CONTEXT_MAP` / `REPROLAB_NEGATIVE_LESSONS` to default-on **only** when its arm wins its A/B
  on ≥3 paired runs with no `rubric.overall_score` regression. Otherwise leave OFF (zero residue when off).
- If Arm A itself can't produce an evidence-true run, the review is right: the work is at the *prove-it*
  stage, not the *add-more-machinery* stage. Stabilize the core before touching 8/9.
