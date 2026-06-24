# Lifecycle Inversion — Harness-Owned Primary Mode (implementation plan, 2026-06-22)

> Branch `feat/grounded-self-improvement-harness-reliability`. **Increment 2** of the
> root-reliability line. **All new behavior flag-gated, default-OFF → byte-identical when
> off** (the existing `rlm.completion` root-driven path is untouched). Builds directly on
> increment 1 (bounded repair in the driver, already merged + GPU-validating).

## Why (the inversion)

The RLM paradigm delegates lifecycle control flow to the root model. The only **keyless**
roots (claude-oauth, foundry/gpt-chat) are unreliable at driving the mandatory backbone —
they degenerate (read paper → loop `FINAL_VAR`, never `implement_baseline`). The harness has
three *reactive* backstops (forced-iteration refusals, autodrive, the once-only reactive
driver) that only fire *after* the root has already failed.

**Inversion:** make the harness the **proactive primary driver** of the lifecycle, keyed on
the already-canonical `root_progress.infer_required_stage`. The root model is consulted only
through primitives (`implement_baseline`'s executor sub-agent, `propose_improvements`). This
makes "no experiment ever ran" structurally impossible instead of detected-and-scolded, and
makes root orchestration reliability irrelevant for reproduction.

This is the user-chosen "full inversion." It ships flag-gated; after the GCP A/B proves it
(≥3 paired SDAR runs per CLAUDE.md), increment 3 flips the default ON and deletes the
now-dead reactive code (the "supersede the patch stack").

## Scope of increment 2

Primary mode = harness drives **backbone → repair → improvement-climb → finalize**, root
loop skipped. New flag `OPENRESEARCH_LIFECYCLE_PRIMARY` (default OFF). Used for the
keyless/unreliable root (set via the SDAR `--run-spec`/`sdar_gcp.env`); a funded reliable
root leaves it off and uses the normal root-driven path ("Both — cheapest reliable").

### Integration facts (verified, do not re-derive)
- Completion call-site: `run.py:3618` `result_obj = await asyncio.to_thread(_run_completion_on_worker)`.
- In-scope at that point: `custom_tools` (the WRAPPED tools dict), `context_dict["paper_text"]`,
  `context_dict["rubric_spec"]`, `ctx.latest_rubric_target` (the target), `iteration_policy`
  (already on `ctx._forced_iteration_policy`), `emit`.
- `_finalize(result_obj, run_failed, ...)` (run.py:3728): `result_obj is not None` →
  `build_final_report(result_obj)`; else a `verdict="failed"` shell. `build_final_report`
  (report.py:921) parses `result.response` (JSON), then the metric-projection
  (`OPENRESEARCH_METRIC_PROVENANCE`) re-projects `baseline_metrics` from the on-disk artifact.
  The evidence-gate + verdict-cap can only DOWNGRADE an over-claim — so an optimistic synth
  verdict is safe.
- `_hard_stop_with_report` already produces a fully-scored report from disk with no
  `result_obj` — proof that finalize-from-disk works.
- `propose_improvements(current_results: dict, rubric_scores: dict, k=None)` returns a
  **list** of `ImprovementHypothesis` dicts (primitives.py:7865).
- Repairable run signal: `result["outcome"] == "repairable"` (canonical).

## Change A — `lifecycle_driver.py`: expose `verify_result` (one line)

`drive_lifecycle_chain` already holds `verify_result` locally and sets
`summary["rubric_score"]`. Add `summary["verify_result"] = verify_result` (after the
rubric_score line) so the primary driver can feed it to `propose_improvements`. Add
`"verify_result": None` to the canonical no-op summary dict. **No other change** to
`drive_lifecycle_chain` — the reactive path stays byte-identical.

## Change B — `lifecycle_driver.py`: new `run_lifecycle_primary`

A new public function. It REUSES the unchanged `drive_lifecycle_chain` for backbone+repair
and for each improvement's run+repair+verify (start_stage `need_experiment`), adding only the
propose→implement glue + the climb loop.

```python
def run_lifecycle_primary(
    *, tools, ctx, paper_text, rubric_spec, emit,
    target_score: float | None = None,
    max_repair_iterations: int = 2,
    max_improve_iterations: int = 2,
    min_remaining_s: float = 300.0,
) -> dict:
    """Harness-owned proactive reproduction: drive the full backbone (+repair) to a
    scored baseline, then climb toward target via propose_improvements. Fail-soft,
    wall-clock-aware. Returns a summary dict (driven, rubric_score, improved,
    stopped_reason)."""
```

Algorithm:
1. **Backbone:** `base = drive_lifecycle_chain(start_stage="need_baseline",
   max_repair_iterations=max_repair_iterations, ...)`. Seed the aggregate summary from it
   (`driven`, `rubric_score`, `verify_result`). If `base["verify_result"]` is None (backbone
   never reached a score), return early (`stopped_reason = base.get("stopped_reason") or
   "no_baseline_score"`).
2. **Improvement climb** (only if `max_improve_iterations > 0`):
   - `verify_result = base["verify_result"]`; `score = verify_result.get("overall_score")`;
     `target = verify_result.get("target_score")` or `target_score`.
   - `improved = 0`
   - `while score is not None and target is not None and score < target and improved <
     max_improve_iterations:`
     - wall-clock gate (reuse the same `min_remaining_s` check as the driver's `_step`;
       factor a module-level `_wallclock_ok(ctx, min_remaining_s)` — it already exists, reuse it).
       If low → `stopped_reason="low_wallclock"`, break.
     - `improved += 1`
     - `hyps = <fail-soft call> propose_improvements(verify_result, {"overall_score": score,
       "target_score": target})` — via the tools dict (`tools["propose_improvements"]["tool"]`),
       wrapped in try/except (fail-soft, never raise). If empty/not a non-empty list → break.
     - `chosen = hyps[0]`
     - `<fail-soft> implement_baseline({"repair_context": {"improvement": chosen,
       "prior_verify": verify_result}})` via the tools dict. (Patch-mode applies the
       improvement against on-disk code.) On explicit `ok=False` or exception → break.
     - `sub = drive_lifecycle_chain(start_stage="need_experiment",
       max_repair_iterations=max_repair_iterations, ...)` — runs the new code (+repair) and
       re-verifies. Extend the aggregate `driven` with `sub["driven"]`.
     - `verify_result = sub.get("verify_result") or verify_result`;
       `new_score = verify_result.get("overall_score")`. Adopt only if improved
       (`if new_score is not None and (score is None or new_score >= score): score = new_score`)
       — never regress the reported score (best-of-climb).
   - `summary["improved"] = improved`
3. `summary["rubric_score"] = score`; `summary["verify_result"] = verify_result`. Return.

Use a tiny local fail-soft tool caller (mirror the driver's `_get_tool`/`_call`) for the
propose + implement glue. Every step swallows exceptions and degrades to a clean break — a
broken improvement step must never lose the already-earned baseline score.

## Change C — `run.py`: primary-mode branch at the completion call-site

Helpers near `_drive_max_repair` (run.py:1023):
```python
def _lifecycle_primary_enabled() -> bool:
    return os.environ.get("OPENRESEARCH_LIFECYCLE_PRIMARY", "").strip() in ("1", "true", "yes")

def _drive_max_improve() -> int:
    raw = os.environ.get("OPENRESEARCH_LIFECYCLE_MAX_IMPROVE", "").strip()
    return int(raw) if raw.isdigit() else 2   # 0 allowed → backbone only
```

At the completion call-site (run.py:3605-3618), branch the `result_obj` assignment:
```python
if _lifecycle_primary_enabled():
    from backend.agents.rlm.lifecycle_driver import run_lifecycle_primary
    summary = await asyncio.to_thread(
        run_lifecycle_primary,
        tools=custom_tools, ctx=ctx,
        paper_text=context_dict.get("paper_text"),
        rubric_spec=context_dict.get("rubric_spec"),
        emit=emit,
        target_score=ctx.latest_rubric_target,
        max_repair_iterations=_drive_max_repair(),
        max_improve_iterations=_drive_max_improve(),
    )
    result_obj = _synth_result_from_summary(summary, ctx)
    run_failed = result_obj is None
else:
    result_obj = await asyncio.to_thread(_run_completion_on_worker)
    # ... existing root-usage drain block (unchanged) ...
```
The existing `except`/`finally` + the `_finalize(result_obj=..., run_failed=..., ...)` call
are UNCHANGED. (The root-usage drain block stays inside the `else`.)

`_synth_result_from_summary(summary, ctx) -> RLMChatCompletion | None`:
- If `summary.get("rubric_score") is None` → return None (honest: no score; `_finalize`
  ships the `failed` shell + regrade-from-disk).
- Else build a minimal report dict:
  `verdict = "reproduced" if score >= (ctx.latest_rubric_target or 1.0) else "partial"`;
  `reproduction_summary = "Harness-driven lifecycle (primary mode): drove
  understand→implement→run→verify" + (f", climbed {n} improvement(s)" if improved)`.
  `baseline_metrics = {}` (the metric-projection fills it from disk).
- Construct and return the rlms `RLMChatCompletion` with
  `response=json.dumps(report_dict)`, `usage_summary={}`, `metadata={}`. **If the rlms
  constructor is awkward, return None instead** (fallback: ships the real regrade score with a
  conservative verdict) and note it in the implementation report. Build-final-report + the
  honesty gates do the rest.

## Tests (TDD)

- `tests/rlm/test_lifecycle_driver.py`:
  - `drive_lifecycle_chain` now exposes `verify_result` in the summary (assert present;
    byte-identical otherwise — existing tests still pass).
  - `run_lifecycle_primary`: backbone-only (`max_improve=0`) returns the baseline score;
    improvement loop fires when `score < target`, calls propose_improvements + implement +
    drive(need_experiment), stops on target reached / cap / empty hypotheses / low wall-clock;
    never regresses the score; fail-soft when propose/implement raise (returns baseline score).
    Use fake tools (mirror existing fixtures) recording call order + args.
- `tests/rlm/test_run_lifecycle_drive.py` (or a new `test_run_lifecycle_primary.py`):
  - `_lifecycle_primary_enabled()` flag parse; `_drive_max_improve()` default 2 / `0` honored.
  - `_synth_result_from_summary`: None when rubric_score None; reproduced/partial verdict by
    score-vs-target; constructs a parseable `response`.
  - **Byte-identical-off**: with the flag unset, the completion branch takes the `else` and the
    existing path is unchanged (assert run_lifecycle_primary is NOT called; mock/spy).

## Acceptance
```
.venv/bin/python -m pytest tests/rlm/ -q
.venv/bin/python -m pytest tests/ -q            # full suite green
uvx ruff@0.15.16 check backend/agents/rlm/lifecycle_driver.py backend/agents/rlm/run.py
```
All green; ruff clean; byte-identical when `OPENRESEARCH_LIFECYCLE_PRIMARY` unset.

## GCP validation (operator-run, after merge)
`ROOT=claude-oauth SMOKE=0 PROV=spot OPENRESEARCH_LIFECYCLE_PRIMARY=1 PROJECT_ID=sdar_primary_v1
scripts/sdar_gcp_e2e.sh run`. Success = a real rubric score with NO degenerate-loop churn
(the root loop never runs; the harness drives implement→run→repair→verify→[climb]→finalize).

## Increment 3 (future, after A/B)
Flip `OPENRESEARCH_LIFECYCLE_PRIMARY` default ON for keyless roots; remove the now-redundant
reactive machinery (the once-only reactive driver branch, autodrive, the forced-iteration
lifecycle-backbone refusals) — keeping the genuinely-orthogonal pieces (evidence gates,
regrade, best-of-run floor, finalize). Update CLAUDE.md + system_overview.md.
