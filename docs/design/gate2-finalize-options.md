# Gate-2-fail Final-Report Options

**Context:** When Gate 2 returns `partial_reproduction`, the orchestrator halts at `GATE_2_PASSED` (stage 8/14). `final_report.{json,md}` are only written at `RESEARCH_MAP_GENERATED` (stage 9). Result: `final_benchmark_report.md` shows the seed template, `demo_status.json.benchmark.verdict` stays `"pending_pipeline_result"`, and `finalize_benchmark()` in `live_runs.py:1250` never runs because it depends on `final_report.json` existing.

The rubric verdict (e.g. `overall=0.270`) is already on `state.baseline_verification` — we just don't surface it.

## What the codebase already provides

- `backend/agents/report_generator.py:1023` `write_final_report(report, dir)` — writes both `.json` and `.md`. Reusable.
- `backend/agents/report_generator.py` `generate_final_report(...)` — composes the FinalReport from state. Accepts partial state.
- `backend/services/events/live_runs.py:1250` `finalize_benchmark()` — reads `final_report.json`, updates `demo_status.json.benchmark`.
- `state.baseline_verification` (RubricVerification) — populated at line 1822, persisted in checkpoint at 254.

## Option A — minimal: write a partial final report on Gate 2 fail (recommended)

**One-line change in spirit, ~15 LOC in practice.** At `backend/agents/orchestrator.py:~1823` (right after `advance_stage(GATE_2_PASSED)`), check if the run is about to halt (no improvement loop will run). If so:

```python
# pseudo
if not _will_attempt_improvements(state):
    partial = generate_final_report(
        state,
        baseline_verification=state.baseline_verification,
        improved_verification=None,
        partial=True,
    )
    write_final_report(partial, self._project_dir)
    state.advance_stage(PipelineStage.COMPLETE, self.runs_root)
    return state
```

Optionally bump `generate_final_report` to accept a `partial: bool` so the markdown header reads "Partial Reproduction" instead of "Reproduction Report" — cosmetic, ~3 LOC.

**Why this is right:**
- Zero new files. Uses existing writer.
- `finalize_benchmark()` starts working — the UI's benchmark panel finally updates.
- The halt semantics don't change: pipeline still stops at Gate 2 fail, just now with an honest artifact.
- Resume-safe: the new advance_stage to COMPLETE means a re-run won't re-trigger Gate 2.

**Blast radius:** one branch in orchestrator + one optional kwarg on `generate_final_report`. ~30 minutes of work.

## Option B — medium: actually run the improvement loop on Gate 2 fail

Trace why the improvement orchestrator wasn't entered for `prj_b9306d43600e3d5c` (the run we're holding as evidence). `rubric_max_improvement_iterations=2` should have triggered at least one improvement-selection + Gate 3 cycle when `baseline_verification.overall=0.27 < target=0.70`.

Hypothesis: there's a missing `if state.baseline_verification.overall < settings.rubric_target_score: <continue to improvements>` decision somewhere. Either the condition isn't being evaluated, or `_will_attempt_improvements()` returns False for a reason that's wrong.

**Why this is appealing:**
- The whole point of the rubric verifier is to **drive improvement**, not just to halt. If the loop never fires, that feature is decorative.

**Why this is risky:**
- One improvement iteration on a docker baseline that did a smoke test is going to be 5–30 more minutes of LLM + container work. The user just spent ~100 minutes; another loop could be another hour.
- The improvement orchestrator can fail in new ways. Without Option A landed first, a failure here STILL leaves no final report.

**Recommendation:** ship Option A first; investigate Option B as a follow-up because regardless of whether improvements run, Option A gives an honest terminal artifact.

## Option C — bigger: unified `finalize_run()` for every terminal state

Refactor so every orchestrator exit path (success, gate fail, fail-soft, exception) routes through one `finalize_run(state, terminal_reason)` that:
1. Writes `final_report.{json,md}` (partial if applicable).
2. Persists final checkpoint.
3. Triggers `finalize_benchmark()` so the UI's benchmark panel always reflects the real terminal state.

Treats final-report writing as a contract of "the run ended" rather than "stage 9 completed."

**Why this is the right long-term move:**
- Today, exception-during-stage-N leaves the project in a half-finished state with no usable artifact. Same UX problem as the Gate 2 case.
- Eliminates the divergence between `pipeline.py:run_offline()` (line 514 writes) and `orchestrator.py` (line 2263 writes) — they currently duplicate the same code.

**Cost:** ~80-150 LOC, touches every `state.advance_stage(...)` call site plus the run loop, plus the resume logic at line 2307+. Needs careful test coverage on resume behavior because `advance_stage` is where checkpoints save.

## Recommendation

1. **Ship Option A now** as a small follow-up PR to the merged Tier 1+2 PR. Unblocks the user's "nothing changed in the final report" confusion and starts giving the UI honest data.
2. **Add a small task to investigate Option B** — figure out WHY `prj_b9306d43600e3d5c` didn't enter the improvement loop. If it's a config gating issue (`rubric_target_score`, `rubric_max_improvement_iterations`), fix is one line. If it's a deeper code path issue, that's its own design discussion.
3. **Defer Option C** until the team has explicit appetite for a refactor sprint. The unified-terminal-state design is correct but Option A bridges the gap with much less risk.

## What I'd actually patch (Option A specifics)

| File | Change |
|---|---|
| `backend/agents/report_generator.py` | Add `partial: bool = False` kwarg to `generate_final_report`; bias the markdown header + `reproduction_status` field accordingly |
| `backend/agents/orchestrator.py` | New private `_finalize_partial(state)` helper called from the Gate-2-fail halt path. Writes the partial report, advances to COMPLETE, returns. |
| `tests/` | One test: pass a state with low baseline_verification.overall, assert `final_report.json` exists and has `reproduction_status="partial_reproduction"` after run. |

Total: ~30 LOC of changes + ~25 LOC of test. Single PR, mergeable in an hour.

---

## Verification plan after Option A lands

```powershell
# Run a pipeline through to Gate 2 fail (or use the existing prj_b9306d43600e3d5c by resuming).
# Then check:
Test-Path logs\<TS>\prj_<id>\final_report.md       # should be True
(Get-Content logs\<TS>\prj_<id>\final_report.json | ConvertFrom-Json).reproduction_status
# should be "partial_reproduction"

(Get-Content logs\<TS>\prj_<id>\demo_status.json | ConvertFrom-Json).benchmark.verdict
# should be "partial_reproduction" (not "pending_pipeline_result")
```

Both writes flow from the new helper; if either is wrong the helper is wrong.
