# Pipeline validation on `mechanistic-understanding` — design (2026-05-29)

**Goal:** prove that ReproLab reproduces an arbitrary ML paper end-to-end (ingest → understand → plan → implement → run_experiment → verify_against_rubric → final_report) on a paper that is significantly less complex than SDAR but still exercises the full pipeline against a real LM. Result: ONE definitive data point that the system reproduces real ML papers, decoupling "SDAR-specific complexity" from "general harness brittleness."

## Why this paper, why now

Across 7 SDAR attempts the pipeline has reached `run_experiment` cleanly only 3 times — each death from a different root cause (asyncio deadlock BUG-NEW-023, CPU-only surrogate BUG-NEW-030/033, Dockerfile prose BUG-NEW-042). Each fix lands one bug but a new one surfaces, because SDAR's surface area (multi-environment + multi-model + GPU-required + strict algorithmic invariants checked by the rubric) is ~10× a typical paper's. We have *no* baseline of "this paper reproduced successfully" against which to compare SDAR's failure modes. Without that baseline we cannot distinguish "SDAR is hard" from "the harness is broken."

`mechanistic-understanding` (DPO toxicity reduction on GPT2-medium / Llama2-7b — 96 rubric leaves, mostly Code Development + Code Execution) is the right validator because:
1. Single environment, single model class — eliminates SDAR's multi-env complexity
2. Supervised fine-tune (not RL) — eliminates the entire class of RL-loop bugs
3. Ships a hand-built rubric in the bundle — removes auto-rubric variance as a confounder
4. Real LM (GPT2-medium = 350MB, Llama2-7b = 13GB) — matches the *class* of paper SDAR represents
5. Already has prior signal: CLAUDE.md notes the `transformers`/`datasets` dep-detection bug surfaced on this paper
6. The bundle bypasses ingest entirely (`_cmd_reproduce_rlm_paperbench`, `backend/cli.py:1068`), so we're testing pure orchestration, not the parser

## Run command

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  nohup .venv/bin/python -m backend.cli reproduce mechanistic-understanding \
    --mode rlm --model claude-oauth --sandbox runpod --provider anthropic \
    --max-wall-clock 7200 --max-pod-seconds 7200 --max-usd 15 \
    --preflight-sanity \
    > /tmp/mech-understanding-attempt1.log 2>&1 &
```

Notes:
- Full paper scope (no `REPROLAB_BASELINE_EXTRA_GUIDANCE` restriction) — the bundle's rubric covers both GPT2-medium and Llama2-7b
- `--sandbox runpod` is the production target; we accept the capacity-exhaustion risk as a real-world signal
- `--max-usd 15` ceiling — Llama2-7b on RTX 4090 community tier (~$0.34/hr) for 2h is ~$0.68; rest of the budget covers retries
- `--preflight-sanity` is the new flag from P3 below
- `claude-oauth` for both root + sub-agents — zero Anthropic API balance burn (Claude Code subscription only)

## Prevention guards — land BEFORE the run

Three guards motivated by past failure modes. All three land in one PR before attempt 1.

### P1 — `commands.json` manifest existence check

**Symptom it catches:** the 2-3s no-op retry pattern from BUG-NEW-042 (sub-agent claimed success but wrote nothing of substance); any case where `commands.json` references a file the sub-agent did not actually write.

**Location:** `backend/agents/rlm/primitives.py::_harvest_baseline_artifacts` (around line 145).

**Logic:** after validating `commands.json` parses as a non-empty list, iterate every command entry. For each arg that ends in a runnable suffix (`.py`, `.sh`, `.bash`, `.ps1`) or a config suffix (`.json`, `.yaml`, `.yml`, `.txt`), resolve relative to `code_dir` and check it exists. Skip flag-like args (start with `-`) and treat env-style args (`KEY=VAL`) as not-paths.

**On failure:** return `_baseline_error_envelope(error_code="commands_missing_file", repairable=True, missing_files=[...])`. Add `"commands_missing_file"` to `_RUN_EXPERIMENT_REPAIRABLE_FAILURES`.

**Tests:** valid (every file present); invalid (commands references missing file); valid (no file-like args at all — pure shell echo); invalid (file referenced is a directory, not a regular file).

### P2 — bundle-mode Dockerfile lock

**Symptom it catches:** stronger form of BUG-NEW-042 — eliminates the prose-stomp class entirely for bundle runs. When we run against a PaperBench bundle, the bundle's environment is canonical; there is no legitimate reason for `implement_baseline` to ever modify `project_dir/Dockerfile`.

**Location:**
- `backend/agents/rlm/context.py`: add a `bundle_mode: bool = False` field to `RunContext`
- `backend/cli.py::_cmd_reproduce_rlm_paperbench`: set `bundle_mode=True` when constructing the `RunContext`
- `backend/agents/rlm/primitives.py::implement_baseline`: when `ctx.bundle_mode is True`, after the sub-agent returns, ALWAYS restore the pre-snapshot of `project_dir/Dockerfile` from the snapshot already captured by BUG-NEW-042 logic. Emit a `dashboard_event` (`code="bundle_dockerfile_lock"`) if a write was detected and reverted.
- `backend/agents/prompts/baseline_implementation.py`: when bundle mode, append a system-prompt line: "DO NOT WRITE Dockerfile. The environment is fixed by the PaperBench bundle and managed by the orchestrator. Any Write to Dockerfile will be reverted."

**Tests:** bundle mode → Dockerfile preserved regardless of post-write content (valid OR invalid); non-bundle mode → existing BUG-NEW-042 shape-guard behavior (auto-restore on invalid only).

### P3 — pre-run sanity dispatch (`--preflight-sanity` flag)

**Symptom it catches:** stale Docker daemon, RunPod auth failure, expired SSH key, network partition — all errors that today would corrupt 20+ min of a real run instead of failing fast.

**Location:**
- `backend/cli.py::_build_parser`: add `--preflight-sanity` flag (default off, opt-in)
- `backend/cli.py::cmd_reproduce`: when flag is set, BEFORE the mode-dispatch block (rdr / rlm-paperbench / main), run the sanity primitive sequence inline (effectively `_cmd_reproduce_sanity` factored as a callable). Pipe its result to stderr.
- Sanity = the existing CPU echo+touch primitive that writes a static `metrics.json` — already implemented as `_cmd_reproduce_sanity`. Factor the body into `_run_sanity_check(runs_root, sandbox_mode) -> tuple[bool, str]` and call it from both places.
- On sanity failure: print `[preflight-sanity] FAILED: <reason>` and `return 4` (new exit code for preflight failure, distinct from the existing 3 = budget). Abort the real run.
- Default: the flag is OFF. Pre-flight sanity runs only when the operator explicitly passes `--preflight-sanity`. Tests must not depend on it being on by default.

**Tests:** flag is wired into argparse; sanity success → real run proceeds; sanity failure → real run aborts with exit code 4 and prints the reason; flag absent → no behavior change (backward compatible).

## Same-failure-twice rule

The 5-attempt cap from the SDAR retry-monitor runbook is replaced for this sprint by a tighter rule:

> **If the same `failure_class` (or same pre-run failure marker) fires in two consecutive attempts, STOP and add a guard before attempt 3.**

Rationale: SDAR proved that grinding 7 attempts on a moving target of distinct root causes burns budget without converging. The opposite — same root cause twice — is the signal we should fix the harness, not retry. Encoded as a hard rule rather than a soft heuristic.

Operationally: after each attempt failure, the runbook entry's first line records the `failure_class`. Before launching attempt N+1, scan attempts N and N-1; if both share the same failure_class, do not relaunch — land a guard first.

Exception: `runpod_capacity` and other `_RUN_EXPERIMENT_RETRYABLE_FAILURES` (transient network/capacity issues) do not count toward the rule — those are not harness bugs.

## Success criteria

The run is successful iff ALL of:
1. `runs/<id>/final_report.json` is written and contains `rubric_score > 0.3` (above noise; perfect score is unrealistic on first try)
2. Run completes within 90 min wall-clock
3. `verify_against_rubric` was called at least 2 times (the forced-iteration policy from Lane H fired at least once)
4. No unhandled `dockerfile_shape_guard` warnings (if present, all must have `restored: true`)
5. Total cost < $5 USD (sanity ceiling; with `claude-oauth` should be near zero)
6. No `commands_missing_file` failures (P1 should hold)
7. No `bundle_dockerfile_lock` events fired (P2's silent preservation worked — if it fires, log a warning but do not fail the run)

## Failure handling — the doc loop

Every attempt failure becomes a `BUG-NEW-NNN` entry in a new runbook (`docs/runbooks/2026-05-29-mech-understanding-validation.md`) plus a memory file when the bug is load-bearing across sessions. Each entry contains:
1. Symptom (verbatim event or quoted root-model text)
2. Failure class (from `_persist_experiment_result` or the runtime error)
3. Root cause once known
4. Fix or "deferred" with reason
5. Validation path for the fix

This is the same doc-loop convention from `docs/runbooks/2026-05-29-monitoring-loops.md` — established convention, no change.

## What we do AFTER success

ONE clean end-to-end success on `mechanistic-understanding` proves the system reproduces real ML papers but does not prove it generalizes. Next steps in priority order:

1. **Document the success.** New runbook entry. Update `CLAUDE.md` "Baseline test paper" section to add `mechanistic-understanding` as the smoke-test baseline (SDAR remains the stretch target).
2. **Run SNSE next** (`sequential-neural-score-estimation`). A second clean success on a different paper class (pure math, no LM) makes the generality claim much stronger and validates that P1-P3 generalize.
3. **Re-attempt SDAR** with confidence that any remaining bugs are SDAR-specific complexity, not general harness brittleness. Apply learnings from mech-understanding + SNSE runs as additional paper hints or pre-flight guards.

## Implementation order

Strict sequential — each step gated by the previous landing and tests passing:

1. **P1 — `commands.json` manifest check.** ~50 LOC in `primitives.py`, 4 tests. Land + commit.
2. **P2 — bundle-mode Dockerfile lock.** ~80 LOC across `context.py` + `cli.py` + `primitives.py` + `prompts/baseline_implementation.py`, 4 tests. Land + commit.
3. **P3 — `--preflight-sanity` flag.** ~80 LOC across `cli.py` (flag wiring + sanity factoring), 3 tests. Land + commit.
4. **Backend up.** `./start.sh` (or the uvicorn invocation). Verify `/health` responds.
5. **Sprint kickoff.** Launch attempt 1 via the run command above. Arm Loops 1 / 2 / 3 / 4 from `docs/runbooks/2026-05-29-monitoring-loops.md`. Open a fresh `docs/runbooks/2026-05-29-mech-understanding-validation.md` for the sprint log.

Total pre-run code work: ~210 LOC + 11 tests. Estimate: 60–90 min of focused implementation before attempt 1.

## Architecture impact

No new architectural concepts. The three guards extend existing patterns:
- P1 extends the existing `_harvest_baseline_artifacts` artifact validation
- P2 extends the existing BUG-NEW-042 shape guard with a stricter mode flag
- P3 extends the existing `_cmd_reproduce_sanity` subcommand as an opt-in pre-flight

`RunContext.bundle_mode` is the only new public field. It's a boolean — easy to thread through, no compatibility surface.

## Out of scope (intentional)

- Run-result classifier telemetry / structured postmortems (deferred — bigger design, separate spec)
- Sandbox-tier escalation (auto fallback runpod → docker for small models) (deferred)
- Auto-paper-hint generation for new papers (deferred — relevant for the post-success generalization track)
- Frontend changes (none needed — the lab UI already renders all the new events)
- Migration to a different RLM library, model, or sandbox backend (none planned)
