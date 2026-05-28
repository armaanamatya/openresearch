# 2026-05-27 RLM reproduction stability plan

Status: fix plan from the `2512.24601` sanity run and prior trace-mining audit.

This document summarizes what is blocking reliable reproduction, which issues are
known to exist on `main`, what to fix first, and what to defer. It intentionally
does not include GEPA implementation details except where they affect the
reproduction loop.

## Executive Summary

The next bottleneck is not prompt optimization. It is the reproduction plumbing
before GPU execution.

The sanity run `prj_ac41983c934a3432` was launched on `feat/gepa-phase2-5` with
`claude-oauth` and `sandbox=runpod`. It completed in ~22 minutes with `$0.00`
RunPod spend because no pod was ever created. It reached `build_environment`, but
then failed at `implement_baseline` three times and produced rubric `0.0`.

This confirms the same funnel problem seen in the eight-run trace-mining corpus:
runs mostly die before useful `run_experiment` execution. Running larger GEPA,
HALO, or SkillOpt experiments before fixing this would mostly optimize dead
paths.

## Evidence

### Sanity Run

| Field | Value |
|---|---|
| Project | `prj_ac41983c934a3432` |
| Paper | `2512.24601` |
| Started | `2026-05-27T21:23:08Z` |
| Completed | `2026-05-27T21:45:24Z` |
| Final event | `run_complete.status=failed` |
| Demo status | `status=completed`, `run_state.kind=failed` |
| Rubric | `0.0 / 0.6`, all leaves `degraded_no_metrics` |
| RunPod pods | `0` |
| RunPod spend | `$0.00` |

Primitive outcome:

| Primitive | Result |
|---|---|
| `understand_section` | 3 ok |
| `extract_hyperparameters` | ok |
| `resolve_gpu_requirements` | ok, A6000 selected |
| `detect_environment` | ok |
| `build_environment` | ok |
| `plan_reproduction` | ok |
| `implement_baseline` | 3 errors |
| `run_experiment` | 1 error before pod creation |
| `verify_against_rubric` | ok, score 0 |
| `propose_improvements` | ok, but emitted an empty candidate |

### Corpus Audit

The programmatic trace miner over eight historical `2512.24601` runs showed:

- `3/8` runs died with OOM-like `exit code -9`.
- `5/8` runs contained repeated `claude_agent_sdk` async-generator cleanup
  errors.
- `7/8` runs never reached a meaningful `run_experiment`.
- The only two completed runs scored `0.00` and `0.15`, both below target.

## What Exists On Main

Checked against `origin/main` after the sanity run.

| Issue | Main status | Evidence |
|---|---|---|
| `implement_baseline` pre-emit stall returns error dict | Present | `backend/agents/rlm/primitives.py` has `sdk_pre_emit_stall` handling. |
| `run_experiment` receives error dict instead of path | Guard present, root still vulnerable | Main has the type guard and hint, but the root can still waste a primitive by chaining the bad value. |
| `worker_report_failed.error=null` | Present | `backend/agents/worker_reports.py` emits `report.get("error")`; primitive result errors are not copied into the report. |
| `demo_status.status=completed` while verdict failed | Present by design | `backend/agents/rlm/run.py` separates process completion from reproduction verdict, but UI/operator language conflates them. |
| CLI runs invisible to `/runs` / Recent UI | Present | `/runs` delegates to registry-backed `service.list_runs`; CLI-created `runs/<id>` dirs are not registered. |
| Derived `run_state` false-stuck behavior | Branch-only | `run_state.py` is not on `main` yet. |
| HMR websocket failures | Environment-only | Observed on a stale worktree dev server; not proven as a repo bug. |

## Priority Fix Plan

### P0 — Make `implement_baseline` return a reliable result contract

Problem:

`implement_baseline` can return three incompatible shapes:

- a string path on success,
- an error dict on pre-emit stall,
- an exception / malformed SDK result even when files were written to `code/`.

The root model then treats the error dict like a path and calls
`run_experiment`, causing a second-order failure.

Fix:

1. Normalize `implement_baseline` to a typed envelope:

```json
{
  "ok": true,
  "code_path": "runs/<id>/code",
  "files": ["train.py", "commands.json", "requirements.txt"]
}
```

or

```json
{
  "ok": false,
  "error_code": "sdk_pre_emit_stall",
  "error": "...",
  "repairable": true
}
```

2. Add a compatibility shim if the root still expects a string, but make the
system prompt and primitive wrapper prefer the envelope.
3. If `commands.json` and minimal runnable artifacts exist after SDK failure,
harvest the directory and return `ok=true`.
4. If artifacts are incomplete, return `ok=false` with a precise missing-file
list.

Acceptance tests:

- Simulate pre-emit stall: root must not call `run_experiment`.
- Simulate post-write SDK failure with valid `commands.json`: primitive must
  return `ok=true` with `code_path`.
- Simulate post-write SDK failure with missing `commands.json`: primitive must
  return `ok=false`, no `run_experiment`.

### P0 — Block bad `run_experiment` calls at the orchestration boundary

Problem:

`run_experiment` has a type guard, but the root still spends a primitive call on
an obviously invalid value.

Fix:

1. Add a wrapper-level precondition: if the previous primitive returned
   `ok=false` or an error dict, short-circuit to `propose_improvements` or a
   repair heartbeat.
2. Add a root prompt invariant:
   “Never pass an `implement_baseline` result to `run_experiment` unless it is a
   string path or an envelope with `ok=true` and `code_path`.”
3. Emit `iteration_boundary_recommended` only once per failure, not as noisy
   repeated state.

Acceptance tests:

- Unit test root/tool wrapper flow where `implement_baseline` returns
  `{success: false}`.
- Assert no `experiment_completed` event is emitted for invalid `code_path`.

### P0 — Fix or isolate the Claude SDK cleanup failure path

Problem:

`RuntimeError: aclose(): asynchronous generator is already running` is noisy in
the root path and reproduction-blocking in sub-agent paths. The 240-second
pre-emit watchdog prevents infinite hangs but still wastes minutes and often
returns no code.

Fix options:

1. Upgrade/pin `claude-agent-sdk` after verifying the async-generator cleanup
   behavior.
2. Wrap SDK calls in a subprocess boundary for `implement_baseline`; if the
   subprocess exits badly but artifacts exist, harvest artifacts.
3. Add a hard `aclose()` timeout around SDK cleanup.
4. Add a local fail-fast retry: if pre-emit stall happens once, retry with a
   deterministic baseline template rather than another SDK sub-agent call.

Acceptance tests:

- Forced SDK cleanup exception does not hang caller.
- Forced cleanup exception after artifact write still returns usable code path.
- Forced cleanup exception before artifact write returns a repairable error in
  under 30 seconds in test mode.

### P1 — Improve worker-report observability

Problem:

`worker_report_failed` events have `error=null` even when the primitive failed
with a detailed error dict.

Fix:

1. When a primitive returns `ok=false`, copy the primitive error into the worker
   report.
2. Include `failure_class`, `contract_violations`, and `repairable` where
   available.
3. Add a `source` field: `exception`, `primitive_result`, `contract_guard`,
   `timeout`, or `sdk_stall`.

Acceptance tests:

- Failed `implement_baseline` event contains non-empty `error`.
- Failed `run_experiment` contract guard surfaces `failure_class`.

### P1 — Separate process status from reproduction verdict

Problem:

The final sanity run had:

- `run_complete.status=failed`
- `demo_status.status=completed`
- `demo_status.run_state.kind=failed`

That is technically explainable, but confusing.

Fix:

Use explicit axes in `demo_status.json`:

```json
{
  "process_status": "completed",
  "verdict": "failed",
  "run_state": {"kind": "failed"}
}
```

UI copy should say “process completed; reproduction failed,” not just
“completed.”

Acceptance tests:

- Failed reproduction with final report renders failed verdict.
- Successful process with partial result renders partial, not green completed.

### P1 — Register CLI-created runs with the UI or scan filesystem

Problem:

Runs launched from the CLI write `runs/<id>`, but `/runs` and the Recent list
only show registry-backed runs created through the API. The lab UI cannot open a
CLI run by `projectId`.

Fix options:

1. Add `/runs?include_unregistered=true` that merges registry and filesystem
   scan.
2. Register CLI runs through the same service used by `POST /runs`.
3. Make `/runs/{project_id}` fall back to filesystem state if registry lookup
   misses.

Acceptance tests:

- CLI-created run appears in Recent.
- `/lab?projectId=<cli-run>` opens the run detail.
- Chat route returns a clear “not steerable because not API-registered” message
  if steering is unavailable.

### P1 — Add a real cheap/sanity reproduction mode

Problem:

The current “sanity” run still performs a full RLM decomposition and calls the
heavy `implement_baseline` sub-agent. This consumes Claude subscription usage
before it tests RunPod.

Fix:

Add `--sanity` or `--mode smoke` semantics:

- cap `sub_rlm` calls to 2-3,
- skip full `implement_baseline`,
- write a deterministic tiny baseline template,
- run one no-op or tiny Python command on the selected sandbox,
- verify artifact plumbing, metrics schema, events, and cleanup.

This mode answers: “Can this machine/account launch and observe a RunPod job?”
It should not try to reproduce a paper.

Acceptance tests:

- `--sanity --sandbox runpod --max-usd 1` reaches pod create or fails with a
  precise pre-pod reason.
- No more than 3 LLM/sub-RLM calls.
- No heavy code-writing sub-agent invocation.

### P2 — Memory/OOM instrumentation

Problem:

`3/8` historical runs died with `exit code -9`.

Fix:

1. Log RSS at primitive boundaries.
2. Log container memory limits and host memory.
3. Add Docker memory floor to docs and preflight.
4. For large PDFs, avoid loading raw PDF, parsed text, and full context into
   every sub-agent prompt.

Acceptance tests:

- Every run emits memory snapshots.
- OOM-like exit gets classified as `oom_killed`, not `failed_other`.

### P2 — Candidate schema validation

Problem:

`propose_improvements` emitted an empty candidate:

```json
{"id": "", "description": "", "category": "", "title": "candidate", "reasoning": ""}
```

Fix:

Reject or repair candidates missing `id`, `description`, or `reasoning`.

Acceptance tests:

- Empty candidate is refused and triggers a repair prompt.
- UI never displays a blank candidate card.

## What Not To Do Yet

Do not spend the `$7` RunPod balance on another full paper reproduction until:

1. `implement_baseline` returns a reliable envelope.
2. Bad `run_experiment` chaining is blocked.
3. CLI runs are visible in the UI or the backend is launched with a correct
   `REPROLAB_RUNS_ROOT`.
4. A cheap/sanity mode can test RunPod without full RLM/code-writing.

Do not run large GEPA/HALO/SkillOpt comparisons yet. The current bottleneck is
before the prompt-optimization surface; most runs never get to the improvement
loop.

## Minimal Next PR Stack

1. **PR A: primitive contract hardening**
   - normalize `implement_baseline` return shape,
   - block `run_experiment` on invalid path,
   - populate worker report errors.

2. **PR B: status and UI visibility**
   - split `process_status` and `verdict`,
   - make CLI runs visible/openable from `/lab?projectId=...`.

3. **PR C: cheap sanity mode**
   - deterministic baseline template,
   - bounded sub-RLM count,
   - no full paper reproduction,
   - optional RunPod pod smoke under `$1`.

4. **PR D: memory/OOM observability**
   - RSS snapshots,
   - OOM classification,
   - Docker memory floor docs.

## Verification Command Set

Run after PR A/B/C:

```bash
python -m pytest \
  tests/test_worker_reports.py \
  tests/rlm/test_run_state.py \
  tests/rlm/test_sse_bridge.py \
  tests/test_mine_traces.py
```

Then run a zero/near-zero-cost smoke:

```bash
python -m backend.cli reproduce 2512.24601 \
  --model claude-oauth \
  --sandbox runpod \
  --sanity \
  --max-run-gpu-usd 1.0 \
  --max-pod-seconds 900 \
  --max-wall-clock 1800
```

Expected result for the smoke is not a good rubric score. Expected result is:

- bounded Claude usage,
- clear pod/pre-pod state,
- no `implement_baseline` sub-agent stall,
- no invalid `run_experiment` call,
- UI can open the run,
- worker-report errors are populated if anything fails.

## 2026-05-27 Implementation Verification Notes

Applied P0/P1 hardening in the working tree:

- `claude-agent-sdk` dependency floor was raised from `>=0.1.80` to
  `>=0.2.87`; the local venv had `0.2.82`, while PyPI lists `0.2.87` as a
  newer release. Artifact harvesting remains as the local fallback if SDK
  cleanup still fails.
- `implement_baseline` now returns a normalized envelope:
  `{"ok": true, "code_path": ..., "files": [...]}` or
  `{"ok": false, "error_code": ..., "error": ..., "repairable": ...}`.
- SDK failure paths harvest usable `code/` artifacts when `commands.json` and a
  runnable source file exist; incomplete artifacts return precise
  `missing_files`.
- `run_experiment` accepts a valid `ok=true` envelope, but contract-guards
  invalid code paths without emitting `experiment_completed` or spawning a pod.
- The primitive wrapper blocks `run_experiment` before calling the primitive
  body when the root passes an error dict.
- `worker_report_failed` now includes primitive error details:
  `failure_class`, `contract_violations`, `repairable`, and `source`.
- `demo_status.json` now carries explicit `process_status` and `verdict` axes.
- `/runs?include_unregistered=true` is explicit; filesystem-backed CLI runs
  with `demo_status.json` appear in `/runs` and open via `/runs/{project_id}`.
- `--sanity` now runs a deterministic tiny sandbox smoke and skips full RLM and
  the code-writing sub-agent.
- Final-report calibration recompute is now opt-in via
  `REPROLAB_UPDATE_CALIBRATION=true`, preventing smoke/pre-flight-only runs from
  overwriting `data/calibration.json`.
- `propose_improvements` now rejects blank candidate records, and the wrapper
  also filters blank candidate cards before emitting UI events.
- Primitive wrappers now emit `primitive_resource` start/end events with
  best-effort process RSS.
- `run_experiment` sandbox execution now emits `sandbox_resource_limits` and
  returns `resource_limits`, `exit_code`, and `cause_kind` for persistence and
  diagnosis.
- OOM-like exits and runtime causes now classify as `oom_killed`; exit `-9`
  and `137` still trigger the GPU escalation detector.
- Docker/local-container smoke runs should keep the memory floor at the
  `SandboxConfig.memory_limit` default (`4g`) or explicitly pass
  `--sandbox-memory` for papers with larger data/model footprints. Treat
  `oom_killed` as a sizing failure first, not a generic code failure.

Verification:

```bash
python -m pytest \
  tests/test_worker_reports.py \
  tests/rlm/test_sse_bridge.py \
  tests/rlm/test_implement_baseline.py \
  tests/rlm/test_run_experiment.py \
  tests/rlm/test_binding.py \
  tests/rlm/test_primitive_cache_validators.py \
  tests/test_live_runs_listing.py \
  tests/test_cli_sanity_mode.py \
  tests/test_cli_budget_flags.py \
  tests/rlm/test_run.py
```

Result after the calibration opt-in test was added: `200 passed`.

Additional P2 hardening checks:

```bash
python -m pytest \
  tests/rlm/test_failure_classifier.py \
  tests/rlm/test_cuda_oom_detection.py \
  tests/rlm/test_sandbox_retry.py \
  tests/rlm/test_binding.py
```

Result: `63 passed`.

Broad regression after P2 candidate/OOM/resource instrumentation:

```bash
python -m pytest \
  tests/test_worker_reports.py \
  tests/rlm/test_sse_bridge.py \
  tests/rlm/test_implement_baseline.py \
  tests/rlm/test_run_experiment.py \
  tests/rlm/test_binding.py \
  tests/rlm/test_primitive_cache_validators.py \
  tests/test_live_runs_listing.py \
  tests/test_cli_sanity_mode.py \
  tests/test_cli_budget_flags.py \
  tests/rlm/test_run.py \
  tests/rlm/test_report.py \
  tests/rlm/test_propose_improvements.py \
  tests/rlm/test_failure_classifier.py \
  tests/rlm/test_cuda_oom_detection.py \
  tests/rlm/test_sandbox_retry.py
```

Result: `245 passed`.

The original verification command could not run exactly because these files are
not present on this branch:

- `tests/rlm/test_run_state.py`
- `tests/test_mine_traces.py`

Cheap RunPod sanity smoke:

```bash
python -m backend.cli reproduce 2512.24601 \
  --model claude-oauth \
  --sandbox runpod \
  --sanity \
  --max-run-gpu-usd 1.0 \
  --max-pod-seconds 900 \
  --max-wall-clock 1800
```

First attempt failed before any sandbox or pod work because the new sanity
helper incorrectly called `project_id_for()` with a raw string. Fixed by using a
local hash-derived `prj_...` id.

Second attempt succeeded:

| Field | Value |
|---|---|
| Project | `prj_4c0368d9d5194b41` |
| Status | `completed` |
| Process status | `completed` |
| Verdict | `reproduced` for the smoke template |
| Metrics | `{"sanity_ok": 1.0}` |
| Primitive provider | `sanity-template` |

Observed issue: the sanity path initially did not create `cost_ledger.jsonl`
because it makes no LLM calls. Fixed by touching an empty ledger file at run
start so monitors have a stable file to tail.

Pod cleanup check:

- `runpodctl` is not installed on this machine.
- Built-in dry-run sweeper command
  `python -m backend.services.runtime.pod_sweeper --dry-run --max-age-seconds 0`
  reported `0/0 swept`, so no active RunPod pods were visible to this account.

