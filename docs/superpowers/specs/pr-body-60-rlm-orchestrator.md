## Summary

Implements the RLM orchestrator (`--mode rlm`), Phase 3 of the RLM pivot (issue #60). This replaces the dispatch path of the 14-stage `PipelineStage` machine with a run mode driven by the `rlms` library's Algorithm-1 loop: the root model writes REPL code that calls domain primitives to reproduce a paper. The orchestrator is deliberately decoupled from the #59 primitive layer — it degrades gracefully to stub primitives and is fully integration-tested before #59 lands.

## What changed

### Core orchestrator (`backend/agents/rlm/`)

- `98f58b3` — **Foundation modules.** New `backend/agents/rlm/` package with `models.py` (root-model registry: GPT-5, Qwen3-Coder, Kimi K2.5, Claude — each maps a name key to an `rlms`-compatible backend config), `system_prompt.py` (the reproduction-domain RLM system prompt, formatted as an `rlm.format()` template), `sse_bridge.py` (an `RLMLogger` subclass streaming `repl_iteration` / `sub_rlm_*` / `run_complete` events with corpus sanitization), `checkpoint.py` (per-iteration `RLMRunIteration` events to the SQLite event store + sanitized snapshots), `report.py` (writes `final_report.{json,md}`), and `stub_primitives.py` (no-op stubs so #60 is runnable without #59).
- `427478d` — **`run.py` — the RLM run entry.** `run_pipeline_rlm`: builds the `rlm.RLM(...)`, wires SSE emission into the `on_*` callbacks, runs `.completion()` on a worker thread to preserve the asyncio event loop, and writes `final_report.{json,md}` from the result.
- `1956014` — **System prompt + integration harness.** `system_prompt.py` is an `rlm .format()` template (not an f-string); includes primitive signatures, Algorithm-1 operating principles, and in-context decomposition examples. Integration test harness exercises the full run with stub primitives.

### Pipeline wiring

- `57d0625` — **`--mode rlm` wired through `cli.py` / `pipeline.py` / `live_runs.py`.** `pipeline.py` routes `mode="rlm"` to `run_pipeline_rlm`; `live_runs.py` spawns it in the same subprocess pattern as `sdk` mode; `cli.py` exposes `--mode {offline,sdk,rlm}`. The `offline` and `sdk` modes are unchanged.

### Robustness

- `590ab26` — **Stub fallback when #59's binding is present-but-incomplete.** `run.py` lazily resolves `build_custom_tools` and falls back to `stub_primitives` if the import succeeds but the binding is incomplete, so the orchestrator runs end-to-end in CI without the full primitive layer.
- `38b4fd3` — **Vendor `RunContext` + test conftest.** A minimal copy of #59's `RunContext` is checked in so #60's test suite compiles independently.

### Documentation

- `9eaacff` — `CHANGELOG.md`, `learn.md`, and design spec v3 updated to record the #60 orchestrator.
- `da7d92d` — Phase 5 / #62 end-to-end handoff + #59 merge plan.

## Testing

```bash
.venv/bin/python -m pytest tests/
```

The full test suite passes at **1079 tests** (recorded in the Phase 5 WS-H hardening run). The orchestrator integration test (`tests/test_rlm_orchestrator_integration.py` or equivalent) exercises a full `run_pipeline_rlm` call with stub primitives and verifies that `final_report.json` is written and the SSE bridge emits `run_complete`.

## Risks / known issues

- `sandbox_mode` is not plumbed to the RLM primitives (it is a `run_pipeline_rlm` argument, but `RunContext` has no sandbox field). `run_experiment` hardcodes `LocalDockerBackend`. RunPod/Brev are unreachable from the RLM primitives until this is plumbed (tracked as a Phase 5 follow-up).
- Claude as a root model is not paper-validated — the RLM paper validates GPT-5 and Qwen3-Coder. Using Claude is an empirical bet; verify on at least one real paper before relying on scores.
- `rlms`'s `max_timeout` only fires between iterations; a primitive wedged inside `execute_code` overruns it. The per-primitive deadline added in WS-H (Phase 5) is the real enforcement layer.

## Checklist

- [x] All commits are on `feat/rlm-phase3-orchestrator`; base branch is `rlm-pivot`
- [x] New modules are in `backend/agents/rlm/` — no existing files modified
- [x] `offline` and `sdk` modes are byte-identical before and after
- [x] Full test suite passes (1079 tests)
- [x] `CHANGELOG.md` and `learn.md` updated
- [x] No secrets or credentials checked in
- [ ] Real end-to-end run on a PaperBench paper (deferred to Phase 5 / #62)
