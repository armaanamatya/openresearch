# Tier 2 — Backend Observability Hook (Implementation Plan)

**Depends on:** Tier 1 merged. Tier 2 reads `REPROLAB_LOG_DIR` (set by `dev.ps1`/`dev.sh`).
**Scope:** Backend code only — frontend untouched. Recorder is **opt-in via env**, so production / non-dev runs are unaffected.

---

## Sub-tiering (ship incrementally)

### Tier 2a — Root logger config (smallest, biggest immediate payoff)

**New file:** `backend/observability/run_logging.py`
- `configure_root_logger()` — idempotent; reads `REPROLAB_LOG_DIR` (fallback `REPROLAB_RUNS_ROOT`); attaches a `FileHandler` writing `pipeline.log` to that dir; sets root level INFO. No-op if env unset.
- Format: `%(asctime)s %(levelname)-7s %(name)s :: %(message)s`.

**Wire-up:** ~2 lines each in
- `backend/app.py` → `create_app` startup
- `backend/cli.py` → `main`

**Effect:** every existing `logger.info(...)` across `orchestrator.py`, `baseline_implementation.py`, `experiment_runner.py`, `resilience/engine.py`, etc. lands in one `logs/<TS>/pipeline.log`. Zero changes to call sites.

**Risk:** essentially nil. `FileHandler` is the most boring stdlib logging primitive.

---

### Tier 2b — Per-agent transcripts (the main payoff)

**Extends:** `backend/observability/run_logging.py`
- `AgentTranscriptRecorder(project_dir, agent_id, seq, prompt)` — writes:
  - `prompt.md` on construction
  - `trace.log` (append) for `StreamText`
  - `tool_calls.jsonl` (append) for `StreamToolCall` — captures sub-agent dispatches too (they appear as tool calls in the runtime stream)
  - `usage.json` overwrite-on-each-frame for `StreamUsage`
  - `result.txt` + `meta.json` on `finalize(status, error)`
- `contextvars.ContextVar` so the recorder is task-local. No signature changes to `run_agent_with_resilience`.
- Context-manager API + try/finally everywhere so `asyncio.CancelledError` doesn't leak handles.

**Wire-up:**
- `backend/agents/orchestrator.py:506` (`_invoke_agent`): increment `self._agent_seq`, construct recorder, `set_recorder(token)`, finalize in try/finally.
- `backend/agents/resilience/engine.py:229-264` (the event loop): inside each `isinstance(event, ...)` branch, additionally call `rec.record_text/tool/usage` if `get_recorder()` returns one.
- Retry handling: when the engine's recovery policy triggers a retry, write a `--- retry N ---` separator into `trace.log` so the transcript is honest about what actually happened.

**Disk cost:** trace.log grows at model-generation speed (~MB per long stage). Acceptable in dev. Add rotation as a follow-up if it bites.

**Risk:** medium. Touches a hot async loop. Mitigations:
- Recorder calls are all best-effort try/except — never propagate I/O exceptions into the resilience engine.
- File writes are line-buffered; OS handles the locking for `logging.FileHandler`.
- Contextvars are asyncio-safe (each task has its own copy by design).

---

### Tier 2c — Verify subcommand

**New file:** `scripts/verify_run.py`
- Walks `logs/<TS>/manifest.json`.
- Cross-checks: every `pipeline.jsonl` `stage_start` has a matching `stage_end`; every agent invocation has a non-empty `result.txt`; no zero-byte text files.
- Exits non-zero on gaps. CI-friendly.

**Risk:** none — pure offline analysis.

---

## Output layout (recap from §3 of unified-logging-launcher.md)

```
logs/<TS>/
├── pipeline.log             # NEW (Tier 2a) — root logger output
├── pipeline.jsonl           # NEW (Tier 2a) — structured stage events
└── prj_<project_id>/
    ├── dashboard_events.jsonl    # EXISTING — left untouched
    └── agents/                   # NEW (Tier 2b)
        └── 01-paper-understanding/
            ├── prompt.md
            ├── trace.log
            ├── tool_calls.jsonl
            ├── usage.json
            ├── result.txt
            └── meta.json
```

Note: `pipeline.log` / `pipeline.jsonl` go at `logs/<TS>/` (launcher dir) — they cover the whole launch, not a single project. Per-agent dirs nest under `prj_<id>/agents/` because they belong to one pipeline run.

---

## Sequencing & pre-conditions

1. **Tier 1 must be shipped first.** Per-agent transcripts presume `REPROLAB_LOG_DIR` resolves to a Windows-native path that the backend can actually open. The WSL-path bug from 2026-05-17 manifested as zero `prj_*` siblings — Tier 2 would have written into the wrong directory under the same bug.
2. **Test order within Tier 2:** 2a → smoke test (one PR) → 2b → smoke test (second PR) → 2c (third PR). Keeps blast radius small per merge.
3. **Don't bundle.** Three PRs, three reviews. Tier 2b is the only one that touches a hot async path; reviewers should see it isolated.

---

## Open questions (answer before 2b lands)

1. **Sequence numbering across orchestrator restarts.** Checkpoints exist (`orchestrator.py:294`). If a run resumes from checkpoint, should `seq` continue from the last value (requires persisting it) or restart at 1? Defaulting to restart-at-1 is simpler; the launcher's `<TS>` already disambiguates.
2. **Sub-agent depth.** A `StreamToolCall` for `Task` (the sub-agent dispatch tool) carries the sub-agent's full transcript inline. Capture the call only (one line in `tool_calls.jsonl`), or also burst it into a `agents/01-.../sub-agents/01-.../` tree? Recommend: call-only for v1; nested tree is a Tier 3 feature if anyone asks.
3. **Async exception coverage.** If `runtime.run_agent` raises before any event arrives, `finalize` runs in the orchestrator's except branch. Sanity-check that `result.txt` is written empty (not absent) so the verify subcommand can distinguish "agent ran, returned nothing" from "agent never ran".
