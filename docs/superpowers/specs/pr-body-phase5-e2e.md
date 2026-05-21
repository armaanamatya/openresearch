> DRAFT — a late api-key fail-fast fix and real reproduction-run results land after this draft. Refresh the "What changed" tail and add a run-results section before opening the PR.

## Summary

Phase 5 of the RLM pivot (issue #62): merges the completed primitive layer (#59) and orchestrator (#60), vendors three real PaperBench bundles, production-hardens the RLM surface for real reproduction runs, adds the bundle→RLM bridge script and post-run leaf scorer, and adds a structural Brev sandbox backend. The goal of Phase 5 is the first real end-to-end reproduction: `--mode rlm` on a real PaperBench paper producing `final_report.json` with a real rubric score. A real run is in progress; scores will be added before the PR is opened.

## What changed

### WS-A — merge #59 (RLM primitive layer) + #60 (orchestrator)

- `f23de61` — **Merge #59 (`feat/rlm-phase2-foundation` at Task 14 — COMPLETE) into `feat/rlm-phase5-e2e`.** All 14 primitive tasks merged; one real conflict (`__init__.py __all__`) resolved as a union. Primitives: `understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment` (with repair loop), `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric` (with honesty backstop), `propose_improvements` (variable-length, paper-specific).
- `2bba002` — Merge #59 review fixes into the branch.
- `11d706c` — Pin the `implement_baseline` manifest-path fix.
- `c9b446b`, `1112761`, `bf4a66b`, `ba2e6f7`, `e170fcb`, `8d689cf` — Final-review fixes and integration tests: `build_custom_tools` wiring, primitive descriptions, capstone integration test tightened.
- `3c7363b`, `dc7b095` — Interim merges of #59 at Task 11 and Task 9 checkpoints during parallel development.

### WS-B — real PaperBench bundles vendored

- `771547a` — **Three real PaperBench bundles** vendored from `openai/preparedness` into `third_party/paperbench/<id>/` (`paper.md`, `addendum.md`, `rubric.json`): **`ftrl`** (178 rubric leaves), **`sequential-neural-score-estimation`** (92), **`mechanistic-understanding`** (96). Replaces the synthetic placeholder that was in `ftrl/`. Artifacts stored in Git LFS on `openai/preparedness`; fetched via the LFS batch API (the LFS store redirects to `openai/frontier-evals`).

### WS-H — production-hardening (4-batch audit)

- `87e02d2` — **Production-hardening of the RLM surface.** A 4-agent audit (~47 findings) produced 4 fix batches:
  - **Batch P — per-primitive deadlines.** `RunContext` gained `deadline_utc: datetime | None` and `remaining_s()`. A `run_with_deadline(coro, ctx, cap_s)` helper wraps each primitive's async body in `asyncio.wait_for(min(cap_s, remaining_s()))` and, on `TimeoutError`, runs teardown before returning a fail-soft error dict. The three long primitives — `build_environment` (3 × 1800 s repair + aggregate cap), `implement_baseline` (was blocked forever on `pool.submit(...).result()`), and `run_experiment` (N commands × 3600 s) — all route through it.
  - **Batch O — `max_usd` cost cap + corpus-leak redaction.** `RLMLogger.log()` compares `cost_ledger.total_usd()` to `run_budget.max_usd` between iterations; on breach it requests run stop and `_finalize` marks `status="failed"` (→ non-zero CLI exit). Cost reporting drops the always-zero `sub_usd` field. `redact_corpus(text, sentinels)` is applied at every remaining egress: `sse_bridge` stdout/stderr prefixes and `report.py`'s final-report strings.
  - **Batch L — run-status integrity.** Non-atomic `demo_status.json` writes in `live_runs.py` are now tempfile + `os.replace` on every code path. The watchdog `os._exit` writes a terminal `status=failed` and a terminal SSE frame before exiting. `stop_run` escalates from SIGTERM to SIGKILL after 10 s.
  - **Batch R — RunPod backend hardening.** `gpu_mode` and `network_disabled` flags are honoured. HTTP 401/403 from the RunPod API is classified `retryable=False`. Pod-death detection avoids a 900 s spin. Cleanup is `asyncio.shield`-ed from cancellation. SSH host-key handling and idempotent destroy are in place.

### WS-F — Phase 5 documentation

- `1651317` — `CHANGELOG.md`, `learn.md`, `system_overview.md`, and `CLAUDE.md` updated to reflect all Phase 5 changes.

### WS-E — bundle→RLM bridge + post-run leaf scorer

- `240a7e9` — **`scripts/rlm_paperbench.py`** (the bundle→RLM-run bridge): loads a vendored PaperBench bundle, builds the claim map + rubric spec, and calls `run_pipeline_rlm`. **`backend/evals/paperbench/leaf_scorer.py`** (post-run authoritative scorer): flattens rubric leaves → batched LLM grading → weighted roll-up to produce the PaperBench-comparable score. `amend_final_report` writes scores back into `runs/<id>/final_report.json`.

### WS-G — Brev sandbox backend (structural, unverified)

- `9ab2c4a` — **`backend/services/runtime/brev_backend.py`**: structural Brev sandbox backend that shells out to the `brev` CLI (brev.dev has no public REST API). Marked **unverified** — no `BREV_API_KEY` available during development. Not wired into the default sandbox selector.

### Fix — api_key plumbing

- `ab78dd4` — **`models.py` did not plumb `api_key` into `backend_kwargs`**, causing `rlms`'s `AnthropicClient` to fail on initialization. Fixed. This was the blocker on reproduction attempt 2.

## Testing

```bash
.venv/bin/python -m pytest tests/
```

Full suite passes at **1079 tests** (recorded at the end of the WS-H hardening batch). The test count includes primitive-layer unit tests (all 14 tasks), orchestrator integration tests (stub-primitive end-to-end), and the existing backend suite.

## Risks / known issues

The following are drawn directly from the handoff (§8) and the Phase 5 plan:

- **Docker Desktop WSL integration flakiness.** The real reproduction run requires stable local Docker. Docker Desktop's WSL integration dropped twice during Phase 5 development — this is the #1 operational risk for any local run.
- **Brev backend unverified.** `brev_backend.py` is structural only; `BREV_API_KEY` is unavailable. Do not route production runs through it.
- **RunPod key invalid.** The `REPROLAB_RUNPOD_API_KEY` in `.env` returns HTTP 401. RunPod runs are blocked until the key is replaced.
- **`sandbox_mode` not plumbed to RLM primitives.** `run_experiment` hardcodes `LocalDockerBackend`; `RunContext` has no sandbox field. RunPod/Brev are unreachable from within an RLM run until this is plumbed (deferred post-Phase-5).
- **`run_experiment`'s per-primitive deadline cannot kill the underlying worker thread.** The `asyncio.wait_for` returns on timeout, but the thread running the Docker command continues. `run.py`'s process watchdog (`max_usd` / `max_wall_clock`) is the hard backstop.

## Checklist

- [x] All commits are on `feat/rlm-phase5-e2e`; base branch is `rlm-pivot`
- [x] #59 (primitive layer) and #60 (orchestrator) fully merged
- [x] Full test suite passes (1079 tests)
- [x] `CHANGELOG.md` and `learn.md` updated (WS-F, commit `1651317`)
- [x] No secrets or credentials checked in
- [x] Corpus-leak redaction applied at all egress points (WS-H Batch O)
- [x] Atomic `demo_status.json` writes on every code path (WS-H Batch L)
- [ ] Real end-to-end run completed with rubric score (WS-E — run in progress; add results before opening)
- [ ] Second paper run completed (Phase 5 deliverable: ≥2 papers with real rubric scores)
- [ ] Brev backend verified with a real key (deferred)
- [ ] `sandbox_mode` plumbed to `RunContext` → `run_experiment` (deferred post-Phase-5)
