# RLM Phase 5 — Production-Hardening Plan (WS-H)

> Synthesis of the 2026-05-21 four-agent audit (orchestrator core / primitive
> layer / runtime-sandbox / CLI-live-runs). ~47 raw findings, evaluated against
> the codebase, de-duplicated, tiered. This is the plan of record for the WS-H
> hardening; it drives the fix batches in §4.

## 1. Rejected (audit overreach — no change)

- **A1-H3 "file-open per iteration → fd churn."** A run does ≤20 sequential
  open/flush/close cycles on the snapshot JSONL — not a scalability problem, and
  no fd is leaked (context-managed). Holding one fd open for a whole multi-hour
  run is the *worse* option. No change.

## 2. Cross-cutting mechanisms (Opus-designed — batches implement these)

Three root-level abstractions; scattered patches are not acceptable.

- **M-DEADLINE — one run deadline, enforced per primitive.** `RunContext` gains
  `deadline_utc: datetime | None` (set by `run.py` from the wall-clock budget at
  run start) and `remaining_s()`. A helper `run_with_deadline(coro, ctx, cap_s)`
  wraps a primitive's async body in `asyncio.wait_for(min(cap_s, remaining))`
  and, on `TimeoutError`, runs the primitive's teardown (sandbox `destroy`)
  before returning a fail-soft error dict. Every long primitive
  (`build_environment`, `implement_baseline`, `run_experiment`) routes through it.
- **M-REDACT — one corpus-leak chokepoint at every egress.** A
  `redact_corpus(text, sentinels)` helper (sentinel = first 200 chars of each
  `context` corpus value). Applied wherever model/primitive output reaches a
  durable or streamed surface: `sse_bridge` stdout/stderr prefixes **and**
  `report.py`'s final-report strings — not only `sanitize_iteration`.
- **M-BUDGET — max_usd enforced, honestly reported.** The `RLMLogger.log()`
  callback compares `cost_ledger.total_usd()` to `run_budget.max_usd` between
  iterations; on breach it requests run stop and `_finalize` marks
  `status="failed"` (→ non-zero CLI exit). Cost reporting drops the always-zero
  `sub` field for an honest single `llm_usd` until per-primitive usage exists.

## 3. Tiered findings

### Tier 1 — MUST (a live run is unsafe or incorrect without these)
- **T1 · R2 per-primitive deadlines** — `run_experiment` (N commands × 3600 s),
  `implement_baseline` (`pool.submit(...).result()` blocks forever),
  `build_environment` (3 × 1800 s repair, no aggregate cap), repair LLM call
  unbounded. → M-DEADLINE. *(primitives.py, context.py)*
- **T2 · Cost cap** — `run_budget.max_usd` accepted, never enforced; a
  budget-exhausted rlm run currently exits 0 (green). → M-BUDGET.
  *(run.py, report.py, cli.py)*
- **T3 · Corpus-leak hardening** — Algorithm-2 invariant holds for iteration
  events but not the final report or primitive stdout prefixes. → M-REDACT.
  *(sse_bridge.py, report.py)*
- **T4 · Run-status integrity** — non-atomic `demo_status.json` write (a crash
  bricks the UI); watchdog `os._exit` leaves `status=running`, no terminal SSE
  frame; `stop_run` SIGTERM-only (a wedged run never dies). Fix: atomic writes
  on every path; a terminal status on every exit; SIGKILL escalation.
  *(live_runs.py, run.py, cli.py)*

### Tier 2 — SHOULD (robustness for a live demo)
- **T5 · Primitive input fail-soft** — primitives run LLM-written REPL code, so
  inputs are adversarial. Several raise opaque exceptions: `k="3"` →
  `out[:"3"]` TypeError; empty `env_id`; uncaught `ValueError` on bad LLM JSON;
  `float()` on non-numeric rubric fields; non-dict `method_spec`. Fix: uniform
  validation + fail-soft error dicts across all 9 (the D3 pattern). *(primitives.py)*
- **T6 · RunPod backend** — honor `gpu_mode`/`network_disabled` (currently
  ignored); 401/403 → `retryable=False` (the live blocker would otherwise loop);
  pod-death detection in the SSH wait (no 900 s spin); cleanup `asyncio.shield`-ed
  from cancellation; SSH host-key handling; idempotent destroy. *(runpod_backend.py,
  aggregate.py)*
- **T7 · Honest cost reporting** — `sub_usd` always 0.0; primitive-internal LLM
  cost unrecorded. → M-BUDGET reporting half. *(report.py, binding.py)*

### Tier 3 — NICE (polish / future-proofing)
- local-docker: `_ensure_image` build timeout; `copy_in` file mode 0o644;
  `tarfile` `filter='data'`; `NotFound`/`ImageNotFound` typed error mapping.
- thread-safety: `dashboard_emitter._emit` lock (watchdog vs worker thread);
  `RLMLogger._next_index` lock.
- misc: `models` registry lazy-build; `system_prompt` `count==1` assert;
  `ui_rlm_<provider>_` project-id prefix; `SqliteEventStore` close in `run.py`;
  SSE poll backoff after 60 s.

## 4. Fix batches (disjoint file sets — parallel-safe, commits serialized)

Sonnet implements each batch against this doc + a tight per-batch spec; Opus
reviews every diff; each batch ends green (`tests/rlm/` + targeted tests +
new contract tests for the mechanisms).

| Batch | Files | Findings |
|---|---|---|
| **P** | `primitives.py`, `context.py` | T1 (M-DEADLINE), T5 |
| **O** | `run.py`, `report.py`, `sse_bridge.py`, `models.py`, `system_prompt.py` | T2, T3, T7, A1 misc, Tier-3 misc |
| **L** | `live_runs.py`, `cli.py` | T4 |
| **R** | `runpod_backend.py`, `local_docker.py`, `interface.py`, `aggregate.py` | T6, Tier-3 docker |

Shared contract across batches: `RLMRunResult.status` ∈ {completed, partial,
failed} drives the CLI exit code (O sets it, L reads it). `RunContext` gains
`deadline_utc` + `remaining_s()` (P owns context.py).

Critical path for a robust real WS-E run: **P + O + L** (deadline / cost /
status). Batch R is not on the local-GPU run path and may land afterward.

## 5. Brev backend (WS-G)

Built **after** Batch R hardens `runpod_backend.py` — `brev_backend.py` mirrors
the *hardened* pattern and implements the RuntimeBackend contract (5 async
methods: `create_sandbox` / `exec` / `copy_out` / `copy_in` / `destroy`;
`SandboxConfig` / `Sandbox` / `ExecResult` / `SandboxRuntimeError`). Structurally
complete; unverified live until a `BREV_API_KEY` is provided.

## 6. Doc-update contract

`CHANGELOG.md` entry for the hardening; `learn.md` Rule/How/Why for the R2
deadline lesson and the corpus-leak-egress lesson. `system_overview.md` /
`CLAUDE.md` synced for the new fail-soft / deadline / cost-cap behaviors.
