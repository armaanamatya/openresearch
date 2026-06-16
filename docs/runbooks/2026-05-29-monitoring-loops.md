<!-- doc-meta: status=current; last-verified=2026-06-07 -->
# Monitoring loops — babysitting an SDAR sprint

> **Status:** Current. The lightweight loops for watching a long reproduction
> run, plus the `BUG-NEW-NNN` doc-loop convention. Read before driving a retry
> sprint.

## The loops

1. **Run status** — `runs/<id>/demo_status.json` (atomic). Terminal states:
   `completed`, `failed`, `stopped`, `killed` (SIGTERM/SIGHUP handler),
   `interrupted` (orphan sweep). A run stuck on `running` with no recent
   `updatedAt` is wedged — check the live log.
2. **Event firehose** — `runs/<id>/dashboard_events.jsonl` (append-only SSE
   log). Tail it; `repl_iteration`, `primitive_call`, `rubric_score`,
   `run_warning` are the high-signal types.
3. **Experiment results** — `runs/<id>/experiment_runs.jsonl`: one row per
   `run_experiment` with `success` / `failure_class` / `suggested_fix`. The
   first place to look when training "succeeded" but the score is 0.
4. **Cost** — `runs/<id>/cost_ledger.jsonl` (per-primitive USD) against
   `--max-usd`.
5. **Crash-aware live monitor** — `scripts/watch_run.py` surfaces a wedged or
   crashed run in real time.
6. **GPU** — `nvidia-smi` / the local GPU allocator leases
   (`backend/services/runtime/local_gpu_allocator.py`); a run that can't lease a
   card sits idle.

## `BUG-NEW-NNN` doc-loop convention

When a sprint surfaces a new defect, file it inline as `BUG-NEW-NNN` in the
relevant spec/runbook with: symptom → root cause → the resulting *rule* (kept in
`CLAUDE.md`) → the fixing commit/test. Incident *narratives* stay in the
spec/runbook; only the durable *rule* graduates to `CLAUDE.md`.

## Kill / restart

`scripts/loops/kill_and_restart.sh` handles SIGKILL (which the CLI cannot trap).
For graceful termination, a plain `kill <pid>` (SIGTERM) now flips the run to
`killed` via `cli._install_termination_handlers` (STAB-3).
