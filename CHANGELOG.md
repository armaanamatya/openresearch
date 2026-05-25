# Changelog

Notable changes to ReproLab. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Forced-iteration policy: when the root model calls `FINAL_VAR` but the rubric score is below target and iterations are below minimum, the orchestrator refuses and forces continued improvement.
- Rubric guard: agent-written `train.py` calls `assert_metrics_schema(...)` at end-of-script; missing keys raise `RubricGuardFailure` with structured repair context.
- Pre-flight validator: catches scope shortcuts and surrogate models before pod dispatch.
- Dev TUI: per-run signal aggregation in a single terminal view.
- Dynamic GPU selection: RLM root estimates VRAM requirements per paper; resolver maps to cheapest RunPod SKU from 8-GPU catalog (RTX 4090 through H200). Auto-escalation on CUDA OOM (up to 2 tiers). Per-GPU and per-run cost caps.
- Constellation canvas: force-directed graph replaces the 4-node tree; every primitive call and sub-RLM visible with progressive disclosure.
- Outcome canonicalization: natural-language outcome synonyms (success/ok/passed) mapped to canonical set; unknown values pass through as-is.

### Fixed
- Outcome strict-reject regression: natural-English synonyms were being silently dropped, causing zero outcome events on the wire.
- Run status transitions: runs that finish now reliably flip to `completed` (was stuck on `running` due to WSL2 atexit hang).
- SQLite concurrency: `BEGIN IMMEDIATE` + 30s `busy_timeout` prevents `database is locked` on parallel ingests.
- Leaderboard resilience: malformed legacy `final_report.json` files no longer crash the entire endpoint.

## [2026-05-23]

### Added
- Chat steering: bidirectional user-to-RLM channel via `POST /runs/<id>/messages` + `check_user_messages` / `respond_to_user` primitives. Chat panel docked in lab sidebar.
- Collapsible right sidebar: replaces floating popup; kind-specific panels for paper/work/candidate/subrlm nodes.
- Heartbeat primitive + iteration_heartbeat SSE event for liveness monitoring.
- Stderr watchdog: detects SDK `aclose()` deadlock loops; flags run as degraded without killing it.
- Azure OpenAI provider support (`--model azure`).
- Leaderboard: `GET /leaderboard` + `/leaderboard` frontend page. Filesystem-aggregated, read-only.
- Lab UI: rubric climb panel with score tween, sparkline, per-area status chips, candidate attribution.
- Recommend-next-tool advisor primitive (Reflexion-lite).

### Changed
- REPL primitives: 12 -> 14 (added heartbeat, recommend_next_tool).
- Real-time elapsed clock derives from `startedAt` + interval (was event-span).
- System prompt: anti-decline-bias framing for improvement candidates.
- `run_experiment` timeout: 7200s -> 1800s default (env-var tunable).

### Fixed
- `paper_claims` list-shaped returns no longer crash final-report generation.
- SSR hydration mismatch on elapsed tile.
- React duplicate-key warnings in rubric strip.
- `candidate_id="None"` wire-contract bug (3-layer defense).
- RDR polling resilience: 4s abort timeout, empty-200 convergence.

## [2026-05-22]

### Added
- Rubric-driven harness (`--mode rdr`): deterministic controller decomposes PaperBench rubric into work-clusters, dispatches one coding agent per cluster, repairs weak clusters in a capped loop. No LLM in the control flow.
- RLM lab frontend: exploration-tree canvas, rubric strip, REPL-state rail, report rail, primitive-call history. Pure `fold`-based reducer over SSE events.
- Dynamic best-source paper ingestion: HTML > PDF > OCR cascade via `ResolvingParser`.
- Self-generated rubric for arXiv papers without vendored PaperBench bundles.
- Featherless Qwen3-Coder root model backend.
- Per-primitive deadlines via `RunContext.remaining_s()`.
- Cost cap enforcement between RLM iterations.
- Corpus-leak redaction at every egress point.
- Run-status integrity: atomic writes via tempfile + `os.replace`.
- RunPod backend hardening: auth failure classification, pod-death detection, incremental file sync.

### Fixed
- `run_experiment` Bug A/B/C: stderr capture, image rebuild from project Dockerfile, network enabled for experiments.
- Reverted I3 root-prompt change that caused 21-iteration `understand_section` loop.

## [2026-05-21]

### Added
- RLM orchestrator: 12 domain primitives exposed as REPL callables via `rlms` library.
- Hybrid mode (`--mode rlm`): RDR Phase 1 + RLM adaptive repair.
- CLI entry point: `python -m backend.cli reproduce`.
- PaperBench evaluation framework: leaf scorer, submission builder, bundle support.
- SSE event bridge with corpus-free invariant.
- RunPod GPU sandbox with SSH-based execution.
- Docker sandbox with network/memory/CPU controls.
- Claude OAuth path for sub-agents (no API key needed).
- Checkpoint/resume support for RLM runs.

### Earlier

Core infrastructure: FastAPI backend, Next.js frontend, SQLite event store, paper ingestion pipeline, CQRS persistence, Hermes audit chain.
