# Changelog

All notable changes to OpenResearch land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html). Add new entries to the top
of `[Unreleased]`. When you cut a release, rename `[Unreleased]` to the
version + date and start a new `[Unreleased]` block above it.

## [Unreleased]

### Added
- **PaperBench head-to-head pipeline.** Vendored FTRL bundle scaffold
  (`third_party/paperbench/ftrl/`) + bundle loader, weight-aware rubric
  scorer, submission validator, seeded multi-attempt runner, and the
  `reprolab paperbench {list,summary,run,status}` CLI subcommand. Default
  is `--pipeline` (real LLM run); `--no-pipeline` runs a dry validation
  for CI. Status JSON is persisted to `runs/paperbench/<run_group_id>/`.
- **`/paperbench` page** with paper picker, seed input, dry/pipeline
  toggle, 3 s polling, score-vs-baseline grid (margin colored), rubric
  breakdown, attempts table.
- **Lab observability.** Live progress strip with stage chips, animated
  bar, and 90 s stall warning; structured agent timeline panel
  (per-invocation card with success dot, model badge, duration, error
  message, All / Errors filter); `Copy debug bundle` button +
  `GET /api/lab/debug-bundle` endpoint that returns a compact
  status/log/telemetry/pipeline-state JSON for paste-into-Claude-Code
  triage.
- **Provider-agnostic runtime** with per-agent provider selection
  (`--provider` / `--verification-provider`) and Claude→OpenAI
  auto-fallback when Claude usage limits hit (from upstream).
- **RunPod sandbox backend** for cloud GPU runs (from upstream).
- **`AgentLimitExceeded` typed exception** with `kind` ∈
  `{turns, tool_calls, wall_clock}`, `limit_value`, `elapsed_seconds`,
  and preserved `partial_output`. The orchestrator converts the SDK's
  untyped `"Reached maximum number of turns (N)"` text into the same
  typed exception so callers branch on `kind` instead of string-matching.
- **`agent_wall_clock_seconds`** governor: 20 min in `efficient`, 1 h in
  `max`. Enforced via `asyncio.timeout` around `runtime.run_agent`.
- **`learn.md`** runbook of post-mortems + cross-cutting principles
  (10 entries currently). New bugs land here with a regression-test
  pointer.
- **Layer 1 RLM workspace service** wired into the orchestrator (from
  upstream).

### Changed
- `RuntimeGuard.find_blocked_term` no longer URL-parses arbitrary agent
  output. Lower-cases the haystack only and substring-matches against
  pre-canonicalised terms — fixes the `Invalid IPv6 URL` crash on
  bracketed agent narration under Python 3.12+.
- `_canonicalize_url_term` (formerly `_normalize_guard_text`) wraps
  `urlparse` in `try/except ValueError`; the discovery regex adapter
  gets the same treatment.
- `ExecutionProfile` defaults: `efficient` now caps `max_turns_per_agent=30`
  / `heavy_agent_max_turns=60` / `max_tool_calls_per_agent=80` /
  `agent_wall_clock_seconds=1200`. `max` keeps per-call caps at `None`
  but adds `agent_wall_clock_seconds=3600` so even unbounded runs
  eventually terminate.
- `SqliteEventStore._new_connection` now sets `PRAGMA synchronous=FULL`
  to survive `SIGKILL` of the writer mid-commit.
- `start.sh` resolves the project venv interpreter explicitly.
- `pipeline.py` / `orchestrator.py` / `cli.py` signatures expose both
  the PaperBench multi-attempt kwargs (`seed`, `attempt_id`,
  `run_group_id`, `blacklist_terms`) and the upstream RLM workspace
  kwargs (`workspace_service`, `workspace_id`).

### Fixed
- **`Reached maximum number of turns (15)`** silently aborting every
  PaperBench-class run at turn 16. Caps are now both higher and
  programmatically inspectable; see `learn.md` 2026-05-09.
- **`ValueError: Invalid IPv6 URL`** crashing `paper_understood` on any
  agent output containing brackets — see `learn.md` 2026-05-09.
- **`database disk image is malformed`** on `reprolab.db` after a killed
  pipeline subprocess. Restored from offline backup; `synchronous=FULL`
  prevents recurrence — see `learn.md` 2026-05-09.

### Documentation
- `learn.md` — durable post-mortem + practice runbook.
- `third_party/paperbench/README.md` — instructions for swapping the
  FTRL placeholder bundle for the upstream PaperBench artifacts.
- `CHANGELOG.md` (this file).

### Tooling
- `.gitignore` exclusions for build/test artifacts:
  `frontend/tsconfig.tsbuildinfo`, `_test_logs/`, `reprolab.db.*`,
  Windows `*:Zone.Identifier`, stray pip-version files at repo root.
  `paperbench1.pdf` is whitelisted as the canonical input fixture.
