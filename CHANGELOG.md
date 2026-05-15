# Changelog

All notable changes to OpenResearch land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html). Add new entries to the top
of `[Unreleased]`. When you cut a release, rename `[Unreleased]` to the
version + date and start a new `[Unreleased]` block above it.

## [Unreleased]

### Added
- **Track 4 — environment build-and-repair loop.** The reproduction Dockerfile
  is now built — and repaired on failure — at the `ENVIRONMENT_BUILT` stage
  instead of failing tens of minutes later inside `run_experiment`. A build-only
  `build_image()` primitive (`backend/services/runtime/local_docker.py`)
  compiles the Dockerfile and returns `(ok, tag, error_text)` for a broken
  Dockerfile (repairable) or raises `SandboxRuntimeError` for an infrastructure
  failure (docker daemon down — not repairable). `_run_environment_build_loop`
  in the orchestrator mirrors the Track 3 re-iteration loop: build → on failure
  feed the build error back to `environment-detective` in a new **repair mode**
  → bounded retry, capped by `environment_build_max_attempts` (default 3). When
  the cap is spent without a buildable image the run does **not** dead-end on
  `blocked_requires_human` — it is **fail-soft**: `environment_build_ok` stays
  false, Gate 2's failure is allowed through, and the run completes with an
  honest partial-reproduction verdict. Opt-in via
  `environment_build_validation_enabled` (default on); with it disabled, or on a
  non-docker sandbox, the run behaves exactly as before. The loop runs *within*
  the `ENVIRONMENT_BUILT` stage (the `PipelineStage` enum stays at 14) and is
  resume-safe — `environment_build_attempts` / `environment_build_ok` /
  `environment_build_error` are checkpointed, so a run that died mid-loop picks
  up where it left off. The `environment-detective` prompt also gained
  Dockerfile-hardening rules (slim base + one curated apt layer, per-package pip
  layers, never `COPY` source — reproduction code is volume-mounted) to shrink
  the loop's workload by avoiding the common "missing system lib" failure class.
- **Track 3 Phase F.5 — Codex review #2 hardening.** Folded the genuinely
  actionable findings from the final Codex review (the remainder were verified
  false positives — hallucinated code structure, conflated schemas, a test/doc
  it claimed missing that exists). The re-iteration loop now breaks early when a
  round's verifier produces no new result, instead of burning the remaining
  capped rounds against a dead verifier. `_run_rubric_verifier` gained a
  *mechanical* honesty backstop: when the orchestrator knows the reproduction
  did not execute successfully (`experiment_artifacts.success` is false), every
  area score is hard-capped at 0.35 — the prompt's "code never ran" cap is no
  longer merely advisory. The completion summary's dispersion stat is relabelled
  "SD across N areas" — it was mislabelled "SE", which implies a sampling
  distribution over runs that does not exist for a single run's per-area scores.
- **Track 3 Phase E — completion summary popup + live re-iteration indicator.**
  New `CompletionSummary` component (`repro-lab-client.tsx`) — a floating,
  dismissible, re-openable popup shown on `run.status === "completed"`, matching
  the existing `.agent-window` / `.benchmark-card` design language. It leads
  with the honest 2-line verdict and a headline ("PaperBench X → Ours Y" when a
  published baseline exists, else "Our rubric score Y, +Δ over baseline across N
  iterations"), a meets/below-target chip, a `mean ± SE` stat line over the
  rubric areas (computed client-side), and a truthful per-area breakdown — full
  0-1 bars with a baseline ghost tick so the improvement is visible. It degrades
  to a minimal card when the verifier did not run. A **live re-iteration badge**
  in the workflow header surfaces `improvement_iteration` + the latest verifier
  score while the loop runs, plumbed `pipeline_state.json` →
  `PipelineStateDocument` → `buildLiveDemoDashboard` → `payload.summary`.
  `finalize_benchmark` also emits `baselineRubricAreas`, and the backend
  `BenchmarkSummary` model now carries every Track 3 field so they survive the
  `LiveRunState` round-trip into the UI instead of being dropped by pydantic's
  `extra="ignore"`.
- **Track 3 Phase C.5 — rubric hardening (Codex review #1).** The canonical
  rubric is now resolved **once per run** — a vendored PaperBench bundle's
  rubric for `paperbench_<id>` runs (wiring `BundleRubricSource` via the
  project-id seam), or LLM-generated on the first verifier call — then persisted
  in `PipelineState.rubric_spec` and reused at every checkpoint, so
  `baseline_verification` and `improved_verification` are measured on the same
  rubric. Per-area **weights now come from the persisted spec**; the verifier
  LLM supplies scores only, closing the score+weight gaming hole. A verifier
  failure no longer clobbers a prior good verification (`if x is not None` guard
  at both gates). The verifier context now includes `baseline_result` (the
  code / Dockerfile / command paths it is told to inspect). The rubric-verifier
  prompt was rewritten to score *submitted artifacts* — method/code fidelity,
  data fidelity, execution, evaluation, result-match, provenance — with explicit
  honesty hard caps (no code → ≤0.20, code never ran → ≤0.35, …) instead of
  process/effort areas. Re-iteration rounds namespace their `path_id`s
  (`r<N>_path_<k>`) so workspaces and `path_results` never collide.
  `resolve_rubric_source` degrades on malformed/unreadable bundles, not just
  missing ones. `rubric_target_score` is documented as a heuristic on the
  verifier's own scale (per-version calibration deferred).
- **Track 3 Phase D — completion-report fields + benchmark payload.**
  `FinalReport` gained `rubric_verification`, `baseline_rubric_verification`,
  `paperbench_baseline`, `verification_delta`, `improvement_iterations`, and a
  deterministic, honest `comparison_summary` (`backend/agents/schemas.py`).
  `generate_final_report` populates them from the orchestrator's verification
  state; `_build_comparison_summary` states plainly when self-improvement did
  not help or the target was not met (it never inflates), and
  `_resolve_paperbench_baseline` surfaces the published PaperBench score for
  vendored-bundle runs (`paperbench_<id>` project ids) — `None` for uploaded
  papers. `finalize_benchmark` (`backend/services/events/live_runs.py`) maps the
  new fields onto the UI `benchmark` object (`paperbenchBaseline`,
  `ourRubricScore`, `verificationDelta`, `improvementIterations`, `meetsTarget`,
  `comparisonSummary`, `rubricAreas`), and `DemoBenchmarkSummary` (plus
  `DemoRubricArea` / `DemoPaperbenchBaseline`) was extended to match. Every new
  field is optional and defaults empty — a run with the verifier disabled
  produces a byte-identical report and benchmark.
- **Track 3 Phase C — capped self-improvement re-iteration loop.** After Gate 3,
  `ReproLabOrchestrator` now loops back through improvement-selection + Gate 3
  while the rubric verifier reports below `rubric_target_score`, up to
  `rubric_max_improvement_iterations` rounds
  (`_run_improvement_reiteration_loop`). Termination is guaranteed by the pure
  `_should_reiterate` guard plus the per-round `improvement_iteration`
  increment — no infinite loop. Fail-closed: a disabled verifier, an already-met
  target, a missing verification, an exhausted run budget, or any re-iteration
  error stops the loop and lets the run finish with the best verification so
  far. The loop reuses the existing improvements_selected / improvements_run /
  gate_3_passed stages — the `PipelineStage` enum is unchanged.
  `improvement-orchestrator` is now fed the *latest* verification (the improved
  one on a re-iteration round, the baseline one on the first pass), and
  `improvement_iteration` is checkpointed after each round for crash-resume.
  Target calibration is the single `rubric_target_score` applied identically at
  every checkpoint; the PaperBench published baseline (a different judge on a
  different scale) is surfaced as informational context in the completion
  summary in Phase D, not used as the loop's stopping target.
- **Track 3 Phase B — rubric-verifier wired into the pipeline.** The
  `rubric-verifier` agent now runs at two checkpoints — within Gate 2
  (post-`baseline_run`) producing `baseline_verification`, and within Gate 3
  (post-`improvements_run`) producing `improved_verification`
  (`backend/agents/orchestrator.py`). `PipelineState` gained
  `baseline_verification`, `improved_verification`, `verification_history`, and
  `improvement_iteration`, all round-tripped through the checkpoint; a run with
  the verifier disabled checkpoints byte-for-byte as before. The baseline
  verification's weak rubric areas now feed `improvement-orchestrator` as an
  explicit objective. Fail-closed by contract: a verifier error logs and falls
  back to the heuristic rubric — the run is never blocked. New opt-in config
  keys (`backend/config.py`): `rubric_verifier_enabled` (default on),
  `rubric_verifier_model` (empty = inherit the run model), `rubric_target_score`
  (0.70), `rubric_max_improvement_iterations` (2) — the last two are consumed by
  the Phase C re-iteration loop. `_invoke_agent` gained an optional
  `model_override`. The capped re-iteration loop and bundle/published-baseline
  target calibration land in Phase C.
- **Track 3 Phase A — rubric-verifier scaffold (no pipeline wiring).** Added the
  `rubric-verifier` agent to `AGENT_REGISTRY` with a two-phase prompt
  (`backend/agents/prompts/rubric_verifier.py`): Phase 1 establishes the rubric
  (from a vendored PaperBench bundle or generated from the claim map); Phase 2
  scores each area against actual artifacts with concrete weak-point lists.
  Added `RubricAreaScore` and `RubricVerification` schemas
  (`backend/agents/schemas.py`) — `RubricVerification.from_areas` computes
  `overall_score` and `meets_target` deterministically from per-area scores and
  weights; the LLM-reported values are never trusted. Added the `RubricSource`
  abstraction (`backend/agents/rubric_source.py`) with `BundleRubricSource`,
  `GeneratedRubricSource`, and `resolve_rubric_source` — degrades cleanly to
  `GeneratedRubricSource` when no validated bundle is found. Orchestrator wiring
  is Phase B; nothing in the existing pipeline is changed.
- **Persistent, URL-addressable lab runs.** The active run is now keyed
  by the `?projectId=` query param — `app/lab/page.tsx` is an async
  server component that restores the run server-side, so a refresh or a
  shared link reopens the exact run instead of dropping to the upload
  view. A per-browser `localStorage` pointer (`reprolab:lastRun`)
  auto-resumes an in-flight run when the tab is reopened on a bare
  `/lab` URL; a stale link, a deleted run, and a transient 504 are each
  handled without discarding a live run. New `lib/demo/server-run.ts`
  holds the shared server-only run-fetch helper.
- **Hybrid vision-leaning paper extraction.** A modular `PaperExtractor`
  augmentation pass runs after the base PyMuPDF parser
  (`backend/services/ingestion/parser/extractor.py`). In `hybrid` mode
  (`REPROLAB_PAPER_EXTRACTION_MODE`, default `hybrid`) it renders
  scanned / figure-heavy pages to images and calls Claude vision
  (`vision.py`) for figure / table / equation descriptions and OCR
  text, enriching the parsed `full_text` that downstream agents
  consume. Fail-soft by contract: `extract()` never raises and degrades
  to text-only when vision is unavailable, so `text` mode and the
  behaviour for ordinary text PDFs are unchanged. `ParserAppService`
  now also emits the (pre-existing) `FigureExtracted` event.
- **Apify ArXiv MCP server integration.** When `APIFY_API_TOKEN` is set,
  the Claude agent runtime registers `https://jakub-kopecky--arxiv-mcp-
  server.apify.actor/sse` as an MCP server named `apify-arxiv` (SSE
  transport, `Authorization: Bearer` header) and grants its tools to the
  builder agents listed in `REPROLAB_APIFY_ARXIV_ENABLED_AGENTS`
  (default: `artifact-discovery,paper-understanding`). Tools surface as
  `mcp__apify-arxiv__*` and give those agents direct paper search /
  access / listing against arXiv without round-tripping through generic
  WebSearch. Empty token = MCP wiring is skipped entirely (no cold-start
  handshake, no extra latency).
- **Persistent Runpod pod reuse via `REPROLAB_RUNPOD_POD_ID`.** When set,
  `RunpodBackend.create_sandbox` attaches to the existing pod via SSH
  instead of `POST /pods`, reusing it across pipeline runs. The pod is
  structurally undeletable (never added to `_owned_pod_ids`). If the
  configured pod is missing or stopped, the backend creates a new
  persistent pod and logs the new id at WARNING
  (`RUNPOD_PERSISTENT_POD_CREATED`) so `.env` can be updated.
- **Default sandbox = `runpod` end-to-end.** Lab UI dropdown, settings,
  FastAPI form fallback, `/api/demo` route, Node demo runner, and CLI
  `--sandbox` flag (`backend/cli.py`, `backend/cli_paperbench.py`) all
  default to `runpod`. The ambiguous "Auto Docker" option was removed
  from the Lab UI dropdown — three explicit choices: Runpod / Docker /
  Local.
- **`scripts/runpod_check.sh` preflight.** Exit-coded auth + SSH key
  pair + GPU/account sanity check, optionally `--start-pod` for a paid
  end-to-end smoke. `start.sh` runs the free preflight before booting
  uvicorn; opt-in `START_FULL_SMOKE=1` for the paid smoke.

- **Typed provider resilience for SDK agents.** Agent invocations now use
  a reusable `backend/agents/resilience/` layer with provider-neutral
  failure classification (`QuotaExhausted`, `RateLimited`,
  `TransientError`, turn/tool/wall-clock budget failures, auth and guard
  failures), bidirectional Anthropic↔OpenAI quota fallback, partial-output
  continuation prompts, provider health cooldowns, append-only
  `cost_ledger.jsonl`, and `fallback_summary.json`. CLI runs can enforce
  pre-invocation caps with `--max-usd`, `--max-wall-clock`, and
  `--max-invocations agent=count`.
- **Codex CLI Hermes audit fallback.** Hermes provider chain now appends
  `codex_cli`, which uses `codex exec` with the operator's ChatGPT OAuth
  session after API-key OpenAI is unavailable. The provider never reads
  OAuth tokens; it only checks for the binary and `~/.codex/auth.json`.
  Optional overrides: `REPROLAB_CODEX_CLI_PATH` and
  `REPROLAB_CODEX_AUTH_PATH`. Verified locally with `codex-cli 0.125.0`.
- **Phase 2 research workspace services.** Added deterministic services for
  AST-backed knowledge graph construction and `graph_query()`, reusable
  cross-project memory, multi-paper comparison summaries, Git worktree
  isolation for improvement paths, dataset cache planning, approval
  checkpoints, failure diagnosis, and reproducibility scoring. FastAPI now
  exposes the Phase 2 workspace read model and service endpoints under
  `/phase2/...`; the orchestrator uses worktrees for improvement agents when
  the verified baseline is a Git repository.
- **Full-stack Docker image + compose.** New `Dockerfile` (3-stage:
  Python deps → Next.js build → slim runtime) packages backend and
  frontend together. `docker-compose.yml` mounts the host docker
  socket so the inner `LocalDockerBackend` keeps working without
  nested DinD; mounts `runs/` and `third_party/` as volumes; mounts
  `.env` read-only and sources it inside the entrypoint so
  `docker compose config` does not print local secret values.
  `docker/entrypoint.sh` boots uvicorn and
  `next start` under tini and forwards SIGTERM so `docker stop` is
  fast. `.dockerignore` keeps the build context lean (excludes venvs,
  node_modules, `.next/dev/`, runs, db files). Quick start:
  `docker compose up --build`; lab UI at http://localhost:3000/lab,
  PaperBench UI at http://localhost:3000/paperbench, backend health
  at http://localhost:8000/health.
- **RunPod pod-deletion guardrails.** `RunpodBackend._delete_pod` now
  refuses to delete any pod ID that wasn't created by the same backend
  instance. Defense in depth on top of `delete_on_destroy=false`: even
  if that flag flips, an in-memory allowlist (`_owned_pod_ids`)
  populated only by `_create_pod` blocks deletion of foreign pods. A
  belt-and-suspenders second check verifies the pod's name still
  starts with `reprolab-` before issuing the API DELETE — so a future
  code path that adds a pod ID to the allowlist by mistake still
  can't delete a coworker's pod on the same RunPod account. Locked in
  by `tests/test_runpod_delete_guardrails.py` (4 tests).
- **Hermes audit adapter — robust, self-learning, fallback-aware.**
  `backend/hermes_audit/client.py` no longer fails closed when the Nous
  Hermes runtime is missing. New chain: Nous Hermes → Anthropic SDK →
  OpenAI SDK → degraded `unavailable`. Each provider implements a
  small `AuditProvider` Protocol (`backend/hermes_audit/providers.py`)
  so future backends plug in by registration, not by editing the
  client. Per-provider success / failure counters persist between runs
  to `<runs_root>/.hermes_adapter_memory.json`
  (`backend/hermes_audit/memory.py`); the next run starts with the
  last-good provider first and quarantines a provider after 3
  consecutive failures until it recovers. Hardened JSON extraction
  tries fenced-block, balanced-brace, and prose-prefix-strip
  strategies in order. Every fallback logs to stderr at WARNING; the
  resulting `HermesAuditReport.provider` field shows which auditor
  actually answered. Atomic write (`tempfile + os.replace`) keeps the
  memory file consistent if a process is killed mid-write.
- **Backend `.env` bootstrap.** `backend.__init__` now loads local `.env`
  values into `os.environ` once without clobbering exported variables, so
  Hermes providers and RunPod credentials work consistently from CLI,
  tests, child Python processes, and the Docker entrypoint.
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
- **Layer 2 semantic store** with Chroma vector embeddings + BM25
  fallback, auto-wired through `_make_services` when `chromadb` is
  installed (from upstream, second merge round).
- **PaperBench section in `docs/setup-guide.md`** — 4-step workflow
  from `list` → `summary` → `--no-pipeline` dry → real run + UI link.

### Changed
- **Graph-first lab layout — edge-docked drawers.** The workflow view's
  3-column grid (`graph | 360px | 320px`) is replaced by a full-bleed
  node graph with the node-details panel and the agent timeline as
  collapsible drawers docked to the right viewport edge. Collapsed they
  are labelled tabs; expanded they overlay the graph without reflowing
  it, behave as an accordion, and persist their open state. The header
  action is now a prominent primary "Start New Run" button.
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
- **Live logs surface subprocess stdout, not just stderr.**
  `FileLiveRunService._read_log` read only `runner.stderr.log`, so agent
  activity on stdout was invisible in the UI. It now reads and combines
  both `runner.stdout.log` and `runner.stderr.log` under the existing
  tail cap.
- **Live log lines render verbatim.** `parseLogEntries` piped every line
  through `issueText`, rewriting `failed` → `needs attention` — that hid
  real failures from anyone reading the log. Log lines are now
  unmodified, and the window grew from the last 20 to 80 lines.
- **An un-enriched `run_state` frame no longer regresses the graph.**
  The GET (750 ms) and SSE (250 ms) routes both cap payload enrichment
  and fall back to the un-enriched backend state on timeout; applied
  raw, a payload-less frame blanked the graph's per-path nodes. A new
  `coalesceRunState` carries the last enriched payload / telemetry / log
  forward until the next enriched frame. (Track 2 §5.4 originally
  attributed this to a missing orchestrator emit — the orchestrator
  never builds a payload at all; the frontend reconstructs `pathStates`
  from `path_results` during enrichment.)
- **Dashboard / telemetry JSONL writes flush immediately.**
  `DashboardEmitter._emit` and `AgentTelemetryRecorder.append` relied on
  the context-manager close to flush; an explicit `flush()` makes each
  completed event durable if the run subprocess is killed before close.
- **SIGINT/CancelledError no longer dumps a 50-line stack trace.**
  `backend/cli.py:cmd_reproduce` catches `(KeyboardInterrupt,
  asyncio.CancelledError)` around `asyncio.run`, flips
  `demo_status.json` to `status="stopped"` via an atomic write, and
  exits 130. `orchestrator.py` step loop adds an explicit cancellation
  guard before the generic `except Exception` so the runner log shows a
  clean `|| STOPPED at <stage>` line and `state.save_checkpoint()` runs
  for resume.
- **`demo_status.json` corruption window closed.** Both
  `cli._atomic_write_json` and `live_runs._write_status` now write via
  tempfile + `os.replace`, so a crash during a status flush leaves
  either the previous valid JSON or the new one — never a truncated
  file that breaks `_read_status` downstream.
- **Lab page white-screen under busy backend.** `lab/page.tsx` SSR
  fetch uses `AbortSignal` with a 1.5 s timeout; `/api/demo` GET proxy
  adds a 4 s timeout (returns 504 instead of hanging); a client-side
  recovery poll with exponential backoff (2 s → 15 s cap) refills run
  state when SSR couldn't get a snapshot.
- **Hermes audit schema drift.** The audit adapter now normalizes common LLM
  variants before validating `HermesAuditReport`: object-valued
  `unsupported_claims`, string-valued `evidence_refs`, numeric confidence, and
  free-form `recommended_intervention` text are coerced into the strict public
  schema while preserving the original response in `raw_response`.
- **`Reached maximum number of turns (15)`** silently aborting every
  PaperBench-class run at turn 16. Caps are now both higher and
  programmatically inspectable; see `learn.md` 2026-05-09.
- **`ValueError: Invalid IPv6 URL`** crashing `paper_understood` on any
  agent output containing brackets — see `learn.md` 2026-05-09.
- **`database disk image is malformed`** on `reprolab.db` after a killed
  pipeline subprocess. Restored from offline backup; `synchronous=FULL`
  prevents recurrence — see `learn.md` 2026-05-09.

### Documentation
- `docs/agent-lifecycle.md` — end-to-end agent pipeline stages, task states,
  runtime statuses, verifier decisions, and Phase 2 service attach points.
- `learn.md` — durable post-mortem + practice runbook.
- `third_party/paperbench/README.md` — instructions for swapping the
  FTRL placeholder bundle for the upstream PaperBench artifacts.
- `CHANGELOG.md` (this file).

### Tooling
- `.gitignore` exclusions for build/test artifacts:
  `frontend/tsconfig.tsbuildinfo`, `_test_logs/`, `reprolab.db.*`,
  Windows `*:Zone.Identifier`, stray pip-version files at repo root.
  `paperbench1.pdf` is whitelisted as the canonical input fixture.
- `tests/test_issue16_workspace_service.py::test_build_workspace_auto_embeds_chunks`
  now uses `pytest.importorskip("chromadb")` so the suite stays green
  when the optional `reprolab-backend[semantic]` extras aren't
  installed (matches the pattern in `test_semantic_layer2.py`).
