# Changelog

> ⚠️ **This changelog is behind `main` and is NOT the source of truth for
> recent history.** The newest entry below stops at **2026-05-23**, but
> development continued through 2026-06-03 (RLM stability remediation
> BUG-LR-011..015, OAuth-contamination fix BUG-NEW-038, evidence-gate closure,
> the gated codex repair primitive, the CLAUDE.md compaction, and more — none
> recorded here). For an accurate, current history use **`git log`**. This file
> is kept for the manual narrative it does contain; treat it as a partial,
> hand-maintained log, not a complete record. *(flagged 2026-06-03)*

All notable changes to OpenResearch land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html). Add new entries to the top
of `[Unreleased]`. When you cut a release, rename `[Unreleased]` to the
version + date and start a new `[Unreleased]` block above it.

## [Unreleased]

### Added (infra — Bicep migration L0/L1, OIDC pipeline, uv lock + ruff)
- `infra/azure/bicep/` — full Bicep port of the Terraform L1 (network, AKS, GPU nodepool, ACR, storage, identity) plus a new `monitoring.bicep` (Log Analytics + diagnostic settings, security-baseline). Three deliberate parity breaks fixing latent never-deployed TF defects (storage `publicNetworkAccess`, CSI SMB shared-key, kubelet `listKeys`) — see `infra/azure/bicep/MIGRATION.md`. Terraform stays in-tree, deprecated, until live parity is proven.
- `infra/azure/bicep/main.bicep` + `rg-grants.bicep` (L0) — subscription-scope RG creation + scoped grants; pipeline principal gets Contributor + ABAC-constrained RBAC Administrator (assignable roles limited to the four L1 data-plane roles), main RG only.
- `infra/azure/bicep/bootstrap/` — `admin-bootstrap.sh` (one-time admin path: app registration + environment-scoped federated credential + what-if-gated L0 deploy) and `pipeline-identity.bicep` (Contributor-only adoption path: UAMI + GitHub federation for pre-existing RGs, e.g. `rg-sciartgen-external`).
- `.github/workflows/infra-deploy.yml` — PR validation (bicep build/lint, deliberately zero Azure credentials — no `pull_request` OIDC federation exists), and an environment-gated, branch-pinned deploy job (what-if preview + deployment stack). Codex security review 2026-06-10 drove the PR-credential removal, deny-settings correction, and least-privilege grants.
- `uv.lock` (153 pkgs) + `.python-version` (3.12) — locked Python env via uv; pip/requirements.txt path unchanged. `hermes-agent` removed from extras (its exact pins are unlockable; install per `backend/hermes_audit/providers.py`).
- `[tool.ruff]` in `pyproject.toml` + CI `lint` job (`uv lock --check` + `ruff check`, both pinned) — 652 baseline violations → 0 (355 auto-fixed, 7 hand-fixed, rest config-scoped). Fixed a real latent `NameError` in `primitives.py::_run_baseline_subprocess` (missing `Path` import) surfaced by un-suppressing F821.

### Changed (lab — minimizable score panels keep the constellation graph visible)
- `frontend/src/components/lab/rlm/collapsible-panel.tsx` + `.module.css` — reusable minimizable wrapper (slim header with chevron + live summary, a height-capped internally-scrolling body, collapse state persisted per key in `localStorage`). Wraps `ScorecardPanel` (rubric scores) and `RubricBreakdown` (leaf scores) so neither can grow to consume the whole column.
- `frontend/src/components/lab/rlm/rlm-lab.module.css` — floored the `.workspace` (graph) band at `min-height: clamp(220px, 34vh, 520px)` so the constellation canvas stays visible at all times, including after a run finishes or fails. Previously the unbounded score bands squeezed the `flex: 1 1 0` graph band to zero once a full run's leaf scores landed.
- `frontend/src/components/lab/rlm/collapsible-panel.test.tsx` — guard: default-expanded body, minimize/restore, persisted + hydrated collapse state, `defaultCollapsed`. See `learn.md` 2026-05-31.

### Added (night — constellation UI + dynamic sandbox capability + outcome canonicalization)
- `frontend/src/components/lab/rlm/constellation-canvas.tsx` (695 LOC) + `layout-constellation.ts` (234 LOC) — replace the 4-node Reingold-Tilford tree with a force-directed graph that visualizes every primitive call and every mini-RLM, with progressive disclosure so the default view stays clean.
- `frontend/src/components/lab/rlm/layout-constellation.test.ts` — pinned: empty input, 1+3 candidates, 50-primitive density (no-overlap under `forceCollide(radius+14)`), deterministic seed.
- `d3-force@^3` (~20 KB) + `@types/d3-force` in `frontend/package.json` — only the focused sub-package, not the d3 umbrella.
- `_OUTCOME_ALIASES` + `_canonicalize_outcome()` in `backend/agents/rlm/primitives.py` — case-insensitive map from natural-language outcome synonyms (success/ok/passed → promoted; fail/error → failed; partial → marginal; etc.) to the 6-value canonical set; unknown values pass through as literal.
- `_friendly_candidate_title()` helper in `backend/agents/rlm/binding.py` — compresses 60-char candidate titles to ≤5 words by splitting on colon / em-dash with first-words fallback; emitted alongside the full title as `display_title` on the wire.
- `_compute_constraint_guidance(sandbox_mode, gpu_mode)` helper in `backend/agents/baseline_implementation.py` — pure dynamic decision: `runpod` OR `gpu_mode=max` → no CPU constraint; `{docker, local}` × {off, auto, prefer, None} → CPU smoke-test guidance; unknown → conservative (no guidance).
- `RunContext.gpu_mode` field — threaded from `ExecutionProfile.gpu_mode` so the constraint helper sees the real value instead of always None.
- `tests/agents/test_baseline_implementation_sandbox_aware.py` — 10-case parameterized `TestDynamicComputeGuidance` (runpod/docker/local × off/auto/prefer/max/None) + `TestAuthSurfaceParity` (claude / claude-oauth / gpt-5 produce byte-identical prompts) + `TestGpuModePlumbedThroughRunContext`.
- `tests/rlm/test_binding.py` — `test_record_candidate_outcome_canonicalizes_outcome_aliases` (17 alias→canonical pairs incl. case-insensitivity + space/hyphen normalization) + `..._unknown_outcome_falls_back_to_literal_not_reject` + `..._empty_outcome_falls_back_to_unknown`.

### Changed (night)
- `rlm-lab.tsx` swaps `ExplorationCanvas → ConstellationCanvas`. `ExplorationCanvas + layout-tree` are NOT deleted — preserved as fallback.
- `TreeNode.kind` union extended with `primitive` and `llm_primitive`; `foldPrimitiveCall` appends one tree node per `status=ok` call keyed by `primitivePhase`. `LLM_USING_PRIMITIVES` set (understand_section, propose_improvements, recommend_next_tool, verify_against_rubric, plan_reproduction, extract_hyperparameters, detect_environment) gets the pulsing mini-RLM visual; `NON_VISUALIZED` set (heartbeat, check_user_messages, record_candidate_outcome, respond_to_user) is filtered out.
- `run_experiment` default cap removed — only honors `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` env var OR `ctx.remaining_s()` from `--max-wall-clock`. Without either, `timeout=None` waits indefinitely (per user mandate "no cost cap until set"). The CPU-bound-baseline problem the 1800s cap was working around is now addressed at the agent prompt layer instead.
- `record_candidate_outcome` — strict-reject of any outcome not in `_VALID_OUTCOMES` (commit 1f72e07) replaced with canonicalize-or-accept-literal. Empty/None outcome falls back to the literal string `"unknown"` so the event still emits.
- `_run_baseline_with_sdk` call now threads `gpu_mode=getattr(ctx, "gpu_mode", None)` so the sandbox-aware prompt receives the real value.

### Fixed (night)
- **Constellation viewport-fit + progressive disclosure** (per "scale to fit the display correctly, only show sub-RLM on click"): auto-fit-to-viewport with `MIN_FIT 600×400` floor so sparse runs don't look lost; user-interaction tracking stops auto-refit once they wheel-zoom or drag-pan; activity nodes (primitive / llm_primitive / subrlm) hidden by default; per-structural-node `+N activity` badge fades them in with 200ms transition; edges to hidden nodes are culled (no orphan lines).
- **Outcome strict-reject silent regression**: my commit 1f72e07's strict 6-value validator was dropping all 4 of C5's `record_candidate_outcome` calls because the model passed natural English synonyms instead of the literal canonical strings. Result: 0 outcome events on the wire, gate unreachable for that paper. C2 + C4 + C5 (3 papers, 0 promoted between them) may all have had outcomes silently eaten.
- **Static "docker = CPU forever" heuristic** (per "sandbox shouldn't be cpu only it should be dynamic since we can use runpod"): replaced with `_compute_constraint_guidance(sandbox_mode, gpu_mode)` honest 2D decision. runpod runs no longer get the CPU smoke-test nudge; `--gpu-mode max` overrides the constraint on any sandbox.

### Validated (night)
- **3 total promotions across this + the prior session** (B1 + C1 + C3); **2 distinct papers promoted** (CodeOCR ×2, LLM CodeGen ×1) of 4 attempted before the outcome-regression discovery. Round 3 (C6 + C7) running with the canonicalization fix live to chase the 3rd-distinct.
- **560 tests pass** across `tests/rlm/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite_concurrent.py tests/routes/test_leaderboard_http.py` after all 4 commits.
- **TypeScript clean** (`npx tsc --noEmit`); 214 vitest pass on the constellation surface (npm test infra blocked elsewhere on pre-existing Node 21 / rolldown — CLAUDE.md mandates ≥22.12).

---

### Added (late evening — parallel paper sweep + reliability sprint)
- `tests/services/events/test_live_runs_status_ordering.py` — 5 compile-the-wrapper tests pinning: `write_status("completed")` before `finalize_benchmark()`; finalize wrapped in try/except; `finally:` block with `os._exit`; exit code follows status; failure path also reaches finally.
- `tests/test_eventstore_sqlite_concurrent.py` — 3 tests pinning concurrent SQLite writers under WAL: 2 threads × different aggregates both succeed; 4×20 = 80 appends complete < 25 s; same-aggregate same-version → exactly one wins with `ConcurrencyError` (not `"database is locked"`).
- `tests/rlm/test_run_experiment_timeout.py` — 3 tests pinning new `run_experiment` cap: default 1800 s; `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` env-var override; invalid value falls back to default.
- `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` env var — tunable aggregate cap for `run_experiment` primitive.
- `runs/_archived_legacy_fixtures_20260523/` — 3 legacy test fixtures (`prj_verify_offline_report`, `prj_verify_offline_report2`, `prj_verify_polish`) moved here so the leaderboard scan doesn't trip on their list-shaped `rubric` fields.

### Changed (late evening)
- Demo wrapper template (`_python_script` in `live_runs.py`): `write_status("completed")` now fires BEFORE `finalize_benchmark()`. `finally:` block calls `os._exit(0 if completed else 1)` to bypass atexit hooks that can hang on WSL2.
- SQLite eventstore: all writers now `BEGIN IMMEDIATE` (was bare `BEGIN`); `busy_timeout` raised 5000 → 30000 ms.
- `record_candidate_outcome` primitive: validates `candidate_id` is non-`None`, non-empty, not literal `"None"`/`"null"`; validates `outcome` against the 6-value `_VALID_OUTCOMES` set. Returns `{"success": False, "error": ...}` on bad input.
- `binding.py` candidate-outcome emitter: skips emit unless `success=True` AND both `candidate_id` and `outcome` are present.
- System prompt IMPROVEMENT_LOOP section: rewrote with anti-decline-bias framing — "Success target is at least one PROMOTED candidate per run, not 'every candidate declined for cost reasons'. If all candidates look too big, IMPLEMENT A SCOPED-DOWN SUBSET."
- System prompt also explicitly: "`candidate_id` MUST be the exact `id` from the most recent `propose_improvements` result (e.g. `path_1`, `path_2`)."
- `run_experiment` aggregate cap: 7200 s → 1800 s (env-var tunable).
- `leaderboard.py:_read_run`: `_as_dict(v)` defensive coercer applied to `paper`/`rubric`/`cost`/`models` — single malformed row no longer 500s the whole endpoint.

### Fixed (late evening)
- **"Stuck Running" bug** (`prj_6b9acbfd8afcd789`): runs that produced `final_report.json` cleanly were stuck on `status=running` forever because atexit cleanup hung in WSL2 subprocess.wait. Fixed via wrapper reorder + `os._exit` escape hatch.
- **"Database is locked" on parallel ingest**: triggering two `/runs/arxiv` calls within seconds killed the second at first SQLite write. Fixed via `BEGIN IMMEDIATE` + 30 s `busy_timeout`.
- **`candidate_id="None"` wire-contract bug**: every `candidate_outcome` SSE event had the literal string `"None"` as candidate_id because `str(None)` was emitted unconditionally. Fixed via 3-layer defense (primitive validation + binding skip-on-failure + system_prompt clarification).
- **`/leaderboard` 500 on legacy `final_report` shapes**: 3 test fixtures with list-shaped `rubric` killed the entire endpoint. Fixed via `_as_dict(v)` at the read boundary.
- **B2 wedged on CPU-bound VLM training**: would have waited the full 2 h aggregate cap before iterating. Default reduced to 30 min; env var lets long experiments extend.

### Documentation (late evening)
- `uiprogress.md`: new "late evening" entry with 6-commit table, paper-sweep result table (A1/A2/B1/B2 with rubric + promoted counts), F18–F23 failure-mode rules, manual-intervention runbook for container-namespace PID kills.
- `learn.md`: 6 new postmortems with full Symptom/Root cause/Fix/Lesson/Guardrail entries (stuck-Running, SQLite locked, candidate_id wire bug, anti-decline prompt, leaderboard 500, run_experiment cap).
- This `CHANGELOG.md` section.

### Validated (late evening)
- **First E2E run to hit the "promoted candidate" success gate**: B1 (arXiv 2602.01785 CodeOCR) — 6 iterations, rubric 31.88%, `outcome=promoted` on `path_1`. 3× higher rubric than A1/A2 under the old prompt; first run to actually try a scoped-down subset instead of declining everything. Confirms the full pipeline (stuck-Running fix + SQLite + candidate_id + anti-decline prompt) works end-to-end.
- **729 tests passing, 1 xfailed** across `tests/rlm/ tests/rdr/ tests/routes/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite{,_concurrent}.py` after all 6 commits.

### Fixed (ship-readiness)
- Gated remaining mutating backend routes, including run stop, chat steering, and Phase 2 service POSTs, and added JSON handling for uncaught route errors.
- RDR metricless runs now call the scorer in degraded mode and write `mode`, `models`, `started_at`, `completed_at`, and degraded metadata into final reports.
- RDR SSE now emits the spec cluster events (`cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`) alongside existing `rdr_*` lifecycle events.
- Threaded `RunBudget.max_pod_seconds` through CLI PaperBench paths, live-run subprocesses, RDR, and hybrid Phase 1/Phase 2 budgets.
- Frontend live-run state preserves terminal/chat/candidate/RDR cluster events across long streams and falls back to polling after SSE errors.
- Library, leaderboard, and recent-run surfaces now show explicit backend-outage states instead of silent empty success states.
- Removed stale committed handoff docs so the tracked docs inventory is back at the launch-readiness limit of 14 Markdown files.

### Added (ship-readiness)
- Vendored PaperBench bundle identity guard covering `ftrl`, `mechanistic-understanding`, and `sequential-neural-score-estimation`.

## [2026-05-23]

### Added
- `heartbeat()` REPL primitive + `iteration_heartbeat` SSE event; root calls before long ops; `rlm-header` shows amber "no signal Ns" chip when stale >60s.
- Stderr watchdog (`live_runs.py:_stderr_watchdog`): detects `aclose()` async-generator loop ≥3× in 30s; sets `degraded: True` on `demo_status.json` + emits `run_warning` SSE; never kills subprocess.
- `recommend_next_tool(situation)` REPL primitive — Reflexion-lite advisor; one bounded `llm_query` call → `{tool, reason, alternatives}`.
- Azure OpenAI provider: `azure-gpt-4o` registry entry + aliases (`azure`, `azure-openai`, `gpt-4o-azure`); `AzureOpenAILlmClient` wrapper; `factory.py:_has_azure_openai_credentials`; env vars `AZURE_OPENAI_{API_KEY,ENDPOINT,DEPLOYMENT}`.
- Chat steering: `POST /runs/<id>/messages` endpoint + `check_user_messages` / `respond_to_user` primitives + frontend `SteeringChat` panel docked in sidebar.
- Lab UI: right-docked `NodeDetailSidebar` (replaces floating popup); aggregate counter strip; enriched panels per node kind; preset arXiv chip row in upload view; Upload nav item; `useResizablePanels` hook + `ResizeHandle` for drag-resizable rails.
- `uiprogress.md` regression-prevention log (append-only, F1–F14 rules).

### Changed
- 12 → 14 REPL primitives; pin tests bumped (`test_run.py`, `test_integration_custom_tools.py`, `test_registry.py`).
- System prompt: new Heartbeat, Chat Steering, Decision Advisor, `rlm_query`-preference sections.
- `useRlmRun.foldPrimitiveCall` flips status `queued→running` on first primitive (was only on first `repl_iteration`).
- Real-time elapsed clock derives from `runMeta.startedAt` + ticking interval (was event-span; ticked nothing when wedged).
- RDR polling proxies (clusters/leaf-scores/repair-iterations): 4s `AbortController` + normalize timeout/5xx → 404; `useRdrArtifacts` counts empty 200 as missing.
- `understand_section` + `extract_hyperparameters` return `_meta.hint` on slices >10K chars; `[hint] ` prefix flows through `result_summary`; sidebar shows amber-dot indicator.

### Fixed
- React duplicate `key=""` in `rubric-strip` + `report-rail` — `_rubric_areas` now emits `"area"` (matches `binding.py`); frontend keys carry `__idx_${i}` fallback.
- SSR hydration mismatch on elapsed tile — `nowMs=null` initial state matches server + first-client render; `useEffect` populates on mount.
- `RLMFinalReport.paper_claims` rejected list-shaped root returns → 30-min runs crashed at final-report step; `@field_validator(mode="before")` coerces list → dict keyed by `method`/`claim`/`id`/`name`/`claim_{i}`.
- `wrap_primitive` validation: one conservative coercion pass (int/float/bool → str, digit-str → int; never dict) before bubbling; saves one root iteration per type-mismatch.

### Documentation
- `CLAUDE.md`: Azure auth bullet, primitive count 12→14, new event types, Chat Steering + Collapsible Sidebar sections.
- `system_overview.md`: matching primitive count + SSE event updates; Chat Steering + Sidebar sections.
- `docs/runbooks/e2e-testing.md`: §4f Backend hangs (SDK aclose remedy), §6b chat-steering walkthrough, F7+F8 commit-table entries.
- `uiprogress.md`: two top-level entries (morning + afternoon sprints).

---

### Added
- **`rdr` — rubric-driven paper-reproduction harness (`--mode rdr`).**
  A deterministic Python controller (`backend/agents/rdr/`) decomposes the
  official PaperBench rubric into agent-sized work-clusters and dispatches
  one scoped Claude coding agent per cluster, each with a precisely-engineered
  context window (verbatim leaf requirements + cited paper excerpts +
  dependency artifacts + repair feedback). The cluster outputs are assembled
  into a shared project, run through `run_experiment`, scored against the
  exact rubric by the existing leaf scorer, and weak clusters are re-attacked
  in a capped repair loop fed the scorer's own justifications. **No LLM in
  the control flow** — the wander/loop failure mode of the free-form RLM root
  is structurally impossible.
  - New package `backend/agents/rdr/` (`models`, `decomposer`,
    `context_engineer`, `agent`, `controller`, `run`); new CLI mode
    `reproduce --mode rdr <paper_id>`; new launcher
    `scripts/rdr_paperbench.py` (parallel to `scripts/rlm_paperbench.py`).
  - Provider/model is fully dynamic — the agent reuses the repo's
    `collect_agent_text` (the same SDK path `run_with_sdk` uses) and inherits
    the runtime resolution, so the same harness runs on Claude OAuth (Sonnet)
    locally OR Azure OpenAI without code changes.
  - Bounded everywhere: agent calls go through `asyncio.wait_for(timeout_s)`
    against `ctx.remaining_s()`; per-cluster fail-soft; deterministic
    termination — the controller assembles the report from structured
    artifacts, never an LLM "final answer."
  - All four #62 DC#4 artifacts: `final_report.{json,md}`,
    `iterations/cluster_*.json` per cluster, `repl_state.pickle`
    (corpus-redacted — file counts / notes only, never raw paper text).
    Verdict reconciled against the leaf score via the existing
    `reconcile_verdict_with_score`.
  - 112 rdr tests (decomposer, context engineer, agent contract, controller
    flow + repair + fail-soft, run.py bundle resolution, full offline e2e on
    the real `sequential-neural-score-estimation` bundle). Full suite green:
    1362 passed, 3 skipped.
  - Design spec: `docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`.
- **RLM Phase 4 — RLM lab frontend (feat/rlm-phase4-frontend, #61).** Ships the
  RLM lab UI: a live, branching **exploration-tree canvas** (the centerpiece) wrapped
  in a lab-notebook shell — rich header (paper metadata, project id, status pill,
  iter/cost), rubric score + climb strip (baseline→target bar, Δ sparkline), REPL-state
  rail (variable manifest + primitive list, collapsible), live report rail
  (verdict, stat grid, rubric breakdown, collapsible), and a primitive-call history
  bar. `useRlmRun` is a pure `fold`-based reducer that folds the SSE `dashboard_event`
  stream into `RlmRunState` (tree, rubric series, variable manifest, report).
  `ExplorationCanvas` renders a laid-out node graph via a pure `layoutTree` function
  with pan/zoom, declined-collapse, soft cap, `NodeDetailPopup`, and live pulse
  animations. Built **fixture-first** against a hand-authored 14-iteration recorded
  events fixture (`rlm-run.fixture.ts`); coexists with the old 14-stage pipeline UI
  via a `runMode === "rlm"` branch in `WorkflowView` (the old UI is untouched).
  `replay.ts` + a `?rlmFixture=1` query param on `/lab` drive the fixture in dev/test.
  A Playwright e2e (`e2e/rlm-lab.spec.ts`) drives the full fixture path end-to-end.
  The 3 candidate/rubric event types (`candidate_proposed`, `candidate_outcome`,
  `rubric_score`) are fixture-contract + a documented backend handoff
  in the retired Phase 4 handoff docs), not yet backend-emitted.
- **Dynamic best-source paper ingestion — HTML > PDF > OCR cascade.**
  Figure-heavy arXiv papers produce figure-label noise when PDF-parsed; arXiv's
  LaTeXML HTML rendering is clean (figures are images there). `ResolvingParser`
  (`backend/services/ingestion/parser/`) quality-scores HTML and PDF on every
  run and picks the highest — HTML wins ties. OCR (`OcrPaperParser`, tesseract)
  runs only when both score below the usable threshold. The HTML sibling is
  written by `ArxivFetcher`, which now fail-soft fetches `arxiv.org/html/<id>`
  with structural-marker validation to reject interstitials; the fetch never
  fails the run. `cli.py` wires `ResolvingParser` as the ingestion entry point.
  New deps: `beautifulsoup4`, `Pillow`, `pytesseract`.
- **arXiv RLM runs — self-generated rubric + REST-retrievable status.**
  When an arXiv `--mode rlm` run has no vendored bundle rubric,
  `backend/agents/rlm/rubric_gen.py::generate_rubric_tree` derives a
  PaperBench-shaped weighted rubric tree from the paper text (six standard
  categories, paper-specific leaf criteria) and persists it to
  `runs/<id>/generated_rubric.json`. `score_run.py` finds it automatically;
  `leaf_scorer` labels the result `rubric_source="generated"` — explicitly not
  PaperBench-official. `run_pipeline_rlm` also writes `demo_status.json` at
  start and terminal states, so `GET /runs/{id}` resolves for CLI- and
  script-launched RLM runs.
- **RLM context sources clean full paper text (not the truncated workspace variable).**
  `_build_workspace_claim_map` now reads `parsed_full_text.txt` — the parser's
  direct, complete output — for RLM mode, falling back to the workspace variable
  only when the file is absent. The chunk-reassembly round-trip that the SDK
  retrieval layer needs (and that loses content on some papers) is bypassed; RLM
  offloads the paper whole.
- **Honest leaf-scoring output — `final_report.md` re-rendered on amend.**
  `score_run.py` / `leaf_scorer.amend_final_report` now re-renders
  `final_report.md` (not only the JSON), so `GET /runs/{id}/final-report` serves
  the authoritative leaf score. The markdown surface rubric provenance: a
  generated-rubric score is labelled "self-generated rubric — not
  PaperBench-official"; a bundle score "PaperBench bundle rubric".

- **RLM Featherless root-model backend + hardened key resolution (feat/rlm-phase5-e2e).**
  Adds `qwen3-coder-featherless` to the root-model registry — `Qwen/Qwen3-Coder-480B-A35B-Instruct`
  (the paper-validated RLM root) served via Featherless's OpenAI-compatible endpoint, so an RLM
  run needs no OpenAI/Anthropic key. A new `RootModel.api_key_env` field decouples the API-key
  env var from the backend type (`_env_var_for` helper), letting an `openai`-typed backend
  authenticate with `FEATHERLESS_API_KEY`; `_build_llm_client` mirrors a custom endpoint so the
  primitive LLM client shares the root's host and key. `resolve_root_model` now fails fast —
  with an actionable `ValueError` — when *any* backend's API key is missing (previously only
  OpenRouter was checked; a missing key otherwise surfaced as a cryptic `TypeError` deep inside
  `rlm`).
- **RLM Phase 5 — production-hardened end-to-end run mode (WS-H + WS-B, feat/rlm-phase5-e2e).**
  Completes the #59 (primitive layer) ↔ #60 (orchestrator) merge and makes `--mode rlm` safe
  for real reproduction runs. Four disjoint hardening batches landed on top of the merge:

  - **Per-primitive deadlines (M-DEADLINE / Batch P).** `RunContext` gained `deadline_utc:
    datetime | None` (set once from the wall-clock budget at run start) and `remaining_s()`.
    A `run_with_deadline(coro, ctx, cap_s)` helper wraps a primitive's async body in
    `asyncio.wait_for(min(cap_s, remaining))` and, on `TimeoutError`, runs the primitive's
    teardown (sandbox `destroy`) before returning a fail-soft error dict. The three long
    primitives — `build_environment` (3 × 1800 s repair + aggregate cap), `implement_baseline`
    (previously blocked forever on `pool.submit(...).result()`), and `run_experiment`
    (N commands × 3600 s) — all route through it. `rlms`'s `max_timeout` only fires between
    iterations; a primitive wedged in `execute_code` overruns it; the per-primitive deadline
    is the only real enforcement.

  - **`max_usd` cost cap (M-BUDGET / Batch O).** The `RLMLogger.log()` callback now compares
    `cost_ledger.total_usd()` to `run_budget.max_usd` between iterations; on breach it
    requests run stop and `_finalize` marks `status="failed"` (→ non-zero CLI exit). Cost
    reporting drops the always-zero `sub_usd` field for an honest single `llm_usd`. A
    budget-exhausted run no longer exits 0.

  - **Corpus-leak redaction at every egress (M-REDACT / Batch O).** Algorithm-2's invariant
    — the offloaded paper corpus must never reach a durable or streamed surface — previously
    held only inside `sanitize_iteration`. The fix applies a `redact_corpus(text, sentinels)`
    helper (sentinel = first 200 chars of each `context` corpus value) at every remaining
    egress: `sse_bridge` stdout/stderr prefixes and `report.py`'s final-report strings.

  - **Run-status integrity (Batch L).** Non-atomic `demo_status.json` writes in `live_runs.py`
    (a crash bricks the UI) are now tempfile + `os.replace` on every code path. The watchdog
    `os._exit` now writes a terminal `status=failed` and a terminal SSE frame before exiting.
    `stop_run` escalates from SIGTERM to SIGKILL after 10 s so a wedged run cannot live
    forever.

  - **RunPod backend hardening (Batch R).** `gpu_mode` and `network_disabled` flags are now
    honoured. HTTP 401/403 from the RunPod API is classified `retryable=False` so the retry
    loop does not spin forever on auth failures. Pod-death detection in the SSH wait avoids
    a 900 s spin. Cleanup is `asyncio.shield`-ed from cancellation. SSH host-key handling
    and idempotent destroy are in place.

  - **Real PaperBench bundles vendored (WS-B).** Replaced the placeholder `third_party/
    paperbench/ftrl/` with the real upstream artifacts (`paper.md`, `addendum.md`,
    `rubric.json`) from `openai/preparedness`. Added two genuinely-easy companion papers
    (`sequential-neural-score-estimation`, `mechanistic-understanding`) so rubric scores are
    honestly comparable to PaperBench's published baselines. PaperBench's `paper.md` /
    `addendum.md` are stored in Git LFS on `openai/preparedness`; raw.githubusercontent.com
    returns a pointer — artifacts are fetched via the LFS batch API (the LFS store on
    `openai/preparedness` redirects to `openai/frontier-evals`).

- **RLM orchestrator — `--mode rlm`, Phase 3 of the RLM pivot (#60).** A new run
  mode that replaces the 14-stage `PipelineStage` machine with the `rlms`
  library's Algorithm-1 loop: the root model writes REPL code that calls domain
  primitives to reproduce a paper. New `backend/agents/rlm/` modules — `run.py`
  (`run_pipeline_rlm`: builds the `rlm.RLM`, runs `.completion()` on a worker
  thread, writes `final_report.{json,md}`), `system_prompt.py`, `models.py`
  (root-model registry: GPT-5 / Qwen3-Coder / Kimi K2.5 / Claude),
  `sse_bridge.py` (a corpus sanitizer + an `RLMLogger` subclass streaming
  `repl_iteration` / `sub_rlm_*` / `run_complete` events), `checkpoint.py`
  (per-iteration `RLMRunIteration` events → the SQLite event store + a sanitized
  snapshot), `report.py`, and `stub_primitives.py`. Wired through `cli.py` /
  `pipeline.py` / `live_runs.py`; later releases replaced the legacy modes with
  the current `--mode {rlm,rdr,rlm-pure}` surface. Decoupled from the #59 primitive layer — `run.py` lazily
  resolves `build_custom_tools` and falls back to a stub provider, so the
  orchestrator runs and is integration-tested before #59 lands. The paper is
  offloaded as the `context` REPL variable; the `sanitize_iteration` chokepoint
  keeps it out of every event, log, and snapshot (Algorithm-2 guard).
- **Dockerfile smoke-import layer — closes the build/runtime visibility gap.**
  Track 4 validates `docker build` succeeds, but build-success isn't
  runtime-success: packages can install cleanly and still fail on first
  `import` (transitive deps that aren't declared in setup.py — e.g.
  `gymnasium[mujoco]` needing `imageio`, observed live on the demo paper).
  The `environment-detective` prompts now require the FINAL Dockerfile layer
  to be a no-network `RUN python -c '<smoke>'` step that imports every
  declared framework and lightly instantiates the paper's primary entity
  exactly as the experiment will use it (`gym.make('<env_id>')`, model
  construction, tokenizer load from local-only path). A failure in that
  layer surfaces as a build error → **Track 4's existing repair loop fires
  → env-detective adds the missing dep → rebuild**. Zero new orchestrator
  code; reuses the build-and-repair loop as the engine. Catches the
  import-time class of runtime failures at the right stage instead of dying
  at `baseline_run` minutes later. Reflected in both
  `ENVIRONMENT_DETECTIVE_PROMPT` (base) and `ENVIRONMENT_DETECTIVE_REPAIR_PROMPT`
  (repair) so a repair round that touches the smoke layer keeps it intact.
- **Sandbox execution contract — single source of truth for the agent↔runtime
  interface.** The reproduction sandbox mounts the project read-only at the
  container working dir and a separate writable volume at `$OUTPUT_DIR` — but
  this contract used to live only in the runtime code and the env-var names,
  not in the agent prompts. So `baseline-implementation` and `improvement-path`
  routinely wrote scripts that `mkdir results/` and `tee > log.txt` in the
  read-only mount, and the experiment died on a "Read-only file system" error
  before any code ran. A new `backend/agents/prompts/_sandbox_contract.py`
  defines `SANDBOX_EXECUTION_CONTRACT` — a single brace-free, domain-agnostic
  block stating the mount model, the env vars, and the required write
  patterns (every output under `$OUTPUT_DIR`; cache-hungry tools redirected via
  `HF_HOME`/`TRANSFORMERS_CACHE`/`TRITON_CACHE_DIR`/etc.; `metrics.json` under
  `$OUTPUT_DIR/metrics.json`). The contract is spliced into the three prompts
  that emit sandbox-executable code (`BASELINE_IMPLEMENTATION_PROMPT`,
  `IMPROVEMENT_PATH_PROMPT`, `COMPOSITION_AGENT_PROMPT`) right before each
  `# Output` section — at peak attention as the agent commits to its
  `commands_to_run`. Identical across the docker, local, and runpod sandbox
  modes — write to the contract, it works on all three.
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
  `OpenResearchOrchestrator` now loops back through improvement-selection + Gate 3
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
  view. A per-browser `localStorage` pointer (`openresearch:lastRun`)
  auto-resumes an in-flight run when the tab is reopened on a bare
  `/lab` URL; a stale link, a deleted run, and a transient 504 are each
  handled without discarding a live run. New `lib/demo/server-run.ts`
  holds the shared server-only run-fetch helper.
- **Hybrid vision-leaning paper extraction.** A modular `PaperExtractor`
  augmentation pass runs after the base PyMuPDF parser
  (`backend/services/ingestion/parser/extractor.py`). In `hybrid` mode
  (`OPENRESEARCH_PAPER_EXTRACTION_MODE`, default `hybrid`) it renders
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
  builder agents listed in `OPENRESEARCH_APIFY_ARXIV_ENABLED_AGENTS`
  (default: `artifact-discovery,paper-understanding`). Tools surface as
  `mcp__apify-arxiv__*` and give those agents direct paper search /
  access / listing against arXiv without round-tripping through generic
  WebSearch. Empty token = MCP wiring is skipped entirely (no cold-start
  handshake, no extra latency).
- **Persistent Runpod pod reuse via `OPENRESEARCH_RUNPOD_POD_ID`.** When set,
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
  Optional overrides: `OPENRESEARCH_CODEX_CLI_PATH` and
  `OPENRESEARCH_CODEX_AUTH_PATH`. Verified locally with `codex-cli 0.125.0`.
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
  starts with `openresearch-` before issuing the API DELETE — so a future
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
  `openresearch paperbench {list,summary,run,status}` CLI subcommand. Default
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
- **PaperBench section in `docs/guides/setup-guide.md`** — 4-step workflow
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
- **`rdr` — production-hardened against the Claude SDK `aclose()` deadlock + 6 Codex-surfaced bugs from the first live e2e runs.** The first live runs of `--mode rdr` on `sequential-neural-score-estimation` and `mechanistic-understanding` surfaced one wedge and six smaller bugs. Six commits on top of the harness (`ee51f02`) brought the path to live-stable:
  - **`5b53a10` + `3e95752` — Snapshot exclusion + defensive merge writes.** `_snapshot_code_dir()` walked the entire code dir including `hf_cache`, `.locks`, `wandb`, etc. — restricted-perm lock files written inside the experiment container caused `PermissionError` on the repair-pass rewrite. Added `_EXCLUDE_DIRS` frozenset + hidden-component skip + `try/except OSError` on the snapshot read; also removed a stale `tsnpe_neurips` submodule reference exposed by the cleanup.
  - **`96b0b63` — Scorer + env primitives off the event loop.** `score_reproduction()` internally calls `ClaudeLlmClient.complete()` which calls `asyncio.run(...)`. Invoking it from inside the controller's already-running loop raised `RuntimeError: asyncio.run() cannot be called from a running event loop`. Wrapped via `await asyncio.to_thread(score_reproduction, ...)` at both call sites; same treatment applied to `detect_environment`, `build_environment`, and `run_experiment`.
  - **`3a73903` — Controller watchdog + agent CWD guard.** `_ClusterWatchdog` (`threading.Timer` per cluster) ceilings each cluster at 900s of no progress, writing an emergency `final_report.json` + SSE `watchdog_killed` event before `os._exit(124)`. CWD guard: `_snapshot_repo_root_entries()` before and after each cluster — if the agent escaped its `code_dir` via `git clone <stuff>` etc., the new entries are detected and `shutil.rmtree`-cleaned (caught the agent leaking a `tsnpe_neurips` repo at repo root). Later, the `os.chdir(code_dir)` was removed entirely in favor of the SDK's `ClaudeAgentOptions(cwd=code_dir)` (process-global `chdir` is unsafe under concurrency).
  - **`c928fb7` — Codex adversarial review sweep (2 CRITICAL + 4 IMPORTANT).** CRITICAL: scorer exception propagation (no `try/except` — a leaf-scorer crash crashed the whole run), path traversal in merge-write (`Artifacts.files` paths weren't `is_relative_to`-validated before write). IMPORTANT: event-loop blocking primitives, `os.chdir` process-global, watchdog missing emergency report, CLI args unregistered. All addressed.
  - **`06d0087` — Retry-on-watchdog + `dashboard_event` SSE emission.** `scripts/rdr_paperbench_retry.sh` wraps `rdr_paperbench.py` and re-invokes with `--resume` when the exit code is 124 (watchdog kill), up to `RDR_MAX_RETRIES` (default 3). Resume hydrates `done` from `iterations/cluster_*.json` checkpoints so completed clusters are not re-executed. New `DashboardEmitter.emit(event_type, payload)` generic method + 11 lifecycle events emitted by the controller — UI parity with `--mode rlm`.
- **`4ac89f7` + **`33c787d`** — SDK thread isolation (Workaround B + non-blocking executor).** The wedge at cluster 23 was root-caused (see `learn.md` 2026-05-22) to two compounding defects in `claude-agent-sdk` v0.1.80: (1) `asyncio.shutdown_asyncgens()` concurrently closing three nested async generators raises `RuntimeError: aclose(): asynchronous generator is already running`; (2) `transport.close()` does unbounded `await process.wait()` after SIGKILL which hangs in `futex_wait_queue` on WSL2 when SIGCHLD is lost. Workaround B: `_run_sdk_in_thread()` in `backend/agents/rdr/agent.py` wraps each `collect_agent_text(...)` call in a `concurrent.futures.ThreadPoolExecutor(max_workers=1)` worker that runs the call inside its own `asyncio.run(asyncio.wait_for(...))`, so the SDK's shutdown race is loop-isolated. The wrapper uses explicit `try/finally` + `ex.shutdown(wait=False)` instead of `with ThreadPoolExecutor(...) as ex:` — `__exit__` would call `shutdown(wait=True)`, blocking the controller on a hung worker. `concurrent.futures.TimeoutError` is re-raised as builtin `TimeoutError` so the existing fail-soft path in `reproduce()` is unchanged. The process-level watchdog remains a defense-in-depth net. Five new tests (`tests/rdr/test_agent_thread_isolation.py`), full suite: 1416 passed, 3 skipped.
- **Workspace `paper_text` now equals the parser's full text (I4).** It was
  reassembled from indexed chunks — lossy, and degraded or empty for some
  papers. `build_workspace` now loads `paper_text` from the parser's full-text
  blob (located via the `ParsingCompleted` event's `full_text_blob_path`),
  falling back to chunk-reassembly only when the blob is unavailable.
  `backend/services/context/workspace/service.py`; guard
  `tests/test_issue16_workspace_service.py::test_paper_text_equals_parser_full_text`.
- **`run_experiment` failed in 6 s with empty logs — three compounding bugs
  (`backend/agents/rlm/primitives.py`).** (A) `_execute_in_sandbox` built
  `logs` from `r.stdout` only; a failed command's traceback is on stderr, so
  every failure landed `logs=""` — undiagnosable, and the RLM repair loop got
  an empty `repair_context`. `_combine_command_output` now joins both streams.
  (B) the experiment ran the image `detect_environment` built before any code
  existed (missing `transformers`); `run_experiment` now rebuilds from
  `ctx.project_dir/Dockerfile` via `build_environment` — content-addressed and
  Docker-cached, so an unchanged Dockerfile is a near-instant no-op. (C) the
  experiment sandbox ran `network_disabled`, blocking HuggingFace/PyPI
  downloads; `_execute_in_sandbox` now enables network for the experiment
  container (the paper corpus is never mounted there). See `learn.md`
  2026-05-22; guard `tests/rlm/test_run_experiment_env.py`.
- **`plan_reproduction` no longer fail-softs on list-valued plan fields.**
  `ReproductionContract`'s plan fields accept `str | list[str]` — the root LLM
  routinely returns lists, which previously dropped the whole contract to
  near-empty defaults.
- **`repl_snapshot` no longer clobbers `repl_state.pickle` on a no-code
  iteration.** A pure-reasoning RLM iteration (no code blocks) now preserves the
  prior REPL pickle instead of overwriting it empty.
- **RLM `implement_baseline` survives the Claude OAuth quota; serves API + dev
  modes (merge).** The RLM sub-agent runtime (`_resolve_agent_runtime`) now
  resolves to Claude with SDK-resolved auth — `ANTHROPIC_API_KEY` in production,
  the Claude Code subscription's OAuth login in dev — and threads
  `settings.anthropic_default_model` (Sonnet) through `RunContext.agent_model`
  as the `to_runtime_spec` `model_override`. Without the override the
  `baseline-implementation` agent ran the registry's Opus default and exhausted
  the OAuth quota (`Claude Code returned an error result: success`), finishing a
  run `failed` with no code written. The fallback now validates credentials at
  resolution time (`require_api_key=True`).
- **RLM `PaperClaimMap` tolerates the loosely-shaped dicts an LLM root emits
  (merge).** `claims` / `datasets` / `metrics` accept bare strings (coerced to
  single-key dicts) and `MetricSpec.definition` defaults to `""`. One canonical
  `_coerce_str_items` validator handles all three; it transforms only bare
  strings and passes dicts and pre-built `DatasetRequirement`/`MetricSpec`
  instances through untouched (an earlier filter dropped instances, emptying
  the lists). `PRIMITIVE_DESCRIPTIONS` now publish the `PaperClaimMap` shape.
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
- **`database disk image is malformed`** on `openresearch.db` after a killed
  pipeline subprocess. Restored from offline backup; `synchronous=FULL`
  prevents recurrence — see `learn.md` 2026-05-09.

### Documentation
- `docs/guides/setup-guide.md` — setup, PaperBench workflow, and Phase 2 service attach points.
- `learn.md` — durable post-mortem + practice runbook.
- `third_party/paperbench/README.md` — instructions for swapping the
  FTRL placeholder bundle for the upstream PaperBench artifacts.
- `CHANGELOG.md` (this file).

### Tooling
- `.gitignore` exclusions for build/test artifacts:
  `frontend/tsconfig.tsbuildinfo`, `_test_logs/`, `openresearch.db.*`,
  Windows `*:Zone.Identifier`, stray pip-version files at repo root.
  `paperbench1.pdf` is whitelisted as the canonical input fixture.
- `tests/test_issue16_workspace_service.py::test_build_workspace_auto_embeds_chunks`
  now uses `pytest.importorskip("chromadb")` so the suite stays green
  when the optional `openresearch-backend[semantic]` extras aren't
  installed (matches the pattern in `test_semantic_layer2.py`).
