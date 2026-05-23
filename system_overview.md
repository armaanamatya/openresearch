# System Overview — OpenResearch / ReproLab

> Orientation for new Claude Code / Codex sessions: the non-obvious "why" and
> "how it fits together." For *what's where*, read the code — it's named by
> function. Keep this current.

## Goal

Given a paper (arXiv link or uploaded PDF), ReproLab ingests it, understands the
claimed results, builds an environment, implements and runs a baseline, scores
the reproduction against a PaperBench-style rubric, explores improvements, and
emits a benchmark report comparing the reproduction to the paper's claims.

RLM is the *paradigm the project is built on*, not a paper ReproLab reproduces.
ReproLab reproduces other papers; the `rlms` library is its substrate.

## Architecture

ReproLab is built on the **Recursive Language Model** paradigm (arXiv 2512.24601,
Zhang/Kraska/Khattab, MIT CSAIL). The `rlms` library (`pip install rlms`) is the
engine; our code is the domain layer. The RLM root model never receives the paper
text — it is offloaded as a REPL `context` variable and the model accesses it
programmatically via slices and recursive sub-calls. Twelve domain primitives
(`understand_section`, `extract_hyperparameters`, `detect_environment`,
`build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`,
`verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`,
`check_user_messages`, `respond_to_user`) are exposed as REPL callables in
`backend/agents/rlm/primitives.py`. The root decides what to call and in what
order by writing Python — there is no fixed stage order and no gate control-flow.
The last two primitives are the **chat-steering surface** (see below) and are
pure file I/O, so they work identically under API-key and OAuth auth.

See `docs/design/rlm-pivot-brief.md` for the full design rationale, the
RLM-fidelity invariants, and the primitive contract.

## How it fits together

One Docker image, two processes (`docker/entrypoint.sh` under `tini`):

- **Backend** — FastAPI (`backend/app.py`), internal `:8000`. Each run is a
  **long-lived subprocess** it spawns and tracks.
- **Frontend** — Next.js 16 (`frontend/`), public `:$PORT`. The "lab" UI;
  reaches the backend **server-side only** via `/api/demo/*` proxy routes (no
  CORS — the browser never talks to the backend directly).

Run state is **file-backed**, not a service — `runs/<project_id>/` holds the
status snapshot (`demo_status.json`), per-iteration checkpoints (`rlm_state/`),
`final_report.{json,md}`, `dashboard_events.jsonl`, `cost_ledger.jsonl`,
`experiment_runs.jsonl`, the reproduced `code/`, and Hermes audit artifacts.
SQLite (`REPROLAB_DATABASE_URL`) is the event/persistence store.

## Paper ingestion

`ResolvingParser` (`backend/services/ingestion/parser/resolving_parser.py`)
quality-scores every available source and picks the cleanest one: arXiv's
LaTeXML HTML rendering (written by `ArxivFetcher` as `raw_paper.html`,
fail-soft) outscores PDF on figure-heavy papers because figures are images in
HTML and become label-noise in PDF. `PyMuPdfParser` is the default when HTML is
absent or low-quality; tesseract OCR runs only when both score below the usable
threshold. The winning parse is written to `parsed_full_text.txt`, the run's
canonical full-text artifact — what the RLM root model is seeded from.

## Rubric-driven harness (`--mode rdr`, 2026-05)

The default CLI mode, `--mode rlm`, is hybrid as of PR #80: Phase 1 runs the
RDR controller without repair, then Phase 2 launches RLM adaptive repair only
when weak rubric clusters remain. `--mode rdr` keeps the pure controller path,
and `--mode rlm-pure` keeps the pre-hybrid root-loop escape hatch.

`--mode rdr` is a parallel reproduction path that **makes the official
PaperBench rubric the spine of the run** instead of grading only at the end.
A deterministic Python controller (`backend/agents/rdr/controller.py`)
decomposes the rubric tree into agent-sized work-clusters and dispatches
**one scoped Claude coding agent per cluster**, each with a precisely-engineered
context window (verbatim leaf requirements + cited paper excerpts + dependency
artifacts from prior clusters + repair feedback). Every rubric leaf is a
controller obligation: attempted, scored, and repaired-if-weak in a capped loop
fed the leaf scorer's own justifications.

There is no LLM in the control flow, so the free-form RLM root's wander/loop
failure mode (run 3 / GoRL: 21 root iterations stuck in `understand_section`)
is structurally impossible. The orchestration **reuses existing infrastructure**:
`collect_agent_text` for the Claude SDK agent (the same path `run_with_sdk` uses)
so the agent is **provider-portable by construction** — Claude OAuth (Sonnet)
locally, Azure OpenAI as an alternative — and `run_experiment` /
`score_reproduction` / `write_final_report_rlm` / `reconcile_verdict_with_score`
for environment, scoring, reporting.

`rdr` targets PaperBench bundle papers (the official rubric is required); arXiv
papers can use it secondarily with a generated rubric. The design spec is
`docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`; the package
is `backend/agents/rdr/` (`models`, `decomposer`, `context_engineer`, `agent`,
`controller`, `run`); the launcher is `scripts/rdr_paperbench.py`.

## Run lifecycle (UI ↔ backend)

1. RLM lab UI (`frontend/src/components/lab/rlm/`) starts a run — arXiv link or
   PDF → `POST /api/demo` → backend `POST /runs` / `/runs/upload` / `/runs/arxiv`.
2. Backend spawns the run subprocess, writes `demo_status.json`, returns state.
3. UI opens an **SSE** stream (`/api/demo/events` → backend `/runs/<id>/events`).
4. SSE event types: `repl_iteration`, `primitive_call`, `sub_rlm_spawned`,
   `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`,
   `rubric_score`, `user_message`, `user_message_response`, `run_warning`, and
   `iteration_heartbeat`. RDR runs also emit `rdr_*` lifecycle events and the
   spec-named cluster events `cluster_started`, `cluster_artifact_emitted`,
   `cluster_scored`, and `repair_dispatched`. All RLM iteration events route
   through `sse_bridge.sanitize_iteration` — the single egress chokepoint that
   strips REPL locals and bounds stdout/stderr to metadata prefixes. The paper
   corpus never reaches the stream.
5. On completion the computed `final_report.{json,md}` replaces the placeholder
   benchmark.

**Which run is current is the URL** — `/lab?projectId=<id>` — so a refresh or
a shared link reopens it. A `localStorage` pointer auto-resumes an in-flight run
when the user lands on a bare `/lab`.

## Chat steering surface (2026-05-23)

A bidirectional channel between the user and the running RLM:

- **User → root**: lab UI POSTs through `/api/demo/runs/<id>/messages` to backend
  `/runs/<id>/messages` (`backend/routes/messages.py`); each message is appended
  to `runs/<id>/user_messages.jsonl` and emits a `user_message` SSE event.
- **Root → user**: the system prompt instructs the root to call
  `check_user_messages()` at the start of each iteration. Unread `user`
  messages are surfaced; the root may reply via `respond_to_user(message)` —
  the reply is appended to the same JSONL and emits a `user_message_response`
  SSE event. A per-run cursor (`runs/<id>/_user_message_cursor.json`) tracks
  the read offset atomically.
- **UI**: the new right-docked `NodeDetailSidebar`
  (`frontend/src/components/lab/rlm/node-detail-sidebar.tsx`) hosts a
  `SteeringChat` panel that derives the conversation from the existing SSE
  stream filtered to the two new event types. Optimistic add + replace-on-echo.

Both primitives are auth-surface-agnostic (file I/O only) so the chat works
identically with `--model claude` (API key) and `--model claude-oauth`.

## Collapsible right sidebar (2026-05-23)

The lab's exploration tree carries a 360px right-docked detail sidebar that
replaces the old floating popup. Selection state is lifted to `rlm-lab.tsx`
so the canvas (highlight) and the sidebar (content) share one source of truth.
Content is kind-specific: `paper` → paperMeta as definition list; `work` →
filtered primitiveCalls (`understand_section`/`extract_hyperparameters` by
default, `detect_environment`/`build_environment` when `node.phase ==
"environment"`); `candidate` → category + description + rubricDelta + iteration
response; `subrlm`/`baseline`/`declined-group` → fall back to a "now" block.
Sidebar collapses to a 36px toggle rail. The `SteeringChat` (above) is docked
at the bottom of the expanded sidebar.

## Rubric

When no vendored PaperBench bundle exists, `backend/agents/rlm/rubric_gen.py`
derives a PaperBench-shaped rubric from the paper text and persists it as
`runs/<id>/generated_rubric.json`. Scores from a generated rubric carry
`rubric_source="generated"` and are labelled as non-PaperBench-official in both
the JSON and the re-rendered `final_report.md` served by `GET /runs/{id}/final-report`.

## Where to look

- **Backend** — `app.py` (HTTP), `cli.py` (CLI / non-UI runs),
  `agents/rlm/run.py` (RLM run entry), `agents/rlm/primitives.py` (domain
  primitives), `agents/rlm/sse_bridge.py` (egress chokepoint),
  `services/events/live_runs.py` (subprocess spawn + SSE bridge).
- **Frontend** — `app/lab/page.tsx` → `components/lab/rlm/` (lab UI),
  `app/api/demo/*` (backend proxy routes).
- **Leaderboard** — `backend/routes/leaderboard.py` (`GET /leaderboard`,
  filesystem-aggregated; no SQLite projection at this scale) and
  `frontend/src/app/leaderboard/` (page + sortable table). Read-only; not
  gated by the demo secret.
- **Rubric climb panel** — `frontend/src/components/lab/rlm/rubric-strip.tsx`
  is band 2 of the lab; carries the count-up tween + line-chart sparkline +
  per-area chip row with fail→pass flip highlights + candidate attribution.
  All new state (`previousAreas`, `attributableCandidate`) is derived in the
  `useRlmRun` reducer from existing SSE events — no new event types added.

## Docs

- `docs/design/rlm-pivot-brief.md` — the canonical architecture reference and design rationale.
- `docs/runbooks/e2e-testing.md` — canonical end-to-end testing and debug reference.
- `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` —
  locked launch decisions, including the hybrid default and leaderboard surface.
- `learn.md` — post-mortems: bugs shipped + the guardrail for each.
- `docs/guides/setup-guide.md`, `docs/guides/deployment.md`, `README.md` — setup
  and deployment. README.md also documents the two-surfaces LLM auth model
  and the empty-`ANTHROPIC_API_KEY` + Claude Code OAuth pattern preferred for
  local dev.

## Maintaining this doc

Orientation only — keep it at the "why / how it fits" altitude, never an
inventory of files. When a change makes a statement here wrong, fix it in the
same change.
