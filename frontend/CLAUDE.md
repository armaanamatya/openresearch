<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# frontend/CLAUDE.md

> Loaded when working in the Next.js UI. Root context: ../CLAUDE.md.

## Commands (Next.js 16, Node ≥20.19 <21 or ≥22.12)

```bash
cd frontend && npm ci
export OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000   # server-side proxy target; this IS the default — set only if the backend runs elsewhere
npm run dev          # http://localhost:3000
npm run build
npm run lint         # eslint .
npm test             # vitest run
npx tsc --noEmit     # type-check only
```

E2E tests use Playwright (`frontend/e2e/`); run via `npx playwright test` from `frontend/`.

## Server boundary — no CORS

The browser reaches the backend **server-side only**, through `/api/demo/*` proxy routes — there is no CORS layer, the browser never talks to the backend directly. In the shipped Docker image, `docker/entrypoint.sh` runs FastAPI on an internal `:8000` and Next.js on the public `:$PORT`; `OPENRESEARCH_BACKEND_URL` (above) is that same proxy target in dev. Debug UI-vs-API issues in `frontend/src/app/api/demo/`, not CORS.

## UI ↔ backend run lifecycle

1. RLM lab UI (`frontend/src/components/lab/rlm/`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload` / `/runs/arxiv`).
2. Backend spawns the run subprocess, writes `demo_status.json`, returns initial state.
3. UI opens **SSE** via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE event types:
   - RLM emits: `repl_iteration`, `primitive_call`, `sub_rlm_spawned`, `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`, `rubric_score`, `user_message`, `user_message_response`, `run_warning`, `iteration_heartbeat`, `repo_resolved`, `repo_cloned`.
   - RDR adds: `rdr_*`, `cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`.
   - Campaign adds: `campaign_started`, `attempt_started`, `attempt_assessed`, `campaign_decision`, `campaign_awaiting_operator`, `campaign_terminal`, `campaign_user_message`.
5. Iteration events route through `sse_bridge.sanitize_iteration` — the egress sanitizer that strips REPL locals and bounds stdout/stderr to metadata prefixes; corpus-derived fields pass through `redact_corpus`. Terminal/control events (`run_complete`/`run_fatal`/`run_interrupted`) carry no corpus and are emitted directly, so `sanitize_iteration` is the per-iteration sanitizer, not a literal single function every event object passes through. **The paper corpus never reaches the stream.**

A `localStorage` pointer auto-resumes an in-flight run on a bare `/lab`.

## UI surfaces (chat steering, sidebar, leaderboard)

- **Chat steering** — real-time panel docked in the right sidebar. `POST /runs/<id>/messages` (`backend/routes/messages.py`) appends to `user_messages.jsonl` + emits `user_message`; root polls `check_user_messages()`, replies via `respond_to_user`. Both pure file I/O. System prompt instructs the root to avoid quoting PII-looking message contents verbatim.
- **Collapsible right sidebar** — 360px `NodeDetailSidebar` (`frontend/src/components/lab/rlm/node-detail-sidebar.tsx`); selection state lifted to `rlm-lab.tsx`; kind-specific content (paper/work/candidate/subrlm/baseline); `SteeringChat` docked at bottom; collapses to a 36px rail.
- **Leaderboard + recent-runs panel** — read-only `/leaderboard` ranks runs; reachable from the left-sidebar nav and surfaced as a **recent-runs panel** atop the lab home (`frontend/src/components/lab/recent-runs-panel.tsx`, fed by `GET /leaderboard?order_by=finished_at&limit=`, then filtered to drop `interrupted` orphans + capped at 8). `GET /leaderboard?paper&mode&order_by&limit` (`backend/routes/leaderboard.py`) aggregates `final_report.json` + `demo_status.json` at request time (no SQLite projection, not demo-gated). Each project resolves its **best-scoring attempt** across top-level + `attempts/*` via `backend/services/runs/report_resolution.py` (`resolve_best_report`/`extract_scores` — normalizes nested `rubric.overall_score`/`compute_adjusted_score` + legacy flat top-level `rubric_score`; same extractor feeds run-detail `finalize_benchmark`); rows carry an honest `status` (stale `running`/`queued`→`completed` when a report exists) + `attempts` count, and `order_by=finished_at` is newest-first. The recent-runs panel rows carry Replay links (`?replay=<id>`) into the otherwise-orphaned replay surface (leaderboard rows link `/lab?projectId=`). Frontend: `frontend/src/app/leaderboard/`. Live rubric climb panel = enriched `RubricStrip` derived from existing SSE events.
