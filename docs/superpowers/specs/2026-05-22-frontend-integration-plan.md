# Frontend integration plan — rdr backend + dashboard wiring

_Date: 2026-05-22 · Scope: ONE combined merge (rdr backend + frontend wiring)._
_Binding decision: §8 of `2026-05-22-session-handoff-compaction.md`._

This document is the authoritative implementation guide for wiring the rdr
(rubric-driven reproduction) harness into the existing Next.js frontend.
The combined merge lands BOTH the rdr backend and the full frontend wiring
in a single PR.

---

## 1. Backend HTTP surface for rdr

### 1.1 Existing endpoints re-used

| Endpoint | Change | Purpose |
|---|---|---|
| `POST /runs` | Accept `mode='rdr'` + `paper_id` field (done) | Start an rdr run |
| `GET /runs/<id>` | No change | Poll status / project state |
| `GET /runs/<id>/events` (SSE) | Consume new `rdr_*` event types | Live cluster progress |
| `GET /runs/<id>/final-report` | No change | Download the benchmark MD |

`StartRunRequest` now has a `paper_id: str | None` field (added in
`backend/services/events/live_runs.py`). When `mode='rdr'` and `paper_id` is
set, `_python_script` dispatches `run_pipeline_rdr` instead of the
offline/sdk/rlm paths.

### 1.2 New rdr-specific GET endpoints (added in `backend/app.py`)

#### `GET /runs/<project_id>/clusters`

Returns per-cluster status by reading `runs/<id>/iterations/cluster_*.json`
and `repair_*.json`. Corpus-leak redaction applied (paper_full_text keys
stripped). 404 when run dir absent; 200 + empty list when iterations dir
absent.

```json
{
  "project_id": "<id>",
  "clusters": [
    {
      "index": 0,
      "cluster_id": "<uuid>",
      "title": "Implement the SNPE estimator",
      "leaf_ids": ["leaf-001", "leaf-002"],
      "failed": false,
      "file_count": 4,
      "repair_history": [
        {"pass": 1, "failed": false, "file_count": 6}
      ]
    }
  ]
}
```

#### `GET /runs/<project_id>/repair-iterations`

Summarizes repair passes (count + failed count per pass). Same 404/200-empty
semantics.

```json
{
  "project_id": "<id>",
  "passes": [
    {"pass": 1, "cluster_count": 22, "failed_count": 3}
  ]
}
```

#### `GET /runs/<project_id>/leaf-scores`

Per-leaf scores from `final_report.json`. 404 when report absent. Justification
capped at 1000 chars.

```json
{
  "project_id": "<id>",
  "overall_score": 0.456,
  "leaf_scores": [
    {"id": "<uuid>", "score": 0.8, "justification": "Model is implemented..."}
  ]
}
```

### 1.3 SSE event types emitted by `run_rdr`

The existing SSE bridge (`backend/services/events/live_runs.py`) streams
`dashboard_event` frames from `runs/<id>/dashboard_events.jsonl`. The rdr
controller emits these event sub-types via that file:

| Event type | When | Key fields |
|---|---|---|
| `cluster_started` | Before agent dispatch | `cluster_id`, `cluster_title`, `index`, `total` |
| `cluster_completed` | After agent returns | `cluster_id`, `failed`, `file_count` |
| `experiment_started` | Before env detect + run_experiment | `cluster_count` |
| `experiment_completed` | After run_experiment | `success`, `metrics` |
| `score_started` | Before `score_reproduction` | pass number |
| `score_completed` | After scoring | `overall_score`, `graded`, `leaf_count` |
| `repair_pass_started` | Before each repair pass | `pass`, `weak_count` |
| `repair_cluster_completed` | After each repair-pass cluster | `cluster_id`, `failed` |
| `repair_pass_completed` | After all clusters in a pass | `pass`, `overall_score` |

Note: these events are NOT yet emitted by `run_rdr` — wiring them is part
of this frontend integration work (see §6 pre-merge sequence). The SSE
bridge itself works; it just needs the controller to write the events.

### 1.4 Auth surface

See §4 below.

---

## 2. Frontend lab-shell changes

### 2.1 Mode selector

`frontend/src/lib/demo/demo-run-types.ts` `DemoRunMode` type already updated
to `"offline" | "sdk" | "rlm" | "rdr"` (done).

In `frontend/src/components/lab/lab-shell.tsx`, add `rdr` to the mode
dropdown (the existing `sdk/offline/rlm` selector). The rdr option should
be hidden or labelled "Beta" until the merge bar is met (score > 0.37).

### 2.2 rdr run launcher form additions

When the user selects `mode=rdr`, show an additional input:
- **Paper bundle ID**: text field → `paperId` query param → `paper_id` in
  the POST body.

The form sends `POST /api/demo?mode=rdr&paperId=<bundle>` (JSON body with
`mode`, `paper_id`, plus the standard run knobs).

### 2.3 POST /api/demo proxy changes (done)

`frontend/src/app/api/demo/route.ts`:
- `toRunMode` now accepts `"rlm"` and `"rdr"` (done).
- `toPaperId` helper reads `paperId` from query params (done).
- POST body includes `paper_id` when `runMode === "rdr"` (done).

### 2.4 SSE consumer changes for `rdr_*` events

`coalesceRunState` in the frontend (likely `frontend/src/lib/demo/server-run.ts`
or `lab-shell.tsx`) must handle the new `dashboard_event` sub-types:

```typescript
if (event.type === "cluster_started") {
  // push cluster into run.clusters array
}
if (event.type === "cluster_completed") {
  // update cluster status (failed, file_count)
}
if (event.type === "score_completed") {
  // update overall_score + update leaf scores
}
if (event.type === "repair_pass_started") {
  // show repair pass N / max badge
}
```

The key invariant from CLAUDE.md: "Don't remove the `coalesceRunState` guard
— it prevents UI flicker on transient timeouts." The rdr event handler must
sit inside the existing guard, not replace it.

---

## 3. Visual layout for the rdr run view

### 3.1 Overall structure

The rdr run view is a new tab/panel inside the existing lab run view, activated
when `runMode === "rdr"`.

```
+------------------------------------------------------------------+
|  rdr run: sequential-neural-score-estimation           Score: 0.456  |
+------------------------------------------------------------------+
|  [Code Development]   [Code Execution]   [Result Analysis]       |
|                                                                    |
|  +----------+  +----------+  +----------+  +----------+          |
|  | Cluster 0|  | Cluster 1|  | Cluster 2|  | Cluster 3|          |
|  | SNPE     |  | Train    |  | Run Exp  |  | Plot     |          |
|  | ✓ 0.82   |  | ✓ 0.75   |  | ✗ 0.12  |  | ⟳ ?      |          |
|  +----------+  +----------+  +----------+  +----------+          |
+------------------------------------------------------------------+
|  Repair timeline:  Pass 1 (3 weak) → Pass 2 (1 weak) → Done     |
+------------------------------------------------------------------+
|  Leaf heatmap: 24 leaves | 0.0 [############] 1.0                |
|  [leaf grid — each leaf a colored cell by score]                  |
+------------------------------------------------------------------+
```

### 3.2 Cluster cards

Each cluster card shows:
- Title (truncated to 40 chars)
- Leaf count
- Status: `pending` (grey) / `running` (blue spinner) / `done` (green tick or
  red cross if failed) / `repairing` (amber spinner)
- Score (from leaf-scores endpoint, weighted average for that cluster's leaves)

Cards are laid out in three horizontal lanes by `dominant_category`:
- **Code Development** (left lane)
- **Code Execution** (middle lane)
- **Result Analysis** (right lane)

Within each lane, cards appear in `index` order.

### 3.3 Repair-pass timeline

A horizontal step indicator below the cluster grid:
```
Initial pass  →  Repair 1 (N weak)  →  Repair 2 (N weak)  →  Final score
```
Each step shows cluster count + failed count from `GET /repair-iterations`.

### 3.4 Leaf-score heatmap

A grid of small colored cells, one per leaf. Color scale:
- 0.0 → red (#ef4444)
- 0.5 → amber (#f59e0b)
- 1.0 → green (#22c55e)

Tooltip on hover: leaf ID, score, justification (truncated to 80 chars in
the tooltip).

Fetched via `GET /runs/<id>/leaf-scores`. Refresh when `score_completed`
SSE event arrives.

### 3.5 Overall score badge

A large numerical badge in the run header: `Score: 0.456` (3 decimal places).
Color: grey until score_completed, then green if > 0.37, red otherwise.
Updates in real-time from the SSE `score_completed` event.

---

## 4. Auth dynamic

### 4.1 Three modes

| Mode | Env var value | Behavior |
|---|---|---|
| `demo` (default) | `REPROLAB_AUTH_MODE=demo` | Existing X-Demo-Secret HMAC. Empty secret disables gate (local dev). |
| `jwt` | `REPROLAB_AUTH_MODE=jwt` | Signed JWT Bearer token. Backend verifies with `REPROLAB_JWT_SECRET`. |
| `claude-oauth` | `REPROLAB_AUTH_MODE=claude-oauth` | Claude OAuth identity (Anthropic token). Backend verifies with the Anthropic token introspection endpoint. |

### 4.2 Backend middleware wiring (FastAPI)

Add a `REPROLAB_AUTH_MODE` setting to `backend/config.py`. In `create_app`,
install a dependency or middleware:

```python
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)

def _auth_gate(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    x_demo_secret: str | None = Header(default=None),
) -> None:
    mode = settings.auth_mode  # "demo" | "jwt" | "claude-oauth"
    if mode == "demo":
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
    elif mode == "jwt":
        _enforce_jwt(credentials)
    elif mode == "claude-oauth":
        _enforce_claude_oauth(credentials)
```

Apply via `app.add_middleware` or as a router-level dependency on all
protected routes (the `/runs` POST/DELETE family). Read-only GET endpoints
(`/runs/<id>`, `/clusters`, etc.) can be ungated for v1.

### 4.3 `_enforce_jwt` sketch

```python
import jwt  # pyjwt
def _enforce_jwt(credentials: HTTPAuthorizationCredentials | None) -> None:
    if credentials is None:
        raise HTTPException(403, "Bearer token required")
    try:
        jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:
        raise HTTPException(403, f"Invalid JWT: {exc}") from exc
```

Requires `REPROLAB_JWT_SECRET` env var. The frontend sends
`Authorization: Bearer <token>` obtained from the login flow.

### 4.4 `_enforce_claude_oauth` sketch

```python
import httpx
async def _enforce_claude_oauth(credentials: ...) -> None:
    # Introspect the Anthropic access token
    r = httpx.get(
        "https://api.anthropic.com/v1/auth/introspect",
        headers={"Authorization": f"Bearer {credentials.credentials}"},
    )
    if r.status_code != 200:
        raise HTTPException(403, "Invalid Claude OAuth token")
    # Optionally bind to allowed user list: settings.allowed_claude_users
```

Requires `REPROLAB_ALLOWED_CLAUDE_USERS` env var (comma-separated email
list, empty = allow any valid Claude user).

### 4.5 Frontend auth-mode selector

Add an `REPROLAB_AUTH_MODE` awareness to the frontend's `backendBaseUrl` helper:
- In `demo` mode: pass `x-demo-secret` header (existing behavior, unchanged).
- In `jwt` mode: read `REPROLAB_JWT_TOKEN` from env (server-side only) or
  from a browser `localStorage` key `reprolab_jwt` and attach as Bearer.
- In `claude-oauth` mode: use the Anthropic Claude OAuth flow; store the
  token in `localStorage`; refresh on 403.

A login UI (modal or page) is needed for `jwt` and `claude-oauth` modes.
For v1, a simple `<input type="password">` → POST `/auth/token` → store in
`localStorage` is sufficient for `jwt` mode.

---

## 5. Testing strategy

### 5.1 Backend route tests (already implemented)

`tests/test_rdr_routes.py` covers all three GET endpoints with:
- 200 + valid data for a populated run dir.
- 404 for missing run dir.
- 200 + empty list for run dir without iterations.
- Corpus-redaction: `paper_full_text` key does not appear in response.
- Justification truncation at 1000 chars.
- `StartRunRequest` validates `mode='rdr'` and `paper_id`.
- `POST /runs` returns 202 with `mode='rdr'`.

### 5.2 SSE emission tests

Tests for `dashboard_events.jsonl` emission from `run_rdr` belong in
`tests/rdr/test_controller.py` (Sonnet A's scope). A minimal contract:

```python
def test_run_rdr_emits_cluster_started_event(tmp_path, fake_rdr_fixtures):
    # Run a short offline-mocked rdr pass; check dashboard_events.jsonl
    events_path = tmp_path / project_id / "dashboard_events.jsonl"
    events = [json.loads(l) for l in events_path.read_text().splitlines()]
    types = [e["event"] for e in events]
    assert "cluster_started" in types
    assert "score_completed" in types
```

### 5.3 Frontend Playwright e2e

`frontend/e2e/rdr_mode.spec.ts` — offline-mocked rdr flow:

1. Mock `POST /api/demo` → return a fixture `LiveRunState` with `runMode='rdr'`.
2. Mock `GET /api/demo?projectId=<id>` → return cluster-progress states.
3. Mock `GET /api/demo/events` SSE → emit `cluster_started`, `cluster_completed`,
   `score_completed` events in sequence.
4. Assert:
   - rdr mode option is visible in the mode selector.
   - Cluster cards render in the three lanes.
   - Overall score badge updates when `score_completed` fires.
   - Leaf heatmap renders after score event.

Use `page.route()` to intercept the backend calls without a live server.
Add to CI: `npx playwright test frontend/e2e/rdr_mode.spec.ts`.

### 5.4 Route test command (acceptance gate)

```bash
.venv/bin/python -m pytest tests/ -q -k "rdr_routes or app"
# Expected: 85 passed, 1 skipped (chromadb)
```

---

## 6. Pre-merge sequence

The following must be ready in order:

| # | Item | Owner | Blocking? |
|---|---|---|---|
| 1 | seqnn3 live run scores > 0.37 | Sonnet A (running) | Yes — merge bar |
| 2 | `run_rdr` emits `dashboard_event` frames to `dashboard_events.jsonl` | Sonnet A | Yes — SSE wiring |
| 3 | rdr GET endpoints (clusters, repair-iterations, leaf-scores) | Done (Sonnet B) | Yes |
| 4 | `POST /runs` + `StartRunRequest` accept `mode='rdr'` + `paper_id` | Done (Sonnet B) | Yes |
| 5 | `/api/demo` proxy passes `mode='rdr'` and `paper_id` | Done (Sonnet B) | Yes |
| 6 | `DemoRunMode` type includes `"rdr"` | Done (Sonnet B) | Yes |
| 7 | `lab-shell.tsx` rdr mode option + paper_id form field | Not started | Yes |
| 8 | SSE consumer handles `rdr_*` event types in `coalesceRunState` | Not started | Yes |
| 9 | Cluster card / lane / repair-timeline / leaf-heatmap components | Not started | Yes |
| 10 | Auth middleware (`REPROLAB_AUTH_MODE`) | Not started | No — can ship as `demo` only |
| 11 | Playwright e2e test for rdr mode (offline-mocked) | Not started | No — can merge without, add post-merge |
| 12 | Retry-on-watchdog (`--resume` CLI flag + controller logic) | Not started | No |
| 13 | Backward-compat smoke (offline/sdk/rlm) | Not started | Yes |
| 14 | Full test suite green (pytest) | On each commit | Yes |
| 15 | Final docs commit (CHANGELOG, learn.md, system_overview.md) | Before merge | Yes |
| 16 | New clean branch + 10-15 noteworthy commits | Before merge | Yes |

Items marked "Yes" in "Blocking?" must all be green before the combined merge PR
is opened.

**Suggested parallel split between sessions:**
- Sonnet A: items 1, 2, 12 (rdr controller + SSE emission + resume flag).
- Sonnet B / follow-on session: items 7, 8, 9, 10, 11 (frontend components).
- Both: item 14 (keep tests green throughout).

---

## 7. Open questions for the user

1. **Visual layout**: lanes-by-category (Code-Dev / Code-Exec / Result-Analysis)
   vs a flat list vs a DAG view? The lane layout is proposed here, but a flat
   list is simpler and may be preferred for v1.

2. **Repair-pass UI**: show per-leaf score deltas between initial and repaired
   (requires keeping initial scores alongside repair scores), or just
   per-cluster failed/not-failed status? Delta view is more informative but
   requires storing the pre-repair `leaf_scores` separately.

3. **Auth-mode default**: `demo` (existing HMAC) or skip auth altogether for
   v1 frontend? If the frontend is internal-only, `demo` with an empty secret
   is fine. If it's public, JWT or Claude OAuth is needed before launch.

4. **Auth-mode login UI**: for `jwt` / `claude-oauth` modes, what is the
   login entry point? Separate `/login` page vs a modal vs a header bar widget?

5. **Paper bundle discovery**: should the frontend offer a dropdown of
   available PaperBench bundles (listing `third_party/paperbench/` dirs from
   a new `GET /rdr/bundles` endpoint), or is a text-field-only input OK for v1?

6. **SSE backpressure**: for a 22-cluster run, expect ~50-60 SSE events (2
   per cluster initial + 2 per cluster repair × up to 2 passes). Is that
   acceptable, or should events be batched (e.g. emit a `cluster_batch` every
   5 clusters)?

7. **Score rendering precision**: 3 decimal places (0.456) vs 1 decimal
   percentage (45.6%) vs letter grade? The existing benchmark panel uses
   percentage; rdr should match for visual consistency.

---

## Appendix: files changed / to change

### Already changed (Sonnet B this session)

| File | Change |
|---|---|
| `backend/app.py` | 3 new GET endpoints + corpus-redaction helpers |
| `backend/services/events/live_runs.py` | `RunMode` + `rdr` in Literal; `StartRunRequest.paper_id`; rdr branch in `_python_script` |
| `frontend/src/app/api/demo/route.ts` | `toRunMode` accepts rdr; `toPaperId`; POST body includes `paper_id` for rdr |
| `frontend/src/lib/demo/demo-run-types.ts` | `DemoRunMode` includes `"rlm"` and `"rdr"` |
| `tests/test_rdr_routes.py` | 15 tests covering all 3 endpoints + schema |

### Still to change (frontend components)

| File | Change |
|---|---|
| `frontend/src/components/lab/lab-shell.tsx` | rdr mode option; paper_id field |
| `frontend/src/lib/demo/server-run.ts` | `coalesceRunState` handles `rdr_*` events |
| `frontend/src/components/rdr/` (new dir) | Cluster lanes, repair timeline, leaf heatmap |
| `frontend/src/app/api/demo/events/route.ts` | Proxy clusters/repair/leaf-scores calls if needed |
| `backend/config.py` | `REPROLAB_AUTH_MODE` + `REPROLAB_JWT_SECRET` settings |
| `backend/app.py` | Auth middleware |
| `backend/agents/rdr/controller.py` | Emit `dashboard_event` frames at cluster transitions |
