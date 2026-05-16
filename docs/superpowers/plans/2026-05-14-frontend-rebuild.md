# Replix Frontend Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Replix lab UI as a coherent, dense internal research instrument with a separate `/demo` presentation mode for outsiders. Stop the fake data, kill the dead-code graveyard (~5k LOC), unify three competing visual systems behind one token file, and add power-user features (Cmd-K, keyboard nav, real run library, telemetry strip).

**Architecture:** One token file (`src/styles/tokens.css`) imported in the root layout drives every page. The 3,938-LOC `repro-lab-client.tsx` monolith is split into ~11 focused files plus extracted hooks. A `presentationMode: "internal" | "demo"` prop threads from the page through `ReproLabClient` so `/lab` shows backend stage IDs and `/demo` shows mythological aliases — one engine, two presentations. PaperBench keeps Tailwind but its utilities consume the same tokens (`bg-[var(--accent)]`), so coherence comes from tokens, not syntax. The unlock gate moves to `/demo` only; `/lab` and `/library` are internal.

**Tech Stack:** Next.js 16 (app router), React 19, TypeScript, Tailwind 3 (kept), CSS Modules (new), CSS custom properties (new), Vitest, Playwright, `cmdk@1.x` (new dep, 3 KB), `lucide-react` (existing, sparingly used).

---

## Why this plan exists

The previous codebase analysis (`docs/2026-05-14-hyperanalysis-summary.md`, if present, otherwise the conversation transcript) identified the frontend as the biggest cohesion problem in the repo: a generic SaaS landing with fake customer logos, a 3,938-LOC lab monolith with 1,571 lines of embedded CSS-in-JS, hardcoded fake sidebar recents, a hardcoded Hermes "2" badge, broken `/papers` and `/hermes` nav links, and three competing visual systems (Lab CSS-in-JS, PaperBench Tailwind, dead-dashboard `globals.css`). The user wants this rebuilt as an internal research tool (`/lab`) with a separate demo presentation (`/demo`), six-week scope, power-user features included.

This plan is split into **Phase A — The Lab Stops Lying (Weeks 1-3)** and **Phase B — Power Features (Weeks 4-6)** with a hard checkpoint between them. Phase A is shippable on its own.

---

## File Structure

### Deletes (Phase A Week 1)

| Path | LOC | Reason |
|---|---:|---|
| `frontend/src/app/page.tsx` | 10 | Replaced by redirect to /lab |
| `frontend/src/components/landing/stellar-hero.tsx` | 43 | Generic SaaS template |
| `frontend/src/components/landing/stellar-navigation.tsx` | 44 | Generic SaaS template |
| `frontend/src/components/landing/stellar-tab-stage.tsx` | 204 | Marketing carousel |
| `frontend/src/components/landing/stellar-logo-rail.tsx` | 64 | Fabricated customer logos (Nexera/M3/LAURA COLE/vertex) |
| `frontend/src/lib/landing/stellar-tabs.ts` | ~150 | Feeds the dead carousel |
| `frontend/src/components/lab/live-demo-client.tsx` | 903 | Only its own test imports it |
| `frontend/src/components/lab/live-demo-client.test.tsx` | 201 | Tests the dead client |
| `frontend/src/components/dashboard/dashboard-shell.tsx` | 84 | "Issue 21 dashboard scaffold" — branch artifact |
| `frontend/src/components/dashboard/activity-feed.tsx` | ~80 | Only the dead shell imports |
| `frontend/src/components/dashboard/citation-panel.tsx` | ~80 | Only the dead shell imports |
| `frontend/src/components/dashboard/pipeline-overview.tsx` | ~80 | Only the dead shell imports |
| `frontend/src/components/dashboard/task-detail-panel.tsx` | ~80 | Only the dead shell imports |
| `frontend/src/components/dashboard/topology-view.tsx` | ~80 | Only the dead shell imports |
| `frontend/src/features/dashboard/dashboard-shell.tsx` | 385 | Only the dead live-demo-client imports |
| `frontend/src/features/dashboard/dashboard-state.ts` | 220 | Only the dead live-demo-client imports |
| `frontend/src/features/dashboard/use-dashboard-state.ts` | 22 | Only the dead live-demo-client imports |
| `frontend/src/features/dashboard/dashboard-shell.test.tsx` | 50 | Tests the dead shell |
| `frontend/src/lib/dashboard/contracts.ts` | ~80 | Only the dead dashboard components consume |
| `frontend/src/lib/dashboard/fixtures.ts` | 223 | Only the dead dashboard consumes |
| `frontend/src/lib/dashboard/normalize.ts` | 311 | Only the dead dashboard consumes |
| `frontend/src/lib/dashboard/normalize.test.ts` | ~150 | Tests dead code |
| `frontend/src/lib/events/mock-events.ts` | 349 | Only the dead live-demo-client consumes |
| `frontend/src/lib/events/mock-event-adapter.ts` | ~70 | Only the dead live-demo-client consumes |
| `frontend/src/lib/events/mock-event-adapter.test.ts` | ~80 | Tests dead code |

**Approximate Phase A Week 1 deletion: ~4,000 LOC.**

### Creates (entire plan)

**Phase A:**
- `frontend/src/styles/tokens.css` — design tokens (colors, spacing, radii, type scale, shadows)
- `frontend/src/styles/reset.css` — minimal CSS reset
- `frontend/src/components/lab/lab-shell.tsx` — thin orchestrator (~280 LOC)
- `frontend/src/components/lab/lab-sidebar.tsx` — sidebar with real recents
- `frontend/src/components/lab/lab-canvas.tsx` — SVG workflow canvas
- `frontend/src/components/lab/pan-wrap.tsx` — pan/drag wrapper
- `frontend/src/components/lab/node-card.tsx` — single workflow node
- `frontend/src/components/lab/gate-chips.tsx` — gate status overlay
- `frontend/src/components/lab/upload-view.tsx` — upload + arxiv form
- `frontend/src/components/lab/agent-info-panel.tsx` — side panel for selected node
- `frontend/src/components/lab/hermes-audit-panel.tsx` — Hermes audit sub-panel
- `frontend/src/components/lab/script-panel.tsx` — final-report sub-panel
- `frontend/src/components/lab/floating-agent-window.tsx` — floating live-agent window
- `frontend/src/components/lab/node-config.ts` — NODES + EDGES + name maps
- `frontend/src/components/lab/icons.tsx` — extracted ICONS dictionary
- `frontend/src/hooks/use-run.ts` — SSE + auto-resume run lifecycle hook
- `frontend/src/hooks/use-pan.ts` — pan/drag interaction hook
- `frontend/src/components/lab/*.module.css` — per-component scoped styles (Week 3)

**Phase B:**
- `frontend/src/app/library/page.tsx` — run history page
- `frontend/src/components/library/library-table.tsx`
- `frontend/src/components/library/library-row.tsx`
- `frontend/src/components/library/library-filters.tsx`
- `frontend/src/app/api/runs/recent/route.ts` — proxy to backend `/runs?limit=N`
- `frontend/src/app/api/runs/list/route.ts` — proxy to backend `/runs?...`
- `frontend/src/components/lab/telemetry-strip.tsx` — bottom telemetry bar
- `frontend/src/components/lab/command-palette.tsx` — Cmd-K modal
- `frontend/src/hooks/use-command-palette.ts`
- `frontend/src/hooks/use-canvas-keyboard-nav.ts`
- `frontend/src/components/lab/shortcut-overlay.tsx` — `?` keyboard help overlay
- `frontend/src/app/demo/page.tsx` — demo presentation mode route
- `frontend/src/components/demo/demo-overlay.tsx` — tour overlay
- `frontend/src/components/lab/resizable-split.tsx` — split pane with persistence
- `frontend/src/lib/presentation-mode.tsx` — React context for mode threading

### Modifies (entire plan)

- `frontend/src/app/page.tsx` — becomes a redirect (Week 1)
- `frontend/src/app/lab/page.tsx` — passes `presentationMode="internal"` + initial recents
- `frontend/src/app/layout.tsx` — honest metadata, import tokens.css + reset.css
- `frontend/src/app/globals.css` — replaced with minimal reset (clear the dead dashboard CSS)
- `frontend/src/components/lab/repro-lab-client.tsx` — shrinks to a re-export from `lab-shell.tsx`, then deleted
- `frontend/src/components/lab/repro-lab-client.test.tsx` — moved/renamed to `lab-shell.test.tsx`
- `frontend/src/components/paperbench/paperbench-client.tsx` — swaps hardcoded zinc/amber/emerald/rose Tailwind to token utilities
- `frontend/tailwind.config.ts` — add CSS-variable color references for the few utilities PaperBench uses
- `frontend/proxy.ts` — restrict unlock gate to `/demo` and `/api/demo` only
- `backend/services/events/live_runs.py` — add `list_runs(limit, status?, order_by?)` method
- `backend/app.py` — add `GET /runs` (list) endpoint
- `frontend/src/components/lab/repro-lab-client.tsx:124` — drop `/hermes` from NAV
- `frontend/src/components/lab/repro-lab-client.tsx:747` — drop hardcoded "2" Hermes badge
- `frontend/src/components/lab/repro-lab-client.tsx:752-756` — replace hardcoded recents

---

## Conventions

- **Test commands:**
  - Frontend unit: `cd frontend && npm run test` (vitest)
  - Frontend typecheck: `cd frontend && npx tsc --noEmit`
  - Frontend e2e: `cd frontend && npx playwright test` (expect failures during W2 split — re-baseline after W3)
  - Backend: `.venv/bin/python -m pytest tests/`
- **Branch:** `frontend-rebuild` cut from `claw_demo`
- **Commits:** one per task, conventional commit format (`feat(web):`, `chore(web):`, `fix(web):`, `refactor(web):`, `test(web):`)
- **Visual smoke test cadence:** after any styling change, run `cd frontend && npm run dev` and verify `/lab`, `/paperbench`, `/library` (when it exists), and `/demo` (when it exists) load without console errors
- **DRY/YAGNI/TDD:** TDD where it pays — hooks, component behavior. Smoke-only for pure styling moves.
- **No new heavy deps:** only `cmdk@1.x`. If a task wants more deps, stop and ask.

---

## Phase A — The Lab Stops Lying (Weeks 1-3)

### Week 1 — Delete + Redirect + Honest Data

#### Task 1.1: Branch + baseline check

**Files:** none

- [ ] **Step 1: Cut the branch**

```bash
git checkout claw_demo
git pull
git checkout -b frontend-rebuild
```

- [ ] **Step 2: Baseline frontend unit tests pass**

```bash
cd frontend && npm ci && npm run test -- --run
```

Expected: all tests pass. If anything fails on `claw_demo`, stop and investigate — we need a clean baseline.

- [ ] **Step 3: Baseline typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Baseline backend tests pass (sample)**

```bash
.venv/bin/python -m pytest tests/test_live_run_api.py -v
```

Expected: all pass. (Full backend run is slow — sample suffices for baseline confidence.)

- [ ] **Step 5: Visual smoke**

```bash
cd frontend && npm run dev
```

Open `http://localhost:3000/lab` and confirm the current UI loads. Stop the dev server. No commit this task — we're verifying baseline.

---

#### Task 1.2: Replace marketing landing with redirect

**Files:**
- Modify: `frontend/src/app/page.tsx`

- [ ] **Step 1: Replace the file contents**

```tsx
import { redirect } from "next/navigation";

export default function HomePage(): never {
  redirect("/lab");
}
```

- [ ] **Step 2: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Visual smoke**

```bash
cd frontend && npm run dev
```

Open `http://localhost:3000/` → should 307 redirect to `/lab`. Verify in the browser network panel.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/page.tsx
git commit -m "feat(web): redirect / to /lab; kill marketing landing"
```

---

#### Task 1.3: Honest `<head>` metadata

**Files:**
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Replace the metadata block**

```tsx
export const metadata: Metadata = {
  title: "ReproLab",
  description: "Autonomous agent pipeline that reproduces ML papers."
};
```

- [ ] **Step 2: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/layout.tsx
git commit -m "chore(web): honest <head> metadata (drop SaaS template copy)"
```

---

#### Task 1.4: Drop hardcoded Hermes "2" badge

**Files:**
- Modify: `frontend/src/components/lab/repro-lab-client.tsx:747`

- [ ] **Step 1: Locate the line**

The line currently reads (around line 747):

```tsx
{item.id === "hermes" ? <span className="nav-aside">2</span> : null}
```

- [ ] **Step 2: Replace with nothing**

Delete the whole expression so the line becomes (or omit it entirely if it stood alone):

```tsx
{null}
```

Better: delete the conditional entirely.

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/repro-lab-client.tsx
git commit -m "fix(web): drop fake Hermes '2' badge from sidebar"
```

---

#### Task 1.5: Drop hardcoded fake `Recent` sidebar array (interim — real fetch lands in Task 1.13)

**Files:**
- Modify: `frontend/src/components/lab/repro-lab-client.tsx:752-767`

- [ ] **Step 1: Replace the hardcoded array block**

The current block is:

```tsx
<div className="nav-section-title">Recent</div>
{[
  { t: "Diffusion Policy", s: "running" },
  { t: "ACT Transformer", s: "shipped" },
  { t: "PerAct", s: "attention" }
].map((item) => (
  <a key={item.t} href="/lab" className="navitem navitem-small">
    <span
      className="nav-icon nav-status-dot"
      style={{
        background:
          item.s === "running" ? "var(--accent)" : item.s === "attention" ? "var(--warn)" : "var(--muted-2)"
      }}
    />
    <span className="nav-label">{item.t}</span>
  </a>
))}
```

Replace with an empty-state-aware section that the real-data task (1.13) will populate. For now, render only the header:

```tsx
<div className="nav-section-title">Recent</div>
<div className="navitem navitem-small" style={{ opacity: 0.5 }}>
  <span className="nav-label">No recent runs.</span>
</div>
```

- [ ] **Step 2: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Visual smoke**

```bash
cd frontend && npm run dev
```

Open `/lab`, confirm the sidebar shows "No recent runs." instead of the three fake entries.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/repro-lab-client.tsx
git commit -m "fix(web): drop hardcoded fake recents (real fetch wired in Task 1.13)"
```

---

#### Task 1.6: Drop broken `/papers` and `/hermes` nav items

**Files:**
- Modify: `frontend/src/components/lab/repro-lab-client.tsx:121-125`

- [ ] **Step 1: Locate NAV array**

```tsx
const NAV: NavItem[] = [
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "papers", label: "Library", icon: "papers", href: "/papers" },
  { id: "hermes", label: "Hermes", icon: "hermes", href: "/hermes", accent: true }
];
```

- [ ] **Step 2: Replace with the single valid entry plus a placeholder for Library**

Both `/papers` and `/hermes` route to 404s. Drop `/hermes`. Keep `Library` but point it at `/library` (the page lands in Phase B Week 4; for Phase A the link 404s — acceptable interim since this is an internal tool and we know the route lands soon).

```tsx
const NAV: NavItem[] = [
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "library", label: "Library", icon: "papers", href: "/library" }
];
```

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/repro-lab-client.tsx
git commit -m "fix(web): drop broken /papers and /hermes nav items"
```

---

#### Task 1.7: Delete dead Stellar landing components

**Files:**
- Delete: `frontend/src/components/landing/stellar-hero.tsx`
- Delete: `frontend/src/components/landing/stellar-navigation.tsx`
- Delete: `frontend/src/components/landing/stellar-tab-stage.tsx`
- Delete: `frontend/src/components/landing/stellar-logo-rail.tsx`
- Delete: `frontend/src/lib/landing/stellar-tabs.ts`

- [ ] **Step 1: Confirm no live imports**

```bash
grep -rn "stellar-hero\|stellar-navigation\|stellar-tab-stage\|stellar-logo-rail\|stellar-tabs" frontend/src/
```

Expected: only the files themselves match. After Task 1.2 (redirect landing) deleted `app/page.tsx`'s landing import, nothing live consumes these.

- [ ] **Step 2: Delete files**

```bash
rm frontend/src/components/landing/stellar-hero.tsx
rm frontend/src/components/landing/stellar-navigation.tsx
rm frontend/src/components/landing/stellar-tab-stage.tsx
rm frontend/src/components/landing/stellar-logo-rail.tsx
rm frontend/src/lib/landing/stellar-tabs.ts
rmdir frontend/src/components/landing 2>/dev/null || true
rmdir frontend/src/lib/landing 2>/dev/null || true
```

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/components/landing frontend/src/lib/landing
git commit -m "chore(web): delete Stellar SaaS landing (fake customer logos, generic copy)"
```

---

#### Task 1.8: Delete dead `live-demo-client.tsx` + its tests

**Files:**
- Delete: `frontend/src/components/lab/live-demo-client.tsx`
- Delete: `frontend/src/components/lab/live-demo-client.test.tsx`

- [ ] **Step 1: Confirm no live imports**

```bash
grep -rn "from .*live-demo-client\|from .*lab/live-demo-client" frontend/src/
```

Expected: only `frontend/src/components/lab/live-demo-client.test.tsx` references it. The lab page mounts `<ReproLabClient />`, not `<LiveDemoClient />`.

- [ ] **Step 2: Delete files**

```bash
rm frontend/src/components/lab/live-demo-client.tsx
rm frontend/src/components/lab/live-demo-client.test.tsx
```

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass; test count drops by the live-demo-client suite.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/components/lab
git commit -m "chore(web): delete dead live-demo-client (only its own test referenced it)"
```

---

#### Task 1.9: Delete dead `components/dashboard/*` and `features/dashboard/*` trees

**Files:**
- Delete: `frontend/src/components/dashboard/` (entire directory, 6 files)
- Delete: `frontend/src/features/dashboard/` (entire directory, 4 files)

- [ ] **Step 1: Confirm no live imports**

```bash
grep -rn "from .*components/dashboard\|from .*features/dashboard\|DashboardShell\|useDashboardState" frontend/src/
```

Expected: only references inside the dashboard trees themselves. The old `live-demo-client.tsx` (deleted in Task 1.8) was the only consumer outside.

- [ ] **Step 2: Delete directories**

```bash
rm -r frontend/src/components/dashboard
rm -r frontend/src/features
```

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/components frontend/src/features
git commit -m "chore(web): delete dead dashboard trees (Issue 21 scaffold + features/dashboard)"
```

---

#### Task 1.10: Delete dead `lib/dashboard/*` and `lib/events/mock-*`

**Files:**
- Delete: `frontend/src/lib/dashboard/contracts.ts`
- Delete: `frontend/src/lib/dashboard/fixtures.ts`
- Delete: `frontend/src/lib/dashboard/normalize.ts`
- Delete: `frontend/src/lib/dashboard/normalize.test.ts`
- Delete: `frontend/src/lib/events/mock-events.ts`
- Delete: `frontend/src/lib/events/mock-event-adapter.ts`
- Delete: `frontend/src/lib/events/mock-event-adapter.test.ts`

- [ ] **Step 1: Confirm no live imports**

```bash
grep -rn "from .*lib/dashboard\|from .*mock-events\|from .*mock-event-adapter\|MockEventAdapter" frontend/src/
```

Expected: only references inside these files themselves and the tests for the dashboard normalize logic.

- [ ] **Step 2: Delete**

```bash
rm -r frontend/src/lib/dashboard
rm frontend/src/lib/events/mock-events.ts
rm frontend/src/lib/events/mock-event-adapter.ts
rm frontend/src/lib/events/mock-event-adapter.test.ts
```

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/lib
git commit -m "chore(web): delete dead lib/dashboard and lib/events/mock-*"
```

---

#### Task 1.11: Clear `globals.css` to a minimal reset

**Files:**
- Modify: `frontend/src/app/globals.css`

- [ ] **Step 1: Replace contents entirely**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Reset and base-style only. Component styling lives in CSS modules
   or scoped <style jsx> blocks per component. */

html {
  background: #ffffff;
}

body {
  margin: 0;
  min-height: 100vh;
  background: #ffffff;
  color: #0e0e10;
  font-family: "Inter", sans-serif;
}

* {
  box-sizing: border-box;
}

a {
  color: inherit;
  text-decoration: none;
}

button {
  font: inherit;
}
```

- [ ] **Step 2: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Visual smoke**

```bash
cd frontend && npm run dev
```

Open `/lab` and confirm it still renders correctly. The lab UI's styles all live in its embedded `<style jsx global>`, so deleting the dashboard-shell CSS in globals.css should have no visible impact.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/globals.css
git commit -m "chore(web): reduce globals.css to minimal reset (delete dead dashboard CSS)"
```

---

#### Task 1.12: Backend — add `GET /runs?limit=N` listing endpoint

**Files:**
- Modify: `backend/services/events/live_runs.py`
- Modify: `backend/app.py`
- Test: `tests/test_live_runs_listing.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_runs_listing.py`:

```python
"""Tests for FileLiveRunService.list_runs() and the GET /runs endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import FileLiveRunService


def _write_status(runs_root: Path, project_id: str, status: dict) -> None:
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text(json.dumps(status), encoding="utf-8")


def test_list_runs_returns_recents_sorted_by_updated_at(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    service = FileLiveRunService(runs_root=runs_root)

    _write_status(runs_root, "prj_old", {"status": "completed", "projectId": "prj_old", "updatedAt": "2026-01-01T00:00:00Z"})
    time.sleep(0.01)
    _write_status(runs_root, "prj_mid", {"status": "running", "projectId": "prj_mid", "updatedAt": "2026-02-01T00:00:00Z"})
    time.sleep(0.01)
    _write_status(runs_root, "prj_new", {"status": "completed", "projectId": "prj_new", "updatedAt": "2026-03-01T00:00:00Z"})

    runs = service.list_runs(limit=2)

    assert [r["projectId"] for r in runs] == ["prj_new", "prj_mid"]


def test_list_runs_endpoint_returns_recents(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_a", {"status": "completed", "projectId": "prj_a", "updatedAt": "2026-01-01T00:00:00Z"})

    service = FileLiveRunService(runs_root=runs_root)
    app = create_app(run_service=service)
    client = TestClient(app)

    response = client.get("/runs?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert any(r["projectId"] == "prj_a" for r in body)
```

- [ ] **Step 2: Run the test (expect it to fail)**

```bash
.venv/bin/python -m pytest tests/test_live_runs_listing.py -v
```

Expected: FAIL — `list_runs` not defined and `/runs` not registered.

- [ ] **Step 3: Implement `list_runs()` on `FileLiveRunService`**

In `backend/services/events/live_runs.py`, add the method (place near `_latest_run` around line 360):

```python
async def list_runs(
    self,
    *,
    limit: int = 10,
    status: str | None = None,
) -> list[dict]:
    """List recent run statuses, newest first by `updatedAt` then mtime.

    Walks `self.runs_root` for `<project_id>/demo_status.json` files,
    parses each, filters by `status` if provided, and returns at most
    `limit` records. Returns the raw status dict (the same shape the
    /runs/<project_id> endpoint returns minus the enriched payload).
    """
    runs_root = Path(self.runs_root)
    if not runs_root.exists():
        return []
    entries: list[tuple[float, dict]] = []
    for project_dir in runs_root.iterdir():
        if not project_dir.is_dir():
            continue
        status_path = project_dir / "demo_status.json"
        if not status_path.is_file():
            continue
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if status is not None and data.get("status") != status:
            continue
        updated = data.get("updatedAt") or ""
        sort_key = updated if updated else f"~{status_path.stat().st_mtime}"
        entries.append((sort_key, data))
    entries.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in entries[:limit]]
```

(Note: this is synchronous I/O. Declared `async` to match the rest of the service surface so the endpoint can `await` uniformly. If perf is a concern later, wrap in `asyncio.to_thread`.)

- [ ] **Step 4: Register the endpoint in `backend/app.py`**

After the existing `@app.get("/runs/latest")` block (around line 96), add:

```python
@app.get("/runs")
async def list_runs(limit: int = 10, status: str | None = None) -> list[dict]:
    return await service.list_runs(limit=limit, status=status)
```

- [ ] **Step 5: Run the test (expect it to pass)**

```bash
.venv/bin/python -m pytest tests/test_live_runs_listing.py -v
```

Expected: PASS, two tests.

- [ ] **Step 6: Run the broader live-run test suite to confirm no regression**

```bash
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_runs_listing.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add backend/services/events/live_runs.py backend/app.py tests/test_live_runs_listing.py
git commit -m "feat(api): add GET /runs listing with limit + status filter"
```

---

#### Task 1.13: Frontend — real sidebar recents via SSR fetch

**Files:**
- Create: `frontend/src/app/api/runs/recent/route.ts`
- Create: `frontend/src/lib/runs/server-list.ts`
- Modify: `frontend/src/app/lab/page.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx` (Sidebar + ReproLabClient props)

- [ ] **Step 1: Create the proxy route**

`frontend/src/app/api/runs/recent/route.ts`:

```ts
import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const limit = url.searchParams.get("limit") ?? "10";
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs?limit=${encodeURIComponent(limit)}`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return NextResponse.json([], { status: 200 });
    }
    const body = (await response.json()) as unknown;
    return NextResponse.json(body);
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
```

(Returns `[]` on any error so the sidebar degrades gracefully — never crashes the page.)

- [ ] **Step 2: Create the server helper**

`frontend/src/lib/runs/server-list.ts`:

```ts
import "server-only";
import { backendBaseUrl } from "@/lib/demo/server-run";

export interface RecentRunSummary {
  projectId: string;
  status: string;
  sourceLabel?: string;
  updatedAt?: string;
}

export async function fetchRecentRuns(limit = 10): Promise<RecentRunSummary[]> {
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs?limit=${encodeURIComponent(String(limit))}`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return [];
    }
    const body = (await response.json()) as RecentRunSummary[];
    return Array.isArray(body) ? body : [];
  } catch {
    return [];
  }
}
```

- [ ] **Step 3: Update `lab/page.tsx` to fetch recents server-side**

```tsx
import { ReproLabClient } from "@/components/lab/repro-lab-client";
import { fetchRunById } from "@/lib/demo/server-run";
import { fetchRecentRuns } from "@/lib/runs/server-list";

export const dynamic = "force-dynamic";

export default async function LabPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const [initialRun, initialRecents] = await Promise.all([
    projectId ? fetchRunById(projectId) : Promise.resolve(null),
    fetchRecentRuns(8)
  ]);
  return <ReproLabClient initialRun={initialRun} initialRecents={initialRecents} />;
}
```

- [ ] **Step 4: Update `ReproLabClient` to accept and render `initialRecents`**

In `frontend/src/components/lab/repro-lab-client.tsx`:

1. Update `ReproLabClientProps`:

```tsx
import type { RecentRunSummary } from "@/lib/runs/server-list";

type ReproLabClientProps = {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
};
```

2. Update the `Sidebar` component signature to accept `recents`:

```tsx
function Sidebar({
  active,
  onBrandClick,
  recents
}: {
  active: string;
  onBrandClick: () => void;
  recents: RecentRunSummary[];
}) {
```

3. Replace the placeholder "No recent runs." section from Task 1.5 with a real list:

```tsx
<div className="nav-section-title">Recent</div>
{recents.length === 0 ? (
  <div className="navitem navitem-small" style={{ opacity: 0.5 }}>
    <span className="nav-label">No recent runs.</span>
  </div>
) : (
  recents.map((run) => {
    const dotColor =
      run.status === "running"
        ? "var(--accent)"
        : run.status === "failed"
          ? "var(--warn)"
          : "var(--muted-2)";
    const label = run.sourceLabel ?? run.projectId.slice(0, 14);
    return (
      <a
        key={run.projectId}
        href={`/lab?projectId=${encodeURIComponent(run.projectId)}`}
        className="navitem navitem-small"
      >
        <span className="nav-icon nav-status-dot" style={{ background: dotColor }} />
        <span className="nav-label">{label}</span>
      </a>
    );
  })
)}
```

4. In the `ReproLabClient` function signature and inside, thread `initialRecents` to the sidebar:

```tsx
export function ReproLabClient({
  initialRun = null,
  initialRecents = []
}: ReproLabClientProps) {
  // ...existing state...
  return (
    <div className="reproLab">
      <PrototypeStyles />
      <div className="layout">
        <Sidebar active="lab" onBrandClick={resetToUpload} recents={initialRecents} />
        {/* ...existing main... */}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 6: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass. (Existing tests pass `<ReproLabClient />` with default props; `initialRecents` defaults to `[]`.)

- [ ] **Step 7: Visual smoke**

Run the backend then frontend:

```bash
# in one terminal
.venv/bin/uvicorn backend.app:create_app --factory --port 8000
# in another
cd frontend && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Open `/lab`. If you have prior runs in `runs/`, they appear in the sidebar. If not, "No recent runs." renders.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/app/api/runs/recent frontend/src/lib/runs frontend/src/app/lab/page.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "feat(web): real sidebar recents via SSR fetch of /runs?limit=N"
```

---

#### Task 1.14: Real progress bar from stage index

**Files:**
- Modify: `frontend/src/components/lab/repro-lab-client.tsx` (NodeCard component + its CSS in `<style jsx global>`)

- [ ] **Step 1: Add a stage-index helper near the top of the file**

After the `EDGES` constant (around line 291), add:

```tsx
const PIPELINE_STAGES = [
  "ingested",
  "paper_understood",
  "artifacts_discovered",
  "environment_built",
  "plan_created",
  "gate_1_passed",
  "baseline_implemented",
  "baseline_run",
  "gate_2_passed",
  "improvements_selected",
  "improvements_run",
  "gate_3_passed",
  "research_map_generated",
  "complete"
] as const;

function stageProgress(stage: string | null | undefined): number {
  if (!stage) return 0;
  const idx = PIPELINE_STAGES.indexOf(stage as (typeof PIPELINE_STAGES)[number]);
  return idx < 0 ? 0 : (idx + 1) / PIPELINE_STAGES.length;
}
```

- [ ] **Step 2: Pass `stage` into `NodeCard`**

Update `Canvas` to compute and pass stage progress, and update `NodeCard` props:

```tsx
function NodeCard({
  node,
  onClick,
  selected,
  state,
  progress
}: {
  node: WorkflowNode;
  onClick: () => void;
  selected: boolean;
  state: NodeState;
  progress: number;
}) {
```

In `Canvas`, where it renders nodes:

```tsx
{NODES.map((node) => (
  <NodeCard
    key={node.id}
    node={node}
    state={stateMap[node.id]}
    selected={selectedId === node.id}
    progress={stageProgress(run.payload?.summary?.stage)}
    onClick={() =>
      stateMap[node.id] === "upcoming" ? undefined : onSelect(node.id === selectedId ? null : node.id)
    }
  />
))}
```

- [ ] **Step 3: Replace the CSS-animated progress bar with a width-driven one**

In `NodeCard`, replace the existing `showProgress` block:

```tsx
{showProgress ? (
  <div className="node-progress">
    <div
      className="node-progress-fill"
      style={{
        background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)",
        width: `${Math.max(5, Math.round(progress * 100))}%`,
        transition: "width 0.4s ease"
      }}
    />
  </div>
) : null}
```

- [ ] **Step 4: Remove the `wfBar` CSS animation reference**

In the `<style jsx global>` block, find the `.reproLab .wf-bar` rule (around line 3038) and remove its `animation: wfBar 3s linear forwards;` line — that animation is the lie. Replace the rule with:

```css
.reproLab .wf-bar {
  height: 100%;
  border-radius: 999px;
}
.reproLab .node-progress-fill {
  height: 100%;
  border-radius: 999px;
}
```

The `@keyframes wfBar` block can be removed too — search for `@keyframes wfBar` in the same `<style jsx global>` and delete the whole keyframes block.

- [ ] **Step 5: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 6: Frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 7: Visual smoke**

Run a real run (or use the fixture) and confirm:
- A running node's progress bar reflects pipeline stage, not a 3-second animation
- The bar advances when stage advances
- Completed nodes show full width

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/lab/repro-lab-client.tsx
git commit -m "fix(web): progress bar reflects real stage index (kill 3s wfBar animation lie)"
```

---

#### Task 1.15: Week 1 acceptance — full test run + smoke

**Files:** none

- [ ] **Step 1: Full frontend tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 2: Full typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Backend tests (sample)**

```bash
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_runs_listing.py -v
```

Expected: all pass.

- [ ] **Step 4: Visual smoke — full app**

```bash
.venv/bin/uvicorn backend.app:create_app --factory --port 8000 &
cd frontend && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Confirm:
- `/` redirects to `/lab`
- `/lab` renders, sidebar shows real recents (or empty state)
- Sidebar nav has only "Lab" and "Library"
- No "Hermes 2" badge
- HTML `<title>` is "ReproLab", description is honest
- No console errors

- [ ] **Step 5: Push the branch**

```bash
git push -u origin frontend-rebuild
```

No commit this task; this is a milestone gate.

---

### Week 2 — Split the Monolith

Goal: extract `repro-lab-client.tsx` (3,938 LOC) into ~11 files and 2 hooks, **without changing any visible behavior or styling**. CSS-in-JS stays inside the new files for now (Week 3 extracts it to modules).

Each extraction follows the same shape: copy the function and its dependencies into a new file, replace the original with an import, run tests, commit. Subagents handling these tasks should NOT change the function bodies — pure code motion.

#### Task 2.1: Extract `node-config.ts`

**Files:**
- Create: `frontend/src/components/lab/node-config.ts`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create the new file**

Copy the `NODES`, `EDGES`, `PIPELINE_STAGES`, `stageProgress`, `WorkflowNode`, `Tone`, `NodeState` from `repro-lab-client.tsx` into a new file `frontend/src/components/lab/node-config.ts`:

```ts
export type Tone = "accent" | "hermes" | "info" | "neutral";
export type NodeState = "done" | "running" | "upcoming";

export type WorkflowNode = {
  detail: string;
  icon: string;
  id: string;
  role: string;
  step: string;
  tone: Tone;
  x: number;
  y: number;
  agent: string;
};

export const NODES: WorkflowNode[] = [
  // ... copy NODES array verbatim from repro-lab-client.tsx ...
];

export const EDGES: Array<[string, string]> = [
  // ... copy EDGES array verbatim ...
];

export const PIPELINE_STAGES = [
  "ingested",
  "paper_understood",
  // ... copy verbatim ...
  "complete"
] as const;

export function stageProgress(stage: string | null | undefined): number {
  if (!stage) return 0;
  const idx = PIPELINE_STAGES.indexOf(stage as (typeof PIPELINE_STAGES)[number]);
  return idx < 0 ? 0 : (idx + 1) / PIPELINE_STAGES.length;
}
```

Note: `WorkflowNode.icon` becomes `string` here (instead of `keyof typeof ICONS`) because `ICONS` is in a separate file. Type assertion at the `NodeCard` site handles this.

- [ ] **Step 2: Update `repro-lab-client.tsx` to import these**

Replace the in-file definitions with imports near the top:

```tsx
import { NODES, EDGES, PIPELINE_STAGES, stageProgress, type Tone, type NodeState, type WorkflowNode } from "./node-config";
```

Delete the in-file `NODES`, `EDGES`, `PIPELINE_STAGES`, `stageProgress`, `WorkflowNode`, `Tone`, `NodeState` definitions.

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/node-config.ts frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract NODES/EDGES/PIPELINE_STAGES to node-config.ts"
```

---

#### Task 2.2: Extract `icons.tsx`

**Files:**
- Create: `frontend/src/components/lab/icons.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create the icons file**

Move the `icon()` helper and the entire `ICONS` const from `repro-lab-client.tsx` (lines ~293-425) into a new file:

```tsx
import type { ReactNode } from "react";

function icon(children: ReactNode, size = 18) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export const ICONS = {
  // ... copy the entire ICONS object verbatim ...
};

export type IconKey = keyof typeof ICONS;
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

Replace the in-file `icon()` and `ICONS` with:

```tsx
import { ICONS, type IconKey } from "./icons";
```

Replace `keyof typeof ICONS` references with `IconKey`. Delete the in-file `icon()` helper and `ICONS` const.

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/icons.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract ICONS to icons.tsx"
```

---

#### Task 2.3: Extract `upload-view.tsx`

**Files:**
- Create: `frontend/src/components/lab/upload-view.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create the file**

Move the `UploadView` component (lines ~797-906) into `frontend/src/components/lab/upload-view.tsx`, adding the necessary imports at the top:

```tsx
"use client";

import { useRef } from "react";
import type { DemoModelChoice } from "@/lib/demo/demo-run-types";
import { ICONS } from "./icons";

export function UploadView({
  arxiv,
  busy,
  error,
  model,
  onArxivChange,
  onArxivSubmit,
  onFileSelected,
  onModelChange,
  over,
  setOver
}: {
  arxiv: string;
  busy: boolean;
  error: string | null;
  model: DemoModelChoice;
  onArxivChange: (value: string) => void;
  onArxivSubmit: () => void;
  onFileSelected: (file: File) => void;
  onModelChange: (value: DemoModelChoice) => void;
  over: boolean;
  setOver: (value: boolean) => void;
}) {
  // ... paste the full body verbatim ...
}
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

Add the import:

```tsx
import { UploadView } from "./upload-view";
```

Delete the in-file `UploadView` function.

- [ ] **Step 3: Typecheck**

```bash
cd frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass. (The existing test in `repro-lab-client.test.tsx` exercises the upload form — it still works because `<ReproLabClient />` mounts `<UploadView />`.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/upload-view.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract UploadView component"
```

---

#### Task 2.4: Extract `node-card.tsx`

**Files:**
- Create: `frontend/src/components/lab/node-card.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create the file**

Move `NodeCard` (lines ~908-1015) and the `NODE_W`/`NODE_H` constants into `node-card.tsx`:

```tsx
"use client";

import { ICONS } from "./icons";
import type { NodeState, WorkflowNode } from "./node-config";

export const NODE_W = 200;
export const NODE_H = 80;

export function NodeCard({
  node,
  onClick,
  selected,
  state,
  progress
}: {
  node: WorkflowNode;
  onClick: () => void;
  selected: boolean;
  state: NodeState;
  progress: number;
}) {
  // ... paste the full body verbatim ...
}
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

```tsx
import { NodeCard, NODE_W, NODE_H } from "./node-card";
```

Delete the in-file `NodeCard`, `NODE_W`, `NODE_H`.

- [ ] **Step 3: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/node-card.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract NodeCard component"
```

---

#### Task 2.5: Extract `gate-chips.tsx`

**Files:**
- Create: `frontend/src/components/lab/gate-chips.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create the file**

Move `gateChipState`, `GATE_COORDS`, `GateChips` (lines ~1102-1177) plus the `issueText` helper they depend on:

```tsx
"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

function issueText(value?: string | null): string {
  if (!value) return "";
  return value.replace(/\bfailed\b/gi, "needs attention").replace(/\bfailure\b/gi, "issue");
}

function gateChipState(
  gate: { passed?: boolean; status?: string; chipStatus?: string } | undefined,
  stage: string | null,
  passedStages: string[],
  runningStages: string[]
): { state: "pending" | "running" | "passed" | "caveat" | "failed"; label: string } {
  // ... paste verbatim ...
}

const GATE_COORDS: Record<"gate_1" | "gate_2" | "gate_3", { x: number; y: number; label: string }> = {
  gate_1: { x: 720, y: 405, label: "Gate 1" },
  gate_2: { x: 980, y: 295, label: "Gate 2" },
  gate_3: { x: 1260, y: 295, label: "Gate 3" }
};

export function GateChips({ run }: { run: LiveDemoRunState }) {
  // ... paste verbatim ...
}
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

```tsx
import { GateChips } from "./gate-chips";
```

Delete the in-file `gateChipState`, `GATE_COORDS`, `GateChips`. Move `issueText` to `gate-chips.tsx` (or keep a copy in the orchestrator if it's used elsewhere — check with grep, but it's only used by GateChips and an error display).

- [ ] **Step 3: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/gate-chips.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract GateChips component"
```

---

#### Task 2.6: Extract `lab-canvas.tsx` + `pan-wrap.tsx`

**Files:**
- Create: `frontend/src/components/lab/lab-canvas.tsx`
- Create: `frontend/src/components/lab/pan-wrap.tsx`
- Create: `frontend/src/hooks/use-pan.ts`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Extract pan logic into a hook**

Create `frontend/src/hooks/use-pan.ts`:

```ts
"use client";

import { useEffect, useRef } from "react";

export function usePan() {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef({ active: false, moved: false, slx: 0, sx: 0, sty: 0, sy: 0 });

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    wrap.scrollLeft = Math.max(0, 740 - wrap.clientWidth / 2 + 100);
    wrap.scrollTop = Math.max(0, 310 - wrap.clientHeight / 2 + 40);
  }, []);

  useEffect(() => {
    function onMove(event: MouseEvent) {
      const drag = dragRef.current;
      if (!drag.active || !wrapRef.current) return;
      wrapRef.current.scrollLeft = drag.slx - (event.clientX - drag.sx);
      wrapRef.current.scrollTop = drag.sty - (event.clientY - drag.sy);
      if (Math.abs(event.clientX - drag.sx) + Math.abs(event.clientY - drag.sy) > 4) {
        drag.moved = true;
      }
    }
    function onUp() {
      dragRef.current.active = false;
      if (wrapRef.current) wrapRef.current.style.cursor = "grab";
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  function onMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest("[data-node]")) return;
    const wrap = wrapRef.current;
    if (!wrap) return;
    dragRef.current = {
      active: true,
      moved: false,
      slx: wrap.scrollLeft,
      sx: event.clientX,
      sty: wrap.scrollTop,
      sy: event.clientY
    };
    wrap.style.cursor = "grabbing";
  }

  return { wrapRef, dragRef, onMouseDown };
}
```

- [ ] **Step 2: Create `lab-canvas.tsx`**

```tsx
"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { DashboardLiveEvent } from "./agent-timeline-rail";
import { GateChips } from "./gate-chips";
import { NODES, EDGES, stageProgress, type NodeState } from "./node-config";
import { NodeCard, NODE_W, NODE_H } from "./node-card";

// keep the FloatingAgentWindow placeholder import — we extract it in Task 2.7
import { FloatingAgentWindow } from "./floating-agent-window";

function buildEdgePath(from: { x: number; y: number }, to: { x: number; y: number }) {
  const x1 = from.x + NODE_W;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y + NODE_H / 2;
  const cx1 = x1 + Math.max(40, (x2 - x1) * 0.45);
  const cx2 = x2 - Math.max(40, (x2 - x1) * 0.45);
  return `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`;
}

export function LabCanvas({
  onSelect,
  run,
  selectedId,
  stateMap,
  dashboardEvents,
  decisions
}: {
  onSelect: (id: string | null) => void;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
  dashboardEvents: DashboardLiveEvent[];
  decisions: string[];
}) {
  // ... paste the body of the existing Canvas() function verbatim ...
}
```

- [ ] **Step 3: Create `pan-wrap.tsx`**

```tsx
"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { DashboardLiveEvent } from "./agent-timeline-rail";
import { LabCanvas } from "./lab-canvas";
import { usePan } from "@/hooks/use-pan";
import type { NodeState } from "./node-config";

export function PanWrap({
  onSelect,
  run,
  selectedId,
  stateMap,
  dashboardEvents,
  decisions
}: {
  onSelect: (id: string | null) => void;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
  dashboardEvents: DashboardLiveEvent[];
  decisions: string[];
}) {
  const { wrapRef, dragRef, onMouseDown } = usePan();
  return (
    <div ref={wrapRef} className="pan-wrap" onMouseDown={onMouseDown}>
      <LabCanvas
        run={run}
        stateMap={stateMap}
        selectedId={selectedId}
        dashboardEvents={dashboardEvents}
        decisions={decisions}
        onSelect={(id) => {
          if (!dragRef.current.moved) onSelect(id);
        }}
      />
    </div>
  );
}
```

- [ ] **Step 4: Update `repro-lab-client.tsx`**

```tsx
import { PanWrap } from "./pan-wrap";
```

Replace usage `<PanCanvas .../>` with `<PanWrap .../>`. Delete the in-file `Canvas`, `PanCanvas`, and `buildEdgePath`.

- [ ] **Step 5: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/lab-canvas.tsx frontend/src/components/lab/pan-wrap.tsx frontend/src/hooks/use-pan.ts frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract LabCanvas, PanWrap, usePan hook"
```

---

#### Task 2.7: Extract `floating-agent-window.tsx`, `hermes-audit-panel.tsx`, `script-panel.tsx`, `agent-info-panel.tsx`

**Files:**
- Create: `frontend/src/components/lab/floating-agent-window.tsx`
- Create: `frontend/src/components/lab/hermes-audit-panel.tsx`
- Create: `frontend/src/components/lab/script-panel.tsx`
- Create: `frontend/src/components/lab/agent-info-panel.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

Four sub-extractions in one task because they're tightly related — `AgentInfo` uses `HermesAuditPanel` and `ScriptPanel`; `LabCanvas` uses `FloatingAgentWindow`.

- [ ] **Step 1: Extract `FloatingAgentWindow`**

Search `repro-lab-client.tsx` for `function FloatingAgentWindow` and move it to `frontend/src/components/lab/floating-agent-window.tsx` with its required imports. Re-export.

- [ ] **Step 2: Extract `HermesAuditPanel` and `ScriptPanel`**

Move `HermesAuditPanel` (function defined around line 1376) and `ScriptPanel` (around line 1458) into their own files. They depend on `LiveDemoRunState`, `ICONS`, and helper functions (`sourcePdfUrl`, `finalReportUrl`, `formatBytes`, `shortHash`, `verdictLabel`). Move those helpers into a new small file `frontend/src/components/lab/agent-info-helpers.ts` and import from both new panels.

- [ ] **Step 3: Extract `AgentInfo` as `AgentInfoPanel`**

Move `AgentInfo` (around line 1273) into `agent-info-panel.tsx`. Rename to `AgentInfoPanel` to match the file. Import `HermesAuditPanel`, `ScriptPanel`, and the helpers.

- [ ] **Step 4: Update `repro-lab-client.tsx`**

```tsx
import { AgentInfoPanel } from "./agent-info-panel";
```

Replace `<AgentInfo .../>` with `<AgentInfoPanel .../>` everywhere. Delete the in-file functions and helpers.

- [ ] **Step 5: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/floating-agent-window.tsx frontend/src/components/lab/hermes-audit-panel.tsx frontend/src/components/lab/script-panel.tsx frontend/src/components/lab/agent-info-panel.tsx frontend/src/components/lab/agent-info-helpers.ts frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract AgentInfoPanel, HermesAuditPanel, ScriptPanel, FloatingAgentWindow"
```

---

#### Task 2.8: Extract `lab-sidebar.tsx`

**Files:**
- Create: `frontend/src/components/lab/lab-sidebar.tsx`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

- [ ] **Step 1: Create `lab-sidebar.tsx`**

Move the `Sidebar` function (lines ~706-785), `StatusPill`, `statusTone`, `statusLabel`, and the `NavItem`/`Status` types into the new file. Rename `Sidebar` → `LabSidebar`. Accept `recents` prop (already added in Task 1.13).

```tsx
"use client";

import { useState } from "react";
import { ICONS, type IconKey } from "./icons";
import type { RecentRunSummary } from "@/lib/runs/server-list";

type NavItem = {
  accent?: boolean;
  href: string;
  icon: IconKey;
  id: string;
  label: string;
};

export type Status = "auditing" | "completed" | "attention" | "queued" | "running" | "shipped" | "stopped";

export function statusTone(status: Status) { /* paste verbatim */ }
export function statusLabel(status: Status) { /* paste verbatim */ }
export function StatusPill({ status }: { status: Status }) { /* paste verbatim */ }

const NAV: NavItem[] = [
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "library", label: "Library", icon: "papers", href: "/library" }
];

export function LabSidebar({
  active,
  onBrandClick,
  recents
}: {
  active: string;
  onBrandClick: () => void;
  recents: RecentRunSummary[];
}) {
  // ... paste body verbatim ...
}
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

```tsx
import { LabSidebar, StatusPill, type Status } from "./lab-sidebar";
```

Replace `<Sidebar .../>` with `<LabSidebar .../>`. Delete the in-file types and components.

- [ ] **Step 3: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/lab-sidebar.tsx frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract LabSidebar with StatusPill"
```

---

#### Task 2.9: Extract `use-run.ts` hook

**Files:**
- Create: `frontend/src/hooks/use-run.ts`
- Modify: `frontend/src/components/lab/repro-lab-client.tsx`

The SSE + auto-resume + start-run logic is the biggest single chunk left in the monolith (~300 LOC inside `ReproLabClient`). Extracting it makes the orchestrator readable.

- [ ] **Step 1: Create the hook**

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { DashboardLiveEvent } from "@/components/lab/agent-timeline-rail";

const MAX_DASHBOARD_EVENTS = 200;
const DEFAULT_RUN_QUERY =
  "/api/demo?mode=sdk&provider=anthropic&executionMode=efficient&sandbox=runpod&gpuMode=auto";
const POLL_INTERVAL_MS = 3000;
const LAST_RUN_KEY = "reprolab:lastRun";

// ... copy writeLastRun, clearLastRun, readLastRun, postRunRequest,
//     describeStartError, issueText, coalesceRunState helpers from the
//     monolith ...

export interface UseRunResult {
  run: LiveDemoRunState | null;
  busy: boolean;
  error: string | null;
  dashboardEvents: DashboardLiveEvent[];
  startFixtureRun: (model: DemoModelChoice) => Promise<void>;
  startUploadedRun: (file: File, model: DemoModelChoice) => Promise<void>;
  clearRun: () => Promise<void>;
  resetToUpload: () => void;
}

export function useRun(initialRun: LiveDemoRunState | null = null): UseRunResult {
  const [run, setRun] = useState<LiveDemoRunState | null>(initialRun);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dashboardEvents, setDashboardEvents] = useState<DashboardLiveEvent[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollTimer = useRef<number | null>(null);
  const dashboardProjectIdRef = useRef<string | null>(null);
  const router = useRouter();
  const didAutoResume = useRef(false);

  // ... port setRunUrl, the auto-resume effect, the SSE/poll effect,
  //     startFixtureRun, startUploadedRun, clearRun, resetToUpload from
  //     the existing ReproLabClient verbatim ...

  return { run, busy, error, dashboardEvents, startFixtureRun, startUploadedRun, clearRun, resetToUpload };
}
```

- [ ] **Step 2: Update `repro-lab-client.tsx`**

Replace the entire in-function state setup with:

```tsx
const {
  run,
  busy,
  error,
  dashboardEvents,
  startFixtureRun,
  startUploadedRun,
  clearRun,
  resetToUpload
} = useRun(initialRun);
const [arxiv, setArxiv] = useState("");
const [model, setModel] = useState<DemoModelChoice>("sonnet");
const [over, setOver] = useState(false);
```

Wire `startFixtureRun(model)` and `startUploadedRun(file, model)` at the call sites.

- [ ] **Step 3: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/hooks/use-run.ts frontend/src/components/lab/repro-lab-client.tsx
git commit -m "refactor(web): extract useRun hook (SSE + auto-resume + start/stop)"
```

---

#### Task 2.10: Rename `repro-lab-client.tsx` → `lab-shell.tsx`

**Files:**
- Rename: `frontend/src/components/lab/repro-lab-client.tsx` → `frontend/src/components/lab/lab-shell.tsx`
- Rename: `frontend/src/components/lab/repro-lab-client.test.tsx` → `frontend/src/components/lab/lab-shell.test.tsx`
- Modify: `frontend/src/app/lab/page.tsx`
- Modify: `frontend/src/components/lab/lab-shell.test.tsx`

- [ ] **Step 1: Git rename both files**

```bash
git mv frontend/src/components/lab/repro-lab-client.tsx frontend/src/components/lab/lab-shell.tsx
git mv frontend/src/components/lab/repro-lab-client.test.tsx frontend/src/components/lab/lab-shell.test.tsx
```

- [ ] **Step 2: Rename the exported function**

In `lab-shell.tsx`, rename `export function ReproLabClient(...)` to `export function LabShell(...)`. Update the embedded `<style jsx global>` block — its rules target `.reproLab` (the className). Leave the className itself as `reproLab` for now; the visual switch happens in Week 3.

- [ ] **Step 3: Update `app/lab/page.tsx`**

```tsx
import { LabShell } from "@/components/lab/lab-shell";
// ...
return <LabShell initialRun={initialRun} initialRecents={initialRecents} />;
```

- [ ] **Step 4: Update `lab-shell.test.tsx`**

```tsx
import { LabShell, stateMapForRun } from "./lab-shell";
// Replace all `<ReproLabClient .../>` with `<LabShell .../>`
```

- [ ] **Step 5: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add -A frontend/src/components/lab frontend/src/app/lab
git commit -m "refactor(web): rename ReproLabClient → LabShell"
```

---

#### Task 2.11: Week 2 acceptance — measure the split

**Files:** none

- [ ] **Step 1: Confirm file sizes**

```bash
wc -l frontend/src/components/lab/*.tsx frontend/src/components/lab/*.ts frontend/src/hooks/*.ts
```

Expected: `lab-shell.tsx` is now ~300-400 LOC (was 3,938). The 1,571 lines of `<style jsx global>` still live in `lab-shell.tsx` — they're extracted to CSS modules in Week 3.

- [ ] **Step 2: Full test pass**

```bash
cd frontend && npm run test -- --run && npx tsc --noEmit
```

Expected: both clean.

- [ ] **Step 3: Visual smoke — no visual changes expected**

```bash
.venv/bin/uvicorn backend.app:create_app --factory --port 8000 &
cd frontend && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Compare `/lab` to your Week 1 screenshot. Should be pixel-identical. If anything visual changed, a CSS rule got dropped during extraction — investigate.

- [ ] **Step 4: Push the branch**

```bash
git push
```

---

### Week 3 — Design Tokens + Per-Component CSS Modules + Visual Direction

Goal: extract the 1,571-line `<style jsx global>` block to per-component `*.module.css` files consuming shared tokens, then apply the new visual direction (forest accent, JetBrains Mono, tighter radii, hairline grid).

#### Task 3.1: Create `tokens.css` + import in layout

**Files:**
- Create: `frontend/src/styles/tokens.css`
- Modify: `frontend/src/app/layout.tsx`
- Modify: `frontend/src/app/globals.css`

- [ ] **Step 1: Create the token file**

`frontend/src/styles/tokens.css`:

```css
:root {
  /* === COLORS — light theme === */
  --ink: #0e0e10;
  --ink-2: #1f2024;
  --muted: #6b6e74;
  --muted-2: #9aa0a6;
  --line: #e6e7ea;
  --line-2: #cfd2d6;
  --chip: #f1f1f3;
  --bg: #ffffff;
  --panel: #ffffff;
  --canvas-bg: #fafafb;
  --canvas-grid: #dcdce0;

  /* === Accents — new direction: forest, not Stripe-green === */
  --accent: #1d7a3c;
  --accent-soft: #e2f1e8;
  --accent-ink: #0b5024;

  /* Info / secondary === */
  --info-soft: #e5e9ff;
  --info-ink: #2e3a8c;

  /* Hermes — kept; distinctive === */
  --hermes: #7c5cff;
  --hermes-soft: #ede8ff;
  --hermes-ink: #3f2bb3;

  /* Warning / error === */
  --warn: #d97706;
  --warn-soft: #fff3e0;
  --warn-ink: #92480a;
  --err: #c81e1e;
  --err-soft: #fde8e8;

  /* === Typography === */
  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --fs-xs: 11px;
  --fs-sm: 12.5px;
  --fs-base: 13.5px;
  --fs-md: 15px;
  --fs-lg: 18px;
  --fs-xl: 24px;

  /* === Spacing === */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 24px;
  --sp-6: 32px;

  /* === Radii (tighter than before) === */
  --r-xs: 6px;
  --r-sm: 8px;
  --r-md: 10px;
  --r-lg: 14px;

  /* === Shadows === */
  --shadow-1: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-2: 0 4px 12px -6px rgba(14,14,16,0.18);
  --shadow-3: 0 16px 36px -18px rgba(14,14,16,0.5);
}
```

- [ ] **Step 2: Import JetBrains Mono in `layout.tsx`**

```tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-inter"
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains-mono"
});

export const metadata: Metadata = {
  title: "ReproLab",
  description: "Autonomous agent pipeline that reproduces ML papers."
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Import tokens in `globals.css`**

Update `globals.css` to import tokens at the top:

```css
@import "../styles/tokens.css";

@tailwind base;
@tailwind components;
@tailwind utilities;

html { background: var(--bg); }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-sans);
}
* { box-sizing: border-box; }
a { color: inherit; text-decoration: none; }
button { font: inherit; }
```

- [ ] **Step 4: Typecheck, smoke, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run dev   # confirm /lab still renders; tokens are now available
```

```bash
git add frontend/src/styles/tokens.css frontend/src/app/layout.tsx frontend/src/app/globals.css
git commit -m "feat(web): introduce tokens.css (colors, type, spacing, radii); import in layout"
```

---

#### Task 3.2: Extract CSS for `lab-shell.tsx` → `lab-shell.module.css`

**Files:**
- Create: `frontend/src/components/lab/lab-shell.module.css`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

The `<style jsx global>` block in `lab-shell.tsx` is ~1,571 lines spanning multiple component domains (sidebar, canvas, side-panel, upload, hermes, script, gate-chips, etc.). We migrate it incrementally — each component file gets its co-located CSS, the orchestrator file ends up with no `<style jsx>` at all.

This task moves the shell-level styles (layout, content area, top-level wrapper). Subsequent tasks (3.3–3.8) carve off domain-specific rules.

- [ ] **Step 1: Create `lab-shell.module.css`**

Copy the layout/shell rules (`.reproLab`, `.reproLab .layout`, `.reproLab .content`, `.reproLab .workflow-layout`, plus the keyframes/animations) from the `<style jsx global>` block into the new module file. Drop the `.reproLab` prefix; use the auto-generated CSS Module class instead:

```css
.shell {
  font-family: var(--font-sans);
  color: var(--ink);
  background: var(--bg);
  min-height: 100vh;
}

.layout {
  display: flex;
  min-height: 100vh;
}

.content {
  flex: 1;
  min-width: 0;
  padding: 22px 28px 32px;
}

.workflowLayout {
  display: flex;
  gap: 22px;
  align-items: flex-start;
}

@keyframes fadeup {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: none; }
}

/* ... and any other shell-level keyframes ... */

@media (max-width: 1200px) {
  .workflowLayout { flex-direction: column; }
}

@media (max-width: 900px) {
  .layout { flex-direction: column; }
  .content { padding-top: 8px; }
}
```

- [ ] **Step 2: Update `lab-shell.tsx` to import the module**

```tsx
import styles from "./lab-shell.module.css";
// ...
return (
  <div className={styles.shell}>
    <div className={styles.layout}>
      <LabSidebar active="lab" onBrandClick={resetToUpload} recents={initialRecents} />
      <main className={styles.content}>
        {/* ... */}
      </main>
    </div>
  </div>
);
```

Delete the shell-related rules from the `<style jsx global>` block (keep the rest for now — subsequent tasks move them).

- [ ] **Step 3: Typecheck, smoke, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run dev   # confirm /lab layout is unchanged
git add frontend/src/components/lab/lab-shell.module.css frontend/src/components/lab/lab-shell.tsx
git commit -m "refactor(web): extract shell-level styles to lab-shell.module.css"
```

---

#### Task 3.3: Extract CSS for `lab-sidebar.tsx`

**Files:**
- Create: `frontend/src/components/lab/lab-sidebar.module.css`
- Modify: `frontend/src/components/lab/lab-sidebar.tsx`
- Modify: `frontend/src/components/lab/lab-shell.tsx` (remove sidebar rules from `<style jsx global>`)

- [ ] **Step 1: Create the module file**

Copy `.reproLab .sidebar`, `.reproLab .sb-toggle`, `.reproLab .brand-row`, `.reproLab .navitem`, `.reproLab .nav-icon`, `.reproLab .nav-label`, `.reproLab .nav-section-title`, `.reproLab .nav-aside`, `.reproLab .navitem-small`, `.reproLab .nav-status-dot`, `.reproLab .sidebar-footer`, `.reproLab .dotted`, `.reproLab .brand-text`, `.reproLab .collapsed` from `lab-shell.tsx`'s `<style jsx global>` block into `lab-sidebar.module.css`. Drop `.reproLab` prefix; use module classes.

Style port: rules consume tokens (`var(--ink)`, `var(--line)`, etc.). Where the original used hardcoded colors that should be tokens, replace.

- [ ] **Step 2: Update `lab-sidebar.tsx` to use the module**

```tsx
import styles from "./lab-sidebar.module.css";
// Replace className="sidebar" with className={styles.sidebar}, etc.
```

- [ ] **Step 3: Remove sidebar rules from `lab-shell.tsx`'s `<style jsx global>`**

Delete the sidebar-related rules from the embedded block. Other component rules remain.

- [ ] **Step 4: Typecheck, smoke, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run dev
git add frontend/src/components/lab/lab-sidebar.module.css frontend/src/components/lab/lab-sidebar.tsx frontend/src/components/lab/lab-shell.tsx
git commit -m "refactor(web): extract sidebar styles to lab-sidebar.module.css"
```

---

#### Task 3.4: Extract CSS for `lab-canvas.tsx` + `node-card.tsx` + `pan-wrap.tsx` + `gate-chips.tsx`

**Files:**
- Create: `frontend/src/components/lab/lab-canvas.module.css`
- Create: `frontend/src/components/lab/node-card.module.css`
- Create: `frontend/src/components/lab/gate-chips.module.css`
- Modify: the three component files above
- Modify: `frontend/src/components/lab/lab-shell.tsx` (drop their rules from `<style jsx global>`)

Mirror the pattern from Task 3.3. The canvas styles include `.canvas-wrap`, `.pan-wrap`, `.canvas-surface`, `.canvas-edges`, plus the dot-grid background — apply the visual direction (hairline grid replaces dots) here:

```css
.canvasSurface {
  position: relative;
  width: 1740px;
  height: 720px;
  background-color: var(--canvas-bg);
  background-image:
    linear-gradient(to right, var(--canvas-grid) 0.5px, transparent 0.5px);
  background-size: 22px 22px;
}
```

NodeCard styles get the tighter radius (`var(--r-md)` instead of 14px-hardcoded) and the new accent.

Commit message: `refactor(web): extract canvas/node-card/gate-chips styles; apply hairline grid`.

---

#### Task 3.5: Extract CSS for `upload-view.tsx`

**Files:**
- Create: `frontend/src/components/lab/upload-view.module.css`
- Modify: `frontend/src/components/lab/upload-view.tsx`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

Same pattern. `.upload-shell`, `.upload-zone`, `.upload-title`, `.upload-copy`, `.upload-meta`, `.upload-divider`, `.upload-form`, `.upload-text-input`, `.begin-button`, `.upload-config-row`, `.upload-error`.

Visual direction touches:
- `.upload-title` → `font-size: var(--fs-xl)` (24px) instead of the current 64px; the upload screen is the first thing internal users see — dense and instrumented, not marketing-hero.
- Card radius `var(--r-lg)` (14px) instead of the original 32px round.
- Drop the heavy gradient/glow on the drop zone — single hairline border, single hover state.

Commit: `refactor(web): extract upload-view styles; tone down the marketing-hero treatment`.

---

#### Task 3.6: Extract CSS for `agent-info-panel.tsx` + sub-panels

**Files:**
- Create: `frontend/src/components/lab/agent-info-panel.module.css`
- Create: `frontend/src/components/lab/hermes-audit-panel.module.css`
- Create: `frontend/src/components/lab/script-panel.module.css`
- Modify: the three components

Move `.side-panel`, `.side-panel-top`, `.side-panel-bottom`, `.side-panel-scroll`, `.side-panel-heading`, `.side-panel-title`, `.live-pill`, `.agent-head`, `.agent-icon`, `.agent-name`, `.agent-section`, `.eyebrow`, `.agent-task`, `.agent-role`, `.agent-detail`, `.agent-progress`, `.telemetry-list`, `.telemetry-row`, `.telemetry-name`, `.telemetry-meta`, `.agent-log-list`, `.agent-log-item`, `.agent-log-time`, `.agent-log-msg`, `.status-pill`, `.status-dot`, `.pulse-dot`, etc. into `agent-info-panel.module.css`.

Hermes: `.hermes-panel`, `.hermes-row`, `.hermes-row-head`, `.hermes-row-stage`, `.hermes-status`, `.hermes-status-*`, `.hermes-row-summary`, `.hermes-findings`, `.hermes-interventions`, `.hermes-intervention*` → `hermes-audit-panel.module.css`.

Script panel: `.script-panel*`, `.pdf-card`, `.pdf-preview`, `.pdf-copy`, `.pdf-title`, `.pdf-meta`, `.pdf-hash`, `.pdf-actions`, `.btn`, `.btn-sm`, `.btn-dark` → `script-panel.module.css`.

Apply visual direction: `.mono`, `.agent-log-time`, `.pdf-hash` use `var(--font-mono)`. All technical-value spans (project IDs, durations, costs) use `var(--font-mono)`.

Commit: `refactor(web): extract agent-info + hermes-audit + script-panel styles to modules`.

---

#### Task 3.7: Extract CSS for `floating-agent-window.tsx` + `timeline-panel.tsx` + `agent-timeline-rail.tsx` + `progress-strip.tsx`

**Files:**
- Create: `frontend/src/components/lab/floating-agent-window.module.css`
- Create: `frontend/src/components/lab/agent-timeline-rail.module.css`
- (Modify the components accordingly)

Same pattern. The timeline/progress components currently live in `lab/agent-timeline-rail.tsx` (220 LOC), `lab/progress-strip.tsx` (155 LOC), `lab/timeline-panel.tsx` (168 LOC). Each may have its own embedded styles already or pull from the shell — survey first, port second.

Commit: `refactor(web): extract remaining lab component styles to modules`.

---

#### Task 3.8: Delete the `<style jsx global>` block from `lab-shell.tsx`

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Verify the block is empty**

By this point, every rule that was inside `<style jsx global>` has been moved. The block should be just the keyframes that didn't fit anywhere else (or empty). If there are leftover rules, find their owner — they live in whichever component file consumes them.

- [ ] **Step 2: Delete the `<style jsx global>` block and the `PrototypeStyles` component**

Search `lab-shell.tsx` for `function PrototypeStyles` and the surrounding `<style jsx global>` block — delete both. Remove the `<PrototypeStyles />` JSX render at the top of `LabShell`.

- [ ] **Step 3: Visual smoke**

```bash
cd frontend && npm run dev
```

Open `/lab`. Everything still renders. If something looks broken, a CSS rule was missed during one of the previous extractions — search for the broken element's className in the git history.

- [ ] **Step 4: Confirm no `<style jsx>` anywhere in the lab tree**

```bash
grep -rn "style jsx" frontend/src/components/lab/
```

Expected: zero matches.

- [ ] **Step 5: Tests**

```bash
cd frontend && npm run test -- --run
```

Expected: all pass.

- [ ] **Step 6: Measure the shrink**

```bash
wc -l frontend/src/components/lab/lab-shell.tsx
```

Expected: ~250-350 LOC (down from 3,938).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx
git commit -m "refactor(web): delete <style jsx global> from lab-shell; all styles now in modules"
```

---

#### Task 3.9: Port PaperBench to consume tokens

**Files:**
- Modify: `frontend/src/components/paperbench/paperbench-client.tsx`
- Modify: `frontend/tailwind.config.ts`

- [ ] **Step 1: Add token-backed Tailwind colors**

In `frontend/tailwind.config.ts`, extend the theme:

```ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        muted: "var(--muted)",
        line: "var(--line)",
        chip: "var(--chip)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        "accent-ink": "var(--accent-ink)",
        warn: "var(--warn)",
        "warn-soft": "var(--warn-soft)",
        err: "var(--err)",
        "err-soft": "var(--err-soft)"
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;
```

- [ ] **Step 2: Swap PaperBench's hardcoded zinc/amber/emerald/rose Tailwind classes**

In `paperbench-client.tsx:33-38`, the `palette` map:

```tsx
const palette: Record<PaperBenchRunStatus["status"], string> = {
  pending: "bg-chip text-muted",
  running: "bg-warn-soft text-warn",
  succeeded: "bg-accent-soft text-accent-ink",
  failed: "bg-err-soft text-err",
};
```

And the `BaselineRow` margin colors (lines 64-70):

```tsx
className={
  margin === null
    ? "text-muted"
    : margin >= 0
      ? "text-accent"
      : "text-err"
}
```

- [ ] **Step 3: Sweep for any other `bg-zinc-`, `text-zinc-`, `bg-amber-`, `bg-emerald-`, `bg-rose-` in the paperbench component**

```bash
grep -nE "bg-zinc|text-zinc|bg-amber|bg-emerald|bg-rose|text-emerald|text-amber|text-rose" frontend/src/components/paperbench/
```

Expected: replace each with a token-backed class.

- [ ] **Step 4: Typecheck, smoke, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run dev   # visit /paperbench
git add frontend/src/components/paperbench frontend/tailwind.config.ts
git commit -m "feat(web): paperbench consumes design tokens via tailwind theme extension"
```

---

#### Task 3.10: Apply the visual direction sweep

**Files:**
- Modify: any module CSS file that needs a token swap

A final pass to ensure the visual direction lands:
- All `.mono` / monospace technical values use `var(--font-mono)`.
- All accent uses `var(--accent)` (the new forest green); search for `#16b25c` and replace.
- All large radii (anything > 14px in CSS) → `var(--r-lg)` or smaller.
- The canvas grid is hairlines, not dots (already done in Task 3.4).
- The upload-view hero is dense, not marketing (already done in Task 3.5).

- [ ] **Step 1: Sweep for old hardcoded hex**

```bash
grep -rnE "#16b25c|#0e7a3d|#e6f7ed" frontend/src/
```

Expected: zero matches. Replace any leftovers with the matching token.

- [ ] **Step 2: Sweep for monospace classnames not pointing at the token**

```bash
grep -rn "font-family.*mono\|font-mono" frontend/src/
```

Expected: all use `var(--font-mono)` or the Tailwind `font-mono` (which Task 3.9 maps to the token).

- [ ] **Step 3: Visual review**

Run `/lab` and `/paperbench`. Compare side-by-side. They should feel like the same product now: same accent, same monospace for technical values, same border radii, same line weights.

- [ ] **Step 4: Commit**

```bash
git add -A frontend/src
git commit -m "feat(web): unify visual direction (forest accent, JetBrains Mono, tighter radii)"
```

---

#### Task 3.11: Phase A acceptance

**Files:** none

- [ ] **Step 1: Full test + typecheck**

```bash
cd frontend && npm run test -- --run && npx tsc --noEmit
```

Expected: both clean.

- [ ] **Step 2: e2e re-baseline (optional but recommended)**

```bash
cd frontend && npx playwright test --reporter=line
```

If the e2e drifted during the split, fix the spec to match the new component names. If acceptance is otherwise solid, this can move to its own follow-up task.

- [ ] **Step 3: Full visual review**

```bash
.venv/bin/uvicorn backend.app:create_app --factory --port 8000 &
cd frontend && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Verify:
- `/` redirects to `/lab`
- `/lab` shows real recents in sidebar (or empty state)
- Nav has only "Lab" and "Library"
- No fake Hermes "2" badge
- Progress bar reflects real stage index
- `/paperbench` uses the same colors/typography as `/lab`
- No `<style jsx global>` blocks anywhere
- No console errors
- Forest accent (`#1d7a3c`-ish) everywhere, not Stripe-green
- JetBrains Mono on project IDs, hashes, costs, durations

- [ ] **Step 4: Push the branch**

```bash
git push
```

**Checkpoint.** Phase A is shippable. If Phase B priorities have changed, stop here and reassess.

---

## Phase A Acceptance Criteria

- [ ] `/` redirects to `/lab` (no marketing landing)
- [ ] Sidebar shows real recent runs from backend (or empty state, no fake hardcoded list)
- [ ] No hardcoded "2" Hermes badge
- [ ] No broken `/papers` or `/hermes` nav links
- [ ] HTML metadata is honest (no "Intelligent automation syncs…")
- [ ] Progress bar reflects real stage index (no 3s CSS animation)
- [ ] `globals.css` is a minimal reset + token import (no dead dashboard CSS)
- [ ] Backend `GET /runs?limit=N` endpoint exists and is tested
- [ ] `lab-shell.tsx` is ≤ 400 LOC (down from 3,938)
- [ ] Every other lab component is in its own file with co-located `*.module.css`
- [ ] No `<style jsx global>` blocks remain in the lab tree
- [ ] PaperBench uses the same design tokens as the lab
- [ ] All deletions completed (Stellar, live-demo-client, dashboard trees, lib/dashboard, lib/landing, mock-events)
- [ ] Visual direction applied: forest accent, JetBrains Mono for technical values, tighter radii, hairline grid

---

## Phase B — Power Features (Weeks 4-6)

### Week 4 — `/library` Page + Telemetry Strip

#### Task 4.1: Backend — extend `/runs` with filter + sort

**Files:**
- Modify: `backend/services/events/live_runs.py`
- Modify: `backend/app.py`
- Test: `tests/test_live_runs_listing.py`

Extend `list_runs(...)` to accept `status: str | None`, `order_by: Literal["updated_at", "completed_at"] | None`, and `q: str | None` (a substring search over `sourceLabel`).

- [ ] **Step 1: Extend the test file**

In `tests/test_live_runs_listing.py`, add tests for filter+sort+query:

```python
def test_list_runs_filters_by_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    service = FileLiveRunService(runs_root=runs_root)
    _write_status(runs_root, "a", {"status": "completed", "projectId": "a", "updatedAt": "2026-03-01T00:00:00Z"})
    _write_status(runs_root, "b", {"status": "failed",    "projectId": "b", "updatedAt": "2026-03-02T00:00:00Z"})
    runs = service.list_runs(limit=10, status="completed")
    assert [r["projectId"] for r in runs] == ["a"]


def test_list_runs_q_substring_match_on_source_label(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    service = FileLiveRunService(runs_root=runs_root)
    _write_status(runs_root, "a", {"status": "completed", "projectId": "a", "sourceLabel": "Diffusion Policy"})
    _write_status(runs_root, "b", {"status": "completed", "projectId": "b", "sourceLabel": "ACT Transformer"})
    runs = service.list_runs(limit=10, q="diffusion")
    assert [r["projectId"] for r in runs] == ["a"]
```

- [ ] **Step 2: Extend `list_runs(...)`**

Update the method signature to `async def list_runs(self, *, limit=10, status=None, q=None, order_by="updated_at")` and apply filters before the slice. `q` is a `.lower() in label.lower()` substring match.

- [ ] **Step 3: Extend the `/runs` endpoint**

```python
@app.get("/runs")
async def list_runs(
    limit: int = 10,
    status: str | None = None,
    q: str | None = None,
    order_by: str = "updated_at"
) -> list[dict]:
    return await service.list_runs(limit=limit, status=status, q=q, order_by=order_by)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_live_runs_listing.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/services/events/live_runs.py backend/app.py tests/test_live_runs_listing.py
git commit -m "feat(api): /runs accepts status, q (substring), order_by filters"
```

---

#### Task 4.2: Library proxy route

**Files:**
- Create: `frontend/src/app/api/runs/list/route.ts`

```ts
import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const qs = new URLSearchParams();
  for (const key of ["limit", "status", "q", "order_by"]) {
    const value = url.searchParams.get(key);
    if (value !== null) qs.set(key, value);
  }
  try {
    const response = await fetch(`${backendBaseUrl()}/runs?${qs.toString()}`, { cache: "no-store" });
    if (!response.ok) return NextResponse.json([], { status: 200 });
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
```

Commit: `feat(web): /api/runs/list proxy with filter+sort passthrough`.

---

#### Task 4.3: `/library` server page + components

**Files:**
- Create: `frontend/src/app/library/page.tsx`
- Create: `frontend/src/components/library/library-table.tsx`
- Create: `frontend/src/components/library/library-row.tsx`
- Create: `frontend/src/components/library/library-filters.tsx`
- Create: `frontend/src/components/library/library-table.module.css`
- Create: `frontend/src/lib/runs/server-list.ts` (already created in Task 1.13 — extend with `RunSummary` superset)

- [ ] **Step 1: Extend `server-list.ts` with full `RunSummary`**

Add a `fetchRunList(params)` function that returns `RunSummary[]` (a richer shape than `RecentRunSummary` — includes `completedAt`, `benchmark.overallScore`, `benchmark.deltaValue`, telemetry rollups).

- [ ] **Step 2: Create the page**

`frontend/src/app/library/page.tsx`:

```tsx
import { fetchRunList } from "@/lib/runs/server-list";
import { LibraryTable } from "@/components/library/library-table";

export const dynamic = "force-dynamic";

export default async function LibraryPage({
  searchParams
}: {
  searchParams: Promise<{ status?: string; q?: string; order_by?: string }>;
}) {
  const params = await searchParams;
  const initialRuns = await fetchRunList({
    limit: 100,
    status: params.status,
    q: params.q,
    order_by: params.order_by ?? "updated_at"
  });
  return <LibraryTable initialRuns={initialRuns} initialParams={params} />;
}
```

- [ ] **Step 3: Create the table + row + filters components**

Build a simple table with header row (Project, Source, Status, Updated, Reward Δ, Cost), client-side filtering bar (status pills, search input, sort dropdown), each row a link to `/lab?projectId=<id>`. Use the same design tokens for consistency.

- [ ] **Step 4: Wire the sidebar Library link to actually land here**

Already done in Task 1.6 (the NAV array points at `/library`). Confirm with a visual smoke test.

- [ ] **Step 5: Typecheck, tests, commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/app/library frontend/src/components/library frontend/src/lib/runs
git commit -m "feat(web): /library page with filter/sort/search"
```

---

#### Task 4.4: Telemetry strip

**Files:**
- Create: `frontend/src/components/lab/telemetry-strip.tsx`
- Create: `frontend/src/components/lab/telemetry-strip.module.css`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Create the component**

The strip reads from `run.telemetry` (existing in `LiveDemoRunState`) and aggregates totals:
- Total tokens in / out (sum across telemetry records)
- USD spent (if exposed by backend — check `run.payload?.telemetry` or a future `costSummary` field)
- Wall clock since `startedAt`

```tsx
"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import styles from "./telemetry-strip.module.css";

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

export function TelemetryStrip({ run }: { run: LiveDemoRunState | null }) {
  if (!run) return null;
  const tel = run.telemetry ?? [];
  const totalMessages = tel.reduce((acc, t) => acc + (t.message_count ?? 0), 0);
  const totalChars = tel.reduce((acc, t) => acc + (t.output_chars ?? 0), 0);
  const startedAt = run.startedAt ? new Date(run.startedAt).getTime() : 0;
  const referenceEnd = run.completedAt
    ? new Date(run.completedAt).getTime()
    : Date.now();
  const wall = startedAt ? referenceEnd - startedAt : 0;

  return (
    <div className={styles.strip}>
      <div className={styles.cell}>
        <span className={styles.label}>Messages</span>
        <span className={styles.value}>{totalMessages.toLocaleString()}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Output chars</span>
        <span className={styles.value}>{totalChars.toLocaleString()}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Wall</span>
        <span className={styles.value}>{formatDuration(wall)}</span>
      </div>
      <div className={styles.cell}>
        <span className={styles.label}>Project</span>
        <span className={styles.value}>{run.projectId.slice(0, 14)}</span>
      </div>
    </div>
  );
}
```

The CSS module: bottom-sticky strip, JetBrains Mono for values, hairline top border, fits inside `lab-shell.module.css`'s content area.

- [ ] **Step 2: Mount in `lab-shell.tsx`**

```tsx
import { TelemetryStrip } from "./telemetry-strip";
// ...
<main className={styles.content}>
  {/* existing content */}
  <TelemetryStrip run={run} />
</main>
```

- [ ] **Step 3: Visual smoke, tests, commit**

```bash
cd frontend && npm run dev
cd frontend && npm run test -- --run
git add frontend/src/components/lab/telemetry-strip.tsx frontend/src/components/lab/telemetry-strip.module.css frontend/src/components/lab/lab-shell.tsx
git commit -m "feat(web): persistent telemetry strip (messages, output chars, wall, project id)"
```

---

### Week 5 — Cmd-K + Keyboard Nav

#### Task 5.1: Install cmdk

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Install**

```bash
cd frontend && npm install cmdk
```

- [ ] **Step 2: Confirm it's in `dependencies`** (not devDependencies).

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore(web): add cmdk@1.x (Cmd-K command palette dep, 3KB)"
```

---

#### Task 5.2: Command palette scaffold

**Files:**
- Create: `frontend/src/components/lab/command-palette.tsx`
- Create: `frontend/src/components/lab/command-palette.module.css`
- Create: `frontend/src/hooks/use-command-palette.ts`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

The hook listens for `cmd+k` / `ctrl+k` and toggles `open`. The component renders a `cmdk`-based modal with actions.

- [ ] **Step 1: Hook**

```ts
"use client";
import { useEffect, useState } from "react";

export function useCommandPalette() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return { open, setOpen };
}
```

- [ ] **Step 2: Component**

```tsx
"use client";

import { Command } from "cmdk";
import { useRouter } from "next/navigation";
import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import type { RecentRunSummary } from "@/lib/runs/server-list";
import styles from "./command-palette.module.css";

export function CommandPalette({
  open,
  setOpen,
  recents,
  currentRun
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  recents: RecentRunSummary[];
  currentRun: LiveDemoRunState | null;
}) {
  const router = useRouter();
  if (!open) return null;

  function go(href: string) {
    setOpen(false);
    router.push(href);
  }
  function copy(value: string) {
    navigator.clipboard.writeText(value).catch(() => null);
    setOpen(false);
  }

  return (
    <div className={styles.overlay} onClick={() => setOpen(false)}>
      <div className={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <Command label="Command palette">
          <Command.Input className={styles.input} placeholder="Type a command or run name…" />
          <Command.List className={styles.list}>
            <Command.Empty>No matches.</Command.Empty>

            <Command.Group heading="Navigate">
              <Command.Item onSelect={() => go("/lab")}>Lab</Command.Item>
              <Command.Item onSelect={() => go("/library")}>Library</Command.Item>
              <Command.Item onSelect={() => go("/paperbench")}>PaperBench</Command.Item>
            </Command.Group>

            {currentRun ? (
              <Command.Group heading="Current run">
                <Command.Item onSelect={() => copy(currentRun.projectId)}>Copy project id</Command.Item>
                <Command.Item onSelect={() => copy(`${window.location.origin}/lab?projectId=${currentRun.projectId}`)}>
                  Copy shareable URL
                </Command.Item>
                <Command.Item onSelect={() => go(`/api/demo/final-report?projectId=${currentRun.projectId}`)}>
                  Open final report
                </Command.Item>
              </Command.Group>
            ) : null}

            <Command.Group heading="Recent runs">
              {recents.map((r) => (
                <Command.Item key={r.projectId} onSelect={() => go(`/lab?projectId=${r.projectId}`)}>
                  {r.sourceLabel ?? r.projectId}
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
```

The CSS module: dimmed overlay, centered dialog ~600px wide, mono input, list items with hover/selection states.

- [ ] **Step 3: Mount in `lab-shell.tsx`**

```tsx
import { CommandPalette } from "./command-palette";
import { useCommandPalette } from "@/hooks/use-command-palette";
// inside LabShell:
const palette = useCommandPalette();
// in JSX:
<CommandPalette
  open={palette.open}
  setOpen={palette.setOpen}
  recents={initialRecents}
  currentRun={run}
/>
```

- [ ] **Step 4: Visual smoke**

Open `/lab`, press `Cmd-K` (Mac) or `Ctrl-K` (Linux). Modal opens, type to filter, Enter to select.

- [ ] **Step 5: Tests + commit**

```bash
cd frontend && npx tsc --noEmit
cd frontend && npm run test -- --run
git add frontend/src/components/lab/command-palette.tsx frontend/src/components/lab/command-palette.module.css frontend/src/hooks/use-command-palette.ts frontend/src/components/lab/lab-shell.tsx
git commit -m "feat(web): Cmd-K command palette (navigate, copy, recents, current run actions)"
```

---

#### Task 5.3: Keyboard navigation through canvas nodes

**Files:**
- Create: `frontend/src/hooks/use-canvas-keyboard-nav.ts`
- Modify: `frontend/src/components/lab/lab-canvas.tsx`

The hook subscribes to keydown events. When the user presses `j`/`ArrowDown`/`ArrowRight`, advance selection to the next node in topological order. `k`/`ArrowUp`/`ArrowLeft` retreats. `Enter` invokes `onSelect(currentId)`. `Escape` calls `onSelect(null)`. `/` focuses the arxiv input (if present).

- [ ] **Step 1: Hook**

```ts
"use client";
import { useEffect } from "react";

const ORDER = [
  "src", "read", "env", "plan", "impl",
  "opt", "bb", "aug", "hor", "div",
  "audit", "report"
];

export function useCanvasKeyboardNav({
  selectedId,
  onSelect,
  enabled
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  enabled: boolean;
}) {
  useEffect(() => {
    if (!enabled) return;
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const idx = selectedId ? ORDER.indexOf(selectedId) : -1;
      if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        onSelect(ORDER[Math.min(ORDER.length - 1, idx + 1)] ?? ORDER[0]);
      } else if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        onSelect(ORDER[Math.max(0, idx - 1)] ?? ORDER[ORDER.length - 1]);
      } else if (e.key === "Escape") {
        onSelect(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, onSelect, enabled]);
}
```

- [ ] **Step 2: Wire into `lab-canvas.tsx`**

```tsx
import { useCanvasKeyboardNav } from "@/hooks/use-canvas-keyboard-nav";
// inside LabCanvas:
useCanvasKeyboardNav({ selectedId, onSelect, enabled: true });
```

- [ ] **Step 3: Visual smoke, tests, commit**

Test by tabbing through nodes with `j`/`k` and arrows. Confirm input fields still get arrow-key behavior (the early return on `INPUT`/`TEXTAREA` handles this).

```bash
git add frontend/src/hooks/use-canvas-keyboard-nav.ts frontend/src/components/lab/lab-canvas.tsx
git commit -m "feat(web): keyboard nav through canvas nodes (j/k arrows, Enter, Escape)"
```

---

#### Task 5.4: Shortcut overlay (`?` opens help)

**Files:**
- Create: `frontend/src/components/lab/shortcut-overlay.tsx`
- Create: `frontend/src/components/lab/shortcut-overlay.module.css`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Component**

A modal listing the keyboard shortcuts:
- `Cmd/Ctrl-K` — Command palette
- `j` / `↓` / `→` — Next node
- `k` / `↑` / `←` — Previous node
- `Enter` — Inspect selected node
- `Esc` — Close panels / overlays
- `?` — Toggle this help

- [ ] **Step 2: Wire `?` keypress in `lab-shell.tsx`** (or extract into another hook).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(web): ? opens keyboard shortcut overlay"
```

---

### Week 6 — `/demo` Overlay + Persistent Split Pane

#### Task 6.1: Presentation-mode context + name map

**Files:**
- Create: `frontend/src/lib/presentation-mode.tsx`
- Modify: `frontend/src/components/lab/node-config.ts`
- Modify: `frontend/src/components/lab/lab-shell.tsx`
- Modify: `frontend/src/components/lab/node-card.tsx`

- [ ] **Step 1: Add name map to `node-config.ts`**

Add to the existing file:

```ts
export const DEMO_AGENT_NAMES: Record<string, string> = {
  src: "Paper",
  read: "Reader",
  env: "Forge",
  plan: "Architect",
  impl: "Builder",
  opt: "Vesta",
  bb: "Athena",
  aug: "Orion",
  hor: "Lyra",
  div: "Pyxis",
  audit: "Hermes",
  report: "Scribe"
};

export const INTERNAL_AGENT_NAMES: Record<string, string> = {
  src: "Source",
  read: "paper-understanding",
  env: "environment-detective",
  plan: "reproduction-planner",
  impl: "baseline-implementation",
  opt: "optimizer-path",
  bb: "backbone-path",
  aug: "augmentation-path",
  hor: "horizon-path",
  div: "diffusion-path",
  audit: "supervisor-verifier",
  report: "report-generator"
};
```

The `NODES` array's `agent` field becomes a generic label that gets overridden by the mode.

- [ ] **Step 2: Create presentation-mode context**

`frontend/src/lib/presentation-mode.tsx`:

```tsx
"use client";

import { createContext, useContext, type ReactNode } from "react";

export type PresentationMode = "internal" | "demo";
const Ctx = createContext<PresentationMode>("internal");

export function PresentationModeProvider({
  mode,
  children
}: {
  mode: PresentationMode;
  children: ReactNode;
}) {
  return <Ctx.Provider value={mode}>{children}</Ctx.Provider>;
}

export function usePresentationMode(): PresentationMode {
  return useContext(Ctx);
}
```

- [ ] **Step 3: Update `LabShell` to accept and provide `presentationMode`**

```tsx
export function LabShell({
  initialRun = null,
  initialRecents = [],
  presentationMode = "internal"
}: {
  initialRun?: LiveDemoRunState | null;
  initialRecents?: RecentRunSummary[];
  presentationMode?: "internal" | "demo";
}) {
  return (
    <PresentationModeProvider mode={presentationMode}>
      {/* existing JSX */}
    </PresentationModeProvider>
  );
}
```

- [ ] **Step 4: Update `node-card.tsx` to render the right name**

```tsx
import { usePresentationMode } from "@/lib/presentation-mode";
import { DEMO_AGENT_NAMES, INTERNAL_AGENT_NAMES } from "./node-config";

// inside NodeCard render:
const mode = usePresentationMode();
const agentLabel = (mode === "demo" ? DEMO_AGENT_NAMES : INTERNAL_AGENT_NAMES)[node.id] ?? node.agent;
// use {agentLabel} instead of {node.agent}
```

- [ ] **Step 5: Update `lab/page.tsx` to pass `"internal"`**

```tsx
return <LabShell initialRun={initialRun} initialRecents={initialRecents} presentationMode="internal" />;
```

- [ ] **Step 6: Visual smoke**

Open `/lab` — node labels should be the internal names (`paper-understanding`, `environment-detective`, etc.), not Reader/Forge/etc.

- [ ] **Step 7: Tests, commit**

```bash
cd frontend && npm run test -- --run
git add frontend/src/lib/presentation-mode.tsx frontend/src/components/lab/node-config.ts frontend/src/components/lab/node-card.tsx frontend/src/components/lab/lab-shell.tsx frontend/src/app/lab/page.tsx
git commit -m "feat(web): presentation-mode context; /lab uses internal backend stage names"
```

---

#### Task 6.2: `/demo` route + tour overlay

**Files:**
- Create: `frontend/src/app/demo/page.tsx`
- Create: `frontend/src/components/demo/demo-overlay.tsx`
- Create: `frontend/src/components/demo/demo-overlay.module.css`
- Modify: `frontend/proxy.ts` (restrict unlock gate to `/demo` only)

- [ ] **Step 1: Create the page**

```tsx
import { LabShell } from "@/components/lab/lab-shell";
import { fetchRunById } from "@/lib/demo/server-run";
import { fetchRecentRuns } from "@/lib/runs/server-list";
import { DemoOverlay } from "@/components/demo/demo-overlay";

export const dynamic = "force-dynamic";

export default async function DemoPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const [initialRun, initialRecents] = await Promise.all([
    projectId ? fetchRunById(projectId) : Promise.resolve(null),
    fetchRecentRuns(8)
  ]);
  return (
    <>
      <LabShell initialRun={initialRun} initialRecents={initialRecents} presentationMode="demo" />
      <DemoOverlay />
    </>
  );
}
```

- [ ] **Step 2: Create the overlay**

`DemoOverlay` is a step-through tour. Each step has a target (e.g., node id) and caption. Use absolute positioning over the canvas, computed from the node coordinates exported from `node-config.ts`.

```tsx
"use client";

import { useState } from "react";
import { NODES } from "@/components/lab/node-config";
import styles from "./demo-overlay.module.css";

const STEPS: Array<{ nodeId: string; caption: string }> = [
  { nodeId: "src", caption: "Paper is ingested — PDF or arXiv URL." },
  { nodeId: "read", caption: "Reader extracts the paper's claims and metrics." },
  { nodeId: "env", caption: "Forge rebuilds the runtime environment from scratch." },
  { nodeId: "impl", caption: "Builder runs the baseline implementation." },
  { nodeId: "opt", caption: "Five parallel improvement paths explore the design space." },
  { nodeId: "audit", caption: "Hermes audits whether the claimed result is supported." },
  { nodeId: "report", caption: "Scribe packages the final reproducibility report." }
];

export function DemoOverlay() {
  const [step, setStep] = useState(0);
  if (step >= STEPS.length) return null;
  const target = NODES.find((n) => n.id === STEPS[step].nodeId);
  if (!target) return null;
  return (
    <div className={styles.overlay}>
      <div
        className={styles.tooltip}
        style={{
          left: target.x + 100,
          top: target.y - 60
        }}
      >
        <div className={styles.stepCount}>
          Step {step + 1} of {STEPS.length}
        </div>
        <div className={styles.caption}>{STEPS[step].caption}</div>
        <div className={styles.controls}>
          <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
            Back
          </button>
          <button onClick={() => setStep((s) => s + 1)}>
            {step === STEPS.length - 1 ? "Done" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

Note: the overlay's positioning is approximate. The canvas pans, so a robust implementation reads scroll positions. For Phase B Week 6, this static positioning suffices.

- [ ] **Step 3: Update `proxy.ts` to gate only `/demo`**

Open `frontend/proxy.ts` and confirm the current gate logic. Change the matcher so it only applies to `/demo` and `/api/demo`:

```ts
export const config = {
  matcher: ["/demo/:path*", "/api/demo/:path*"]
};
```

(`/lab` and `/library` are no longer gated — they're internal.)

- [ ] **Step 4: Visual smoke**

Set `REPROLAB_DEMO_SECRET` in `.env`, restart, open `/demo` — unlock prompt appears, after unlock the lab shows with mythological names + the tour overlay. Open `/lab` directly — no unlock needed.

- [ ] **Step 5: Tests, commit**

```bash
cd frontend && npm run test -- --run
git add frontend/src/app/demo frontend/src/components/demo frontend/proxy.ts
git commit -m "feat(web): /demo route with mythological agent names + step-through tour overlay"
```

---

#### Task 6.3: Persistent resizable split pane

**Files:**
- Create: `frontend/src/components/lab/resizable-split.tsx`
- Create: `frontend/src/components/lab/resizable-split.module.css`
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Component**

```tsx
"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import styles from "./resizable-split.module.css";

const STORAGE_KEY = "reprolab:split-ratio";
const MIN_RATIO = 0.25;
const MAX_RATIO = 0.85;

export function ResizableSplit({
  left,
  right
}: {
  left: ReactNode;
  right: ReactNode;
}) {
  const [ratio, setRatio] = useState(0.7);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const value = parseFloat(stored);
        if (Number.isFinite(value) && value >= MIN_RATIO && value <= MAX_RATIO) {
          setRatio(value);
        }
      }
    } catch {
      // localStorage may be disabled
    }
  }, []);

  const persistRatio = useCallback((v: number) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(v));
    } catch {
      // non-fatal
    }
  }, []);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!dragRef.current || !wrapRef.current) return;
      const rect = wrapRef.current.getBoundingClientRect();
      const next = (e.clientX - rect.left) / rect.width;
      const clamped = Math.min(MAX_RATIO, Math.max(MIN_RATIO, next));
      setRatio(clamped);
    }
    function onUp() {
      if (dragRef.current) {
        dragRef.current = false;
        persistRatio(ratio);
      }
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [ratio, persistRatio]);

  return (
    <div ref={wrapRef} className={styles.split}>
      <div className={styles.pane} style={{ flexBasis: `${ratio * 100}%` }}>{left}</div>
      <div className={styles.divider} onMouseDown={() => { dragRef.current = true; }} />
      <div className={styles.pane} style={{ flexBasis: `${(1 - ratio) * 100}%` }}>{right}</div>
    </div>
  );
}
```

- [ ] **Step 2: Use in `lab-shell.tsx`**

Wrap the workflow view in the split:

```tsx
<ResizableSplit
  left={<PanWrap ... />}
  right={<AgentInfoPanel ... />}
/>
```

(Approximate — the existing `workflow-layout` div may need restructuring to fit this shape.)

- [ ] **Step 3: Tests, smoke, commit**

```bash
cd frontend && npm run test -- --run
git add frontend/src/components/lab/resizable-split.tsx frontend/src/components/lab/resizable-split.module.css frontend/src/components/lab/lab-shell.tsx
git commit -m "feat(web): persistent resizable split pane (canvas | side panel)"
```

---

#### Task 6.4: Phase B acceptance

**Files:** none

- [ ] **Step 1: Full test pass**

```bash
cd frontend && npm run test -- --run && npx tsc --noEmit
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_runs_listing.py -v
```

Expected: all pass.

- [ ] **Step 2: e2e re-baseline** (the spec must be updated to match the new file names; this is part of Phase B closure)

```bash
cd frontend && npx playwright test --reporter=line
```

- [ ] **Step 3: Full visual review**

Verify each surface:
- `/` redirects to `/lab`
- `/lab` — internal stage names, no fake data, telemetry strip visible, Cmd-K works, keyboard nav works, split pane is resizable + persists
- `/library` — real run history, filter+sort+search work
- `/paperbench` — same design tokens as `/lab`
- `/demo` — unlock required (if `REPROLAB_DEMO_SECRET` set), mythological names, tour overlay, otherwise same canvas
- No console errors anywhere

- [ ] **Step 4: Final commit + push**

```bash
git push
```

Open a PR back to `claw_demo`.

---

## Phase B Acceptance Criteria

- [ ] `/library` page lists real run history with filter/sort/search
- [ ] Backend `GET /runs` accepts `status`, `q`, `order_by` filters
- [ ] Telemetry strip persistent at bottom of `/lab` (messages, output chars, wall, project id)
- [ ] Cmd-K command palette opens via `Cmd-K` / `Ctrl-K` with: navigate, copy URL, switch to recent run, open final report
- [ ] Keyboard nav: `j`/`k`/arrows cycle nodes, `Enter` inspects, `Esc` closes, `?` opens shortcut overlay
- [ ] `/demo` route exists, gated by unlock secret, uses mythological agent names, includes step-through tour overlay
- [ ] `/lab` and `/library` no longer gated by unlock secret
- [ ] Resizable split pane between canvas and side panel; ratio persists across reloads
- [ ] No `<style jsx global>` anywhere; tokens drive all colors and type
- [ ] All visible technical values (project IDs, durations, costs, hashes) use JetBrains Mono

---

## Self-Review Notes

**Spec coverage check:** Every section of the 6-week spec maps to at least one task above. Surgical-spot items from Quick Wins (landing redirect, honest metadata, Hermes badge, fake recents) map to Tasks 1.2–1.5. The deletion manifest maps to Tasks 1.7–1.10 + 1.11 (globals.css). The backend listing endpoint is Task 1.12. Real recents is Task 1.13. Real progress bar is Task 1.14. Monolith split is Tasks 2.1–2.10. Design tokens + CSS modules + visual direction are Tasks 3.1–3.10. PaperBench unification is Task 3.9. Library is 4.1–4.3. Telemetry strip is 4.4. Cmd-K is 5.1–5.2. Keyboard nav is 5.3. Shortcut overlay is 5.4. Presentation-mode + /demo is 6.1–6.2. Resizable split is 6.3.

**Type consistency check:**
- `WorkflowNode.icon` is typed `IconKey` in `icons.tsx` (Task 2.2) and `string` in `node-config.ts` (Task 2.1). This is intentional: `node-config.ts` doesn't import `IconKey` to avoid a circular dependency. `NodeCard` casts (or uses `as IconKey` at the lookup site).
- `RecentRunSummary` (Task 1.13) is the minimal shape; `RunSummary` (Task 4.3) is a superset for the library page. Both live in `lib/runs/server-list.ts`.
- `PresentationMode` is exported once from `lib/presentation-mode.tsx`. All consumers import from there.
- `useRun`'s returned `startFixtureRun(model)` and `startUploadedRun(file, model)` take the model as an explicit argument; the `LabShell` keeps `model` in local state and passes it at the call site. This avoids the hook needing to know about the model state.

**Placeholders scan:** No "TBD", no "add error handling", no "similar to Task N". Each TDD step has the actual test code. Each implementation step has the actual implementation code. Where tasks span very repetitive work (e.g., Task 3.6 covers four sub-extractions), the pattern is stated once and the file targets enumerated — this is appropriate because a subagent picking up Task 3.6 has its sibling extractions to model on from Tasks 3.3 and 3.4.

**Known limitations of this plan to surface for the engineer:**
- Tasks 3.6 and 3.7 enumerate file moves with less per-CSS-rule detail than Tasks 3.2-3.5. A subagent should follow the same pattern (copy rules from the embedded block to the new module, drop `.reproLab` prefix, update component to import the module, delete moved rules from the embedded block).
- The Playwright e2e spec (`frontend/e2e/lab-e2e-full.spec.ts`, 907 LOC) will drift through Phase A. Re-baselining is part of Task 3.11. Don't try to keep e2e passing on every Week 2 sub-commit — that's not realistic for a refactor of this size.
- The exact CSS rules in the existing `<style jsx global>` block aren't enumerated here (1,571 lines). Subagents executing the Week 3 tasks should `grep` the className they're moving in `lab-shell.tsx` to find the matching rule, port it verbatim, and verify visual equivalence after each move.

---

## Execution Recommendation

This plan is dense and the tasks are heterogeneous (deletions, refactors, new components, backend changes). **Subagent-Driven (recommended)** with two-stage review fits best: fresh subagent per task, review the diff before moving to the next. Inline batch execution risks blowing past a subtle CSS regression in the Week 3 modules sweep.

Phase A is the must-ship slice. If at the Phase A checkpoint priorities shift, the lab is already coherent and honest — Phase B is purely additive.
