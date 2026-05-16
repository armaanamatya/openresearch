# Replix Frontend Polish Pass — Implementation Plan (Phase C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the residual visual + functional issues surfaced by the Playwright audit of the rebuilt frontend (commits up to `1ce3fad` on `frontend-rebuild`), and verify every interactive feature actually works.

**Architecture:** Surgical fixes on top of the rebuilt branch — no architectural changes. Three focused groups: (1) mythological agent names still leak into 3 sites where the workflow view shows internal stage labels in the canvas but mythological names in the right panel — finish the swap; (2) the 1740×720 SVG canvas only shows 3-4 of 12 nodes on a 1200px viewport — re-coordinate the NODES so the whole pipeline fits at the default split ratio; (3) Playwright-driven smoke verification of every interactive feature (Cmd-K, keyboard nav, drag-resize, filters, demo tour) we added in Phase B but never actually exercised end-to-end.

**Tech Stack:** No new deps. Existing: Next.js 16, React 19, TypeScript, Tailwind 3, CSS Modules + global CSS, Vitest, Playwright, `cmdk@1.x`.

---

## Why this plan exists

The post-Phase-B Playwright walk-through (see the audit summary in this conversation) surfaced 5 real bugs and ~6 interactive features we shipped without manually verifying. The plan is short (~20 tasks across 6 groups) because the rebuild itself is sound — what's left is **calibration**, not redesign:

- The `presentation-mode` context only routes labels through `NodeCard`. Three sibling render sites (`RunOverview` subagent list, side-panel title `${selected.agent} activity`, `FloatingAgentWindow` active-agent badge) still pull `node.agent` from the raw `NODES` config, so `/lab` shows `paper-understanding` on the canvas but `Vesta/Athena/...` in the right panel. That asymmetry is the single most user-visible bug.
- The 1740×720 SVG canvas was designed for a no-side-panel desktop layout. After Phase B introduced the persistent resizable split, the visible canvas pane at the default 70/30 ratio is ~700-800px wide. Only 3-4 of 12 nodes are visible without panning. Re-coordinate.
- Phase B added Cmd-K, keyboard nav, `?` shortcut overlay, drag-resize split, demo tour overlay, library filters, debounced search. None of these were Playwright-tested end-to-end — only typecheck + smoke. Verify each one.
- The 3 failing tests in `src/app/api/demo/route.test.ts` are pure production drift (test asserts no `model` field, prod sends `model: "sonnet"`; test asserts no `content-type` header, prod sends it; test expects 400 for empty multipart, prod returns 202). Repair.
- Two cosmetic carry-overs: dead `.edge-drawer-*` CSS in `lab-shell.css` (EdgeDrawer was removed when ResizableSplit replaced it), and the dot-grid canvas background (plan called for hairline grid).

---

## Test baselines (must not regress)

- **Frontend vitest**: 3 failed / 44 passed (47 total). The 3 failures are the test-drift cases in `api/demo/route.test.ts` — fixed in Task C.4. After Task C.4 the target is **0 failed / 47 passed**.
- **Backend pytest**: 23 passed (`tests/test_live_runs_listing.py` + `tests/test_live_run_api.py`). No backend changes in this phase.
- **Frontend typecheck**: clean.

Each task in this plan verifies typecheck + tests before committing. The script:

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected output for tasks C.1.* and earlier: `Tests  3 failed | 44 passed (47)`. Tasks C.4 onward: `Tests  0 failed | 47 passed (47)`.

---

## File Structure

### Files to modify

| Path | Why |
|---|---|
| `frontend/src/components/lab/lab-shell.tsx` | RunOverview subagent list + side-panel title use mythological `node.agent`; need to read `usePresentationMode` and look up `INTERNAL_AGENT_NAMES` (C.1.1, C.1.2). Also drop the orphaned `DRAWER_KEY` constant if still present (C.3.3 sweep). |
| `frontend/src/components/lab/floating-agent-window.tsx` | Active-agent badge uses `activeNode.agent`; same fix (C.1.3). |
| `frontend/src/components/lab/node-config.ts` | Reflow all 12 NODES coordinates so the canvas fits in 1200px (C.2.1). |
| `frontend/src/components/lab/gate-chips.tsx` | `GATE_COORDS` reference edge midpoints derived from the old NODES coordinates; recompute (C.2.1). |
| `frontend/src/components/lab/lab-canvas.tsx` | The SVG width attribute is hard-coded `width={1740}` — change to the new canvas width (C.2.1). |
| `frontend/src/components/lab/lab-canvas.css` | `.canvas-surface { width: 1740px; height: 720px; background-image: radial-gradient(...) }` — change to new dimensions + hairline grid (C.2.2, C.3.1). |
| `frontend/src/components/lab/floating-agent-window.tsx` | The anchor calculation references `1740` (canvas width) and `720` (height) as literals. Update to new canvas size (C.2.1). |
| `frontend/src/components/lab/lab-shell.css` | Strip the `.edge-drawer-*` rules left over after EdgeDrawer was replaced by ResizableSplit (C.3.2). |
| `frontend/src/app/api/demo/route.test.ts` | Repair the 3 drifted assertions (C.4.1). |
| `frontend/src/components/library/library-table.tsx` | Drop the Cost column entirely (backend has no USD field and we don't want to show "—" forever) OR keep but document the gap (C.3.4). |

### Files to create

| Path | Why |
|---|---|
| `tools/seed-fake-run.sh` | Idempotent script that writes a complete fake-run fixture (demo_status.json + agent_telemetry.jsonl + a fake source-pdf + final report) so the UI can be audited end-to-end without firing a real reproduction. Used by Tasks C.5.* for Playwright smoke (C.7.1). |
| `frontend/e2e/lab-smoke-interactive.spec.ts` | Playwright spec that exercises Cmd-K, keyboard nav, `?` overlay, sidebar collapse, resizable split drag + persist, library filters, demo tour next/back — one spec per feature, fast iteration (C.5.*). |

### Files NOT to touch

- `frontend/src/components/lab/lab-shell.tsx` — beyond C.1.1 and C.1.2, no other changes here this phase.
- `frontend/src/hooks/use-run.ts` — SSE + auto-resume is working; leave it.
- `backend/**` — no backend changes this phase.
- `frontend/src/styles/tokens.css` — leave the design tokens alone; this phase is about consuming them correctly, not redefining them.

---

## Conventions

- **Test commands:**
  - Frontend unit: `cd frontend && npm run test -- --run` (vitest)
  - Frontend typecheck: `cd frontend && npx tsc --noEmit`
  - Frontend e2e: `cd frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts`
  - Backend pytest: `.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_runs_listing.py -q`
- **Branch:** continue on `frontend-rebuild` (current branch). No new branch needed.
- **Commits:** one per task, conventional commit format (`fix(web):`, `feat(web):`, `chore(web):`, `test(web):`).
- **Visual smoke:** for each visual change, re-screenshot via Playwright before committing. The dev servers are running on `:3001` (frontend) and `:8000` (backend). If they're down, start with the snippet in **Setup** below.

---

## Setup (if dev servers are not running)

```bash
# Terminal 1 — backend
cd /Volumes/CS_Stuff/Replix
.venv/bin/uvicorn backend.app:create_app --factory --port 8000

# Terminal 2 — frontend (port 3000 is taken by OrbStack on this machine)
cd /Volumes/CS_Stuff/Replix/frontend
REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- -p 3001
```

Verify:

```bash
curl -fs http://127.0.0.1:8000/health
curl -fs http://127.0.0.1:3001 > /dev/null && echo "frontend up"
```

---

## Phase C.1 — Stop the mythological-name leak (3 tasks)

The `presentation-mode` context exists and `NodeCard` reads it, but three sibling sites still pull `node.agent` directly from `NODES`. Result on `/lab`: canvas nodes show `paper-understanding`, but the side panel title says `Reader activity`. This is the #1 visual inconsistency in the product.

### Task C.1.1: RunOverview subagent list reads from `INTERNAL_AGENT_NAMES`

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx` — the `RunOverview` function around line 130-210

- [ ] **Step 1: Locate the subagent list block**

```bash
cd /Volumes/CS_Stuff/Replix
grep -nE "subagents = |item\.node\.agent" frontend/src/components/lab/lab-shell.tsx
```

Expected output: a `const subagents = ["opt", "bb", "aug", "hor", "div"].map(...)` block and a JSX site rendering `{item.node.agent}`.

- [ ] **Step 2: Read the surrounding 30 lines for context**

```bash
sed -n '/const subagents = /,/<\/div>/p' frontend/src/components/lab/lab-shell.tsx | head -40
```

You should see a `.map(...)` that produces `{ id, node: NODES.find(...), state: stateMap[id] }` and renders `<span className="subagent-name">{item.node.agent}</span>`.

- [ ] **Step 3: Import the presentation-mode helpers at the top of `lab-shell.tsx`**

The file already has imports from `./node-config`. Extend that import line:

```tsx
import {
  EDGES,
  NODES,
  PIPELINE_STAGES,
  deriveStage,
  stageProgress,
  stateMapForRun,
  DEMO_AGENT_NAMES,
  INTERNAL_AGENT_NAMES,
  type NodeState,
  type Tone,
  type WorkflowNode
} from "./node-config";
import { usePresentationMode } from "@/lib/presentation-mode";
```

(`usePresentationMode` is exported from `frontend/src/lib/presentation-mode.tsx`; the line is `export function usePresentationMode(): PresentationMode { ... }`.)

- [ ] **Step 4: Inside `RunOverview`, read the presentation mode and compute the right label per subagent**

Find the `subagents` const (around line ~145 after the import). Right above it, add the mode read. Then replace `{item.node.agent}` with the mapped label. Concretely:

```tsx
const mode = usePresentationMode();
const agentNames = mode === "demo" ? DEMO_AGENT_NAMES : INTERNAL_AGENT_NAMES;

const subagents = ["opt", "bb", "aug", "hor", "div"].map((id) => ({
  id,
  node: NODES.find((entry) => entry.id === id)!,
  state: stateMap[id],
  label: agentNames[id] ?? NODES.find((entry) => entry.id === id)!.agent
}));
```

And in the JSX where it renders the list:

```tsx
<span className="subagent-name">{item.label}</span>
```

- [ ] **Step 5: Typecheck**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 6: Tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: `Tests  3 failed | 44 passed (47)` — unchanged.

- [ ] **Step 7: Visual check**

Reload `http://127.0.0.1:3001/lab?projectId=prj_smoke_demo`. The right-side panel's "Improvement sub-agents" list should show `optimizer-path`, `backbone-path`, `augmentation-path`, `horizon-path`, `diffusion-path` (NOT `Vesta`, `Athena`, etc.).

- [ ] **Step 8: Commit**

```bash
cd /Volumes/CS_Stuff/Replix
git add frontend/src/components/lab/lab-shell.tsx
git commit -m "fix(web): RunOverview subagent list reads INTERNAL_AGENT_NAMES

Three sites in the rebuilt frontend still pulled node.agent directly
from NODES (which holds the mythological names as the default field
value). NodeCard was fixed in Phase B Week 6 — finish the swap here.

First of three sibling fixes: the right-panel 'Improvement sub-agents'
list now respects usePresentationMode(), so /lab shows
optimizer-path/backbone-path/... and /demo shows Vesta/Athena/...

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.1.2: Side-panel title `${selected.agent} activity` reads `INTERNAL_AGENT_NAMES`

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx` — the `RightPanel` function

- [ ] **Step 1: Locate the side-panel title**

```bash
cd /Volumes/CS_Stuff/Replix
grep -nE "selected\.agent|side-panel-title" frontend/src/components/lab/lab-shell.tsx
```

You'll find a JSX line like:

```tsx
<div className="side-panel-title">{selected ? `${selected.agent} activity` : "Live activity"}</div>
```

inside the `RightPanel` function.

- [ ] **Step 2: Compute the label in `RightPanel`**

`RightPanel` already receives `run`, `selectedId`, `stateMap`, `error`. Add the presentation-mode read at the top of the function body:

```tsx
function RightPanel({
  error,
  run,
  selectedId,
  stateMap
}: {
  error: string | null;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  const selected = selectedId ? NODES.find((node) => node.id === selectedId) ?? null : null;
  const logEntries = parseLogEntries(run);
  const telemetry = telemetryForSelectedNode(run, selectedId);
  const failedNodeId = failedNodeIdForRun(run, stateMap);
  const mode = usePresentationMode();
  const selectedLabel = selected
    ? (mode === "demo" ? DEMO_AGENT_NAMES : INTERNAL_AGENT_NAMES)[selected.id] ?? selected.agent
    : null;
```

Then replace the JSX line:

```tsx
<div className="side-panel-title">{selectedLabel ? `${selectedLabel} activity` : "Live activity"}</div>
```

- [ ] **Step 3: Typecheck + tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero typecheck errors, `Tests  3 failed | 44 passed (47)` unchanged.

- [ ] **Step 4: Visual check**

Reload `/lab?projectId=prj_smoke_demo`, click one of the canvas node cards (e.g., the leftmost SOURCE). The side-panel title at the bottom-half should now read `Source activity` (internal label), not `Paper activity`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx
git commit -m "fix(web): side-panel title uses INTERNAL_AGENT_NAMES

Second of three sibling fixes for the mythological-name leak. The
RightPanel's section header (currently '\${selected.agent} activity')
pulled the default from NODES.agent — now respects presentationMode.

/lab shows 'paper-understanding activity'; /demo shows 'Reader activity'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.1.3: FloatingAgentWindow active-agent badge reads `INTERNAL_AGENT_NAMES`

**Files:**
- Modify: `frontend/src/components/lab/floating-agent-window.tsx`

- [ ] **Step 1: Locate the badge JSX**

```bash
cd /Volumes/CS_Stuff/Replix
grep -nE "activeNode\.agent" frontend/src/components/lab/floating-agent-window.tsx
```

You'll find two lines:

```tsx
<span className="agent-window-active" title={`Active agent: ${activeNode.agent}`}>
  {activeNode.agent}
</span>
```

- [ ] **Step 2: Add presentation-mode imports**

At the top of `floating-agent-window.tsx`:

```tsx
import { NODES, DEMO_AGENT_NAMES, INTERNAL_AGENT_NAMES, type NodeState } from "./node-config";
import { usePresentationMode } from "@/lib/presentation-mode";
```

(The existing `NODES` import stays; just add the two name maps and the hook.)

- [ ] **Step 3: Compute the label inside the component**

Find the line `const activeNode = useMemo(...)`. Right after that block, add:

```tsx
const mode = usePresentationMode();
const activeLabel =
  (mode === "demo" ? DEMO_AGENT_NAMES : INTERNAL_AGENT_NAMES)[activeNode.id] ?? activeNode.agent;
```

Then replace the two badge JSX usages:

```tsx
<span className="agent-window-active" title={`Active agent: ${activeLabel}`}>
  {activeLabel}
</span>
```

- [ ] **Step 4: Typecheck + tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero typecheck errors, `3 failed | 44 passed`.

- [ ] **Step 5: Visual check**

Reload `/lab?projectId=prj_smoke_demo`. The floating agent window (top-right of the canvas) badge should now read `paper-understanding` (or whichever stage is `running`), not `Reader`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/lab/floating-agent-window.tsx
git commit -m "fix(web): FloatingAgentWindow badge uses INTERNAL_AGENT_NAMES

Third and last sibling fix. The floating live-agent window's
'active agent' badge now respects presentationMode.

After this commit, all four agent-label render sites on /lab show
backend stage IDs consistently. The mythological names only appear
on /demo, as designed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C.2 — Workflow canvas fits in 1200px viewport (2 tasks)

The canvas is currently 1740×720, fixed pixel coordinates. With the persistent resizable split at the default 70/30 ratio, the canvas pane gets ~700-800px wide on a typical 1200-1440px viewport. Only 3-4 of 12 nodes are visible — the user has to drag-pan to reach the optimizer/audit/report nodes.

Fix: re-coordinate all 12 NODES so the whole pipeline fits in ~1100px width × ~600px height. Keep node card dimensions (200×80) — those are dictated by the label text length.

### Task C.2.1: Reflow NODES + update GATE_COORDS + canvas dimensions

**Files:**
- Modify: `frontend/src/components/lab/node-config.ts` — `NODES` array (lines ~50-200)
- Modify: `frontend/src/components/lab/gate-chips.tsx` — `GATE_COORDS` constant (around line 31-35)
- Modify: `frontend/src/components/lab/lab-canvas.tsx` — `<svg width={1740} height={720}>` (around line 50)
- Modify: `frontend/src/components/lab/floating-agent-window.tsx` — `1740` and `720` literals in `anchorX` / `anchorY` (around line 64-67)

The new coordinate plan (canvas 1200×640):

| node id | old x,y | new x,y |
|---|---|---|
| src | 20, 310 | 20, 280 |
| read | 260, 310 | 200, 280 |
| env | 500, 200 | 380, 180 |
| plan | 500, 420 | 380, 380 |
| impl | 740, 310 | 560, 280 |
| opt | 1020, 60 | 760, 40 |
| bb | 1020, 200 | 760, 160 |
| aug | 1020, 340 | 760, 280 |
| hor | 1020, 480 | 760, 400 |
| div | 1020, 620 | 760, 520 |
| audit | 1300, 310 | 940, 280 |
| report | 1540, 310 | 1100, 280 |

Canvas dimensions: 1200×640 (was 1740×720). All nodes are 200×80, so `report` at x=1100 fits with 100px slack to the right edge; `div` at y=520 fits with 40px slack to the bottom (NODE_H=80, so bottom of node = 600).

Edge midpoints for `GATE_COORDS`:

- gate_1: plan(380,380) → impl(560,280), midpoint horizontally = (380+200+560)/2 ≈ 570, vertically = (380+40+280+40)/2 = 370. **New: x=545, y=370**.
- gate_2: impl(560,280) → bb(760,160), midpoint = (560+200+760)/2 ≈ 760, vertically = (280+40+160+40)/2 = 260. **New: x=755, y=260**.
- gate_3: bb(760,160) → audit(940,280), midpoint = (760+200+940)/2 ≈ 950, vertically = 260. **New: x=945, y=260**.

(Exact midpoint math: gate_N x = (source_node.x + NODE_W + target_node.x) / 2; y = (source_node_center_y + target_node_center_y) / 2.)

- [ ] **Step 1: Update NODES coordinates**

Open `frontend/src/components/lab/node-config.ts` and edit each NODE's `x` and `y` per the table above. Touch only the `x` and `y` fields — leave `id`, `agent`, `step`, `icon`, `tone`, `role`, `detail` alone.

Example diff for the first two entries:

```diff
-    id: "src",
-    x: 20,
-    y: 310,
+    id: "src",
+    x: 20,
+    y: 280,
```

Repeat for all 12 entries.

- [ ] **Step 2: Update GATE_COORDS**

Open `frontend/src/components/lab/gate-chips.tsx`. Find `const GATE_COORDS: Record<...>`. Replace its body with:

```tsx
// Edge midpoints in the 1200x640 SVG. Computed from NODES coordinates
// (NODE_W=200, NODE_H=80): right-edge of source node + left-edge of target node.
//   Gate 1: plan(380,380)→impl(560,280)  midpoint (545, 370)
//   Gate 2: impl(560,280)→bb(760,160)    midpoint (755, 260)
//   Gate 3: bb(760,160)→audit(940,280)   midpoint (945, 260)
const GATE_COORDS: Record<"gate_1" | "gate_2" | "gate_3", { x: number; y: number; label: string }> = {
  gate_1: { x: 545, y: 370, label: "Gate 1" },
  gate_2: { x: 755, y: 260, label: "Gate 2" },
  gate_3: { x: 945, y: 260, label: "Gate 3" }
};
```

- [ ] **Step 3: Update the SVG width/height in lab-canvas.tsx**

Open `frontend/src/components/lab/lab-canvas.tsx`. Find the `<svg>` element and change its dimensions:

```tsx
<svg width={1200} height={640} className="canvas-edges" aria-hidden="true">
```

- [ ] **Step 4: Update FloatingAgentWindow canvas-size literals**

Open `frontend/src/components/lab/floating-agent-window.tsx`. Find the `anchorX` / `anchorY` derivation (around line 62-70):

```tsx
// Anchor to the right of the active node; flip left when it would spill
// off the 1200x640 canvas surface, and clamp within it.
const anchorRight = activeNode.x + NODE_W + 28;
const anchorX =
  anchorRight + size.w > 1200 ? Math.max(8, activeNode.x - size.w - 28) : anchorRight;
const anchorY = Math.min(Math.max(8, activeNode.y - 14), 640 - size.h - 8);
```

Replace `1740` → `1200` and `720` → `640`.

- [ ] **Step 5: Typecheck + tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero typecheck errors, `3 failed | 44 passed`. The `lab-shell.test.tsx` tests don't assert on coordinates so this should not regress.

- [ ] **Step 6: Visual check**

Reload `/lab?projectId=prj_smoke_demo`. The canvas should now show ALL 12 nodes within a 1200px-wide pane: src on the left, report on the right, with the 5 improvement paths stacked vertically in the middle column. Gate chips overlay the correct edges.

If a gate chip looks misaligned, the midpoint math was wrong — recompute and adjust the GATE_COORDS y/x.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/lab/node-config.ts frontend/src/components/lab/gate-chips.tsx frontend/src/components/lab/lab-canvas.tsx frontend/src/components/lab/floating-agent-window.tsx
git commit -m "fix(web): reflow workflow canvas to 1200x640 — all 12 nodes fit at default split

The 1740x720 canvas was sized for a no-side-panel layout. After Phase B
introduced the persistent resizable split (default 70/30), the canvas
pane on a 1200-1440px viewport only fit 3-4 of the 12 nodes. The user
had to drag-pan to find the audit/report nodes.

Re-coordinate every node so the whole pipeline fits in 1200x640:
  - x: shrunk from max 1540 to max 1100
  - y: shrunk from max 620 to max 520
  - NODE_W=200, NODE_H=80 unchanged (label readability preserved)

GATE_COORDS recomputed for the new edge midpoints; the SVG width/height
attributes updated; FloatingAgentWindow anchor calculation updated to
clamp inside the new bounds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.2.2: Update `lab-canvas.css` for new dimensions

**Files:**
- Modify: `frontend/src/components/lab/lab-canvas.css`

- [ ] **Step 1: Locate the canvas-surface rule**

```bash
cd /Volumes/CS_Stuff/Replix
grep -nE "canvas-surface" frontend/src/components/lab/lab-canvas.css
```

You'll find:

```css
.canvas-surface {
  position: relative;
  width: 1740px;
  height: 720px;
  background-image: radial-gradient(#dcdce0 1px, transparent 1px);
  background-size: 22px 22px;
  background-color: var(--canvas-bg);
}
```

- [ ] **Step 2: Update the dimensions only (grid swap is Task C.3.1)**

Replace `width: 1740px;` with `width: 1200px;` and `height: 720px;` with `height: 640px;`. Leave the background-image rule alone for now.

- [ ] **Step 3: Visual check**

Reload `/lab?projectId=prj_smoke_demo`. The canvas should now have no extra empty space on the right or below — the surface matches the node bounding box exactly.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/lab-canvas.css
git commit -m "fix(web): canvas-surface CSS matches new 1200x640 dimensions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C.3 — Visual polish (4 tasks)

### Task C.3.1: Replace dot-grid canvas background with hairline grid

The original visual direction (from `docs/superpowers/plans/2026-05-14-frontend-rebuild.md`) called for a horizontal hairline grid on the canvas — "lab notebook, not Mural/Whimsical." Currently the canvas shows a radial-gradient dot pattern.

**Files:**
- Modify: `frontend/src/components/lab/lab-canvas.css`

- [ ] **Step 1: Replace the dot-grid rule**

In `frontend/src/components/lab/lab-canvas.css`, find:

```css
.canvas-surface {
  position: relative;
  width: 1200px;
  height: 640px;
  background-image: radial-gradient(#dcdce0 1px, transparent 1px);
  background-size: 22px 22px;
  background-color: var(--canvas-bg);
}
```

Replace with:

```css
.canvas-surface {
  position: relative;
  width: 1200px;
  height: 640px;
  background-color: var(--canvas-bg);
  background-image:
    linear-gradient(to right, var(--canvas-grid) 0.5px, transparent 0.5px);
  background-size: 22px 22px;
}
```

The horizontal hairlines every 22px give it a lab-notebook feel. `--canvas-grid` is already defined in `tokens.css` as `#dcdce0`.

- [ ] **Step 2: Visual check**

Reload `/lab?projectId=prj_smoke_demo`. The canvas background should show very faint vertical hairlines every 22px instead of dots.

If the hairlines are too prominent, soften by setting the alpha — change `var(--canvas-grid)` to `rgba(220, 220, 224, 0.4)` directly.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/lab/lab-canvas.css
git commit -m "feat(web): hairline canvas grid replaces dot-grid

Visual direction from the rebuild plan: 'lab notebook, not Mural/
Whimsical.' Vertical hairlines every 22px instead of a radial-gradient
dot field.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.3.2: Strip dead `.edge-drawer-*` CSS from `lab-shell.css`

When Phase B Week 6 replaced `EdgeDrawer` with `ResizableSplit`, the CSS rules for the old drawer were left in place. Sweep them.

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.css`

- [ ] **Step 1: List the dead rules**

```bash
cd /Volumes/CS_Stuff/Replix
grep -nE "^\.edge-drawer" frontend/src/components/lab/lab-shell.css
```

You'll see lines for `.edge-drawer`, `.edge-drawer.open`, `.edge-drawer-tab`, `.edge-drawer-tab-label`, `.edge-drawer-head`, `.edge-drawer-head-label`, `.edge-drawer-close`, `.edge-drawer-body`.

- [ ] **Step 2: Verify no consumer remains**

```bash
grep -rn "edge-drawer" frontend/src/
```

Expected: only the lab-shell.css matches (the JSX site was removed in Phase B Week 6 commit `e403d70`). If anything else shows up, STOP and investigate before deleting.

- [ ] **Step 3: Delete the rules**

Open `frontend/src/components/lab/lab-shell.css` and remove all `.edge-drawer*` rule blocks. They form one continuous section — easy to identify.

- [ ] **Step 4: Typecheck + tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero errors, `3 failed | 44 passed`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/lab-shell.css
git commit -m "chore(web): strip dead .edge-drawer-* CSS from lab-shell.css

EdgeDrawer was replaced by ResizableSplit in Phase B Week 6 (commit
e403d70). The CSS rules for the old drawer were never cleaned up — no
JSX consumer references them. Sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.3.3: Drop the `DRAWER_KEY` constant if still present

Same context as C.3.2 — leftover from EdgeDrawer.

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

- [ ] **Step 1: Check if DRAWER_KEY is still referenced**

```bash
cd /Volumes/CS_Stuff/Replix
grep -n "DRAWER_KEY" frontend/src/components/lab/lab-shell.tsx
```

If it returns matches, proceed; if it returns nothing, this task is already done — skip to commit step (or merge with another task).

- [ ] **Step 2: Remove the const and any reference**

Find lines that declare `const DRAWER_KEY = "reprolab:openDrawer";` and any `localStorage.getItem(DRAWER_KEY)` / `localStorage.setItem(DRAWER_KEY, ...)` lines. Delete them. The `detailsOpen` / `toggleDetails` state should also be gone after the ResizableSplit swap — if `detailsOpen` is unused, remove it.

- [ ] **Step 3: Typecheck + tests**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero errors, `3 failed | 44 passed`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx
git commit -m "chore(web): drop unused DRAWER_KEY constant + EdgeDrawer state

Carry-over from the EdgeDrawer → ResizableSplit swap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.3.4: Library: drop the `Cost` column (no backend USD field yet)

The Cost column always renders `—` because `LiveDemoRunState` has no USD field. Showing a permanently-empty column is noise.

**Files:**
- Modify: `frontend/src/components/library/library-table.tsx`
- Modify: `frontend/src/components/library/library-table.module.css` (grid template if it sizes columns)

- [ ] **Step 1: Remove the `Cost` header**

In `frontend/src/components/library/library-table.tsx`, find the `<th>` block. There's a line like `<th className={styles.thRight}>Cost</th>` — delete it.

- [ ] **Step 2: Remove the `Cost` cell from each row**

Same file, find the row map and remove the `<td>` rendering `—` (or whatever USD-derived value). Keep the row to 6 columns: Project, Source, Status, Updated, Reward Δ, Score.

- [ ] **Step 3: Update colspan for the empty-state row**

There's a line like `<td className={styles.emptyCell} colSpan={7}>`. Change `colSpan={7}` to `colSpan={6}`.

- [ ] **Step 4: Add a comment explaining why**

Right above the `<thead>`, drop a one-line comment:

```tsx
{/* Cost column intentionally absent — backend has no USD field on
    LiveDemoRunState. Add when cost-ledger.jsonl rollup is plumbed
    through. */}
```

- [ ] **Step 5: Typecheck + tests + visual**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Reload `/library`. The table should have 6 columns now.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/library/library-table.tsx frontend/src/components/library/library-table.module.css
git commit -m "chore(web): drop empty Cost column from /library

Backend has no USD field on LiveDemoRunState — the column was rendering
'—' for every row. Remove the column rather than leave permanent
visual noise. Comment in the JSX explains how to add it back when
cost-ledger.jsonl rollup is plumbed through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C.4 — Test drift repair (1 task)

### Task C.4.1: Fix the 3 drifted tests in `api/demo/route.test.ts`

The test file asserts on production fetch shapes that have moved. Three failures:
1. "forwards uploaded-paper runs..." — test asserts body without `model` and without `content-type` header; prod sends both
2. "starts fixture runs..." — same drift
3. "rejects multipart requests without a pdf file..." — test expects 400, prod returns 202

**Files:**
- Modify: `frontend/src/app/api/demo/route.test.ts`

- [ ] **Step 1: Read the test file to understand structure**

```bash
sed -n '1,40p' frontend/src/app/api/demo/route.test.ts
```

Look at how the existing tests mock fetch + assert call arguments.

- [ ] **Step 2: Run only the failing tests to see exact errors**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx vitest run src/app/api/demo/route.test.ts 2>&1 | tail -60
```

You'll see 3 assertion failures. Each one shows expected vs received. The fix for each is to update the test's expected payload to match what prod actually sends.

- [ ] **Step 3: Fix "starts fixture runs through the FastAPI JSON route"**

The test currently asserts something like:

```ts
expect(fetch).toHaveBeenCalledWith(
  "http://backend.test/runs",
  expect.objectContaining({
    body: '{"mode":"sdk","provider":"openai","verificationProvider":"anthropic","executionMode":"max","sandbox":"docker","gpuMode":"prefer"}',
    method: "POST"
  })
);
```

Update the body string to include `,"model":"sonnet"}` before the closing brace, and add the `headers` expectation:

```ts
expect(fetch).toHaveBeenCalledWith(
  "http://backend.test/runs",
  expect.objectContaining({
    body: '{"mode":"sdk","provider":"openai","verificationProvider":"anthropic","executionMode":"max","sandbox":"docker","gpuMode":"prefer","model":"sonnet"}',
    headers: expect.objectContaining({ "content-type": "application/json" }),
    method: "POST"
  })
);
```

- [ ] **Step 4: Fix "forwards uploaded-paper runs to the FastAPI upload route"**

Similar shape — find the equivalent block (looks like `expect(fetch).toHaveBeenCalledWith(..., expect.objectContaining({ body: ... }))`). The body for upload runs is FormData, not JSON — the test probably asserts on a content-disposition or similar. Inspect what prod sends:

```bash
sed -n '/startUploadedRun/,/}/p' frontend/src/hooks/use-run.ts | head -25
```

You'll see the FormData fields the hook sets: mode, provider, executionMode, sandbox, gpuMode, model, paper. Update the test's assertion to expect those fields.

- [ ] **Step 5: Fix "rejects multipart requests without a pdf file before proxying"**

This test asserts `response.status` is 400. Open `frontend/src/app/api/demo/route.ts` and search for "Upload a PDF" — if the rejection still exists, the test mock is wrong. If the rejection was removed (prod now returns 202 and the BACKEND validates), the assertion should be `expect(response.status).toBe(202)` and the body should reflect whatever the backend echoes. Update accordingly.

- [ ] **Step 6: Run only this test file**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx vitest run src/app/api/demo/route.test.ts 2>&1 | tail -10
```

Expected: all 3 tests pass.

- [ ] **Step 7: Run the full vitest**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npm run test -- --run 2>&1 | grep -E "Test Files|Tests "
```

Expected: `Tests  0 failed | 47 passed (47)`. This is the long-awaited green baseline.

- [ ] **Step 8: Update the baseline doc**

In `docs/superpowers/plans/baseline-2026-05-14.md`, add a final line:

```markdown
**Updated after Task C.4 (polish-pass):** 0 failed / 47 passed (47 total). All known-broken tests repaired.
```

And strike through the `route.test.ts` row in the disposition table:

```markdown
| ~~`src/app/api/demo/route.test.ts`~~ | ~~3~~ | **Repaired in Task C.4** — production drift (model field + content-type header + 202 vs 400 on empty multipart). |
```

- [ ] **Step 9: Commit**

```bash
git add frontend/src/app/api/demo/route.test.ts docs/superpowers/plans/baseline-2026-05-14.md
git commit -m "test(web): repair 3 drifted /api/demo route tests — baseline now 0/47

Production-code drift, never the test suite's fault:
  - POST body now includes 'model: sonnet' (default selector)
  - fetch() call includes content-type: application/json header
  - Empty multipart returns 202 (backend validates), not 400

Bring the test assertions in line with what production actually does.

Tests now 0 failed / 47 passed — first fully-green baseline since the
frontend-rebuild branch was cut.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C.5 — Interactive smoke verification (6 tasks)

Phase B added Cmd-K, keyboard nav, sidebar collapse, resizable split, library filters, and the demo tour overlay. We typecheck-passed and ship-committed them, but never ran any of them through Playwright. Each task here writes (or extends) one Playwright spec that verifies the feature actually works end-to-end. If any spec finds a bug, fix it in the same task.

### Task C.5.1: Seed fake-run utility script

Before the smoke specs can run, we need a deterministic way to put a "run" into the backend's `runs/` directory so the UI has something to render. This script is reusable for future audits.

**Files:**
- Create: `tools/seed-fake-run.sh`

- [ ] **Step 1: Create the script**

```bash
mkdir -p /Volumes/CS_Stuff/Replix/tools
cat > /Volumes/CS_Stuff/Replix/tools/seed-fake-run.sh <<'SHELL'
#!/usr/bin/env bash
# Seed a fake in-flight run for UI audits.
#
# Writes runs/<project_id>/{demo_status.json,agent_telemetry.jsonl}
# so /lab?projectId=<id> renders the workflow view, /library shows
# the run, and the telemetry strip has non-zero numbers.
#
# Usage:
#   tools/seed-fake-run.sh                 # default project id
#   tools/seed-fake-run.sh prj_my_smoke    # custom project id
#   tools/seed-fake-run.sh --clean         # remove all prj_*_smoke fixtures

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--clean" ]]; then
    find runs -maxdepth 1 -type d -name "prj_*_smoke" -exec rm -r {} +
    echo "removed fake fixtures"
    exit 0
fi

project_id="${1:-prj_diffusion_smoke}"
run_dir="runs/${project_id}"
mkdir -p "${run_dir}"

cat > "${run_dir}/demo_status.json" <<JSON
{
  "projectId": "${project_id}",
  "outputDir": "${run_dir}",
  "runMode": "sdk",
  "llmProvider": "anthropic",
  "executionMode": "efficient",
  "sandboxMode": "runpod",
  "gpuMode": "auto",
  "model": "sonnet",
  "status": "running",
  "sourceKind": "uploaded_pdf",
  "sourceLabel": "Diffusion Policy reproduction",
  "sourceNote": "Smoke fixture",
  "startedAt": "2026-05-14T15:00:00Z",
  "updatedAt": "2026-05-14T15:08:00Z",
  "pid": 1,
  "payload": {
    "summary": { "stage": "baseline_implemented", "doneCount": 6, "totalCount": 14 },
    "decisionLog": ["gate_1: verified", "Baseline: CartPole-v1 selected"],
    "gates": {
      "gate_1": { "passed": true, "status": "verified", "chipStatus": "passed", "detail": "Plan OK" },
      "gate_2": { "passed": false, "chipStatus": "running" },
      "gate_3": { "passed": false, "chipStatus": "pending" }
    },
    "pathStates": { "opt": "upcoming", "bb": "upcoming", "aug": "upcoming", "hor": "upcoming", "div": "upcoming" }
  },
  "log": "[15:00] paper-understanding started\n[15:01] paper-understanding completed\n[15:02] environment-detective started\n[15:04] reproduction-planner: contract written\n[15:04] Gate 1: verified\n[15:05] baseline-implementation started",
  "telemetry": []
}
JSON

cat > "${run_dir}/agent_telemetry.jsonl" <<JSON
{"agent_id":"paper-understanding","model":"claude-sonnet-4-6","duration_seconds":41.2,"message_count":12,"output_chars":8420,"success":true}
{"agent_id":"environment-detective","model":"claude-sonnet-4-6","duration_seconds":18.7,"message_count":6,"output_chars":2150,"success":true}
{"agent_id":"reproduction-planner","model":"claude-sonnet-4-6","duration_seconds":25.3,"message_count":8,"output_chars":3890,"success":true}
{"agent_id":"baseline-implementation","model":"claude-sonnet-4-6","duration_seconds":240.0,"message_count":31,"output_chars":18250}
JSON

echo "seeded ${project_id}"
echo "open: http://127.0.0.1:3001/lab?projectId=${project_id}"
SHELL
chmod +x /Volumes/CS_Stuff/Replix/tools/seed-fake-run.sh
```

- [ ] **Step 2: Test the script**

```bash
cd /Volumes/CS_Stuff/Replix
tools/seed-fake-run.sh prj_test_smoke
curl -s "http://127.0.0.1:8000/runs/prj_test_smoke" | python3 -m json.tool | head -10
```

You should see a JSON document with `"projectId": "prj_test_smoke"`. The pid=1 will likely still get downgraded to "failed" by the live-pid check, but `telemetry` should now show 4 records.

- [ ] **Step 3: Verify telemetry actually shows in the UI**

```bash
open "http://127.0.0.1:3001/lab?projectId=prj_test_smoke"
```

The telemetry strip at the bottom should now read MESSAGES ≥ 57, OUTPUT ≥ 32K (not 0/0). If it's still 0/0, the seed isn't being picked up — investigate.

- [ ] **Step 4: Commit**

```bash
git add tools/seed-fake-run.sh
git commit -m "chore(tools): add seed-fake-run.sh for UI audits

Idempotent script that writes runs/<project_id>/{demo_status.json,
agent_telemetry.jsonl} so the workflow view can be Playwright-tested
without firing a real reproduction. Reusable for future audits.

  tools/seed-fake-run.sh                # default project id
  tools/seed-fake-run.sh prj_my_smoke   # custom
  tools/seed-fake-run.sh --clean        # tear down all prj_*_smoke

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.5.2: Playwright spec — Cmd-K palette opens, filters, navigates

**Files:**
- Create: `frontend/e2e/lab-smoke-interactive.spec.ts`

- [ ] **Step 1: Create the spec with the Cmd-K test**

```ts
import { test, expect } from "@playwright/test";

const LAB_URL = "http://127.0.0.1:3001/lab";

test.beforeEach(async ({ page }) => {
  await page.goto(LAB_URL);
});

test("Cmd-K opens the command palette and fuzzy-search works", async ({ page }) => {
  // Open the palette.
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.getByRole("dialog", { name: /command palette/i }).or(
    page.locator('[cmdk-root]')
  );
  await expect(palette).toBeVisible({ timeout: 2000 });

  // Type into the input — cmdk's <Command.Input>.
  const input = palette.locator('input[cmdk-input]').first();
  await input.fill("library");

  // The "Library" navigate item should be matched + visible.
  const item = palette.getByText("Library", { exact: false }).first();
  await expect(item).toBeVisible();

  // Esc closes.
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden({ timeout: 2000 });
});
```

- [ ] **Step 2: Run the spec**

```bash
cd /Volumes/CS_Stuff/Replix/frontend
npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
```

Expected: pass. If it fails because the dialog role is different (cmdk might not set `role="dialog"`), use the `[cmdk-root]` selector fallback already included in the test. If still failing, inspect the rendered HTML and adjust the selector.

- [ ] **Step 3: Commit**

```bash
cd /Volumes/CS_Stuff/Replix
git add frontend/e2e/lab-smoke-interactive.spec.ts
git commit -m "test(e2e): Cmd-K opens palette, fuzzy-searches, Esc closes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.5.3: Playwright spec — keyboard nav + `?` overlay

**Files:**
- Modify: `frontend/e2e/lab-smoke-interactive.spec.ts`

- [ ] **Step 1: Seed a run so the canvas has nodes to navigate**

The Playwright spec needs a seeded run. Add a beforeAll hook OR run `tools/seed-fake-run.sh prj_kbd_smoke` once via shell and use that project id.

Add this test block to the existing spec:

```ts
test("j/k cycles canvas nodes; ? opens shortcut overlay; Esc closes", async ({ page }) => {
  // Use the seeded fixture project so the canvas renders.
  await page.goto(`${LAB_URL}?projectId=prj_diffusion_smoke`);

  // Wait for the canvas to render.
  await expect(page.locator('.canvas-surface')).toBeVisible({ timeout: 5000 });

  // Click into the canvas area to ensure focus is not on a button.
  await page.locator('.canvas-surface').click({ position: { x: 50, y: 50 } });

  // Press j — should select the first node.
  await page.keyboard.press("j");

  // The side panel title should change away from "Live activity"
  // (the default when nothing is selected) to "<agent-id> activity".
  const sidePanelTitle = page.locator('.side-panel-title');
  await expect(sidePanelTitle).not.toHaveText("Live activity", { timeout: 2000 });

  // Press ? to open the shortcut overlay.
  await page.keyboard.press("?");
  const overlay = page.getByText("Keyboard shortcuts", { exact: false }).first()
    .or(page.locator('[data-shortcut-overlay]'));
  await expect(overlay).toBeVisible({ timeout: 2000 });

  // Esc closes the overlay.
  await page.keyboard.press("Escape");
  await expect(overlay).toBeHidden({ timeout: 2000 });
});
```

- [ ] **Step 2: Seed the run if not already present**

```bash
cd /Volumes/CS_Stuff/Replix && tools/seed-fake-run.sh prj_diffusion_smoke
```

- [ ] **Step 3: Run the spec**

```bash
cd /Volumes/CS_Stuff/Replix/frontend
npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
```

Expected: both tests pass. If the shortcut overlay selector misses, look at `frontend/src/components/lab/shortcut-overlay.tsx` and adjust the selector to whatever distinctive text it renders (e.g., "Cmd/Ctrl-K" is a label inside it).

- [ ] **Step 4: Commit**

```bash
cd /Volumes/CS_Stuff/Replix
git add frontend/e2e/lab-smoke-interactive.spec.ts
git commit -m "test(e2e): j/k cycles nodes; ? overlay opens and closes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.5.4: Playwright spec — resizable split drag + localStorage persistence

**Files:**
- Modify: `frontend/e2e/lab-smoke-interactive.spec.ts`

- [ ] **Step 1: Add the test**

```ts
test("resizable split divider drags and persists ratio in localStorage", async ({ page }) => {
  await page.goto(`${LAB_URL}?projectId=prj_diffusion_smoke`);

  const divider = page.locator('.resizable-split-divider').first();
  await expect(divider).toBeVisible({ timeout: 5000 });

  // Get the initial bounding rect of the left pane.
  const leftPane = page.locator('.resizable-split-pane').first();
  const before = await leftPane.boundingBox();
  if (!before) throw new Error("no left pane bbox");

  // Drag the divider 200px to the left.
  const dividerBox = await divider.boundingBox();
  if (!dividerBox) throw new Error("no divider bbox");
  const startX = dividerBox.x + dividerBox.width / 2;
  const startY = dividerBox.y + dividerBox.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX - 200, startY, { steps: 12 });
  await page.mouse.up();

  // The left pane should now be narrower.
  await page.waitForTimeout(150); // let React re-render
  const after = await leftPane.boundingBox();
  if (!after) throw new Error("no left pane bbox after drag");
  expect(after.width).toBeLessThan(before.width - 100);

  // localStorage should have the new ratio.
  const ratio = await page.evaluate(() => window.localStorage.getItem("reprolab:split-ratio"));
  expect(ratio).not.toBeNull();
  expect(parseFloat(ratio ?? "1")).toBeLessThan(0.7);

  // Reload — ratio should restore.
  await page.reload();
  const restored = await leftPane.boundingBox();
  if (!restored) throw new Error("no left pane bbox after reload");
  expect(Math.abs(restored.width - after.width)).toBeLessThan(20);
});
```

- [ ] **Step 2: Run + commit**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
cd /Volumes/CS_Stuff/Replix && git add frontend/e2e/lab-smoke-interactive.spec.ts && git commit -m "test(e2e): split divider drags + persists ratio across reload

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.5.5: Playwright spec — library filters narrow the table

**Files:**
- Modify: `frontend/e2e/lab-smoke-interactive.spec.ts`

- [ ] **Step 1: Seed two runs with different statuses**

```bash
cd /Volumes/CS_Stuff/Replix
tools/seed-fake-run.sh prj_completed_smoke
# Then edit demo_status.json to set "status": "completed".
sed -i '' 's/"status": "running"/"status": "completed"/' runs/prj_completed_smoke/demo_status.json
```

- [ ] **Step 2: Add the test**

```ts
test("library status filter narrows the table", async ({ page }) => {
  await page.goto("http://127.0.0.1:3001/library");

  // Both seeded runs should be visible under "All".
  await expect(page.locator('table tbody tr').filter({ hasText: "prj_diffusion_smoke" })).toBeVisible({ timeout: 5000 });
  await expect(page.locator('table tbody tr').filter({ hasText: "prj_completed_smoke" })).toBeVisible();

  // Click the "Completed" filter pill.
  await page.getByRole("tab", { name: "Completed" }).click();
  await page.waitForTimeout(500); // debounced fetch

  // Only the completed run should remain.
  await expect(page.locator('table tbody tr').filter({ hasText: "prj_completed_smoke" })).toBeVisible();
  await expect(page.locator('table tbody tr').filter({ hasText: "prj_diffusion_smoke" })).toBeHidden();
});
```

- [ ] **Step 3: Run + commit**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
cd /Volumes/CS_Stuff/Replix && git add frontend/e2e/lab-smoke-interactive.spec.ts && git commit -m "test(e2e): library status filter narrows the table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C.5.6: Playwright spec — demo tour next/back

**Files:**
- Modify: `frontend/e2e/lab-smoke-interactive.spec.ts`

- [ ] **Step 1: Confirm unlock is disabled**

The unlock middleware only fires when `REPROLAB_DEMO_SECRET` is set in `.env`. For local smoke, leave it unset so `/demo` is accessible.

- [ ] **Step 2: Add the test**

```ts
test("demo tour overlay steps forward and back", async ({ page }) => {
  await page.goto("http://127.0.0.1:3001/demo");

  // Tour overlay should be at step 1 by default.
  await expect(page.getByText(/STEP 1 OF 7/i)).toBeVisible({ timeout: 3000 });

  // Click Next twice.
  await page.getByRole("button", { name: /next/i }).click();
  await expect(page.getByText(/STEP 2 OF 7/i)).toBeVisible();
  await page.getByRole("button", { name: /next/i }).click();
  await expect(page.getByText(/STEP 3 OF 7/i)).toBeVisible();

  // Back returns to step 2.
  await page.getByRole("button", { name: /back/i }).click();
  await expect(page.getByText(/STEP 2 OF 7/i)).toBeVisible();

  // Close (×) dismisses the overlay.
  await page.getByRole("button", { name: /close/i }).or(
    page.locator('button:has-text("×")')
  ).click();
  await expect(page.getByText(/STEP 2 OF 7/i)).toBeHidden();
});
```

- [ ] **Step 3: Run + commit**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
cd /Volumes/CS_Stuff/Replix && git add frontend/e2e/lab-smoke-interactive.spec.ts && git commit -m "test(e2e): demo tour next/back step through, close dismisses

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C.6 — Acceptance (1 task)

### Task C.6.1: Full acceptance + summary

**Files:** none (verification only)

- [ ] **Step 1: Full typecheck**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 2: Full vitest**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npm run test -- --run 2>&1 | grep -E "Test Files|Tests "
```

Expected: `Tests  0 failed | 47 passed (47)`.

- [ ] **Step 3: Full Playwright interactive smoke**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -15
```

Expected: 6 tests pass (one per Phase C.5.* task).

- [ ] **Step 4: Backend tests**

```bash
cd /Volumes/CS_Stuff/Replix && .venv/bin/python -m pytest tests/test_live_runs_listing.py tests/test_live_run_api.py -q 2>&1 | tail -3
```

Expected: 23 passed.

- [ ] **Step 5: Visual walkthrough of all 5 routes**

Open each in the browser, look for any regression vs. the post-rebuild baseline screenshots:

- `http://127.0.0.1:3001/` → redirects to `/lab`
- `http://127.0.0.1:3001/lab` → dense upload view, sidebar shows real recents
- `http://127.0.0.1:3001/lab?projectId=prj_diffusion_smoke` → all 12 nodes visible without panning, INTERNAL names on every label site (no `Vesta`/`Athena`/`Hermes` text anywhere)
- `http://127.0.0.1:3001/library` → 6-column table (no Cost), filter pills, search, sort
- `http://127.0.0.1:3001/paperbench` → benchmark form
- `http://127.0.0.1:3001/demo` → mythological names + tour overlay step 1 visible

- [ ] **Step 6: Tear down fake runs (optional)**

```bash
cd /Volumes/CS_Stuff/Replix && tools/seed-fake-run.sh --clean
```

- [ ] **Step 7: No commit — this is a checkpoint**

If everything passes, the polish pass is complete. The branch is ready to push:

```bash
git push -u origin frontend-rebuild
```

---

## Phase C Acceptance Criteria

- [ ] No mythological agent names (`Vesta`, `Athena`, `Orion`, `Lyra`, `Pyxis`, `Hermes`, `Scribe`, `Reader`, `Forge`, `Architect`, `Builder`, `Paper`) appear anywhere on `/lab` — all four render sites use `INTERNAL_AGENT_NAMES`.
- [ ] `/demo` still shows mythological names everywhere (presentation-mode context routes correctly).
- [ ] All 12 workflow nodes are visible on a 1200px viewport without panning. Gate chips overlay the correct edge midpoints.
- [ ] Canvas background is a faint vertical hairline grid, not a dot field.
- [ ] No `.edge-drawer-*` rules in any CSS file. No `DRAWER_KEY` reference in any TS file.
- [ ] `/library` table has 6 columns (no Cost).
- [ ] Frontend vitest: **0 failed / 47 passed (47)**.
- [ ] Frontend Playwright interactive: **6 tests pass** in `e2e/lab-smoke-interactive.spec.ts`.
- [ ] Backend pytest: 23 passed (no regression).
- [ ] Frontend typecheck: clean.

---

## Quick wins to ship before Monday (no rebuild required)

Three of Phase C's items are surgical and ship cleanly even without doing the whole plan:

1. **Mythological-name swap at the 3 leak sites** — Tasks C.1.1 + C.1.2 + C.1.3 together are ~20 lines of changes across 2 files. Single PR, restores visual consistency.
2. **Canvas reflow** — Task C.2.1 is mechanical NODES coordinate edits + 3 constant updates. 10 minutes. The biggest user-visible improvement.
3. **Test repair** — Task C.4.1 brings the test suite to 0 failures. Closes the long-standing baseline rot.

The three combined are ~1 hour of work and clear ~80% of the user-visible cruft.

---

## Risks / defaults

- **`docs/superpowers/plans/baseline-2026-05-14.md` gets updated twice in this plan** — once in Task C.4 (mark `route.test.ts` repaired). If the user re-runs old baseline checks, they should reference the polish-pass updates.
- **Canvas reflow may misalign gate chips at the new midpoints.** The math is checked in Task C.2.1, but a visual mismatch after the commit means recomputing the GATE_COORDS y/x by hand against the new NODES table.
- **No new heavy deps.** No shadcn, no Radix.
- **The 1200×640 canvas is still desktop-only.** Tablets/mobile remain out of scope for the lab tool.
- **Playwright specs run against the live dev server.** They will fail with `ERR_CONNECTION_REFUSED` if the dev server is down — Setup section at top documents how to start it.
- **Seed-fake-run script writes to `runs/`** which is gitignored at the repo root (`/runs/`). The seed is ephemeral; tear down with `tools/seed-fake-run.sh --clean`.

---

## Self-Review Notes

**Spec coverage check:** Every issue surfaced in the post-rebuild audit summary maps to a task above:
- Internal-mode visual inconsistency (3 sites) → Tasks C.1.1, C.1.2, C.1.3
- Canvas 12 nodes don't fit in 1200px → Task C.2.1 + C.2.2
- Dot-grid vs hairline (visual direction carry-over) → Task C.3.1
- Dead `.edge-drawer-*` CSS → Task C.3.2
- DRAWER_KEY leftover → Task C.3.3
- Library cost column shows `—` → Task C.3.4
- 3 failing tests in `api/demo/route.test.ts` → Task C.4.1
- Cmd-K/keyboard nav/split drag/library filters/demo tour never end-to-end tested → Tasks C.5.2 through C.5.6, backed by the seed script C.5.1
- Final acceptance → Task C.6.1

**Placeholder scan:** No "TBD", no "add error handling", no "similar to Task N". Each code step shows actual code or actual commands. The Playwright selector patterns use multiple fallback selectors (e.g., `[cmdk-root]` OR `[role="dialog"]`) because cmdk's exact DOM attributes depend on its version; this is documented inline.

**Type consistency check:**
- `INTERNAL_AGENT_NAMES` and `DEMO_AGENT_NAMES` are imported from `./node-config` consistently (Tasks C.1.1, C.1.2, C.1.3).
- `usePresentationMode` from `@/lib/presentation-mode` consistently (same).
- `GATE_COORDS` keys remain `"gate_1" | "gate_2" | "gate_3"` literal type — no signature change.
- Playwright spec uses `expect(...).toBeVisible({ timeout: ... })` consistently rather than mixing in `waitForSelector`.

**Known limitations of this plan:**
- The canvas reflow keeps the SVG at fixed pixel coords. If the user later wants a fluid canvas (CSS grid or viewBox-scaled SVG), that's a follow-up — not in scope here.
- The seed script doesn't write `final_report.md` or `code/paper.pdf`, so the `report` node's `ScriptPanel` will render with "Benchmark report" pending state. Adding those is one-line additions to the script — left for the engineer's discretion.
- The Playwright specs in C.5.* assume the dev server is running. CI would need a `webServer:` config in `playwright.config.ts`. Out of scope for this polish pass.

---

## Execution Recommendation

This plan is short (~14 tasks) and the tasks are independent enough that **Subagent-Driven** with reviewer + reviewer is the right shape. Each task is small enough that one fresh subagent can complete it in a couple of minutes; the controller's review catches any Playwright selector or coordinate misalignment.

If C.2.1 (canvas reflow) hits visual issues, that's the one task where the subagent's pixel math may need correction — review the screenshot carefully before approving.
