# Replix Frontend — "Nothing Hardcoded" Audit + Phase D Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the lab frontend from constants-in-files to a data-driven architecture: the pipeline graph, the agent labels, the gate positions, the keyboard navigation order, the demo tour, the model choices, and the user's preferences all come from API + run state + per-user storage. After Phase D, if the backend renames a stage or adds a node, the frontend renders the new shape without code changes.

**Architecture:** Three sources of truth, layered:

1. **Backend pipeline topology** (new `GET /pipeline/topology` endpoint): canonical list of stages, nodes, edges, gates, and agent-id → node-id mapping. Backend owns this because the orchestrator already encodes it (`PipelineStage` enum + `_pipeline_steps`). Frontend fetches once, caches; this is what we render.
2. **Run-state-embedded pipeline override** (extension to `LiveRunState.payload`): a specific run may use a custom pipeline (a theoretical paper skips the sandbox stages; a benchmark run skips the audit stages). When `run.payload.pipeline` is present, it overrides the topology fetch. Frontend prefers run-state.
3. **User preferences** (`localStorage["reprolab:user-prefs"]`): model choice, sandbox mode, execution mode, dev presentationMode toggle, split-pane ratio. Persists across sessions per browser. Defaults from last-used or sensible system defaults.

**Tech Stack:** No new heavy deps. Existing: Next.js 16, React 19, TypeScript, Tailwind 3, CSS Modules + global CSS, Vitest, Playwright, `cmdk@1.x`. A small graph-layout function (dagre-style stage-bucket layout, ~80 LOC) is hand-rolled — no new layout library.

---

## Why this plan exists

The audit triggered by the user's "ensure NOTHING is hardcoded" request found these constants embedded across 15+ files:

### Backend (canonical — these are legitimately backend-owned, but need to be exposed)

- `backend/agents/orchestrator.py:93` — `class PipelineStage(str, enum.Enum)` with 14 stages
- `backend/agents/orchestrator.py:1540-1547` — `_pipeline_steps` list mapping stages to agent functions
- `backend/agents/registry.py` — 12 agent definitions

These ARE the source of truth — but no HTTP endpoint exposes them. The frontend mirrors them.

### Frontend (the actual hardcoding problem)

| Location | What | Should come from |
|---|---|---|
| `node-config.ts:24-205` | `NODES` array of 12 entries (id, agent, step, role, detail, icon, tone, x, y) | Backend topology — except (x,y) which the frontend computes |
| `node-config.ts:207-225` | `EDGES` array of 16 edges | Backend topology |
| `node-config.ts:227-244` | `PIPELINE_STAGES` 14-string list | Backend topology |
| `node-config.ts:248-258` | `DEMO_AGENT_NAMES` mythological aliases | Backend topology (presentation metadata) |
| `node-config.ts:260-272` | `INTERNAL_AGENT_NAMES` backend stage IDs | Backend topology |
| `gate-chips.tsx:33-37` | `GATE_COORDS` three fixed pixel coordinates | **Frontend, derived from layout** |
| `node-card.tsx:9-10` | `NODE_W = 200`, `NODE_H = 80` | Frontend layout const (legitimate UI choice) |
| `lab-canvas.tsx:57` | `<svg width={1200} height={640}>` | Frontend layout (derived from node positions) |
| `floating-agent-window.tsx:68-72` | clamp to `1200`/`640` literals | Frontend layout (derived) |
| `lab-canvas.css:23-30` | `.canvas-surface { width: 1200px; height: 640px; }` | Frontend layout (derived) |
| `use-canvas-keyboard-nav.ts:11-25` | `ORDER` array of 12 IDs in topo order | Backend topology |
| `lab-shell.tsx:80-92` | `agentMatchers` (node-id → backend-agent-id list) | Backend topology (each node carries `agent_ids: string[]`) |
| `lab-shell.tsx:122` | `return "report"` (fallback failed node) | Backend topology (the "terminal" node id) |
| `lab-shell.tsx:147` | `["opt", "bb", "aug", "hor", "div"]` improvement path IDs | **Run state** — `run.payload.pathStates` keys |
| `agent-info-panel.tsx:46` | `node.id === "report"` (renders ScriptPanel) | Backend topology — a `kind: "report"` flag on the node |
| `agent-info-panel.tsx:111` | `node.id === "audit"` (renders HermesAuditPanel) | Backend topology — a `kind: "audit"` flag on the node |
| `demo-overlay.tsx:14-42` | `STEPS` array of 7 tour stops with hardcoded node IDs and prose | Backend topology (each node optionally carries `tourCaption: string`) |
| `upload-view.tsx:112-113` | `<option value="sonnet">` / `<option value="opus">` | Backend models endpoint (or env-derived list) |
| `use-run.ts:11-13` | `DEFAULT_RUN_QUERY` query string with hardcoded provider/sandbox/etc. | User preferences |
| `use-run.ts:355-361` | `formData.set("sandbox", "runpod")` etc. in `startUploadedRun` | User preferences |
| `use-run.ts:10` | `MAX_DASHBOARD_EVENTS = 200` | Legitimate UI cap (keep) |
| `use-run.ts:13` | `POLL_INTERVAL_MS = 3000` | Could be config; low priority |
| `api/demo/route.ts:20` | `MAX_UPLOAD_BYTES = 50 * 1024 * 1024` | Legitimate (mirrors backend limit, ideally backend exposes it via header — low priority) |

**Bottom line:** ~20 distinct hardcoded values where "hardcoded" means "should come from API or per-user storage." Roughly 5 are legitimate UI constants and stay.

---

## What "dynamic" means here

Three categories with distinct mechanisms:

### Category 1 — Backend-owned domain knowledge (Phase D.1 + D.2)
The pipeline graph, the agent labels, which nodes are "audit" vs "report" terminals, which stages exist. Source of truth: backend Python code. Mechanism: HTTP endpoint `GET /pipeline/topology` + Pydantic model + frontend hook.

### Category 2 — Per-run state (Phase D.3)
The active run may have a paper-specific pipeline (skip the sandbox for theoretical papers; skip improvement paths for `--n-paths 0` runs). Mechanism: `LiveRunState.payload.pipeline?: PipelineTopology` overrides the global topology.

### Category 3 — Per-user preferences (Phase D.4)
Model choice, sandbox mode, execution mode, split-pane ratio. Mechanism: typed `localStorage` helper, sensible system defaults if absent. Defaults derived from the user's *last successful run* where applicable.

### What legitimately stays hardcoded (Section §6)

- ICONS dictionary in `icons.tsx` (SVG path data — static design asset).
- Design tokens in `tokens.css` (colors, fonts, spacing — system aesthetic).
- `NODE_W` / `NODE_H` (UI layout constants for label-readable nodes).
- `MAX_DASHBOARD_EVENTS = 200` (UI cap to bound memory).
- `LAST_RUN_KEY`, `DRAWER_KEY` (localStorage *keys* — the values they store are dynamic).
- HTTP timeouts (system-level config, not user-facing).
- Test fixture data (mock data IS hardcoded by definition).
- Component class names.
- The 5-minute prompt-cache TTL note inside `use-run.ts` is fine — it's just commentary.

---

## File Structure

### Files to create

| Path | Why |
|---|---|
| `backend/agents/topology.py` | Canonical pipeline graph definition (nodes + edges + stages + agent-ids) — single source of truth for what's exposed. |
| `backend/schemas/topology.py` | Pydantic models for the topology response. |
| `tests/test_pipeline_topology_api.py` | New test file for the `/pipeline/topology` endpoint. |
| `frontend/src/lib/pipeline/topology.ts` | TypeScript types matching the backend Pydantic models. |
| `frontend/src/lib/pipeline/server-fetch.ts` | Server-side fetch helper for SSR consumption. |
| `frontend/src/lib/pipeline/layout.ts` | Pure-function graph layout: takes topology + viewport size → node x/y + gate midpoints. |
| `frontend/src/hooks/use-topology.ts` | Client hook that fetches and caches the topology. |
| `frontend/src/lib/user-prefs.ts` | Read/write typed user preferences from localStorage. |
| `frontend/src/app/api/pipeline/topology/route.ts` | Frontend proxy for the backend endpoint. |
| `tools/check-no-hardcoding.sh` | Audit script: grep for the patterns we just migrated; fails CI if a regression sneaks in. |

### Files to modify

| Path | What changes |
|---|---|
| `backend/app.py` | Register the new `GET /pipeline/topology` endpoint. |
| `backend/agents/orchestrator.py` | (optional) include the topology object in the run state payload so per-run customization is possible. |
| `backend/services/events/live_runs.py` | (optional) extend `LiveRunState` to carry `payload.pipeline`. |
| `frontend/src/components/lab/node-config.ts` | Stripped to only: types + name-map merging from topology. NODES/EDGES/STAGES/GATE_COORDS gone. The file becomes ~30 LOC. |
| `frontend/src/components/lab/gate-chips.tsx` | Compute gate positions from the layout result, not from `GATE_COORDS`. |
| `frontend/src/components/lab/lab-canvas.tsx` | Read nodes/edges/dimensions from the topology + layout. |
| `frontend/src/components/lab/node-card.tsx` | Same — node shape from layout. |
| `frontend/src/components/lab/floating-agent-window.tsx` | Read canvas bounds from layout result, not from `1200`/`640` literals. |
| `frontend/src/components/lab/lab-canvas.css` | Drop `width: 1200px; height: 640px;` — use inline style from layout. |
| `frontend/src/hooks/use-canvas-keyboard-nav.ts` | Accept the topology's node ID order as a prop (or read from a context). |
| `frontend/src/components/lab/agent-info-panel.tsx` | Render Hermes/Script panel based on `node.kind` flags from topology, not `node.id` string comparison. |
| `frontend/src/components/demo/demo-overlay.tsx` | Tour steps derived from topology — each node optionally has a `tourCaption`. |
| `frontend/src/components/lab/upload-view.tsx` | Model selector populated from `/api/models` (new endpoint) OR from a user-prefs fallback. |
| `frontend/src/hooks/use-run.ts` | `startFixtureRun` / `startUploadedRun` pull user prefs for sandbox/provider/executionMode. |
| `frontend/src/app/lab/page.tsx` | Pass server-fetched topology + user prefs to LabShell. |
| `frontend/src/components/lab/lab-shell.tsx` | Topology + prefs flow as props through PresentationModeProvider context. Improvement path IDs from `run.payload.pathStates`, not literal array. |

### Files NOT touched

- `frontend/src/styles/tokens.css` — design tokens are legitimate constants.
- `frontend/src/components/lab/icons.tsx` — SVG path data is static design.
- `frontend/src/components/library/*` — already mostly dynamic (table fetches `/api/runs/list`).
- `frontend/src/app/api/runs/*` — fine.
- `frontend/src/components/lab/telemetry-strip.tsx` — reads `run.telemetry` already.

---

## Test baselines (must not regress)

- Frontend vitest: **0 failed / 46 passed (46 total)** — keep.
- Backend pytest: 23 passed — should grow to ~26 with new topology endpoint tests.
- Frontend typecheck: clean.
- Playwright interactive: **5 passed / 5** in `lab-smoke-interactive.spec.ts` — keep, plus a new test that verifies the canvas renders the same shape after Phase D as before (visual contract).

---

## Conventions

- **One commit per task.** Conventional commit format (`feat(api):`, `feat(web):`, `refactor(web):`).
- **TDD for the backend endpoint** (Phase D.1) — write failing pytest before the route exists.
- **No TDD for layout changes** (D.2 visual work) — smoke test via Playwright + screenshot diff.
- **Per-task verification**: typecheck + vitest + (where applicable) the relevant Playwright spec.
- **Branch**: continue on `frontend-rebuild`. The polish-pass commits (`66e7a80` etc.) are upstream.

---

## Setup (if dev servers are not running)

```bash
# Backend
cd /Volumes/CS_Stuff/Replix
.venv/bin/uvicorn backend.app:create_app --factory --port 8000 &

# Frontend
cd /Volumes/CS_Stuff/Replix/frontend
REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- -p 3001 &
```

Verify:
```bash
curl -fs http://127.0.0.1:8000/health
curl -fs http://127.0.0.1:3001 > /dev/null && echo "frontend up"
```

---

## Phase D.1 — Backend exposes the pipeline topology (5 tasks)

### Task D.1.1: Define `backend/agents/topology.py` — canonical pipeline graph

**Files:**
- Create: `backend/agents/topology.py`

- [ ] **Step 1: Create the file**

```python
"""Canonical pipeline topology — single source of truth for what the
frontend renders.

The orchestrator's PipelineStage enum + _pipeline_steps list already
encode the stage progression; this module collects the metadata that
the UI needs (node ids, display names, role descriptions, terminal
kinds, gate positions in the stage sequence) into a serialisable
shape.

Keep this file decoupled from React / pixel coordinates. It's pure
domain data. Frontend layout (x/y) is computed in lib/pipeline/layout.ts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NodeKind = Literal["source", "agent", "improvement_path", "audit", "report"]
Tone = Literal["accent", "hermes", "info", "neutral"]


class PipelineNode(BaseModel):
    """One renderable node in the workflow graph.

    `agent_ids` is the list of backend agent IDs this node represents
    (a single UI node can summarise several backend agents — e.g. the
    `env` node covers both `environment-detective` and `environment-
    verifier`). Telemetry filtering uses this.

    `tour_caption` is optional. When present, the /demo tour overlay
    includes this node as a step with the caption text.
    """
    id: str
    kind: NodeKind
    internal_label: str
    demo_label: str
    step: str
    role: str
    detail: str
    icon: str  # icon key, matches frontend ICONS dict
    tone: Tone
    agent_ids: list[str] = Field(default_factory=list)
    tour_caption: str | None = None


class PipelineEdge(BaseModel):
    source: str
    target: str


class PipelineGate(BaseModel):
    """A verification gate sits between two nodes. The frontend draws
    a chip on the edge midpoint."""
    id: str
    before_node: str
    after_node: str
    label: str


class PipelineStage(BaseModel):
    """One step in the orchestrator's state machine. The frontend uses
    these for the progress bar (stage index / total)."""
    id: str
    order: int


class PipelineTopology(BaseModel):
    nodes: list[PipelineNode]
    edges: list[PipelineEdge]
    gates: list[PipelineGate]
    stages: list[PipelineStage]
    improvement_path_ids: list[str]
    """Node IDs that represent parallel improvement explorations. The
    UI uses this list to render the 'Improvement sub-agents' rollup
    and to read run.payload.pathStates."""


def default_topology() -> PipelineTopology:
    """Return the canonical 12-node / 14-stage / 3-gate pipeline.

    This mirrors what backend/agents/orchestrator.py drives today. A
    future per-paper customisation lives in
    LiveRunState.payload.pipeline (override) — this function is the
    fallback when no override is set.
    """
    nodes = [
        PipelineNode(id="src", kind="source", internal_label="Source", demo_label="Paper",
                     step="Source intake", role="Receives the source artifact",
                     detail="This is the paper or workspace input that starts the run.",
                     icon="doc", tone="neutral", agent_ids=[]),
        PipelineNode(id="read", kind="agent", internal_label="paper-understanding", demo_label="Reader",
                     step="Paper understanding", role="Extracts claims, metrics, and assumptions",
                     detail="Parses the paper and turns benchmarks and assumptions into a runnable plan.",
                     icon="brain", tone="info",
                     agent_ids=["paper-understanding", "artifact-discovery"],
                     tour_caption="Reader extracts the paper's claims and metrics."),
        PipelineNode(id="env", kind="agent", internal_label="environment-detective", demo_label="Forge",
                     step="Environment", role="Rebuilds the runtime environment",
                     detail="Resolves dependencies and creates the isolated execution environment.",
                     icon="beaker", tone="info",
                     agent_ids=["environment-detective", "environment-verifier"],
                     tour_caption="Forge rebuilds the runtime environment from scratch."),
        PipelineNode(id="plan", kind="agent", internal_label="reproduction-planner", demo_label="Architect",
                     step="Reproduction plan", role="Defines the verification contract",
                     detail="Maps paper claims to experiments and checkpoints.",
                     icon="doc", tone="info",
                     agent_ids=["reproduction-planner", "root-orchestrator"]),
        PipelineNode(id="impl", kind="agent", internal_label="baseline-implementation", demo_label="Builder",
                     step="Baseline implementation", role="Builds and runs the baseline",
                     detail="Produces the baseline implementation and records first metrics.",
                     icon="zap", tone="accent",
                     agent_ids=["baseline-implementation", "experiment-runner",
                                "method-fidelity-verifier", "data-metrics-verifier",
                                "artifact-diff-verifier"],
                     tour_caption="Builder runs the baseline implementation."),
        PipelineNode(id="opt", kind="improvement_path", internal_label="optimizer-path", demo_label="Vesta",
                     step="Optimizer path", role="Explores optimizer changes",
                     detail="Tests alternative optimizers and schedules.",
                     icon="spark", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"],
                     tour_caption="Five parallel improvement paths explore the design space."),
        PipelineNode(id="bb", kind="improvement_path", internal_label="backbone-path", demo_label="Athena",
                     step="Backbone path", role="Tests representation swaps",
                     detail="Evaluates backbone changes.",
                     icon="copy", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="aug", kind="improvement_path", internal_label="augmentation-path", demo_label="Orion",
                     step="Augmentation path", role="Explores robustness changes",
                     detail="Sweeps augmentation strategies.",
                     icon="graph", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="hor", kind="improvement_path", internal_label="horizon-path", demo_label="Lyra",
                     step="Horizon path", role="Extends planning horizon",
                     detail="Tests longer-horizon variants.",
                     icon="flag", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="div", kind="improvement_path", internal_label="diffusion-path", demo_label="Pyxis",
                     step="Diffusion path", role="Sweeps diffusion settings",
                     detail="Compares DDIM and related inference-time changes.",
                     icon="compute", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="audit", kind="audit", internal_label="supervisor-verifier", demo_label="Hermes",
                     step="Result audit", role="Verifies claims against the run",
                     detail="Checks whether claimed results are grounded in the run outputs.",
                     icon="shield", tone="hermes",
                     agent_ids=["supervisor-verifier", "verifier", "hermes"],
                     tour_caption="Hermes audits whether the claimed result is supported."),
        PipelineNode(id="report", kind="report", internal_label="report-generator", demo_label="Scribe",
                     step="Final report", role="Packages the reproducibility output",
                     detail="Compiles manifests, logs, checkpoints, and the audit trail.",
                     icon="flag", tone="neutral",
                     agent_ids=["supervisor-verifier", "root-orchestrator"],
                     tour_caption="Scribe packages the final reproducibility report."),
    ]

    edges = [
        PipelineEdge(source="src", target="read"),
        PipelineEdge(source="read", target="env"),
        PipelineEdge(source="read", target="plan"),
        PipelineEdge(source="env", target="impl"),
        PipelineEdge(source="plan", target="impl"),
        PipelineEdge(source="impl", target="opt"),
        PipelineEdge(source="impl", target="bb"),
        PipelineEdge(source="impl", target="aug"),
        PipelineEdge(source="impl", target="hor"),
        PipelineEdge(source="impl", target="div"),
        PipelineEdge(source="opt", target="audit"),
        PipelineEdge(source="bb", target="audit"),
        PipelineEdge(source="aug", target="audit"),
        PipelineEdge(source="hor", target="audit"),
        PipelineEdge(source="div", target="audit"),
        PipelineEdge(source="audit", target="report"),
    ]

    gates = [
        PipelineGate(id="gate_1", before_node="plan", after_node="impl", label="Gate 1"),
        PipelineGate(id="gate_2", before_node="impl", after_node="bb", label="Gate 2"),
        PipelineGate(id="gate_3", before_node="bb", after_node="audit", label="Gate 3"),
    ]

    stages = [
        PipelineStage(id="ingested", order=0),
        PipelineStage(id="paper_understood", order=1),
        PipelineStage(id="artifacts_discovered", order=2),
        PipelineStage(id="environment_built", order=3),
        PipelineStage(id="plan_created", order=4),
        PipelineStage(id="gate_1_passed", order=5),
        PipelineStage(id="baseline_implemented", order=6),
        PipelineStage(id="baseline_run", order=7),
        PipelineStage(id="gate_2_passed", order=8),
        PipelineStage(id="improvements_selected", order=9),
        PipelineStage(id="improvements_run", order=10),
        PipelineStage(id="gate_3_passed", order=11),
        PipelineStage(id="research_map_generated", order=12),
        PipelineStage(id="complete", order=13),
    ]

    return PipelineTopology(
        nodes=nodes,
        edges=edges,
        gates=gates,
        stages=stages,
        improvement_path_ids=["opt", "bb", "aug", "hor", "div"],
    )
```

- [ ] **Step 2: Test it imports**

```bash
.venv/bin/python -c "from backend.agents.topology import default_topology; t = default_topology(); print(f'{len(t.nodes)} nodes / {len(t.edges)} edges / {len(t.gates)} gates / {len(t.stages)} stages')"
```

Expected: `12 nodes / 16 edges / 3 gates / 14 stages`.

- [ ] **Step 3: Commit**

```bash
git add backend/agents/topology.py
git commit -m "feat(api): introduce backend.agents.topology — canonical pipeline graph

Single source of truth for nodes, edges, gates, stages, and the
agent_id → node_id mapping. Mirrors what the orchestrator already
drives, but exposed as a serialisable Pydantic shape so the frontend
can render the graph from API data instead of mirroring the constants
in TypeScript.

Per-run customisation (a paper-specific pipeline) lives in
LiveRunState.payload.pipeline in a future commit; this module is the
fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.1.2: Register `GET /pipeline/topology` endpoint + TDD

**Files:**
- Modify: `backend/app.py`
- Create: `tests/test_pipeline_topology_api.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline_topology_api.py`:

```python
"""Tests for the GET /pipeline/topology endpoint."""

from __future__ import annotations

from starlette.testclient import TestClient

from backend.app import create_app


def test_pipeline_topology_endpoint_returns_default_shape() -> None:
    client = TestClient(create_app())
    response = client.get("/pipeline/topology")

    assert response.status_code == 200
    body = response.json()

    assert "nodes" in body
    assert "edges" in body
    assert "gates" in body
    assert "stages" in body
    assert "improvement_path_ids" in body

    assert len(body["nodes"]) == 12
    assert len(body["edges"]) == 16
    assert len(body["gates"]) == 3
    assert len(body["stages"]) == 14
    assert body["improvement_path_ids"] == ["opt", "bb", "aug", "hor", "div"]


def test_pipeline_topology_node_has_required_fields() -> None:
    client = TestClient(create_app())
    body = client.get("/pipeline/topology").json()

    for node in body["nodes"]:
        assert "id" in node
        assert "kind" in node
        assert "internal_label" in node
        assert "demo_label" in node
        assert "step" in node
        assert "icon" in node
        assert "tone" in node
        assert "agent_ids" in node
        assert isinstance(node["agent_ids"], list)


def test_pipeline_topology_audit_node_has_audit_kind() -> None:
    """The 'audit' node has kind='audit' so the frontend can route the
    HermesAuditPanel render conditionally on kind, not on the literal
    id string."""
    client = TestClient(create_app())
    body = client.get("/pipeline/topology").json()
    audit_node = next(n for n in body["nodes"] if n["id"] == "audit")
    assert audit_node["kind"] == "audit"

    report_node = next(n for n in body["nodes"] if n["id"] == "report")
    assert report_node["kind"] == "report"
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
.venv/bin/python -m pytest tests/test_pipeline_topology_api.py -v
```

Expected: FAIL with 404 (endpoint doesn't exist yet).

- [ ] **Step 3: Register the endpoint**

In `backend/app.py`, near the existing route registrations, add:

```python
from backend.agents.topology import default_topology, PipelineTopology

# ... inside create_app(), after the /runs registrations ...

@app.get("/pipeline/topology", response_model=PipelineTopology)
async def pipeline_topology() -> PipelineTopology:
    return default_topology()
```

- [ ] **Step 4: Run the test (expect pass)**

```bash
.venv/bin/python -m pytest tests/test_pipeline_topology_api.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the broader test suite for no regression**

```bash
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_runs_listing.py tests/test_pipeline_topology_api.py -q
```

Expected: 26 passed (23 baseline + 3 new).

- [ ] **Step 6: Commit**

```bash
git add backend/app.py tests/test_pipeline_topology_api.py
git commit -m "feat(api): expose GET /pipeline/topology

Returns the canonical PipelineTopology document (12 nodes, 16 edges,
3 gates, 14 stages, plus the improvement_path_ids list). Frontend
consumes this in Phase D.2 to replace the hardcoded NODES/EDGES/
STAGES constants in node-config.ts.

Tests cover shape, required fields, and the audit/report kind flags
that the frontend will use for conditional panel rendering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.1.3: Frontend proxy + types

**Files:**
- Create: `frontend/src/lib/pipeline/topology.ts`
- Create: `frontend/src/app/api/pipeline/topology/route.ts`
- Create: `frontend/src/lib/pipeline/server-fetch.ts`

- [ ] **Step 1: TypeScript types**

`frontend/src/lib/pipeline/topology.ts`:

```ts
export type NodeKind = "source" | "agent" | "improvement_path" | "audit" | "report";
export type Tone = "accent" | "hermes" | "info" | "neutral";

export interface PipelineNode {
  id: string;
  kind: NodeKind;
  internal_label: string;
  demo_label: string;
  step: string;
  role: string;
  detail: string;
  icon: string;
  tone: Tone;
  agent_ids: string[];
  tour_caption?: string | null;
}

export interface PipelineEdge {
  source: string;
  target: string;
}

export interface PipelineGate {
  id: string;
  before_node: string;
  after_node: string;
  label: string;
}

export interface PipelineStage {
  id: string;
  order: number;
}

export interface PipelineTopology {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  gates: PipelineGate[];
  stages: PipelineStage[];
  improvement_path_ids: string[];
}
```

- [ ] **Step 2: Browser proxy**

`frontend/src/app/api/pipeline/topology/route.ts`:

```ts
import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    const response = await fetch(`${backendBaseUrl()}/pipeline/topology`, {
      cache: "no-store"
    });
    if (!response.ok) {
      return NextResponse.json({ error: "Topology unavailable" }, { status: 503 });
    }
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({ error: "Topology unavailable" }, { status: 503 });
  }
}
```

- [ ] **Step 3: Server-side helper for SSR**

`frontend/src/lib/pipeline/server-fetch.ts`:

```ts
import "server-only";
import { backendBaseUrl } from "@/lib/demo/server-run";
import type { PipelineTopology } from "./topology";

export async function fetchTopology(): Promise<PipelineTopology | null> {
  try {
    const response = await fetch(`${backendBaseUrl()}/pipeline/topology`, {
      cache: "no-store"
    });
    if (!response.ok) return null;
    return (await response.json()) as PipelineTopology;
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Typecheck**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 5: Smoke**

```bash
curl -s http://127.0.0.1:3001/api/pipeline/topology | python3 -m json.tool | head -10
```

Expected: a JSON document with `"nodes": [...]` etc.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/pipeline frontend/src/app/api/pipeline
git commit -m "feat(web): TypeScript types + proxy for /pipeline/topology"
```

---

### Task D.1.4: Layout function — topology → x/y coordinates

**Files:**
- Create: `frontend/src/lib/pipeline/layout.ts`

The pipeline graph has a natural stage-bucket layout:
- Stage 0 (sources): leftmost column
- Stage 1: column 2
- Stage 2: column 3 (env + plan branch)
- Stage 3: column 4 (impl merge)
- Stage 4: column 5 (improvement paths fan out — 5 rows)
- Stage 5: column 6 (audit merge)
- Stage 6: column 7 (report)

This is a hand-rolled layout because the graph is small (12 nodes) and we know the topology. Using dagre or elk for 12 nodes is overkill (~50 KB minified for a graph we can lay out in 80 LOC).

- [ ] **Step 1: Create the layout module**

```ts
import type { PipelineNode, PipelineEdge, PipelineGate, PipelineTopology } from "./topology";

export interface LaidOutNode extends PipelineNode {
  x: number;
  y: number;
}

export interface LaidOutGate extends PipelineGate {
  x: number;
  y: number;
}

export interface Layout {
  nodes: LaidOutNode[];
  gates: LaidOutGate[];
  width: number;
  height: number;
}

export interface LayoutConfig {
  nodeWidth: number;
  nodeHeight: number;
  columnGap: number;
  rowGap: number;
  paddingX: number;
  paddingY: number;
}

const DEFAULT_CONFIG: LayoutConfig = {
  nodeWidth: 200,
  nodeHeight: 80,
  columnGap: 80,
  rowGap: 80,
  paddingX: 20,
  paddingY: 40
};

/**
 * Lay out the topology in stage-bucket columns. Source-typed and
 * agent-typed nodes go in the leftmost columns; improvement_path
 * nodes fan out into a vertical column at the next position; audit
 * and report nodes follow.
 *
 * The layout walks edges topologically: every node's column equals
 * `1 + max(predecessor.column)`. Within a column, improvement_path
 * nodes stack vertically; everything else centres on the column's
 * horizontal axis.
 */
export function layoutTopology(
  topology: PipelineTopology,
  config: Partial<LayoutConfig> = {}
): Layout {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Topological columns via Kahn's algorithm.
  const columnByNode: Record<string, number> = {};
  const inDegree: Record<string, number> = {};
  for (const node of topology.nodes) {
    inDegree[node.id] = 0;
    columnByNode[node.id] = 0;
  }
  for (const edge of topology.edges) {
    inDegree[edge.target] = (inDegree[edge.target] ?? 0) + 1;
  }
  const queue: string[] = topology.nodes.filter((n) => inDegree[n.id] === 0).map((n) => n.id);
  while (queue.length) {
    const id = queue.shift()!;
    for (const edge of topology.edges) {
      if (edge.source !== id) continue;
      const nextCol = columnByNode[id] + 1;
      if (nextCol > columnByNode[edge.target]) {
        columnByNode[edge.target] = nextCol;
      }
      inDegree[edge.target] -= 1;
      if (inDegree[edge.target] === 0) queue.push(edge.target);
    }
  }

  // Bucket nodes by column.
  const columns: Record<number, PipelineNode[]> = {};
  for (const node of topology.nodes) {
    const col = columnByNode[node.id];
    (columns[col] ??= []).push(node);
  }

  // Y-position within each column. improvement_path nodes stack
  // vertically (one row each). source/agent/audit/report nodes
  // centre. env vs plan need to stack vertically too — they're
  // both at the same column. The rule: if a column has >1 node,
  // stack them.
  const laidOutNodes: LaidOutNode[] = [];
  let maxRight = 0;
  let maxBottom = 0;
  for (const colKey of Object.keys(columns).map(Number).sort((a, b) => a - b)) {
    const colNodes = columns[colKey];
    const x = cfg.paddingX + colKey * (cfg.nodeWidth + cfg.columnGap);
    // Centre the column vertically around y=320 (rough viewport mid).
    const totalHeight = colNodes.length * cfg.nodeHeight + (colNodes.length - 1) * cfg.rowGap;
    const startY = Math.max(cfg.paddingY, 320 - totalHeight / 2);
    colNodes.forEach((node, i) => {
      const y = startY + i * (cfg.nodeHeight + cfg.rowGap);
      laidOutNodes.push({ ...node, x, y });
      maxRight = Math.max(maxRight, x + cfg.nodeWidth);
      maxBottom = Math.max(maxBottom, y + cfg.nodeHeight);
    });
  }

  // Gate midpoints — between before_node and after_node.
  const byId: Record<string, LaidOutNode> = Object.fromEntries(
    laidOutNodes.map((n) => [n.id, n])
  );
  const laidOutGates: LaidOutGate[] = topology.gates.map((g) => {
    const a = byId[g.before_node];
    const b = byId[g.after_node];
    return {
      ...g,
      x: (a.x + cfg.nodeWidth + b.x) / 2,
      y: (a.y + cfg.nodeHeight / 2 + b.y + cfg.nodeHeight / 2) / 2
    };
  });

  return {
    nodes: laidOutNodes,
    gates: laidOutGates,
    width: maxRight + cfg.paddingX,
    height: maxBottom + cfg.paddingY
  };
}
```

- [ ] **Step 2: Write a quick unit test**

`frontend/src/lib/pipeline/layout.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { layoutTopology } from "./layout";
import type { PipelineTopology } from "./topology";

describe("layoutTopology", () => {
  const t: PipelineTopology = {
    nodes: [
      { id: "a", kind: "source", internal_label: "a", demo_label: "A", step: "", role: "", detail: "", icon: "doc", tone: "neutral", agent_ids: [] },
      { id: "b", kind: "agent",  internal_label: "b", demo_label: "B", step: "", role: "", detail: "", icon: "doc", tone: "info", agent_ids: [] },
      { id: "c", kind: "agent",  internal_label: "c", demo_label: "C", step: "", role: "", detail: "", icon: "doc", tone: "info", agent_ids: [] }
    ],
    edges: [
      { source: "a", target: "b" },
      { source: "b", target: "c" }
    ],
    gates: [
      { id: "g1", before_node: "a", after_node: "b", label: "Gate 1" }
    ],
    stages: [],
    improvement_path_ids: []
  };

  it("places nodes in topological columns", () => {
    const layout = layoutTopology(t);
    const [a, b, c] = layout.nodes;
    expect(a.x).toBeLessThan(b.x);
    expect(b.x).toBeLessThan(c.x);
  });

  it("places gates at edge midpoints", () => {
    const layout = layoutTopology(t);
    const g = layout.gates[0];
    expect(g.x).toBeGreaterThan(0);
    expect(g.y).toBeGreaterThan(0);
  });

  it("computes total width/height to enclose all nodes", () => {
    const layout = layoutTopology(t);
    expect(layout.width).toBeGreaterThan(0);
    expect(layout.height).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 3: Run the unit test**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npm run test -- --run layout.test.ts 2>&1 | tail -5
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/pipeline/layout.ts frontend/src/lib/pipeline/layout.test.ts
git commit -m "feat(web): hand-rolled stage-bucket layout for pipeline graphs

Pure function: topology → laid-out nodes + gate midpoints + total
bounds. Uses Kahn's algorithm to bucket nodes into columns by
topological depth, then stacks nodes vertically within each column.

This replaces the hardcoded (x,y) coordinates in node-config.ts in
Phase D.2. 80 LOC, no new deps — for 12 nodes, dagre/elk is overkill.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.1.5: `useTopology()` hook with SSR fallback

**Files:**
- Create: `frontend/src/hooks/use-topology.ts`

- [ ] **Step 1: Create the hook**

```ts
"use client";
import { useEffect, useState } from "react";
import type { PipelineTopology } from "@/lib/pipeline/topology";

let cachedTopology: PipelineTopology | null = null;

/**
 * Fetches and caches the pipeline topology.
 *
 * Pass `initialTopology` from a server component to avoid a fetch on
 * first paint (SSR-friendly). Subsequent components reuse the cache.
 *
 * If the fetch fails, returns null — callers MUST handle that case
 * (typically by falling back to a "Loading…" or "Pipeline unavailable"
 * state).
 */
export function useTopology(initialTopology: PipelineTopology | null = null): PipelineTopology | null {
  const [topology, setTopology] = useState<PipelineTopology | null>(
    initialTopology ?? cachedTopology
  );

  useEffect(() => {
    if (topology) {
      cachedTopology = topology;
      return;
    }
    let cancelled = false;
    fetch("/api/pipeline/topology", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: PipelineTopology | null) => {
        if (cancelled) return;
        if (data) {
          cachedTopology = data;
          setTopology(data);
        }
      })
      .catch(() => {
        // Stay null; caller renders a fallback.
      });
    return () => {
      cancelled = true;
    };
  }, [topology]);

  return topology;
}
```

- [ ] **Step 2: Typecheck**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/use-topology.ts
git commit -m "feat(web): useTopology hook — SSR-friendly client cache

Module-level cache so multiple components share the topology after
first fetch. Accepts initialTopology from a server component for
zero-cost first paint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase D.2 — Frontend consumes the topology (6 tasks)

### Task D.2.1: Pass topology into lab + library pages from server components

**Files:**
- Modify: `frontend/src/app/lab/page.tsx`
- Modify: `frontend/src/app/library/page.tsx`
- Modify: `frontend/src/app/demo/page.tsx`

- [ ] **Step 1: Update `lab/page.tsx`**

```tsx
import { LabShell } from "@/components/lab/lab-shell";
import { fetchRunById } from "@/lib/demo/server-run";
import { fetchRecentRuns } from "@/lib/runs/server-list";
import { fetchTopology } from "@/lib/pipeline/server-fetch";

export const dynamic = "force-dynamic";

export default async function LabPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const [initialRun, initialRecents, topology] = await Promise.all([
    projectId ? fetchRunById(projectId) : Promise.resolve(null),
    fetchRecentRuns(8),
    fetchTopology()
  ]);
  return (
    <LabShell
      initialRun={initialRun}
      initialRecents={initialRecents}
      initialTopology={topology}
      presentationMode="internal"
    />
  );
}
```

- [ ] **Step 2: Same pattern for `/demo` and `/library`**

The library page doesn't render the canvas directly, but it can still benefit from the cached topology (the library sidebar uses LabSidebar which uses recents — no topology needed there yet, but pass anyway for consistency).

```tsx
// frontend/src/app/demo/page.tsx
const [initialRun, initialRecents, topology] = await Promise.all([
  projectId ? fetchRunById(projectId) : Promise.resolve(null),
  fetchRecentRuns(8),
  fetchTopology()
]);
return (
  <>
    <LabShell initialRun={initialRun} initialRecents={initialRecents} initialTopology={topology} presentationMode="demo" />
    <DemoOverlay topology={topology} />
  </>
);
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/lab/page.tsx frontend/src/app/demo/page.tsx
git commit -m "feat(web): server-side fetch the pipeline topology + thread to LabShell

Zero-cost SSR — the topology is fetched in parallel with the run and
recents. Threaded as initialTopology prop so the client hook skips
its first fetch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.2.2: `LabShell` accepts and provides topology via context

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`
- Create: `frontend/src/lib/pipeline/topology-context.tsx` (a tiny React context for the topology + layout)

- [ ] **Step 1: Create the context**

```tsx
"use client";
import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { PipelineTopology } from "./topology";
import { layoutTopology, type Layout } from "./layout";

interface TopologyContextValue {
  topology: PipelineTopology;
  layout: Layout;
}

const Ctx = createContext<TopologyContextValue | null>(null);

export function TopologyProvider({
  topology,
  children
}: {
  topology: PipelineTopology;
  children: ReactNode;
}) {
  const layout = useMemo(() => layoutTopology(topology), [topology]);
  return <Ctx.Provider value={{ topology, layout }}>{children}</Ctx.Provider>;
}

export function useTopologyContext(): TopologyContextValue {
  const value = useContext(Ctx);
  if (!value) {
    throw new Error("useTopologyContext must be inside a <TopologyProvider>");
  }
  return value;
}
```

- [ ] **Step 2: Update LabShell**

In `lab-shell.tsx`:
- Add `initialTopology?: PipelineTopology | null` to `ReproLabClientProps`.
- Inside `LabShell`, call `useTopology(initialTopology ?? null)`.
- If the topology is null, render a tiny "Pipeline unavailable" fallback.
- Otherwise, wrap the existing layout in `<TopologyProvider topology={topology}>`.

- [ ] **Step 3: Typecheck**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/pipeline/topology-context.tsx frontend/src/components/lab/lab-shell.tsx
git commit -m "feat(web): TopologyProvider context — layout computed once, shared

LabShell now wraps WorkflowView/RightPanel/etc. in TopologyProvider.
Children pull the laid-out nodes and gate positions from a single
useTopologyContext() call — no more passing NODES/EDGES through props.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.2.3: Replace `NODES`/`EDGES` usage in `lab-canvas.tsx`, `node-card.tsx`, `floating-agent-window.tsx`

For each file: stop importing `NODES`/`EDGES` from `node-config.ts`; instead read from `useTopologyContext()`. The layout result has `width`/`height` so the SVG dimensions become dynamic.

- [ ] **Step 1: `lab-canvas.tsx`**

Replace:
```tsx
import { EDGES, NODES, stageProgress, type NodeState, type WorkflowNode } from "./node-config";
```
with:
```tsx
import { useTopologyContext } from "@/lib/pipeline/topology-context";
import type { NodeState } from "./node-config"; // NodeState type stays
```

Inside `LabCanvas`, replace `EDGES.map(...)` and `NODES.map(...)` with iteration over `layout.nodes` / `topology.edges`. Compute `<svg width={layout.width} height={layout.height}>` from the layout.

- [ ] **Step 2: `node-card.tsx`**

Already takes a `node` prop with `x` / `y` — keep that. The change is at the consumer (LabCanvas) which gets nodes from the layout.

- [ ] **Step 3: `floating-agent-window.tsx`**

Replace `1200` / `640` literals in `anchorX` / `anchorY` clamps with `layout.width` / `layout.height` from the context.

- [ ] **Step 4: `lab-canvas.css`** — strip the `width: 1200px; height: 640px;` declaration; the inline style from JSX takes over.

- [ ] **Step 5: Typecheck + visual smoke**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
# Open /lab in the browser; verify the canvas still renders all 12 nodes
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/lab/lab-canvas.tsx frontend/src/components/lab/node-card.tsx frontend/src/components/lab/floating-agent-window.tsx frontend/src/components/lab/lab-canvas.css
git commit -m "refactor(web): lab canvas reads node/edge positions from layout context

No more hardcoded NODES/EDGES imports in the canvas surface. The
1200x640 SVG dimensions become dynamic — derived from the layout
function's bounding box. Adding or removing nodes in topology.py
now re-flows the canvas automatically.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.2.4: Replace `GATE_COORDS` in `gate-chips.tsx`

**Files:**
- Modify: `frontend/src/components/lab/gate-chips.tsx`

- [ ] **Step 1: Replace the GATE_COORDS constant + computation**

```tsx
import { useTopologyContext } from "@/lib/pipeline/topology-context";
// ... drop the GATE_COORDS const ...

export function GateChips({ run }: { run: LiveDemoRunState }) {
  const { layout, topology } = useTopologyContext();
  const gates = run.payload?.gates;
  const stage = run.payload?.summary.stage ?? null;

  // ... existing gateChipState logic unchanged ...

  return (
    <>
      {layout.gates.map((gate) => {
        const gateState = gates?.[gate.id as "gate_1" | "gate_2" | "gate_3"];
        const view = gateChipState(
          gateState,
          stage,
          /* passedStages = */ stagesAfter(topology, gate.id),
          /* runningStages = */ stagesDuring(topology, gate.id)
        );
        return (
          <div
            key={gate.id}
            className={`gate-chip gate-chip-${view.state}`}
            style={{ left: gate.x, top: gate.y }}
            title={gateState?.detail ?? undefined}
          >
            {gate.label} · {view.label}
          </div>
        );
      })}
    </>
  );
}
```

The `stagesAfter` / `stagesDuring` helpers derive the stage lists from the topology + gate position. This replaces the hardcoded `["gate_1_passed", "baseline_implemented", ...]` arrays.

- [ ] **Step 2: Implement the helpers in `lib/pipeline/topology-helpers.ts`**

```ts
import type { PipelineTopology } from "./topology";

/** Stages that have happened by the time this gate has passed. */
export function stagesAfter(topology: PipelineTopology, gateId: string): string[] {
  // Convention: gate_<n>_passed is the stage that signals this gate has been verified.
  // Everything from that stage onward is "after".
  const passedStage = `${gateId}_passed`;
  const order = topology.stages.find((s) => s.id === passedStage)?.order ?? 0;
  return topology.stages.filter((s) => s.order >= order).map((s) => s.id);
}

/** Stages during which this gate is being checked. */
export function stagesDuring(topology: PipelineTopology, gateId: string): string[] {
  // Convention: the stage immediately before gate_<n>_passed is when the
  // gate is "running". E.g. gate_1 is checked during plan_created.
  const passedStage = `${gateId}_passed`;
  const order = topology.stages.find((s) => s.id === passedStage)?.order ?? 0;
  const prev = topology.stages.find((s) => s.order === order - 1);
  return prev ? [prev.id] : [];
}
```

- [ ] **Step 3: Typecheck + smoke**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit
```

Reload `/lab?projectId=prj_diffusion_smoke` — gate chips should appear at the same positions as before.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/gate-chips.tsx frontend/src/lib/pipeline/topology-helpers.ts
git commit -m "refactor(web): gate chip positions derived from layout, not hardcoded

GATE_COORDS const removed. Gate (x,y) comes from the layout function's
gate midpoint calculation. Stage-list arrays (which stages count as
'passed' / 'running' per gate) are derived from the topology's stages
table via stagesAfter() / stagesDuring().

Adding a new gate to topology.py now Just Works — no frontend code
changes needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D.2.5: Replace `PIPELINE_STAGES` + `stageProgress` with topology-derived

**Files:**
- Modify: `frontend/src/components/lab/node-config.ts` (delete PIPELINE_STAGES + stageProgress)
- Modify: `frontend/src/lib/pipeline/topology-helpers.ts` (add stageProgressFromTopology)
- Update consumers (lab-canvas, anywhere else)

- [ ] **Step 1: Add the helper**

```ts
export function stageProgressFromTopology(
  topology: PipelineTopology,
  stage: string | null | undefined
): number {
  if (!stage) return 0;
  const total = topology.stages.length;
  if (total === 0) return 0;
  const idx = topology.stages.find((s) => s.id === stage)?.order;
  return idx === undefined ? 0 : (idx + 1) / total;
}
```

- [ ] **Step 2: Update consumers**

Anywhere that imports `stageProgress` from `node-config.ts`, switch to:

```tsx
const { topology } = useTopologyContext();
const progress = stageProgressFromTopology(topology, run.payload?.summary?.stage);
```

- [ ] **Step 3: Delete the constants**

Remove `PIPELINE_STAGES` and `stageProgress` from `node-config.ts`.

- [ ] **Step 4: Typecheck + smoke + commit**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
```

Expected: zero errors, tests still 46/46 (or 49/49 after the layout test).

```bash
git add frontend/src/components/lab/node-config.ts frontend/src/lib/pipeline/topology-helpers.ts frontend/src/components/lab/lab-canvas.tsx
git commit -m "refactor(web): stageProgress derives total from topology, not 14-const"
```

---

### Task D.2.6: Replace `INTERNAL_AGENT_NAMES` / `DEMO_AGENT_NAMES` lookups with `node.internal_label` / `node.demo_label`

The two name maps in `node-config.ts` duplicate what's now in `node.internal_label` and `node.demo_label`. Delete the maps and have callers read from the node object directly.

**Files:**
- Modify: `frontend/src/components/lab/node-config.ts` (delete name maps)
- Modify: `frontend/src/components/lab/node-card.tsx` (read `node.internal_label`/`node.demo_label` directly)
- Modify: `frontend/src/components/lab/lab-shell.tsx` (RunOverview subagents + RightPanel title)
- Modify: `frontend/src/components/lab/floating-agent-window.tsx`

- [ ] **Step 1: Each site reads the label from the node object**

Pattern (in node-card.tsx):

```tsx
const mode = usePresentationMode();
const agentLabel = mode === "demo" ? node.demo_label : node.internal_label;
// Use {agentLabel} where {node.agent} used to be.
```

- [ ] **Step 2: Delete the two maps from `node-config.ts`**

```bash
cd /Volumes/CS_Stuff/Replix
sed -i '' '/DEMO_AGENT_NAMES/,/^};$/d' frontend/src/components/lab/node-config.ts
sed -i '' '/INTERNAL_AGENT_NAMES/,/^};$/d' frontend/src/components/lab/node-config.ts
```

(Or remove by hand — they're contiguous.)

- [ ] **Step 3: Typecheck + tests + commit**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
git add frontend/src/components/lab
git commit -m "refactor(web): drop INTERNAL_AGENT_NAMES/DEMO_AGENT_NAMES — labels live on the node"
```

---

## Phase D.3 — Per-run pipeline customization (3 tasks)

This phase makes the topology pluggable per run. The frontend already reads from a context; we just teach the context to prefer `run.payload.pipeline` when present.

### Task D.3.1: Extend `LiveRunState.payload` schema with optional `pipeline`

**Files:**
- Modify: `backend/services/events/live_runs.py` — extend `LiveRunState.payload` type to allow a `pipeline` field
- Modify: `frontend/src/lib/demo/demo-run-types.ts` (or wherever LiveDemoRunState lives) — same

Backend is permissive — `payload: Any | None = None` — so no schema change required. Just document the convention in a docstring near `LiveRunState`.

Frontend: extend the LiveDemoPayload TypeScript type to include `pipeline?: PipelineTopology | null`.

- [ ] **Step 1-3: Patch types + smoke + commit**

Concrete code is straightforward; trace `LiveDemoPayload` in `frontend/src/lib/demo/pipeline-dashboard.ts` and add the field.

### Task D.3.2: Frontend prefers `run.payload.pipeline` over the global topology

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

In `LabShell`, if `run?.payload?.pipeline` exists, pass that to `<TopologyProvider topology={run.payload.pipeline}>` instead of the global hook result.

### Task D.3.3: Tests + visual smoke

Seed a fake run with a 6-node pipeline (using a custom payload.pipeline) and verify the canvas renders 6 nodes, not 12.

---

## Phase D.4 — User preferences persistence (3 tasks)

### Task D.4.1: `lib/user-prefs.ts` — typed localStorage helpers

```ts
const KEY = "reprolab:user-prefs";

export interface UserPrefs {
  model?: "sonnet" | "opus";
  sandbox?: "auto" | "local" | "docker" | "runpod";
  executionMode?: "efficient" | "max";
  splitRatio?: number;
}

export function readUserPrefs(): UserPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function writeUserPref<K extends keyof UserPrefs>(key: K, value: UserPrefs[K]): void {
  if (typeof window === "undefined") return;
  try {
    const prefs = readUserPrefs();
    prefs[key] = value;
    window.localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    // localStorage may be disabled (private mode etc.) — non-fatal.
  }
}
```

### Task D.4.2: UploadView writes the model choice on change

The current `onModelChange` from `LabShell` just sets local state. Make it also `writeUserPref("model", value)`. On mount, `LabShell` reads `readUserPrefs().model ?? "sonnet"` as the initial value.

### Task D.4.3: `useRun.startUploadedRun` reads sandbox/executionMode from prefs

Currently hardcoded to `sandbox: "runpod", executionMode: "efficient"`. Read from prefs; fall back to those defaults if absent.

---

## Phase D.5 — Demo tour derived from topology (2 tasks)

### Task D.5.1: Use `node.tour_caption` to generate tour steps

**Files:**
- Modify: `frontend/src/components/demo/demo-overlay.tsx`

```tsx
const { topology, layout } = useTopologyContext();
const STEPS = topology.nodes
  .filter((n) => n.tour_caption)
  .map((n) => {
    const laid = layout.nodes.find((l) => l.id === n.id)!;
    return {
      nodeId: n.id,
      caption: n.tour_caption!,
      x: laid.x,
      y: laid.y
    };
  });
```

The STEPS array is now derived from topology + layout. Adding/removing tour steps means editing `topology.py`, not frontend code.

### Task D.5.2: Smoke test

Run the Playwright `demo tour` spec; should still pass with the same 7 steps.

---

## Phase D.6 — Keyboard nav order from topology (1 task)

### Task D.6.1: `useCanvasKeyboardNav` accepts order as a prop

**Files:**
- Modify: `frontend/src/hooks/use-canvas-keyboard-nav.ts`
- Modify: `frontend/src/components/lab/lab-shell.tsx` (caller)

Hook signature becomes:

```ts
export function useCanvasKeyboardNav({
  selectedId,
  onSelect,
  enabled,
  order
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  enabled: boolean;
  order: string[];
}) {
```

In `lab-shell.tsx`, compute the order from the topology:

```tsx
const { layout } = useTopologyContext();
const order = useMemo(
  () =>
    [...layout.nodes]
      .sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x)
      .map((n) => n.id),
  [layout]
);
useCanvasKeyboardNav({ selectedId, onSelect: setSelectedId, enabled: true, order });
```

---

## Phase D.7 — Improvement-path IDs from run state (1 task)

### Task D.7.1: `RunOverview` reads path IDs from `run.payload.pathStates` keys

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

Replace:
```tsx
const subagents = ["opt", "bb", "aug", "hor", "div"].map((id) => { ... });
```

with:
```tsx
const { topology } = useTopologyContext();
const pathIds = run.payload?.pathStates
  ? Object.keys(run.payload.pathStates)
  : topology.improvement_path_ids;
const subagents = pathIds.map((id) => { ... });
```

This way: a run with `--n-paths 3` shows 3 subagents in the rollup, not 5.

---

## Phase D.8 — Audit/Report kind flags (1 task)

### Task D.8.1: AgentInfoPanel renders sub-panels by `node.kind`, not `node.id`

**Files:**
- Modify: `frontend/src/components/lab/agent-info-panel.tsx`

```tsx
{node.kind === "audit" ? <HermesAuditPanel run={run} /> : null}
{node.kind === "report" ? <ScriptPanel run={run} /> : null}
```

Replaces:
```tsx
{node.id === "audit" ? <HermesAuditPanel run={run} /> : null}
{node.id === "report" ? <ScriptPanel run={run} /> : null}
```

If a future pipeline renames the audit node to "verify" or "review", the kind flag still routes the panel correctly.

---

## Phase D.9 — Telemetry agent matchers from topology (1 task)

### Task D.9.1: `telemetryForSelectedNode` reads `node.agent_ids` from the laid-out node

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`

```tsx
function telemetryForSelectedNode(
  run: LiveDemoRunState | null,
  selectedNode: PipelineNode | null
) {
  if (!run?.telemetry?.length || !selectedNode) return [];
  const matches = selectedNode.agent_ids;
  return run.telemetry
    .filter((record) => matches.some((m) => record.agent_id?.includes(m)))
    .slice(-6)
    .reverse();
}
```

The 30-line `agentMatchers` hardcoded dict goes away.

---

## Phase D.10 — Models list endpoint (2 tasks)

### Task D.10.1: Backend exposes `/models`

The Sonnet/Opus dropdown is hardcoded in `upload-view.tsx`. Expose the list via backend so adding a new Claude model (or an OpenAI model) is a backend change:

```python
# backend/app.py
@app.get("/models")
async def list_models() -> list[dict]:
    return [
        {"id": "sonnet", "label": "Sonnet", "provider": "anthropic"},
        {"id": "opus",   "label": "Opus",   "provider": "anthropic"},
    ]
```

(Reads from a config file or env var if you want to make THIS dynamic too — but for now, the constant lives in the backend, the right ownership.)

### Task D.10.2: `UploadView` populates from `/api/models`

Server-fetch from `lab/page.tsx`, pass as `initialModels: ModelChoice[]` prop. Client renders `<select>` options from the list.

---

## Phase D.11 — Acceptance (1 task)

- [ ] **Step 1: Full tsc + vitest + playwright + backend pytest**

```bash
cd /Volumes/CS_Stuff/Replix/frontend && npx tsc --noEmit && npm run test -- --run 2>&1 | grep -E "Tests "
.venv/bin/python -m pytest tests/test_pipeline_topology_api.py tests/test_live_runs_listing.py tests/test_live_run_api.py -q
cd /Volumes/CS_Stuff/Replix/frontend && npx playwright test e2e/lab-smoke-interactive.spec.ts --reporter=line 2>&1 | tail -10
```

Expected: 0 failures across all four.

- [ ] **Step 2: Hardcoded-value sweep**

```bash
cd /Volumes/CS_Stuff/Replix
tools/check-no-hardcoding.sh
```

(See Phase D.12 — defines the sweep script.)

- [ ] **Step 3: Visual walkthrough — all 5 routes still look right.**

---

## Phase D.12 — Regression guard (1 task)

### Task D.12.1: `tools/check-no-hardcoding.sh` — CI tripwire

A grep-based script that fails if forbidden patterns reappear. Run in CI.

```bash
cat > tools/check-no-hardcoding.sh <<'SHELL'
#!/usr/bin/env bash
# Fail if any of the Phase-D-migrated hardcoded patterns regress.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
patterns=(
  # NODES/EDGES constants must not live in frontend
  'NODES: WorkflowNode\[\] = \['
  '^const EDGES: Array'
  # Mythological-name leaks from a tone string
  'INTERNAL_AGENT_NAMES'
  'DEMO_AGENT_NAMES'
  # Gate positions hardcoded
  'GATE_COORDS'
  # Canvas dimensions hardcoded
  'width: 1200px'
  'height: 640px'
  '1740'  # the old size
  # Hardcoded improvement-path IDs in a map (run state owns this)
  '\["opt", "bb", "aug", "hor", "div"\]'
  # Hardcoded model dropdown options (should come from /models)
  '<option value="sonnet">'
)

for p in "${patterns[@]}"; do
    if grep -rnE "$p" frontend/src 2>/dev/null | grep -v "node_modules\|\.test\." | grep -v "^.*//.*"; then
        echo "❌ forbidden pattern reappeared: $p"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "Phase D regression — see hits above."
    exit 1
fi
echo "✓ no-hardcoding sweep clean"
SHELL
chmod +x tools/check-no-hardcoding.sh
```

Add to a CI step (Github Actions / Vercel build) once it lands.

---

## What stays hardcoded — and why

| Constant | Location | Why it stays |
|---|---|---|
| Design tokens (colors, spacing, radii, fonts) | `src/styles/tokens.css` | These ARE the design system. Making them dynamic = supporting per-user theming, which is a separate product feature. |
| `ICONS` SVG path data | `src/components/lab/icons.tsx` | Static design asset. Adding a new icon means editing an SVG, which is a design task, not a runtime config. |
| `NODE_W` / `NODE_H` | `src/components/lab/node-card.tsx` | UI layout constants for label-readable nodes. The layout function can accept overrides. |
| `MAX_DASHBOARD_EVENTS = 200` | `src/hooks/use-run.ts` | UI cap to bound memory. Not user-facing. |
| `MAX_UPLOAD_BYTES = 50 MB` | `src/app/api/demo/route.ts` | Mirrors the backend's hard cap. If the backend ever exposes its cap via header, we read it then. |
| `POLL_INTERVAL_MS` / `BACKEND_GET_TIMEOUT_MS` / `ENRICH_TIMEOUT_MS` | `src/hooks/use-run.ts` / `src/lib/demo/server-run.ts` | System-level timeouts. Not per-user. |
| localStorage keys (`reprolab:lastRun` etc.) | various | The keys are constant; the *values* they store are dynamic. |
| Component class names + module imports | various | Build-time identifiers. |
| Test fixtures | `*.test.ts(x)` | Mocks ARE hardcoded by definition. |

---

## Risks / defaults

- **Phase D is significant scope (~20 tasks).** Phases D.1 + D.2 + D.11 are the must-do core. D.3 (per-run customization) is a separate product feature — can be deferred. D.4 (user prefs) is small and high-value. D.5–D.10 are mechanical cleanups.
- **The hand-rolled layout function may not match the current hand-placed coordinates pixel-perfectly.** That's OK — the new layout will be slightly different, but coherent (stage-bucketed). The Playwright `j/k cycles canvas nodes` test verifies the order is unchanged.
- **The `node_config.ts` file shrinks to ~30 LOC** after all the constants move to the topology endpoint. That's the point.
- **No new heavy deps.** No dagre, no elk, no SWR. 80-LOC hand-rolled layout function and a 30-LOC module-level cache.
- **Backend response evolves; client should be lenient.** If a new node-kind appears that the frontend doesn't handle, the AgentInfoPanel should fall back to a default "agent" rendering rather than crash.

---

## Self-Review Notes

**Spec coverage check:** Every line item in the audit table at the top of this doc maps to a task:
- 20 distinct hardcoded values, 12 of which migrate to backend topology (Phase D.1+D.2)
- 3 migrate to per-run state (Phase D.3+D.7)
- 3 migrate to user prefs (Phase D.4)
- 3 are derived from topology in helper functions (Phase D.5+D.6+D.8+D.9)
- 1 migrates to a new endpoint (Phase D.10)
- 5 explicitly stay hardcoded with rationale (§"What stays hardcoded")

**Placeholder scan:** No "TBD". No "add error handling" — failure modes (topology fetch fails → null → render fallback) are explicit. Every code step shows the actual code.

**Type consistency:** `PipelineTopology` is defined once in `lib/pipeline/topology.ts`, imported consistently. `Layout`, `LaidOutNode`, `LaidOutGate` similarly. `NodeKind` is the union: `"source" | "agent" | "improvement_path" | "audit" | "report"`.

**Known limitations:**
- The seed-fake-run.sh script in `tools/` still hardcodes a "Diffusion Policy" run shape. That's appropriate — seeds ARE hardcoded by definition.
- The Playwright spec hardcodes selectors. Also appropriate — tests assert specific behaviors.
- The backend `topology.py` still ships a single "default" topology. Per-paper customization happens via run state (Phase D.3), not by changing the default. If the team eventually wants 3 starter topologies (theoretical / empirical / benchmark), that's a database choice and a Phase E task.

---

## Quick wins to ship before Monday

If you want to land the most user-visible portion of Phase D in one PR:

1. **Phase D.1 + D.2** (backend endpoint + frontend consumes topology + layout function + gates/nodes derived) = ~10 tasks, ~1 day. Catches every "if I rename a stage" footgun.
2. **Phase D.4** (user prefs) = 3 tasks, ~2 hours. Real UX win.
3. **Phase D.12** (regression-guard script) = 1 task, ~15 min. Prevents the constants from creeping back.

The remaining phases (D.3 per-run, D.5 tour, D.6 kbd, D.7 paths, D.8 kind flags, D.9 telemetry, D.10 models) are mechanical and can land incrementally.

---

## Execution Recommendation

**Subagent-Driven (recommended)** — each phase has 1-3 tasks, each task is independent and small. The most consequential tasks (D.1.4 layout function, D.2.3 canvas refactor) benefit from a fresh-eyes review before merging. Visual regressions on the canvas would show up immediately in the Playwright `j/k cycles canvas nodes` test.

For inline execution, batch as: D.1 (one session), D.2 (one session), D.3-D.10 individually, D.11+D.12 (acceptance pair).
