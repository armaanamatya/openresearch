# GEPA Native Floating Panel — Design Spec

**Date:** 2026-06-02  
**Branch:** feat/gepa-integration  
**Status:** Approved for implementation

## Problem

`REPROLAB_GEPA_OPTIMIZATION` is feature-flagged and the SSE events for GEPA
(`gepa_phase_start`, `gepa_candidate_proposed/accepted/rejected`,
`gepa_phase_complete`) already flow through `dashboard_events.jsonl` and are
partially parsed by `use-rlm-run`. However:

1. The lab UI renders `gepa_candidate` tree nodes but displays no phase-level
   summary (score timeline, candidate list, delta).
2. The `gepa-viz` server-based approach (`frontend/src/app/api/gepa-viz/`) is
   broken: `gepa-viz==0.1.0` is listed in `requirements.txt` but not installed
   in the venv, so the proxy always 503s.
3. There is no path to view per-primitive GEPA optimization results at all.

## Solution

Build a **native floating React panel** (`GEPAPhasePanel`) that reads GEPA
phase data accumulated from the existing SSE stream. Zero external dependencies,
zero process management. The gepa-viz proxy remains as-is (does not need
removal; it will 503 gracefully and is now unused).

Three parallel fix tracks:

| Track | Scope |
|-------|-------|
| A — GEPA panel | New component + hook changes + rlm-lab wiring |
| B — Pending bug fixes | Commit BUG-046/047/048/050 + lab-shell hydration (already written) |
| C — BUG-049 | Same-tier retry before GPU tier escalation on TRANSIENT_500 |

---

## Track A — GEPA Native Panel

### 1. Data Model

Two new interfaces, added to `frontend/src/hooks/use-rlm-run.ts` and exported:

```ts
export interface GEPACandidateRecord {
  candidate_id: string;
  prompt_preview: string;
  score?: number;
  score_delta?: number;
  reason?: string;       // rejection reason when status === 'rejected'
  status: 'proposed' | 'accepted' | 'rejected';
}

export interface GEPAPhaseData {
  primitive_name: string;
  status: 'running' | 'complete';
  max_metric_calls?: number;  // from gepa_phase_start
  candidates: GEPACandidateRecord[];
  // Populated on gepa_phase_complete:
  seed_score?: number;
  best_score?: number;
  delta?: number;
  total_metric_calls?: number;
  duration_s?: number;
}
```

`RlmRunState` gains a new field:

```ts
gepaPhases: Map<string, GEPAPhaseData>  // keyed by primitive_name
```

Initial value: `new Map()`.

### 2. Hook Changes — `use-rlm-run.ts`

Add `GepaPhaseStartEvent` and `GepaPhaseCompleteEvent` to imports from
`rlm-events.ts` (both are already defined there; they just aren't imported).

Add four fold helpers (replacing the current no-op branches):

**`foldGepaPhaseStart`** — creates a new `GEPAPhaseData` entry with
`status: 'running'`, `max_metric_calls` from the event, and empty `candidates`.
If an entry already exists for this primitive (re-runs within one session), it
is replaced.

**`foldGepaPhaseComplete`** — finds the existing phase entry and merges in
`seed_score` (= `ev.baseline_score`), `best_score` (= `ev.final_score`),
`delta`, `total_metric_calls`, `duration_s`, and flips `status` to `'complete'`.

**`foldGepaCandidateProposed`** — existing function already creates a
`gepa_candidate` tree node. Extend it to also push a new `GEPACandidateRecord`
(`status: 'proposed'`) onto the matching `gepaPhases` entry. If no phase entry
exists yet (event ordering edge case), create a stub phase entry.

**`foldGepaCandidateAccepted` / `foldGepaCandidateRejected`** — extend to
update the matching `GEPACandidateRecord` in `gepaPhases` (set `score`,
`score_delta`, `reason`, flip `status`). Tree-node updates are unchanged.

The `fold` switch case for `gepa_phase_start` and `gepa_phase_complete` changes
from `return seeded` to the new helpers.

`RlmRunState` and `INITIAL_RLM_STATE` are updated accordingly. `useRlmRun`
return type picks up `gepaPhases` automatically since it returns the full state.

### 3. New Component — `gepa-phase-panel.tsx`

**File:** `frontend/src/components/lab/rlm/gepa-phase-panel.tsx`  
**Companion CSS:** `gepa-phase-panel.module.css`

Props:

```ts
interface GEPAPhasePanelProps {
  phaseData: GEPAPhaseData;
  onClose: () => void;
}
```

Structure (top to bottom):

1. **Draggable header** — amber dot, `GEPA · {primitive_name}`, close ×.
   Drag via `onMouseDown` on the header; `mousemove`/`mouseup` on `document`
   update an `offset` state (`[dx, dy]`). Starting position: fixed, bottom-right
   of the canvas (CSS `bottom: 24px; right: 388px` — clears the 360px sidebar).

2. **Score timeline** — horizontal bar chart. One bar per candidate in
   `candidates` order, colored by status: accepted = green `#22c55e`, rejected =
   red `#ef4444`, proposed (in-flight) = amber `#f59e0b` with a pulse animation.
   Height proportional to `score` (0–1 range). Seed score bar rendered first
   (gray, uses `phaseData.seed_score` when available). Minimum bar height 4px so
   zero-score bars are visible. Tooltip on hover: `score: 0.XX`.

3. **Candidate list** — scrollable (`max-height: 200px`). Each row:
   - Left accent border: green/red/amber by status.
   - Status badge (`✓ accepted` / `✗ rejected` / `… proposed`).
   - Score + delta (only when `score` is present): `0.81 (+0.14)`.
   - `prompt_preview` in a `<code>` element, truncated to 80 chars.
   - Rejection `reason` shown as a dim sub-line when present.

4. **Summary footer** — three stat blocks: `seed → best = delta`, plus
   `metric calls: N / max`. When `status === 'running'`: "optimizing…" pulse
   next to calls counter. When `total_metric_calls === 0` (timeout / 0
   candidates): warning line "seed prompt used — optimization timed out".

Panel is `position: fixed`, `z-index: 50`, amber border `#f59e0b55`,
`background: #111827`, `border-radius: 10px`, `width: 320px`,
`box-shadow: 0 12px 40px rgba(0,0,0,0.5)`.

No external chart library. The bar chart is `div` elements with inline height.

### 4. Integration — `rlm-lab.tsx`

Add state:

```ts
const [openGEPAPrimitive, setOpenGEPAPrimitive] = useState<string | null>(null);
```

Modify the `onSelectNode` / `setSelectedNodeId` call site: when the selected
node is `kind === "gepa_candidate"`, also call
`setOpenGEPAPrimitive(node.gepaInfo!.primitive_name)`. Clicking any non-GEPA
node does NOT close the panel — it stays open. The close button is the only
dismiss path, keeping the panel independent of the canvas selection.

Pass `gepaPhases` from `state.gepaPhases` through to where `GEPAPhasePanel` is
rendered.

Add `GEPAPhasePanel` at the bottom of the JSX return, outside the
canvas/sidebar layout:

```tsx
{openGEPAPrimitive && state.gepaPhases.get(openGEPAPrimitive) && (
  <GEPAPhasePanel
    phaseData={state.gepaPhases.get(openGEPAPrimitive)!}
    onClose={() => setOpenGEPAPrimitive(null)}
  />
)}
```

The panel receives live-updating `phaseData` as events continue to arrive
(React re-renders naturally when the map entry changes).

### 5. Edge Cases

| Scenario | Behavior |
|----------|----------|
| `REPROLAB_GEPA_OPTIMIZATION=off` | No `gepa_*` events → `gepaPhases` stays empty → no GEPA nodes in canvas → panel never mounts |
| Phase still running | `status: 'running'` → header shows amber pulse; candidate rows appear as events arrive |
| BUG-048 timeout (0 candidates) | `total_metric_calls === 0`, candidates empty → summary footer shows timeout warning |
| User opens panel for one primitive, then clicks another | `setOpenGEPAPrimitive` replaces the open primitive; panel swaps content |
| `gepa_candidate_proposed` arrives before `gepa_phase_start` | Stub entry created in `gepaPhases` without `max_metric_calls`; safe |

---

## Track B — Pending Bug Fixes (commit only)

Five diffs already written to working tree, need a commit:

| Bug | File | Fix |
|-----|------|-----|
| BUG-NEW-046 | `backend/agents/paper_grounding.py` | `ast.literal_eval` to extract `name` from dict-repr dataset strings |
| BUG-NEW-047 | `backend/agents/environment_detective.py` | `compatibility_notes` says "LOCAL CPU dev machine" to prevent root from concluding CPU-only |
| BUG-NEW-048 | `backend/config.py` | `gepa_timeout_plan_s=180`, `gepa_timeout_baseline_s=90`, `gepa_timeout_improve_s=180` |
| BUG-NEW-050 | `backend/agents/rlm/primitives.py` | Fall back to last successful `experiment_runs.jsonl` entry when `results` has no metrics |
| Hydration | `frontend/src/components/lab/lab-shell.tsx` | All localStorage reads moved to `useEffect` (SSR hydration mismatch) |

---

## Track C — BUG-049 GPU Escalation

**File:** `backend/agents/rlm/primitives.py` (the `run_experiment` escalation
handler).

**Current behavior:** `TRANSIENT_500` and `CAPACITY_EXHAUSTED` both immediately
consume an escalation slot and upgrade the GPU tier.

**Fix:** Add a same-tier retry counter on the `TRANSIENT_500` path. Before
consuming an escalation slot, retry the same GPU tier up to 2 times with
exponential backoff (15s, 30s). Only escalate if retries are exhausted.
`CAPACITY_EXHAUSTED` continues to escalate immediately (the pod is genuinely
full at that tier).

Implementation note: the retry counter is local to the `run_experiment`
invocation and does not persist across primitive calls. The existing
`REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS` cap applies to escalations only (not
same-tier retries).

---

## Files Changed

| File | Change |
|------|--------|
| `frontend/src/hooks/use-rlm-run.ts` | Add `GEPACandidateRecord`, `GEPAPhaseData` types; extend fold functions; add `gepaPhases` to `RlmRunState` |
| `frontend/src/components/lab/rlm/gepa-phase-panel.tsx` | **New** — floating panel component |
| `frontend/src/components/lab/rlm/gepa-phase-panel.module.css` | **New** — panel styles |
| `frontend/src/components/lab/rlm/rlm-lab.tsx` | Add `openGEPAPrimitive` state; GEPA node click handler; render `GEPAPhasePanel` |
| `backend/agents/paper_grounding.py` | BUG-046 (already written) |
| `backend/agents/environment_detective.py` | BUG-047 (already written) |
| `backend/config.py` | BUG-048 (already written) |
| `backend/agents/rlm/primitives.py` | BUG-050 (already written) + BUG-049 (new) |
| `frontend/src/components/lab/lab-shell.tsx` | Hydration fix (already written) |

## Out of Scope

- `gepa-viz` subprocess management / proxy repair — the proxy is left as-is.
- The `gepa==0.0.27` vs `0.1.1` requirements.txt version mismatch — tracked
  separately; does not affect the panel.
- New tests for the panel component — the hook changes are pure functions
  amenable to unit tests; UI testing deferred until a GEPA-enabled run is
  available for E2E.
