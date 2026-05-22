# RLM Pivot Phase 4 — Frontend Redesign · Design Spec

> **Status:** approved design (brainstormed 2026-05-21, visual companion v1→v3).
> **Scope:** Phase 4 (#61) — the RLM lab UI. Frontend-only, fixture-first.
> **Canonical inputs:** `docs/design/rlm-pivot-brief.md` §9/§11; GitHub issue #61;
> `docs/design/phase4-6-execution-prompt.md`. Where this spec and #61's *layout*
> conflict, this spec wins (the graph-centric layout is a user-directed change,
> 2026-05-21); #61's *content* and event intent are otherwise honored.
> **Next step:** `superpowers:writing-plans` → the implementation plan.

---

## 1. Goal & done condition

Replace the fixed 14-stage pipeline UI with a lab that renders a live **RLM run**
as a dynamic, branching **exploration tree** — nodes appearing as the root model
auto-chooses where to go — wrapped in a dense "lab notebook" shell (rubric score,
REPL state, live report).

**Done when:**

1. The lab renders a complete RLM run through all of issue #61's content —
   header, rubric score+climb, REPL state, the exploration tree, root reasoning,
   the live report, primitive-call history — driven by a recorded events fixture.
2. The exploration tree populates dynamically from `candidate_proposed` /
   `candidate_outcome`; no fixed stage strip appears anywhere for an `rlm` run.
3. The old 14-stage UI (`offline`/`sdk` runs) is untouched and still works.
4. `npm run lint`, `npx tsc --noEmit`, `npm test` (vitest), and the Playwright
   e2e all pass.

The UI obeys the **honesty rule** (brief §1/§9): it shows only what the backend
actually produced. No fabricated slots, no "real-time" overclaim, no marketing
aesthetic.

---

## 2. Scope

**In scope (Phase 4, this spec):**

- The `<RlmLab>` component tree and the v3 layout (§7).
- `useRlmRun` — a pure reducer folding the RLM event stream into `RlmRunState` (§5).
- TypeScript types for the 8 RLM event shapes (§4).
- A hand-authored **events fixture** + a **replay harness** (§10).
- The `WorkflowView` `runMode === "rlm"` branch (§3).
- vitest unit tests + a Playwright e2e (§11).
- A **backend-emission handoff doc** for the 3 not-yet-emitted events (§13).

**Explicitly OUT of scope:**

- Backend code. The 3 missing events (`candidate_proposed`, `candidate_outcome`,
  `rubric_score`) are *not* emitted by Phase 4 — they are specified as a fixture
  contract + handoff doc for a later backend task. Phase 4 touches no file under
  `backend/`.
- Phase 5 (end-to-end runs) and Phase 6 (cleanup, dead-code deletion).
- Deleting the old pipeline UI (`progress-strip.tsx`, `gate-chips.tsx`, the
  14-stage `LabCanvas`, the old SSE event types) — that is Phase 6's job. Phase 4
  leaves them in place and runnable.

---

## 3. Architecture & integration

**Data flow** (both paths converge on the same reducer):

```
LIVE path   real RLM run → dashboard_events.jsonl → live_runs.py
            → SSE "dashboard_event" frames → /api/demo/events (proxy)
            → useRun() → dashboardEvents[]  ─┐
FIXTURE path  rlm-run.fixture.ts → replay harness → event[] ─┤
                                                              ▼
                                            useRlmRun(events) → RlmRunState
                                                              ▼
                                  <RlmLab>  (rendered by WorkflowView when
                                            run.runMode === "rlm")
```

- **`useRun()` is reused unchanged** as the SSE transport. It already parses
  `dashboard_event` frames generically into a `dashboardEvents[]` array — the RLM
  event types ride that channel with **zero transport/proxy changes**.
- **`useRlmRun(events)` is the one new hook** — a `useMemo` that folds the event
  array through a pure `fold(state, event)` function into `RlmRunState`. Pure →
  replayable, unit-testable with no browser.
- **Coexistence:** `WorkflowView` (in `lab-shell.tsx`) gains one branch —
  `run.runMode === "rlm"` → `<RlmLab>`; otherwise → the existing 14-stage
  `LabCanvas`, untouched. One route (`/lab`), one shell, one upload flow.
- **Rejected alternative:** a separate `/lab/rlm` route with its own shell + SSE
  hook. Rejected — duplicates header/upload/transport wiring, splits the lab into
  two URLs; the brief mandates the old and new paths coexist on one lab.

**Event-retention note:** `useRun` currently caps `dashboardEvents[]` at 200. A
long RLM run can exceed that and truncate the reducer's history. Live-wiring
(later) must raise this cap for `rlm` runs; the fixture path is uncapped. See §14.

---

## 4. The event model

The lab consumes **8 event types**. Five are already emitted by the backend; three
are not — those three are the **fixture contract** (and the §13 handoff). All event
`data` arrives inside an SSE `dashboard_event` frame.

`variable_update` and `root_reasoning` from issue #61's table are **not separate
events** — their data already lives inside `repl_iteration` (`code_blocks[].vars`
and `response`). The reducer derives them; no new event needed.

### 4.1 Existing events (emitted today — schemas verified in backend source)

```ts
// backend/agents/rlm/sse_bridge.py :: _repl_iteration_event
interface ReplIterationEvent {
  event: "repl_iteration"; timestamp: string;
  iteration: number;                 // 1-based
  response: string;                  // root model's reasoning text, ≤4000 chars
  code_blocks: Array<{
    code: string;
    stdout_meta: { length: number; prefix: string; has_traceback: boolean };
    stderr_meta: { length: number; prefix: string; has_traceback: boolean };
    vars: Record<string, { type: string; size: number }>;   // the REPL var manifest
    sub_calls: number;
  }>;
  sub_calls: number;
  timing: number | null;
}

// backend/agents/dashboard_emitter.py :: primitive_call
interface PrimitiveCallEvent {
  event: "primitive_call"; timestamp: string;
  primitive: string;                 // "understand_section" | … (the 9 primitives)
  status: "start" | "ok" | "error";
  args_summary: Record<string, unknown>;   // value-free (type/length only)
  result_summary: string | null;           // value-free, e.g. "list[3]"
  iteration: number | null;
  rubric_delta: number | null;
}

interface SubRlmSpawnedEvent {
  event: "sub_rlm_spawned"; timestamp: string;
  depth: number; model: string; prompt_preview: string;   // preview ≤200 chars
}
interface SubRlmCompleteEvent {
  event: "sub_rlm_complete"; timestamp: string;
  depth: number; model: string; duration_ms: number; error: string | null;
}
interface RunCompleteEvent {
  event: "run_complete"; timestamp: string;
  status: "completed" | "partial" | "failed";
  iterations: number;
  rubric_score: number | null; cost_usd: number | null;
  final_report_path: string | null;
}
```

### 4.2 New events — the fixture contract (NOT emitted yet)

These are designed here so the fixture's field names *are* the contract a later
backend task implements (§13).

```ts
interface CandidateProposedEvent {
  event: "candidate_proposed"; timestamp: string;
  iteration: number;
  round: number;            // 1-based; one propose_improvements() call = one fan
  parent_id?: string;       // node this branches from. STRONGLY RECOMMENDED from
                            // the backend; if absent the reducer infers it (§5.3).
  candidate: {
    id: string;             // stable, e.g. "c5"
    title: string;          // short, model-generated (improvement name) — not corpus
    category: string;       // free-form tag, e.g. "optimizer" — not corpus
    description: string;    // 1–2 sentences, model-generated
    reasoning: string;      // why the root proposed it
  };
}

interface CandidateOutcomeEvent {
  event: "candidate_outcome"; timestamp: string;
  iteration: number;
  candidate_id: string;
  outcome: "running" | "promoted" | "marginal" | "failed" | "skipped" | "declined";
  rubric_delta: number | null;   // score change this candidate produced, if measured
}

interface RubricScoreEvent {
  event: "rubric_score"; timestamp: string;
  iteration: number;
  score: number;            // overall, 0–1
  target: number;           // rubric target, 0–1
  areas: Array<{ area: string; score: number; weight: number;
                 status: "pass" | "partial" | "fail" }>;
}
```

`DashboardEvent` (the RLM union) lives in a **new** file
`frontend/src/lib/events/rlm-events.ts` — the existing `contract.ts` is left
untouched (Phase 6 retires it).

**Corpus safety.** Every consumed field is metadata or short model-generated text.
The paper corpus never appears in any event (the backend's `sanitize_iteration` /
`redact_corpus` guarantee it). Candidate titles/categories/descriptions are
improvement *hypotheses* the model wrote — not paper text — so they are safe to
render. The UI never reconstructs or displays `context`.

---

## 5. Data model — `RlmRunState` & the `useRlmRun` reducer

### 5.1 `RlmRunState`

```ts
interface RlmRunState {
  status: "queued" | "running" | "completed" | "partial" | "failed";
  iterationCount: number;

  rubric: {
    current: number | null;
    baseline: number | null;                       // first rubric_score
    target: number | null;
    series: Array<{ iteration: number; score: number }>;   // sparkline / timeline
    areas: RubricArea[];                            // latest breakdown
  };

  variables: Record<string, {                       // REPL-state rail
    type: string; size: number; firstSeenIteration: number;
  }>;

  iterations: IterationView[];                      // every repl_iteration, for detail
  currentIteration: IterationView | null;           // the live frontier

  tree: TreeNode[];                                 // flat; each node has parentId
  primitiveCalls: PrimitiveCallView[];              // for the history bar
  subRlms: SubRlmView[];

  report: {
    finalReportPath: string | null;
    costUsd: number | null;
    counts: { iterations: number; primitiveCalls: number;
              proposed: number; promoted: number };
  } | null;                                         // null until run_complete
}
```

`TreeNode`, `IterationView`, `RubricArea`, `PrimitiveCallView`, `SubRlmView` are
defined alongside `RlmRunState` in `use-rlm-run.ts`.

### 5.2 The reducer

`useRlmRun(events)` = `useMemo(() => events.reduce(fold, INITIAL), [events])`.
`fold(state, event): RlmRunState` is **pure** and exported for unit tests.

| Event | Fold effect |
|---|---|
| `repl_iteration` | `iterationCount = iteration`; push `IterationView`; set `currentIteration`; merge `code_blocks[].vars` into `variables`; iteration's `response` is its root-reasoning. Extend the trunk (§6). |
| `primitive_call` | append to `primitiveCalls`; drives phase derivation + node iteration-ranges (§6). |
| `candidate_proposed` | add a `candidate` `TreeNode`; parent from `parent_id` or inferred (§5.3). |
| `candidate_outcome` | set the candidate node's `outcome` + `rubricDelta`. |
| `rubric_score` | push `{iteration, score}` to `rubric.series`; set `rubric.current`/`areas`; `baseline` if first; attach `score` to the frontier node. |
| `sub_rlm_spawned` / `_complete` | add/update a `SubRlmView` + a `subrlm` `TreeNode`. |
| `run_complete` | set `status`, `report`, final `rubric.current`. |

### 5.3 The frontier rule (parent inference) — **pin this**

When `candidate_proposed` has no `parent_id`, the reducer assigns one:

> **Parent of round N's fan** = the most-recent node with `outcome === "promoted"`;
> if a round produced no promotion, fall back to the previous fan's parent; if no
> promotion has ever occurred, the parent is the `baseline` node. Ties (same
> iteration) break by `propose_improvements` call order.

This is a **heuristic** — it can mis-parent when the root triages and builds on a
*marginal-but-cheap* candidate instead of the highest-scoring one (brief §8.7), or
when `candidate_proposed` arrives before the prior round's `candidate_outcome`.
When wrong, the tree is *approximately* right and every node still carries its
true `iterationRange`, so the user can always see what attached to what. The fix
is exact `parent_id` from the backend — hence "strongly recommended" in §13.

The fixture (§10) **must** include a no-promotion round and an out-of-order
`proposed`/`outcome` pair so tests pin this behavior.

---

## 6. The node & tree model

A node is a **real move the RLM made**, backed by real events — nothing decorative.

**Node kinds:** `paper` (root) · `work` (a trunk step toward the baseline) ·
`baseline` (the first scored milestone — created at the first `rubric_score` event) · `candidate` (an improvement
hypothesis) · `subrlm` (a recursion marker) · `declined-group` (a synthesized
node collapsing a round's declined candidates).

**True structure vs. phase labels.** The tree's *real* structure is:
`paper → a chain of work-nodes → baseline → candidate fans (deepening)`. The
labels **"comprehension / environment / baseline-build"** are a *display annotation* —
computed by grouping consecutive iterations by which primitives ran
(`understand_section`/`extract_hyperparameters` → comprehension;
`detect_environment`/`build_environment` → environment;
`implement_baseline`/`run_experiment` → baseline-build). **Phases are not load-bearing**
— if the root revisits `understand_section` mid-improvement, the tree still
renders correctly; only the convenience label may look unusual. Consumers of this
spec must not treat phase labels as structural truth.

**Trunk-node granularity — flag for user review.** v3 (approved) consolidates the
trunk: one `comprehension` node spans iterations 1–3, etc. — matching the
screenshot the user liked. Consequence: the live "nodes pop up" dynamism is
strongest at the **candidate fans and outcome transitions**; trunk nodes
appear/extend progressively but are few. If the user wants **one node per
`primitive_call`** in the trunk instead (finer-grained pop-in), that is a small
design change — raised here so it can be settled at spec review, not at
implementation.

**Tree assembly:** trunk nodes are created/extended as `primitive_call`s land;
the `baseline` node is the first `verify_against_rubric`; each
`candidate_proposed` adds a leaf under its round's parent; a promoted candidate
becomes the parent of the next round's fan (the tree deepens along the chosen
path); declined candidates in a round collapse into one `declined-group` node.

---

## 7. The UI — v3 layout & component decomposition

### 7.1 Layout (four bands)

```
┌ RlmHeader ───────────────────────────────────────────────────────┐
│ paper title · authors · venue · arXiv | prj_id | status·iter | $  │
├ RubricStrip ──────────────────────────────────────────────────────┤
│ 0.53  [baseline▲──fill──▲target]  ▁▃▅▇ climb sparkline  +Δ        │
├ RlmWorkspace (3 zones) ───────────────────────────────────────────┤
│ ReplStateRail │      ExplorationCanvas          │  ReportRail     │
│  (collapsible │   — dominant zone —             │  (collapsible   │
│   ~190px)     │   the live exploration tree     │   ~240px)       │
│  variables    │   pan/zoom · nodes pop in live  │  verdict        │
│  primitives   │   click node → NodeDetailPopup  │  stat grid      │
│               │                                 │  rubric break   │
├ PrimitiveHistoryBar ──────────────────────────────────────────────┤
│ ▸ primitive call history — N calls (collapsed by default)         │
└───────────────────────────────────────────────────────────────────┘
```

Single vertical scroll; the workspace row is the tall band. Both rails collapse so
the canvas can take full width.

### 7.2 Component tree

All new components live in `frontend/src/components/lab/rlm/`.

| Component | Consumes | Renders |
|---|---|---|
| `RlmLab` | `run`, `events` | calls `useRlmRun`; composes the four bands; owns rail-collapse state (`selectedNodeId` is owned by `ExplorationCanvas` — node selection is canvas-internal) |
| `RlmHeader` | `run`, `state.status`, `state.iterationCount` | paper metadata, project id, status pill (reuse `status.tsx`), iter/cost |
| `RubricStrip` | `state.rubric` | score number, baseline→target bar, climb sparkline, Δ-vs-baseline / Δ-vs-target |
| `ReplStateRail` | `state.variables`, primitive list | variable rows (name · type · size, dimmed if unset), "N/M set", primitives list; collapsible |
| `ExplorationCanvas` | `state.tree`, `selectedNodeId` | `layoutTree` → positioned nodes; pan/zoom via `PanWrap`; renders `TreeNode`s + `TreeEdges` + `NodeDetailPopup` |
| `TreeNode` | one `TreeNode` | a node card — kind-specific (paper/work/baseline/candidate/subrlm/declined-group); outcome color; live pulse |
| `TreeEdges` | positioned nodes | one SVG layer; edges colored by child outcome, dashed for failed/declined |
| `NodeDetailPopup` | selected node + its `IterationView`s | floating card: iteration code, stdout meta, reasoning, rubric delta, category |
| `ReportRail` | `state.report`, `state.rubric` | verdict pill, 6-tile stat grid, live rubric breakdown (areas flip pass/partial/fail); collapsible |
| `PrimitiveHistoryBar` | `state.primitiveCalls` | collapsible, reverse-chronological list |

**Reuse:** `PanWrap` + `usePan` (pan/zoom), `status.tsx` (status pill),
`styles/tokens.css` (all colors/spacing/type), the `resizable-split` pattern if
useful. New components follow the lab's **CSS-module** convention
(`*.module.css`), not the plain-`.css` `:global` style.

---

## 8. The exploration tree — detailed design

The canvas is the hardest component; the layout is its own task (§16.6).

- **`layoutTree(nodes): { positioned, edges }`** — a **pure** function (no React),
  unit-tested with synthetic trees. Horizontal layout: `x` by depth, `y` by
  sibling stacking (a simple Reingold-Tilford-style pass — assign each leaf a slot,
  each parent the midpoint of its children). Returns absolute positions + edge
  endpoints. Stable ordering: chronological while running; the run may re-sort
  siblings by outcome after `run_complete`.
- **Edges** — one SVG layer behind the nodes; an edge is colored by its *child's*
  outcome; `failed`/`declined` edges are dashed.
- **Soft cap** — at most 8 sibling candidates rendered per fan; the rest collapse
  into a `+N more` node that expands on click.
- **Declined-collapse** — candidates with `outcome === "declined"` in a round
  render as one `declined-group` node ("N declined"), expandable.
- **Sub-RLM** — a circular `R` node; clicking it drills into the sub-run (Phase 4:
  shows the sub-RLM's prompt preview + duration + error in the popup; a nested
  sub-tree view is a noted follow-up, not required for the done condition).
- **Selection** — clicking any node sets `selectedNodeId`; `NodeDetailPopup` opens
  anchored near it. The live frontier node is auto-selected until the user clicks.
- **Live behavior** — new nodes fade+scale in over 200ms; a `running` node pulses
  (low-contrast); all motion is disabled under `prefers-reduced-motion`.

---

## 9. Visual design

- All color/space/type from `frontend/src/styles/tokens.css` — forest accent
  (`#1d7a3c`), hermes purple (`#7c5cff`), JetBrains Mono for technical values,
  tight radii. Flat surfaces, hairline (0.5–1px) borders, sentence case, generous
  whitespace.
- **Outcome palette:** promoted = forest green · marginal = amber (`--warning`) ·
  failed = coral (`--error`) · running = purple (`#7c5cff`, pulsing) ·
  skipped/declined = grey (`--text-tertiary`, dashed border).
- **Animations:** status dot 2s pulse, new node fade-in 200ms, running node
  low-contrast pulse — all gated on `prefers-reduced-motion: reduce`.
- **No** orbital clouds, glow, dark gradients, or "self-improving superintelligence"
  language (brief §9, #61 visual section).
- Rails collapse to icon-width; the layout reflows to a single column under a
  narrow viewport (the canvas stays pannable).
- **Accessibility:** outcome is shown by a text label + icon, never color alone;
  tree nodes are keyboard-focusable with descriptive ARIA labels; the
  node-detail popup is dismissable and restores focus to its node.

---

## 10. The fixture & replay harness

**Fixture** — `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`,
exporting a typed `DashboardEvent[]`. Hand-authored (Phase 4 runs no backend),
conforming exactly to the §4 schemas. It can later be regenerated from a real run
once the backend emits all 8 types.

The fixture **must** exercise, in one realistic ~14-iteration run:

- the full trunk (comprehension → environment, incl. a 2-attempt build → baseline);
- ≥ 2 `propose_improvements` rounds, ≥ 10 nodes total, **mixed outcomes**
  (promoted, marginal, failed, running, skipped, declined);
- **a round that promotes nothing** (pins the §5.3 fallback);
- **an out-of-order `candidate_proposed` / `candidate_outcome` pair**;
- a declined-collapse (≥ 2 declined in one round);
- a `sub_rlm_spawned` + `sub_rlm_complete` pair;
- a `rubric_score` with ≥ 1 `fail` area (a weak rubric node);
- a terminal `run_complete`.

**Replay harness** — `frontend/src/components/lab/rlm/replay.ts`: replays the
fixture into the `useRlmRun` input, either instantly (tests) or timed (manual /
Playwright). Exposed in dev via a `?rlmFixture=1` query param on `/lab` that makes
`WorkflowView` mount `<RlmLab>` against the replayed fixture instead of a live run.
The query param is dev/test-only and documented as such.

---

## 11. Testing strategy

- **vitest — the pure core (highest value):** `fold` reducer tested by replaying
  fixture slices and asserting `RlmRunState` (tree shape, frontier rule incl. the
  no-promotion + out-of-order cases, rubric series, variable manifest, report
  counts). `layoutTree` tested with synthetic trees (depth, sibling stacking,
  soft-cap).
- **vitest — components:** each renders against a fixture-derived `RlmRunState`;
  assert key elements (rubric number, node count + outcome classes, rail
  collapse, popup open on node click).
- **Playwright — `frontend/e2e/rlm-lab.spec.ts`:** load `/lab?rlmFixture=1`,
  assert the tree populates, candidate nodes appear with outcome colors, the
  rubric climbs, the report fills, a node click opens the popup.
- Every task ends green on `npm run lint`, `npx tsc --noEmit`, `npm test`.

---

## 12. Coexistence & transition

- **Untouched:** `useRun`, the `/api/demo/*` proxy routes, `live_runs.py`, the
  14-stage `LabCanvas`/`progress-strip`/`gate-chips`/`node-config`, `contract.ts`,
  all of `backend/`.
- **Modified (frontend only):** `lab-shell.tsx` — the one `WorkflowView`
  `runMode === "rlm"` branch. (`use-run.ts`'s 200-event cap: raise only when
  live-wiring; not required for the fixture-first done condition.)
- **Not deleted:** nothing. Removing the old pipeline UI and the old SSE event
  types is **Phase 6** (#63). Phase 4 keeps `main` runnable.

---

## 13. Backend-emission handoff spec

Delivered as a separate doc
(`docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md`) so a
later backend task can make the fixture-fed UI go fully live. Summary:

- **`candidate_proposed`** — emit one per item returned by `propose_improvements`,
  from its `wrap_primitive` wrapper in `backend/agents/rlm/binding.py`. The
  `ImprovementHypothesis` already carries title/category/description — all
  model-generated, corpus-safe. **Include `parent_id`** (the node id of the result
  being improved) — strongly recommended; without it the UI falls back to the
  §5.3 heuristic, which mis-parents under triage.
- **`rubric_score`** — emit from the `verify_against_rubric` wrapper; the
  `RubricVerification` result already carries the per-area breakdown.
- **`candidate_outcome`** — the hardest: outcomes are known only *after* a
  candidate is run and re-verified, so it cannot be emitted from a single
  primitive wrapper. It needs run-level correlation in `run.py` (track which
  candidate id the root is pursuing, emit when its `verify_against_rubric`
  resolves) — or an explicit signal in the root's REPL code. Flagged as the
  non-trivial part of the handoff.

This doc is a Phase-4 deliverable but implementing it is **not** Phase 4 scope.

---

## 14. Edge cases & honesty

- **Initial / empty:** before any event, `<RlmLab>` shows only the paper node and
  a "waiting for the first iteration" state — no fabricated nodes.
- **Failed / partial / timed-out runs:** `run_complete.status` drives an honest
  verdict; a `failed` run shows the tree as far as it got + the failure; no fake
  "complete".
- **The 0.35 metric cap (review finding I7):** when the backend caps a degraded
  run's rubric at 0.35, the UI shows 0.35 with a visible "degraded — no real
  metrics" note. It must not imply a real score.
- **Cost (review finding M4):** primitive cost-ledger rows are zero-usage today;
  `cost_usd` comes from `run_complete`. Show only what `run_complete` provides;
  label it honestly or omit it — never display a fabricated per-primitive cost.
- **Event-cap truncation:** the 200-event cap (§3) can drop early events on a long
  live run; the reducer renders what it has and never crashes on a missing parent
  (an orphan node attaches to `baseline` with a "parent unknown" marker).
- **Corpus invariant:** the UI never displays `context`/`paper_text`; it only
  renders the §4 metadata fields.
- **Narrow viewport:** rails collapse; the canvas stays pannable.

---

## 15. File-by-file change list

**New — `frontend/src/`**

- `lib/events/rlm-events.ts` — the 8 event interfaces + the `RlmDashboardEvent` union.
- `hooks/use-rlm-run.ts` — `RlmRunState`, the pure `fold`, the `useRlmRun` hook.
- `components/lab/rlm/layout-tree.ts` — the pure tree-layout function.
- `components/lab/rlm/rlm-lab.tsx` + `.module.css`
- `components/lab/rlm/rlm-header.tsx` + `.module.css`
- `components/lab/rlm/rubric-strip.tsx` + `.module.css`
- `components/lab/rlm/repl-state-rail.tsx` + `.module.css`
- `components/lab/rlm/exploration-canvas.tsx` + `.module.css`
- `components/lab/rlm/tree-node.tsx` + `.module.css`
- `components/lab/rlm/tree-edges.tsx`
- `components/lab/rlm/node-detail-popup.tsx` + `.module.css`
- `components/lab/rlm/report-rail.tsx` + `.module.css`
- `components/lab/rlm/primitive-history-bar.tsx` + `.module.css`
- `components/lab/rlm/replay.ts`
- `components/lab/rlm/__fixtures__/rlm-run.fixture.ts`
- colocated `*.test.ts(x)` for the reducer, `layoutTree`, and each component.
- `e2e/rlm-lab.spec.ts`

**Modified — frontend only**

- `components/lab/lab-shell.tsx` — the `WorkflowView` `runMode === "rlm"` branch.

**New — docs**

- `docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md` (§13).

---

## 16. Build sequence (feeds `superpowers:writing-plans`)

Each item is one bite-sized, independently-reviewable task; each ends green on
lint + types + tests. TDD throughout — failing test first.

1. **Event types** — `rlm-events.ts`: the 8 interfaces + union. (no behavior)
2. **The fixture** — `rlm-run.fixture.ts` covering every §10 scenario.
3. **`useRlmRun` reducer** — the pure `fold` + `RlmRunState`; TDD against the
   fixture, incl. the §5.3 frontier rule (no-promotion + out-of-order cases).
4. **`layoutTree`** — the pure tree-layout function; TDD with synthetic trees.
5. **Presentational components** — `RlmHeader`, `RubricStrip`, `ReplStateRail`,
   `ReportRail`, `PrimitiveHistoryBar`; fixture-driven, each tested.
6. **`ExplorationCanvas`** — `TreeNode` + `TreeEdges` + `NodeDetailPopup` +
   pan/zoom + soft-cap collapse + declined-collapse. (the hard task — kept whole)
7. **`<RlmLab>` shell** — composes the bands; the `WorkflowView`
   `runMode === "rlm"` branch; rail-collapse + selection state.
8. **Replay harness** — `replay.ts` + the `?rlmFixture=1` dev path.
9. **Playwright e2e** — `rlm-lab.spec.ts`.
10. **Backend-emission handoff doc** — §13 written up.

Items 1–5 are pure-frontend with no backend coupling; 6 is the largest single task; 7 wires it together;
8–10 verify and hand off.

---

## 17. Open questions / risks

- **R1 — trunk granularity** (§6): consolidated phase nodes vs. one node per
  `primitive_call`. v3 (approved) uses consolidated; confirm at spec review.
- **R2 — `candidate_outcome` correlation** (§13): the genuinely hard part of the
  backend handoff; Phase 4 only specifies it.
- **R3 — parent inference** (§5.3): the heuristic mis-parents under root triage;
  mitigated by `parent_id` from the backend and by always showing iteration
  ranges.
- **R4 — live-wiring later:** the 200-event cap and the timed-replay-vs-live
  difference are deferred to the live-wiring step; the fixture path is unaffected.
- **R5 — sub-RLM drill depth:** Phase 4 ships popup-level sub-RLM detail; a nested
  sub-tree view is a noted follow-up, not in the done condition.
