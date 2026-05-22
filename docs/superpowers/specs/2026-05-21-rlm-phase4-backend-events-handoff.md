# RLM Phase 4 — Backend-Emission Handoff: Three Missing SSE Events

**Audience:** Backend engineer implementing the SSE emission for the three
events Phase 4's UI consumes from fixture/replay but the backend does not yet emit.

**Status:** Phase 4 deliverable. Implementing the emission is NOT Phase 4 scope —
Phase 4 touches no file under `backend/`. This doc is the contract a later backend
task implements.

---

## 1. Purpose & scope

Phase 4 (issue #61) built the RLM lab UI **fixture-first**: the frontend
components, the `useRlmRun` reducer, and the Playwright e2e all work today
against a hand-authored events fixture. The UI already renders all 8 RLM event
types, including the 3 described here.

The backend currently emits 5 of those 8 types:

| Event | Emitted today | Source |
|---|---|---|
| `repl_iteration` | yes | `sse_bridge.ReproLabRLMLogger.log` |
| `primitive_call` | yes | `binding.wrap_primitive` → `DashboardEmitter.primitive_call` |
| `sub_rlm_spawned` | yes | `sse_bridge.make_on_subcall_start` |
| `sub_rlm_complete` | yes | `sse_bridge.make_on_subcall_complete` |
| `run_complete` | yes | `run._finalize` → `sse_bridge.build_run_complete_event` |
| `candidate_proposed` | **no** | — |
| `candidate_outcome` | **no** | — |
| `rubric_score` | **no** | — |

Emitting the 3 missing events is a **pure transport addition** — it does not
change the existing 5 events, the reducer, the UI, or any frontend file. Once
the backend emits them, the fixture-fed UI goes fully live with no frontend
changes required.

### Alignment with `frontend_integration.md`

All 3 events must observe the same rules the existing 5 already follow
(see `frontend_integration.md`):

1. **Emit a typed event** for anything the UI must show live — never poll.
2. **Route every new event through `make_emit`** in `sse_bridge.py` — the single
   egress chokepoint; never widen the surface.
3. **Keep events additive and value-free** — no raw corpus, no full REPL locals.
4. **Persist to `dashboard_events.jsonl`** so a reconnecting client replays them.

The "dedicate vs. derive" criterion from `frontend_integration.md` (§ "Phase 4
notes"):

> Promote any [derivable event] to a dedicated backend event only if the UI
> needs push-granularity the derivation can't give.

All 3 events meet that bar:
- `candidate_proposed` — a `primitive_call` for `propose_improvements` is
  value-free by design: its `result_summary` is just `"list[7]"`. The candidate
  titles, categories, and reasoning cannot be reconstructed from it.
- `rubric_score` — similarly, `verify_against_rubric` returns
  `RubricVerification.model_dump()` — a dict — so `binding.py`'s `_result_summary`
  produces a value-free key list such as `"dict[areas, confidence, overall_score, ...]"`.
  Per-area scores cannot be derived.
- `candidate_outcome` — outcomes are only known after a candidate runs and is
  re-verified; by definition there is no single primitive call to derive from.

---

## 2. Event schemas (authoritative TypeScript source: `frontend/src/lib/events/rlm-events.ts`)

Copy these field names and types verbatim — the reducer's type guards depend on them.

### 2.1 `candidate_proposed`

```ts
interface CandidateProposedEvent {
  event: "candidate_proposed";
  timestamp: string;                  // ISO-8601 UTC
  iteration: number;                  // 1-based root-loop iteration
  round: number;                      // 1-based; one propose_improvements() call = one fan
  parent_id?: string;                 // node this branches from (see §5)
  candidate: {
    id: string;                       // stable within the run, e.g. "c5"
    title: string;                    // short name, model-generated — NOT corpus
    category: string;                 // free-form tag, e.g. "optimizer" — NOT corpus
    description: string;              // 1–2 sentences, model-generated
    reasoning: string;                // why the root proposed it
  };
}
```

### 2.2 `candidate_outcome`

```ts
interface CandidateOutcomeEvent {
  event: "candidate_outcome";
  timestamp: string;                  // ISO-8601 UTC
  iteration: number;                  // iteration when outcome was determined
  candidate_id: string;               // matches candidate_proposed.candidate.id
  outcome:
    | "running"
    | "promoted"
    | "marginal"
    | "failed"
    | "skipped"
    | "declined";
  rubric_delta: number | null;        // overall_score change this candidate produced
}
```

### 2.3 `rubric_score`

```ts
interface RubricScoreEvent {
  event: "rubric_score";
  timestamp: string;                  // ISO-8601 UTC
  iteration: number;                  // 1-based root-loop iteration
  score: number;                      // overall, 0–1
  target: number;                     // rubric target, 0–1
  areas: Array<{
    area: string;
    score: number;
    weight: number;
    status: "pass" | "partial" | "fail";
  }>;
}
```

---

## 3. `candidate_proposed` — emission point and field mapping

**Where to emit:** `backend/agents/rlm/binding.py`, inside `wrap_primitive`,
immediately after a successful `propose_improvements` call returns. Emit one
`candidate_proposed` event per item in the returned list.

**Current code path:** `wrap_primitive` wraps `propose_improvements` and emits a
single `primitive_call(name, "ok", result_summary="list[N]")` on success. The 3
new events fire *after* that `primitive_call(ok)` — one per returned hypothesis.

### Field mapping: `ImprovementHypothesis` → `candidate_proposed.candidate`

`primitives.propose_improvements` returns `list[ImprovementHypothesis.model_dump()]`
(see `backend/agents/schemas.py:250`). The `ImprovementHypothesis` schema is:

```python
class ImprovementHypothesis(BaseModel):
    path_id: str
    hypothesis: str
    rationale: str
    expected_outcome: str
    compute_estimate: str
    risk: RiskLevelField
    expected_value_score: float
    category: str
```

The frontend `candidate` shape requires `title`, `description`, and `reasoning` —
fields that do not exist verbatim on `ImprovementHypothesis`. The backend
implementation must derive or add them:

| `candidate` field | Source | Notes |
|---|---|---|
| `id` | `path_id` | Direct mapping. |
| `title` | **no direct field** | Must be derived: first sentence of `hypothesis`, a short slug, or a new `title: str = ""` field added to `ImprovementHypothesis`. A short slug from `hypothesis` is the simplest option; a new field lets the LLM supply it explicitly. |
| `category` | `category` | Direct mapping. |
| `description` | `hypothesis` or `expected_outcome` | `hypothesis` is the 1–2 sentence description; `expected_outcome` may be appended. |
| `reasoning` | `rationale` | Direct mapping. |

**Recommendation:** add a `title: str = ""` field to `ImprovementHypothesis` and
update the `propose_improvements` LLM prompt to supply it. This avoids fragile
first-sentence splitting, keeps the field model-generated, and matches the UI's
intent (a short name for the candidate node label).

### Iteration number

`wrap_primitive` does not currently know the root-loop iteration index —
`DashboardEmitter.primitive_call`'s `iteration` parameter is `None` today (see
`dashboard_emitter.py:202-207`). Supplying `iteration` on the new events requires
the same mechanism the spec defers for `primitive_call` itself: `RunContext` must
carry the current iteration index (e.g. `ctx.current_iteration: int = 0`), which
`ReproLabRLMLogger.log` increments before calling the wrapper's event (or which
`run.py` injects via a callback). This is a pre-requisite for all 3 new events.

### Round number

`round` is the per-run count of `propose_improvements` calls. `RunContext` should
carry a `propose_round: int = 0` counter that `wrap_primitive` increments on each
`propose_improvements` call.

### Corpus safety

`ImprovementHypothesis` fields are improvement *hypotheses* the model wrote about
the paper — not paper text. `hypothesis`, `rationale`, `expected_outcome`,
`category`, and a derived `title` are all model-generated metadata, safe to stream.
They must still route through `make_emit` (never call `dashboard._emit` directly)
to stay within the single egress chokepoint.

---

## 4. `rubric_score` — emission point and field mapping

**Where to emit:** `backend/agents/rlm/binding.py`, inside `wrap_primitive`,
immediately after a successful `verify_against_rubric` call returns its
`RubricVerification` dict.

### Field mapping: `RubricVerification` → `rubric_score`

`primitives.verify_against_rubric` returns `RubricVerification.model_dump()`
(see `backend/agents/schemas.py:365`). Key fields:

| `rubric_score` field | Source | Notes |
|---|---|---|
| `score` | `overall_score` | Computed deterministically by `RubricVerification.from_areas` — never from the model's self-report. |
| `target` | `target_score` | Set from the run's rubric spec. |
| `areas[].area` | `areas[].area` | Direct. |
| `areas[].score` | `areas[].score` | Direct. |
| `areas[].weight` | `areas[].weight` | Direct. |
| `areas[].status` | **no field on `RubricAreaScore`** | Must be derived. Suggested thresholds: `score >= 0.7` → `"pass"`, `score >= 0.4` → `"partial"`, otherwise `"fail"`. These are a UI affordance, not a rubric gate; the exact thresholds are a design choice for the implementing engineer. |

`RubricAreaScore` does not carry a `status` field — the backend must derive it.
The thresholds above match the natural rubric scale (`target_score` is typically
0.7) but should be made configurable or at minimum documented as constants.

### Fail-soft behavior

`verify_against_rubric` is fail-soft: on LLM/parse error it returns
`{"success": False, "error": ...}`. The wrapper must check `result.get("success"
) is not False` before emitting — do not emit `rubric_score` for a failed
verification.

---

## 5. `candidate_outcome` — the correlation challenge

`candidate_outcome` is the most complex of the three to emit because outcomes are
only known **after** a candidate has been run and re-verified. A single primitive
wrapper cannot emit it; it requires run-level state correlation.

### Why it is hard

The lifecycle of one candidate looks like:

```
propose_improvements()          → emits candidate_proposed  (iteration N, round K)
<root runs the candidate>
run_experiment(…)               → some primitive
verify_against_rubric(…)        → emits rubric_score at iteration M
<root or run.py decides outcome>
                                → must emit candidate_outcome  (candidate_id=C, outcome=promoted|failed|…)
```

The verify wrapper does not know which candidate the root is currently pursuing —
that is run-level state. The outcome (`promoted`, `marginal`, `failed`, etc.) is a
judgment call the root makes *after* seeing the verification score.

### Option A — run-level correlation in `run.py`

`run.py` already maintains the `ReproLabRLMLogger` across the full run. The
correlation logic could live there:

1. When `candidate_proposed` is emitted, record `{candidate_id → iteration}` in
   a dict on `RunContext` (or the logger).
2. When `verify_against_rubric` returns, compare the new `overall_score` to the
   prior best. The root's choice (promote/fail) can be inferred from the score
   delta against the rubric target.
3. Emit `candidate_outcome` immediately after `rubric_score`.

**Limitation:** this is an approximation — the root model may apply logic the
backend cannot observe (e.g. promoting a marginal candidate because it is
orthogonal to a prior improvement). The backend's inferred outcome may not match
the root's actual intent.

### Option B — explicit signal in the root prompt

Add a `record_candidate_outcome(candidate_id, outcome)` primitive to
`PRIMITIVE_REGISTRY` (a no-op computation, purely for event emission). The root
REPL code calls it after each evaluation:

```python
outcome = "promoted" if score > rubric_target else "failed"
record_candidate_outcome(candidate_id=cid, outcome=outcome)
```

**Advantage:** the outcome reflects the root's actual decision. The primitive
wrapper emits `candidate_outcome` directly from `wrap_primitive` — no run-level
correlation needed.

**Limitation:** requires a prompt change and a new primitive, and the root model
must call it reliably.

### Option C — hybrid

Use Option A (score-delta inference) as the default, and add Option B's primitive
as an override. If the root calls `record_candidate_outcome`, that outcome wins;
otherwise the inferred outcome fires after `verify_against_rubric`.

### Recommendation

Option B is the cleanest: the outcome is authoritative (from the root), emission
is local to a wrapper, and there is no inference logic to maintain. The primitive
is trivially simple — it just emits the event. Option A is acceptable if the
prompt cannot be changed; flag the approximation clearly in code comments.

---

## 6. `parent_id` — cost and recommendation

`CandidateProposedEvent.parent_id` is optional (`parent_id?: string`). When it is
absent, the `useRlmRun` reducer applies a heuristic (the §5.3 frontier rule from
the Phase 4 design spec):

> Parent of round N's fan = the most-recent node with `outcome === "promoted"`;
> if a round produced no promotion, fall back to the previous fan's parent; if no
> promotion has ever occurred, the parent is the `baseline` node.

This heuristic can mis-parent when the root triages and builds on a marginal-but-
cheap candidate instead of the highest-scoring one, or when `candidate_proposed`
arrives before the prior round's `candidate_outcome`. When wrong, the tree is
approximately correct but node parentage may be visually misleading.

**Cost:** `parent_id` faces the same correlation challenge as `candidate_outcome`
— the `wrap_primitive` wrapper does not know which node the root is currently
building on. Emitting it accurately requires the same run-level state (Option A or
B above). For Option B (`record_candidate_outcome` primitive), the root can supply
`parent_id` in the same call.

**Strong recommendation:** emit `parent_id`. The heuristic fallback is available
but should not be the primary path. If Option B is implemented, `parent_id` comes
for free from the root's REPL code. If Option A, derive it from the correlation
dict (the `candidate_id` that last produced a `promoted` outcome).

---

## 7. Summary checklist for the backend implementer

```
[ ] Add `title: str = ""` to ImprovementHypothesis in schemas.py and update
    the propose_improvements LLM prompt to supply it.

[ ] Add `ctx.current_iteration: int` to RunContext (incremented by
    ReproLabRLMLogger.log before each wrapped call fires).

[ ] Add `ctx.propose_round: int` to RunContext (incremented per
    propose_improvements call in wrap_primitive).

[ ] Plumb make_emit (or a locked emit callable) onto RunContext — wrap_primitive
    today only has ctx.dashboard.primitive_call, which bypasses the threading.Lock
    that make_emit provides. Options: (a) add `emit: Callable[[dict], None]` to
    RunContext (set from make_emit in run.py), or (b) add typed methods to
    DashboardEmitter (matching the existing primitive_call pattern — note this path
    lacks the lock from make_emit).

[ ] In wrap_primitive (binding.py), after a successful propose_improvements:
    - For each ImprovementHypothesis in the result list:
      - emit candidate_proposed via make_emit (see sse_bridge.make_emit)
      - use path_id → id, title → title, category → category,
        hypothesis → description, rationale → reasoning
      - include parent_id if available (see §6)

[ ] In wrap_primitive (binding.py), after a successful verify_against_rubric:
    - Derive areas[].status from score thresholds (pass/partial/fail)
    - Emit rubric_score via make_emit
    - Do NOT emit on a failed verification (success=False return)

[ ] Implement candidate_outcome emission via Option B (new primitive) or
    Option A (run-level correlation in run.py); see §5.

[ ] Add event builders to sse_bridge.py (following the existing pattern of
    build_run_complete_event, build_sub_rlm_spawned_event, etc.) for all 3
    new events.

[ ] Verify all 3 new events route through make_emit, never dashboard._emit
    directly.

[ ] Update frontend_integration.md's SSE event table with the 3 new rows.
```

---

## 8. File reference

| File | Relevance |
|---|---|
| `backend/agents/rlm/binding.py` | `wrap_primitive` — emit `candidate_proposed` and `rubric_score` here |
| `backend/agents/rlm/sse_bridge.py` | `make_emit`, `sanitize_iteration`, event builder pattern — all new builders go here |
| `backend/agents/rlm/run.py` | `run_pipeline_rlm`, `ReproLabRLMLogger` — `candidate_outcome` Option A correlation lives here |
| `backend/agents/rlm/context.py` | `RunContext` — add `current_iteration` and `propose_round` |
| `backend/agents/dashboard_emitter.py` | `primitive_call` — note its `iteration=None` gap; same fix needed for the new events |
| `backend/agents/schemas.py` | `ImprovementHypothesis` (add `title`), `RubricVerification`, `RubricAreaScore` (derive `status`) |
| `backend/agents/rlm/primitives.py` | `propose_improvements`, `verify_against_rubric` — source of the returned objects |
| `frontend/src/lib/events/rlm-events.ts` | Authoritative TypeScript schemas — field names and types are the wire contract |
| `frontend_integration.md` | Egress rules, the derive-vs-dedicate criterion, the SSE event table to update |
