# Phase 3 — BES on RDR (2026-06-07)

**Status:** 🟡 PROPOSED. The actual BES work. **Extends the RDR controller behind `REPROLAB_BES_*` flags, default OFF** (decision D1). Flag-off ⇒ RDR behaves bit-for-bit as today.
**Goal:** Add the BES-unique deltas to RDR so the env × baseline matrix is covered by *competing candidates in parallel* and (v2) assembled *honestly*.

> **Codex review (2026-06-07) — applied.** (1) **v1 ships competing candidates only; evolve/splice is deferred to v2** — surviving-cell splice is *not free*: discarded candidates have no executed cells (cells exist only after `run_experiment`, which runs once), so splice needs per-candidate GPU execution or a cell-granular redesign; and the `aggregate_cell_metrics` signature/manifest requirement was wrong (§4). (2) **The pre-run gate is mode-agnostic** (`REPROLAB_RDR_PREFLIGHT_GATE`), moved to Phase 2 — not a BES delta. (3) Seam corrected to insert **before** `:920` (the `:914-935` range *contains* the experiment); `scan_code_dir` is **top-level glob, not recursive**; `done[cid].files` **cannot** attribute a file to a cluster (§2). (4) Parity needs a real **`bes_enabled` master gate** over every child flag (§5).

---

## 1. Grounding — RDR is one-attempt-per-cluster; the three deltas are additive

**Verified (Agent 4):** `run_rdr` owns `decompose → cluster loop → assemble → env → experiment → score → repair → report` (`controller.py:3` docstring). It dispatches **one** scoped attempt per cluster + serial repair — confirmed:
- `_run_cluster_batch` builds exactly one task per `(idx, cluster)` (`controller.py:567-586`).
- `_dispatch_one_cluster` stores one `Artifacts` per `cluster.id` (`:472-500`); the agent is a single SDK call with no internal sampling (`agent.py:511-520`).
- The repair loop **replaces** the weak cluster's artifact in place (`:1003-1058`), keyed by `cluster.id` (overwrites).

So there is **no** candidate pool, no select-best-of-N. RDR's parallelism is *across distinct clusters* (`:566` semaphore), orthogonal to BES's *N attempts at the same sub-goal*. The deltas below are genuinely new — but **v1 ships competing candidates only** (§3); evolve/splice is **v2** (§4); the pre-run gate is **mode-agnostic** (§2, owned by Phase 2). (RDR's *repair* is already parallel + feedback-driven via `_failed_leaves_for_cluster:186` → `prior_feedback` (`agent.py:256-263`), so the "back-pass" half is ~done.)

## 2. Pre-run gate wiring (mode-agnostic — owned by Phase 2; ship FIRST)

*Renamed from "Delta 3": the gate is mode-agnostic (flag `REPROLAB_RDR_PREFLIGHT_GATE`, not `bes_*`) and belongs to Phase 2. The RDR-specific wiring is here.*

**Seam (Codex-corrected):** insert the gate **after `:911` (env build) and before the `run_experiment` call at `:920`** — the `:914-935` range *contains* the experiment call, so insert above it, not "into" it. Today: assemble `commands.json` (`:843`) → detect/build env (`:853-911`) → `run_experiment` (`:920`).

**Verified:** RDR has **no** pre-run AST gate today — `scan_code_dir`'s only product caller is `pre_flight_validator.py:1344` (RLM side); the RDR experiment chain never calls it.

**Change (PROPOSED):** when the gate flag is on, call `scan_code_dir(code_dir)` before `:920`. **Note (Codex):** `scan_code_dir` is **top-level `glob('*.py')`, not recursive** — only the env-contract check recurses (`preflight_ast.py:830-835`); widen the scan if subdir code must be gated. On violation, regenerate **without** the experiment via the existing `_run_cluster_batch` repair path, bounded by a max-regens flag (default 1).
- **⚠ Cluster attribution (Codex):** a violation file **cannot** be mapped to its owning cluster via `done[cid].files` — the agent snapshots *every* text file in the shared `code/` into each artifact (`agent.py:280-321`,`:524-540`), so multiple clusters claim the same file. Either re-dispatch **all** Code-Dev clusters on a violation, or first add **per-dispatch file-delta provenance** (before/after diff per cluster).
- **Reuse:** `scan_code_dir`; `_run_cluster_batch`; the violation→repair format exists RLM-side (`primitives.py:1418`,`:1431`).
- **New:** one `AgentContext.preflight_violations` field (`models.py:91`, additive); one emit type `rdr_preflight_blocked`.

## 3. Delta 1 — competing candidates (N attempts per sub-goal → SELECT)

**Seam:** wrap the single `reproduce` call inside `_dispatch_one_cluster` (`controller.py:472-500`) — the only place a cluster artifact is produced; every caller funnels through it.

**The real cost is isolation, not the loop.** Today every cluster's agent writes into the **shared** `code/` dir (`agent.py:463`); N candidates would stomp each other. So competing candidates require a **per-candidate scratch dir**, scored in isolation, with only the winner merged into `code/`.

**Data model (new — `backend/agents/rdr/candidates.py`):**
```python
@dataclass
class Candidate:
    candidate_id: str          # f"{cluster.id}#{n}"
    cluster_id: str
    parent_id: str | None      # provenance for splice (Delta 2)
    scratch_dir: Path          # project_dir/"candidates"/candidate_id/"code"
    artifacts: Artifacts
    score: float | None = None
    failed_leaves: list[str] = field(default_factory=list)
```
Candidate pool = `dict[str, list[Candidate]]` keyed by `cluster.id` (parallels the existing `done` dict).

**SELECT — reuse the scorer, statically:**
- `_cluster_score` (`:134`) + `_failed_leaves_for_cluster` (`:186`) rank candidates. Tie-break: fewest `failed_leaves`. `bes_select_metric` chooses.
- **Crucial cost fact (verified):** `score_reproduction` takes a *dir* arg (`:952`), so per-candidate scoring is the **static leaf scorer (no GPU)** over each scratch dir. The expensive `run_experiment` (`:920`) still runs **once**, on the winner-merged `code/`. **BES is N× *token* cost, not N× *GPU* cost** — do not conflate.

**Cost lever:** N multiplies *agent dispatches* — the dominant token cost. `total_agent_dispatches` (`:988`) becomes `len(clusters) * n`. `bes_candidates_per_cluster` **must** default to 1 (= parity) and be capped.

## 4. Delta 2 — evolve/splice (DEFERRED to v2 — not free)

**⚠ Codex blockers — splice as first drafted does not work:**
1. **No cells to splice.** "Union the *surviving* cells across candidates" assumes each candidate's cells ran — but cell results are created **only inside `run_experiment`** (`primitives.py:3663-3686`), which RDR calls **once** (`controller.py:920-926`), on the winner-merged code. Discarded candidates have **no executed cells**. So splice needs **either (a) executing each candidate's cells (GPU × N — not free)** or **(b) a cell-granular candidate redesign** where each candidate *is* a cell tied to the code that produced it.
2. **Wrong API/contract.** `aggregate_cell_metrics` is `(matrix_result, cells, …)` with **no `run_id`** (`cell_matrix.py:445-453`) and **requires the full attempted-cell manifest** to retain missing/OOM/error cells as failures (`:456-464`,`:511-540`). A naive union of only *surviving* cells would silently drop attempted failures and **inflate** the score — an honesty violation.
3. **Cluster-file splice (2a) stays blocked** — `RubricLeaf` (`models.py:27-41`) and `Artifacts` (`models.py:73-88`) carry no leaf→file provenance, so per-file recombination has no signal; `_merge_cluster_files` (`:348-389`) only merges *different* clusters.

**v1 ships competing candidates (§3) only — no splice.** The genuinely-"evolutionary" recombination is **v2**, contingent on a design decision: pay GPU × N for per-candidate cell execution, or restructure candidates at cell granularity. **Honesty invariant for any v2 splice:** pass the **full** cell manifest (not just survivors) to `aggregate_cell_metrics`, so dropped/failed cells stay counted and splice can never raise the score by discarding attempted failures.

## 5. Flag surface + parity contract

Add to `backend/config.py` `Settings` (after `:217`, the dynamic-GPU block); read via `get_settings()` (`:292`), never module-level:

```python
bes_enabled: bool = Field(default=False)                       # MASTER — off ⇒ every child flag inert
bes_candidates_per_cluster: int = Field(default=1, ge=1, le=8) # 1 = today's path (parity)
bes_select_metric: str = Field(default="cluster_score")        # cluster_score | failed_leaves
bes_splice_enabled: bool = Field(default=False)                # v2 (deferred — §4)
# pre-run gate is mode-agnostic (Phase 2), NOT a bes_* flag:
# rdr_preflight_gate: bool = Field(default=False)              # REPROLAB_RDR_PREFLIGHT_GATE
```

**Parity (bit-for-bit):** make it *structural*, not behavioral-by-luck —
- **Master gate (Codex):** wrap every BES branch in one outer `if not settings.bes_enabled: <legacy path>`; child flags (`bes_candidates_per_cluster`, `bes_splice_enabled`) are read **only inside** the enabled branch — so `bes_enabled=False` forces parity regardless of any child flag's value.
- `bes_candidates_per_cluster == 1` (inside the enabled branch) → `if n == 1: art = await <existing :473 call>` (literal early-return; no scratch dir, no pool) so `_merge_cluster_files`, checkpoints (`:524`), and `total_agent_dispatches` (`:988`) are byte-identical.
- `bes_splice_enabled` → v2, no-op in v1.
- the mode-agnostic `rdr_preflight_gate` (Phase 2) off → env-build → `run_experiment` directly, as today.

**Regression test (parity anchor):** the suite already injects a counting `reproduce_fn` (`test_controller_parallel.py:37`,`:149-165` asserts `concurrent_peak==1`; `test_controller.py:275-304`). Add `test_bes_flags_off_dispatch_count_unchanged` — with all `bes_*` defaults, `reproduce_fn` call count == `len(clusters)` and the `final_report.json` + the deterministic `iterations/cluster_{i}_{id}.json` filenames (`:272`) are identical to the pre-BES commit (a clean golden-equality assertion). **Also add `test_bes_enabled_false_overrides_child_flags`** — set `bes_candidates_per_cluster=4` + `bes_splice_enabled=True` but `bes_enabled=False`, and assert byte-identical parity (proves the master gate, not just the child defaults).

## 6. C1 (deferred) — route default-mode baseline-construction through RDR

The serial `write→error→restart` pain lives in the RLM half: `implement_baseline` (`primitives.py:1322`) + serial `repair_context` (`:1397`) re-invokes one agent to fix everything. Routing it through RDR's parallel dispatch is attractive but is a **separate migration, NOT in-scope for BES** (verified, Agent 4):
- Needs a **new method-spec→module decomposer** (RDR's `decompose` is over the *rubric*, not the baseline plan).
- Must reconcile two code-dir-ownership models and give up / port `implement_baseline`'s warm-retry cache + contract plumbing.
- Must reconcile the `ForcedIterationPolicy` loop (RLM) with RDR's dispatch-count iteration model.
- **Blast radius:** BES is flag-gated over `run_rdr`; C1 alters the default path every non-flagged run uses (no parity escape hatch).

**Sequence:** land BES deltas 3→1→2b first; file C1 as a follow-on gated on delta-1 shipping, with its own design pass.

## 7. Testing

- Delta 3: a non-subclassing `*Env` is blocked + regenerated pre-GPU; `bes_preflight_gate=False` skips the gate.
- Delta 1: with `n=3`, three scratch dirs are scored statically, the winner is merged, `run_experiment` runs once; `n=1` is byte-identical to today.
- Delta 2b: `splice_cell_candidates` unions surviving cells and re-aggregates; OFF leaves `exp` unchanged.
- Parity: the golden flags-off test in §5.

## 8. Definition of done

- [ ] `REPROLAB_BES_*` flags added with a **`bes_enabled` master gate**; **flags-off parity proven by the golden test** AND the master-override test.
- [ ] Mode-agnostic pre-run gate (Phase 2) lands first; **Delta 1 (competing candidates, isolated scratch dirs, static SELECT) is BES v1**.
- [ ] **Delta 2 (evolve/splice) deferred to v2** with the GPU-cost / full-manifest honesty caveats (§4); 2a (cluster-file splice) explicitly not shipped (no leaf→file provenance).
- [ ] Per-candidate scoring is static (no extra GPU in v1); `total_agent_dispatches` reflects the N× token cost in the trace.
- [ ] C1 filed as a separate follow-on, not built here.
- [ ] **Rollback:** `bes_enabled=False` (master) restores today's RDR behavior with no residue; flags are the kill switch.

## 9. Expected effect

Faster, broader matrix coverage (parallel competing candidates) + honest partial assembly (cell-metric splice) + fewer GPU-burning errors on the RDR path (preflight gate). Whether this beats Phases 0–1 on $/rubric-point is the **A/B to measure** — run BES-on vs BES-off on a fixed paper set and compare pass-rate, $/paper, wall-clock, and rubric.
