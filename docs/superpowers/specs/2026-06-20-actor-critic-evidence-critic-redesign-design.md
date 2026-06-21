# Actor–Critic Harness: Unified Evidence Critic, Adversarial Validation, and Self-Learning

> **Doc status:** Proposed · design spec · authored 2026-06-20.
> **Lineage:** builds directly on `2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`
> (three-tier trust model: zero-metrics veto, lifecycle ledger, external validator, fix-first loop, recipes).
> This spec *unifies* that work into a single actor–critic architecture, strengthens the deterministic
> signal, makes the existing grok validator fire on every terminal path, and closes the self-learning loop.
> **Grounding:** the 2026-06-20 four-scout audit of `runs/` (133 runs, mean score 0.123), the orchestrator,
> the evidence layer, and the flag/config/test surface. Current-state claims cite verified `file:line`.

---

## 1. Motivation (grounded in the audit)

The harness already contains every ingredient of an actor–critic system, but the ingredients are
**scattered, mostly turned off, and never wired into a loop**:

1. **No single critic.** "Is this evidence real?" is answered by ≥6 ad-hoc predicates across
   `evidence_gate.py`, `zero_metrics_detection.py`, `stub_detection.py`, `leaf_scorer.py` (A7),
   `evidence_key.py`, `convergence_evidence.py`, `metric_reality_smoke.py` — different scopes
   (run / leaf / value), different stages, different defaults. There is no shared signal a loop
   could optimize against.

2. **The strongest checks are OFF in production.** The SDAR-v6 fabrication (real GPU training,
   all-0.0 metrics with *real* keys) is caught only by `ZERO_METRICS_GUARD` (default-OFF). The
   per-leaf fabrication veto (A7) is default-OFF (`leaf_scorer.py:291`) while a *different* check
   sharing the env var `OPENRESEARCH_EVIDENCE_GATE` is default-ON (`report.py:1509`) — a confirmed
   split-default bug.

3. **The critic is trivially evadable.** The grok external validator runs only on the happy
   finalize path (`run.py:3583`) and the FINAL_VAR gate (`run.py:3081`). The two paths a
   fabricating-or-timing-out run actually takes — `_finalize_fatal_primitive_abort` (`run.py:1365`)
   and `_hard_stop_with_report` (`run.py:1507`) — only `load_verdict` (read a previously persisted
   verdict); they never run the panel. The 2026-06-20 6h-SIGTERM run skipped validation for exactly
   this reason.

4. **`rerun_agrees` — the strongest anti-gaming check — is a permanent stub** (`external_validator.py:385`:
   "pending (P3) — not vetoed at P2").

5. **The within-run "advantage" leaks the grade.** The best-of-run floor selects by LLM-grade MAX,
   not evidence state, so a noisy high grade on weak evidence permanently inflates the shipped score.

6. **The caution has a cost.** ~290 `OPENRESEARCH_*` flags, ≥21 default-OFF behavioral gates, a
   ~57-paired-SDAR-run / ~$150 / 100+ hour validation backlog. Safe-by-default has become un-shippable.

The **red line** stays absolute: *the fitness signal is the deterministic evidence layer, never the
LLM grade.* This spec makes that line a single object, turns it on, and builds the loop on top of it.

---

## 2. The actor–critic model (the unifying frame)

> **Actor** = the RLM root (proposes + executes via primitives).
> **Critic** = one deterministic evidence signal (`EvidenceAudit`) + the existing **grok** validator as
> its adversarial layer.
> **Self-learning** = that critic signal — never the grade — drives within-run candidate
> selection/repair and cross-run recipe/lesson memory.

| Asked for | Pillar |
|---|---|
| "more deterministic evaluation" | **P1** `EvidenceAudit` unification |
| "strengthen evaluation" | **P2** rerun-agrees, kill grade leak, fix split-default, default-on |
| "integrate with the grok validator" | **P3** grok consumes the audit, fires on every path |
| "actor-critic" | **P4** advantage = Δ evidence; critic ranks candidates + gates repair |
| "self-learning agentic logic" | **P5** recipes/lessons admitted on evidence, injected to the actor |

The architecture is **paper-agnostic by construction**: every predicate keys on
provenance / metrics shape / VRAM / claim-grounding — never on SDAR specifics.

---

## 3. Pillar 1 — `EvidenceAudit`: the unified deterministic critic

The keystone. It is *also* the `run_experiment` decomposition, because the audit is the clean seam
between "run the experiment" and "judge the result."

### 3.1 The abstraction — one aggregator over existing predicates

New module `backend/agents/rlm/evidence_audit.py`. It does **not** reimplement predicate logic — it
**composes** the existing pure predicate modules into one snapshot with one chokepoint:

```python
# evidence_audit.py — the single aggregation point + the only "is evidence real?" entry points.
# Predicate *logic* stays in its current modules; this composes them.

@dataclass(frozen=True)
class EvidenceAudit:
    backed_by_ledger: bool          # >=1 in-process run_experiment ok call (binding ledger)
    provenance_present: bool        # code/provenance.json exists & well-formed
    metrics_non_degenerate: bool    # zero_metrics_detection: not all-zero / not bit-constant
    metric_keys_real: bool          # stub_detection: >=1 real-metric key, not only placeholders
    vram_plausible: bool | None     # antifab: net peak >= floor; None if unmeasured (remote)
    reasons: tuple[str, ...]        # human-readable veto reasons (for repair_context + SSE)
    fingerprint: str                # evidence_key over audited disk state -> the advantage signal
    @property
    def run_level_clean(self) -> bool: ...   # the ONE run-level predicate

def audit_evidence(ctx) -> EvidenceAudit              # reads disk state; used at scoring/finalize/validator
def result_is_fabricated(result, ctx) -> str | None  # run_experiment veto seam (reason or None)
def leaf_substantiated(leaf, metrics, scope) -> bool  # the A7 per-leaf check, re-homed here
```

`fingerprint` reuses `evidence_key.py`; `metrics_non_degenerate` delegates to `zero_metrics_detection`;
`metric_keys_real` to `stub_detection`; `vram_plausible` to the existing antifab logic;
`leaf_substantiated` is the A7 logic from `leaf_scorer.py` moved behind this interface.

### 3.2 Where it plugs in (replaces, does not add)

| Consumer | Today | After |
|---|---|---|
| `run_experiment` (primitives.py) | 3 separate guards (antifab, stub, zero-metrics) | one `result_is_fabricated(...)` veto seam |
| `leaf_scorer.score_reproduction` | inline A7 block (`leaf_scorer.py:1985`) | `leaf_substantiated(...)` |
| verdict gate (`report.py:_apply_evidence_gate`) | bespoke two-flag check | `audit_evidence(ctx).run_level_clean` |
| recipe/lesson admission | four separate predicate calls | `run_level_clean` |
| grok validator (P3) | recomputes `not_all_constant`/`provenance_present` | consumes the `EvidenceAudit` |
| fix-first loop (P4) | ad-hoc evidence fingerprint | `audit.fingerprint` |

### 3.3 The refactor it unlocks

`run_experiment` (2,253 lines, 119 bare `except`s) splits because the audit is the post-processing seam:

```
run_experiment:
    raw   = dispatch(backend).run(...)        # Local/Docker/RunPod/Azure runners behind one protocol
    reason = result_is_fabricated(raw, ctx)   # ONE shared veto, identical across all backends
    return apply_veto(raw, reason)
```

`SandboxExperimentRunner` protocol with one `run(...) -> ExperimentResult` per backend; shared pre/post
logic (audit, scope-check, cost ledger) lives in thin wrappers. Each backend becomes unit-testable
against a mock sandbox; OOM/stub/zero-metrics logic can no longer cross-contaminate backends.

### 3.4 Flag collapse

One master flag `OPENRESEARCH_EVIDENCE_AUDIT` replaces `ZERO_METRICS_GUARD`, `STUB_METRICS_GUARD`,
`ANTIFAB_GUARD`, the verdict-gate `EVIDENCE_GATE`, the leaf-gate `EVIDENCE_GATE`, and `EVIDENCE_FINGERPRINT`
— and **fixes the split-default** by making one variable mean one thing. See §9 for the migration.

---

## 4. Pillar 2 — Strengthening the critic signal

Four changes, all flowing through the `EvidenceAudit` from P1.

1. **Implement `rerun_agrees`** (`external_validator.py:182`/`:385`). Re-execute the *single cheapest
   cell* (smallest model × smallest dataset, one seed) and compare the headline metric to the recorded
   value within a relative tolerance (default **5%**). Result is recorded as a **new
   `rerun_agrees: bool | None` field on `EvidenceAudit`** (added in this pillar; `None` under the P1
   foundation) and becomes a real validator veto. GPU-cost-bearing → **budget-gated and sampled (1 cell)**,
   default-OFF behind `OPENRESEARCH_RERUN_AGREES`, recommended-ON only for fidelity-critical papers.
   `None` when budget/sandbox can't support it (never a false veto).

2. **Kill the grade-trusting best-of-run leak.** `_apply_best_of_run_floor` selects the floor by the
   **median grade keyed to the evidence fingerprint**, not the global grade MAX. This is the existing
   `EVIDENCE_FINGERPRINT` behavior promoted to be the default under the P1 master flag (no separate flag).

3. **Fix the `EVIDENCE_GATE` split-default.** Both call sites resolve through `audit_evidence`; the verdict
   gate stops reading `OPENRESEARCH_EVIDENCE_GATE` with a `"1"` default. One variable, one meaning. A
   regression test asserts both paths read the same effective state.

4. **Verdict/score consistency.** At the single write chokepoint (`write_final_report_rlm`): recompute
   `meets_target` from the final authoritative score (it is `None` in 100% of the corpus today), and
   **downgrade `verdict=reproduced` → `partial` when `score < target`** (the `pb_ftrl_1779413937`
   reproduced-at-0.0 bug).

---

## 5. Pillar 3 — Grok adversarial critic on every finalize path

### 5.1 The structural enabler: one finalize pipeline

The four finalize paths (`_finalize` 3380, `_finalize_fatal_primitive_abort` 1365,
`_hard_stop_with_report` 1507, and `_salvage_partial_report` 1467) diverge on which trust operations
run (validator, champion-restore, best-of-run floor, regrade). Replace the divergence with one object:

```python
@dataclass
class FinalizeContext:
    report; ctx; project_dir; emit; stop_kind   # normal | fatal_abort | hard_stop

def finalize_pipeline(fc: FinalizeContext) -> RLMRunResult:
    # ONE ordered trust pipeline, identical invariants for every terminal path:
    regrade(fc) -> champion_restore(fc) -> salvage_floor_if_hardstop(fc) ->
    run_validation_panel(fc) -> write_final_report_rlm(fc) -> emit_sse(fc) -> write_demo_status(fc)
```

Each of the three terminal handlers constructs the appropriate `FinalizeContext` and delegates. This
**closes the "induce an abort to skip the critic" hole**: the validator panel and champion-restore now
fire on abort and hard-stop, not just the happy path. It also removes the duplicate `mine_lessons`
call and the triple best-of-run-floor application the audit found.

### 5.2 Grok consumes the audit

`run_validation_panel` receives the `EvidenceAudit` and machine-checks its typed predicates against the
audit fields instead of recomputing them. The grok panel (cross-family vs the Sonnet executor =
`independent` separation per `role_models.py`) keeps its existing min-aggregation veto and
fingerprint-keyed verdict cache. `rerun_agrees` (P2) becomes a live predicate it can request. The
machine-checked veto stands even under `weak`/`degraded` separation; only the LLM suspicion-selection
is affected by separation strength (documented, unchanged).

**On a hard stop with no budget for a panel,** the pipeline stamps `validation: {status: "unavailable",
reason}` rather than silently shipping unvalidated — the critic's absence is recorded, not hidden.

---

## 6. Pillar 4 — The within-run actor–critic loop

### 6.1 Advantage = Δ evidence, never Δ grade

The actor's progress signal between attempts is the change in `EvidenceAudit.fingerprint` (P1) —
the fix-first loop already keys on an evidence fingerprint; this makes `audit.fingerprint` the canonical
one. A repair "made progress" iff the evidence state changed, regardless of grade movement.

### 6.2 Critic-ranked candidate selection

BES competing candidates are ranked by the critic, in order: (1) `EvidenceAudit.run_level_clean`
(hard filter — fabricated candidates are ineligible), (2) the deterministic leaf score on real
evidence, (3) the grok validator as tie-break for surviving candidates. The LLM grade is **never** the
selection key. This makes BES safe to run on the unified signal (the audit found the old static SELECT
margin sat inside grader noise).

### 6.3 Fix-first repair, gated on real improvement

A critic veto (audit or validator) feeds the distinct repair-refusal class in `ForcedIterationPolicy`;
progress is keyed to the evidence fingerprint changing; bounded by `OPENRESEARCH_REPAIR_MAX_ITERATIONS`;
stops honestly as `repair_exhausted` (excluded from the degenerate-loop counter — a stuck-but-trying
repair ships the honest verdict).

### 6.4 The `ForcedIterationPolicy` refactor (opted in)

`should_refuse` (339 lines, 61 branches, side effects embedded) becomes an **ordered rule chain**:

```python
class RefusalRule(Protocol):
    def check(self, state: PolicyState) -> PolicyDecision | None: ...

RULES = [WallClockFloor(), TerminalFailure(), IterationBudget(), ValidatorGate(),
         NoExperimentEver(), RepairFloor(), RubricFloor(), MinIterations()]
# should_refuse: first rule returning non-None wins; signature-stamping & callbacks
# move to the interceptor AFTER the decision (no side effects inside the decision).
```

Each rule is a small pure function with its own test; the actor–critic gates (validator, repair, rubric)
are just rules in the chain. The 35-field dataclass shrinks to a `PolicyState` the rules read.

---

## 7. Pillar 5 — Cross-run self-learning

The persistent policy memory of the actor–critic.

- **Recipes** (`recipe_library.py`) and **negative lessons** (`lesson_distiller.py`) are admitted
  **only** on `EvidenceAudit.run_level_clean` + deterministic `meets_target` (read off `metrics.json`),
  **never** the LLM grade. The brace-tracking static-import guard that structurally forbids reading grade
  fields for admission stays.
- Admission happens inside `finalize_pipeline` (§5.1) so it survives a wall-clock kill (today the lesson
  miner is skipped on hard-stop).
- Injection: active recipes/lessons for the same paper/`arxiv_id` enter the next attempt's implementer
  prompt (distinct from the paper's own data recipes), capped and advisory.
- This makes the loop *self-improving across runs*: the critic's evidence verdicts are the only thing
  that writes to memory, and memory biases the actor's next proposal.

---

## 8. The refactor (opted in) — summary

| God-unit | Today | After | Lands in |
|---|---|---|---|
| `run_experiment` | 2,253 lines, 5 backends inline | `SandboxExperimentRunner` protocol + per-backend runners + shared audit seam | §3.3 |
| finalize paths | 4 divergent handlers | one `FinalizeContext` + `finalize_pipeline` | §5.1 |
| `ForcedIterationPolicy.should_refuse` | 339 lines / 61 branches | ordered `RefusalRule` chain | §6.4 |
| `run_pipeline_rlm` | 1,292 lines | `_build_run_context` + `_execute_rlm_loop` + `_dispatch_finalize` | here |

`run_pipeline_rlm` splits into three testable phases: context/credential/prompt assembly, the
`rlm.completion` loop + exception handling, and finalize dispatch (which calls `finalize_pipeline`).
Each phase is independently unit-testable; the entry point becomes a ~30-line orchestrator.

---

## 9. Flag consolidation & migration

**`OPENRESEARCH_EVIDENCE_AUDIT` master flag**, rolled out so nothing breaks:

1. **Phase 1 — byte-identical.** Build `evidence_audit.py` + `finalize_pipeline` + the runner protocol.
   Existing guards/paths *delegate* to them; every legacy flag still works; full suite green; the
   default-off contract test proves zero behavior change.
2. **Phase 2 — opt-in full strength.** `OPENRESEARCH_EVIDENCE_AUDIT=1` activates the unified veto +
   median-evidence floor + validator-on-every-path. A/B as **one** batch (3 paired SDAR runs, not 18).
3. **Phase 3 — default-on + deprecate.** Flip the master default to ON; legacy flags become
   deprecated aliases (kept for `.env` back-compat, removed from docs).

Secondary consolidation (no behavior change): rename the four behavioral `REPROLAB_*` constants to
`OPENRESEARCH_*` keeping the bridge (fixes the post-import monkeypatch gotcha); collapse the duplicate
`DYNAMIC_GPU_ENABLED` alias; purge the dead `SCOPE_INCLUSION` doc references.

**Net:** ~6 flags → 1; ~18 of the ~57 queued validation runs eliminated.

---

## 10. Hardening track (separate, opt-in)

Justified by the run corpus but orthogonal to the actor–critic; sequenced independently so it never
blocks the core. Recommended because it addresses the **top preventable killers** the logs show:

1. **Credential preflight** (13 runs dead-on-arrival, 401). A health ping with the configured credential
   before the run subprocess spawns; fail fast with an actionable message. Extend `pre_flight_validator.py`.
2. **Orphan/stall salvage** (13+ runs killed with no report). Wire `sweep_orphaned_runs` to
   `finalize_pipeline` (hard-stop kind) before SIGKILL, salvaging on-disk evidence into a scored partial.
3. **`detect_environment` hardening** (38% failure rate — the worst primitive). Static validator on the
   inferred Dockerfile `FROM` line: known-base catalog + devel-vs-runtime against whether the paper
   compiles CUDA, before any GPU cost.
4. **`meets_target` population** — covered by P2 §4 (listed here for traceability).

---

## 11. Invariants

- **The red line:** admission to memory, candidate selection, the best-of-run floor, and the verdict
  gate read `EvidenceAudit`, never the LLM grade. The static-import guard enforces it for recipes.
- **Backward-compat:** master flag OFF (and all legacy flags at their current defaults) ⇒ byte-identical
  behavior, proven by `test_default_off_contract.py`.
- **Determinism:** `audit_evidence` is a pure function of on-disk state; same disk → same audit →
  same fingerprint. (`rerun_agrees` is the one non-deterministic input and is explicitly optional.)
- **Fail-soft vs fail-closed:** the deterministic audit fails *soft* on its own internal error (never
  blocks a run on a bug in the critic); the validator-client construction fails *closed* (a misconfigured
  validator raises rather than silently judging with the executor's own lineage — unchanged from today).
- **Egress:** no new corpus reaches disk or SSE; `EvidenceAudit.reasons` are predicate names + metric
  keys, never paper prose.

---

## 12. Testing strategy

- **Predicate unit tests:** each composed predicate, incl. the SDAR-v6 all-0.0-real-keys vector as a
  regression fixture that MUST veto under the master flag.
- **Property test:** `audit_evidence` determinism given disk state.
- **Contract test:** master flag OFF ⇒ byte-identical (extends `test_default_off_contract.py`).
- **Finalize-parity test:** all three terminal kinds run the identical trust pipeline (the validator
  fires on abort + hard-stop, not just happy path) — the regression that motivated §5.1.
- **Backend-runner tests:** each `SandboxExperimentRunner` against a mock sandbox in isolation.
- **Rule-chain tests:** each `RefusalRule` as a pure function.
- **Selection test:** BES ranks a fabricated-but-high-grade candidate below a clean lower-grade one.
- **Final gate:** **Codex code-reviews the full branch** after implementation (operator-run), atop
  continuous Opus diff review; Sonnet executes against this spec.

---

## 13. Rollout & phasing

1. **P1 foundation (byte-identical):** `evidence_audit.py`, runner protocol + `run_experiment` split,
   `finalize_pipeline` + the four-path delegation, `run_pipeline_rlm` three-phase split. Suite green.
2. **P2 signal strength:** `rerun_agrees`, median-evidence floor, split-default fix, verdict/score
   consistency + `meets_target`.
3. **P3 validator-everywhere:** grok consumes the audit; panel fires on every path via `finalize_pipeline`.
4. **P4 loop:** advantage = Δ fingerprint; critic-ranked BES; fix-first gating; `RefusalRule` chain.
5. **P5 self-learning:** evidence-only admission inside `finalize_pipeline`; prompt injection.
6. **Flag flip:** the one-batch A/B (§9 phase 2) → default-on (§9 phase 3).
7. **Hardening track:** independently, recommended-first credential preflight + orphan salvage.

Each phase ships behind the OFF master flag (byte-identical) until the A/B clears.

---

## 14. Non-goals

- No model fine-tuning / weight updates (the user chose inference-time actor–critic).
- No new validator transport — reuse the existing grok `OPENRESEARCH_VALIDATOR_BACKEND` path.
- No new sandbox backend; the runner protocol wraps the existing four.
- No rewrite of working predicate logic — `evidence_audit.py` *composes* the existing modules.
- No CLAUDE.md restructure beyond removing the confirmed drift (duplicate sections, RunPod-image /
  cloud-type contradictions, the "12 vs 17 primitives" mismatch).

## 15. Decisions baked in (flag any you'd change in review)

- `rerun_agrees` tolerance **5% relative**, **1 cheapest cell**, default-OFF (recommended-ON for
  fidelity-critical). 
- Master flag **default-ON only after** the one-batch 3-paired-SDAR A/B clears.
- Legacy flags kept as **deprecated aliases**, not deleted.
- Hardening track **included but sequenced last** (credential preflight + orphan salvage first within it).
