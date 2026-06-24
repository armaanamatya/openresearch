# ADR — The evidence-first actor / critic / validator architecture

> **Status:** Accepted (design) · 2026-06-21 · Phase 1 of the SDAR unification
> mega-prompt (`docs/superpowers/prompts/2026-06-20-sdar-unification-megaprompt.md`).
> Grounded in the read-only audit of `feat/grounded-harness-integration` (HEAD
> `f67304b2`, integration commit `d4c4271c`).
>
> **Purpose.** PR #110 landed a second design for parts of the evidence machinery
> alongside the trunk's. This ADR establishes the **single target architecture**,
> names every flag and its default, declares the **two canonical evidence
> sources**, and records what gets deduplicated vs. what is correctly separate —
> so the #110 features can be enabled without shipping two parallel evidence
> stacks. **Key finding of the audit: the duplication is far smaller than feared;
> most layers already compose or are single-sourced.** This ADR mostly *blesses
> the current composition* and points at three concrete, scoped changes.

---

## 1. The pipeline (target state — matches today's wiring, minus the noted forks)

```
                    ┌─────────────────────────────────────────────────────────┐
   ACTOR            │ RLM root (run.py + 17 primitives) — writes code, runs    │
                    │ experiments, calls verify_against_rubric                 │
                    └───────────────┬─────────────────────────────────────────┘
                                    │ run_experiment result
   IN-LOOP CRITIC (per-result)      ▼
     evidence_audit.apply_result_veto  [OPENRESEARCH_EVIDENCE_AUDIT, default OFF]
       composes stub_detection + zero_metrics_detection + VRAM plausibility
       → degrades a result to failure_class="fabrication_suspected"
                                    │
   IN-LOOP CRITIC (per-leaf)        ▼  (during leaf_scorer grading)
     evidence_gate.gate_decision  [OPENRESEARCH_LEAF_EVIDENCE_GATE, default OFF]
       vetoes a RESULT-claiming leaf score→0.0 when no on-disk per_model cell
       substantiates it (leaf_scorer._result_leaf_substantiated)
                                    │
   IN-LOOP REFUSAL                  ▼
     run.py::_claim_gate  [OPENRESEARCH_VALIDATOR_CHECK_REPORT etc.]
       refuses FINAL_VAR when report narrative makes ungrounded claims
       (claim_grounding engine)
                                    │
   PRE-GPU GATE                     ▼
     code_review_gate  [OPENRESEARCH_CODE_REVIEW_GATE, needs EXTERNAL_VALIDATOR]
       cross-family reviewer blocks fake/wrong-metrics code before GPU dispatch
                                    │
   FINALIZE  ──────────────────────▼────────────────────────────────────────────
     write_final_report_rlm (ALL finalize paths) runs, in order:
       1. _apply_evidence_gate  [OPENRESEARCH_EVIDENCE_GATE, default ON]  ◀── P0
            · primary no-evidence verdict downgrade  (FORGE DEFENSE — do not weaken)
            · AND evidence_audit.run_level_clean      (composed in today)
       2. rubric merge + verdict floor
       3. validation panel stamp (external_validator)  [OPENRESEARCH_EXTERNAL_VALIDATOR]
       4. best-of-run floor (write-chokepoint; idempotent; skipped when degraded/stop)
       5. score-fidelity verdict cap (reconcile_verdict_with_score)
       6. two-axis clamp  [OPENRESEARCH_TWO_AXIS_VERDICT]
       7. report_claim_gate  [OPENRESEARCH_REPORT_CLAIM_GATE]  (claim_grounding engine)
     build_final_report (CLEAN path only) additionally runs:
       · _apply_best_of_run_floor · _apply_champion_artifact
         [OPENRESEARCH_CHAMPION_ARTIFACT] · _reconcile_verdict_against_evidence
   EXTERNAL VALIDATOR (adversarial, fail-CLOSED transport, fail-soft panel)
     external_validator panel on every finalize path; reuses grader_transport
```

## 2. The two canonical sources of truth

1. **Run-level evidence snapshot — `evidence_audit.audit_evidence(dir)`.** The
   single deterministic snapshot of on-disk + ledger evidence
   (`backed_by_ledger`, `metrics_non_degenerate`, `metric_keys_real`,
   `rerun_agrees`). Already consumed by the verdict gate
   (`report.py::_apply_evidence_gate`, AND-composed) and intended for
   recipe-admission + the validator. **Other layers should read this snapshot,
   not re-derive run-level evidence.**
2. **Transport resolver — `grader_transport.build_transport_client` /
   `sample_completions`.** The single SDK-dispatch + sampler core for every
   sub-role client (grader, validator, verifier). Backends, azure
   `model→deployment` resolution, and fail-policy live here once.

Two **grounding notions** are correctly distinct and each single-sourced — do
not merge them:
- **Text-claim grounding** (`claim_grounding.py`): report-sentence number ↔
  metric identity. One engine, three flag-gated consumers (in-loop refusal,
  validator predicate, finalize cap).
- **Leaf-cell substantiation** (`leaf_scorer._result_leaf_substantiated`): rubric
  leaf-token ↔ successful `per_model` cell subject. Feeds the per-leaf gate.

## 3. Decisions (the recommendation matrix, with corrected risk gradient)

| # | Pair | Decision | Risk | Status |
|---|---|---|---|---|
| 1 | validator deployment resolution (`run.py::_validator_separation_tier`) ↔ `build_transport_client` azure branch | **SUBSUME** into one resolver | **LOW** — behavior-preserving | ✅ **DONE** (`resolve_azure_deployment`, commit `92e51df4`) |
| 2 | `evidence_audit.run_level_clean` ↔ `_apply_evidence_gate` (default **ON**) | **COMPOSE (already composes) — no structural merge** | **HIGH** — touches default-ON forge defense | ✅ **RESOLVED — no code change** (see below) |
| 3 | `evidence_audit.apply_result_veto` (per-result) ↔ `evidence_gate` (per-leaf) | KEEP-SEPARATE-BY-STAGE | n/a | no change |
| 4 | `evidence_audit` ↔ `champion_artifact` | KEEP-SEPARATE (orthogonal) | n/a | no change |
| 5 | claim-grounding consumers | already one engine | n/a | no change |
| 6 | finalize-path coverage asymmetry (champion + evidence-reconcile only on clean path) | **INTENTIONAL — not a defect** | **HIGH** — watchdog/SIGTERM blast radius | ✅ **RESOLVED — no code change** (see below) |

### Decision 1 — SUBSUME the transport deployment resolver (DO NOW)
`run.py::_validator_separation_tier` (run.py ~2103-2175) re-implements
`build_transport_client`'s azure `model→deployment` override; the comment at
~2146 admits it is "same logic as build_transport_client's azure branch," kept in
sync by comment only. **Change:** expose the resolved deployment from the
transport builder (e.g. a pure `resolve_transport_deployment(backend, model)` or
returning the resolved id) and have `_validator_separation_tier` consume it.
Behavior-preserving; gate on the existing transport + validator-wiring tests.

### Decision 2 — COMPOSE evidence_audit into the verdict gate, never RETIRE forge logic — RESOLVED: no code change
`_apply_evidence_gate` runs under `OPENRESEARCH_EVIDENCE_GATE` which is **default
ON** — this is the **P0 anti-forge defense** (the ledger/forged-row cross-check;
`test_evidence_gate_forge.py` + the replay tests are its dedicated CI gate). The
audit found the two **already compose**: `audit_evidence(...).run_level_clean` is
AND-ed into `_apply_evidence_gate` (`report.py:1524-1535,1617`), so a run already
gets ONE effective evidence verdict. A *structural* merge — folding the gate's
forged-row/ledger derivation into `audit_evidence` — would touch default-ON P0
code for **marginal behavioural benefit** (the composition already achieves the
unified verdict) at **high risk**. **Decision: keep the composition; do NOT
restructure.** `audit_evidence` remains the canonical run-level snapshot that the
gate *consumes*; the forge cross-check stays exactly where it is. If a future need
ever forces a merge, the non-negotiable gate is `test_evidence_gate_forge.py` +
the replay tests (the default-OFF contract is NOT sufficient), and the forged-row
logic is never deleted — only relocated intact.

### Decision 6 — finalize coverage asymmetry — RESOLVED: intentional, no code change
`_apply_champion_artifact` and `_reconcile_verdict_against_evidence` run only
inside `build_final_report` (clean `_finalize`); the fatal-abort and hard-stop
paths skip them. **Investigated (2026-06-21) — this is intentional and correct:**
- `_reconcile_verdict_against_evidence` is **downgrade-only** (report.py:602/629,
  "NEVER upgrades — only downgrades", `reproduced→partial`). Both non-clean paths
  already cap the verdict at `partial` **before** it would run
  (`_finalize_fatal_primitive_abort:1482` = `"partial" if evidence else "failed"`;
  `_salvage_partial_report:1577` = `reconcile_verdict_with_score("partial", …)`).
  So the missing reconcile is a **no-op** under that cap — adding it changes
  nothing. (And it could never *upgrade*, so there was never an upgrade hazard
  from the reconcile itself.)
- `_apply_champion_artifact` **can upgrade** (it restores the best-median artifact
  and ships its grade) and performs **file I/O** (`restore_snapshot` copies
  `code/`). Skipping it on the dying-process salvage path is the **deliberate
  conservative choice**: a hard-stop/SIGTERM finalizer ships the honestly-earned
  `partial`, it does not run a code-restore under time pressure to lift the score.

**Decision: leave both paths as-is.** The asymmetry encodes the correct policy
(clean path may optimise to the best artifact; salvage ships a conservative,
already-capped partial). No change to the watchdog/SIGTERM handlers.

## 4. Invariants this ADR locks in
- Every layer stays **independently flag-gated**; unset ⇒ byte-for-byte today.
- The **forge defense** (`OPENRESEARCH_EVIDENCE_GATE` default ON + its CI tests)
  is never weakened by a unification step.
- The **fail-CLOSED validator transport** stays fail-closed; the **fail-OPEN
  grader transport** stays fail-open. Do not merge their policies.
- No new run-level evidence derivation outside `audit_evidence`; no new transport
  dispatch outside `build_transport_client`.

## 5. Phase 1 outcome + sequencing note (for the operator)
**Phase 1 is complete.** The audit's central finding held up under scrutiny: the
evidence/critic/validator machinery is **already well-composed and single-sourced**;
the only genuine fork was the Azure deployment resolver (Decision 1, deduped). The
two "scary" items dissolved on investigation — Decision 2 already composes (a
structural merge is marginal value at high risk → no change), and Decision 6's
asymmetry is the intended conservative policy (→ no change). So Phase 1 ships as
**one real dedupe + an ADR that blesses the existing composition**, with zero
changes to default-ON forge defense or the watchdog/SIGTERM handlers.

**Sequencing:** the remaining gate is purely process — land/review **#115**
(history consolidation) and **#116** (#110 integration + this ADR + Decision 1)
before Phase 2 (BES SELECT → unified critic) builds on top. This is the operator's
call; nothing further should be stacked on the unmerged tower without it.
