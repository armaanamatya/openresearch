# Eval Integrity (Track A) — Two-Track Verdict Authority on Deterministic Result-Fidelity

- **Date:** 2026-07-09
- **Status:** Design (approved in brainstorming; Codex adversarial review pending)
- **Track:** A (eval integrity). Runs in parallel with Track D (cloud reliability,
  `2026-07-09-cloud-reliability-track-d-design.md`).
- **Follow-on specs (out of scope here):** B (eval-of-eval + rubric trust), C (HITL
  oversight), E (self-improvement re-grounding). Seams for B and C are placed here; the
  work is not.

## 1. Problem

The reproduction verdict today **inverts the north-star invariant** ("the fitness signal is
the deterministic evidence layer, never the LLM grade"). Grounded in `runs/prj_adam_local_1`:

1. **North-star inversion.** The shipped headline `verdict="reproduced"` is minted solely by
   the **LLM grade** (`rubric_score=0.7848`, byte-identical to a manual grok re-grade)
   crossing a `0.60` reconcile ceiling. The one deterministic paper-claim signal correctly
   says `replication_verdict="inconclusive", replication_credit=0.0` and is relegated to a
   non-headline side field.
2. **Verdict fragmentation.** Five disagreeing verdict-ish fields ship in one run
   (`verdict="reproduced"`, `verdict_pre_regrade="partial"`, `replication_verdict="inconclusive"`,
   `implementation_verdict="partial"`, `demo_status.verdict="unknown"`), written by many
   sites (`report.py`, `run.py:1286`, `two_axis_report.py`, `report_claim_gate.py`,
   `cli.py`, plus a manual hand-patch) with **no single reconciliation authority**.
3. **Grader brittleness.** An unparseable batch or SDK error silently zeros every leaf in
   the batch (`leaf_scorer.py` `_grade_batch` ~1937/2196), and there is **no in-run
   cross-provider fallback** — recovery today is a manual off-harness `scripts/regrade_adam_local.py`.
4. **Presence over result fidelity.** 14/25 full-credit leaves reward "code implements
   Algorithm 1"; the paper's actual numeric claim is never measured, yet the run ships
   "reproduced".

### 1.1 The load-bearing insight: the machinery already exists

The two-track verdict is **already implemented** behind `OPENRESEARCH_TWO_AXIS_VERDICT`
(default-OFF), and is **structurally dead on the arXiv/RLM path**:

- `repro_spec_extractor.py` — LLM extracts falsifiable quantitative claims into a frozen
  `rlm_state/repro_spec.json` (**this is "the LLM builds the ruler"**): per-claim
  `metric_name`, `direction`, `estimate_kind`, `claimed_effect`, `equivalence_margin`,
  `is_primary`, conservative `ambiguous` handling, blinded A6a re-extraction.
- `repro_spec_extractor.seed_bundle_from_metrics()` — reads the measured value from
  `code/metrics.json` at a `metric_key`/`model_key`/`env_key` path (**the deterministic
  measurement**).
- `two_axis_report.py` (`load_claims`, `compute_and_attach`, `fidelity_score_from_rubric`)
  + `reproducibility_verdict.compute_reproducibility_verdict` — the verdict engine, with a
  fail-closed **upgrade clamp** (a verdict upgrade requires ≥1 success-compatible in-process
  `run_experiment` ledger call, so a forged certificate cannot lift a verdict).

Three concrete deaths keep it inert on arXiv runs:

- **D1 — fidelity always 0.0.** `fidelity_score_from_rubric` iterates a PaperBench
  `rubric['areas']` shape; generated rubrics are a flat `sub_tasks` tree with no `areas`, so
  fidelity falls back to `overall_score` (often `None`→`0.0`). Impl verdict caps at
  `partial` → replication forced `inconclusive` on essentially every arXiv run.
- **D2 — the number is never located.** The extractor emits `metric_name` (prose, e.g.
  `"per_iteration_wall_clock_slowdown_factor"`), but `seed_bundle_from_metrics` needs a
  `metric_key` **path into `metrics.json`**. Nothing binds prose→path, so the deterministic
  reader finds nothing → `unmeasured` → `inconclusive`.
- **D3 — a grade-derived verdict wins the headline.** Two independent grade→verdict paths
  survive: (i) `report.py` score-based upgrades (`score≥0.60→reproduced`, ~lines 1952/2082),
  and (ii) when two-axis attach runs, `two_axis_report` overwrites `report["verdict"]` with
  `legacy_verdict` (`two_axis_report.py:303`), which `reproducibility_verdict` maps from the
  *implementation* fidelity — itself computed from the LLM leaf grade
  (`reproducibility_verdict.py:306/429`). So even the "deterministic" path projects the grade
  into the headline; a manual regrade can then re-mutate `verdict` again on top. **Fixing only
  the `0.60` projection is insufficient** — the `faithful→reproduced` legacy mapping must also
  be severed (§4.3).

**Therefore Track A promotes, completes, and makes-authoritative the existing abstraction —
it does not build a parallel one.**

## 2. Goals / Non-goals

**Goals**
- G1. **One authoritative reproduction verdict**, computed in exactly one place, keyed on the
  deterministic result-fidelity + evidence — never the raw LLM grade. All other sites read it.
- G2. **Result-fidelity actually measured** for the common claim kinds (numeric, relative,
  trend), by closing D1/D2 so the existing engine produces a real signal on arXiv runs.
- G3. **Verdict taxonomy** `reproduced | contradicted | partial | inconclusive` with honest
  semantics (see §4.3).
- G4. **Grader robustness** — no silent batch-zero; in-run cross-provider fallback; the LLM
  grade is relabeled a diagnostic (`impl_fidelity`), explicitly not the verdict. Retire the
  manual regrade script.
- G5. **Seams** for B (ruler-quality gate before freeze) and C (human contract approval).

**Non-goals**
- Building a new ruler/verdict engine (it exists — we complete it).
- The gold-set / cross-family eval-of-eval (Spec B).
- Wiring the ApprovalService / any UI (Spec C) — only the seam is placed.
- Self-improvement objective changes (Spec E) — though G1 makes E a small change later.
- Any qualitative-claim auto-pass. Qualitative claims cap at `partial` by design.

## 3. Locked decisions (from brainstorming)

- **Two-track** (Q1): the verdict keys on result-fidelity + evidence; the LLM grade is a
  diagnostic (`impl_fidelity`), never the verdict.
- **Tiered claim-spec, LLM-builds-ruler / deterministic-checker-measures** (Q2), taxonomy
  `reproduced|contradicted|partial|inconclusive`.
- Rollout is **flag-gated, default-OFF, byte-identical when off**; default-flip needs the
  repo's ≥3 paired-A/B + grader-σ gate (Track D owns the flip mechanics; Track A ships behind
  the existing `OPENRESEARCH_TWO_AXIS_VERDICT` master plus a new authority sub-flag).

## 4. Design

### 4.1 The ruler — promote `repro_spec_extractor` + close the binding gap (D2)

The ruler is `rlm_state/repro_spec.json`, produced by the **existing**
`repro_spec_extractor.extract_and_write` (LLM builds it, conservatively). The one addition:

- **`metric_binding`** — a new per-claim object bound at extraction time that maps the prose
  `metric_name` to a concrete measurement path the deterministic reader already understands:
  `{metric_key, model_key?, env_key?, baseline_key?, agg?}`. Two producers, in priority order:
  1. **Deterministic bind** — after the run produces `code/metrics.json`, a pure helper
     (`metric_binding.bind_claims(repro_spec, metrics)`) matches each claim's `metric_name`
     + `scope` tokens against the actual metrics.json key tree (reusing the tokenization
     already in `leaf_scorer`), emitting the path when the match is unambiguous.
  2. **LLM-proposed bind (fallback)** — when deterministic matching is ambiguous, the extractor
     asks the LLM for a candidate `metric_key` path (it *proposes where to look*, never judges
     the result).

  **Acceptance is a deterministic scope+unit gate, regardless of who proposed the path
  (closes review finding #2).** "The path exists in `metrics.json`" is necessary but NOT
  sufficient — `accuracy` for the wrong split/model/env exists yet is the wrong number. A
  candidate bind is ACCEPTED only when a pure check confirms the resolved path's scope keys
  (`model_key`/`env_key`/`baseline_key`/split) deterministically match the claim's declared
  `scope` (model/dataset/split from `repro_spec`) AND the metric's unit/`direction` are
  consistent with the claim. This gate is what rejects a scope-mismatched LLM proposal, so the
  LLM can never introduce a wrong bind. **Asymmetry:** a `fail`/`contradicted` (§4.3) requires a
  scope-verified bind; an unverifiable or ambiguous bind can never produce a `fail` — it stays
  `unmeasured`, so a mis-bind cannot manufacture a false `contradicted` (a false contradiction
  is the worst error). A claim whose bind fails the gate stays `unmeasured` → `partial`/
  `inconclusive`, never a false `pass`/`fail`. The accepted bind is written back into the
  **frozen** `repro_spec.json` so it is auditable (seam for C: this is what a human approves/edits).

### 4.2 `result_fidelity` — the deterministic checker

A pure module `backend/agents/rlm/result_fidelity.py` (stdlib-only core, no LLM):
`evaluate(repro_spec, run_dir) -> ResultFidelity` where

```
ResultFidelity = {
  per_claim: [ {claim_id, kind, status: "pass"|"fail"|"unmeasured", measured, target, margin, reason} ],
  result_fidelity_score: float,   # fraction of PRIMARY claims that pass (secondary weighted lower)
  primary_all_measured: bool,
  any_contradicted: bool,
}
```

It reuses `repro_spec_extractor.seed_bundle_from_metrics` (via the §4.1 bind) to read the
measured value, then applies a per-kind deterministic test:

- **numeric** — `|measured - target| <= equivalence_margin` (fields already present).
- **relative** — ordering + ratio/effect within `equivalence_margin` on the sign-folded
  `claimed_effect` (the extractor already folds lower-is-better into the sign).
- **trend** — sign/slope of the produced curve (`history.*` series in `metrics.json`) matches
  the claimed direction over the claimed window. New, small; reads the series the extractor's
  `direction` already implies.
- **qualitative / ambiguous / unresolved-bind** — `unmeasured` (never auto-pass).

This module *supersedes* the D2-blind seed-only path inside `reproducibility_verdict` by
feeding it a real per-claim result; `reproducibility_verdict.compute_reproducibility_verdict`
keeps ownership of mapping `(result_fidelity, fidelity, evidence)` → the two axes.

### 4.3 `VerdictAuthority` — the single, last, grade-free reconciliation point

A new pure module `backend/agents/rlm/verdict_authority.py` with exactly one entry:
`decide(*, result_fidelity, evidence_gate, fidelity_certificate, claim_gate_cap=None, ruler_quality=None) -> Verdict`.
**It takes no LLM-grade / `impl_fidelity` input** — the grade is structurally absent from the
decision (see "Sever" below).

**Scope (explicit):** governs the **reproduction verdict only** — `final_report.verdict`,
`implementation_verdict`, `replication_verdict`, and the `demo_status.verdict` mirror. It does
**not** touch the separate verdict namespaces: `run_watchdog` (`ok`/`kill`/`warn`),
`minimal_viable` (`viable`/`not_viable`), `campaign_report`'s `plan_only`. Those keep their words.

**Sever the grade→verdict path (closes findings #1/#4/#10/#11).** Today the grade reaches the
headline two ways: `report.py` score upgrades (~1952/2082) AND `two_axis_report` overwriting
`report["verdict"]` with `legacy_verdict` (`:303`), which `reproducibility_verdict` projects from
the *implementation* fidelity (LLM-grade-derived, `:306/429`). **Both are removed.** After the change:
- `VerdictAuthority.decide` keys **only** on `result_fidelity` (deterministic per-claim result) +
  the evidence gate + optional `claim_gate_cap`/`ruler_quality`. No fidelity/grade term exists in
  the signature — the sever is structural, not a convention.
- `implementation_verdict` and `impl_fidelity` remain **diagnostic axes** written beside the
  verdict, never folded into it. `two_axis_report` stops writing `report["verdict"]`;
  `reproducibility_verdict`'s fidelity→legacy-headline mapping is deleted (it keeps computing the
  two *diagnostic* axes).
- Every other historical verdict writer/mutator runs **before** the authority and feeds it as an
  input, or is excluded. In particular `report_claim_gate` (which today mutates
  `report_dict["verdict"]` *after* two-axis attach, `report.py:2202`) becomes a pre-authority
  **input** — its cap is the `claim_gate_cap` argument — not a post-hoc mutation.
- `VerdictAuthority` is invoked once, as the **last** verdict writer, at the finalize chokepoint
  (`report.write_final_report_rlm`), and stamps all surfaces atomically.
- **Guard test:** a static + runtime assertion that NO module writes `report.verdict` /
  `demo_status.verdict` for a reproduction run *after* `VerdictAuthority.decide` (catches a future
  post-authority mutator regressing the sever).

**Taxonomy — deterministic, explicit precedence (closes finding #5).** Over the `is_primary`
claims, apply in order:
1. **No measurable primary** — no claim is genuinely `is_primary` in the paper, or the only
   primaries are `ambiguous`/unresolved-bind → **`inconclusive`** (`reason=no_measurable_target`).
   The extractor's "auto-promote the first claim to primary" (`repro_spec_extractor.py:981`) is a
   spec-shape convenience and is **ignored for the verdict** — an arbitrarily-promoted secondary
   never yields a headline verdict.
2. **Any primary `fail`** → **`contradicted`** (a measured miss dominates; never hidden inside
   `inconclusive`).
3. **Any primary `unmeasured`** (none `fail`) → **`partial`** (faithfully attempted, unmeasured).
4. **All primaries `pass`** on the declared `eval_split` AND evidence gate satisfied (≥1
   success-compatible in-process `run_experiment` ledger row for the cited cells + real on-disk
   metrics) → **`reproduced`**.

Precedence is `inconclusive(no-target) → contradicted → partial → reproduced`; `contradicted`
strictly outranks `unmeasured`, so a failing primary is never masked by an unmeasured sibling.
This replaces the current "collapse by weakest status" (`reproducibility_verdict.py:517`) which
lumps `fail` and `unmeasured` together.

**Consumers must read the authority, not the grade (closes finding #3).** The campaign terminal
gate and scope-escalation today key on `final_report.meets_target` (grade-derived,
`campaign_policy.py:781/618`; `attempt_assessment.py:318`). Track A repoints the campaign
`REPRODUCED`/escalation gates to require the authoritative `verdict == "reproduced"` — a
grade-high/`meets_target=True` run whose deterministic verdict is `inconclusive`/`contradicted`
can no longer terminate the campaign as reproduced. The *full* objective re-grounding
(propose_improvements, BES ranking) stays Track E; only the terminal correctness gate moves here,
because leaving it grade-keyed would defeat the whole unification.

### 4.4 Fix the three structural deaths

- **D1** — `fidelity_score_from_rubric` gains a flat-`sub_tasks` path: derive fidelity from
  the generated rubric's leaf tree (weighted by the existing per-leaf `weight`) when `areas`
  is absent, instead of collapsing to `overall_score`/0.0. This fidelity feeds **only the
  `implementation_verdict` diagnostic axis**, never the headline verdict (§4.3 sever) — it makes
  the *diagnostic* honest on arXiv runs, it does not re-enter the decision.
- **D2** — closed by §4.1 (`metric_binding`).
- **D3** — closed by §4.3 (the legacy projection no longer writes `verdict`).
- **Certificate (evidence, not grade)** — `fidelity_certificate_builder` registers the paper's
  declared constants (from `repro_spec`/hyperparameters) so `mutation_confirmed` can
  legitimately go green and *strengthen* the evidence gate's anti-forgery signal. It is an
  input to the evidence gate, not an independent verdict cap: a `reproduced` still requires the
  in-process ledger row + a measured primary-claim `pass`; a non-registrable certificate is an
  evidence gap that keeps the existing upgrade clamp conservative, not a silent global pin to
  `partial`.

### 4.5 Grader hardening (the grade becomes a diagnostic)

- **No silent batch-zero.** In `leaf_scorer._grade_batch`, a parse/SDK failure retries on an
  in-run **fallback provider** (`grader_transport` gains an ordered fallback list, e.g.
  primary Foundry-Sonnet → grok → oauth-Sonnet) before any leaf is scored `0.0`; a leaf that
  cannot be graded after fallback is marked `ungraded` (excluded from the diagnostic), **never
  silently 0.0**, and raises a loud `grader_unavailable` warning.
- **Relabel + fully sever (closes finding #10).** The rolled-up leaf grade is surfaced as
  `impl_fidelity` (diagnostic) in `final_report` and is **not** an input to
  `VerdictAuthority.decide` at all (§4.3). Consequence: a grader provider outage or a batch
  parse failure changes only the diagnostic, **never** the verdict — the whole point of severing.
- **Retire** `scripts/regrade_adam_local.py` once the in-run path is robust (kept only as a
  debugging tool, not a finalize dependency).

### 4.6 Seams for B and C (placed, not wired)

- **B (eval-of-eval):** `VerdictAuthority.decide` accepts an optional `ruler_quality` gate
  result; when absent (Track A alone) it is treated as `trusted`. Spec B fills it with the
  gold-set separation verdict.
- **C (HITL):** `repro_spec.json` is written to a `pending_contract` state and frozen by a
  single `freeze_contract(run_dir)` call. Track A calls `freeze_contract` immediately
  (auto-freeze); Spec C interposes the async ApprovalService offer + default-auto timeout on
  that one call. No lifecycle restructuring needed later.

## 5. Data flow & artifacts

```
paper ─► repro_spec_extractor.extract_and_write ─► rlm_state/repro_spec.json  (ruler; frozen)
run   ─► code/metrics.json ─► metric_binding.bind_claims ─► repro_spec.json[+metric_binding]
                                    │
                                    ▼
        result_fidelity.evaluate ─► per-claim pass/fail/unmeasured  (DETERMINISTIC)
leaf_scorer._grade_batch ─► impl_fidelity  (DIAGNOSTIC, fallback-hardened)
                                    │
                                    ▼
        reproducibility_verdict + VerdictAuthority.decide ─► ONE verdict, stamped everywhere
```

`final_report.json` changes: `verdict ∈ {reproduced,contradicted,partial,inconclusive}` (one
authority); add `result_fidelity{per_claim,score}`; rename headline grade to
`impl_fidelity` (keep `rubric_score` as a back-compat alias for one release). `demo_status.verdict`
is stamped from the same authority at finalize (no longer hardcoded `unknown`).

## 6. Key interfaces

```python
# metric_binding.py
def bind_claims(repro_spec: dict, metrics: dict) -> dict          # returns repro_spec + metric_binding
# result_fidelity.py
def evaluate(repro_spec: dict, run_dir: Path) -> ResultFidelity   # pure, no LLM
# verdict_authority.py  — NO grade / impl_fidelity input (structural sever, §4.3)
def decide(*, result_fidelity, evidence_gate, fidelity_certificate,
           claim_gate_cap=None, ruler_quality=None) -> Verdict
def freeze_contract(run_dir: Path) -> Path                        # seam for Spec C
```

## 7. Testing & acceptance

- **Acceptance (headline, runnable now, no cloud):** re-run the verdict on the on-disk
  `runs/prj_adam_local_1` → `verdict` must flip from `"reproduced"` to **`"partial"`**
  (impl_fidelity ≈ 0.78, primary SFO claim `unmeasured` because SFO was omitted). If it still
  says `reproduced`, Track A failed.
- **Contradicted path:** a synthetic run whose measured number violates the claim → `contradicted`
  (not `inconclusive`).
- **Per-kind unit tests** for `result_fidelity` (numeric/relative/trend/unmeasured), reusing
  the `repro_spec_extractor` test fixtures.
- **Binding tests:** deterministic bind resolves the Adam metrics.json paths; ambiguous →
  `unmeasured`, never a wrong path.
- **Single-authority guard test:** static + runtime assertion that only `verdict_authority`/the
  finalize chokepoint writes a reproduction `verdict`, and that no writer runs *after* it
  (catches a post-authority mutator like `report_claim_gate`).
- **Taxonomy precedence tests:** any-primary-`fail`→`contradicted` even with an unmeasured
  sibling; no genuine primary → `inconclusive(no_measurable_target)` (auto-promoted secondary
  ignored); mixed pass/unmeasured (none fail) → `partial`.
- **Grade-severance test:** perturbing the LLM grade / simulating a grader outage does NOT change
  the `verdict` on fixed deterministic artifacts.
- **Scope-bind test:** a metrics path that exists but whose scope (model/env/split) mismatches
  the claim is rejected → `unmeasured`, never a `pass`/`fail`.
- **Campaign-consumer test:** a `meets_target=True` but `verdict=inconclusive` attempt does NOT
  let the campaign terminate `REPRODUCED`.
- **Grader fallback test:** a primary-grader parse failure falls back and still grades; a total
  failure yields `ungraded` + warning, never `0.0`.
- **Off-state byte-identical test:** master flag off ⇒ prior behavior unchanged.

## 8. Rollout

- Ships behind `OPENRESEARCH_TWO_AXIS_VERDICT` (existing master) + a new
  `OPENRESEARCH_VERDICT_AUTHORITY` sub-flag; both off ⇒ byte-identical to today.
- The default-ON flip is Track D's A/B responsibility (≥3 paired SDAR runs + σ-gate). The
  *acceptance* re-grade of the Adam local run gates the flip on the eval side.
- `impl_fidelity`/`rubric_score` dual-write for one release for downstream/UI compatibility.

## 9. Risks & mitigations

- **R1 — a wrong `metric_binding` produces a false `contradicted`.** Mitigation: unresolved/
  ambiguous binds are `unmeasured` (never fail); `contradicted` requires a resolved bind + a
  measured value outside the equivalence margin; the extractor's conservative `ambiguous` flag
  already forces `inconclusive` on shaky claims; Spec B's separation test catches lenient/
  wrong rulers before auto-freeze.
- **R2 — fewer "reproduced" verdicts** (honest but a product-optics change). Mitigation:
  `partial` carries a clear "faithfully implemented, result unmeasured" story; this is the
  point of the two-track decision.
- **R3 — grader fallback adds latency/cost.** Mitigation: fallback only fires on a parse/SDK
  failure (rare), bounded to the existing per-batch budget.

## 10. Out of scope (own specs)

- **B** — gold-set separation + cross-family panel + `spec_validator` teeth (fills
  `ruler_quality`).
- **C** — ApprovalService wiring at the 4 gates + SSE/UI (interposes on `freeze_contract`).
- **E** — repoint the campaign objective from `meets_target(grade)` → `result_fidelity`.
