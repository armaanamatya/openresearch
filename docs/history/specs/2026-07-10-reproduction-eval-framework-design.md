<!-- doc-meta: status=draft; last-verified=2026-07-10 -->
# Reproduction Evaluation Framework — Design

> **Status:** draft (design approved in brainstorming 2026-07-10; Codex-reviewed 2026-07-10 —
> all 7 findings applied; pending operator review).
> **Scope:** one design spec linking **three sequenced tracks**. Each track gets its own
> implementation plan via `writing-plans`; this document is the shared architecture + invariants.
> **Supersedes nothing.** Extends the in-flight eval-integrity work
> ([`2026-07-09-eval-integrity-track-a-design.md`](2026-07-09-eval-integrity-track-a-design.md),
> plan [`2026-07-10-track-a-eval-integrity.md`](../plans/2026-07-10-track-a-eval-integrity.md)).

## 1. Motivation

OpenResearch reproduces ML papers end-to-end. The next layer — per CLAUDE.md's "where it's
going" — is a **reproducibility evaluation layer** that (a) certifies *whether* a paper
reproduces, (b) emits a per-run **report**, and (c) produces a structured, provenance-rich
evidence base that **deepinvent.ai**'s improvement/patent-generation layer consumes
(reproduce → provenance-rich evidence → whitespace/improvement → patent draft). The eval is
**decoupled** from that downstream layer and **evidence-gated**.

A proposed 10-step framework (weighted composite of 11 evaluator dimensions, a reference
experiment-DAG, typed data models, verdict levels) motivated this design. We adopt its
*taxonomy* and *rigor* but reject the parts that invert the north-star. This spec records what
we build, what we deliberately do not, and why — grounded in the current code.

### 1.1 What already exists (ground truth, 2026-07-10)

A background subsystem map (7 agents, ~992k tokens) + a Codex verification pass established:

- **No declarative task-DAG exists anywhere** (networkx/graphlib are not even dependencies).
  Four orchestration regimes: (1) default free-form `rlm.completion()` REPL loop
  (`run.py:4504`) — the root writes arbitrary Python calling 19 primitives, `FINAL_VAR`
  terminates; (2) `OPENRESEARCH_LIFECYCLE_PRIMARY` (default-OFF) linear imperative chain
  (`lifecycle_driver.py:259-273`); (3) RDR cluster pipeline (`decomposer.py:188-197` —
  `WorkCluster` with a *fixed 3-tier* `depends_on` used only to inject predecessor artifacts, no
  scheduling); (4) the **outer campaign** deterministic state machine over whole-run *attempts*
  (`reproduction_campaign.py:330`) with a fail-closed durable ledger and a zero-LLM `DECIDE`
  rule table (`campaign_policy.py:769`). Experiments communicate **only** through shared on-disk
  artifacts.
- **Scoring is three deterministic layers, never a scalar LLM grade.** Layer 1: PaperBench leaf
  scorer (`leaf_scorer.py:1669`) — LLM-grades rubric *leaves* in batches, rolls up the weighted
  tree (`roll_up:76`, a skipped subtree → `None`, dropped from numerator+denominator, **never a
  phantom 0.0**), with `DEGRADED_LEAF_CEILING=0.35` and paper-hint invariant caps. Layer 2: two
  axes (`OPENRESEARCH_TWO_AXIS_VERDICT`) — `implementation_verdict` from
  `fidelity_score_from_rubric` (`two_axis_report.py:70`) + `replication_verdict`. Layer 3
  (Track A): **`verdict_authority.decide()`** (`verdict_authority.py:340`) — the single
  **grade-free** verdict writer (no grade parameter, structural sever), deciding from
  `result_fidelity` per-claim pass/fail/unmeasured + an `evidence_gate` bool + a downward-only
  `claim_gate_cap`. 4-level taxonomy: `inconclusive / contradicted / partial / reproduced`.
- **The evidence gate is fail-closed** (`report.py:1507-1527`, default-ON): a success+metrics row
  needs ≥1 forge-proof in-process `run_experiment_ok_calls`; a success row with 0 ok-calls is
  *forged* → `failed`. **It returns `False` when `run_experiment_ok_calls is None`** (replay /
  CLI-direct / RDR / any **out-of-process** path) — passed into `decide()` at
  `report.py:2405-2416`, and `verdict_authority.py:367-370` grants `reproduced` only when the
  gate holds. So an out-of-process re-grade currently caps at `partial` (see §6.6, finding 1).
- **Provenance is already three-way** on `RLMFinalReport` (`report.py:32`, fields ~`:74`):
  `reported_metrics` (paper/self-attested, *never scored*), `paper_claims` (*never scored as
  truth*), `baseline_metrics` (agent-*measured*, projected under `OPENRESEARCH_METRIC_PROVENANCE`,
  back-linked by `experiment_run_id` + `metrics_sha256`).
- **A structural-analogue composite already exists but off the verdict path and grade-tainted:**
  `ReproductionScore.composite_score()` (`backend/evals/schemas.py:26`, method `:42`) =
  `0.1·build + 0.2·run + 0.4·metric_match + 0.3·fidelity_score` — it folds an LLM
  `fidelity_score` (0.3) into the blend. Unwired to `final_report.verdict`.
- **An LLM-graded signal already participates in campaign DECIDE.** `campaign_policy.decide()`
  ANDs `final_report.meets_target` (`campaign_policy.py:798`) — a rubric signal the LLM leaf
  grader produces, and which skills can sharpen (`leaf_scorer.py:1930-1934/:2421-2464`) — into
  the reproduced terminal. Since 50d0f8fe it is **AND-gated by the grade-free authority verdict**
  (`campaign_policy.py:621-623/789-798`), so `meets_target` is *necessary but never sufficient*.
  The design constraint (§3.1, §2) is therefore about **new** fields, not this existing path.
- **Of the 11 proposed dimensions, only numerical-reproduction keys on the evidence layer
  today.** execution/environment/dataset are measured as verdict *gates*, not 0–1 scores;
  DAG-planning, debugging-recovery, scientific-analysis have **no evaluator surface**;
  efficiency/autonomy are surfaced but never verdict-affecting. `backend/evals/schemas.py:55`
  (`HypothesisScore`) / `:84` (`IntegrityReport`, with `data_leakage`/`selective_reporting`) is
  an **unwired** innovation-eval branch.
- **Cost/compute plumbing is siloed and partly misleading:** `cost_ledger.jsonl`
  (`resilience/cost.py:24`, priced `:80-99`) is **LLM-token-only, blind to GPU$/Foundry**; GPU
  cost is *reconstructed at run level* (`services/pricing/timing.py:197-224`) and there is **no
  true per-experiment start timestamp** (`timing.py:107-118`); `experiment_runs.jsonl` rows carry
  `wall_time_s` but no gpu_hours/retry (`primitives.py:4816-4877`).
- **Human-intervention tracking is greenfield:** operator ingress is the message routes
  (`routes/messages.py:84-148`) + campaign approval/resume (`reproduction_campaign.py:450-457`
  / `:547-554`); `respond_to_user` (`primitives.py:8986-9028`) writes *assistant* replies, not
  operator actions. Nothing records *what / when / why* a human changed.

## 2. Goals / Non-goals

**Goals.** A per-run **diagnostic scorecard** + typed `EvaluationReport` (json + md) that
(1) records every proposed dimension with **artifact-anchored provenance**, (2) keeps the
deterministic dimensions as **downward-only verdict gates** and the LLM-judged dimensions as
**display-only** diagnostics, (3) is fed a reference from **skills** (paper-type + dataset), not
hardcoded examples, (4) runs **100% on a cloud VM** with no laptop dependency, (5) structures
the run as a recorded **observed-DAG** for the report + the downstream patent layer.

**Non-goals (explicitly not built).** A composite score that gates the verdict · the proposal's
7 composite-threshold verdict levels · removing the free-form root (spectrum point S3) · a
hardcoded SDAR example · **any new scorecard/`EvaluationReport` field writing `rubric`,
`overall_score`, `target_score`, or `meets_target`** (the rubric surface that already feeds
DECIDE — finding 2) · any new signal entering `decide()` / DECIDE / a self-improvement fitness
term · a general graph engine where a recorded artifact suffices.

## 3. Shared invariants (bind all tracks)

1. **Evidence-not-grade, structurally.** `verdict_authority.decide()` stays the **sole** verdict
   writer. Every new signal runs *before* `decide()` as a **downward-only gate** (like
   `claim_gate_cap`) or is **display-only**. Nothing new may raise a verdict. **No new field may
   write the rubric surface** (`rubric`/`overall_score`/`target_score`/`meets_target`) that
   already feeds DECIDE (finding 2). All new writers must pass `assert_verdict_surface_unchanged`
   (`verdict_authority.py:147-155` governs `VERDICT_SURFACE_KEYS`), which Track 0 **extends to
   cover `demo_status.json`** (finding 3).
2. **Provenance is typed and three-way.** Every scorecard value is tagged
   `paper_reported | agent_measured | evaluator_computed` and anchored to an `evidence_bundle`
   sha256 receipt. Generated/inferred values are never treated as reported ground truth.
3. **Skill = structure · paper text = values · deterministic evidence = pass/fail.** Reference
   knowledge lives in skills; paper-specific target values come from span-grounded extraction;
   only measured artifacts produce a pass. No paper-specific logic in evaluator code.
4. **Cloud-VM, no laptop.** Root, executor, LLM-judge evaluators, and adversarial verification
   are cloud-resident and cloud-funded (auth via long-lived token in secret store, Track-D). The
   eval consumes only on-disk/GCS artifacts. Out-of-process re-grade requires the new **ok-receipt**
   (§6.6, finding 1) — the persisted `evidence_bundle` alone does **not** clear the
   `run_experiment_ok_calls is None` ceiling.
5. **Flag discipline.** Every new capability is `os.environ`-gated, default-OFF, **byte-identical
   when off**; a default flip needs ≥3 paired A/B runs + the grader-σ gate.

## 4. Architecture & sequencing

Three tracks, sequenced **Track 0 → (Track E ∥ Track G)**:

```
Track 0  Finish Track A  ──►  the grade-free verdict surface is committed + guarded
                               │
              ┌────────────────┴─────────────────┐
        Track E                              Track G
   Eval scorecard +                   DAG refactor S1→S2
   EvaluationReport                   (observed event-log → edges →
   (consumes the DAG)                  opt-in scheduled backbone)
```

Track E starts against an **S0-derived** graph (reconstructed post-hoc) and upgrades to the
**S1-recorded** graph when Track G lands — no throwaway coupling. Track G is flag-gated and never
blocks Track E.

## 5. Track 0 — finish Track A (prerequisite)

Track A is mid-Task-6: the 507-line "sever" (all six historical grade→verdict writers gated behind
`OPENRESEARCH_TWO_AXIS_VERDICT AND OPENRESEARCH_VERDICT_AUTHORITY`; `write_final_report_rlm`
invokes `_va.decide()` unconditionally as the true-final writer) is **uncommitted and untested**.

- [ ] Add `tests/agents/rlm/test_single_verdict_authority_guard.py` (the tripwire fires on a
      post-authority mutation).
- [ ] Add `test_verdict_authority_offstate.py` (byte-identical when either flag off).
- [ ] Add `tests/acceptance/test_adam_verdict_reground.py` — frozen `runs/prj_adam_local_1`
      re-grade **must land `inconclusive`** (its ruler holds one ambiguous primary → Rule 1),
      **not** `partial`.
- [ ] **Extend `assert_verdict_surface_unchanged` / `VERDICT_SURFACE_KEYS` to snapshot + re-read
      `demo_status.json`** (`report.py:2455-2468` writes it separately from `final_report.json`;
      the tripwire currently checks only the report dict at `report.py:2519-2525`) — finding 3.
- [ ] Commit the sever.

Rationale: these tests are the guardrail that stops any new dimension writer (Track E) from
re-mutating the verdict. Everything in Track E is designed to run *before* `decide()` or be
display-only, and to pass this tripwire.

## 6. Track E — eval scorecard + typed `EvaluationReport`

### 6.1 Dimension contract

Each of the 11 dimensions is a scorecard **row**: `{status enum, provenance, evidence_refs[],
detail}` — **not a bare float**. A row may **gate** (downward-only, before `decide()`) **only if
it is keyed on a deterministic on-disk artifact**. LLM-judged rows are **display-only** and
labeled as such; they enrich the report and the patent substrate but never touch the verdict, and
never write the rubric surface (§3.1).

| Dimension | Gates? | Source / plan |
|---|---|---|
| Numerical reproduction | **GATE** | `result_fidelity.evaluate` + `reproducibility_verdict._grade_claim` seed CI + `metric_binding` (already the authority's core) |
| Execution completeness | **GATE** | `_has_experiment_evidence` + forge-proof `run_experiment_ok_calls`; formalize as a row |
| Environment setup | **GATE** | `env_health.jsonl` verified exclusions (a dead env is excluded, never a fake 0) |
| Dataset resolution | **GATE** | `_detect_data_unavailable_leaves` exclusions (curated aliases) |
| Tables / figures | **GATE-lite** | `fig_*.json` sidecars (grader text-only, never PNG) + **new** lightweight `TableCell` model |
| Autonomy / human-intervention | display | **new deterministic** `HumanIntervention` records → autonomy metric; never gates |
| Efficiency | display | **new instrumentation** (§6.5); never gates, never a fitness term |
| Paper understanding | display | `fidelity_score_from_rubric` + rubric prose; feeds patents |
| DAG planning | display | derived-DAG vs skill reference DAG (coverage %); §7 |
| Debugging / recovery | display | `failure_capsules.jsonl` + `FailureAttribution` + repair ledger |
| Scientific analysis | display | reuse unwired `HypothesisScore` / `IntegrityReport` (`data_leakage`, `selective_reporting`) — high patent value |

Net **new deterministic** build is small: human-intervention capture + per-experiment efficiency
instrumentation (§6.5). The LLM-judged rows wire existing signals into display rows, never gates.

### 6.2 Composite (offline, display/rank only)

Refactor the existing `ReproductionScore.composite_score()` (`backend/evals/schemas.py:42`) to be
**deterministic-dominated** — drop or heavily curtail the `0.3·fidelity_score` LLM term and
re-weight onto `metric_match`/`run`/`build`. It stays strictly on the `backend/evals/` surface as
a **report + paper-ranking** stat and **can never reach `final_report.verdict` or
`demo_status.verdict`** (nor `rubric`/`meets_target`, §3.1). Weights are configurable; all
component rows are preserved separately.

### 6.3 Data models

No proposal model name exists verbatim; each maps to an existing analogue. **Track E adds a thin
typed *adapter/view* layer in `backend/evals/` that composes `backend/agents/schemas.py` types —
it does not fork them** (finding 7):

| Proposal model | Existing analogue → plan |
|---|---|
| `PaperSpecification` | `PaperClaimMap` (`agents/schemas.py:88`) + `repro_spec.json` `ComparisonSpec` + skill structure → a typed **view** composing them; no new source of truth |
| `ExperimentSpecification` | `ReproductionContract` (`agents/schemas.py:403`) — whole-paper; add per-experiment view from the DAG node |
| `DAGNode` / `ExperimentDAG` | greenfield (§7); reuse the SQLite node/edge pattern of `KnowledgeGraphService` |
| `ExecutionAttempt` | `AttemptAssessment` (`attempt_assessment.py:100`) — campaign-level; add per-node link |
| `FailureRecord` | `FailureAttribution` (`failure_attribution.py:87`) + 33-class `FAILURE_CLASSES` + `failure_capsules.jsonl` |
| `HumanIntervention` | **new** (§6.5) |
| `MetricResult` | `MetricSpec` (`agents/schemas.py:60`) / `MetricDelta` over the untyped on-disk `metrics.json` |
| `ArtifactRecord` | `ExperimentArtifacts` (`agents/schemas.py:475`) + `evidence_bundle` receipt (sha256) |
| `ReproductionComparison` | `MetricDelta` (`agents/schemas.py:705`: `relative_error`, Cohen's d, `ci95`) + per-claim CI |
| `EvaluationReport` | `RLMFinalReport` (`report.py:32`) + `RubricVerification.from_areas` (`agents/schemas.py:651`) → a typed superset carrying the scorecard rows |

Metric-specific and experiment-specific tolerances are honored (already in `equivalence_margin` /
`MetricDelta`); no single universal tolerance. Stochastic claims compare distributions across
seeds (already the seed-CI path), not a single run.

### 6.4 Reference via skills

The reference for a paper is composed at eval time from the **selected skills** (existing
`consult_skill` #19 / `OPENRESEARCH_SKILL_SELECT` → `active_skills.json` machinery):

- **paper-type skill** (`rl-reproduction`, `llm-reproduction`, …; `sdar-reproduction` /
  `tool-rl-reproduction` already exist) → the reference experiment-DAG template, expected metric
  families, standard baselines, eval protocol, common failure→recovery patterns, compute profile.
- **dataset skill** (`alfworld`, `wikitext-103`, …) → expected splits, sample counts, label
  distribution, checksum, preprocessing, access restrictions.

Extraction fills paper-specific values with **paper-text-span** provenance (offsets into
`parsed_full_text.txt`). Human curation is deferred (tiered ground truth); the pluggable
skill source keeps it dynamic. **Leniency guard:** a skill supplies *structure only* — it can
never supply a pass; pass/fail keys solely on measured artifacts. This is load-bearing because
skills already sharpen the LLM leaf grade that feeds `meets_target`/DECIDE (finding 2); a
dedicated test asserts a skill cannot flip a claim to pass.

### 6.5 New deterministic instrumentation

- **`HumanIntervention`** (finding 5) — one append-only `human_interventions.jsonl` writer,
  hooked at the **operator-ingress** points: the run/campaign message routes
  (`routes/messages.py:84-148`), campaign **approval/resume** (`reproduction_campaign.py:450-457`
  / `:547-554` / `:651-657`), mode changes, and any explicit manual-artifact/edit API. **Not**
  `respond_to_user` (that is the assistant side). Each record: `{ts, kind, what, why,
  artifact_diff, blocking}`; `kind` ∈ {credentials/access · clarification · dataset-approval ·
  env-repair · code-fix · config-choice · scientific-judgment · compute-approval · manual-artifact}.
  Credentials/legally-required access weighted differently from technical/scientific assistance.
  Feeds an **autonomy** display metric; **never** gates.
- **Per-experiment efficiency** (finding 6) — this is a real sub-task, not a field add: current
  GPU cost is reconstructed at *run* level and there is no per-experiment start timestamp. Add a
  **separate GPU ledger** (do not overload the token `cost_ledger`) plus, at `run_experiment`
  persistence time, `start_ts`/`end_ts`, a `gpu_plan` snapshot, `retry_id`, and provider/rate
  metadata on each `experiment_runs.jsonl` row. Efficiency stays display-only, never a fitness term.

### 6.6 Cloud-VM eval + out-of-process re-grade receipt (finding 1)

The scorecard and report generation run on the VM as part of finalize (in-process, where
`run_experiment_ok_calls` exists). To make a run **re-gradable out-of-process** (VM replay, CLI,
RDR) without being floored at `partial`, add a **durable, forge-resistant ok-receipt**: at
`run_experiment` success, persist `{experiment_run_id, ok, metrics_sha256, ts}` into the run dir
(extending the `evidence_bundle`). Teach `_authority_evidence_gate` to consume the receipt(s)
when `run_experiment_ok_calls is None`, reconstructing the forge-proof from the sha256-anchored
receipt (a receipt is written *only* on a genuine in-process ok, so the forge-resistance holds).
The `evidence_bundle` **as it exists today carries metrics/provenance only** and does **not** by
itself clear the ceiling — this receipt is new work, and its `_authority_evidence_gate` change
must pass the Track-0 tripwire. LLM-judge + adversarial-verification evaluators are cloud-funded
subprocesses. No step reads laptop-local state.

### 6.7 Verdict levels

Keep the **4-level** evidence-keyed authority taxonomy
(`reproduced / partial / contradicted / inconclusive`). Map the proposal's `INVALID_REPRODUCTION`
→ `contradicted`. The lifecycle words (`NOT_STARTED / SETUP_ONLY / PARTIAL_EXECUTION / …`) live in
the orthogonal **run-status** namespace, artifact/`stop_reason`-keyed — **never** derived from a
score.

### 6.8 Acceptance (the eval's own gate)

- Frozen **Adam** re-grade lands `inconclusive` (shared with Track 0).
- A **UCPO** artifact re-grade produces a coherent scorecard + observed-DAG.
- SDAR is a **stress goal**, never the framework's correctness oracle.

## 7. Track G — DAG refactor (S1 → S2, flag-gated)

### 7.1 S1 — observed run graph (phased; finding 4)

Typed producer→consumer edges are **not** derivable from today's `experiment_runs.jsonl` rows,
so S1 is two steps:

- **S1a — node-only observed event log.** Write a durable `dag_nodes.jsonl` during the run: one
  node per experiment/stage from signals already emitted (`experiment_runs.jsonl`
  `experiment_run_id/artifact_dir/wall_time_s/env_id/model_id`, lifecycle stages, failure
  capsules), each `{id, name, category, status, attempt_count, wall_s, gpu_hours, estimated_cost,
  actual_cost, failure_type, artifacts[]}`. Matches the campaign's fail-closed write-ahead
  durability (atomic + fsync). Free-form execution **unchanged**.
- **S1b — typed edges.** Add explicit **edge-emitting hooks at primitive boundaries** (a
  consumer records the producer `metrics_sha256`/`artifact_dir` it read) to build
  producer→consumer edges. Only after S1b does the observed graph carry dependency edges.

Consumed by Track E (planning-coverage row, report skeleton) and the downstream patent layer.

### 7.2 S2 — scheduled backbone (opt-in)

Behind `OPENRESEARCH_DAG_BACKBONE` (default-OFF), evolve `lifecycle_driver` from a linear chain
into a **scheduler** over the S1 graph: parallel frontier for independent experiments
(baselines/ablations/seeds), **failed-node propagation** (a failed node never marks downstream
complete — the same bug-class as phantom-0.0), per-node resume. It routes **through**
`verdict_authority.decide()` + campaign `DECIDE`, never around. Validation: cycle detection,
dependency existence, valid topological order, partial-completion, failed-node non-propagation.
The free-form root stays the default; S2 becomes a cloud default **only** after ≥3 paired A/B
runs show no quality regression.

### 7.3 Why not S3

Removing the free-form root deletes the capability behind every proven run (Adam 0.831, UCPO
0.25, SDAR) and buys nothing against the four stated goals (provenance, planning-score,
durability/parallelism, report structure) that S2 does not already deliver. Excluded.

## 8. Testing

Deterministic unit tests (proposal Step 8, adapted): cycle detection · missing-dep detection ·
valid topological order · partial DAG completion · **failed-node non-propagation** · numeric
tolerance + CI overlap · missing metrics → `unmeasured` (never auto-pass) · multi-seed aggregation
· human-intervention weighting · GPU-ledger cost aggregation · **composite never reaches the
verdict surface** · **scorecard/`EvaluationReport` fields never alter `meets_target` /
`AttemptAssessment` / `campaign_policy.decide()`** (finding 2 regression) · **a skill cannot flip
a claim to pass** (finding 3, §6.4) · unsupported discrepancy claim rejected · serialization
round-trip. Edge cases: empty paper spec · no successful experiment · missing dataset · missing σ
· single seed · NaN/Inf metrics · interrupted execution · duplicated nodes · inconsistent units ·
fabricated/unverifiable artifact → severe penalty (evidence gate). Plus the two acceptance
re-grades (§6.8) and the out-of-process re-grade receipt (§6.6).

## 9. Risks & limitations

- **Out-of-process re-grade ceiling** — the **new ok-receipt** (§6.6) is the fix; until it lands,
  VM re-grades of non-ledger runs cap at `partial` by design. The `evidence_bundle` alone does
  not clear it.
- **Skill-as-reference leniency loop** — mitigated by the structure/values/evidence separation
  (§3.3, §6.4) + the "skill cannot flip a pass" test; sharper because skills already influence
  the `meets_target` grade that feeds DECIDE (finding 2).
- **Auto-extracted reference is agent-derived** until human curation — acceptable for triage,
  flagged in the report's confidence field; patents pull from human-curated specs only.
- **S1b edge hooks + S2 parallel experiments** are the higher-effort parts of Track G; both are
  behind the flag + A/B, and S1a ships value (node log) before either.
- **GPU ledger** is a genuine sub-task (per-experiment timestamps + GPU plan snapshot), not a
  field add (finding 6); efficiency stays off the verdict path regardless.
- **Composite re-weighting** is a judgment call (commented in code); it must remain off the
  verdict path regardless of weights.

## 10. Open questions

1. Exact weights for the refactored offline composite (deferred to A/B tuning; not verdict-affecting).
2. Table data-model depth — minimal `TableCell{row_id, col_id, value, provenance}` vs a richer
   table type (start minimal).
3. Whether the ok-receipt lives inside the `evidence_bundle` file or a sibling `ok_receipts.jsonl`
   (implementation detail for the Track-E plan; must be atomic + fsync either way).

## 11. References

- In-flight: `2026-07-09-eval-integrity-track-a-design.md`, plan
  `../plans/2026-07-10-track-a-eval-integrity.md`; Track-D
  `2026-07-09-cloud-reliability-track-d-design.md`.
- Code anchors (verified 2026-07-10): `verdict_authority.py:340` (+`:147-155` surface,
  `:367-370` gate), `report.py:1507-1527` (evidence gate) / `:2405-2416` (into decide) /
  `:2455-2468` (demo_status write) / `:2519-2525` (tripwire) / `:32`,`~:74` (provenance),
  `leaf_scorer.py:1669` / `:1930-1934` / `:2421-2464`, `two_axis_report.py:70`,
  `result_fidelity.py:296`, `reproducibility_verdict.py:324`, `metric_binding.py:225`,
  `backend/evals/schemas.py:26`/`:42`/`:55`/`:84`, `backend/agents/schemas.py:60`/`:88`/`:403`/
  `:475`/`:651`/`:705`, `reproduction_campaign.py:330`/`:450-457`/`:547-554`,
  `campaign_policy.py:769`/`:621-623`/`:798`, `lifecycle_driver.py:259-273`,
  `decomposer.py:188-197`, `attempt_assessment.py:100`, `primitives.py:4816-4877`/`:8986-9028`,
  `routes/messages.py:84-148`, `services/pricing/timing.py:197-224`, `resilience/cost.py:24`,
  `run.py:4504`, `evidence_bundle.py`.

## 12. Review history

- **2026-07-10 — Codex adversarial review** (session `019f4b42`): 2 blockers + 5 majors, all
  verified-legitimate and applied — out-of-process receipt (§6.6/§3.4/§9), DECIDE `meets_target`
  scoping + test (§2/§3.1/§8), `demo_status` tripwire (§5), S1 phasing (§7.1), HumanIntervention
  capture points (§6.5), GPU-ledger sub-task (§6.5), anchor corrections (§1.1/§6.3/§11).
