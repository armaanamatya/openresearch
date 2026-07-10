<!-- doc-meta: status=active-handoff; last-verified=2026-07-10 -->
# Reproduction Eval Framework — Track E/G End-to-End Handoff (2026-07-10)

> **Purpose.** Resume the approved 3-track build in a fresh session and finish it end-to-end.
> This doc carries the exact file:line anchors, interfaces, and gotchas so you do **not** re-recon.
> **Design spec (approved):** [`docs/superpowers/specs/2026-07-10-reproduction-eval-framework-design.md`](../superpowers/specs/2026-07-10-reproduction-eval-framework-design.md)
> **Track E plan (executable):** [`docs/superpowers/plans/2026-07-10-track-e-eval-scorecard.md`](../superpowers/plans/2026-07-10-track-e-eval-scorecard.md)
> **Track A plan (context):** [`docs/superpowers/plans/2026-07-10-track-a-eval-integrity.md`](../superpowers/plans/2026-07-10-track-a-eval-integrity.md)

## 0. Start here (30-second orientation)

- **Branch:** `feat/gke-gpu-path-reproduction-reliability` (ahead of `deepinvent` remote, **unpushed** — push only on operator request).
- **Sequence chosen by operator:** **Track E first, then Track G.** Delegation posture: **guarded hybrid** — lead implements evaluator-core/verdict-adjacent tasks; delegate only mechanical, fenced pieces to Sonnet with hardened guardrails (§7); review every diff.
- **What's done:** Track 0 (finish Track A) + Track E Tasks 1, 2, 3a + the critical inert-authority fix. **All committed, `-n auto` green.**
- **What's next (in order):** Track E **3b → 4 → 5 → 6 → 7 → 8**, then **Track G** (S1a → S1b → S2), then **WS4/WS5** (the real coverage levers).
- **First command in a fresh session:** `git -C /home/abheekp/openresearch log --oneline -8` then read the two plan docs above, then §5 Task 3b here.

## 1. What this build is

A per-run **diagnostic scorecard** + typed **`EvaluationReport`** (json+md) recording all 11 evaluator dimensions with artifact-anchored provenance — deterministic dimensions as **downward-only verdict gates**, LLM-judged dimensions **display-only** — feeding deepinvent.ai's patent layer (reproduce → provenance-rich evidence → whitespace → patent). The discrete evidence verdict (`verdict_authority.decide`) stays the **sole** gate. Three sequenced tracks: **Track 0** (finish the grade→verdict sever, DONE) → **Track E** (scorecard, in progress) ∥ **Track G** (observed-DAG → opt-in scheduler).

## 2. Committed this turn (verify with `git log`)

| Commit | What | Tests |
|---|---|---|
| `15880345` | **Critical fix** — `normalize_repro_spec_claims` (idempotent nested→flat claim lift) in `result_fidelity.py`. Extractor writes claims NESTED (`comparison:{…}`); evaluator read FLAT → every real run's verdict was pinned `inconclusive`. Now the authority produces real signal. | in `test_result_fidelity.py` |
| `c1f384c4` | **Track 0** — `report.py` demo_status.json verdict tripwire (finding-3) + `test_verdict_authority_offstate.py`. | 8 + guard suite |
| `6b296d4c` | **E Task 1** — `ReproductionScore.composite_score` deterministic-dominated (`CompositeWeights` `{build .15, run .25, metric_match .60, fidelity 0}`) + `test_eval_composite.py` structural guard. | `tests/test_eval_composite.py` |
| `7f47c4a3` | **E Task 2** — `human_intervention.py` + 5 operator-ingress hooks (`messages.py` ×2, `reproduction_campaign.py` ×3) + `autonomy_metric`. | `tests/agents/rlm/test_human_intervention.py` (6) |
| `4ff678b0` | **E Task 3a** — `gpu_ledger.py` writer + `aggregate_gpu_cost`. **Module only; wiring is 3b.** | `tests/agents/rlm/test_gpu_ledger.py` (5) |
| `f56ae9dc`,`267c3c7d` | Track E plan + Task-3 split note. | — |

**Also committed earlier this session by a subagent, verified-correct + kept:** `4a1ab068` (sever + `is_enabled`/surface-guard), `429ace14` (Adam acceptance test), `4f0042e8` (WS3 durable-orchestration **design doc** — agent-authored, needs a full lead+operator review before WS3 coding).

## 3. Load-bearing invariants — READ before touching any verdict-adjacent file

1. **`verdict_authority.decide()` is the SOLE verdict writer.** Every new signal runs *before* `decide()` as a **downward-only gate** (like `claim_gate_cap`) or is **display-only**. Nothing new may raise a verdict.
2. **Verdict-surface tripwire.** `verdict_authority.assert_verdict_surface_unchanged(stamped, current, context=...)` governs `VERDICT_SURFACE_KEYS = ("verdict","implementation_verdict","replication_verdict")` on BOTH `final_report.json` and `demo_status.json` (`report.py` ~2519-2545). **Any new finalize-path writer must pass it** — run `tests/agents/rlm/test_single_verdict_authority_guard.py` after every `report.py` change.
3. **No new field writes the rubric surface** (`rubric`/`overall_score`/`target_score`/`meets_target`) — that surface already feeds campaign DECIDE (finding 2).
4. **Composite is report/rank-only** — confined to `backend/evals/` (schemas → store SQLite). `test_eval_composite.py::test_composite_never_reaches_verdict_surface` enforces no verdict module imports `backend.evals.schemas`/`store`. (NB: `report.py:1200` legitimately imports `backend.evals.paperbench.leaf_scorer` — the grader — which is a DIFFERENT module.)
5. **Flag discipline.** New capability = `backend/agents/rlm/feature_flags.py::env_truthy("FLAG")`, default-OFF, byte-identical off, hermetic OFF+ON test pair (`tests/CLAUDE.md`). Master flag for the scorecard: `OPENRESEARCH_EVAL_SCORECARD`. Sub-flags already coined: `OPENRESEARCH_HUMAN_INTERVENTION_LOG` (done), `OPENRESEARCH_GPU_LEDGER` (done), `OPENRESEARCH_OK_RECEIPT` (Task 4).
6. **Skill = structure · paper text = values · deterministic evidence = pass/fail.** A skill can NEVER supply a pass (Task 7 leniency guard).

## 4. Gotchas (so you don't lose an hour)

- **Serial-order test quirk (pre-existing, benign):** 3 tests in `test_single_verdict_authority_guard.py` (`*demo_status*`, `*live_sequence*`, `*fail_closed_mirrors*`) fail when the whole `tests/agents/rlm/` dir runs **serially**, but pass under `-n auto` (the real CI mode) and in isolation. An unknown other test in the dir leaks state; it is NOT your change. Always validate with `-n auto`.
- **The design spec's §5 "sever uncommitted/untested" snapshot is STALE** — Track A was ~90% committed at session start. Trust the code + git log over the spec's prose.
- **`isolation: worktree` branches from a STALE base** (~181 commits behind HEAD — missing `feature_flags.py`/`reproduction_campaign.py`). Delegated shared-file edits land against wrong code. **Do primitives.py/report.py-touching tasks in-tree**, or verify the worktree base == feature HEAD first. See memory `feedback_subagent_git_guardrails`.
- **Adam acceptance docstring is now stale** (Track E plan **Task 0**, doc-only): with the normalize fix the Adam claim is a *visible ambiguous primary* (→ `inconclusive` via Rule 1), not "invisible." Update `tests/acceptance/test_adam_verdict_reground.py`'s docstrings; assertions unchanged.

## 5. Remaining Track E tasks (exact anchors — do these in order)

All anchors verified 2026-07-10; match by symbol (lines drift ±10). Each task is TDD: write failing test → confirm fail → implement → confirm pass (`-n auto`) → `uvx ruff@0.15.16 check <files>` → commit.

### Task 3b — GPU-ledger wiring (in-tree; module already exists)
- **Module (done):** `backend/agents/rlm/gpu_ledger.py::append_gpu_ledger(project_dir, *, experiment_run_id, start_ts, end_ts, gpu_plan, provider, rate_usd_per_hr)`.
- **Wire in `backend/agents/rlm/primitives.py`:** capture `start_ts` in `run_experiment` near the `gpu_plan` load (`primitives.py:7034-7040`); stamp `start_ts`/`end_ts`/`gpu_plan`/`retry_id` onto the result via `setdefault` at the `_stamp_manifest_ids(result, …)` call seam (`primitives.py:7471` — `experiment_run_id`/`env_id`/`gpu_plan`/`_retry_idx` all in scope); call `append_gpu_ledger` at the persist success path (`primitives.py:4959`, just after the `experiment_runs.jsonl` write in `_persist_experiment_result`:4880/4954). `end_ts` ≈ the persist `entry["timestamp"]`. **Off-flag byte-identical** (gate every addition on `gpu_ledger_enabled()`). Provider/rate from `gpu_plan`/pricing (`services/pricing`).
- **Test:** extend `tests/agents/rlm/test_gpu_ledger.py` or a new primitives test — assert the experiment row has NO `start_ts` key when the flag is off.

### Task 4 — Out-of-process ok-receipt (lead; verdict-adjacent)
- **New:** `backend/agents/rlm/ok_receipt.py` — `write_ok_receipt(project_dir,*,experiment_run_id,ok,metrics_sha256,ts)` (atomic append to `rlm_state/experiment_ok_receipts.jsonl`, written ONLY on genuine in-process success), `count_ok_receipts(project_dir)` (distinct `experiment_run_id` with `ok is True` AND non-empty `metrics_sha256`), `ok_receipt_enabled()` → `OPENRESEARCH_OK_RECEIPT`.
- **Persist** at `primitives.py:4959` (success path; `ok=result.get("success") is True`, `metrics_sha256` from `_manifest_enrichment`:4862, `experiment_run_id` from `_stamp_manifest_ids`:4867/7471, `ts=entry["timestamp"]`).
- **Consume** in `report.py`: `run_experiment_success_count(ctx)` (**`report.py:1906-1926`**) returns `None` when `ctx.cost_ledger is None` (out-of-process). Add a fallback: when `ctx.cost_ledger is None` AND flag on, return `count_ok_receipts(ctx.project_dir)` — this lifts the `partial` ceiling. `_authority_evidence_gate` (`report.py:1507`) then flips `False→True`. **Must pass the Track-0 tripwire + `tests/rlm/test_evidence_gate_forge.py`.**

### Task 5 — Typed `EvaluationReport` adapter + `ScorecardRow` (lead; core)
- **New:** `backend/evals/evaluation_report.py`. `ScorecardRow(BaseModel)` = `{dimension, status: Literal["pass","fail","unmeasured","excluded","display"], provenance: Literal["paper_reported","agent_measured","evaluator_computed"], gates: bool, evidence_refs: list[str], detail: str}`. `EvaluationReport(BaseModel)` composes (never forks) `RLMFinalReport` (`report.py:32`, fields 39-232) — copies `verdict` **read-only**, plus `scorecard: list[ScorecardRow]`, `composite: float|None` (Task 1), `provenance_bundle_sha256`, `autonomy`, `gpu_efficiency`. `from_run(project_dir)` reads on-disk artifacts. `gate_caps() -> str|None` = most-severe downward cap from `gates=True` non-pass rows via `reproducibility_verdict._ROLLUP_ORDER` (display rows ignored) — a **pre-`decide()` input**, never a write. `to_markdown()`.
- Analogues to compose (do NOT fork): `agents/schemas.py` `RubricVerification`+`from_areas`:651/668, `MetricDelta`:705 (`relative_error_vs_paper`:718, Cohen's d `effect_size`:721, `ci95_half_width`:722), `ExperimentArtifacts`:475, `MetricSpec`:60, `PaperClaimMap`:88.

### Task 6 — Populate 11 dimensions + wire at finalize (lead; core — THE scorecard)
- **New:** `backend/evals/scorecard.py::build_scorecard(project_dir) -> list[ScorecardRow]` + `write_evaluation_report(project_dir) -> Path|None`. **Wire** in `report.py::write_final_report_rlm` after the authority stamp, flag-gated `OPENRESEARCH_EVAL_SCORECARD`; fold any downward gate cap into the existing `claim_gate_cap` **before** `decide()` (NOT a post-decide writer). Off ⇒ no sidecar, no cap (prove via the tripwire + a new `test_scorecard_offstate.py`).
- **Dimension → source (spec §6.1 table):** GATE rows — numerical (`result_fidelity`), execution (`_has_experiment_evidence` + ok-receipt), environment (`env_health.jsonl`), dataset (`_detect_data_unavailable_leaves`), tables/figures (`fig_*.json` sidecars, GATE-lite). DISPLAY rows — autonomy (`human_intervention.autonomy_metric`), efficiency (`gpu_ledger.aggregate_gpu_cost`), paper-understanding (`two_axis_report.fidelity_score_from_rubric`), DAG-planning (**post-hoc S0 graph** from `experiment_runs.jsonl` until Track G S1 lands), debugging (`failure_capsules.jsonl` + `FailureAttribution`), scientific-analysis (`HypothesisScore`/`IntegrityReport`).

### Task 7 — Skill-as-reference + leniency guard (lead)
- **New:** `backend/evals/reference_from_skills.py::compose_reference(project_dir)` reads `rlm_state/active_skills.json` → structure only (expected metric families, baselines, eval protocol, dataset expectations), provenance `evaluator_computed`. **Load-bearing test:** a skill-supplied "expected pass" can NEVER flip a `result_fidelity` per-claim `status` (pass keys solely on measured artifacts).

### Task 8 — Acceptance + §8 battery (lead)
- **New:** `tests/acceptance/test_eval_scorecard_acceptance.py`. Frozen **Adam** `runs/prj_adam_local_1` → coherent scorecard, headline verdict stays `inconclusive` (never lifted by a display row/composite). **UCPO** `runs/prj_ucpo_optA_1` → coherent scorecard + post-hoc observed-DAG rows. Copy fixtures to `tmp_path` — NEVER mutate `runs/`. §8 invariants: missing metrics → `unmeasured`; composite off-verdict; no scorecard field alters `meets_target`/`AttemptAssessment`/`campaign_policy.decide()`; skill can't flip a pass; fabricated artifact → severe (evidence gate); serialization round-trip.

## 6. Track G — DAG (after Track E; flag-gated). Recon map

Neither `networkx` nor `graphlib` is a dependency — use stdlib `graphlib.TopologicalSorter` (importable, zero new deps). No declarative task-DAG exists today.

- **S1a — node-only observed log:** write `runs/<id>/dag_nodes.jsonl` during the run, one node per experiment/stage from `experiment_runs.jsonl` (`_persist_experiment_result` `primitives.py:4880`, row keys: `experiment_run_id`/`artifact_dir`/`wall_time_s`/`env_id`/`model_id`/`failure_class`/`outcome`/`metrics_sha256`; **absent:** gpu_hours/estimated_cost/actual_cost/producer→consumer edges). Node `{id,name,category,status,attempt_count,wall_s,gpu_hours,estimated_cost,actual_cost,failure_type,artifacts[]}`. Reuse durable append `reproduction_campaign.py:205 CampaignLedger.append_row` (atomic + torn-tail + fsync) — the experiment writer at `primitives.py:4958` is weaker (no fsync). Free-form execution unchanged.
- **S1b — typed edges:** edge-emitting hooks at primitive boundaries (consumer records the producer `metrics_sha256`/`artifact_dir` it read).
- **S2 — scheduled backbone** behind `OPENRESEARCH_DAG_BACKBONE` (default-OFF): evolve `lifecycle_driver.py` (`run_lifecycle_primary`:499; linear chain `drive_lifecycle_chain`:183, stages understand→detect_env→plan→implement→run→verify at :338-487; stage constants :172-176) into a scheduler over the S1 graph — parallel frontier, **failed-node non-propagation** (a failed node never marks downstream complete — same bug-class as phantom-0.0), per-node resume. Routes THROUGH `verdict_authority.decide()` + campaign DECIDE, never around. Default flip needs ≥3 paired A/B.
- **Node/edge storage pattern (reuse):** `backend/services/context/graph/service.py:40 KnowledgeGraphService` — tables `knowledge_graph_{nodes,edges}` (:52-87), `upsert_node`/`upsert_edge` (:98/:118), deterministic ids `graph_node_id`/`graph_edge_id` (:14/:27). Models `backend/services/context/graph/model.py` `GraphNode`:31/`GraphEdge`:44. NB: that's a shared cross-project SQLite DB; `dag_nodes.jsonl` is per-run — reuse the *shape*, keep per-run jsonl.
- **Failure inputs:** `failure_capsule.py:140 build_capsule` (schema :175-188), `failure_attribution.py:87 FailureAttribution`, `failure_classifier.py:31 FAILURE_CLASSES` (33 classes).
- **Testing (spec §8):** cycle detection · missing-dep · valid topological order · partial completion · **failed-node non-propagation** · duplicated nodes.

## 7. Delegation guardrails (guarded hybrid)

**Lead implements:** Tasks 4, 5, 6, 7, 8 (verdict-adjacent/core), Track G S1b/S2, Task 3b (primitives.py). **Delegatable to Sonnet (fenced, new-file-heavy):** none strictly required now; if you do, EVERY subagent prompt MUST: forbid all git state commands (`commit`/`add`/`amend`/`checkout`/`stash`/`reset`/`rebase`); restrict writes to an explicit allowlist; "never edit/delete an existing test — STOP and report if one blocks you"; "if a production file outside your list needs changing, STOP and report." **After it returns, `git status` + `git log`/`reflog` to verify footprint before trusting.** Prefer **in-tree** over `isolation: worktree` (stale-base gotcha, §4). See memory `feedback_subagent_git_guardrails`, `feedback_delegation`, `feedback_opus_plans_reviews`.

## 8. Verify / run

```bash
cd /home/abheekp/openresearch
.venv/bin/python -m pytest tests/agents/rlm/ tests/test_eval_composite.py tests/test_evaluation_report.py -n auto   # real CI mode
.venv/bin/python -m pytest tests/ -n auto                     # full suite before any commit
uvx ruff@0.15.16 check .                                       # lint
# acceptance fixtures already on disk: runs/prj_adam_local_1 (inconclusive), runs/prj_ucpo_optA_1
```

Off-state proof for every flag: assert byte-identical output with the flag unset (the OFF half of the pair).

## 9. The bigger goal (sequence after Track E/G)

Track E/G make the eval **honest + structured + patent-ready**. The lever for *"reproduce the vast majority of ML papers"* is **WS4 (CPU-class durable lane** — most ML is CPU-class; remote cells are GPU-only today, floor `gpu_count≥1`; the only proven turnkey run, Adam, is CPU-local**)** + **WS5 (repo-first grounding default-on** where a paper links code**)**. Both are scoped in the Track-A/D memory (`project_tracks_a_d_implementation`) and the 2026-07-09 design specs (untracked on-disk: `2026-07-09-eval-integrity-track-a-design.md`, `-cloud-reliability-track-d-design.md`). **WS3 durable cloud-native controller** design (`4f0042e8`) is drafted but needs a lead+operator review before coding. Recommended order: **E 3b–8 → G (S1a→S1b→S2) → WS4 → WS5 → WS3**.

## 10. Memory pointers (auto-loaded next session)

`project_tracks_a_d_implementation` (scope + this turn's update), `feedback_subagent_git_guardrails` (the incident + worktree gotcha), `feedback_delegation`/`feedback_opus_plans_reviews` (Opus designs+reviews, Sonnet executes), `project_adam_cpu_reproduction` (WS4 context), `project_github_repo_first_reproduction` (WS5 context).
