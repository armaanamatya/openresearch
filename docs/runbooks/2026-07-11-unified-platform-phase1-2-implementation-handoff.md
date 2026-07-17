# Unified Reproduction Platform — Handoff + Phase-3 Fan-Out Playbook

- **Date:** 2026-07-11
- **Branch:** `feat/gke-gpu-path-reproduction-reliability` (55 ahead of `main`, **unpushed, no PR**)
- **Master spec:** `docs/superpowers/specs/2026-07-11-unified-reproduction-platform-parallel-build.md`
- **Status:** Track E eval scorecard **wired end-to-end + validated on real data**; WS3 Phase-1
  durability primitives + WS5 grounding module built. All flag-gated **default-OFF**, byte-identical
  off, **verdict-inert**. Remaining WS3/WS4 cluster wiring + WS2 flips + WS1-H1 + WS-Ext are deferred
  (§4) — each for a real reason (operator-gated GPU, or a verdict-adjacent design call).

---

## 0. START HERE — how to run the next session (read this section first)

**You are picking up a large multi-workstream build with a FRESH context. This doc is the single
source of truth. Read it fully, then open the cited specs (§7) only for the detail you need.**

**Operator directive:** *fan out sub-agents, be efficient, work in parallel — WITHOUT sacrificing
quality.* Run it as a **phased `ultracode` Workflow**, not a free-for-all. The session's lead model
(Opus) owns design, **reviews EVERY diff**, and **owns ALL commits**; sub-agents write code against a
tight, anchored spec and never touch git.

**The clobber-safe rule (hard lesson — the 2026-07-10 incident: two agents in one working tree
reverted each other's uncommitted work twice):**
- Parallel agents MUST own **disjoint file sets**. A shared "hot" file gets **exactly ONE owner per
  phase**.
- *Axis A* (brand-new / pure modules): unlimited concurrency — disjoint by construction.
- *Axis B* (hot-file integration): one designated owner-agent per file per phase; it lands all of that
  phase's edits to that file, the lead reviews + commits, freeing it for the next phase.
- **Do NOT use `isolation: worktree`** for hot-file edits — the worktree base is ~181 commits stale, so
  a delegated edit lands against wrong code. Do hot-file work **in-tree with single-owner
  serialization** (unless you first verify the worktree base == feature HEAD).

**Subagent guardrails — put these VERBATIM in every implementer prompt:**
- FORBIDDEN: every git state command (`commit`/`add`/`amend`/`checkout`/`switch`/`stash`/`reset`/
  `rebase`/`merge`/`rm`/`clean`). Read-only `git status`/`diff`/`log` on their own files only.
- Write ONLY the files in an explicit **allowlist**. **NEVER edit or delete an existing test** — STOP
  and report. If a file outside the allowlist needs changing — STOP and report (`out_of_scope_needed`).
- New capability = `feature_flags.env_truthy("FLAG")`, **default-OFF, byte-identical off**, shipped with
  a hermetic OFF+ON test pair (TDD: failing test first).
- Return a **structured summary** (`files_created`, `tests_passed`, `off_state_verified`,
  `out_of_scope_needed`). The lead **verifies the git footprint** (`git status`/`git log`) and re-runs
  the tests BEFORE trusting the summary, then commits.
- This session's Phase-1 fan-out (6 agents) ran clean under exactly these guardrails — no clobber, no
  rogue commits. Reuse that shape.

**Do NOT delegate / do NOT parallelize:**
- The **verdict-adjacent files** (`report.py`, `run.py` H1) — lead-implement them, review against the
  north-star (§5).
- The **operator-gated GPU work** (durability drill, A/B flips) — an agent literally cannot run it
  (real A100 money); it's the operator's step.

---

## 1. What already landed (DONE — do NOT redo)

| Commit | What |
|---|---|
| `b41c7627` | Track E Task 8 acceptance — scorecard coherent + verdict-preserving on frozen Adam/UCPO runs |
| `0fbaf0bb` | **Phase 2** — wire Track E into `run_experiment` persist + `report.py` finalize; add G-S1a observed-DAG recorder |
| `87d1b7a9` | WS5 repo-first grounding module (structure-only) |
| `1aba3db9` | WS3 durability primitives (`job_fence` + `run_controller`) |
| `c6b5e54b` | **Phase 1** Track E scorecard modules (EvaluationReport + scorecard + ok-receipt + skill-ref) |
| `8aeeea55` | cited Track A/D + eval-framework design specs |

**The whole Track E eval scorecard is wired end-to-end + validated on real data:** producers in
`primitives.py` (`_persist_experiment_result` chokepoint) → sidecar at `report.py` finalize →
acceptance on the real `runs/prj_adam_local_1` (reproduced), `prj_ucpo_optA_2` (partial),
`prj_ucpo_optA_1` (failed). Plus WS3 Phase-1 primitives + WS5 grounding module.

## 2. New modules + flags (all default-OFF, byte-identical off)

| Flag | Module | Effect |
|---|---|---|
| `OPENRESEARCH_OK_RECEIPT` | `agents/rlm/ok_receipt.py` | forge-resistant out-of-process success receipt; lifts the partial re-grade ceiling in `report.run_experiment_success_count` when the in-memory ledger is absent |
| `OPENRESEARCH_EVAL_SCORECARD` | `evals/scorecard.py` + `evals/evaluation_report.py` | writes `evaluation_report.{json,md}` sidecar at finalize (11 dimensions; verdict copied read-only) |
| `OPENRESEARCH_GPU_LEDGER` | `agents/rlm/gpu_ledger.py` (prior) + wiring | per-experiment `gpu_ledger.jsonl` + `start_ts/end_ts/gpu_plan/retry_id` row fields |
| `OPENRESEARCH_DAG_BACKBONE` | `agents/rlm/dag_nodes.py` (new) | G-S1a observed-DAG `dag_nodes.jsonl` recorder |
| `OPENRESEARCH_DURABLE_CONTROLLER` | `agents/rlm/run_controller.py` + `services/runtime/job_fence.py` | WS3 controller/fence **primitives only** — cluster wiring deferred (§4) |
| `OPENRESEARCH_USE_AUTHOR_REPO` (existing) | `services/ingestion/repo_first_grounding.py` | structure-only author-repo grounding (module built; run.py wiring deferred) |

`evals/reference_from_skills.py` is a pure read-only helper (no flag; unwired; structurally verdict-inert).

## 3. Do NOT rebuild — known surprises (these save wasted agent runs)

- **E-1 composite ALREADY done** — `CompositeWeights` + `DEFAULT_COMPOSITE_WEIGHTS` + the
  deterministic-dominated `ReproductionScore.composite_score` are in `evals/schemas.py`. Do not touch it.
- **`gcs_blob` CAS + `blob_lease.py` ALREADY landed** (`0f691a30`): `upload_bytes(if_generation_match=…)`
  returns the generation, `read_bytes_with_generation` exists, `BlobLease.{acquire,renew,is_current}`
  exist. **BUT `blob_lease.reap_older_generations` is a deliberate `NotImplementedError` stub** — the
  reaper is Phase-3 work (the RBAC to do it is already granted).
- `gpu_ledger.py` + `human_intervention.py` already exist (aggregators `aggregate_gpu_cost` /
  `autonomy_metric`).
- **`result_fidelity` per-claim `status` is 3-valued** — `pass` / `fail` / `unmeasured`. `ambiguous` is a
  boolean passthrough, `contradicted` is the aggregate `any(status=="fail")`. A RELATIVE claim needs a
  finite `baseline_value` or it stays `unmeasured`. (This is the crux of the WS-Ext work.)
- **`scorecard.py` top-imports `report._has_experiment_evidence`** → any `report.py` → scorecard wiring
  MUST be a **lazy import** (inside the function) to avoid a cycle. (Phase 2 already does this correctly.)
- `run_controller.classify_controller_exit` maps campaign exit-3 (money-halt) → `"crash"`; Phase 3 should
  special-case exit-3 as terminal, not a respawn.
- **`_persist_experiment_result` has 9 call sites** — hooking that one function covers every
  `run_experiment` path (that's how the Phase-2 telemetry covers all paths).
- Flags added this branch (all default-OFF, in `docs/reference/flags.md`, `gen_flag_registry --check`
  passes): `OK_RECEIPT`, `EVAL_SCORECARD`, `GPU_LEDGER`, `DAG_BACKBONE`, `DURABLE_CONTROLLER`.

## 4. Environment + verification

- venv `.venv` (Python 3.11+; 3.12 in CI). Tests are **socket-hermetic** (`--disable-socket`) — a new
  test must never dial out; inject fakes.
- Full suite: `.venv/bin/python -m pytest tests/ -n auto`. Lint: `uvx ruff@0.15.16 check .`. Flag
  registry: `.venv/bin/python scripts/gen_flag_registry.py` then `--check`.
- **Between phases** (and before trusting any agent): `-n auto` + `ruff` + confirm off-state
  byte-identical for every touched flag + **run `tests/agents/rlm/test_single_verdict_authority_guard.py`
  after ANY `report.py`/`run.py`/`demo_status` change** + confirm `tests/rlm/test_registry.py` still says
  19 primitives.
- **Establish the pre-existing failure baseline BEFORE editing** so you don't attribute it to your change.
  Known-pre-existing (NOT regressions): 3 × `tests/rlm/test_accelerator.py::TestResolveAuto` (the operator
  `.env` has real `AZURE_FOUNDRY_API_KEY`, so `resolve_accelerator("auto")` returns a live endpoint where
  the test asserts `None`) + ~18 other credential/keychain/OCR/oauth/repo-hygiene probes. This session's
  broad regression was **5962 passed, only those 3 env-failures** (confirmed identical with the diff stashed).
- Phase-1+2 focused repro:
  ```bash
  .venv/bin/python -m pytest \
    tests/services/runtime/test_job_fence.py tests/agents/rlm/test_run_controller.py \
    tests/agents/rlm/test_ok_receipt.py tests/test_evaluation_report.py \
    tests/test_reference_from_skills.py tests/test_scorecard_dimensions.py \
    tests/agents/rlm/test_scorecard_offstate.py tests/services/ingestion/test_repo_first_grounding.py \
    tests/agents/rlm/test_dag_nodes.py tests/agents/rlm/test_persist_telemetry_wiring.py \
    tests/agents/rlm/test_report_eval_wiring.py tests/acceptance/test_eval_scorecard_acceptance.py \
    tests/agents/rlm/test_single_verdict_authority_guard.py tests/rlm/test_registry.py -q   # 186 passed
  ```

## 5. Phase-3 fan-out plan (the actionable parallel work)

Everything below is flag-gated on `OPENRESEARCH_DURABLE_CONTROLLER` (WS3/WS4) and byte-identical off.
All anchors verified read-only 2026-07-11 (match by symbol; lines drift ±10). **The durability drill is
the ONLY real validation of the WS3/WS4 group — it is operator-gated, so this code ships flag-gated +
off-state-proven and stays drill-pending until the operator runs it.**

### Wave B — hot-file integration (Axis B, ONE owner per file; the four files are disjoint → run the 4 owners CONCURRENTLY, max 4)

**owner(`agents/rlm/k8s_job_cell_runner.py`) — WS3 fence + WS4 cell:**
- `_job_name(cell_id, run_id="")` **:499** → when `run_controller.durable_controller_enabled()`, route
  through `job_fence.fenced_job_name(run_id, cell_id, gen)` (`gen` = `BlobLease` token generation). Off ⇒ legacy string.
- `run_matrix` `output_blob_prefix = f"runs/{run_id}/{_BLOB_CELLS_PREFIX}"` **:1643** → `job_fence.fenced_blob_prefix(run_id, gen, cell_id=…)`.
- catch-all `except Exception` **:1323-1332** → add `ApiException.status==409` branch: read Job status → `job_fence.adopt_or_submit(...)` → attach to `_watch_job` instead of `STATUS_ERROR`.
- `_watch_job` `time.monotonic()+active_deadline_seconds` **:879** → persisted absolute-epoch deadline (re-read on adopt ⇒ inherit *remaining* budget).
- WS4: cell `gpu_count=0` path.

**owner(`services/runtime/k8s_job_backend.py`) — WS2 guard + WS4:**
- `exec(...)` **:740** stages no code → add WS2 guard: fail loud `monolithic_exec_unstaged` for gcp/gke unstaged.
- WS4: `gpu_count = max(1, int(gpu_count or 1))` **:205** → allow `0` + drop the unconditional `nvidia.com/gpu` limits/requests **:226-227** for the CPU-class lane.
- deadline `time.monotonic()+timeout` **:960** → persisted epoch. (Optional) `GcsStore.upload_bytes` **:363** — forward `if_generation_match` + add `read_bytes_with_generation` only if a writer needs CAS *through* `GcsStore` (`BlobLease` sidesteps it by calling `gcs_blob` directly).

**owner(`services/events/live_runs.py`) — WS3 SaaS path:**
- `Popen python -c cmd_reproduce` **:1019** → a `sandbox=gcp` branch that submits the controller (`run_controller.build_controller_command` → the helm Deployment) instead of the local Popen. Off ⇒ byte-identical Popen path.

**owner(`infra/gcp/helm` + `blob_lease.py` reaper) — WS3 controller + reaper:**
- wire `orchestrator-deployment.yaml` (`values.yaml orchestrator.enabled:false` **:192**) to `campaign --project-id <stable> --resume`.
- implement `blob_lease.reap_older_generations` (stub today) using the already-granted `role.yaml:19-21` list+delete on `batch/jobs`.
- GCS-mirror the campaign ledger; promote GCS `status.json` (`_try_reconcile_status`) ahead of local `cell_manifest.json` (`cell_scheduler.should_skip_cell:159`) as the primary resume-skip source.

*(Optional Wave A first: a pure `deadline.py` helper for the persisted-epoch logic, unit-tested, so the
two owners that need it consume a tested primitive — Axis A, parallel with everything.)*

### Lead-only (do NOT delegate)
- **owner(`run.py`) — WS1 H1** (demo_status terminal-only): the two non-terminal `status="running"` writes
  at **:3310** (run-start snapshot) + **:3399** (root-validation stamp), plus `_cost_summary_loop` **:2633**.
  Verdict-adjacent — **DESIGN the exact semantics first** (the run-start snapshot is legitimately
  non-terminal for the live UI). Small careful change, not a fan-out target.
- **WS-Ext** (`agents/rlm/repro_spec_extractor.py`): emit an EXPLICIT per-claim `kind` (never inferred —
  `result_fidelity` must dispatch only on an explicit kind; a false pass/contradicted is the worst error);
  thread `baseline_value`/`proposed_value` through `_normalize_claim_from_llm` (LLM already prompts for
  them, currently discarded) so RELATIVE claims measure; relax the over-conservative A6 cross-check.
  **Recon `repro_spec_extractor.py` for exact anchors first** (the design sub-spec agent stalled — it's
  not written). Acceptance: a real non-ambiguous claim measures; Adam stays `inconclusive`; no false `contradicted`.

### Operator-gated (money — an agent cannot run these)
- **WS3 durability drill (headline):** launch a `sandbox=gcp` run, **kill the controller pod** mid-training
  → a successor adopts the in-flight Job by name, resumes from the GCS ledger, finalizes a real metric.
  Plus the split-brain drill (stale token refused) + reaper drill (no orphaned A100 bills).
- **WS2 default-flips:** ≥3 paired SDAR A/B runs + the grader-σ gate per Tier-1 guard; CI-enforce
  `OPENRESEARCH_REQUIRE_STAMPED_AB`. **No default was flipped this session** (correct — the re-tiering makes
  `GKE_SYNTH_CELL`/`CELL_RESUME_AUTO`/`IMPL_ABANDON_GUARD` NOT Tier-0; `PREFLIGHT_UNION_SCOPE` needs the
  bidirectional false-block audit).

### Track G S1b/S2
`dag_nodes.py` records S1 nodes (edges empty today). S1b = typed edges; S2 = the opt-in scheduler in
`lifecycle_driver.py` behind `OPENRESEARCH_DAG_BACKBONE`. Orchestration/observability only — never a verdict signal.

**Dependency edges that MUST hold:** WS4 ⟶ after the WS3 fence+controller land; WS2 flips ⟶ after the A/B;
Track G S2 ⟶ after S1; WS-Ext code ⟶ after its recon.

## 6. North-star (binds EVERY agent — non-negotiable)

`verdict_authority.decide()` is the **sole** verdict writer. Every new signal is a pre-`decide()`
**downward-only** gate or **display-only**; nothing new may raise a verdict — with one principled
exception: forge-resistant deterministic **evidence** (like the ok-receipt) may inform `decide()`'s
*inputs* (never a grade / skill / composite, and never a post-`decide` write). The eval scorecard stays
**decoupled / report-only** — **do NOT fold `EvaluationReport.gate_caps()` into `decide()`** (that would
couple the eval to the verdict and violate the invariant; Phase 2 deliberately did not). After ANY
`report.py`/`run.py`/`demo_status` change, `assert_verdict_surface_unchanged` must pass
(`test_single_verdict_authority_guard.py`, `-n auto`).

## 7. Source specs / plans (detail refs — read on demand)

- Master orchestration + clobber-safe model: `docs/superpowers/specs/2026-07-11-unified-reproduction-platform-parallel-build.md`
- WS3 durability design: `docs/superpowers/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md`
- WS2 cloud-reliability (Track D): `docs/superpowers/specs/2026-07-09-cloud-reliability-track-d-design.md`
- Eval-integrity (Track A): `docs/superpowers/specs/2026-07-09-eval-integrity-track-a-design.md`
- Track E impl plan (full task interfaces + provided tests): `docs/superpowers/plans/2026-07-10-track-e-eval-scorecard.md`
- Track E/G master design: `docs/superpowers/specs/2026-07-10-reproduction-eval-framework-design.md`
</content>
