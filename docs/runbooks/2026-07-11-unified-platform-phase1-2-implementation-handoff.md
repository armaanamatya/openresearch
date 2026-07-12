# Unified Reproduction Platform — Phase 1 + 2 Implementation Handoff

- **Date:** 2026-07-11
- **Branch:** `feat/gke-gpu-path-reproduction-reliability`
- **Spec:** `docs/superpowers/specs/2026-07-11-unified-reproduction-platform-parallel-build.md`
- **Scope delivered:** the complete, fully-verified **Track E eval scorecard** (wired
  end-to-end, producers → finalize sidecar) + **WS3 Phase-1 durability primitives** +
  **WS5 repo-first grounding module**. Everything flag-gated **default-OFF**, byte-identical off,
  and **verdict-inert** (the north-star single-writer tripwire passes with every flag on).

## 1. What landed (commits, newest first)

| Commit | What |
|---|---|
| `b41c7627` | Track E Task 8 acceptance — scorecard coherent + verdict-preserving on frozen Adam/UCPO runs |
| `0fbaf0bb` | **Phase 2** — wire Track E into `run_experiment` persist + `report.py` finalize; add G-S1a observed-DAG recorder |
| `87d1b7a9` | WS5 repo-first grounding module (structure-only) |
| `1aba3db9` | WS3 durability primitives (`job_fence` + `run_controller`) |
| `c6b5e54b` | **Phase 1** Track E scorecard modules (EvaluationReport + scorecard + ok-receipt + skill-ref) |
| `8aeeea55` | cited Track A/D + eval-framework design specs |

**Delegation model used:** Phase 1's 6 new disjoint modules were built by a parallel agent
fan-out (`ultracode` workflow), each with hard git + file-allowlist guardrails; every diff was
lead-reviewed before commit. Phase 2's two crown-jewel files (`report.py`, `primitives.py`) were
lead-implemented directly (verdict/evidence-critical — not delegated).

## 2. New modules + flags (all default-OFF, byte-identical off)

| Flag | Module | Effect |
|---|---|---|
| `OPENRESEARCH_OK_RECEIPT` | `agents/rlm/ok_receipt.py` | forge-resistant out-of-process success receipt; lifts the partial re-grade ceiling in `report.run_experiment_success_count` when the in-memory ledger is absent |
| `OPENRESEARCH_EVAL_SCORECARD` | `evals/scorecard.py` + `evals/evaluation_report.py` | writes `evaluation_report.{json,md}` sidecar at finalize (11 dimensions; verdict copied read-only) |
| `OPENRESEARCH_GPU_LEDGER` | `agents/rlm/gpu_ledger.py` (3a, prior) + wiring | per-experiment `gpu_ledger.jsonl` + `start_ts/end_ts/gpu_plan/retry_id` row fields |
| `OPENRESEARCH_DAG_BACKBONE` | `agents/rlm/dag_nodes.py` (new) | G-S1a observed-DAG `dag_nodes.jsonl` recorder |
| `OPENRESEARCH_DURABLE_CONTROLLER` | `agents/rlm/run_controller.py` + `services/runtime/job_fence.py` | WS3 controller/fence **primitives only** — cluster wiring is deferred (§4) |
| `OPENRESEARCH_USE_AUTHOR_REPO` (existing) | `services/ingestion/repo_first_grounding.py` | structure-only author-repo grounding (module built; run.py wiring deferred) |

`reference_from_skills.py` is a pure read-only helper (no flag; unwired, structurally verdict-inert).

## 3. Verification (what was actually run)

- New/wiring/acceptance tests: **all green** (job_fence 31, ok_receipt 4, evaluation_report 5,
  reference 11, repo_first 14, scorecard 36, report wiring 8, dag_nodes 4, persist telemetry 11,
  Task-8 acceptance 5).
- **Verdict tripwire** (`test_single_verdict_authority_guard.py`) + forge + off-state: green,
  **including with `OPENRESEARCH_EVAL_SCORECARD` on and the authority active** — the scorecard
  sidecar cannot move the verdict surface.
- **Primitive registry still 19** (`test_registry.py`) — `scorecard.py`'s import of `report.py`
  and the `primitives.py` edits did not perturb the count.
- Broad regression `tests/agents/rlm/ tests/rlm/ …`: **5962 passed, 3 failed**. The 3 are
  **pre-existing + env-dependent** (`test_accelerator.py::TestResolveAuto` — they fail identically
  with the Phase-2 diff stashed, because the operator's `.env` has real `AZURE_FOUNDRY_API_KEY` so
  `resolve_accelerator("auto")` returns a live endpoint where the test asserts `None`).
- `ruff@0.15.16` clean on every touched file; `docs/reference/flags.md` regenerated (`--check` passes).

**Repro:**
```bash
.venv/bin/python -m pytest tests/agents/rlm/test_report_eval_wiring.py \
  tests/agents/rlm/test_persist_telemetry_wiring.py tests/agents/rlm/test_dag_nodes.py \
  tests/acceptance/test_eval_scorecard_acceptance.py \
  tests/agents/rlm/test_single_verdict_authority_guard.py tests/rlm/test_registry.py -q
```

## 4. Deferred — Phase 3 (exact anchors, so the next session lands them precisely)

These are **flag-gated cluster changes** whose only real validation is the **operator-gated GPU
durability drill** — that's why they were not shipped blind this session. All anchors verified via
read-only recon 2026-07-11 (match by symbol; lines drift ±10).

### 4.1 WS3 fencing into the cell runner (`agents/rlm/k8s_job_cell_runner.py`)
- `_job_name(cell_id, run_id="")` **:499** — when `run_controller.durable_controller_enabled()`,
  route through `job_fence.fenced_job_name(run_id, cell_id, gen)` (`gen` = the `BlobLease` token
  generation). Off-flag keeps the legacy string.
- `run_matrix` `output_blob_prefix = f"runs/{run_id}/{_BLOB_CELLS_PREFIX}"` **:1643** → gen-scoped
  `job_fence.fenced_blob_prefix(run_id, gen, cell_id=...)`.
- the catch-all `except Exception` at **:1323-1332** → add an `ApiException.status==409` branch that
  reads the Job status and adopts it (`job_fence.adopt_or_submit`) instead of `STATUS_ERROR`.
- `_watch_job` `time.monotonic()+active_deadline_seconds` **:879** → a persisted absolute-epoch
  deadline (re-read on adopt so a resumed run inherits *remaining* budget).

### 4.2 WS3 + WS4 in the job backend (`services/runtime/k8s_job_backend.py`)
- `exec(...)` **:740** stages no code — add the **WS2 guard**: fail loud `monolithic_exec_unstaged`
  for gcp/gke unstaged (the working path is `gke_cell_synth`).
- **WS4:** `gpu_count = max(1, int(gpu_count or 1))` **:205** + the unconditional `nvidia.com/gpu`
  limits/requests **:226-227** — allow `gpu_count=0` (drop the GPU resource block) for the CPU-class
  lane. Also `GcsStore.upload_bytes` **:363** discards the CAS generation — extend it to forward
  `if_generation_match` + expose `read_bytes_with_generation` if any writer needs CAS *through*
  `GcsStore` (`BlobLease` sidesteps this today by calling `gcs_blob` directly).
- deadline `time.monotonic()+timeout` **:960** → persisted epoch (as §4.1).

### 4.3 WS3 controller / SaaS path
- `services/events/live_runs.py` `Popen python -c cmd_reproduce` **:1019** → a `sandbox=gcp` branch
  that submits the controller (`run_controller.build_controller_command` → the helm Deployment).
- `infra/gcp/helm`: `orchestrator-deployment.yaml` exists, `values.yaml` `orchestrator.enabled:false`
  **:192** — wire it to `campaign --project-id <stable> --resume`. The reaper RBAC is **already
  granted** (`role.yaml:19-21` grants `list`+`delete` on `batch/jobs`); implement
  `blob_lease.reap_older_generations` (today a deliberate `NotImplementedError` stub) to use it.
- GCS-mirror the campaign ledger; promote GCS `status.json` (`_try_reconcile_status`) ahead of local
  `cell_manifest.json` (`cell_scheduler.should_skip_cell:159`) as the primary resume-skip source.

### 4.4 WS1 H1 — demo_status terminal-only (**verdict-adjacent; design before coding**)
`run.py::_write_demo_status` **:1033**; the two non-terminal `status="running"` writes at **:3310**
(run-start snapshot) + **:3399** (root-validation stamp), plus the `_cost_summary_loop` re-writer
**:2633**. The stale-read is subtle and adjacent to the verdict surface — do not delegate; decide the
exact semantics first (the run-start snapshot is legitimately non-terminal for the live UI).

### 4.5 WS2 default-flips — **operator-gated (money)**
No default was flipped this session (correct: the re-tiering makes `GKE_SYNTH_CELL`/`CELL_RESUME_AUTO`/
`IMPL_ABANDON_GUARD` **not** Tier-0). Each Tier-1 flip needs ≥3 paired SDAR A/B runs + the grader-σ
gate, with `OPENRESEARCH_REQUIRE_STAMPED_AB` CI-enforced. `PREFLIGHT_UNION_SCOPE` needs the
bidirectional false-block audit.

### 4.6 WS-Ext — extractor measurement-unblock (**design-only; the Phase-1 sub-spec agent stalled**)
Intent (from the unified spec + the confirmed `result_fidelity` taxonomy — per-claim status is
**pass/fail/unmeasured**, `ambiguous` is a boolean, RELATIVE claims need a finite `baseline_value`):
in `agents/rlm/repro_spec_extractor.py`, (a) emit an **explicit** per-claim `kind` (never inferred —
`result_fidelity` must dispatch only on an explicit kind; a false pass/contradicted is the worst
error), (b) thread `baseline_value`/`proposed_value` through `_normalize_claim_from_llm` (already
LLM-prompted, currently discarded) so RELATIVE claims measure, (c) relax the over-conservative A6
cross-check. **Needs a recon pass of `repro_spec_extractor.py` for exact anchors before coding**;
acceptance = a real non-ambiguous claim measures, Adam stays `inconclusive`, no false `contradicted`.

### 4.7 Track G S1b/S2
`dag_nodes.py` records S1 nodes (edges empty today). S1b = typed edges; S2 = the opt-in scheduler in
`lifecycle_driver.py` behind `OPENRESEARCH_DAG_BACKBONE`. Orchestration only — never a verdict signal.

## 5. Operator-gated validation still owed (money)
- **Durability drill (WS3 headline):** launch a `sandbox=gcp` run, kill the CONTROLLER pod
  mid-training → a successor adopts the in-flight Job by name, resumes from the GCS ledger, finalizes
  a real metric. Plus the split-brain + reaper drills.
- **WS2 A/B** per Tier-1 guard before any default flip.

## 6. North-star reminder (binds all future work here)
`verdict_authority.decide()` is the SOLE verdict writer. Every new signal is a **pre-`decide()`
downward-only gate** or **display-only**; nothing new may raise a verdict. The eval scorecard stays
**decoupled** (report/rank-only) — that is why Phase 2 deliberately did **not** fold the scorecard's
`gate_caps()` into `decide()`. Run `test_single_verdict_authority_guard.py` after any `report.py`
change.
