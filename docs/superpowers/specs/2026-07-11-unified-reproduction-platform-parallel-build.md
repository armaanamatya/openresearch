# Unified Reproduction Platform — Integrated Spec + Clobber-Safe Parallel Build Plan

- **Date:** 2026-07-11
- **Status:** Master integration spec. Unifies every in-flight track into ONE source of truth + a parallel-execution plan a fleet of agents can run without clobbering each other.
- **Supersedes for coordination:** the two 2026-07-09 Track A/D specs, the WS3 durability design (`4f0042e8`), and the reproduction-eval-framework Track E/G handoff (`2c70fb48`) — those remain the *detailed* references; this doc is the *orchestration authority*.

## 0. Goal + operator directives (the fixed points)

Build a platform that **reproduces the vast majority of ML papers unattended, on cloud, with an honest score**, feeding deepinvent.ai's patent layer. The operator's standing directives:

- **100% cloud / VM — the laptop is never required.** A run survives the launcher disconnecting; the driver executes in-cluster (or on a controller VM). Production 24/7, SaaS-grade, scale-out.
- **Honest scoring.** The verdict keys on the deterministic evidence layer, **never** the LLM grade (north-star). Higher score = more *real* reproductions, not a lifted label.
- **GCP-first.** Azure is deferred (redundancy, not a customer constraint — §5). The durability layer is designed cloud-agnostically so Azure is a later adapter, not a rewrite.
- **WS3 durability is prioritized now** (operator override of the Track E/G handoff's "WS3-last" ordering) — but because WS3 (runtime/infra files) and Track E (evals files) are largely file-disjoint, they run **in parallel**, so this is not an either/or.

**North-star invariant (unbreakable):** `verdict_authority.decide()` is the SOLE verdict writer. Every new signal is either a **pre-`decide()` downward-only gate** or **display-only**. Nothing new may raise a verdict.

## 1. Unified workstream map + status

| WS | What | Status | Primary file cluster (ownership) |
|---|---|---|---|
| **WS1** | Eval integrity — grade-free single-writer verdict (sever) | ✅ **DONE** (committed spine `b7d79c93`→`15880345`; sever `4a1ab068`; guard `c1f384c4`) | `metric_binding`/`result_fidelity`/`verdict_authority`/`report.py` sever |
| **WS1-fix** | H1 demo_status stale-read; "make it measure" is UPSTREAM (extractor) | ⏳ pending | H1→`run.py` (fold into WS3); measure→`repro_spec_extractor.py` (see WS-Ext) |
| **WS2** | Track D flag promotion (re-tiered) + guarded default-GCP path | ⏳ pending | flag defaults + `k8s_job_backend.exec` guard + `gke_cell_synth` default |
| **WS3** | Durable cloud-native orchestration (CAS lease built) | 🔨 **foundation done** (`blob_lease` `0f691a30`) | `gcs_blob`/`blob_lease`/`k8s_job_*`/controller/`live_runs`/`infra/gcp` |
| **WS4** | CPU-class durable lane (`gpu_count=0` under WS3) | ⏳ pending (depends on WS3) | `k8s_job_backend`/`k8s_job_cell_runner`/routing |
| **WS5** | Repo-first grounding default-on where a paper links code | ⏳ pending | ingestion/grounding path (`OPENRESEARCH_USE_AUTHOR_REPO`) |
| **WS-Ext** | Extractor: emit explicit test-`kind` + thread baseline values + relax over-conservative A6 (unblocks `result_fidelity` measurement) | ⏳ pending (recon-proven required) | `repro_spec_extractor.py` |
| **E** | Eval scorecard — typed `EvaluationReport`, 11 dimensions, deterministic-dominated composite | 🔨 **partial** (Tasks 1/2/3a committed; 3b–8 pending) | `backend/evals/*` + `gpu_ledger`/`human_intervention`/`ok_receipt` + `primitives.py`/`report.py` wiring |
| **G** | Observed-DAG → opt-in scheduler behind `OPENRESEARCH_DAG_BACKBONE` | ⏳ pending (after E) | `dag_nodes` writer (`primitives.py`) + `lifecycle_driver` scheduler |

## 2. The clobber-safe parallel-execution model (the core of this doc)

**Hard lesson (2026-07-10 incident):** two agents editing the same working tree reverted each other's uncommitted work twice. **Rule: parallel agents MUST own DISJOINT file sets. A shared "hot" file gets exactly ONE owner per phase.**

### 2.1 Hot files (contention points — never edited by two concurrent agents)
`backend/agents/rlm/primitives.py` · `backend/agents/rlm/report.py` · `backend/agents/rlm/run.py` · `backend/services/runtime/k8s_job_backend.py` · `backend/agents/rlm/k8s_job_cell_runner.py` · `backend/agents/rlm/lifecycle_driver.py` · `backend/agents/rlm/cell_scheduler.py` · `backend/agents/rlm/reproduction_campaign.py`.

### 2.2 Two-axis schedule
- **Axis A — new / pure modules (fully parallel, disjoint by construction).** Every new file is its own owner; unlimited concurrency.
- **Axis B — hot-file integration (serialized per file).** Each hot file has ONE designated owner-agent per phase; that agent lands all of the phase's edits to that file, then commits, freeing it for the next phase.

### 2.3 Phase plan + dependency DAG

```
Phase 0 (DONE): WS1 spine + sever · E T1/2/3a · WS3 CAS lease
        │
Phase 1 — PARALLEL new modules (Axis A, ~8 agents, all disjoint new files):
  WS3-fence   backend/services/runtime/job_fence.py   (gen→jobname/blobpath + adopt-by-name, PURE)
  WS3-ctrl    backend/agents/rlm/run_controller.py    (controller submit/reconnect wrapper; READS k8s_job_cell_runner)
  E-5         backend/evals/evaluation_report.py      (ScorecardRow + EvaluationReport, composes RLMFinalReport)
  E-4         backend/agents/rlm/ok_receipt.py        (out-of-process ok-receipt)
  E-7         backend/evals/reference_from_skills.py  (skill-as-reference + leniency guard)
  WS5         backend/services/ingestion/repo_first_grounding.py (new grounding module)
  WS-Ext-recon (design-only: extractor kind/baseline/A6 change — write the sub-spec, no code yet)
        │
Phase 2 — SERIALIZED hot-file integration (Axis B, ONE owner per file, run these file-groups in parallel since the FILES are disjoint across groups):
  owner(primitives.py):        E-3b gpu-ledger wiring → WS2 cell-synth default → WS4 gpu_count=0 routing → G-S1a dag-nodes  (all in primitives.py → ONE agent, sequential within)
  owner(report.py):            E-6 scorecard wiring (fold cap into claim_gate_cap, PRE-decide)
  owner(k8s_job_backend.py):   WS3 fence integration + WS2 monolithic-exec guard + WS4 gpu_count=0 Job path
  owner(k8s_job_cell_runner.py): WS3 fence integration (jobname/blobpath gen + adopt-on-409) + WS4 cell gpu_count=0
  owner(run.py):               WS1 H1 (terminal-only demo_status) + WS3 controller wiring
  owner(lifecycle_driver.py):  (Phase 3 — G-S2 scheduler; idle in Phase 2)
        │
Phase 3 — integration, defaults, drills (mostly serialized on the now-integrated hot files):
  WS3: controller Deployment (infra/gcp helm) → GCS-mirror ledger → reaper (wire blob_lease.reap) → durability drill
  WS2: default-flip the re-tiered flags AFTER the ≥3 paired-A/B + σ-gate (CI-enforce REQUIRE_STAMPED_AB)
  E-6/E-8: scorecard finalize + acceptance battery
  G-S1b/S2: typed edges + scheduler behind OPENRESEARCH_DAG_BACKBONE
  WS-Ext: implement the extractor change (its own review) → re-run WS1 acceptance (Adam still inconclusive, but a synthetic real claim now measures)
```

Dependency edges that MUST hold: WS4 ⟶ after WS3 fence+controller land; WS2 default-flip ⟶ after WS1 authority + the A/B σ-gate; E-6 ⟶ after E-4/E-5; G-S2 ⟶ after G-S1; WS-Ext code ⟶ after its sub-spec review.

### 2.4 Concurrency ceiling per phase
Phase 1 = up to ~8 concurrent (all disjoint new files). Phase 2 = up to **6 concurrent** — one per hot-file-group above (the groups are file-disjoint from each other). Phase 3 mostly serial (shared integrated files + real-cluster drills).

### 2.5 Worktree caveat
`isolation: worktree` currently branches from a **stale base (~181 commits behind HEAD)** — a delegated shared-file edit lands against wrong code. **Do hot-file work in-tree with single-owner serialization, NOT worktrees**, unless the worktree base is first verified == feature HEAD.

## 3. Per-workstream briefs (owner · files · interfaces · acceptance · detail-ref)

**WS2 — flag promotion + default GCP path** (detail: `2026-07-09-cloud-reliability-track-d-design.md`).
- Re-tiered (review correction): `HARDEXIT_CLEANUP` truly Tier-0; **`IMPL_ABANDON_GUARD`/`GKE_SYNTH_CELL`/`CELL_RESUME_AUTO` are NOT Tier-0** (they change shipped output) → audited/A-B. `PREFLIGHT_UNION_SCOPE` needs a population false-block audit. **CI-enforce `OPENRESEARCH_REQUIRE_STAMPED_AB`** (today unenforced).
- Guard `k8s_job_backend.exec` for gcp/gke unstaged → fail loud `monolithic_exec_unstaged`; default `GKE_SYNTH_CELL` ON (it routes uncellified projects to the working path).
- Acceptance: plain `sandbox=gcp` monolith stages code + runs, no hand-set env; off-state byte-identical; each Tier-1 flip has ≥3 paired A/B stamped.

**WS3 — durable cloud-native orchestration** (detail: `2026-07-10-durable-cloud-native-orchestration-ws3-design.md`; foundation `blob_lease` DONE).
- `job_fence.py` (Phase 1, pure): `fenced_job_name(run_id,cell_id,attempt|gen)`, `fenced_blob_prefix(run_id,gen)`, `adopt_or_submit(...)` decision logic. `run_controller.py` (Phase 1): submit the controller + reconnect/resume reader (wraps `k8s_job_cell_runner`).
- Phase 2/3 integration: fence into `k8s_job_cell_runner._job_name` (`:499`) + `run_matrix` blob prefix (`:1643`) + adopt-on-409 (`:1323-1332`); persisted absolute-epoch deadlines (replace `time.monotonic()` at `_watch_job:879`/`k8s_job_backend:960`); wire the dormant `orchestrator-deployment.yaml` → `campaign --project-id <stable> --resume`; GCS-mirror the campaign ledger; promote GCS `status.json` (`_try_reconcile_status`) to the PRIMARY resume-skip source ahead of local `cell_manifest.json` (`cell_scheduler.should_skip_cell:159`); reaper via `blob_lease.reap_older_generations` + the already-granted `list/delete batch/jobs` RBAC; `live_runs.py:1019` gains a gcp branch that submits the controller.
- Acceptance: **durability drill — kill the CONTROLLER pod mid-training → a successor adopts the in-flight Job by name, resumes from the GCS ledger, finalizes a real metric.** Split-brain drill (stale token refused). Reaper drill (no orphaned A100 bills). Off-state byte-identical.

**WS4 — CPU-class durable lane** (depends on WS3; detail: memory `project_adam_cpu_reproduction`).
- `gpu_count=0` K8s-Job path in `k8s_job_backend`/`k8s_job_cell_runner` (today `gpu_count=max(1,…)` at `k8s_job_backend.py:205`) + routing so a CPU-class paper (no GPU signal) runs unattended on the durable cloud lane, not the laptop.
- Acceptance: a CPU paper reproduces unattended on cloud (no laptop); the current only-success (Adam) runs on this lane.

**WS5 — repo-first grounding** (detail: memory `project_github_repo_first_reproduction`).
- Default `OPENRESEARCH_USE_AUTHOR_REPO` where a paper links code; ground the implementation from the author repo before synthesis.
- Acceptance: a linked-repo paper grounds from the repo by default; off-state unchanged.

**WS-Ext — extractor measurement unblock** (recon-proven MUST-FIX-UPSTREAM; sub-spec in Phase 1, code in Phase 3).
- `repro_spec_extractor.py`: emit an explicit per-claim `kind` (numeric/relative/trend/qualitative, deterministically gated); thread `baseline_value`/`proposed_value` through `_normalize_claim_from_llm` (already LLM-prompted, currently discarded) so RELATIVE claims (the common case) measure; relax the over-conservative A6 cross-check that marks 100% of real claims `ambiguous`. `result_fidelity` dispatches only on an EXPLICIT kind — never an inferred one (a false pass/contradicted is the worst error).
- Acceptance: a real non-ambiguous claim now measures pass/contradicted; Adam stays `inconclusive` (genuinely ambiguous); no false `contradicted`.

**Track E / Track G** — as in `2026-07-10-track-e-eval-scorecard.md` + the handoff. Integrated here as Phase-1 modules (E-4/5/7) + Phase-2 wiring (E-3b/6 on primitives/report) + Phase-3 (E-8, G). All display-or-downward-gate only; never raise a verdict.

## 4. Invariants + guardrails (bind EVERY agent)

1. **North-star** (§0) — `decide()` sole writer; new signals pre-`decide` downward-cap or display-only; the `assert_verdict_surface_unchanged` tripwire on `final_report.json` AND `demo_status.json` must pass after any `report.py` change (run `test_single_verdict_authority_guard.py`, `-n auto`).
2. **Flag discipline** — new capability = `feature_flags.env_truthy("FLAG")`, default-OFF, byte-identical off, hermetic OFF+ON test pair. Every default-flip is a one-line default + the A/B evidence, never a code change.
3. **Subagent git guardrails (from the incident):** every implementer prompt MUST forbid ALL git state commands (`commit`/`add`/`amend`/`checkout`/`stash`/`reset`/`rebase`), restrict writes to an explicit file allowlist, "never edit/delete an existing test — STOP and report," "if a file outside your list needs changing, STOP." The controller `git status`+`git log` verifies footprint before trusting, and commits. Judge tests by the summary line (SDK async-teardown makes exit codes lie).
4. **Determinism/evidence** — measured artifacts on disk are the truth; skills/paper-text are structure/values, never a pass.

## 5. Azure decision + open questions

- **Azure (OPEN — operator):** deferred as GCP-first redundancy. The `BlobLease` is already cloud-agnostic (ETag/`If-Match` seam). Promote to a parallel track ONLY if Azure is a hard customer/tenancy constraint — then bootstrap AKS + GPU quota + Workload-Identity and add the Azure `BlobLease`/controller adapters. **Confirm which reading holds before Phase 3.**
- **Sequencing (RESOLVED):** WS3 runs in parallel with Track E (file-disjoint); the handoff's "WS3-last" is overridden — durability is not gated behind the scorecard.
- **Cost visibility** stays partial on GPU; WS3's reaper + controller-owned node lifetime makes it observable — a later cost-accounting spec builds on it.

## 6. Orchestration recipe (how to actually run it in parallel)

Run as a **phased workflow**, not a free-for-all:
1. **Phase 1** — fan out ~8 agents, one per new module (Axis A), each with a tight brief + the §4 guardrails + a single-file(s) allowlist. Controller reviews each diff, commits serially (disjoint files → no git race).
2. **Phase 2** — fan out ONE agent per hot-file-group (§2.3), max 6 concurrent; each owns its file(s) exclusively for the phase. Controller reviews + commits each before starting a dependent phase-3 step on that file.
3. **Phase 3** — mostly serial: integration, default-flips (gated on A/B), and the real-cluster durability drill (operator-gated GPU spend).
4. Between phases the controller runs `-n auto` + `ruff` and confirms off-state byte-identical for every touched flag.

This partition is what turns "implement the entire thing in parallel" from a clobber-fest into a safe fan-out.
