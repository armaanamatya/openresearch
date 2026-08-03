<!-- doc-meta: status=live; started=2026-08-01 -->
# Feature-ablation results — per-feature reproduction scores

Live results of the per-feature ablation on GCP single-VM (L4), **`sonnet-foundry`** (real
Claude via Foundry API key — **NOT OAuth**, thinking-patched `_anthropic_thinking_patch.py`).
Each arm = the fixed honest baseline (`configs/ablation/baseline_run_spec.json`) plus ONE
feature (`configs/ablation/arms.json`); `all_on` = every feature. Scored by the auto-generated
PaperBench rubric (`rubric.overall_score`). Method + launch: `docs/runbooks/2026-08-01-feature-ablation-gcp-runbook.md`.

## Scoreboard (paper: ResNet arXiv 1512.03385, seed 1, L4)

| Feature (arm) | What it adds | Rubric score | Verdict | Δ vs baseline | Date/time (UTC) | Status |
|---|---|---:|---|---:|---|---|
| **baseline** | fixed honest infra, no test features | **0.466** (rubric) | **partial** (credited; target 0.6) | — (reference) | 2026-08-03 14:40 UTC | ✅ completed + credited (base_rn3) |
| bes | best-of-N candidate pool | _pending_ | — | — | — | ⏳ queued |
| champion | champion-artifact + evidence-fingerprint rails | _pending_ | — | — | — | ⏳ queued |
| recipes | cross-run positive recipes | _pending_ | — | — | — | ⏳ queued |
| expmem | cross-run experience memory | _pending_ | — | — | — | ⏳ queued |
| lessons | cross-run negative lessons | _pending_ | — | — | — | ⏳ queued |
| audit | evidence-audit deterministic critic | _pending_ | — | — | — | ⏳ queued |
| leafgate | per-leaf evidence gate (anti-fabrication) | _pending_ | — | — | — | ⏳ queued |
| **all_on** | all 7 features combined | _pending_ | — | — | — | ⏳ queued |

> Scores populate as each ~1.5 h run completes. The baseline is validated running end-to-end
> (root → rubric-gen → executor producing code → GPU training → grade) on `sonnet-foundry`.

## ⚠️ Why the runs kept dying — ROOT CAUSE (corrected 2026-08-02)

**It was never GCP host maintenance and never uncontrollable cloud flakiness. It was the
`--max-run-duration` cap we set on the VMs.** Evidence (GCP `system_event` audit log +
timing math):

| VM | Started | Stopped | Ran for | Cap (`maxRunDuration`) | Log method |
|---|---|---|---:|---|---|
| `base-vm` | 03:51:54 PDT | 09:52:41 PDT | **6.013 h** | 21600 s = **6 h** | `compute.instances.deferredStop` |
| `allon-vm` | 03:52:06 PDT | 09:52:49 PDT | **6.012 h** | 21600 s = **6 h** | `compute.instances.deferredStop` |
| first `or-ablation-run` | — | 06:55 UTC | ~cap | (short) + `DELETE` action | `compute.instances.deferredDelete` |

- `compute.instances.deferredStop` / `deferredDelete` is **exactly** what GCP emits when
  `maxRunDuration` expires (with `instanceTerminationAction: STOP` / `DELETE` respectively). A
  genuine host-maintenance event logs `hostError` or a migrate/terminate event — **not** this.
- Both VMs stopped **8 s apart** only because they **started 12 s apart**; each hit its identical
  6 h cap. The "host maintenance killed both simultaneously" reading was a misdiagnosis.
- **The score was never a compute-time problem — it was a COLLECTION problem** (corrected again
  2026-08-02 after recovering base-vm's disk). base-vm's `base_rn` run **completed in ~3 h**
  (started 10:54 UTC, `final_report.json` written 14:01 UTC) — well inside the 6 h cap. The VM
  then sat **idle** for ~3 h until the cap STOPped it at 16:52 UTC, and we lost SSH before ever
  pulling the artifact. The report sat on the disk the whole time. So the 6 h cap is real hygiene
  but was **not** the blocker; **not scp-ing the score the moment it landed** was.
- **`--instance-termination-action=STOP` is vindicated, not incidental:** STOP (not DELETE)
  preserved the boot disk, which is the *only* reason we recovered `final_report.json`,
  `rubric_evaluation.json`, and `experiment_runs.jsonl` after the fact (now in
  `runs_logs/recovered/base_rn/`). Had we used DELETE (as the first run did), the completed result
  would have been destroyed.

**The fixes (all in our control):**
1. **Raise the cap** to safely exceed real completion time — `--max-run-duration=64800s` (18 h),
   or drop it and rely on a monitored teardown. Never cap below the observed run time.
2. **`--instance-termination-action=STOP`, never `DELETE`** — a cap-stop then preserves the disk so
   we can restart and pull artifacts. The first run used `DELETE` and destroyed the results.
3. **scp `final_report.json` to local disk the moment it appears** (poll the run dir every ~60 s) —
   so even an unexpected stop never loses a completed score.
4. **Shrink time-to-score** — pre-bake torch/deps into the image or startup script so bootstrap
   doesn't burn hours, and/or reduce the ResNet cell-matrix scope for the 1-seed screen.

> Prior chat/notes that blamed "GCP host maintenance" or "relentless GCP flakiness" for these two
> kills are **superseded by this entry**. The transient SSH-255 / scp-EOF issues were real but
> secondary and retryable; the runs-never-finished blocker was the 6 h cap.

**Where the 6 h cap came from (so we don't reintroduce it):** the **code path is safe** —
`VmComputeProvider._DEFAULT_MAX_RUN_DURATION_S = 100800` (28 h). The 6 h cap was in the **manual
`gcloud` launch** used by the direct-recipe (`--max-run-duration=21600s`), not the provider. Two
more 6 h defaults to override on any manual/`reproduce` launch: `campaign_policy.DEFAULT_ATTEMPT_WALL_S
= 21600.0` (app per-attempt wall) and `--max-wall-clock` — both self-terminate at 6 h unless raised.

## Testing sequence (the campaign roadmap)

Deliberate order — **baseline → Tree-A → Tree-B → combo of both**:

1. **Baseline** (control, no test features) — establishes the reference score every Δ is measured
   against. *(running now: `base_rn`, ResNet.)*
2. **Tree-A** (within-run features) — the 7 single-feature arms (`bes`, `champion`, `recipes`,
   `expmem`, `lessons`, `audit`, `leafgate`) + `all_on` (all 7 combined). Each = baseline + that
   feature; Δ vs baseline = its contribution. Runs via `reproduce`. **This is the current fan-out.**
3. **Tree-B** (scheduler authority: **freeze / branch / revive / true-kill**) — measured separately
   because it is **not** a within-run feature: it only exists in a **`campaign`** cohort loop
   (`OPENRESEARCH_SCHEDULER_AUTHORITATIVE`), and it produces **zero receipts** today (cells don't emit
   the 5-field checkpoint contract → July runs: 4 branches spawned, 0 receipts, `0.203 inconclusive`).
   **GATED on a code build first: A1 (train_cell checkpoint emission) + A2 (per-branch isolation).**
   Then scored as a serial-vs-authority A/B campaign. **Not runnable in the Tree-A ablation** — adding
   the `SCHEDULER_*` flags to a `reproduce` arm would be a silent no-op.
   **Phase-3 starting points (do NOT rebuild from scratch)** — the scheduler-authority code and the
   local-transport harness were pruned to backup tags, not deleted:
   `git show backup/pruned/scheduler-authority-runtime` (+ `authoritative-scheduler-runtime`,
   `scheduler-authority-enrichment`) = the receipt-gated authority controller/runtime groundwork;
   `git show backup/pruned/gke-local-transport` = the **local cell-matrix transport** that lets the
   cohort loop run locally/hermetically for **$0** — validate Tree-B there before any cloud spend.
4. **Combo of both** (Tree-A features + Tree-B authority together) — only after (2) and (3) are each
   validated. Answers whether the within-run features and the cross-branch scheduler compose.

> ⚠️ `all_on` in the Tree-A ablation = **all 7 Tree-A features**, NOT freezing/Tree-B. Do not read
> it as "everything." Tree-B and the combo require the A1/A2 build (Phase 3 prerequisite).

## Method (so the numbers are trustworthy)
- **Auth: `sonnet-foundry` (Foundry API key), never OAuth** (operator directive). Root +
  executor + grader + verifier all `sonnet-foundry`.
- **Baseline infra ON in every arm** (reliability + anti-fabrication + grader-fidelity +
  feasibility scope), so a score reflects the feature, not a broken run. Only the one ablated
  feature differs per arm.
- **Δ vs baseline** is the per-feature contribution. `all_on` vs Σ(individual Δ) tells you if
  features compose or are redundant.
- Fidelity caveat: a full per-feature verdict needs ≥3 seeds through the grader-σ gate
  (`asha_authority_gate`); this scoreboard is the 1-seed screen.

## First real baseline result (recovered 2026-08-02) — `base_rn`, ResNet 1512.03385, seed 1, L4

The first end-to-end-completed arm. Full chain ran on `sonnet-foundry` (planner/executor/verifier/
grader all `claude-sonnet-5` via Foundry) in ~3 h, 20 iterations. **rubric.overall_score = 0.433**,
**verdict = failed** (target 0.6). The `failed` verdict is *correct and fail-closed*, for two
independent reasons — do **not** read 0.433 as a suppressed pass:

1. **Real but partial reproduction (the science signal).** Shallow nets matched the paper, deep nets
   diverged monotonically — a classic missing-**LR-warmup** / deep-residual-config gap:

   | cell | test-error % | paper ballpark | |
   |---|---:|---:|---|
   | plain20 | 9.59 | ~9–10 | ✅ |
   | resnet20-optA | **8.64** | 8.75 | ✅ excellent |
   | resnet20-optB | 9.12 | — | ✅ |
   | resnet32-optA | 35.6 | ~7.5 | ❌ diverged |
   | resnet44-optA | 45.9 | ~7.2 | ❌ |
   | resnet56-optA | 49.8 | ~6.97 | ❌ |
   | resnet110-optA | 62.9 | ~6.4 | ❌ (needs 0.01→0.1 LR warmup) |

2. **Evidence gate refused credit (working as designed).** `experiment_runs.jsonl`: call 1 (13:29)
   = all 11 cells crashed (`cell_execution_error`); call 2 (13:59) = 11/11 leaf cells `status: ok`
   with the numbers above, **but** `success=false` and `verdict_clamped: "success_compatible=0 …
   code/metrics.json is root-writable, so neither a full-credit grade nor a verdict upgrade can be
   granted."** Without the 5-field checkpoint contract (**A1 gap**) the model-written metrics can't
   be trusted — the "evidence, not grade" invariant holding the line.

**Path to a real pass (never by weakening the gate):** (a) fix deep-net training (LR warmup for
resnet110, config for 32/44/56); (b) land the checkpoint-evidence contract so a completed
`run_experiment` is `success_compatible` on trustworthy on-disk evidence. **Secondary, to verify:**
`_all_models_failed_violation` (primitives.py, default-OFF flag `OPENRESEARCH_PER_MODEL_STATUS_GATE`)
checks **model-level** `status`, but the cells route nests `status` at the leaf — it may misfire on
nested `per_model`; even if fixed it would **not** grant this run a score (the root-writable clamp +
deep-net divergence stand). Recovered artifacts: `runs_logs/recovered/base_rn/`.

## ⏸️ CURRENT STATE + HOW TO RESUME (2026-08-03, pre-compaction)

**Scores in hand:** baseline `base_rn3` = **partial 0.466** (credited, committed). `all_on_rn5`
(Tree-A, all 7 features) = **still running on allon-vm** — its lifecycle poller pulls the score +
baseline Δ when it lands (`runs_logs/recovered/all_on_rn5/`).

**Tree-B:** authority chain **hermetically validated** (4 tests, committed) AND the missing
**rung-step → trainer iters wire is built + tested** (commits `834ba242`/`2c0002a1`/`46628d85`,
465 tests green, env-gated byte-identical-off). ResNet authority spec added
(`configs/resnet_authority_spec.json`, committed `bb32338a`). The **first-ever GPU authority
campaign has NOT been launched yet** (held before compaction).

**VM states:** allon-vm RUNNING (all_on_rn5); treeb-vm TERMINATED (bootstrapped, ready);
**base-vm TERMINATED — operator's friend's, DO NOT TOUCH**; adam-tier3-vm old/terminated.

**To launch the ResNet Tree-B GPU campaign (the one remaining GPU run):**
1. `gcloud compute instances start treeb-vm --zone us-central1-a --project deepinvent-ext-ut`
2. scp the 3 wired files to the VM (treeb-vm has pre-wire code):
   `backend/agents/rlm/primitives.py`, `backend/agents/rlm/reproduction_campaign.py`,
   `configs/resnet_authority_spec.json`.
3. On the VM (venv MUST be on PATH):
   ```bash
   cd ~/or && export PATH=$HOME/or/.venv/bin:$HOME/.local/bin:$PATH
   OPENRESEARCH_MIN_DISK_GB=0 OPENRESEARCH_SCHEDULER_TREE=1 OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1 \
   OPENRESEARCH_ROLE_MODELS='{"executor":"sonnet-foundry","grader":"sonnet-foundry","verifier":"sonnet-foundry"}' \
   setsid nohup python -m backend.cli campaign 1512.03385 --campaign-driver unified --sandbox local \
     --authority-spec configs/resnet_authority_spec.json --root-model sonnet-foundry \
     --max-llm-usd 15 --max-gpu-usd 20 --max-gpu-hours 12 --project-id treeb_resnet \
     > ~/treeb_resnet.log 2>&1 &
   ```
   (`campaign` uses `--root-model` + `OPENRESEARCH_ROLE_MODELS` env — NOT `--model`/`--models`.)
4. It's NOVEL — early-checkpoint ~14 min in (authority constructed? branches spawned?
   `CELL_ITER_BUDGET` in the log? errors?) before committing to a long poll. Verify Tree-B via the
   `branch-tree:<campaign>` events + `python -m backend.agents.rlm.asha_shadow_report <run_dir>`,
   never the ledger. Expect fresh integration issues (novel run). Resume-for-rung-climb is DEFERRED
   (cold-start climb still fires freeze/promote/kill; see Tree-B run-log entry).

## Run log
- **2026-08-03 — TREE-B RUNG-STEP WIRE BUILT (Phase-3 enabler).** The authority campaign never ran
  on hardware because the ladder `to_step` was stamped into the branch payload but **never consumed**
  into the trainer's iteration budget (the trainer ran its own `cells.json iters`, ignoring the
  ladder). Fixed: `reproduction_campaign._branch_iter_budget_enforcement` threads
  `OPENRESEARCH_CELL_ITER_BUDGET=to_step` into the branch `enforcement["env"]` (the only surface the
  driver forwards to the child), and `primitives._apply_cell_iter_budget` (in `_execute_cell_matrix`,
  before `normalize_cell_axes`) deterministically rewrites every cell's `iters`→budget (+ clamps
  `lr_milestones`), the field the ResNet trainer already honors. Env-gated: **byte-identical when
  `OPENRESEARCH_CELL_ITER_BUDGET` unset**. TDD: `tests/rlm/test_cell_iter_budget.py` (7) +
  cohort/env-thread tests; 465 authority/campaign tests green. **Resume-for-rung-climb DEFERRED** —
  `gpu_cell_runner:804` unconditionally overwrites `OPENRESEARCH_CELL_CHECKPOINT_DIR` per-cell, so a
  cross-branch checkpoint pointer is clobbered; a cold-start rung climb still produces a valid
  checkpoint+metric, so freeze/promote/kill is demonstrated without it. Commits `834ba242`,
  `2c0002a1`, `46628d85`; ResNet spec `bb32338a`.
- **2026-08-03 — TREE-B AUTHORITY VALIDATED HERMETICALLY (Phase 1+2 done).** The scheduler
  authority chain (freeze/branch/revive/true-kill) was found **already built** — a recon claim that
  the checkpoint *producer* was missing turned out false: `gpu_cell_runner` always sets
  `OPENRESEARCH_CELL_CHECKPOINT_DIR`, the trainer scaffold emits the 5-field checkpoint, and
  `reproduction_campaign._authority_dispatch_impl` reads it → `build_raw_receipt` →
  `controller.record_cell_receipt` → applies ASHA. The real gap was **it had never run with a real
  trainer** (all tests used synthetic checkpoint bytes). New hermetic test
  `tests/rlm/test_authority_e2e_real_checkpoint.py` (4 tests) drives the REAL controller from a REAL
  tiny-CPU-torch trainer's checkpoint through a real **promote + freeze + true-kill** with the
  matching `branch_lineage` DomainEvents, plus fail-closed / kill-gating / NaN-rejection guards.
  **No production changes needed** (chain already correct). 143/143 authority+scheduler regression
  green. Design+plan: `docs/superpowers/{specs,plans}/2026-08-03-tree-b-authority-e2e-validation*`.
  Remaining: **Phase 3** = one real GPU authority campaign (`--authority-spec-path
  configs/adam_authority_spec.json` + `SCHEDULER_TREE=1 SCHEDULER_AUTHORITATIVE=1`).
- **2026-08-03 — BASELINE PROPERLY CREDITED (base_rn3): `partial`, 0.466.** First run where a
  completed cells-route reproduction is credited instead of clamped to `failed` — validates the
  `all_models_failed` guard fix (leaf-status descent) + the launch fix end-to-end. Ran the full
  ~6 h (08:38→14:40 UTC), **2 experiments both `success=True, status=complete`**, `verdict_clamped:
  None`. Contrast the recovered `base_rn` (0.433, `failed`, clamped) — same paper/seed, now
  credited because the guard no longer false-fires on the nested `per_model`.
  - **New gotcha found + fixed — venv MUST be on PATH.** The cell trainer resolves its interpreter
    via `command -v python3`, which finds SYSTEM `/usr/bin/python3` (no torch) unless the run venv
    is on PATH. Launching `.venv/bin/python -m backend.cli` is NOT enough — the child cells don't
    inherit the venv. Two runs (`base_rn2`,`all_on_rn2`) died with `overall:None` / preflight
    `No module named 'torch'` before this was caught. Fix: `export PATH=$HOME/or/.venv/bin:$PATH`
    before launch (or launch from an activated venv). Now in the runbook.
  - **Reliable launch = GCE startup-script**, not interactive SSH. OS-Login/IAP rate-limits SSH
    hard under rapid calls; a startup-script launches the run on boot (verify via the serial
    console — both API, no SSH) and SSH is used only for sparse collection.
  - `all_on_rn3` was SIGTERM-interrupted mid-run (iter 14) by an orchestrator bug (macOS bash 3.2
    has no `declare -A`) → invalid; re-running clean as `all_on_rn4`.
- **2026-08-01** — GCP pipeline validated on `sonnet-foundry`: root crash fixed, rubric-gen
  fixed (thinking-disable patch), executor generates code. Two prior Foundry-Sonnet blockers
  found + fixed (see `docs/runbooks/2026-08-01-remote-run-llm-auth.md`).
- **2026-08-02 — baseline (`base_rn`) confirmed TRAINING ON GPU** (L4 at 22% util / 476 MiB;
  `train_cell.py` cell `plain20__cifar10__s42` running, torch loaded, CUDA engaged). Full chain
  validated end-to-end on `sonnet-foundry` + torch: rubric → CUDA-correct codegen → real GPU
  training. Score pending run completion (~1–2 h for the ResNet cell matrix). SSH to the VM is
  intermittently 255-rate-limited; the log stdout is buffered (monitor via disk artifacts +
  `nvidia-smi`, not the log tail).
- **2026-08-01 (3rd blocker)** — first baseline finished in ~15 min with a **null score**:
  `train_cell.py` failed `ModuleNotFoundError: No module named 'torch'` (`cell_execution_error`).
  **The VM bootstrap installed only `backend/requirements.txt` (orchestrator deps), not the ML
  training deps.** Fix: after `backend/requirements.txt`, ALSO
  `uv pip install torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121`
  into the run venv on EVERY arm VM. Baseline re-launched with torch (real GPU training this
  time). This is now in the fan-out cron + the runbook — do NOT fan out without it.
