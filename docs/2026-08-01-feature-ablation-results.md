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
| **baseline** | fixed honest infra, no test features | **0.433** (rubric) | **failed** (evidence gate; target 0.6) | — (reference) | 2026-08-02 14:01 UTC | ✅ completed, recovered from disk |
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

## Run log
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
