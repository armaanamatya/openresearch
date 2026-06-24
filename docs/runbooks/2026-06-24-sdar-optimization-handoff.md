# SDAR reproduction — optimization & fidelity handoff (2026-06-24)

> **Mission for the next session:** the end-to-end *pipeline* now works (ingest →
> implement → real 3-model training incl. the 7B → grade → finalize, no crashes).
> The open problem is **result quality / paper fidelity**: at the current
> cost-bounded scope the agentic-env rewards are small or zero, so the run
> reproduces the *method and pipeline* faithfully but does not yet demonstrate the
> paper's headline "SDAR beats baselines on ALFWorld/WebShop/Search-QA" result.
> Investigate how to optimize/improve the results and reproduce the paper better,
> end to end. **Do NOT stop the live run** (see §1) — it's gathering the first full
> 30-cell datapoint; let it finalize, then iterate.

Paper: SDAR (arXiv 2605.15155) — Self-Distilled Agentic RL. Baseline shape in
`backend/agents/prompts/paper_hints.py::PAPER_HINTS["2605.15155"]`.

---

## 1. Live run — KEEP IT GOING

```
project_id : sdar_full_v2
VM         : sdar-ultra · us-central1-c · a2-ultragpu-1g (1×A100-80GB) · STANDARD (on-demand, no preempt)
project    : deepinvent-ext-ut   (CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud)
launched   : 2026-06-24 01:10Z (resume) · outer wall-clock 25h · self-stops on finish
grid       : 30 cells (manifest), ~16 ok + 1 running, ~13 to go (~57%); ETA ~2-4h to graded finalize
monitor    : tail -f runs/_full_v2_resume_mon.log   (a detached scripts/sdar_min_monitor.sh)
guidance   : runs/.cache/sdar_scope_guidance_full.txt (84 lines — the implementer spec)
```

**Watch it / pull results (read-only, won't disturb it):**
```bash
# progress
tail -5 runs/_full_v2_resume_mon.log
# cell statuses + rewards (one SSH)
gcloud compute ssh abheekp@sdar-ultra --zone us-central1-c --project deepinvent-ext-ut --quiet --command \
 'cd /home/abheekp/openresearch; P=runs/sdar_full_v2; OUT=$(ls -dt $P/code/outputs/*/|head -1);
  for d in "$OUT"*/; do [ -f "$d/metrics.json" ] && python3 -c "import json;m=json.load(open(\"$d/metrics.json\"));print(\"'$(basename $d)'\",m.get(\"status\"),m.get(\"reward\"))"; done'
```
**When it finalizes:** `final_report.json` lands in `runs/sdar_full_v2/`. The VM self-stops
(`OPENRESEARCH_SDAR_NO_AUTOSTOP=0`) — to read the report after shutdown, flip to CPU:
`PROJECT_ID=sdar_full_v2 OPENRESEARCH_GCP_INSTANCE=sdar-ultra OPENRESEARCH_GCP_ZONE=us-central1-c scripts/sdar_gcp_e2e.sh inspect`
(a2-ultragpu can't CPU-flip with the GPU attached + is stocked-out to restart — if needed, `gcloud compute instances delete sdar-ultra --keep-disks=boot` then create a small e2 with `--disk=name=sdar-ultra,boot=yes` to read the disk; see §5).

---

## 2. What works end-to-end (do NOT re-debug these — fixed this session)

| Was broken | Fix (committed on `reconcile/grounded-self-improvement-on-main`) |
|---|---|
| Root degenerates (claude-oauth loops FINAL_VAR) | PRIMARY lifecycle mode drives the chain; `OPENRESEARCH_LIFECYCLE_PRIMARY=1` |
| Cells time out (unbatched rollouts) | guidance: batched rollouts + 20-step budget (FIX 1-3) |
| `preflight_smoke` false-fails on repo-first orphan deps | scoped to the cell entry-point import closure (`6d2f2ba5`) |
| Partial grid → scoreless `failed` | lifecycle driver always grades partial evidence + best-of-run (`7aaaff0b`) |
| `ModuleNotFoundError: backend.services.runs` on fresh VM | rsync `--exclude 'runs/'` → `'/runs/'` anchor (`7aaaff0b`) |
| self_stop nukes the scarce VM on a fast crash | fast-crash guard keeps the VM up <`OPENRESEARCH_SDAR_FASTCRASH_S` (`f32c12c8`) |
| 400GB disk fills (306GB per-cell checkpoints) | create-poll disk default 1TB (`572a4536`) + guidance FIX 5 (no per-cell checkpoints) |
| 7B OOM on 80GB | never happened — 7B trains at gpus:1 on one 80GB card |

**Compute reality (confirmed):** 1×A100-80GB GPU is *sufficient* (7B fits gpus:1, no
sharding). 4-GPU/2-GPU 40GB blocks are STOCKED OUT in all us-central1 zones; A100-80GB
quota is 4 (auto-granted) but `a2-ultragpu` capacity is scarce and offered ONLY in
us-central1-a/-c. The bottleneck for the *full grid* was **disk (need ~1TB), not GPU**.

---

## 3. Results so far (16/30 cells) — the actual gap to close

```
Search-QA : 6/6 cells non-zero reward (max ~0.0215) — real learning signal, scales with model
ALFWorld  : 1/6 non-zero (only qwen2.5-7B grpo, 0.0078) — the rest ~0
WebShop   : 0/6 — all zero
```
Viz: the per-cell reward chart was sent to the operator (`/tmp/sdar_results.png`; regenerate
from `runs/sdar_full_v2/code/outputs/*/metrics.json`).

**Known issues to investigate (these are the mission):**
1. **ALFWorld ≈ 0 reward.** Small models don't complete multi-turn ALFWorld in a 20-step
   budget; only the 7B scratched a signal. Reward-ceiling, not a crash.
2. **WebShop = 0 everywhere.** The env was `env_unavailable` (server didn't come up), now runs
   "ok" but returns 0 — the WebShop server isn't truly serving episodes.
3. **Metric-computation fidelity (IMPORTANT).** The per-cell `accuracy`/`success_rate` ≈
   `reward × 100` *exactly* (reward 0.0087 → "acc" 0.872) — i.e. accuracy looks *derived from*
   reward, NOT an independent EM/F1 (Search-QA) or task-success {0,1} (ALFWorld) measurement.
   **Verify the agent's `train_cell.py` computes real eval metrics**, not a scaled reward.
4. **SDAR-vs-baseline inconclusive.** On Search-QA sdar≈grpo within noise (1.7B: sdar 0.0087 >
   grpo 0.0046; 3B: ~tie). Rewards too small/noisy to confirm the paper's claim.

---

## 4. Optimization plan — how to reproduce the paper better (prioritized)

The single biggest lever is **training budget** (the 20-step cap is the dominant
reward-ceiling cause). Investigate in this order, A/B-ing against the current sdar_full_v2 baseline:

1. **More training steps (20 → 100-150, the paper's regime).** This is the headline lever for
   real reward on the agentic envs. Cost: 1×80GB sequential is slow (~30 cells × longer cells).
   Mitigate with (a) more GPUs for parallelism (a2-ultragpu-2g/4g if capacity returns; quota=4
   A100-80GB) OR (b) narrow to the 9 headline cells (3 models × 3 envs, drop the 12 ablations)
   so the budget buys depth not breadth. Edit `TRAINING_STEPS` in `runs/.cache/sdar_scope_guidance_full.txt`.
2. **Fix the reward/eval-metric computation (issue #3).** Pull `runs/sdar_full_v2/code/train_cell.py`
   and confirm: Search-QA reward = real EM/F1 vs gold; ALFWorld reward = real task-success {0,1};
   `accuracy` is measured on held-out eval, NOT `reward×100`. If it's derived, the "results" are
   an artifact — fix the metric wiring first (cheapest, highest-fidelity win). Reinforce in guidance.
3. **WebShop env (issue #2).** Investigate why the server returns 0 — `OPENRESEARCH_WEBSHOP_PYTHON`
   venv + the server startup (it was `env_unavailable` then "ok"-but-0). Either get it serving real
   episodes or keep it a documented gap (don't let it dilute the grade — it's already excluded
   from inclusion scope by the leaf scorer for non-WebShop runs).
4. **Use the author's training recipe more faithfully (repo-first).** The run is repo-first
   (`OPENRESEARCH_USE_AUTHOR_REPO=1`, clones ZJU-REAL/SDAR into `runs/<id>/repo/`). Check whether
   the agent's `train_cell.py` actually adapts the authors' GRPO/OPSD training loop + hyperparameters
   (lambda_SDAR=0.01, beta=5.0, G-group sampling) vs a from-scratch reimplementation. Closer
   adaptation = better fidelity. Consider `OPENRESEARCH_REPRODUCTION_MODE=reference` to force a
   clean-room read of the real loop.
5. **More compute → parallel + deeper.** If `a2-ultragpu-2g/4g` capacity returns (poll it), run the
   grid in parallel (more steps in the same wall-clock). The runner already create-polls; set
   `OPENRESEARCH_GCP_GPU_MACHINE_TYPE=a2-ultragpu-2g OPENRESEARCH_SDAR_MIN_GPUS=2` + the 7B can stay
   gpus:1 on an 80GB card while a second cell runs concurrently.
6. **Seeds + eval slice.** Currently seed 0 + a small eval slice (n_eval=8). For a paper-grade
   number, more seeds + a larger eval slice (variance + stable accuracy) — costs more compute.

**A/B discipline:** the repo has an A/B harness — stamp `experiment_arm`, use
`scripts/ab_compare.py --paper 2605.15155` for paired deltas; `OPENRESEARCH_REUSE_RUBRIC=1` pins
the rubric so the grader doesn't drift between arms. Don't conclude "SDAR > baseline" without ≥3
paired runs (the reward signal sits inside grader noise at this scope).

---

## 5. Operating knowledge (runner, gotchas, lessons)

- **Runner:** `scripts/sdar_gcp_optimal_run.sh` — create-polls on-demand a2-ultragpu (free while
  the VM doesn't exist), syncs latest code, writes the optimal env, GREEN-gated launch, watch,
  auto-stop. Env knobs: `PROJECT_ID, OPENRESEARCH_GCP_ZONE/INSTANCE/GPU_MACHINE_TYPE,
  OPENRESEARCH_SDAR_MIN_GPUS, SDAR_VRAM_GB (80 for ultragpu), GUIDANCE_FILE, CREATE_IMAGE,
  CREATE_DISK_GB (default 1000), LIFECYCLE_MAX_IMPROVE, SDAR_LAUNCH_LOCK`.
- **Monitor:** `scripts/sdar_min_monitor.sh` (fragility-free; `scripts/sdar_gcp_watch.sh` dies on
  set -e — prefer the min monitor). Launch detached: `setsid nohup env PID=<id> INSTANCE=<vm>
  ZONE=<z> bash scripts/sdar_min_monitor.sh > runs/_<id>_mon.log 2>&1 < /dev/null &`.
- **GOTCHA — killing a run:** killing the python `reproduce` process triggers `sdar_gcp_run.sh`'s
  `self_stop` → VM shutdown. To stop a run without losing the VM: `pkill -f sdar_gcp_run.sh` (the
  wrapper) FIRST, or `OPENRESEARCH_SDAR_NO_AUTOSTOP=1`. (The fast-crash guard already protects
  <10-min crashes.)
- **GOTCHA — never two pollers stop each other / the daemon never stops a RUNNING VM** (fixed; a
  cross-poller `mkdir` lock + a use-if-correct/bail safe-stop). Don't revert.
- **GOTCHA — scheduling flip order:** set machine-type to a2 FIRST, then STANDARD (e2-STANDARD
  can't TERMINATE); the runner asserts `provisioningModel==STANDARD` before polling.
- **Read a TERMINATED a2-ultragpu disk:** can't CPU-flip (GPU attached) + can't restart (stockout)
  → `gcloud compute instances delete <vm> --keep-disks=boot` then create an e2 with
  `--disk=name=<disk>,boot=yes,auto-delete=no`, SSH, read; clean up after.
- **`gcp_info.md`** is local-only operator notes — gitignored, NEVER track it (a `.gitignore`
  union re-tracked it once; the hygiene test `tests/test_repo_hygiene.py` guards this).

---

## 6. Merge to main — state

`reconcile/grounded-self-improvement-on-main` @ `b465a390` is **fully reconciled with main**
(behind=0; 37 conflicts resolved keep-all-unique; 7285 tests pass, only pre-existing/env failures),
pushed to deepinvent. **PR not yet opened** (gh CLI unauthenticated here) — open it:
`https://github.com/Deepinvent/scientific_article_generator/pull/new/reconcile/grounded-self-improvement-on-main`
(base `main`, draft title/body in this session's history). Do NOT force-push; it fast-forwards.

---

## 7. Where to start (next session)
1. Let sdar_full_v2 finalize; pull `final_report.json` + the 30-cell `metrics.json` (the first full datapoint).
2. **Pull `train_cell.py` and audit the reward/accuracy computation (issue #3)** — this gates whether
   the numbers mean anything. Fix the metric wiring if it's `reward×100`.
3. Then the budget lever (§4.1): a headline-9-cell run at 100+ steps, A/B vs the 20-step baseline.
4. Triage ALFWorld + WebShop (§4.2-4.3).
Related memory: `project_lifecycle_driver`, `project_sdar_gcp_harness_refactor`, `project_github_repo_first_reproduction`.
