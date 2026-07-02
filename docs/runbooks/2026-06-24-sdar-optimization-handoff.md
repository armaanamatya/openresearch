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

---

## 8. Error & fidelity analysis + optimal solution set (2026-06-24, analysis session)

**Live-run state (read-only pull, NOT disturbed):** `sdar_full_v2` is healthy — **19 cells `ok`, 1
running, zero crashes** (no OOM, no `env_unavailable`, no traceback; a single early
`compute_scope_invalid` warning). The pipeline is sound. **The open problems are measurement
fidelity and result quality, NOT failures — there are no crashes to hunt.**

### 8.1 Root cause of issue #3 (metric fidelity) — CONFIRMED by auditing `train_cell.py`
Pulled `runs/sdar_full_v2/code/train_cell.py` (1202 lines) and traced the metric path end-to-end:

- **`accuracy` is never measured — it is `mean_success × {100 | 128}`** (`train_cell.py:962-968`,
  final value `:1010-1012`). `mean_success` is the mean of `step_successes`, collected *inside the
  GRPO training-rollout loop* (`:833-839`): ALFWorld `info["won"]`, Search-QA `info["f1"]`, else
  `info["reward"]` (≡ reward). For these envs reward≡success, so the whole chain collapses to
  `accuracy = reward × 100`.
- **No held-out eval exists.** The only split reference is `ALFWorldEnv(train_eval="train")`
  (`:154`) — no test/eval slice, no `n_eval`, no gold-vs-prediction scoring. **Every number is
  in-sample on the training rollouts.** (The guidance asked for a held-out eval; the agent skipped it.)
- **The ×100/×128 is a units bug.** `won`/`f1`/success are already fractions in [0,1]; scaling by
  100 turns a true success of ~0.008 into `accuracy ≈ 0.83`. That is exactly why `accuracy ==
  reward × 100` in every live cell (7B Search-QA `r=0.00833 → acc=0.8326`). **Reported accuracy
  overstates true success by ~100×.**

**Consequence:** at the current near-zero regime the inflated value still lands inside [0,1] (0.83),
so it looks plausible; if the model ever did well (true F1 0.4) the same code emits `accuracy=40`.
Either way the headline "accuracy" is not a trustworthy, independent, held-out measurement.
**You cannot optimize against a broken yardstick — fix this before drawing any SDAR-vs-baseline
conclusion.** (The honest reading of the current numbers: Search-QA reward ~0.002-0.009 — real but
tiny; ALFWorld/WebShop ≈ 0; SDAR ≈ GRPO within noise.)

### 8.2 The harness gap this exposes (why it generalizes to EVERY paper)
The harness has ~15 anti-fabrication guards (evidence-gate, zero-metrics, stub, VRAM-antifab,
metric-provenance, completeness, …). **All of them verify the *training* side** (real GPU work?
non-zero / non-constant metrics? provenance present?). **None verify the *eval* side** — that a
self-reported `accuracy`/`f1`/`success_rate` is a real held-out measurement against gold, in correct
units. SDAR's run passes every guard yet reports a derived, in-sample, ×100 metric. **Any paper's
agent can do the same** — this is a structural gap, not an SDAR quirk.

> A naive bounds check (`accuracy ∈ [0,1]`) does NOT fix this: it misses the bug exactly when
> results are near-zero (0.83 ∈ [0,1]) and only fires once the model does well. Weak partial, not the fix.

### 8.3 Optimal solution set (robust · elegant · generalizable), prioritized

**Tier 1 — Fidelity (cheap, highest value, helps all papers):**
- **S1 — eval-metric provenance (the root fix; the elegant generalization).** Extend the
  *provenance principle from training to evaluation*. A cell reporting a held-out metric must persist
  an `eval_provenance` sidecar — `{n_eval, sample of (input, prediction, gold, correct)}` — and a
  pure-Python guard recomputes the headline metric from those records and **vetoes/flags** when
  (a) reported ≠ recomputed, (b) the eval set is identical to / drawn from the training rollouts
  (not held-out), or (c) the sidecar is absent for a result-claiming leaf. Same shape as the existing
  `provenance.json` + evidence-gate; flag-gated, default-OFF (byte-identical when off); makes "this
  number is a real measurement" *verifiable, not trusted*. **One canonical guard for the whole class,
  for every paper** — not a per-paper patch.
- **S2 — guidance hardening (immediate, zero-cost; backstop only).** The implementer spec already
  said "accuracy NOT reward×100" and the agent did exactly that, so prose alone is insufficient.
  Make it unambiguous and pair it with S1: *compute EM/F1/success on a held-out slice vs gold; report
  `success_rate`/`em`/`f1` in [0,1]; never scale by 100/128; never set success≡reward; persist the
  eval records.* (`runs/.cache/sdar_scope_guidance_full.txt`.)

**Tier 2 — Result quality (the real lever; compute-bound):**
- **S3 — training budget (the headline lever).** 20 → 100-150 steps on the **9 headline cells**
  (drop the 12 ablations to buy depth, not breadth). THE lever for non-zero agentic-env reward.
  A/B vs the 20-step `sdar_full_v2` control (`OPENRESEARCH_REUSE_RUBRIC=1`,
  `scripts/ab_compare.py --paper 2605.15155`).
- **S4 — WebShop.** Runs "ok" but reward≡0 everywhere → the server isn't serving real episodes.
  Either fix the `OPENRESEARCH_WEBSHOP_PYTHON` server startup, or fold WebShop into
  `scope.gaps`/inclusion-exclusion so a non-serving env can't dilute the grade (never fabricate).
- **S5 — ALFWorld.** Reward-ceiling, not a bug: small models can't finish multi-turn ALFWorld in 20
  steps (only 7B-grpo scratched a signal). S3 (budget) is the primary mitigation.

**Tier 3 — Rigor:**
- **S6.** Do NOT conclude "SDAR > GRPO" — at this scope sdar ≈ grpo within noise (3B/7B grpo slightly
  higher; 1.7B sdar higher). Needs S1 + S3 + ≥3 paired seeds first. Audit repo-first faithfulness:
  does `train_cell.py` adapt ZJU-REAL/SDAR's real GRPO/OPSD loop, or reimplement from scratch?
  (`OPENRESEARCH_REPRODUCTION_MODE=reference` forces a clean-room read of the authors' loop.)

### 8.4 Sequencing (none of this disturbs the live run)
1. Let `sdar_full_v2` finalize — the first full 30-cell datapoint (on the *current, flawed* yardstick;
   still useful as the A/B control + to confirm how the grader treats inflated accuracy).
2. Build **S1** (eval-provenance guard) + apply **S2** (guidance) — both off-VM, safe to do now.
3. Next run = **S3** headline-9 at 100+ steps, S1 on, A/B vs the v2 control.

---

## 9. IMPLEMENTED this session (2026-06-24) + the launch recipe

All three Tier-1/budget items are now built, flag-gated **default-OFF** (byte-identical when unset),
and ready to launch when `sdar_full_v2` finalizes. The live run was NOT touched.

**S1 — eval-metric provenance guard (the root fix; generalizes to every paper).**
- New module `backend/agents/rlm/eval_provenance.py` (pure stdlib, fail-soft), double-duty like
  `provenance.py`:
  - **Producer** `record_eval(output_dir, *, model_key, env, baseline, metric_name, records, …) -> float`
    — auto-copied into `code/` (added to `_HARNESS_CODE_HELPERS`); computes `mean(outcome)` in [0,1],
    writes a verifiable `eval_provenance.json` sidecar, returns the value for `metrics.json`.
  - **Guard** `eval_provenance_should_veto(code_dir) -> (bool, msg)` — wired into `run_experiment`
    postflight (`primitives.py`, right after the metric-semantics block). For each success-status cell
    claiming a positive rate metric (accuracy/success_rate/f1/em/…), it requires a sidecar and vetoes
    (→ repairable `fabrication_suspected`) when the sidecar is absent, a per-example outcome is outside
    [0,1] (catches ×100), or `reported ≠ mean(records)` (catches reward-derived). Conservative: RL-reward-only
    cells (no rate key), zero-valued metrics, boolean `success` flags, and non-success cells are all exempt.
- Flag: **`OPENRESEARCH_EVAL_PROVENANCE_GUARD`** (default OFF).
- This is the complement to the existing `metric_semantics` guard (which only catches rate > 1 — it
  MISSES SDAR's `0.83 = reward×100` because 0.83 ∈ [0,1]); none of the 4 prior fabrication guards
  (stub/zero/metric-semantics/evidence-audit) can catch a plausible, in-range, derived metric.
- Tests: `tests/rlm/test_eval_provenance.py` — 68 pass (incl. the SDAR reward×100 case, absent sidecar,
  out-of-range outcome, >64-example exact recompute, boolean-success exemption, producer round-trip).
- **v1 limitation:** "held-out vs in-sample" is self-declared (`held_out` field), not yet cross-checked
  against training task ids — the robust checks are presence + recompute + per-example range. (v2: persist
  train ids and verify disjointness.)

**S2 — guidance hardened.** `FIX 6 — REAL HELD-OUT EVAL METRIC` added to both
`runs/.cache/sdar_scope_guidance_full.txt` and the new headline-9 file: compute EM/F1/success on a
HELD-OUT slice vs gold; report a fraction in [0,1]; never ×100/×128; never `success ≡ reward`; emit the
sidecar via `record_eval`. (The exact `record_eval` call matches the S1 signature.)

**S3 — headline-9 depth A/B staged.**
- `runs/.cache/sdar_scope_guidance_headline9.txt` — the 9 SDAR headline cells only (3 models × 3 envs),
  `TRAINING_STEPS=100`, no baselines/ablations (buy depth not breadth). Note inside: add the 9 GRPO cells
  (→18) for a SEPARATE SDAR-vs-baseline run.
- Runner `scripts/sdar_gcp_optimal_run.sh` now passes through `OPENRESEARCH_EVAL_PROVENANCE_GUARD` and
  `OPENRESEARCH_REUSE_RUBRIC` (conditional, byte-identical when unset).

**Launch recipe (after `sdar_full_v2` finalizes; on-demand 4×A100, `sdar-a100-od`):**
```bash
cd /home/abheekp/openresearch
GUIDANCE_FILE=runs/.cache/sdar_scope_guidance_headline9.txt \
PROJECT_ID=sdar_headline9_v1 \
OPENRESEARCH_EVAL_PROVENANCE_GUARD=1 \
OPENRESEARCH_REUSE_RUBRIC=1 \
  setsid nohup bash scripts/sdar_gcp_optimal_run.sh > runs/_headline9_od.log 2>&1 < /dev/null &
# monitor: setsid nohup env PID=sdar_headline9_v1 INSTANCE=sdar-a100-od ZONE=us-central1-b \
#   bash scripts/sdar_min_monitor.sh > runs/_headline9_mon.log 2>&1 < /dev/null &
```
With the guard ON, the implementer is FORCED to emit verifiable `eval_provenance.json` per cell (or the
cell is vetoed into repair) — so the headline-9 numbers are honest held-out measurements, not reward×100.

**Compare the depth A/B** (does more training raise the HONEST metric?): the 9 sdar cells of
`sdar_full_v2` (20 steps, inflated accuracy) vs `sdar_headline9_v1` (100 steps, S1-verified). Use
`scripts/ab_compare.py --paper 2605.15155` and/or compare per-cell `eval_provenance.json` metric_value.

**Validation discipline before flipping any default-ON:** the repo rule is ≥3 paired SDAR A/B + the
grader-σ gate. S1 ships default-OFF; the headline-9 run is its first real exercise.

---

## 10. Harness-reliability suite — F2 / F1-v2 / F3 + bug fixes (2026-06-24, "fix all / end-to-end")

Implemented the full forward Tier (§8.2–8.3), all flag-gated **default-OFF** (byte-identical when
unset), Opus-designed · Sonnet-executed · Opus-reviewed (2 robustness bugs caught + fixed in review).
Verified: parse + import clean; **136 new-suite tests + 246 verdict/guard regression tests pass**, no
regressions. The live `sdar_full_v2` run was NOT touched.

**Through-line:** every silent-break mode now has a deterministic, evidence-based gate — fitness is
the on-disk evidence, never the LLM grade — because guidance alone is ignored (the agent emitted
`accuracy=reward×100` *and* skipped FIX 5's no-checkpoint rule).

**F2 — env-liveness (`OPENRESEARCH_ENV_LIVENESS_GATE`, the WebShop fake-zero fix).** Root cause:
`webshop_env` emits `info["unavailable"]` when the server is down, but the agent's trainer swallows it
into `reward=0, status:ok` and no harness postflight read it back. Fix: harness-owned
`agentic_rollout.rollout_episode` writes a per-cell `env_health.jsonl` (n_turns / unavailable / served)
the agent can't suppress; the cells route detects an env that served **0 episodes** and adds a verified
**`env_setup_failed` Exclusion** to `ctx.env_setup_exclusions`, which the EXISTING `_apply_operator_scope`
folds into the scope — fairly *excluded* from the strict score, never scored as a fake 0 that pollutes
the grade / a baseline Δ. `env_name` canonical attrs added to the 3 envs so the exclusion item matches
the rubric leaf text. Code fix: `webshop_env.reset` now finishes-on-unavailable so `env.done`/`last_info`
carry the signal immediately. `backend/agents/rlm/env_liveness.py` + `tests/rlm/test_env_liveness.py` (33).
> ⚠️ End-to-end: F2 makes WebShop's failure HONEST (a fair gap). Making WebShop actually SERVE episodes
> is a separate server-provisioning task (install `web_agent_site` / set `WEBSHOP_URL`) — the handoff
> already deprioritized it ("skip + record the gap, don't fabricate").

**F1-v2 — verified held-out eval (extends the S1 guard).** `record_eval(..., train_ids=…)` persists the
training task-ids; the guard vetoes when eval ids ∩ train ids ≠ ∅ (in-sample, not held-out) — catching
the honest-mean-over-training-rollouts case S1's recompute can't. Surfaced stable task-ids: ALFWorld
`extra.gamefile` → `last_info["task_id"]`, Search-QA `f"{source}:{i}"`. `tests/rlm/test_eval_provenance.py`
(73, +5).

**F3 — no-learning-signal verdict (`OPENRESEARCH_NO_LEARNING_SIGNAL_GATE`).**
`no_learning_signal.detect_no_learning_signal(code/)` reads the persisted reward/loss curves; if EVERY
judgeable cell is flat (reward never rose AND loss never descended), `_finalize` forces
`replication_verdict="inconclusive"` (threaded through `write_final_report_rlm` → `compute_and_attach` →
`compute_reproducibility_verdict`) + emits a `no_learning_signal` warning. Conservative: ANY learning
leaf → not flagged. Deterministic — curves, never the grade. `tests/rlm/test_no_learning_signal.py` (30).

**Bug fixes (found in recon, fixed):**
- **B2 (real — a default-ON guard was a silent dead letter):** `_degenerate_training_violation` read
  `per_model[model]["status"]`, but the cells-route shape is `{env:{baseline:leaf}}` (no top-level
  status) → it skipped every model on every SDAR-style run. Fixed via `_model_result_leaves` (flat +
  nested) with **model-level** condition-(2) — a model with any non-zero leaf is never flagged, so it is
  byte-identical to the monolithic semantics (no new false-positive on the live run; the 246 regression
  tests confirm).
- **B3:** `mine_lessons` was called twice in `_finalize` (E1-reconcile duplicate) → deduped to one
  post-write call.
- **A5 (verified NON-issue):** `ctx.env_setup_exclusions` already `field(default_factory=list)` — the
  proposed fix was a no-op; skipped.
- **Checkpoint bloat (flagged, not code-fixed):** the agent ignored FIX 5 → 6 GB `model_checkpoint.pt`
  per cell (162 GB on the live disk; safe under the 1 TB disk + `gc_runs.py`). A future harness contract
  should forbid per-cell `model.save` (same "guidance ignored → needs a contract" pattern).

**Enable on the staged headline-9 run (§9)** as their first real exercise:
`OPENRESEARCH_ENV_LIVENESS_GATE=1 OPENRESEARCH_NO_LEARNING_SIGNAL_GATE=1 OPENRESEARCH_EVAL_PROVENANCE_GUARD=1`
(+ the F1-v2 disjointness rides the eval-provenance flag). All default-ON flips need the ≥3-paired-A/B + σ gate.
