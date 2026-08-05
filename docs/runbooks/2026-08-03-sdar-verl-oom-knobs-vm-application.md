<!-- doc-meta: status=current; authored=2026-08-03 -->
# SDAR verl OOM knobs — VM-side application (ALFWorld + WebShop)

**Date:** 2026-08-03 · **Status:** current · **Operator-run** (billed VM time — needs approval).

Applies the four memory-safe knobs from the 2026-07 diagnostics
(`docs/periods/2026-07.md` §11) to the authors' launch scripts on the SDAR cache
disk. ALFWorld-3B genuinely trains and learns (`episode/success_rate`
0.383 → 0.508, peaks 0.609) then OOMs mid-FSDP-actor-update at step 79;
`run_webshop_3b_patched.sh` carries the same self-defeating
`expandable_segments` export. The in-repo mirror of these knobs is staged in
`scripts/sdar_authors_repro.sh` (`OOM_SED_EXPRS`); the actual
`run_alfworld_3b*.sh` / `run_webshop_3b*.sh` live ONLY on the VM's cache disk
at `/mnt/sdar-cache/SDAR/examples/sdar_trainer/` — they are NOT in this repo,
so they must be patched on the VM.

## ⚠️ Infra reality check (verified 2026-08-03, `gcloud` as armaan@deepinvent.ai)

The VM and disks this runbook targets **no longer exist**: `sdar-2model-a` is
deleted, and **all** `sdar-*` pd-ssd disks (incl. `sdar-cache-a`) are gone —
the conda envs, HF caches, ALFWorld data, and the authors' repo checkout went
with them. What survives:

- **`sdar-cache-snap`** — 1000 GB snapshot of `sdar-cache` (us-central1-c),
  taken **2026-07-01**. Note this predates the 2026-07-04/05 sessions, so
  `run_alfworld_3b_4gpu.sh` / `run_webshop_3b_patched.sh` are likely NOT on it;
  regenerate them from the authors' base scripts via the same sed knobs
  (in-repo mirror: `scripts/sdar_authors_repro.sh` `OOM_SED_EXPRS`).
- **`gs://deepinvent-ext-ut-sdar-runs/`** — run artifacts only (incl.
  `final_bundles/search_3b_150step_20260703.tar.gz`, the 0.456 proof). No
  envs/assets.

**Recon + knob application DONE (2026-08-03, ~$1 total):** `sdar-cache-snap`
was restored to disk **`sdar-cache-restore`** (us-central1-a, 1000 GB,
pd-balanced, **~$100/mo carrying cost while kept** — delete + re-restore from
the snapshot if training is far off), inventoried via a throwaway
`e2-standard-4` VM (created + deleted same session), and the knobs applied:

- **Present and intact:** `SDAR/` repo with the authors' 9 base trainer
  scripts; conda envs `sdar` / `verl-webshop` / `retriever` (29 G);
  `data/alfworld` (2.3 G); `data/searchR1` (**175 G** — the expensive
  re-download avoided); `hf/hub` with `Qwen2.5-3B-Instruct` + `Qwen3-1.7B`.
- **Knobs applied on-disk** to `run_alfworld_3b.sh` (`16 / 0.5 / True`) and
  `run_webshop_3b.sh` (`0.5 / True`; its micro-batch was already `8`, left
  as-is). Backups `*.bak-20260803`; pre-edit hashes in
  `logs/oom_knobs_pre_20260803.sha256`; `run_search_3b.sh` verified
  byte-identical (`ebfcb63c…`). No script contains `PYTORCH_CUDA_ALLOC_CONF`.
- **Missing (post-snapshot artifacts):** the derived
  `run_alfworld_3b_4gpu.sh` / `run_webshop_3b_patched.sh` (not needed — knobs
  are now in the base scripts; apply `trainer.n_gpus_per_node=8→4` as a
  launch-time sed on ALFWorld), **all WebShop assets** (corpus/index must be
  re-staged before WebShop can train), and 7B weights (out of smallest-two
  scope anyway).

Remaining to train: create an `a2-ultragpu-4g` VM (~$20/hr), attach
`sdar-cache-restore` at `/mnt/sdar-cache`, then run steps 5-6 below
(ALFWorld first; WebShop only after its assets are re-staged).

## The four knobs (ALFWorld + WebShop ONLY — never Search)

| Knob | From | To | Why |
|---|---|---|---|
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 32 | 16 | halve the FSDP backward peak |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.6 | 0.5 | vLLM leaves headroom for the actor update (Search's proven value) |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | False | True | Adam state → CPU |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | present | **REMOVED** | vLLM's `CuMemAllocator` raises `AssertionError: Expandable segments are not compatible with memory pool` at model load (see the command-cell exemption in `backend/agents/rlm/gpu_cell_runner.py`) — the operator's attempt-2 crashed on exactly this |

**`run_search_3b.sh` must stay byte-identical** — it is proven at **0.456**
(2026-07-03 Search-QA-3B run).

## Cost implication — operator approval required

Starting the VM bills immediately (the SDAR VM shape is 4×A100-80,
`a2-highgpu-4g`, **~$14/hr**). The edit itself takes minutes; do NOT leave the
VM running between the edit and the relaunch decision. **Stop the VM when
done.** Every command below is operator-run; nothing here is automated.

## Steps

Identity below uses the historical names (project `deepinvent-ext-ut`, zone
`us-central1-a`, VM `sdar-2model-a` on pd-ssd `sdar-cache-a`) — substitute the
current instance if it was recreated (the July VM was terminated; if no
instance mounts `sdar-cache-a`, create one and attach the cache disk first).

### 1. Start the VM (billed from this moment)

```bash
gcloud compute instances start sdar-2model-a \
  --zone us-central1-a --project deepinvent-ext-ut
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet \
  --command='ls /mnt/sdar-cache/SDAR/examples/sdar_trainer/'
```

### 2. Record pre-edit hashes (Search pin + rollback reference)

```bash
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet --command='
  cd /mnt/sdar-cache/SDAR/examples/sdar_trainer
  sha256sum run_search_3b.sh run_alfworld_3b_4gpu.sh run_webshop_3b_patched.sh \
    | tee /mnt/sdar-cache/logs/oom_knobs_pre_20260803.sha256'
```

### 3. Apply the four knobs (sed, with .bak backups)

```bash
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet --command='
  cd /mnt/sdar-cache/SDAR/examples/sdar_trainer
  sed -i.bak-20260803 \
    -e "s/ppo_micro_batch_size_per_gpu=32/ppo_micro_batch_size_per_gpu=16/g" \
    -e "s/gpu_memory_utilization=0\.6/gpu_memory_utilization=0.5/g" \
    -e "s/actor\.fsdp_config\.optimizer_offload=False/actor.fsdp_config.optimizer_offload=True/g" \
    -e "/PYTORCH_CUDA_ALLOC_CONF=expandable_segments/d" \
    run_alfworld_3b_4gpu.sh run_webshop_3b_patched.sh'
```

Expected effective diff (per file):

```diff
-    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
+    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
-    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
+    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
-    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
+    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
-export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4. Verify

```bash
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet --command='
  cd /mnt/sdar-cache/SDAR/examples/sdar_trainer
  echo "== knobs now in effect (expect 16 / 0.5 / True, and NO expandable_segments) =="
  grep -nE "ppo_micro_batch_size_per_gpu|gpu_memory_utilization|optimizer_offload|PYTORCH_CUDA_ALLOC_CONF" \
    run_alfworld_3b_4gpu.sh run_webshop_3b_patched.sh
  echo "== Search must be UNCHANGED (hash must match the pre-edit record) =="
  sha256sum run_search_3b.sh
  grep run_search_3b.sh /mnt/sdar-cache/logs/oom_knobs_pre_20260803.sha256'
```

Checks, all four must hold:
- `ppo_micro_batch_size_per_gpu=16` and `gpu_memory_utilization=0.5` in both files;
- `optimizer_offload=True` in both files — **if the grep shows NO optimizer_offload
  line at all**, the knob was absent rather than `=False`: add
  `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \` to the trainer
  arg list by hand and re-verify;
- zero `PYTORCH_CUDA_ALLOC_CONF` hits in both files;
- `run_search_3b.sh` sha256 identical to step 2.

### 5. Relaunch the two cells

Same launch shape as the prior attempts (conda env + authors' script; the
Search retrieval server is NOT needed for these two envs):

```bash
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet --command='
  cd /mnt/sdar-cache/SDAR
  export HF_HOME=/mnt/sdar-cache/hf ALFWORLD_DATA=/mnt/sdar-cache/data/alfworld WANDB_MODE=offline
  nohup /mnt/sdar-cache/miniconda/bin/conda run --no-capture-output -p /mnt/sdar-cache/conda/envs/sdar \
    bash examples/sdar_trainer/run_alfworld_3b_4gpu.sh \
    > /mnt/sdar-cache/logs/run_alfworld_3b_4gpu.log 2>&1 & echo alfworld_pid=$!'
# WebShop uses its own env (vllm 0.8.2):
gcloud compute ssh sdar-2model-a --zone us-central1-a --quiet --command='
  cd /mnt/sdar-cache/SDAR
  export HF_HOME=/mnt/sdar-cache/hf WANDB_MODE=offline
  nohup /mnt/sdar-cache/miniconda/bin/conda run --no-capture-output -p /mnt/sdar-cache/conda/envs/verl-webshop \
    bash examples/sdar_trainer/run_webshop_3b_patched.sh \
    > /mnt/sdar-cache/logs/run_webshop_3b_patched.log 2>&1 & echo webshop_pid=$!'
```

Run them sequentially if GPU budget is tight (each wants the memory headroom
the knobs just created). Watch for the step-79 wall: the ALFWorld OOM fired at
step 79 pre-fix, so a run passing ~step 100 with stable
`torch.cuda.memory_reserved` is the success signal.

### 6. Stop the VM (ALWAYS)

```bash
gcloud compute instances stop sdar-2model-a \
  --zone us-central1-a --project deepinvent-ext-ut
gcloud compute instances list --project deepinvent-ext-ut   # confirm nothing RUNNING
```

If the training runs are long-lived, either keep the VM up knowingly (~$14/hr,
operator decision) with an autostop/idle watchdog armed, or stop and resume —
NOTE the harness/authors' scripts have no mid-training checkpoint-resume; a
stop restarts training from step 0.

## References

- Diagnostics + prescription: `docs/periods/2026-07.md` §11 (ALFWorld OOM root
  cause; "Do NOT use expandable_segments" dead end).
- Harness-side handling of the expandable_segments ban:
  `backend/agents/rlm/gpu_cell_runner.py` (command cells never get the harness
  `PYTORCH_CUDA_ALLOC_CONF` default — the authors' launcher owns its CUDA
  memory config).
- In-repo mirror of the same knobs: `scripts/sdar_authors_repro.sh`
  (`OOM_SED_EXPRS`, env-overridable via `SDAR_OOM_*`).
- Proven Search-QA cell (must stay untouched):
  `configs/sdar_execute_cells_phase1.json`.
