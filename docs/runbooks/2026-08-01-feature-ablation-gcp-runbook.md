<!-- doc-meta: status=current; authored=2026-08-01 -->
# Feature-ablation on GCP single-VM — operator runbook

Per-feature scores for the Tree-A features, on the GCP single-VM path. Builds on the
canonical recipe [`2026-07-22-gcp-vm-e2e-run-procedure.md`](2026-07-22-gcp-vm-e2e-run-procedure.md);
read that first. This adds: the arm matrix, per-arm flags, and the paired comparison.

## What this measures (and does NOT)
- **Measures:** each of the 7 Tree-A features' marginal score contribution + an all-combined
  arm, vs a fixed honest baseline. Arms/flags: `configs/ablation/{baseline_run_spec.json,arms.json}`.
- **Does NOT measure Tree-B** (freeze/branch/revive/true-kill) — those produce **no score**
  until the A1/A2 checkpoint-emission + per-branch-isolation build lands. Out of scope here.

## LLM auth — API keys ONLY, NEVER OAuth (read `2026-08-01-remote-run-llm-auth.md`)
> ⛔ **NEVER use OAuth** (`--model claude-oauth`, `CLAUDE_CODE_OAUTH_TOKEN`, `claude login`) —
> operator directive 2026-08-01. API keys only.

The sanctioned remote surface is **`--model sonnet-foundry`** — real Claude via
`AZURE_FOUNDRY_API_KEY` (an API key → copies to the VM `.env`), now thinking-patched
(`_anthropic_thinking_patch.py`; `claude-sonnet-5` defaults to extended thinking, which crashed
the root + produced rubric-less runs on 2026-08-01 until fixed):
```
# stage AZURE_FOUNDRY_ENDPOINT/_API_KEY/_DEPLOYMENT into the VM .env (NO OAuth token), then:
--model sonnet-foundry --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry
```
Alternative: fund an **`ANTHROPIC_API_KEY`** and use **`--model claude`** (standard non-OAuth Claude).
Do NOT use `grok` (not ML-validated → infra-`failed`), `--model gpt-5` (dead `sk-svcacct-` key), or
`--model azure` (`AZURE_OPENAI_ENDPOINT`/`_DEPLOYMENT` empty). The `claude-agent-sdk` executor is a
separate subprocess path — smoke-validate a full `sonnet-foundry` run before fanning out.
Caveats: re-run `scripts/calibrate_grader.py` if you switch grader model; Foundry LLM spend is
invisible in `cost_ledger.jsonl` (track in Azure Cost Management). The ablation baseline no longer
hardcodes `GRADER_BACKEND` — `--models grader=…` drives it.

## Machine type & cost
These papers (All-CNN/Adam/ResNet, CIFAR-scale) fit on **L4** — `g2-standard-8`, ~$0.70/hr,
matching the RTX-4090 the June best-runs used. A100 is unnecessary (and ~5× pricier).
```
export OPENRESEARCH_GCP_GPU_MACHINE_TYPE=g2-standard-8   # 1×L4-24GB
```
Rough cost on L4: screen 27 runs (~$80) + confirm ~45 runs (~$135), ×1.6 retry ≈ **~$350 GPU**
(vs ~$770 on A100). Foundry LLM is separate.

## STEP 0 — validate BEFORE provisioning (no VM spend)
The #1 GCP failure is paying to spin up a VM that dies on dead LLM auth or no GPU capacity.
```bash
# a) Foundry LLM live? (cheap HTTP ping — must be 200 before any run)
python -m backend.cli reproduce --help >/dev/null && echo "CLI OK"
# b) GPU quota + a zone with L4 capacity (stockout is real):
gcloud compute regions describe us-central1 --format='table(quotas.filter("metric~NVIDIA").flatten())'
# c) per-arm run-specs all valid:
for a in $(python scripts/merge_run_spec.py --list); do python scripts/merge_run_spec.py "$a" --validate >/dev/null && echo "$a OK"; done
```

## STEP 1 — SMOKE (one arm, one paper, L4) — NON-NEGOTIABLE GATE
Run exactly one arm end-to-end and confirm a **real scored** `final_report.json`
(`verdict` ∈ {reproduced, partial} with a rubric score — NOT infra-`failed`, NOT 0-receipt).
If the smoke doesn't produce a real score, STOP — the fan-out would just burn money.
Use the `baseline` arm on ResNet (fastest). Follow the "Validated direct recipe" in the
2026-07-22 runbook, with:
- `python scripts/merge_run_spec.py baseline > /tmp/arm_baseline.json` and `scp` it to the VM,
- run `reproduce 1512.03385 --sandbox local --run-spec /tmp/arm_baseline.json --model opus-foundry --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry --project-id smoke_baseline`,
- pull `runs/smoke_baseline/final_report.json` + `rubric_evaluation.json`, confirm the score, DELETE the VM.

## STEP 2 — one arm launch (the unit the fan-out repeats)
> ⚠️ **INSTALL TORCH ON EVERY VM.** After `uv pip install -r backend/requirements.txt` (orchestrator
> deps only), you MUST also install the ML training deps or every training cell fails
> `ModuleNotFoundError: No module named 'torch'` → `cell_execution_error` → **null score**
> (hit 2026-08-01):
> ```bash
> uv pip install --python .venv/bin/python torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121
> ```
> (The harness's `env_pin` local-torch-core did NOT auto-install it on the DLVM uv venv — install explicitly.)

> 🛑 **PUT THE VENV ON PATH AT LAUNCH — else torch is invisible to the cells (root-caused 2026-08-03).**
> The cell trainer resolves its interpreter with `command -v python3`, which finds SYSTEM
> `/usr/bin/python3` (no torch) — NOT the run venv. Launching `.venv/bin/python -m backend.cli` is
> **not** enough; the child cell subprocesses don't inherit the venv, so every cell fails the
> preflight import smoke (`No module named 'torch'`) → `overall:None`, no score. FIX: launch with
> the venv on PATH:
> ```bash
> cd ~/or && export PATH=$HOME/or/.venv/bin:$HOME/.local/bin:$PATH
> python -m backend.cli reproduce ...   # now `command -v python3` -> venv python3 (with torch)
> ```
> **Reliable launch mechanism = a GCE startup-script, not interactive SSH.** OS-Login/IAP
> rate-limits SSH hard under rapid calls (255s); set the launch as `--metadata-from-file
> startup-script=...` (runs on boot as the user, with the PATH export above), start the VM, and
> verify via `gcloud compute instances get-serial-port-output` (`STARTUP_LAUNCHED` marker) — all
> API, zero SSH. Reserve SSH for the single sparse `scp` that collects the score. And **poll with
> plain shell** — macOS bash 3.2 has no `declare -A`; an associative-array dedup silently misfires
> and can SIGTERM a still-running arm.

> 🛑 **CAP + TERMINATION-ACTION — the #1 killer of these runs (root-caused 2026-08-02).**
> The single-L4 ResNet cell-matrix run takes **longer than 6 h**. VMs provisioned with
> `--max-run-duration=21600s` (6 h) auto-STOPped mid-run with `compute.instances.deferredStop`
> and produced **no score** — this was misread as "host maintenance," it was the cap. RULES:
> - **`--max-run-duration=64800s` (18 h) minimum**, or omit the cap and tear down manually. Never
>   cap below the observed completion time.
> - **`--instance-termination-action=STOP`, NEVER `DELETE`** — DELETE on the first run destroyed
>   the results (`deferredDelete`). STOP preserves the boot disk so artifacts survive a stop.
> - **Poll + scp `final_report.json` the instant it appears** (below) so no completed score is ever
>   lost to a stop. Details: `docs/2026-08-01-feature-ablation-results.md` (root-cause section).

For each `(paper, arm, seed)`: provision an L4 VM with an **18 h cap and STOP termination**
(`--max-run-duration=64800s --instance-termination-action=STOP`), stage code + the merged
run-spec, and start a background pull loop that scps `final_report.json` out the moment it lands.
Then run:
```bash
python scripts/merge_run_spec.py <arm> > /tmp/arm_<arm>.json   # scp to VM
# on the VM:
OPENRESEARCH_AB_ARM=<arm> OPENRESEARCH_AB_PAIR_ID=<paper>-<seed> \
  .venv/bin/python -m backend.cli reproduce <arxiv_id> \
    --sandbox local --run-spec /tmp/arm_<arm>.json \
    --model opus-foundry --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry \
    --seed <seed> --force-single-gpu --max-wall-clock 60000 \
    --project-id <paper>_<arm>_s<seed>
# pull runs/<paper>_<arm>_s<seed>/{final_report.json,rubric_evaluation.json,experiment_arm...}; then STOP (not delete) VM
```
> ⚠️ `--max-wall-clock` (app-level) must ALSO exceed real completion time and stay **below** the VM
> `--max-run-duration` cap — `25000s` (~6.9 h) was too short; use `60000s` (~16.7 h) under an 18 h
> VM cap. Two independent timers; whichever is smaller kills the run.
Each run stamps `experiment_arm={arm, ab_pair_id}` into its report (the A/B harness).

## STEP 3 — fan-out (phased)
- **Screen (1 seed):** 9 arms × 3 papers = **27 runs**. Launch ≤ your GPU quota (L4=8) concurrently.
  Drop features whose Δ vs baseline is within grader-σ.
- **Confirm (3 seeds):** surviving arms + `all_on` × 3 papers × seeds {1,2,3}.
- One VM per run; "in parallel" = up to 8 L4 VMs at once (quota). Stagger create calls a few
  seconds apart to avoid provision collisions.

## STEP 4 — per-feature deltas
```bash
# each feature arm is paired against 'baseline' at the same <paper>-<seed>:
python scripts/ab_compare.py --pair-id <paper>-<seed>      # writes runs/_ab/<key>/ab_report.{md,json}
```
Then gate with `python -m backend.agents.rlm.asha_authority_gate` semantics: a feature "wins"
only if the paired Δ > 2·grader-σ, sign-consistent across seeds/papers, AND corroborated by
deterministic evidence (cells converged / lower error) — never the grade alone.

## MANDATORY every run
- **Journaling** (GCP rule): update `docs/progress/2026-08-01-feature-ablation-progress.md`
  ≥ every 30 min for the campaign's life — progress, failures (with log paths), infra issues,
  cost (from Azure Cost Mgmt + `gcloud`, not the ledger), next action.
- **Stray-VM check** after every run: `gcloud compute instances list --project deepinvent-ext-ut`
  — with STOP termination the VM will be TERMINATED (not gone) after its run; confirm it's not
  still RUNNING and idle-billing, then delete it once artifacts are pulled. Run
  `python scripts/gcp_vm_audit.py` on a timer.

## Honest risk notes
- **The runs that "GCP killed" were killed by our own `--max-run-duration` cap, not host
  maintenance** (root-caused 2026-08-02 via `compute.instances.deferredStop` in the audit log +
  exact-6 h timing; see `docs/2026-08-01-feature-ablation-results.md`). The real reliability lever
  is: cap ≥ completion time, STOP not DELETE, and poll-scp the score the instant it lands.
- This path has **never completed a multi-run campaign first-try** (July runs mostly failed; the
  Aug runs died on the 6 h cap above). The smoke gate (Step 1) exists to catch that before the
  fan-out spends — and the smoke MUST run under the corrected cap/termination settings.
- The generic (non-SDAR) VM launch is the **direct-recipe** steps — there is no one-command
  multi-paper campaign launcher yet; the fan-out is a scripted loop over the Step-2 unit.
