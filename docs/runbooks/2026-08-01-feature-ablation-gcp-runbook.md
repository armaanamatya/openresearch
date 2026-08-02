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
For each `(paper, arm, seed)`: provision an **auto-delete** L4 VM (`--max-run-duration=Ns
--instance-termination-action=DELETE`), stage code + the merged run-spec, run:
```bash
python scripts/merge_run_spec.py <arm> > /tmp/arm_<arm>.json   # scp to VM
# on the VM:
OPENRESEARCH_AB_ARM=<arm> OPENRESEARCH_AB_PAIR_ID=<paper>-<seed> \
  .venv/bin/python -m backend.cli reproduce <arxiv_id> \
    --sandbox local --run-spec /tmp/arm_<arm>.json \
    --model opus-foundry --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry \
    --seed <seed> --force-single-gpu --max-wall-clock 25000 \
    --project-id <paper>_<arm>_s<seed>
# pull runs/<paper>_<arm>_s<seed>/{final_report.json,rubric_evaluation.json,experiment_arm...}; DELETE VM
```
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
  — expect no lingering RUNNING instance (auto-delete should have fired). Run
  `python scripts/gcp_vm_audit.py` on a timer.

## Honest risk notes
- This path has **never completed a multi-run campaign first-try** (July runs mostly failed).
  The smoke gate (Step 1) exists to catch that before the fan-out spends.
- The generic (non-SDAR) VM launch is the **direct-recipe** steps — there is no one-command
  multi-paper campaign launcher yet; the fan-out is a scripted loop over the Step-2 unit.
