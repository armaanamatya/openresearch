# SDAR campaign live-validation handoff (2026-07-02) — rerun in a new session

> **Doc status:** Operator/session handoff · written mid-live-validation by the
> implementation session that built the ReproductionCampaign (Phases B+C). The
> next session's job: switch the root orchestrator to the new **claude-sonnet-5
> Foundry deployment**, relaunch the SDAR campaign, and drive it to an honest
> terminal — ideally `REPRODUCED`.

## 0. First actions in the new session (do these before anything else)

1. **A campaign may still be RUNNING and a GPU VM may still be BILLING.**
   - VM: `sdar-2model-a` (us-central1-a, a2-ultragpu-4g = 4×A100-80GB, ~$21/hr
     on-demand, project `deepinvent-ext-ut`).
   - Check: `gcloud compute instances describe sdar-2model-a --zone us-central1-a --format='value(status)'`
   - Campaign #3 (pid 8650 on the VM, project `prj_09047604e591d969`, log
     `runs/_campaign_sdar3.log`) was in `attempt_loop` attempt 1 when this doc
     was written. Its rails: $19 LLM / $255 GPU / 53 GPU-h / 6 attempts / 12h
     campaign wall. **The campaign stops itself; the VM does NOT.** If you are
     not actively continuing: `gcloud compute instances stop sdar-2model-a --zone us-central1-a`
     (boot + cache disks persist; restart is cheap, subject to A100 stockout).
   - A throwaway local campaign (All-CNN `prj_0a3202fc187bb692`, laptop) may
     also be mid-loop — it is capped at ~$8 LLM and CPU-only torch; kill freely:
     `pgrep -f "backend.cli campaign 1412" | xargs kill`.
2. Read state: `runs/prj_09047604e591d969/campaign/attempts.jsonl` (on the VM)
   is the append-only truth: rows `launched`/`assessed`/`decided` keyed by
   attempt_n. `campaign_run1/`, `campaign_run2/` siblings hold the two finished
   campaigns' ledgers + reports.
3. This branch (`reconcile/grounded-self-improvement-on-main`, remote
   **deepinvent** only) carries everything; the VM tree is an **rsync-staged
   copy, NOT a git clone** — sync with rsync (see §6 gotcha #1).

## 1. What exists (built + committed this session)

Commit `f92fe1d4` (+ the live-fix commit after it): the full
**ReproductionCampaign** — the deterministic repeat-until-reproduced outer loop
(INIT → UNDERSTAND → PLAN→LAUNCH→AWAIT→ASSESS→DISTILL→DECIDE →
`REPRODUCED`/`CONTRADICTED`/`INFEASIBLE`/`EXHAUSTED`) + the Phase-C gated
harness self-edit tier. Spec:
`docs/superpowers/specs/2026-07-01-reproduction-campaign-and-self-improving-harness-design.md`
(v2, Codex F1–F16 all implemented). Day-to-day rules: CLAUDE.md "Reproduction
campaign" block. Architecture: system_overview.md campaign section. ~600
hermetic tests; full suite green at the pre-change baseline (16 env-dependent
failures on this WSL box, unrelated).

Key modules (all `backend/agents/rlm/` unless noted):
`reproduction_campaign.py` (fail-CLOSED spend ledger: write-ahead intent rows,
halt-on-unwritable, torn-tail repair, resume protocol), `campaign_policy.py`
(split money meters → real knobs; wall co-tightening enforces GPU-hours;
guard-filtered champion; typed novelty), `attempt_assessment.py` (deterministic
trust reader; validator absence/staleness = quarantine),
`campaign_directives.py` (clean-context: transcript paths fail the build),
`attempt_driver.py` (Live/Unified/Paired drivers; force-quarantine kills the
warm-retry heuristic; seed marker `campaign/seed_staging.json`),
`campaign_composition.py` (build_campaign; the ONLY place stages are wired),
`understanding_gate.py`, `campaign_report.py`, `harness_self_edit.py` (+
`self_edit_surface.json`), `doomed_run_comparator.py`,
`backend/services/runs/attempt_isolation.py::force_archive_incomplete`,
`run_spec_contract.py`. CLI: `python -m backend.cli campaign <paper>
--max-llm-usd X --max-gpu-usd Y --max-gpu-hours Z [...] --resume`. Exit codes:
0 terminal · 2 paused · 3 MONEY-HALT.

## 2. Live-validation results so far (3 campaigns, all on prj_09047604e591d969)

**Campaign #1** (`campaign_run1/`): EXHAUSTED{max_attempts}, 3 attempts, $2.97
LLM / $0 GPU. Proved live: write-ahead intent before launch, force-quarantine
at every launch, crash-resume (re-attach + assess-from-disk), validator-absence
soft-quarantine (F4), lineage arms (runner_up → fresh), capsules → next
directives, `campaign_report.md` deliverable with trust columns + evidence
trajectory (4/6 → 5/6). Attempt 3's fabricated metric was **hard-quarantined by
the eval-provenance guard**.

**Campaign #2** (`campaign_run2/`): EXHAUSTED{max_attempts}, 4 attempts, $2.30
LLM / 0.07 GPU-h. A1 trained ~3.6 GPU-min, grade **vetoed by the grok validator
panel**; A2 preflight_blocked; A3 cell_execution_error with a CLEAN guard
envelope (became champion-so-far, stamped in the terminal); A4 **preflight AST
guard hard-blocked a hardcoded `success_rate` literal** — fabrication refused
before a single GPU-second.

**Campaign #3** (running at handoff): same project, `campaign/` live dir.
Change: **executor=sonnet** (via the VM's `CLAUDE_CODE_OAUTH_TOKEN` in
`~/openresearch/.env`), root/grader/verifier=foundry(grok-4.3), 6 attempts,
extended paper hint. Attempt 1 was mid-implement (32 files in `code/`, Sonnet
writing a real tree — much deeper than foundry's stubs).

**The verdict so far:** the harness is flawless and honest — every fabrication
caught, every failure classified, money accounting exact, terminals honest.
The reproduction bottleneck is **per-attempt agent quality**, which is why the
root/executor upgrade below is the next lever.

## 3. Bugs found LIVE and their fixes (all committed)

1. **`--sandbox` not forwarded to children** → child fell to repo default
   (runpod) and died at preflight on the VM. Fixed in
   `campaign_composition._enforcement_mapping` (+ pinned test
   `test_plan_attempt_sandbox_reaches_child_argv`).
2. **Non-login SSH PATH** → agent subprocesses resolved `/usr/bin/python3`
   (no torch). Fix is OPERATIONAL: launch the campaign with
   `PATH=/home/abheekp/openresearch/.venv/bin:$PATH` (see the recipe, §5).
   The VM base venv has torch 2.12.1+cu130.
3. **DISTILL miners silently no-op'd**: the memory flags
   (`OPENRESEARCH_NEGATIVE_LESSONS/POSITIVE_RECIPES/EXPERIENCE_MEMORY`) lived
   in the child profile only; the CAMPAIGN process env (where `_distill_impl`
   runs) lacked them. Operational fix: set them in the campaign process env
   (recipe §5). **Code follow-up (open):** INIT should apply the profile's
   flags to the campaign process env.
4. **Foundry (grok) as executor writes stubs** (hardcoded metrics — matches
   memory `project_foundry_gptchat_root_not_executor`). Fix: executor=sonnet
   in the run-spec `models` key (campaign #3 onward).
5. **rsync gotcha:** `--exclude runs` also matches `backend/services/runs/` —
   sync the VM with explicit sources and NO bare `runs` exclude.

Also shipped: `PAPER_HINTS["2605.15155"].guidance` extended (+robust-exit,
no-per-cell-checkpoints, 8-bit Adam, `record_eval` held-out eval sidecar,
non-null top-level scalars). 50 hint tests green. The hint channel is how
full-scope technical guidance reaches every attempt (campaign directives are
deliberately transcript-free and carry only structured repairs).

## 4. THE NEW LEVER — claude-sonnet-5 on Azure Foundry as root (and executor)

Screenshot `sonnet5.png` (repo root, laptop): deployment **claude-sonnet-5**
(model version 2, GlobalStandard, 1M TPM / 1000 RPM) on resource
`appradhann-4738`:

- **Endpoint:** `https://appradhann-4738-resource.services.ai.azure.com/anthropic/v1/messages`
- **Protocol: ANTHROPIC Messages API** — NOT the OpenAI-compatible `/openai/v1`
  surface the existing `azure-foundry` provider speaks. Do NOT point
  `AZURE_FOUNDRY_*` at it.
- **Key:** in the Azure portal (Foundry → appradhann-4738 → Models →
  claude-sonnet-5 → Details), and visible in `sonnet5.png`. Never commit it;
  put it in the VM's `~/openresearch/.env`.
- **Provisioning state was `Creating` at capture (Jul 2 11:22 AM)** — verify it
  is Succeeded before wiring (Playground tab or a curl to /v1/messages).

Integration paths, in order of preference:

**Path A (try first — likely zero/near-zero code):** the Anthropic SDK and the
`claude` CLI honor `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`. On the VM set:
```
ANTHROPIC_BASE_URL=https://appradhann-4738-resource.services.ai.azure.com/anthropic
ANTHROPIC_API_KEY=<the Foundry key>
```
then root `--model claude` / `OPENRESEARCH_RLM_ROOT_MODEL=claude`, executor
token `sonnet` (SDK auto-resolves API-key mode), and grader/verifier can stay
foundry or move to `sonnet`. Two things to verify before trusting it: (i) the
root registry's `claude` entry (`backend/agents/rlm/models.py`) — which model
id it pins and whether it honors base-url env (the raw-HTTP root client may
need the base URL threaded); (ii) the deployment name `claude-sonnet-5` must be
what the endpoint expects as `model` in the request body. Smoke it with a
5-line curl before any campaign.
**Beware the repo gotcha:** a non-empty `ANTHROPIC_API_KEY` changes sub-agent
auth resolution everywhere (auto prefers API key over OAuth) — with the
Foundry base URL that is exactly what you want, but it means the OAuth
executor path from campaign #3 is superseded; remove/adjust
`llm_auth_strategy` expectations accordingly.

**Path B (if the root's raw-HTTP client ignores ANTHROPIC_BASE_URL):** add a
root registry entry (`azure-anthropic` / `sonnet5-foundry`) in
`backend/agents/rlm/models.py` mirroring the `claude` entry with
base_url+key+model-id from env (`AZURE_ANTHROPIC_ENDPOINT/_API_KEY/_DEPLOYMENT`
— mirror `foundry_endpoint.py`'s resolver pattern), and a matching
`grader_transport`/`role_models` token so all five tiers can select it. This is
the durable fix; ~1-2h with tests, patterned exactly on the 2026-06-17
azure-foundry unification (CLAUDE.md documents that precedent end-to-end).

Why this matters: the root has been grok-4.3 (works but "not paper-validated";
foundry executor stubs), and claude-oauth-Sonnet-4.x as root had the
degenerate-loop history. **Sonnet-5 as root + executor via a funded 1M-TPM
endpoint is the best-quality configuration this repo has ever had available**
— and the campaign's guards + validator stay the backstop regardless.

## 5. The exact relaunch recipe (VM)

```bash
# 0. VM up + cache disk mounted (idempotent):
gcloud compute instances start sdar-2model-a --zone us-central1-a   # stockout? try sdar-2m / sdar-1model / sdar-a100-8g (see §6)
gcloud compute ssh sdar-2model-a --zone us-central1-a --command \
  'sudo mount /dev/disk/by-id/google-sdar-cache /mnt/sdar-cache 2>/dev/null; ls /mnt/sdar-cache'

# 1. Sync code from the laptop checkout (VM tree is NOT a git repo):
rsync -az -e "ssh -i ~/.ssh/google_compute_engine" \
  --exclude "**/__pycache__" --exclude node_modules \
  backend configs scripts tests pyproject.toml CLAUDE.md \
  abheekp@35.253.12.150:/home/abheekp/openresearch/     # IP may change on restart: gcloud compute instances describe ... EXTERNAL_IP

# 2. If starting FRESH (prior campaign terminal): on the VM,
#    mv runs/prj_09047604e591d969/campaign runs/prj_09047604e591d969/campaign_runN
#    (resume instead: keep campaign/ and add --resume to the command below)

# 3. Launch (adjust root/executor per §4; this was campaign #3's shape):
gcloud compute ssh sdar-2model-a --zone us-central1-a --command '
cd ~/openresearch
env -u OPENAI_API_KEY \
  PATH=/home/abheekp/openresearch/.venv/bin:$PATH \
  HF_HOME=/mnt/sdar-cache/hf \
  ALFWORLD_DATA=/mnt/sdar-cache/data/alfworld \
  WEBSHOP_DATA_DIR=/mnt/sdar-cache/SDAR/agent_system/environments/env_package/webshop/webshop \
  SEARCH_QA_INDEX_DIR=/mnt/sdar-cache/data/searchR1 \
  OPENRESEARCH_WEBSHOP_PYTHON=/mnt/sdar-cache/conda/envs/verl-webshop/bin/python3 \
  OPENRESEARCH_RLM_ROOT_MODEL=azure-foundry \
  OPENRESEARCH_VALIDATOR_BACKEND=azure-foundry \
  OPENRESEARCH_NEGATIVE_LESSONS=1 OPENRESEARCH_POSITIVE_RECIPES=1 OPENRESEARCH_EXPERIENCE_MEMORY=1 \
  setsid nohup .venv/bin/python -m backend.cli campaign 2605.15155 \
  --max-llm-usd 19 --max-gpu-usd 255 --max-gpu-hours 53 \
  --max-attempts 6 --wall-clock-s 43200 \
  --mode unattended --sandbox local --billing-sandbox gcp \
  --gpu-usd-per-hr 5.25 --est-gpu-hours 8 \
  --paper-hint 2605.15155 \
  --run-spec configs/sdar_campaign_run_spec.json \
  > runs/_campaign_sdarN.log 2>&1 < /dev/null &'
```
The VM-side profile `configs/sdar_campaign_run_spec.json` (already on the VM;
`models` currently `executor=sonnet,grader=foundry,verifier=foundry`) carries
the full guard suite + SDAR paths. It must NEVER contain the driver-owned keys
(`OPENRESEARCH_SEED_BEST_ATTEMPT/TARGET_BEST_FLOOR/BASELINE_EXTRA_GUIDANCE/
MAX_RUN_GPU_USD`) — INIT fail-closes at $0 if it does.

Monitoring one-liners (from the laptop):
```bash
ssh -i ~/.ssh/google_compute_engine abheekp@<IP> 'cd ~/openresearch/runs/prj_09047604e591d969 && \
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | paste -sd" | "; \
  grep -c "" campaign/attempts.jsonl; tail -3 runs/_campaign_sdarN.log'
# per-attempt narrative: the assessed/decided rows in campaign/attempts.jsonl
# deliverables at terminal: campaign_report.md (+ campaign/champion_final_report.*)
```
Steering: `POST /runs/prj_09047604e591d969/campaign/messages`
`{"op":"set_mode","mode":"checkpoint"}` (or edit
`campaign/user_messages.jsonl` directly on the VM — one JSON per line).

## 6. GCP inventory + gotchas

- VMs (all stop-able, disks persist): `sdar-2model-a` (a, 4×A100-80, THE one),
  `sdar-2m` (a, same shape, was STOCKOUT), `sdar-1model` (c, 4×A100-80),
  `sdar-a100-8g` (c, 4×A100-40), `sdar-a100-od` (b, 1×A100-40). Zone-a
  stockouts are common; try siblings.
- Cache disk `sdar-cache-a` stays attached to sdar-2model-a as device
  `sdar-cache`; after every VM start it needs the `mount` (recipe step 0).
  Contents: `hf/` (Qwen weights), `data/` (alfworld, searchR1 132GB index),
  `conda/envs/verl-webshop`, `SDAR/` (authors' repo).
- The VM `.env` already has: `AZURE_FOUNDRY_*` (grok-4.3), `CLAUDE_CODE_OAUTH_TOKEN`
  (headless Sonnet OAuth), empty `ANTHROPIC_API_KEY` (until you add the
  Foundry key per §4), a possibly-DEAD `OPENAI_API_KEY` (always launch with
  `env -u OPENAI_API_KEY` — shell/env wins over .env in this repo).
- `gcloud compute instances update --max-run-duration` is NOT supported by the
  installed gcloud — there is NO hard VM billing ceiling; the campaign wall +
  your monitoring + explicit `instances stop` are the belts.
- ssh occasionally 255s transiently; direct `ssh -i ~/.ssh/google_compute_engine abheekp@<IP>`
  is more reliable than `gcloud compute ssh`.

## 7. Open follow-ups (none block a rerun)

1. INIT should apply the profile's `OPENRESEARCH_*` flags to the campaign
   process env (kills live-bug #3's operational workaround).
2. `campaign_composition._decide_impl`'s `est_usd = est_gpu_hours × rate`
   under-counts multi-GPU (misses ×gpu_count) — budget floor fires late; the
   envelope co-tightening still caps actual spend.
3. Pre-existing: `recipe_library.admit_recipe`'s evidence-gate key is never
   written by `report.py` → recipe admission can never fire in production.
4. Rubric-canary → ASSESS guard_flags wiring (helper `canary_tripped()` is
   shipped + tested; one line at the assessment layer).
5. Phase-C replay harvest can't reconstruct un-persisted PolicyConfig knobs
   (documented; skips budget_floor cases).
6. `preflight_blocked` is not in `lesson_distiller.CORRECTABLE` — recurring
   preflight blocks produce capsules but never a promoted lesson.
7. Frontend: campaign event payload interfaces were shape-corrected; no UI
   panel consumes them yet (leaderboard column ships).
8. Root registry has no Anthropic-on-Foundry entry (§4 Path B) — the durable
   integration for claude-sonnet-5.

## 8. Money accounting (this validation, all sources)

- Campaign #1: $2.97 LLM (grok) + ~40 min VM ≈ $14 GPU-idle.
- Campaign #2: $2.30 LLM + ~25 min VM ≈ $9.
- Campaign #3: rails $19/$255/53h/12h — read its ledger for actuals.
- Sonnet executor rides the Claude subscription (OAuth, $0 per-token) until
  the Foundry key path (§4) replaces it.
- Local All-CNN campaign: <$2 LLM, $0 GPU.
