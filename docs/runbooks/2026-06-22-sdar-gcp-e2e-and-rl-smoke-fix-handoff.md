# SDAR-on-GCP e2e + the RL-aware smoke fix — handoff (read first, self-contained)

> Authored 2026-06-22 for a fresh session (context will be cleared). Branch:
> `feat/grounded-self-improvement-harness-reliability`. Everything below is
> committed; the GCP run procedure is now a deterministic script
> (`scripts/sdar_gcp_e2e.sh`).

---

## 0. TL;DR

1. **The pre-GPU "metric reality smoke" was the GCP blocker — it's a harness bug, now FIXED.**
   It imposed a *supervised* `loss>0 + varies` assumption on SDAR's *GRPO* (RL) objective. On
   the smoke's tiny 1–2-rollout slice, GRPO's group-relative advantage collapses to 0 → the
   loss **and** gradient are legitimately ~0 → the smoke wrongly called it "loss disconnected"
   and killed the run before it reached the grid. **Fix (commit `bfd86d52`): RL-aware smoke**
   (`_is_rl_objective` — relax the 0/constant-loss + grad>0 checks for GRPO/PPO objectives;
   keep non-finite/VRAM/supervised teeth). This is why it "worked locally (smoke off), failed
   on GCP (smoke on)."

2. **A clean full SDAR *reproduction* on GCP is then gated by ROOT reliability — NOT the smoke,
   NOT the executor.** No keyless root drives it reliably: `foundry`/gpt-chat-latest churns ~50
   min on `plan_reproduction` placeholder args, `claude-oauth` degenerates (FINAL_VAR refusal
   loop, never reaches `implement_baseline`). This is the open thread (§3).

3. **Run it with one command:** `ROOT=claude-oauth PROV=spot SMOKE=0 PROJECT_ID=sdar_gcp_<id> scripts/sdar_gcp_e2e.sh run` (§4).

---

## 1. State / what's committed (branch `feat/grounded-self-improvement-harness-reliability`)

| Commit | What |
|---|---|
| `b099d637` | Smoke accuracy (token-aware loss keys: `train_loss`/`ce_loss`/… ) + grad-evidence sufficiency + crash-propagation test; foundry cred-preflight routed through `resolve_foundry_credentials` (no false abort). |
| `bfd86d52` | **RL-aware smoke** (`_is_rl_objective`) — the headline fix (§2). |
| this handoff commit | `scripts/sdar_gcp_e2e.sh` (deterministic e2e helper), the `gcp_sdar_preflight.sh` spot-flip bug fix, this doc. |

`tests/rlm` green (2981 passed); `ruff` clean. **Two distinct smoke fixes:** `b099d637` fixed
loss-*key* recognition; `bfd86d52` fixed the loss-*magnitude* premise for RL. Code:
`backend/agents/rlm/metric_reality_smoke.py` (`_is_rl_objective`, `evaluate_smoke_trace`).

---

## 2. The resolved diagnosis (smoke RL-fitness) — confirmed three ways

The user's question was "is this a bad implementation or a broken smoke?" → **broken smoke**, confirmed:
1. **The executor's own `train_cell.py` documented it** (degenerate GRPO advantage → loss 0.0 →
   "the smoke checker flags this as 'loss not connected'") and added a **non-paper entropy hack**
   (`task_loss += 1e-4 * NLL`) purely to defeat the smoke. The smoke was distorting the implementation.
2. **Smoke-off run got *past* the smoke** to `run_experiment` (proving the smoke was the first blocker).
3. **RL theory:** a 1–2-rollout slice gives no reward variance → advantage 0 → GRPO loss/grad legit ~0.
The executor (Sonnet) writes good code; it is **not** the problem. Do not swap it for ChatGPT/Kimi.

---

## 3. OPEN THREAD — why no clean full reproduction yet (ROOT reliability)

Attempts this session (all post-smoke-fix, smoke OFF unless noted):
- **spot + `foundry` (gpt-chat-latest) root, smoke ON** → false-failed the smoke (pre-fix); the run that motivated the fix.
- **spot + `foundry`, smoke OFF** → got past the smoke (XR=1, a cell started loading) but the root churned ~50 min on `plan_reproduction` placeholder-arg failures (arg-contract guard firing), then **spot-preempted** ~1 h in.
- **on-demand a2** → **STOCKED OUT** in us-central1-b (persistent this whole session).
- **spot + `claude-oauth` root, smoke OFF** → **degenerated** (~10 min): root read the paper then looped `FINAL_VAR` without ever calling `implement_baseline` (CELLS=no); detector aborted at `DEGENERATE_REFUSAL_THRESHOLD=3`.

**Key correction (do NOT repeat my earlier mistake):** this is **NOT** an auth/funded-key
problem — `claude-oauth` authenticated fine (`Claude OAuth subscription detected`). A funded
`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` would just be a *different, more reliable* root, not a "fix"
for claude-oauth.

**The real unknown: why does `claude-oauth` reproduce SDAR FINE LOCALLY but degenerate as root on
GCP?** Unresolved. Ranked hypotheses + how to test:
1. **Heavy GCP flag block vs vanilla local (most likely / user's instinct).** The GCP run loads
   ~15 grounded-self-improvement/actor-critic guard flags (validator, evidence-gate, arg-contracts,
   forced-iteration, …) via `runs/.cache/sdar_gcp.env`; a local run is almost certainly vanilla
   `reproduce 2605.15155 --model claude-oauth`. **DISCRIMINATING EVIDENCE NEEDED: ask the user for
   their exact local command + flags.** Then re-run on GCP with *minimal* flags (match local).
2. **`OPENRESEARCH_OAUTH_AUTODRIVE=1` (purpose-built lever, default OFF, experimental).** On the
   degenerate event, instead of aborting it **nudges the root toward `implement_baseline`**
   (`run.py:1021`, `_on_degenerate`). Caveat: issues a *directive*, not a guaranteed primitive call
   — may help, not a guaranteed cure. `scripts/sdar_gcp_e2e.sh` exposes it via `AUTODRIVE=1`.
3. **Non-determinism** — claude-oauth-as-root degenerates *sometimes* (documented known failure
   mode; the harness even prints the warning). A retry might just work.
4. **Rate-limit** — root + executor + grader + verifier all on the one OAuth subscription.

The detector itself is **not** the cause: CELLS=no means the root genuinely never implemented;
without the detector it would loop to the 16-refusal cap with the same 0-cell result.

**Mechanism refs:** `forced_iteration.py` (degenerate counter, `on_degenerate_refusal_loop`),
`run.py:863` (`OPENRESEARCH_OAUTH_AUTODRIVE`), `run.py:1013-1126` (the callback), `root_progress.py`.

### Recommended next experiments (cheapest-first)
1. **Get the user's local run command/flags**, then `SMOKE=0 ROOT=claude-oauth` with a *trimmed*
   `sdar_gcp.env` (drop the heavy guards) → isolates whether a flag flips claude-oauth into the loop.
2. `AUTODRIVE=1 ROOT=claude-oauth PROV=spot scripts/sdar_gcp_e2e.sh run` → does the nudge rescue it?
3. If a funded key is available: `ROOT=gpt-5` (needs `OPENAI_API_KEY`) or `ROOT=claude` (needs
   `ANTHROPIC_API_KEY`) — the reliable-root path; on-demand to avoid preemption (poll for capacity).

---

## 4. Deterministic run procedure — `scripts/sdar_gcp_e2e.sh`

Wraps the VM flips, start, sync, env overrides, launch, monitor, inspect, teardown. Params via env:
`ROOT` (claude-oauth|foundry|gpt-5|claude), `PROV` (spot|ondemand), `SMOKE` (0|1),
`PROJECT_ID` (fresh each run — never reuse a completed dir), `AUTODRIVE` (0|1).

```bash
# Recommended: smoke-off full grid, claude-oauth (true local config), spot, all-in-one:
ROOT=claude-oauth PROV=spot SMOKE=0 PROJECT_ID=sdar_gcp_$(date +%s) \
  scripts/sdar_gcp_e2e.sh run            # = up + setenv + launch + monitor

# Step-by-step (for debugging):
ROOT=claude-oauth PROV=spot SMOKE=0 PROJECT_ID=sdar_gcp_x scripts/sdar_gcp_e2e.sh up      # flip a2/spot, start, sync
ROOT=claude-oauth SMOKE=0 PROJECT_ID=sdar_gcp_x AUTODRIVE=1 scripts/sdar_gcp_e2e.sh setenv # overrides on the VM
PROJECT_ID=sdar_gcp_x ROOT=claude-oauth PROV=spot scripts/sdar_gcp_e2e.sh launch          # GREEN-gated detached launch
PROJECT_ID=sdar_gcp_x scripts/sdar_gcp_e2e.sh monitor                                      # GPU-train/terminal/preempt/degenerate
PROJECT_ID=sdar_gcp_x scripts/sdar_gcp_e2e.sh inspect                                      # CPU-flip, pull -> /tmp/sdar_inspect, stop
scripts/sdar_gcp_e2e.sh down                                                               # stop (halt billing)

# On-demand (no preemption; usually STOCKED OUT — this polls until it frees, then up+sync):
PROV=ondemand scripts/sdar_gcp_e2e.sh poll-ondemand
```

Monitor success signal = **GPU util > 60** (real grid reached). Other exits: DEG (root degenerate),
TERMINAL (run_complete → `inspect`), VM preempted. To background long monitors/polls, run the script
under `nohup`/your background runner.

**Pre-req on a fresh session:** `export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud` (gcloud auth =
`abheek@deepinvent.ai`). The script defaults to project `deepinvent-ext-ut`, zone `us-central1-b`,
instance `sdar-a100-od`. The `claude-oauth` path needs `CLAUDE_CODE_OAUTH_TOKEN` in the synced `.env`
(already present; `setenv` prints whether it's there).

---

## 5. GCP operational gotchas (learned the hard way; encoded in the script)

- **Instance:** `sdar-a100-od`, us-central1-b, `a2-highgpu-4g` (4×A100), warm Qwen/ALFWorld caches
  (machine image `sdar-mi-20260620`). Currently **TERMINATED** (no billing; ~$8–12 total spent this
  session). Verify: `scripts/sdar_gcp_e2e.sh status`.
- **Machine-type / provisioning flips (the #1 time-sink):**
  - `set-machine-type` requires the instance **TERMINATED**.
  - a2 (GPU) **must** use `--maintenance-policy=TERMINATE` (can't MIGRATE).
  - **STANDARD→SPOT needs BOTH `--preemptible` AND `--provisioning-model=SPOT`** (each alone errors:
    "preemptible=false and provisioning_model=SPOT is contradicting" / "for preemptible, only allowed
    model is SPOT"). Do **not** add `--instance-termination-action` with `--preemptible`. (This was a
    real bug in `gcp_sdar_preflight.sh::ensure_provisioning_model`, fixed this session.)
  - **SPOT→STANDARD needs `--no-preemptible --provisioning-model=STANDARD --clear-instance-termination-action`**.
  - To inspect the disk cheaply: flip to `e2-standard-4` **as spot** (TERMINATE is allowed for a
    preemptible e2; e2-STANDARD would need MIGRATE). The script's `inspect` does this.
- **Capacity:** on-demand a2 has been **STOCKED OUT** in us-central1-b all session; **spot has
  capacity but preempts** (a fast root beats the window — hence claude-oauth/gpt-5 over gpt-chat).
  Other regions can't run 4×A100 without a quota bump (A2_CPUS=12 elsewhere vs 48 in us-central1).
- **Cold-boot SSH race:** first SSH after start often "connection refused"; poll `ssh true` until it
  answers (the script does).
- **Config channels:** the run reads `runs/.cache/sdar_gcp.env` on the VM (full guard block + the
  `setenv` overrides — LAST assignment wins); `sync` excludes `runs/` so that file survives. The
  launch's `run_spec.json` carries only a subset; the root is set by `--model $OPENRESEARCH_SDAR_ROOT`.
- **Cost safety:** always `SMOKE`-off runs set `NO_AUTOSTOP=0` → self-stop on completion/preempt.
  After any inspect, confirm `status` = TERMINATED.

---

## 6. Files
- `backend/agents/rlm/metric_reality_smoke.py` — the smoke (`_is_rl_objective`, `evaluate_smoke_trace`).
- `scripts/sdar_gcp_e2e.sh` — deterministic e2e helper (NEW).
- `scripts/gcp_sdar_preflight.sh` — sync/launch/monitor backend (spot-flip fixed).
- `tests/rlm/test_metric_reality_smoke.py` — `TestRLAwareSmoke` + the earlier suites.
- Inspected artifacts of the prior failed run in `/tmp/sdar_inspect/` (may be cleared).
