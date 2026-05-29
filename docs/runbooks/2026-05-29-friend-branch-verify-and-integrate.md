# 2026-05-29 — Verify & integrate `feat/rlm-perf-multienv-accelerator`

**Branch produced:** `feat/integrate-perf-accelerator-into-stability` (off `pipeline-validation-mech-understanding`, merging in `feat/rlm-perf-multienv-accelerator`).

## Context

`feat/rlm-perf-multienv-accelerator` (sww35 / Abheek, 25 commits) added:
- Pluggable accelerator abstraction (`off|auto|local|runpod|azure|endpoint`) with vLLM serve harness `scripts/serve_local_llm.py` (465 LOC) and parallel sweep `scripts/batch_reproduce.py` (814 LOC, adds `--gpus-per-run`).
- Docker-free local-GPU sandbox + `LocalGpuAllocator` (581 LOC) that leases NVIDIA GPUs exclusively per parallel run.
- Shared pip wheel cache + shared HF/dataset cache (`runs/.cache/data`) — first-run cost paid once per sweep, not per paper.
- Root-cause fix for SDAR `env_load_failed`: `/workspace` (RunPod-only) was being used on local sandboxes; `_ensure_local_data_root` now repoints `REPROLAB_RUNPOD_VOLUME_MOUNT_PATH` at a writable shared cache dir for `--sandbox local`, no-op for `runpod`/`docker`.
- Canonical HF owner/name in dataset-setup guidance (`hotpotqa/hotpot_qa`).
- `implement_baseline` pre-emit stall threshold raised 240s → 900s (measured against real SDAR run: agent's first file lands at ~+593s, biggest no-output gap ~402s).
- `_write_demo_status(process_status, verdict)` kwargs added (fixed latent CLI `TypeError`).
- vLLM lifecycle: `start_new_session=True` + process-group SIGTERM/SIGKILL (orphaned tensor-parallel workers were stealing CPU after timeouts); auth-aware readiness probe (probes were 401-ing on api-key-protected servers).
- Rubric: `--paper-hint` regex invariants as deterministic score gate (245 LOC scorer + 681 LOC tests).
- Sonnet for sub-agents (after user instruction "always from now on" — see registry edit below).

## Empirical verification on RunPod (this Mac, no local GPU)

Launched friend's branch with SDAR baseline command (`--mode rlm --sandbox runpod --model claude-oauth --paper-hint 2605.15155 --vram-gb 38 --max-wall-clock 5400 --max-usd 10`). Outcome:

| Metric | Value |
|---|---|
| Wall clock | 44 min 32 s (of 90 min cap) |
| Iterations | 3 |
| Rubric | 0.0 / target 0.6 |
| Cost | $0 (claude-oauth subscription) + $0 RunPod (pod never spun up) |
| Verdict | partial |
| Executor model in `final_report.json` | `claude-sonnet-4-6` ✓ (Sonnet sub-agent switch verified) |

### Timeline

1. 21:16 — Launch. Sub-agent set to Sonnet. Ingest + understand_section + extract_hyperparameters + detect_environment succeed.
2. 21:25 — `build_environment` errors (detect_environment produced a thin Dockerfile: matplotlib + numpy + tqdm only — no torch / transformers / alfworld / webshop).
3. **21:32 – 21:55 — Anthropic API outage from this Mac.** `api.anthropic.com` and `claude.ai` both TCP-dead (`HTTP=000`, timeout). Independently-hosted `status.anthropic.com` still responsive (HTTP 302). GitHub / RunPod unaffected → confirmed Anthropic-side. 4 cascading `implement_baseline` failures with `'API Error: Unable to connect to API (ConnectionRefused)'`.
4. 21:55 — Anthropic recovers. `verify_against_rubric status=ok`, score=0.0 (no impl yet).
5. 21:57 – 22:00 — Root iterates: `propose_improvements` → candidates → `implement_baseline` finally writes code → `run_experiment` fail-fasts in **10 ms** with `failure_class=unknown, error="no commands.json"` (no RunPod waste).
6. 22:00 — Root voluntarily calls `FINAL_VAR` at iter=3 → ships `verdict=partial` with rubric=0 **despite 45 min of wall-clock remaining**.

### Verified working on friend's branch

- ✅ Sonnet sub-agent switch (executor=`claude-sonnet-4-6` in `final_report.json`).
- ✅ `_write_demo_status(process_status, verdict)` no longer raises.
- ✅ `run_complete` carries `mode/models/started_at/completed_at` (cleanup-spec compliant).
- ✅ `run_experiment` fail-fast on `commands_missing_file` saved a pod provision.
- ✅ Survived a 25-min upstream API outage without process death.
- ✅ `_ensure_local_data_root` correctly no-ops for `sandbox=runpod` (line 1055 `if "local" not in key: return`).
- ✅ `paper-hint 2605.15155` applied 6 invariants + scope guidance.

### Gaps surfaced

- ❌ **Premature exit at iter=3 with rubric=0** — friend's branch lacks the BUG-LR-013 forced-iteration guard (None/zero-score `FINAL_VAR` refusal). On `pipeline-validation-mech-understanding`, this would have refused `FINAL_VAR` and kept the loop running until either a real `run_experiment status=ok` or wall-clock cap (90 min). The 45 min of unused runway would have likely produced at least one real implementation attempt.
- ❌ `implement_baseline` doesn't reliably produce `commands.json` under the lossy paper-text fallback path (no `parsed_full_text.txt`). Code was written but the manifest wasn't.

## Integration — `feat/integrate-perf-accelerator-into-stability`

Merge of `feat/rlm-perf-multienv-accelerator` into `pipeline-validation-mech-understanding`:

- **Auto-merged cleanly:** `CLAUDE.md`, `backend/agents/rlm/run.py`, `backend/agents/rlm/system_prompt.py`, `backend/cli.py`, plus all of friend's new files (accelerator.py, batch_reproduce.py, serve_local_llm.py, local_gpu_allocator.py, ...).
- **Conflict 1: `backend/agents/rlm/primitives.py`** — pre-emit stall threshold. HEAD: `1800.0` (safe overshoot), incoming: `900.0` (measured against real SDAR data). Resolution: take friend's `900.0` + his measurement comment (better-justified value).
- **Conflict 2: `backend/services/context/workspace/tools/rlm_query.py`** — `ClaudeAgentOptions.permission_mode` in the text-completion path. HEAD: `"bypassPermissions"` (BUG-NEW-038 fix); incoming: `"default"` with explanation that `tools=[]` already forbids tool use, and `"plan"` made the model return empty results because it expected `ExitPlanMode` instead of a direct answer. Resolution: take friend's `"default"` + comment (more surgical than `bypassPermissions`).

### Post-merge audit

- All 3 real `ClaudeAgentOptions` call sites (`claude_runtime.py:93`, `rlm_query.py:569`, `hermes_audit/providers.py:394`) retain BUG-NEW-038 isolation (`setting_sources=[]` + explicit `mcp_servers`).
- Sonnet sub-agent registry edit re-applied (`baseline-implementation` + `improvement-path` → `claude-sonnet-4-6`); stored to TrueMemory as a permanent preference.
- BUG-NEW-041 (SIGTERM handler → `demo_status="killed"`), BUG-NEW-042 (Dockerfile shape guard), BUG-NEW-043 (subcall traceback + recursion limit), and the forced-iteration / premature-exit guards (BUG-LR-013) all preserved from `pipeline-validation-mech-understanding`.

## What this branch enables

- Local-GPU SDAR runs (`--sandbox local --accelerator local` with vLLM Qwen-Coder 14B) — friend's path that produced the 5-epoch result, now also carrying the stability guards that would have saved today's RunPod attempt.
- RunPod SDAR runs (`--sandbox runpod --accelerator off`) with the forced-iteration guard preventing premature `verdict=partial` exits.
- Parallel multi-paper sweeps via `batch_reproduce.py --gpus-per-run auto` (Abheek's "gpumaxx").

## Not done in this change

- The `commands.json` template gap surfaced in the verification run — `implement_baseline` doesn't consistently write the run manifest under the lossy-paper-text fallback. Tracked separately; see `final_report.json::iterations[2].run_experiment` for the failure shape.
- Empirical re-verification of the merged branch — requires another reproduction run + Anthropic uptime + wall-clock budget. The static + test-suite checks are clean.
