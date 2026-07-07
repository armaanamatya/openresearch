# Issues

## 2026-07-07 — SDAR GKE (`sandbox=gcp`) + Foundry root/executor launch: 3 blockers

Context: launching SDAR (2605.15155) as a GKE Job on 4×A100 with `opus-foundry`
root + `sonnet-foundry` sub-roles, driven from the `/lab` UI (Autonomous toggle +
GPU count = 4), synced live to the local UI. All three fixed; run then drove
cleanly (opus-foundry root OK, skills fired, 0 errors).

### 1. `temperature` is deprecated for Claude Opus 4.8 / Sonnet 5 (400) — FIXED in code
- Symptom: `generate_rubric_tree` (and every grader/verifier/rubric sub-role on the
  Foundry Claude models) hit `400 invalid_request_error: `temperature` is deprecated
  for this model`; run proceeded "rubric-less" (degraded verification, not a crash).
- Root cause: `AnthropicMessagesClient` (`backend/services/context/workspace/tools/
  anthropic_messages_client.py`) hard-pinned `temperature=0`. Opus 4.8 / Sonnet 5
  reject it. The ROOT model is unaffected (rlm-patched client sends no temperature).
- Fix: `_messages_create` probes with `temperature`, and on the specific
  "temperature … deprecated" 400 drops the param + latches `_omit_temperature` so
  later calls skip the probe. Byte-identical for models that accept it (default
  `claude-sonnet-4-6`). Verified live vs `claude-opus-4-8` + `claude-sonnet-5`.

### 2. `sandbox=gcp` needs `google-cloud-storage` in the dispatching venv — FIXED (dep)
- Symptom: `Sandbox preflight failed: GCP sandbox requires 'google-cloud-storage'`
  → run fails cleanly BEFORE any GPU spins up ($0).
- Fix: `.venv/bin/pip install google-cloud-storage` (`kubernetes` + `google-auth`
  were already present). Rule: the machine that DISPATCHES a `sandbox=gcp` run needs
  the GCS client (it stages cell artifacts to `gs://…`). Belongs in a GCP extra.

### 3. Operational gotchas
- `POST /runs/arxiv` requires a scheme — send `https://arxiv.org/abs/<id>`, not the
  bare host (else `400 "URL must start with http:// or https://."`). The `/lab` UI
  prepends it; a raw curl must too.
- `pkill -f "uvicorn backend.app:create_app"` self-matches the shell running it and
  kills its own parent (exit 144). Kill by port/PID, or use a non-self-matching
  pattern.
- `/runs/arxiv` stages a fresh timestamped upload per POST, so the `project_id`
  differs each call (NOT the deterministic bundled-paper hash). Track the
  `projectId` from the response / stderr `project_id=…`.

## 2026-06-17 — SDAR GCP VM run failed before GPU training

Status: mitigated in code/docs by explicit GCP SDAR preflight and asset warmer.

Symptoms:
- The `sdar-a100-8g` VM launched the Grok/Foundry RLM run, but the process ended after a few minutes.
- All 8 A100 GPUs remained idle after the failed cells.
- ALFWorld provisioning failed because `alfworld-download` was not available.
- WebShop provisioning failed because the expected `web_agent_site` server was not installed/running.
- Generated cells hit missing or broken HuggingFace stack imports, especially `transformers`.

Root cause:
- The full-scope SDAR asset contract was implicit. The expensive run could start before the VM had the SDAR cell dependencies, datasets, model weights, ALFWorld game data, WebShop server, and shared cache env vars prepared.

Fix:
- Added `backend/requirements-sdar.txt` for the SDAR-only ML/environment stack.
- Added `scripts/sdar_gcp_assets.py` to install/warm/check SDAR assets and write `runs/.cache/sdar_gcp.env`.
- Added `scripts/gcp_sdar_preflight.sh` to run the checks on the GCP VM before the reproduction command.
- Hardened the GCP wrapper to stage source without `runs/`, venvs, `__pycache__`, or `.pyc` artifacts, and to refuse non-spot GPU VMs by default.

Required operating rule:
- Do not start the full SDAR paper run on GCP until `scripts/gcp_sdar_preflight.sh prepare` returns GREEN on the VM.
- Keep `OPENRESEARCH_REQUIRE_SPOT=true` for normal operation. Use on-demand A100s only as an explicit exception.
