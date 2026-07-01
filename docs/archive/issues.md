# Issues

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
