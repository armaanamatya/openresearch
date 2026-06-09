<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# Running the project — workflow + sandbox×prerequisite matrix

> **Status:** Current. Day-to-day commands live in `CLAUDE.md`; this runbook is
> the operator-facing "how do I actually run it + which sandbox needs what".

## Quickstart

```bash
python3 -m venv .venv                              # fresh clones ship no venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000   # API
.venv/bin/python -m backend.cli reproduce 2605.15155 --provider anthropic --sandbox local   # CLI
```

Cheapest local-dev cost model: OpenAI root (`--model gpt-5`, ~$1/run) + OAuth
subscription for Sonnet sub-agents ($0) + RunPod COMMUNITY GPU. Zero-cost:
`--model claude-oauth` for both surfaces (subject to subscription rate limits).
See `CLAUDE.md` → "RLM auth" for the full credential matrix and gotchas
(no-credit `ANTHROPIC_API_KEY` does **not** fall back to OAuth; shell env shadows
`.env`, BUG-LR-014).

## Sandbox × prerequisite matrix

| `--sandbox` | Local Docker daemon? | GPU | Notes |
|---|---|---|---|
| `local` | **no** | host GPUs via `OPENRESEARCH_GPU_DEVICE_IDS` | `build_environment` is a no-op; experiments run as host subprocesses |
| `docker` | **yes** | local | real `docker build`; network/mem/CPU capped |
| `runpod` (repo default) | **yes** (for the local build, currently wasted — rough edge) | remote pod via SSH | pod boots `OPENRESEARCH_RUNPOD_IMAGE` (`cuda-devel` default); `OPENRESEARCH_RUNPOD_API_KEY` + SSH key required |
| `auto` | yes | resolved | picks docker/runpod by availability |

`start.sh` preflight-checks `docker info` whenever the sandbox is not `local`
(bypass `START_SKIP_PREFLIGHT=1`).

## Troubleshooting

- **Hollow `partial` + "SDK success-with-no-text"** → an auth/SDK credential
  failure, NOT Docker. Read the `/lab` detail-panel blockers.
- **`SandboxRuntimeError(backend_unavailable)`** → Docker daemon not reachable
  for a non-`local` sandbox. Start OrbStack/Docker, or use `--sandbox local`.
- **`disk_exhausted` mid-run** → free disk fell below
  `OPENRESEARCH_DISK_FLOOR_GB` or an HF cache exceeded
  `OPENRESEARCH_HF_CACHE_CAP_GB`. Stream + slice datasets; use lighter variants.

Full architecture: `system_overview.md` and `docs/design/rlm-pivot-brief.md`.
