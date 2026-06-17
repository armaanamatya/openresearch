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
| `runpod` (repo default) | **no** (build_environment short-circuits — ported 2026-06-09) | remote pod via SSH | pod boots `OPENRESEARCH_RUNPOD_IMAGE` (`cuda-devel` default); `OPENRESEARCH_RUNPOD_API_KEY` + SSH key required |
| `auto` | yes | resolved | picks docker/runpod by availability |

`start.sh` preflight-warns on `docker info` for the `docker`/`auto` sandboxes
(non-fatal; bypass `START_SKIP_PREFLIGHT=1`). `local` and `runpod` need no daemon.

## Troubleshooting

- **Hollow `partial` + "SDK success-with-no-text"** → an auth/SDK credential
  failure, NOT Docker. Read the `/lab` detail-panel blockers.
- **`SandboxRuntimeError(backend_unavailable)`** → Docker daemon not reachable
  for a non-`local` sandbox. Start OrbStack/Docker, or use `--sandbox local`.
- **`disk_exhausted` mid-run** → free disk fell below
  `OPENRESEARCH_DISK_FLOOR_GB` or an HF cache exceeded
  `OPENRESEARCH_HF_CACHE_CAP_GB`. Stream + slice datasets; use lighter variants.

## Orchestrator image (GCP/Azure in-cluster root)

The GKE/AKS orchestrator Deployment + CronJob (`infra/gcp/helm`,
`orchestrator.image`) run the root reasoning loop *inside the cluster*:
`python -m backend.cli reproduce <paper> --mode rlm --model claude-oauth
--sandbox gcp`, where the root shells out to the `claude` CLI (which needs Node)
and dispatches training cells outward as K8s Jobs onto the GPU-cell image. So the
orchestrator image is a **control-plane** image — Python backend + Node + the
pinned `claude` CLI, and deliberately **no torch/CUDA** (training runs in the
`gke-cell-base` Job pods, not here).

Dockerfile: `docker/orchestrator/Dockerfile`. Build + push to Artifact Registry:

```bash
# <ar-repo> = REGION-docker.pkg.dev/PROJECT/REPO
#   (terraform -chdir=infra/gcp output -raw artifact_registry_url)
scripts/build_orchestrator_image.sh \
    us-central1-docker.pkg.dev/my-proj/reprolab          # tag defaults to sha-<short>
# then wire the PINNED tag into helm:
helm upgrade reprolab-gke ./infra/gcp/helm \
    --set orchestrator.enabled=true \
    --set orchestrator.image=us-central1-docker.pkg.dev/my-proj/reprolab/reprolab-orchestrator:sha-abc1234
```

The Helm chart **fails fast** (`helm template` errors) if `orchestrator.image` is
empty while a Deployment/CronJob would render — there is no `:latest` fallback.
The `claude` CLI is pinned in the Dockerfile (`ARG CLAUDE_CODE_VERSION`, never
floating); bump it deliberately and rebuild. For `--model azure` roots, also set
`orchestrator.azureOpenai.{endpoint,deployment}` (non-secret config; the API key
comes from Secret Manager); for OAuth roots set
`orchestrator.claudeOauthToken.enabled=true`, which also forces
`OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only` so a present-but-dead
`ANTHROPIC_API_KEY` can't hijack the OAuth path.

Full architecture: `system_overview.md` and `docs/design/rlm-pivot-brief.md`.
