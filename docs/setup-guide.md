# ReproLab Agent — Developer Setup Guide

Everything a contributor needs to install and configure before working on ReproLab.

## Prerequisites

| Requirement | Minimum version | Check command |
| --- | --- | --- |
| Python | 3.11+ | `python --version` |
| Node.js | 20.19+ LTS or 22.12+ LTS | `node --version` |
| npm | 10+ | `npm --version` |
| Git | 2.40+ | `git --version` |
| Docker Desktop | 4.30+ (Engine 26+) | `docker --version` |
| Claude Code CLI | Latest | `claude --version` |

## 1. Python environment

```bash
# Install Python 3.11+ (if not already installed)
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# Windows — download from https://www.python.org/downloads/
# Make sure to check "Add to PATH" during install
```

Create a virtual environment:

```bash
cd openresearch
python3.11 -m venv .venv

# Activate
# macOS/Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

## 2. Claude Code CLI

Claude Code is the LLM backend for all agents during development.

```bash
# Install globally
npm install -g @anthropic-ai/claude-code

# Log in with your Claude Max/Pro subscription
claude login

# Verify
claude --version
claude -p "hello" --output-format json
```

You need an active **Claude Pro or Max subscription**. Max is recommended for multi-agent workloads (higher rate limits).

For production/sustained runs, you'll also need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 3. Agent SDKs (Python)

The Python agent SDKs are installed with the backend dependency set in step 6.
If you want to verify them independently after installing dependencies:

```bash
python -c "import claude_agent_sdk; print('claude-agent-sdk OK')"
python -c "import agents; print('openai-agents OK')"
```

## 4. Docker Desktop

### Install

- **macOS:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Windows:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/) — requires WSL 2 enabled
- **Linux:** Install Docker Engine via your package manager

### Configure

After installing, make sure Docker is running:

```bash
docker run hello-world
```

Recommended Docker Desktop settings:
- **Resources:** At least 4 CPUs, 8 GB RAM allocated to Docker
- **Disk:** At least 30 GB available for images and containers

### GPU support (optional, for GPU papers)

Only needed if you want to run ML training on a local GPU.

**Linux (NVIDIA GPU):**

```bash
# Install NVIDIA drivers (if not already)
sudo apt install nvidia-driver-550

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Windows:** Docker Desktop with WSL 2 supports GPU passthrough automatically if you have NVIDIA drivers installed on the host. No extra toolkit needed inside WSL.

**macOS:** No NVIDIA GPU support. Use CPU-only mode (fine for demo papers).

### Remote GPU support with Runpod

Use this when local Docker is not enough and experiments need a disposable remote GPU.

Install and verify the CLI:

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.zip -OutFile runpodctl.zip
Expand-Archive runpodctl.zip -DestinationPath $env:LOCALAPPDATA\runpodctl
[Environment]::SetEnvironmentVariable('Path', $env:Path + ";$env:LOCALAPPDATA\runpodctl", 'User')
runpodctl version
```

Set the backend environment:

```bash
REPROLAB_RUNPOD_API_KEY=
REPROLAB_RUNPOD_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
REPROLAB_RUNPOD_GPU_TYPE=NVIDIA GeForce RTX 4090
REPROLAB_RUNPOD_GPU_COUNT=1
REPROLAB_RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519
```

Then run a paper with:

```bash
python -m backend.cli reproduce path/to/paper.pdf --mode sdk --sandbox runpod
```

The Runpod backend creates a GPU Pod, exposes SSH on `22/tcp`, uploads generated code to `/workspace/reprolab/<project>/baseline/work`, runs commands from `/work`, syncs `/artifacts` back into the local run directory, and deletes the Pod when the run ends.

## 5. Node.js and frontend dependencies

```bash
# Install Node.js 20.19+ LTS or 22.12+ LTS
# macOS
brew install node@20

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Windows — download from https://nodejs.org/
```

Frontend setup (run from repo root):

```bash
cd frontend   # once the frontend directory exists
npm install
```

## 6. Python dependencies

From the repo root (with your venv activated):

```bash
pip install -r requirements.txt
# or, equivalently:
pip install -r backend/requirements.txt
```

`requirements.txt` at the repo root delegates to `backend/requirements.txt`.
Key packages installed by the backend runtime set:

| Package | Purpose |
| --- | --- |
| `claude-agent-sdk` | Agent orchestration via Claude Agent SDK |
| `openai-agents` | OpenAI Agents SDK runtime support |
| `pymupdf` | PDF parsing for ML papers |
| `docker` | Python Docker SDK for `LocalDockerBackend` |
| `asyncssh` | Remote runtime support |
| `fastapi` | Backend API server |
| `uvicorn` | ASGI server |
| `pydantic` | Data validation and schemas |
| `httpx` | HTTP client support |

## 7. Environment variables

Create a `.env` file in the repo root (this file is gitignored):

```bash
# Required
# (none — Claude Code subscription handles LLM auth via `claude login`)

# Optional — for production/sustained runs
ANTHROPIC_API_KEY=sk-ant-...

# Optional — for web search
TAVILY_API_KEY=tvly-...

# Optional — GPU device selection (default: all available)
CUDA_VISIBLE_DEVICES=0
```

## 8. Verify everything works

Run these checks to confirm your setup is complete:

```bash
# Python
python --version          # 3.11+
python -c "import docker; print('docker SDK OK')"
python -c "import claude_agent_sdk; print('claude-agent-sdk OK')"

# Claude Code
claude --version
claude -p "respond with OK" --output-format json

# Docker
docker --version          # 26+
docker run hello-world

# GPU (optional)
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Node
node --version            # 20.19+ LTS or 22.12+ LTS
npm --version             # 10+
```

## 9. PaperBench head-to-head

Run the agent pipeline against a vendored PaperBench paper bundle and
compare to the published BasicAgent baselines (PaperBench paper, OpenAI,
April 2025, Tables 11/15).

```bash
# 1. Inspect the bundle (no LLM key needed — pure rubric math)
.venv/bin/python -m backend.cli paperbench list
.venv/bin/python -m backend.cli paperbench summary --paper-id ftrl

# 2. Dry validation (writes a placeholder submission, no LLM key)
.venv/bin/python -m backend.cli paperbench run --no-pipeline --paper-id ftrl

# 3. Real run (requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
.venv/bin/python -m backend.cli paperbench run --paper-id ftrl --seeds 0 1 2

# 4. Poll status
.venv/bin/python -m backend.cli paperbench status --run-group-id <id>
```

The web UI mirrors this at `http://localhost:3000/paperbench` (paper
picker, seed input, dry/pipeline toggle, score-vs-baseline grid, live
attempts table). Results land in `runs/paperbench/<run_group_id>/`.

To produce numbers that are honestly comparable to PaperBench's
published Tables 11/15, replace `third_party/paperbench/ftrl/paper.md`,
`addendum.md`, and `rubric.json` with the upstream artifacts. Full
swap-in steps are in `third_party/paperbench/README.md`.

## 10. Phase 2 research workspace services

Phase 2 adds durable workspace services for graph navigation, reusable
cross-project memory, dataset cache state, approval checkpoints, failure
diagnosis, reproducibility scoring, and multi-paper comparison summaries.
For the end-to-end agent state model, see
[`docs/agent-lifecycle.md`](agent-lifecycle.md).

Start the backend:

```bash
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
```

Useful local checks:

```bash
# Combined Phase 2 workspace read model
curl http://127.0.0.1:8000/phase2/projects/<project_id>/summary

# AST knowledge graph query
curl "http://127.0.0.1:8000/phase2/projects/<project_id>/graph?entity_type=function&calls=train"

# Cross-project memory search
curl "http://127.0.0.1:8000/phase2/memory/search?query=torch%20gymnasium"

# Dataset cache planning
curl -X POST http://127.0.0.1:8000/phase2/datasets/plan \
  -H "Content-Type: application/json" \
  -d '{"project_id":"prj_demo","name":"CIFAR-10","version":"official","size_bytes":178257920}'

# Human approval checkpoint
curl -X POST http://127.0.0.1:8000/phase2/approvals/evaluate \
  -H "Content-Type: application/json" \
  -d '{"project_id":"prj_demo","action":"dataset_download","dataset_size_gb":82}'

# Failure diagnosis taxonomy
curl -X POST http://127.0.0.1:8000/phase2/failures/diagnose \
  -H "Content-Type: application/json" \
  -d '{"project_id":"prj_demo","stage":"training","command":"python train.py","stderr":"CUDA out of memory"}'
```

## 11. Docker (run the whole app in one container)

For a one-command boot of backend + frontend together:

```bash
docker compose up --build       # first time
docker compose up -d            # subsequent (detached)
docker compose logs -f app      # tail logs
docker compose down             # stop, keep volumes
```

URLs after boot:
- Lab UI: http://localhost:3000/lab
- PaperBench UI: http://localhost:3000/paperbench
- Backend health: http://localhost:8000/health

What the compose file mounts:
- `/var/run/docker.sock` (host) → so the inner `LocalDockerBackend`
  spawns sandbox containers against the host daemon (no nested DinD).
  **Local-dev only**: this gives the container effective root on host
  Docker. For prod, use `RunPodBackend` (`--sandbox runpod`) instead.
- `./runs` → persists pipeline artifacts, Hermes adapter memory, and
  PaperBench statuses between restarts.
- `./third_party` (read-only) → vendored PaperBench bundles.
- `./.env` (read-only) → keeps `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` / `REPROLAB_RUNPOD_API_KEY` available to the
  entrypoint without printing secret values through `docker compose config`
  (and prevents in-container typos from clobbering your local secrets).

The image is a 3-stage build:
1. `python:3.12-slim` — installs `backend/requirements.txt` into `/opt/venv`
2. `node:20-bookworm-slim` — `npm ci` + `next build` for the frontend
3. `python:3.12-slim` runtime — copies the venv + Next build, adds
   `tini` (PID 1), `docker.io` (CLI for the Python `docker` SDK),
   `nodejs` (to run `next start`), `openssh-client` (for asyncssh /
   RunPod backend)

`docker/entrypoint.sh` boots both servers in parallel and forwards
SIGTERM so `docker stop` is fast (10 s grace) instead of waiting on
the 30 s default SIGKILL.

## Quick reference — what to install

| Tool | Required? | Install time |
| --- | --- | --- |
| Python 3.11+ | Yes | 2 min |
| Claude Code CLI | Yes | 1 min |
| Claude Pro/Max subscription | Yes | — |
| Docker Desktop | Yes | 5 min |
| Node.js 20.19+ LTS or 22.12+ LTS | Yes | 2 min |
| NVIDIA Container Toolkit | Only for GPU papers | 5 min |
| Anthropic API key | Only for production | — |
