# ReproLab Agent — Developer Setup Guide

Everything a contributor needs to install and configure before working on ReproLab.

## Prerequisites

| Requirement | Minimum version | Check command |
| --- | --- | --- |
| Python | 3.11+ | `python --version` |
| Node.js | 20+ (LTS) | `node --version` |
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

## 3. Claude Agent SDK (Python)

```bash
pip install claude-code-sdk
```

Verify:

```python
python -c "from claude_code_sdk import query; print('SDK installed')"
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

## 5. Node.js and frontend dependencies

```bash
# Install Node.js 20+ LTS
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
```

Key packages that will be in `requirements.txt`:

| Package | Purpose |
| --- | --- |
| `claude-code-sdk` | Agent orchestration via Claude Agent SDK |
| `chromadb` | Semantic vector search (Context Layer 2) |
| `nougat-ocr` | PDF parsing for ML papers |
| `docker` | Python Docker SDK for `LocalDockerBackend` |
| `litellm` | LLM abstraction used by RLM |
| `networkx` | Knowledge graph (Phase 2, Graphify) |
| `restrictedpython` | REPL sandbox security |
| `fastapi` | Backend API server |
| `uvicorn` | ASGI server |
| `websockets` | Real-time event streaming to frontend |
| `pydantic` | Data validation and schemas |
| `aiohttp` | Async HTTP for web search |

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
python -c "import chromadb; print('chromadb OK')"
python -c "from claude_code_sdk import query; print('claude-sdk OK')"

# Claude Code
claude --version
claude -p "respond with OK" --output-format json

# Docker
docker --version          # 26+
docker run hello-world

# GPU (optional)
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Node
node --version            # 20+
npm --version             # 10+
```

## Quick reference — what to install

| Tool | Required? | Install time |
| --- | --- | --- |
| Python 3.11+ | Yes | 2 min |
| Claude Code CLI | Yes | 1 min |
| Claude Pro/Max subscription | Yes | — |
| Docker Desktop | Yes | 5 min |
| Node.js 20+ | Yes | 2 min |
| NVIDIA Container Toolkit | Only for GPU papers | 5 min |
| Anthropic API key | Only for production | — |
