# AKS cell base image — CUDA runtime, PyTorch, Azure SDK.
#
# Philosophy: mirrors the repo's runpod image choice — cuda-runtime (NOT devel).
# Reproduction code calls pre-built CUDA libs (PyTorch, etc.) and never invokes
# nvcc or compiles CUDA kernels, so the devel image is ~14 GB of wasted pull time.
#
# Base: nvidia/cuda 12.1.1 + cuDNN 8 runtime on Ubuntu 22.04.
# Matches the CUDA major.minor of the pinned torch==2.2.2 wheel.
#
# Build / push to ACR — BLOCKED in this session (no docker daemon, no az login):
#   # Build:
#   docker build -t reprolab-aks-cell-base:latest docker/aks-cell-base/
#   # Tag for ACR:
#   docker tag reprolab-aks-cell-base:latest <acr-login-server>/reprolab/aks-cell-base:latest
#   # Push (requires `az acr login --name <acr-name>`):
#   docker push <acr-login-server>/reprolab/aks-cell-base:latest
#
# NOTE: `docker build` could not be run in this session.  All instructions have
# been authored to the spec but cannot be validated without a Docker daemon.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="reprolab-aks-cell-base"
LABEL org.opencontainers.image.description="ReproLab AKS single-cell training image"
LABEL org.opencontainers.image.version="1"

# ---- System packages --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        wget \
        build-essential \
        libffi-dev \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

# ---- Upgrade pip + install wheel -----------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- Install pinned requirements ------------------------------------------------
# NOTE: torch needs the +cu121 index for the correct CUDA wheel.
COPY requirements.txt /opt/reprolab/requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /opt/reprolab/requirements.txt

# ---- Copy entrypoint wrapper ------------------------------------------------
COPY aks_cell_entrypoint.py /opt/reprolab/aks_cell_entrypoint.py
RUN chmod +x /opt/reprolab/aks_cell_entrypoint.py

# ---- Working directory ------------------------------------------------------
WORKDIR /workspace

# ---- Cache mounts (Azure Files PVC) are mounted at runtime ------------------
# The entrypoint sets HF_HOME / PIP_CACHE_DIR under /mnt/reprolab-cache at
# runtime; no static ENV for cache paths (overridden per-run by the wrapper).
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Silence tokenizer parallelism warnings inside the single-cell pod
ENV TOKENIZERS_PARALLELISM=false

# ---- Entrypoint -------------------------------------------------------------
# The K8s Job spec overrides CMD args; the bare entrypoint runs the cell.
ENTRYPOINT ["python", "/opt/reprolab/aks_cell_entrypoint.py"]
