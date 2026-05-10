# ──────────────────────────────────────────────────────────────────────────────
# PPO Reproduction Dockerfile
# Paper : Proximal Policy Optimization Algorithms (Schulman et al., 2017)
#         https://arxiv.org/abs/1707.06347
# Implementation : CleanRL v1.0.0 (PyTorch) — incorporates all 37 implementation
#                  details documented in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# Envs  : MuJoCo v4 (-v4 envs via gymnasium, ~v1 dynamics, small perf delta)
#          Atari 49 games via ale-py + AutoROM
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL org.opencontainers.image.description="PPO (Schulman 2017) reproduction — CleanRL + PyTorch + MuJoCo v4 + Atari"
LABEL paper.arxiv="1707.06347"
LABEL implementation.repo="https://github.com/vwxyzjn/cleanrl"
LABEL implementation.commit="v1.0.0"

# ── 1. System packages ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build tools
        git \
        wget \
        curl \
        unzip \
        build-essential \
        cmake \
        # OpenGL / EGL for headless MuJoCo rendering
        libgl1 \
        libgl1-mesa-dev \
        libglew-dev \
        libegl1 \
        libosmesa6 \
        libosmesa6-dev \
        # Required shared libs
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        # mujoco binary patching
        patchelf \
        # Video encoding (for recording evaluation episodes)
        ffmpeg \
        # SWIG required by some gym environments
        swig \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Environment variables ──────────────────────────────────────────────────
# Use EGL for headless GPU rendering; fall back to OSMesa (CPU) if no GPU
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
# Disable tokenizer parallelism warning
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

# ── 3. Python base + pip ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 4. PyTorch (CPU image; swap index-url for CUDA variant) ──────────────────
# For CUDA 11.8: add --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1: add --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
        "torch==2.1.0" \
        "torchvision==0.16.0"

# ── 5. Core numerics ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "numpy>=1.23.5,<2.0" \
        "scipy==1.11.4"

# ── 6. MuJoCo v3 + Gymnasium (modern free DeepMind MuJoCo) ───────────────────
# NOTE: Paper used MuJoCo 1.50 with -v1 envs. We use mujoco>=3 with -v4 envs.
# -v4 dynamics differ slightly from -v1; this is flagged as assumption ENV003.
RUN pip install --no-cache-dir \
        "mujoco==3.1.1" \
        "gymnasium[mujoco]==0.29.1"

# ── 7. Atari ALE ─────────────────────────────────────────────────────────────
# accept-rom-license extra ships the ROMs directly; no separate download needed
RUN pip install --no-cache-dir \
        "ale-py==0.8.1" \
        "AutoROM[accept-rom-license]==0.6.1" && \
    AutoROM --accept-license --quiet || true

# ── 8. CleanRL utilities ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "tyro>=0.5.0,<0.9.0" \
        "rich>=13.0.0" \
        "tensorboard>=2.14.0" \
        "shimmy>=1.3.0" \
        "moviepy>=1.0.3" \
        "imageio>=2.33.0" \
        "opencv-python-headless>=4.8.0" \
        "huggingface_hub>=0.19.0" \
        "stable-baselines3>=2.1.0"

# ── 9. Clone CleanRL at v1.0.0 ───────────────────────────────────────────────
RUN git clone --depth 1 --branch v1.0.0 \
        https://github.com/vwxyzjn/cleanrl /workspace/cleanrl

# Install CleanRL extras if available, else install core
RUN cd /workspace/cleanrl && \
    pip install --no-cache-dir -e ".[mujoco,atari]" 2>/dev/null || \
    pip install --no-cache-dir -e . 2>/dev/null || true

# ── 10. Sanity check ──────────────────────────────────────────────────────────
RUN python -c "\
import torch; \
import gymnasium; \
import mujoco; \
print(f'[OK] torch={torch.__version__}  gymnasium={gymnasium.__version__}  mujoco={mujoco.__version__}'); \
print(f'[OK] CUDA available: {torch.cuda.is_available()}') \
"

# ── 11. Default entry point ───────────────────────────────────────────────────
# Run MuJoCo PPO:
#   python /workspace/cleanrl/cleanrl/ppo_continuous_action.py \
#     --env-id HalfCheetah-v4 --num-envs 1 --num-steps 2048 \
#     --total-timesteps 1000000 --learning-rate 3e-4 --num-minibatches 32 \
#     --update-epochs 10 --clip-coef 0.2 --ent-coef 0.0 --vf-coef 0.5 \
#     --max-grad-norm 0.5
#
# Run Atari PPO:
#   python /workspace/cleanrl/cleanrl/ppo_atari.py \
#     --env-id BreakoutNoFrameskip-v4 --num-envs 8 --num-steps 128 \
#     --total-timesteps 10000000 --learning-rate 2.5e-4 --num-minibatches 4 \
#     --update-epochs 4 --clip-coef 0.1 --ent-coef 0.01
#
# CRITICAL: Always set --adam-eps 1e-5 (NOT PyTorch default 1e-8)
CMD ["/bin/bash"]
