FROM python:3.10-slim
LABEL org.opencontainers.image.description="PPO (Schulman 2017) reproduction — CleanRL + PyTorch + MuJoCo v4 + Atari"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip build-essential cmake \
    libgl1 libgl1-mesa-dev libglew-dev libegl1 \
    libosmesa6 libosmesa6-dev libglib2.0-0 \
    libxext6 libxrender1 libgomp1 patchelf ffmpeg swig \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "torch==2.1.0" "torchvision==0.16.0"
RUN pip install --no-cache-dir "numpy>=1.23.5,<2.0" "scipy==1.11.4"
RUN pip install --no-cache-dir "mujoco==3.1.1" "gymnasium[mujoco]==0.29.1"
RUN pip install --no-cache-dir "ale-py==0.8.1" "AutoROM[accept-rom-license]==0.6.1" \
    && AutoROM --accept-license --quiet || true
RUN pip install --no-cache-dir \
    "tyro>=0.5.0,<0.9.0" rich tensorboard "shimmy>=1.3.0" \
    "moviepy>=1.0.3" "imageio>=2.33.0" "opencv-python-headless>=4.8.0" \
    "huggingface_hub>=0.19.0" "stable-baselines3>=2.1.0"

# Reference codebase (not strictly required since our scripts are self-contained,
# kept as a reference / fallback per ENV005).
RUN git clone --depth 1 --branch v1.0.0 https://github.com/vwxyzjn/cleanrl /workspace/cleanrl \
    && cd /workspace/cleanrl \
    && (pip install --no-cache-dir -e ".[mujoco,atari]" 2>/dev/null \
        || pip install --no-cache-dir -e . 2>/dev/null \
        || true)

# Copy our adapted PPO scripts and config
COPY ppo_continuous_action.py /workspace/code/ppo_continuous_action.py
COPY ppo_atari.py /workspace/code/ppo_atari.py
COPY train.py /workspace/code/train.py
COPY evaluate.py /workspace/code/evaluate.py
COPY config.json /workspace/code/config.json
COPY commands.log /workspace/code/commands.log

WORKDIR /workspace/code

# Sanity check at build time
RUN python -c "import torch, gymnasium, mujoco; \
    print(f'[OK] torch={torch.__version__} gymnasium={gymnasium.__version__} mujoco={mujoco.__version__}')"

CMD ["/bin/bash"]
