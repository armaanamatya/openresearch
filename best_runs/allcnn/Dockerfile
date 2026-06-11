FROM python:3.11-slim

# System deps: git + CUDA shared-library stubs for LD_LIBRARY_PATH resolution
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch 2.2.0 (cu121 wheel) so the GPU is usable.
# The base image is CPU-only; torch itself ships its own CUDA runtime libs
# inside the wheel, so no nvidia-cuda-* apt packages are needed.
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    tqdm==4.66.0

# Working directory is /code (read-only mount in sandbox).
# We do NOT copy code here — the harness mounts /code read-only at runtime.
# All outputs go to $OUTPUT_DIR (injected by harness).
WORKDIR /code

# Default entry point (overridden by harness to run individual cells).
CMD ["python", "/code/train.py"]
