# GKE cell image WITH the verl RL stack, layered on the OFFICIAL vLLM image.
#
# Why FROM vllm/vllm-openai (not a raw CUDA base):
#   The first attempt built torch/vLLM by hand on `nvidia/cuda:...-ubuntu22.04` +
#   `apt install python3.11`. Ubuntu 22.04's apt python3.11 resolves to Python
#   **3.11.0rc1** — a broken release candidate: `inspect.getargspec` is already
#   removed (breaks `pyext`) BUT `sys.get_int_max_str_digits` is not yet added
#   (breaks torch 2.6 `_dynamo` — `AttributeError: module 'sys' has no attribute
#   'get_int_max_str_digits'`). Hand-installing torch can't fix a broken interpreter.
#   The official vLLM image ships a COHERENT, validated torch 2.6 + vLLM 0.8.5 +
#   correct-Python stack — so we layer verl's remaining deps on top of it.
#
# Reusable across verl-based papers: UCPO (arXiv 2605.00365), SDAR (arXiv 2605.15155).
#
# Build via GCP Cloud Build (no local Docker daemon needed):
#   gcloud builds submit --project deepinvent-ext-ut --timeout=3600s \
#     --machine-type=E2_HIGHCPU_8 \
#     --tag us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-verl:v1 \
#     docker/gke-cell-verl/
FROM vllm/vllm-openai:v0.8.5

LABEL org.opencontainers.image.title="reprolab-gke-cell-verl"
LABEL org.opencontainers.image.description="ReproLab GKE cell image with the verl/vLLM RL stack (vLLM 0.8.5 base)"
LABEL org.opencontainers.image.version="1"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# ---- 0) Fail-fast: the base interpreter MUST be sane (this is the whole reason
#         we switched off the raw-CUDA base). torch 2.6 needs get_int_max_str_digits.
RUN python3 -c "import sys; assert sys.version_info[:2] >= (3,10), sys.version; assert hasattr(sys,'get_int_max_str_digits'), 'broken pre-rc2 3.11 interpreter'; import torch; print('base OK: python', sys.version.split()[0], 'torch', torch.__version__)"

# ---- 1) tensordict at verl's ceiling (vLLM doesn't pin it; torch 2.6 is already present)
RUN python3 -m pip install "tensordict==0.6.2"

# ---- 2) verl runtime deps that vLLM does NOT already bring. `pyext` is DROPPED:
# it needs `inspect.getargspec` (removed in 3.11) and is only imported by verl's
# prime_code reward scorer, which UCPO/SDAR (math RLVR) never load.
COPY requirements-verl.txt /opt/reprolab/requirements-verl.txt
RUN python3 -m pip install -r /opt/reprolab/requirements-verl.txt

# ---- 2b) flash-attn: verl's actor worker HARD-imports flash_attn.bert_padding
# (verl/workers/actor/dp_actor.py) at module load, so it is mandatory, not an
# optional GPU extra. The vLLM runtime base has no nvcc, so we install a PREBUILT
# wheel matched EXACTLY to this image's stack (introspected: torch 2.6.0+cu124,
# cp312, CXX11-ABI=False). --no-deps: torch/einops are already present, don't
# let pip re-resolve torch.
RUN python3 -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --no-deps

# ---- 3) verify the full import chain the cell will exercise at runtime
RUN python3 -c "import torch, tensordict, vllm, hydra, omegaconf, ray, datasets, einops; from math_verify import parse; from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input; import flash_attn; print('stack OK: torch', torch.__version__, 'vllm', vllm.__version__, 'tensordict', tensordict.__version__, 'flash_attn', flash_attn.__version__)"

# ---- 4) cell entrypoint (identical contract to gke-cell-base) ----------------
# sys.executable inside the entrypoint == this python3 (torch-enabled), so the
# trainer subprocess runs under the correct interpreter automatically.
COPY gke_cell_entrypoint.py /opt/reprolab/gke_cell_entrypoint.py
RUN chmod +x /opt/reprolab/gke_cell_entrypoint.py

WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/reprolab/gke_cell_entrypoint.py"]
