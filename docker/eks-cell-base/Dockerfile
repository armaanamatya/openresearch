# EKS cell image — CUDA/PyTorch training cell with S3 access through IRSA.
#
# Build from the repository root so the shared cell contract can be copied:
#   docker build -f docker/eks-cell-base/Dockerfile -t <ecr>/reprolab-eks-cell:<digest> .
#
# Do not pass AWS credentials as build args or image environment variables. The
# pod's ServiceAccount supplies short-lived web-identity credentials at runtime.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="reprolab-eks-cell"
LABEL org.opencontainers.image.description="ReproLab EKS single-cell training image (S3 via IRSA)"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl wget build-essential libffi-dev libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel boto3

# Keep the training dependency set aligned with the proven GKE base image. The
# EKS adapter is intentionally a different ENTRYPOINT, not a second trainer.
COPY docker/gke-cell-base/requirements.txt /opt/reprolab/requirements.txt
RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /opt/reprolab/requirements.txt

COPY docker/gke-cell-base/gke_cell_entrypoint.py /opt/reprolab/gke_cell_entrypoint.py
COPY docker/eks-cell-base/eks_cell_entrypoint.py /opt/reprolab/eks_cell_entrypoint.py
RUN chmod +x /opt/reprolab/gke_cell_entrypoint.py /opt/reprolab/eks_cell_entrypoint.py

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python3", "/opt/reprolab/eks_cell_entrypoint.py"]
