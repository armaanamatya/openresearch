FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages — CPU-only PyTorch to keep image lean (ENV003 assumption)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    numpy==1.26.4 \
    tqdm==4.66.0

WORKDIR /code
COPY . /code/

# OUTPUT_DIR is injected at runtime by the sandbox; default for local docker runs
ENV OUTPUT_DIR=/artifacts
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
