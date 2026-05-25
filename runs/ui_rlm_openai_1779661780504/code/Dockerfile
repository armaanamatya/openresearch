FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages — CPU-only PyTorch keeps image small (Assumption ENV003)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    tqdm==4.66.0

WORKDIR /code
COPY . /code/

# OUTPUT_DIR is set by the sandbox runtime; default to /artifacts for local testing
ENV OUTPUT_DIR=/artifacts
RUN mkdir -p /artifacts

CMD ["python", "train.py", "--config", "config.json"]
