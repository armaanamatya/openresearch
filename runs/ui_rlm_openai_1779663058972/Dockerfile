FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages (CPU-only torch for lightweight CartPole baseline)
# ENV001: pytorch 2.2.0 assumed from paper date
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    tqdm==4.66.0

# /code is mounted read-only; outputs go to $OUTPUT_DIR (set by sandbox runtime)
WORKDIR /code

# Single adaptive entrypoint - branches on HAS_GPU internally
CMD ["python", "train.py"]
