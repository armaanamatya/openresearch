FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir gymnasium==0.29.1 matplotlib==3.8.0 numpy==1.26.4 tqdm==4.66.0

# /code is mounted READ-ONLY at runtime; all output goes to $OUTPUT_DIR
WORKDIR /code

# Default OUTPUT_DIR — overridden by the sandbox runner
ENV OUTPUT_DIR=/artifacts
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
