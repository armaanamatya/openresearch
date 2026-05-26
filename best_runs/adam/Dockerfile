FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0
RUN pip install --no-cache-dir matplotlib==3.8.0 numpy==1.26.4 tqdm==4.66.0

WORKDIR /workspace
COPY . /workspace/

CMD ["python", "train.py"]
