FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages — CPU torch + torchvision for MNIST, scipy for .mat loading
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    scipy==1.12.0 \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    tqdm==4.66.0

WORKDIR /code

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
