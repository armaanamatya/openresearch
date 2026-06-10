FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    tqdm==4.66.0

# Make python3 available as python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /code
CMD ["python", "train_cell.py"]
