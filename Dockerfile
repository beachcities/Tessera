FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

WORKDIR /app

# 基本パッケージ
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# pip更新
RUN pip install --upgrade pip

# PyTorch + CUDA 12.6 (公式インデックスから)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Mamba依存
RUN pip install --no-cache-dir \
    triton \
    packaging \
    ninja \
    einops

# Mamba SSM (ビルドに時間かかる)
RUN pip install --no-cache-dir \
    causal-conv1d \
    mamba-ssm

COPY requirements_gpu.txt .

CMD ["bash"]
