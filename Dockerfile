FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /app/requirements.txt

ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/huggingface/sentence-transformers
ENV TORCH_HOME=/models/torch
ENV XDG_CACHE_HOME=/models/cache

RUN mkdir -p \
    /models/huggingface \
    /models/huggingface/sentence-transformers \
    /models/torch \
    /models/cache

RUN python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

RUN python3 - <<'PY'
import onnxruntime as ort
print("onnxruntime:", ort.__version__)
print("device:", ort.get_device())
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDAExecutionProvider missing"
PY

RUN python3 - <<'PY'
from insightface.app import FaceAnalysis

# Download model only; CPU prepare avoids build-time CUDA dependency.
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

print("InsightFace buffalo_l downloaded")
PY

RUN python3 - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Downloaded sentence-transformers/all-MiniLM-L6-v2")
PY

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]