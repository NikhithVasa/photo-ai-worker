FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV CUDNN_FRONTEND_ATTN_DISABLED=1
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r /app/requirements.txt

RUN python - <<'PY'
import torch
import torchvision
import transformers
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('transformers:', transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    print('Qwen2_5_VLForConditionalGeneration import OK')
except Exception as e:
    raise SystemExit(f'Qwen import failed: {e}')
try:
    from transformers import AutoModelForMultimodalLM
    print('AutoModelForMultimodalLM import OK')
except Exception as e:
    try:
        from transformers import AutoModelForImageTextToText
        print('AutoModelForImageTextToText import OK')
    except Exception as e2:
        raise SystemExit(f'Gemma model class import failed: {e}; fallback failed: {e2}')
PY

ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/runpod-volume/huggingface/sentence-transformers
ENV TORCH_HOME=/runpod-volume/torch
ENV XDG_CACHE_HOME=/runpod-volume/cache

RUN mkdir -p \
    /runpod-volume/huggingface \
    /runpod-volume/huggingface/sentence-transformers \
    /runpod-volume/torch \
    /runpod-volume/cache

RUN python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available at build:', torch.cuda.is_available())
print('device count at build:', torch.cuda.device_count())
print('cudnn:', torch.backends.cudnn.version())
PY

RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct')
print('Downloaded Qwen/Qwen2.5-VL-3B-Instruct')
PY

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Downloaded sentence-transformers/all-MiniLM-L6-v2')
PY

COPY handler.py /app/handler.py
COPY runtime_gemma_force_patch.py /app/runtime_gemma_force_patch.py
COPY entrypoint.py /app/entrypoint.py

CMD ["python", "-u", "/app/entrypoint.py"]
