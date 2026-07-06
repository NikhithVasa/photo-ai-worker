# Photo AI Worker Developer Guide

The photo AI worker enriches already-ingested photos with captions, search text, text embeddings, optional image embeddings, and culling/best-photo data.

## Runtime Shape

- Runtime: Python on RunPod.
- Entry point: `entrypoint.py`, which loads `handler.py`.
- Container base: PyTorch CUDA runtime.
- Storage: S3 for media objects and temporary derived assets.
- Database: PostgreSQL/RDS for photo, person, embedding, culling, and best-photo records.
- Model cache: `/runpod-volume` paths for Hugging Face, sentence-transformers, Torch, and general cache files.

## Main Files

| File | Purpose |
| --- | --- |
| `handler.py` | Main worker logic for image-text metadata, embeddings, and culling writes. |
| `entrypoint.py` | Runtime entry point used by the Docker image. |
| `runtime_gemma_force_patch.py` | Runtime patching for Gemma model loading behavior. |
| `Dockerfile` | CUDA image, dependencies, model import checks, and cache warmup. |
| `requirements.txt` | Python runtime dependencies. |
| `test_input.json` | Example local/RunPod input payload. |

## Responsibilities

The worker can write or update:

- `photos.caption`
- `photos.ai_description`
- `photos.search_text`
- `photos.qwen_json`
- `photos.qwen_status`
- `photos.search_embedding`
- `photo_people.qwen_description`
- `photo_people.qwen_json`
- `photo_people.search_text`
- `photo_people.search_embedding`
- `photo_culling_scores`
- `photo_image_embeddings`
- `best_photo_collections`

Names such as `qwen_status` and `qwen_json` are legacy database names. Check runtime logs for the actual provider, such as Gemma or Qwen.

## Important Environment Variables

| Variable | Purpose |
| --- | --- |
| `S3_BUCKET` | Bucket that contains source and generated media. |
| `AWS_DEFAULT_REGION` | AWS region for S3. |
| `RDS_HOST`, `RDS_PORT`, `RDS_DB`, `RDS_USER`, `RDS_PASSWORD` | PostgreSQL connection. |
| `LOCAL_WORK` | Worker scratch directory. |
| `QWEN_MODEL_ID` | Default Qwen model ID. |
| `QWEN_IMAGE_MAX_SIDE`, `QWEN_MAX_NEW_TOKENS`, `QWEN_INFERENCE_BATCH_SIZE` | Image-text inference controls. |
| `TEXT_EMBED_MODEL_ID`, `TEXT_EMBED_BATCH_SIZE` | Text embedding model and batch size. |
| `IMAGE_EMBED_MODEL_ID`, `IMAGE_EMBED_BATCH_SIZE` | Image embedding model and batch size. |
| `CULLING_VERSION`, `CLUSTER_VERSION` | Version labels for generated culling data. |
| `RESET_EXISTING_QWEN` | Regenerate completed image-text/search metadata when true. |

## Build Notes

The Dockerfile validates key imports during build:

- Torch and torchvision versions.
- Qwen model class import.
- Gemma-compatible model class import.
- Hugging Face downloads for Qwen and sentence-transformers models.

If the image builds but runtime fails, compare build-time model availability with the RunPod template environment and mounted cache volume.

## Web App Handoff

The web app expects the worker to populate fields used by search, people views, culling pages, and gallery display. See the web app docs:

- `v0-ai-photo-gallery/docs/developer/data-and-media-flow.md`
- `v0-ai-photo-gallery/docs/developer/operations-runbook.md`

## Debug Checklist

- Confirm the RunPod job input includes the expected album/photo scope.
- Confirm S3 objects exist for the photos being processed.
- Confirm database credentials and network access.
- Check logs for provider selection: Gemma vs Qwen.
- Check whether `RESET_EXISTING_QWEN` is unintentionally skipping or regenerating data.
- Confirm embedding dimensions match database expectations.
