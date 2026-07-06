# Photo AI Worker Agent Notes

Use these notes when making worker changes.

## Start Here

- For image-text generation, captions, and search metadata, read `handler.py`.
- For container dependency or model import issues, read `Dockerfile` and `requirements.txt`.
- For runtime startup behavior, read `entrypoint.py`.

## Change Safety

- Do not rename legacy `qwen_*` fields unless the web app and database migrations are updated together.
- Be careful with model cache paths. RunPod deployments may rely on `/runpod-volume` to avoid cold downloads.
- Do not print secrets. The worker handles database and AWS credentials.
- Keep retry and batching changes conservative; worker bursts can pressure RDS and S3.

## Validation

Documentation-only changes:

```bash
git diff --check
```

Code changes should at least compile:

```bash
python -m py_compile handler.py entrypoint.py runtime_gemma_force_patch.py
```

Container or dependency changes should build the Docker image before deployment.
