import os

import runpod

import handler

if not hasattr(handler, "IMAGE_TEXT_MODEL_PROVIDER"):
    handler.IMAGE_TEXT_MODEL_PROVIDER = os.getenv("IMAGE_TEXT_MODEL_PROVIDER", "gemma")

import runtime_gemma_force_patch  # noqa: E402,F401 - applies monkey patches on import
import runtime_gemma_dtype_patch  # noqa: E402,F401 - fixes Gemma image tensor dtypes


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler.handler})
