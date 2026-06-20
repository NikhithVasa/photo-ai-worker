import runpod

import runtime_gemma_force_patch  # noqa: F401 - applies monkey patches on import
import handler


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler.handler})
