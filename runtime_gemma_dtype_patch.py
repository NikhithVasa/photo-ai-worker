"""Fix Gemma runtime input dtypes.

The legacy runtime Gemma patch moves the processor output to CUDA without
changing dtype. Some Gemma processor outputs can include uint8/Byte image
tensors, which later fail inside Gemma vision LayerNorm with:

    RuntimeError: "LayerNormKernelImpl" not implemented for 'Byte'

This patch is intentionally small and imported after runtime_gemma_force_patch.
It replaces only the input-moving helper used by _gemma_describe_one.
"""

from __future__ import annotations

from typing import Any

import runtime_gemma_force_patch as gemma_patch


def _model_device_and_dtype(model: Any):
    import torch

    try:
        param = next(model.parameters())
        device = param.device
    except Exception:
        device = getattr(model, "device", "cuda")

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    return device, model_dtype


def _move_value_to_model(key: str, value: Any, device: Any, model_dtype: Any) -> Any:
    import torch

    if not torch.is_tensor(value):
        return value

    key_lower = str(key).lower()

    # Token ids, masks, and shape metadata must stay integer/bool.
    if key_lower in {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "cache_position",
        "image_grid_thw",
        "video_grid_thw",
    }:
        return value.to(device)

    # Gemma vision tensors must be floating-point. A uint8 tensor reaching
    # LayerNorm causes: LayerNormKernelImpl not implemented for 'Byte'.
    if "pixel" in key_lower or "image" in key_lower or "vision" in key_lower:
        moved = value.to(device=device)
        if not moved.is_floating_point():
            moved = moved.to(dtype=model_dtype)
            try:
                if moved.numel() and float(moved.detach().max().item()) > 1.5:
                    moved = moved / 255.0
            except Exception:
                pass
        else:
            moved = moved.to(dtype=model_dtype)
        return moved

    if value.is_floating_point():
        return value.to(device=device, dtype=model_dtype)

    return value.to(device)


def _move_inputs_to_model_fixed(inputs: Any, model: Any) -> Any:
    device, model_dtype = _model_device_and_dtype(model)

    if hasattr(inputs, "items"):
        return {
            key: _move_value_to_model(key, value, device, model_dtype)
            for key, value in inputs.items()
        }

    try:
        return inputs.to(device)
    except Exception:
        return inputs


gemma_patch._move_inputs_to_model = _move_inputs_to_model_fixed

print("Gemma dtype patch installed: image tensors forced to floating dtype", flush=True)
