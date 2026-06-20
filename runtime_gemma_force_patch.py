

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List

import handler as h

_ORIGINAL_PROCESS_ALBUM_EVENTS = h.process_album_events
_GEMMA_MODEL = None
_GEMMA_PROCESSOR = None
_FORCE_FLAG_NAMES = (
    "force_gemma",
    "overwrite_gemma",
    "force_qwen",
    "overwrite_qwen",
    "force_image_text",
    "overwrite_image_text",
    "reset_existing_qwen",
    "reset_existing_gemma",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "force", "overwrite"}
    return bool(value)


def _image_text_model_provider() -> str:
    provider = (
        getattr(h, "IMAGE_TEXT_MODEL_PROVIDER", None)
        or os.getenv("IMAGE_TEXT_MODEL_PROVIDER")
        or os.getenv("VISION_MODEL_PROVIDER")
        or "gemma"
    )
    provider = str(provider).strip().lower()

    if provider not in {"qwen", "gemma"}:
        raise RuntimeError(
            f"IMAGE_TEXT_MODEL_PROVIDER must be qwen or gemma; got {provider!r}"
        )

    h.IMAGE_TEXT_MODEL_PROVIDER = provider
    return provider


def _gemma_quantization() -> str:
    quantization = (os.getenv("GEMMA_QUANTIZATION") or "4bit").strip().lower()

    if quantization not in {"none", "4bit", "8bit"}:
        raise RuntimeError("GEMMA_QUANTIZATION must be one of: none, 4bit, 8bit")

    return quantization


def _load_gemma():
    global _GEMMA_MODEL, _GEMMA_PROCESSOR

    if _GEMMA_MODEL is not None and _GEMMA_PROCESSOR is not None:
        return _GEMMA_MODEL, _GEMMA_PROCESSOR

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForMultimodalLM as GemmaModelClass
    except Exception:
        from transformers import AutoModelForImageTextToText as GemmaModelClass

    if hasattr(h, "assert_torch_gpu_ready"):
        h.assert_torch_gpu_ready("Gemma")

    model_id = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-12B-it")
    quantization = _gemma_quantization()
    attn_impl = os.getenv("GEMMA_ATTN_IMPLEMENTATION", "sdpa")
    token = os.getenv("HF_TOKEN") or None

    print(
        f"Loading Gemma model: {model_id}, "
        f"quantization={quantization}, attn={attn_impl}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )

    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto",
        "trust_remote_code": True,
    }

    if token:
        model_kwargs["token"] = token

    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    # Gemma 4 Unified casts raw image/audio inputs to these projection weights'
    # dtypes. Quantizing them stores the weights as uint8 and breaks normalization.
    bnb_skip_modules = [
        "lm_head",
        "model.embed_vision",
        "model.embed_audio",
    ]

    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=bnb_skip_modules,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=bnb_skip_modules,
        )

    try:
        model = GemmaModelClass.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = GemmaModelClass.from_pretrained(model_id, **model_kwargs)

    model.eval()

    _GEMMA_MODEL = model
    _GEMMA_PROCESSOR = processor

    print("Gemma loaded", flush=True)
    return _GEMMA_MODEL, _GEMMA_PROCESSOR


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(model, "device", "cuda")

    try:
        return inputs.to(device)
    except Exception:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }


def _gemma_describe_one(image_path: Path) -> Dict[str, Any]:
    import torch

    model, processor = _load_gemma()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": h.qwen_prompt()},
            ],
        }
    ]

    template_kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }

    try:
        inputs = processor.apply_chat_template(
            messages,
            enable_thinking=_truthy(
                os.getenv("GEMMA_ENABLE_THINKING", "false")
            ),
            **template_kwargs,
        )
    except TypeError:
        inputs = processor.apply_chat_template(messages, **template_kwargs)

    inputs = _move_inputs_to_model(inputs, model)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("GEMMA_MAX_NEW_TOKENS", "320")),
            do_sample=False,
        )

    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h.normalize_qwen_data(h.extract_json(response))


def _gemma_describe_batch(image_paths: List[Path]) -> List[Dict[str, Any]]:
    print(
        "Image-to-text provider=gemma, "
        f"model={os.getenv('GEMMA_MODEL_ID', 'google/gemma-4-12B-it')}, "
        f"images={len(image_paths)}",
        flush=True,
    )

    return [_gemma_describe_one(Path(path)) for path in image_paths]


def _force_image_text_requested(payload: Dict[str, Any], steps: Dict[str, Any]) -> bool:
    return any(_truthy(payload.get(name)) or _truthy(steps.get(name)) for name in _FORCE_FLAG_NAMES)


def rich_image_text_prompt() -> str:
    return """
Return only valid minified JSON. No markdown. No explanation.
The image may have green boxes labeled Person 1, Person 2, etc.

CRITICAL PERSON RULES:
- Use ONLY the visible green labels exactly as written: "Person 1", "Person 2", etc.
- Do NOT identify real people, celebrities, bride/groom names, family names, or infer identity.
- Do NOT create keys like "bride", "groom", "man", "woman", or real names inside people.
- The people object keys must be exact visible Person labels only.
- You may describe role_guess such as bride, groom, guest, child, family, priest, performer, unknown, but only as a visual guess.
- Include only labeled people visible in the image. If a labeled person is mostly hidden, include them with uncertainty.
- Do not invent unseen details. If uncertain, use "uncertain", [], false, or 0.

Goal: produce dense searchable Indian wedding/event photo metadata for text-to-text semantic search, culling, album selection, and person-specific search. Be literal and exhaustive. Capture visible objects, clothes, colors, decor, background, pose, actions, emotions, quality, and likely user search phrases.

Schema:
{
  "caption": "one natural sentence, rich but concise",
  "scene": "3-6 factual sentences describing visible moment, setting, action, people, decor, mood, objects, background, and composition",
  "event_type_guess": "wedding|reception|haldi|sangeeth|engagement|pre-wedding|ceremony|ritual|portrait|group_photo|dance|party|details|unknown",
  "moment_keywords": ["short visible moment tags"],
  "action_keywords": ["standing","smiling","dancing","walking","ritual","posing","hugging","talking","eating","blessing"],
  "emotion_keywords": ["happy","serious","joyful","emotional","calm","romantic","funny","uncertain"],
  "photo_style": "candid|posed|portrait|group|wide_scene|detail_shot|action|ceremony|decor_detail|uncertain",
  "indoor_outdoor": "indoor|outdoor|mixed|uncertain",
  "venue_type": "mandap|stage|hall|garden|temple|home|street|beach|banquet|unknown",
  "time_of_day_guess": "day|night|evening|indoor_lighting|uncertain",
  "lighting_keywords": ["natural light","flash","warm lights","backlit","low light","harsh light","soft light","stage lighting"],
  "color_palette": ["visible dominant colors"],
  "outfit_keywords": ["saree","lehenga","sherwani","suit","kurta","dress","gown","dupatta","traditional","western"],
  "object_keywords": ["visible important objects like bouquet, garland, phone, chair, table, food, microphone, fire, flowers, lights"],
  "background_keywords": ["stage","mandap","flowers","curtains","crowd","trees","water","wall","decor","lights","clean background","busy background"],
  "decoration_present": true,
  "decoration_keywords": "visible stage, flowers, lights, mandap, backdrop, garlands, seating, props, venue decoration",
  "composition_keywords": ["close-up","full body","upper body","centered","wide","symmetry","crowded","clean background","blocked subject","low angle","high angle"],
  "technical_issues": ["blur","closed eyes","bad crop","overexposed","underexposed","blocked face","noise","none"],
  "background_quality": 0,
  "frame_clarity": 0,
  "album_worthy_score": 0,
  "album_worthy_reason": "specific visible reason for album/culling decision",
  "print_worthy_score": 0,
  "duplicate_risk": "low|medium|high|uncertain",
  "best_use": "hero_album|album_candidate|person_cover|group_memory|detail_memory|social_media|reject|uncertain",
  "camera_gaze": {
    "overall": "all|some|none|uncertain",
    "people": {"Person 1": "looking_at_camera|not_looking|eyes_closed|partially_visible|uncertain"}
  },
  "relationships": [
    {"people": ["Person 1", "Person 2"], "relationship_or_interaction": "standing together|hugging|ritual interaction|dancing together|posing together|uncertain"}
  ],
  "people": {
    "Person 1": {
      "visible_keywords": "literal visible description of this labeled person only",
      "role_guess": "bride|groom|guest|family|child|priest|performer|unknown",
      "clothing_keywords": "visible clothing, outfit type, fabric, pattern, embroidery, sleeves, dupatta/veil, footwear if visible",
      "clothing_colors": ["visible clothing colors"],
      "accessory_keywords": "watch, glasses, turban, dupatta, bouquet, phone, purse, sunglasses, etc.",
      "jewelry_count": {"bangles":0,"necklace":0,"earrings":0,"rings":0,"head_jewelry":0,"other":0},
      "jewelry_keywords": "visible jewelry details only",
      "pose_keywords": "standing, sitting, dancing, walking, blessing, holding hands, holding flowers, looking side, etc.",
      "expression": "smiling|laughing|serious|emotional|eyes_closed|neutral|uncertain",
      "face_visibility": "clear|partial|side_face|back_view|blocked|blurred|uncertain",
      "body_visibility": "face_only|upper_body|full_body|partial|uncertain",
      "camera_gaze": "looking_at_camera|not_looking|eyes_closed|partially_visible|uncertain",
      "occlusion_keywords": "blocked by person/object/none/uncertain",
      "personal_photo_quality_score": 0,
      "photo_quality_score": 0,
      "person_cover_score": 0,
      "search_phrases": ["person-specific phrases users might search"]
    }
  },
  "search_text": "dense literal searchable paragraph including event, action, clothing, colors, people labels, decor, background, objects, mood, quality, and likely phrases like wedding dress, red lehenga, groom sherwani, family group, dancing, stage lights"
}

Rules:
- Return exactly one JSON object matching the schema. No trailing commas.
- Scores are integers from 0 to 10.
- Keep arrays useful but not huge: usually 5-12 items.
- Make search_text dense and literal, not poetic.
- Include synonyms users may search only if visually supported, e.g. gown/dress/lehenga/saree when visible.
- Prefer visible evidence over assumptions.
""".strip()


def process_album_events_with_local_image_text(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = h.normalize_steps(payload)
    force_image_text = _force_image_text_requested(payload, steps)

    result = _ORIGINAL_PROCESS_ALBUM_EVENTS(job_id, payload)
    album_ctx = result.get("album")
    db_events = result.get("events") or []

    if not album_ctx or not db_events:
        return result

    should_run_image_text = bool(
        steps.get("qwen")
        or steps.get("gemma")
        or payload.get("run_qwen")
        or payload.get("run_gemma")
        or payload.get("run_image_text")
    )
    should_run_embeddings = bool(steps.get("embeddings") or payload.get("run_embeddings"))

    if should_run_image_text:
        provider = _image_text_model_provider()
        h.update_job_status(
            job_id,
            "running",
            "qwen",
            f"Running {provider} image-text metadata"
            + (" with overwrite" if force_image_text else ""),
        )

        old_reset = h.RESET_EXISTING_QWEN
        try:
            if force_image_text:
                h.RESET_EXISTING_QWEN = True
            qwen_result = h.run_qwen_for_events(album_ctx, db_events)
            qwen_result["force_reset"] = force_image_text
            qwen_result["image_text_provider"] = provider
            result.setdefault("steps", {})["qwen"] = qwen_result
        finally:
            h.RESET_EXISTING_QWEN = old_reset

    if should_run_embeddings:
        h.update_job_status(job_id, "running", "embeddings", "Generating text embeddings from image-text search_text")
        result.setdefault("steps", {})["embeddings"] = h.run_text_embeddings_for_events(album_ctx, db_events)

    if should_run_image_text or should_run_embeddings:
        h.update_job_status(job_id, "completed", "image_text_worker", "Image-text worker completed")

    return result


def apply_patch() -> None:
    provider = _image_text_model_provider()

    if provider == "gemma":
        h.qwen_describe_batch = _gemma_describe_batch
        h.QWEN_INFERENCE_BATCH_SIZE = max(
            1,
            int(os.getenv("GEMMA_INFERENCE_BATCH_SIZE", "1")),
        )
        print(
            "Gemma provider routing installed: "
            "handler.qwen_describe_batch -> Gemma",
            flush=True,
        )
    else:
        print(
            "Qwen provider selected; keeping handler.qwen_describe_batch",
            flush=True,
        )

    h.qwen_prompt = rich_image_text_prompt
    h.process_album_events = process_album_events_with_local_image_text
    print(
        f"Runtime Gemma/Qwen force patch installed provider={provider}",
        flush=True,
    )


apply_patch()
