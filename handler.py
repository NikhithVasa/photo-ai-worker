import gc
import json
import os
import re
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
import psycopg2
import runpod
from PIL import Image, ImageDraw
from psycopg2.extras import Json, RealDictCursor


# ============================================================
# ENV CONFIG
# ============================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "nikhith-ai-photo-gallery-dev")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

RDS_HOST = os.environ.get("RDS_HOST")
RDS_PORT = int(os.environ.get("RDS_PORT", "5432"))
RDS_DB = os.environ.get("RDS_DB")
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

LOCAL_WORK = Path(os.environ.get("LOCAL_WORK", "/tmp/photo-qwen-worker"))
LOCAL_WORK.mkdir(parents=True, exist_ok=True)

RESET_EXISTING_QWEN = os.environ.get("RESET_EXISTING_QWEN", "false").lower() == "true"
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_IMAGE_MAX_SIDE = int(os.environ.get("QWEN_IMAGE_MAX_SIDE", "448"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "320"))
QWEN_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_INFERENCE_BATCH_SIZE", "4"))

TEXT_EMBED_MODEL_ID = os.environ.get("TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBED_BATCH_SIZE = int(os.environ.get("TEXT_EMBED_BATCH_SIZE", "64"))

os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/runpod-volume/huggingface")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/runpod-volume/huggingface/sentence-transformers")
os.environ.setdefault("TORCH_HOME", "/runpod-volume/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/runpod-volume/cache")
for cache_dir in [
    os.environ["HF_HOME"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["TORCH_HOME"],
    os.environ["XDG_CACHE_HOME"],
]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_PROCESS_VISION_INFO = None
_TEXT_EMBED_MODEL = None
_DB_POOL = None
_TABLE_COLUMNS_CACHE: Dict[str, set[str]] = {}

DB_POOL_MIN_CONN = int(os.environ.get("DB_POOL_MIN_CONN", "1"))
DB_POOL_MAX_CONN = int(os.environ.get("DB_POOL_MAX_CONN", "2"))
DB_CONNECT_RETRIES = int(os.environ.get("DB_CONNECT_RETRIES", "8"))
DB_CONNECT_BASE_SLEEP = float(os.environ.get("DB_CONNECT_BASE_SLEEP", "0.75"))
DB_APPLICATION_NAME = os.environ.get("DB_APPLICATION_NAME", "photo_qwen_worker")


# ============================================================
# DB HELPERS
# ============================================================

def assert_env_ready() -> None:
    missing = []
    for key, value in {
        "RDS_HOST": RDS_HOST,
        "RDS_DB": RDS_DB,
        "RDS_USER": RDS_USER,
        "RDS_PASSWORD": RDS_PASSWORD,
        "S3_BUCKET": S3_BUCKET,
    }.items():
        if not value:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


class PooledDbConnection:
    def __init__(self, pool_obj, conn):
        self._pool = pool_obj
        self._conn = conn
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def close(self):
        if self._returned:
            return
        self._returned = True
        try:
            if self._conn and not self._conn.closed:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(self._conn)
        except Exception:
            try:
                self._pool.putconn(self._conn, close=True)
            except Exception:
                pass


def _connect_kwargs() -> Dict[str, Any]:
    return {
        "host": RDS_HOST,
        "port": RDS_PORT,
        "dbname": RDS_DB,
        "user": RDS_USER,
        "password": RDS_PASSWORD,
        "cursor_factory": RealDictCursor,
        "application_name": DB_APPLICATION_NAME,
        "connect_timeout": int(os.environ.get("DB_CONNECT_TIMEOUT", "10")),
        "keepalives": 1,
        "keepalives_idle": int(os.environ.get("DB_KEEPALIVES_IDLE", "30")),
        "keepalives_interval": int(os.environ.get("DB_KEEPALIVES_INTERVAL", "10")),
        "keepalives_count": int(os.environ.get("DB_KEEPALIVES_COUNT", "3")),
    }


def _init_db_pool():
    global _DB_POOL
    if _DB_POOL is not None:
        return _DB_POOL
    assert_env_ready()
    from psycopg2.pool import SimpleConnectionPool

    last_err = None
    for attempt in range(DB_CONNECT_RETRIES):
        try:
            _DB_POOL = SimpleConnectionPool(DB_POOL_MIN_CONN, DB_POOL_MAX_CONN, **_connect_kwargs())
            print(f"DB pool initialized min={DB_POOL_MIN_CONN} max={DB_POOL_MAX_CONN} app={DB_APPLICATION_NAME}", flush=True)
            return _DB_POOL
        except psycopg2.OperationalError as e:
            last_err = e
            sleep_for = min(20.0, DB_CONNECT_BASE_SLEEP * (2 ** attempt))
            print(f"DB pool init failed {attempt + 1}/{DB_CONNECT_RETRIES}: {repr(e)}; sleeping {sleep_for:.2f}s", flush=True)
            time.sleep(sleep_for)
    raise last_err


def get_conn():
    pool_obj = _init_db_pool()
    last_err = None
    for attempt in range(DB_CONNECT_RETRIES):
        try:
            conn = pool_obj.getconn()
            if conn.closed:
                pool_obj.putconn(conn, close=True)
                raise psycopg2.OperationalError("pooled connection was closed")
            return PooledDbConnection(pool_obj, conn)
        except psycopg2.OperationalError as e:
            last_err = e
            sleep_for = min(20.0, DB_CONNECT_BASE_SLEEP * (2 ** attempt))
            print(f"DB connection failed {attempt + 1}/{DB_CONNECT_RETRIES}: {repr(e)}; sleeping {sleep_for:.2f}s", flush=True)
            time.sleep(sleep_for)
    raise last_err


def db_one(sql: str, params: Tuple[Any, ...] = ()):  # noqa: ANN401
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        conn.close()


def db_all(sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def execute_sql(sql: str, params: Tuple[Any, ...] = ()) -> None:
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    finally:
        conn.close()


def table_columns(table_name: str) -> set[str]:
    cached = _TABLE_COLUMNS_CACHE.get(table_name)
    if cached is not None:
        return cached
    rows = db_all(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s;
        """,
        (table_name,),
    )
    cols = {r["column_name"] for r in rows}
    _TABLE_COLUMNS_CACHE[table_name] = cols
    return cols


# ============================================================
# SMALL HELPERS
# ============================================================

def update_job_status(job_id: str, status: str, step: str, message: str = "", extra: Optional[Dict[str, Any]] = None):
    print("[JOB_STATUS]", {"job_id": job_id, "status": status, "step": step, "message": message, "extra": extra or {}}, flush=True)


def vector_to_pg(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def download_file(key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, key, str(local_path))
    return local_path


def upload_file(local_path: Path, key: str, content_type: str) -> None:
    s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs={"ContentType": content_type})


def make_qwen_image(input_path: Path, max_side: int = QWEN_IMAGE_MAX_SIDE) -> Path:
    img = Image.open(input_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = input_path.parent / ("qwen_" + input_path.stem + ".jpg")
    img.save(out, "JPEG", quality=82)
    return out


# ============================================================
# ALBUM / EVENT
# ============================================================

def restore_album_context(album_slug: str, album_name: Optional[str] = None) -> Dict[str, Any]:
    album = db_one("SELECT * FROM albums WHERE slug=%s LIMIT 1;", (album_slug,))
    if not album:
        raise RuntimeError(f"Album not found: {album_slug}. Run face-worker ingest first.")
    return {"album_id": str(album["id"]), "album_slug": album["slug"], "album_name": album.get("name")}


def upsert_events(album_ctx: Dict[str, Any], events: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # Keep this idempotent so qwen-worker can be called with the same payload as face-worker.
    album_id = album_ctx["album_id"]
    restored_events = []
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for event in events:
                    name = event["name"]
                    slug = event["slug"]
                    source_prefix = event.get("source_prefix") or ""
                    cur.execute(
                        """
                        INSERT INTO album_events(album_id, name, slug, source_prefix, sort_order, is_deleted, created_at, updated_at)
                        VALUES(
                            %s::uuid,
                            %s,
                            %s,
                            %s,
                            COALESCE((SELECT MAX(sort_order) + 1 FROM album_events WHERE album_id=%s::uuid), 1),
                            false,
                            now(),
                            now()
                        )
                        ON CONFLICT(album_id, slug)
                        DO UPDATE SET name=EXCLUDED.name, source_prefix=COALESCE(NULLIF(EXCLUDED.source_prefix, ''), album_events.source_prefix), is_deleted=false, updated_at=now()
                        RETURNING *;
                        """,
                        (album_id, name, slug, source_prefix, album_id),
                    )
                    row = cur.fetchone()
                    restored_events.append({
                        "event_id": str(row["id"]),
                        "name": row["name"],
                        "slug": row["slug"],
                        "source_prefix": row.get("source_prefix") or source_prefix,
                    })
    finally:
        conn.close()
    return restored_events


# ============================================================
# QWEN MODEL
# ============================================================

def install_json_repair_if_needed():
    try:
        from json_repair import repair_json
        return repair_json
    except Exception:
        subprocess.check_call(["python3", "-m", "pip", "install", "json-repair"])
        from json_repair import repair_json
        return repair_json


def load_qwen():
    global _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO
    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading Qwen model: {QWEN_MODEL_ID}", flush=True)
    print("torch cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _QWEN_MODEL = model
    _QWEN_PROCESSOR = processor
    _PROCESS_VISION_INFO = process_vision_info
    print("Qwen loaded", flush=True)
    return _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO


def qwen_prompt() -> str:
    return """
Return only valid minified JSON. No markdown. No explanation.
The image may have green boxes labeled Person 1, Person 2, etc.
Use only visible labels. Do not identify real people. Do not invent details.
If uncertain, use "uncertain" or 0.

Schema:
{
  "caption": "",
  "scene": "",
  "decoration_present": true,
  "decoration_keywords": "",
  "background_quality": 0,
  "frame_clarity": 0,
  "camera_gaze": {
    "overall": "all|some|none|uncertain",
    "people": {
      "Person 1": "looking_at_camera|not_looking|uncertain"
    }
  },
  "album_worthy_score": 0,
  "album_worthy_reason": "",
  "people": {
    "Person 1": {
      "visible_keywords": "",
      "jewelry_count": {
        "bangles": 0,
        "necklace": 0,
        "earrings": 0,
        "rings": 0,
        "head_jewelry": 0,
        "other": 0
      },
      "jewelry_keywords": "",
      "photo_quality_score": 0
    }
  },
  "search_text": ""
}

Rules:
- Scores are integers from 0 to 10.
- frame_clarity means sharp, clear, good for printing/framing.
- background_quality means clean, pretty, visually pleasing background.
- album_worthy_score means whether this photo deserves to be in a wow wedding album.
- decoration_present means visible stage, flowers, lights, mandap, backdrop, decor, or venue decoration.
- Count jewelry by category only.
- Include only visible labeled people.
- Keep all strings short.
- No trailing commas.
"""


def extract_json(text: str) -> Dict[str, Any]:
    repair_json = install_json_repair_if_needed()
    raw = (text or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        raw = raw[start:end + 1]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(repair_json(raw))


def clamp_score(v: Any) -> int:
    try:
        return max(0, min(10, int(round(float(v)))))
    except Exception:
        return 0


def safe_int(v: Any) -> int:
    try:
        return max(0, int(v))
    except Exception:
        return 0


def normalize_qwen_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Qwen output is not dict")
    caption = str(data.get("caption") or "")[:300]
    scene = str(data.get("scene") or "")[:500]
    search_text = str(data.get("search_text") or "")[:3000]
    decoration_present = bool(data.get("decoration_present") or False)
    decoration_keywords = str(data.get("decoration_keywords") or "")[:500]
    background_quality = clamp_score(data.get("background_quality"))
    frame_clarity = clamp_score(data.get("frame_clarity"))
    album_worthy_score = clamp_score(data.get("album_worthy_score"))
    album_worthy_reason = str(data.get("album_worthy_reason") or "")[:500]

    camera_gaze = data.get("camera_gaze") or {}
    if not isinstance(camera_gaze, dict):
        camera_gaze = {}
    gaze_overall = str(camera_gaze.get("overall") or "uncertain")
    gaze_people = camera_gaze.get("people") or {}
    if not isinstance(gaze_people, dict):
        gaze_people = {}

    people_obj = data.get("people") or {}
    if not isinstance(people_obj, dict):
        people_obj = {}
    normalized_people: Dict[str, Any] = {}
    for label, value in people_obj.items():
        label = str(label).strip()
        if not label.startswith("Person "):
            continue
        if not isinstance(value, dict):
            value = {"visible_keywords": str(value or "")}
        jc = value.get("jewelry_count") or {}
        if not isinstance(jc, dict):
            jc = {}
        normalized_people[label] = {
            "visible_keywords": str(value.get("visible_keywords") or "")[:500],
            "jewelry_count": {
                "bangles": safe_int(jc.get("bangles")),
                "necklace": safe_int(jc.get("necklace")),
                "earrings": safe_int(jc.get("earrings")),
                "rings": safe_int(jc.get("rings")),
                "head_jewelry": safe_int(jc.get("head_jewelry")),
                "other": safe_int(jc.get("other")),
            },
            "jewelry_keywords": str(value.get("jewelry_keywords") or "")[:500],
            "camera_gaze": str(gaze_people.get(label) or "uncertain"),
            "photo_quality_score": clamp_score(value.get("photo_quality_score")),
        }

    quality_keywords = (
        f"decoration_present={decoration_present}; "
        f"decoration={decoration_keywords}; "
        f"background_quality={background_quality}/10; "
        f"frame_clarity={frame_clarity}/10; "
        f"camera_gaze={gaze_overall}; "
        f"album_worthy_score={album_worthy_score}/10; "
        f"album_worthy_reason={album_worthy_reason}"
    )
    merged_search_text = " | ".join([caption, scene, search_text, quality_keywords]).strip()
    return {
        "photo": {
            "caption": caption,
            "detailed_description": scene,
            "scene_context": scene,
            "search_text": merged_search_text,
        },
        "quality": {
            "decoration_present": decoration_present,
            "decoration_keywords": decoration_keywords,
            "background_quality": background_quality,
            "frame_clarity": frame_clarity,
            "camera_gaze_overall": gaze_overall,
            "album_worthy_score": album_worthy_score,
            "album_worthy_reason": album_worthy_reason,
        },
        "people_map": normalized_people,
        "relationships": [],
        "raw": data,
    }


def qwen_describe_batch(image_paths: List[Path]) -> List[Dict[str, Any]]:
    import torch

    model, processor, process_vision_info = load_qwen()
    messages = [
        {"role": "user", "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": qwen_prompt()}]}
        for path in image_paths
    ]
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    all_image_inputs = []
    all_video_inputs = []
    for m in messages:
        image_inputs, video_inputs = process_vision_info([m])
        all_image_inputs.extend(image_inputs or [])
        if video_inputs:
            all_video_inputs.extend(video_inputs)
    inputs = processor(
        text=texts,
        images=all_image_inputs,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
    outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del inputs, gen, trimmed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [normalize_qwen_data(extract_json(o)) for o in outs]


# ============================================================
# QWEN DB / ANNOTATION
# ============================================================

def patch_annotated_keys(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]
    execute_sql(
        """
        UPDATE photos
        SET photo_uuid=COALESCE(photo_uuid, gen_random_uuid())
        WHERE album_id=%s::uuid AND album_event_id=ANY(%s::uuid[]);

        UPDATE photos p
        SET storage_album_slug=COALESCE(p.storage_album_slug, a.slug),
            storage_event_slug=COALESCE(p.storage_event_slug, e.slug),
            updated_at=now()
        FROM albums a
        JOIN album_events e ON e.album_id=a.id
        WHERE p.album_id=a.id
          AND p.album_event_id=e.id
          AND p.album_id=%s::uuid
          AND p.album_event_id=ANY(%s::uuid[]);

        UPDATE photos
        SET annotated_s3_key='albums/' || storage_album_slug || '/events/' || storage_event_slug || '/annotated/' || photo_uuid::text || '.jpg',
            updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[])
          AND COALESCE(is_deleted, false)=false
          AND (annotated_s3_key IS NULL OR annotated_s3_key='');
        """,
        (album_id, event_ids, album_id, event_ids, album_id, event_ids),
    )


def mark_no_labeled_face_photos_skipped(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    execute_sql(
        """
        UPDATE photos p
        SET qwen_status='skipped_no_labeled_faces',
            qwen_error='No labeled faces for person-labeled Qwen metadata',
            updated_at=now()
        WHERE p.album_id=%s::uuid
          AND p.album_event_id=ANY(%s::uuid[])
          AND p.face_index_status='completed'
          AND COALESCE(p.is_deleted, false)=false
          AND p.qwen_status IN ('pending', 'failed')
          AND NOT EXISTS (
              SELECT 1 FROM faces f WHERE f.photo_id=p.id AND f.person_id IS NOT NULL
          );
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events]),
    )


def reset_retryable_qwen_failures(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    execute_sql(
        """
        UPDATE photos
        SET qwen_status='pending', qwen_error=NULL, updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[])
          AND qwen_status='failed'
          AND (
              qwen_error LIKE 'JSONDecodeError%%'
              OR qwen_error LIKE 'ParamValidationError%%'
              OR qwen_error LIKE 'ValueError%%'
              OR qwen_error LIKE 'RuntimeError%%'
          );
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events]),
    )


def maybe_reset_existing_qwen(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    if not RESET_EXISTING_QWEN:
        return
    execute_sql(
        """
        UPDATE photos
        SET qwen_status='pending',
            qwen_error=NULL,
            caption=NULL,
            ai_description=NULL,
            search_text=NULL,
            qwen_json=NULL,
            search_embedding=NULL,
            search_index_status='pending',
            search_index_error=NULL,
            updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[])
          AND COALESCE(is_deleted, false)=false;

        UPDATE photo_people
        SET qwen_description=NULL,
            qwen_json=NULL,
            search_embedding=NULL,
            updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[]);
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events], album_ctx["album_id"], [e["event_id"] for e in events]),
    )


def fetch_qwen_faces_by_photo(photo_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not photo_ids:
        return {}
    rows = db_all(
        """
        SELECT f.*, pe.person_number
        FROM faces f
        JOIN people pe ON pe.id=f.person_id
        WHERE f.photo_id=ANY(%s::uuid[])
          AND f.person_id IS NOT NULL
        ORDER BY f.photo_id, pe.person_number;
        """,
        (photo_ids,),
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["photo_id"]), []).append(row)
    return grouped


def fetch_event_map(event_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not event_ids:
        return {}
    rows = db_all("SELECT id, name, slug FROM album_events WHERE id=ANY(%s::uuid[]);", (event_ids,))
    return {str(r["id"]): {"name": r["name"], "slug": r["slug"]} for r in rows}


def annotate_photo_with_faces(photo: Dict[str, Any], faces: List[Dict[str, Any]]) -> Path:
    if not faces:
        raise RuntimeError("No labeled faces for photo")
    tmpdir = LOCAL_WORK / "annotated" / str(photo["id"])
    tmpdir.mkdir(parents=True, exist_ok=True)
    local = tmpdir / "ai.webp"
    download_file(photo["ai_input_s3_key"], local)
    img = Image.open(local).convert("RGB")
    draw = ImageDraw.Draw(img)
    for f in faces:
        label = f"Person {f['person_number']}"
        x1, y1, x2, y2 = int(f["bbox_x1"]), int(f["bbox_y1"]), int(f["bbox_x2"]), int(f["bbox_y2"])
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)
        tw = 9 * len(label) + 10
        label_y1 = max(0, y1 - 24)
        draw.rectangle((x1, label_y1, x1 + tw, y1), fill=(0, 255, 0))
        draw.text((x1 + 4, max(0, y1 - 21)), label, fill=(0, 0, 0))
    ann = tmpdir / "annotated.jpg"
    img.save(ann, "JPEG", quality=86)
    upload_file(ann, photo["annotated_s3_key"], "image/jpeg")
    return make_qwen_image(ann)


def get_label_to_id(album_ctx: Dict[str, Any]) -> Dict[str, str]:
    people = db_all("SELECT id, person_number FROM people WHERE album_id=%s::uuid;", (album_ctx["album_id"],))
    return {f"Person {p['person_number']}": str(p["id"]) for p in people}


def save_qwen_photo(photo: Dict[str, Any], data: Dict[str, Any], label_to_id: Dict[str, str], event_by_id: Dict[str, Dict[str, Any]]) -> None:
    photo_data = data.get("photo", {})
    people_map = data.get("people_map", {})
    event_row = event_by_id.get(str(photo["album_event_id"]))
    if event_row:
        data["event"] = {"name": event_row["name"], "slug": event_row["slug"]}

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE photos
                    SET caption=%s,
                        ai_description=%s,
                        search_text=%s,
                        qwen_json=%s,
                        qwen_status='completed',
                        qwen_error=NULL,
                        search_index_status='pending',
                        search_index_error=NULL,
                        updated_at=now()
                    WHERE id=%s::uuid;
                    """,
                    (
                        photo_data.get("caption"),
                        photo_data.get("detailed_description") or photo_data.get("scene_context"),
                        photo_data.get("search_text"),
                        Json(data),
                        str(photo["id"]),
                    ),
                )
                for label, person_data in people_map.items():
                    person_id = label_to_id.get(label)
                    if not person_id:
                        continue
                    qwen_desc = " | ".join([
                        str(person_data.get("visible_keywords") or ""),
                        str(person_data.get("jewelry_keywords") or ""),
                        f"camera_gaze={person_data.get('camera_gaze', 'uncertain')}",
                        f"photo_quality_score={person_data.get('photo_quality_score', 0)}/10",
                    ]).strip()
                    cur.execute(
                        """
                        UPDATE photo_people
                        SET qwen_description=%s,
                            qwen_json=%s,
                            search_text=COALESCE(search_text, '') || ' | ' || %s,
                            search_embedding=NULL,
                            updated_at=now()
                        WHERE photo_id=%s::uuid AND person_id=%s::uuid;
                        """,
                        (qwen_desc, Json(person_data), qwen_desc, str(photo["id"]), person_id),
                    )
    finally:
        conn.close()


def mark_qwen_failed(photo_id: str, err: str) -> None:
    execute_sql(
        """
        UPDATE photos
        SET qwen_status='failed', qwen_error=%s, updated_at=now()
        WHERE id=%s::uuid;
        """,
        (err[:3000], photo_id),
    )


def run_qwen_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    patch_annotated_keys(album_ctx, events)
    maybe_reset_existing_qwen(album_ctx, events)
    reset_retryable_qwen_failures(album_ctx, events)
    mark_no_labeled_face_photos_skipped(album_ctx, events)

    rows = db_all(
        """
        SELECT *
        FROM photos p
        WHERE p.album_id=%s::uuid
          AND p.album_event_id=ANY(%s::uuid[])
          AND p.compression_status='completed'
          AND p.face_index_status='completed'
          AND COALESCE(p.is_deleted, false)=false
          AND p.qwen_status IN ('pending', 'failed')
          AND EXISTS (
              SELECT 1 FROM faces f WHERE f.photo_id=p.id AND f.person_id IS NOT NULL
          )
        ORDER BY p.created_at;
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events]),
    )

    label_to_id = get_label_to_id(album_ctx)
    event_by_id = fetch_event_map([e["event_id"] for e in events])
    ok = failed = 0
    errors = []

    for batch_start in range(0, len(rows), QWEN_INFERENCE_BATCH_SIZE):
        batch_rows = rows[batch_start:batch_start + QWEN_INFERENCE_BATCH_SIZE]
        faces_by_photo = fetch_qwen_faces_by_photo([str(r["id"]) for r in batch_rows])
        prepared_rows = []
        image_paths = []
        for row in batch_rows:
            photo_id = str(row["id"])
            try:
                path = annotate_photo_with_faces(row, faces_by_photo.get(photo_id, []))
                prepared_rows.append(row)
                image_paths.append(path)
            except Exception as e:
                failed += 1
                err = repr(e)
                errors.append({"photo_id": photo_id, "error": err})
                mark_qwen_failed(photo_id, err)

        if not image_paths:
            continue

        try:
            outputs = qwen_describe_batch(image_paths)
        except Exception as e:
            err = repr(e)
            for row in prepared_rows:
                failed += 1
                photo_id = str(row["id"])
                errors.append({"photo_id": photo_id, "error": err})
                mark_qwen_failed(photo_id, err)
            continue

        for row, data in zip(prepared_rows, outputs):
            photo_id = str(row["id"])
            try:
                save_qwen_photo(row, data, label_to_id, event_by_id)
                ok += 1
            except Exception as e:
                failed += 1
                err = repr(e)
                errors.append({"photo_id": photo_id, "error": err})
                mark_qwen_failed(photo_id, err)

    return {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25]}


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def load_text_embed_model():
    global _TEXT_EMBED_MODEL
    if _TEXT_EMBED_MODEL is not None:
        return _TEXT_EMBED_MODEL
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading text embedding model {TEXT_EMBED_MODEL_ID} on {device}", flush=True)
    _TEXT_EMBED_MODEL = SentenceTransformer(TEXT_EMBED_MODEL_ID, device=device)
    return _TEXT_EMBED_MODEL


def encode_texts(texts: List[str]) -> List[str]:
    model = load_text_embed_model()
    arr = model.encode(texts, batch_size=TEXT_EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    return [vector_to_pg(v) for v in arr]


def run_photo_text_embeddings(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = db_all(
        """
        SELECT id, search_text
        FROM photos
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[])
          AND COALESCE(is_deleted, false)=false
          AND qwen_status='completed'
          AND COALESCE(search_text, '') <> ''
          AND search_index_status IN ('pending', 'failed')
        ORDER BY created_at;
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events]),
    )
    ok = failed = 0
    errors = []
    for start in range(0, len(rows), TEXT_EMBED_BATCH_SIZE):
        batch = rows[start:start + TEXT_EMBED_BATCH_SIZE]
        try:
            vectors = encode_texts([str(r["search_text"]) for r in batch])
            conn = get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        for row, vector in zip(batch, vectors):
                            cur.execute(
                                """
                                UPDATE photos
                                SET search_embedding=%s::vector,
                                    search_index_status='completed',
                                    search_index_error=NULL,
                                    updated_at=now()
                                WHERE id=%s::uuid;
                                """,
                                (vector, str(row["id"])),
                            )
                            ok += 1
            finally:
                conn.close()
        except Exception as e:
            err = repr(e)
            failed += len(batch)
            errors.append({"batch_start": start, "error": err})
            execute_sql(
                """
                UPDATE photos
                SET search_index_status='failed', search_index_error=%s, updated_at=now()
                WHERE id=ANY(%s::uuid[]);
                """,
                (err[:3000], [str(r["id"]) for r in batch]),
            )
    return {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25]}


def run_photo_people_text_embeddings(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if "search_embedding" not in table_columns("photo_people"):
        return {"skipped": True, "reason": "photo_people.search_embedding column missing"}
    rows = db_all(
        """
        SELECT id,
               TRIM(COALESCE(search_text, '') || ' | ' || COALESCE(qwen_description, '')) AS text
        FROM photo_people
        WHERE album_id=%s::uuid
          AND album_event_id=ANY(%s::uuid[])
          AND TRIM(COALESCE(search_text, '') || COALESCE(qwen_description, '')) <> ''
          AND search_embedding IS NULL
        ORDER BY created_at;
        """,
        (album_ctx["album_id"], [e["event_id"] for e in events]),
    )
    ok = failed = 0
    errors = []
    for start in range(0, len(rows), TEXT_EMBED_BATCH_SIZE):
        batch = rows[start:start + TEXT_EMBED_BATCH_SIZE]
        try:
            vectors = encode_texts([str(r["text"]) for r in batch])
            conn = get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        for row, vector in zip(batch, vectors):
                            cur.execute("UPDATE photo_people SET search_embedding=%s::vector, updated_at=now() WHERE id=%s::uuid;", (vector, str(row["id"])))
                            ok += 1
            finally:
                conn.close()
        except Exception as e:
            failed += len(batch)
            errors.append({"batch_start": start, "error": repr(e)})
    return {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25]}


def run_text_embeddings_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "photos": run_photo_text_embeddings(album_ctx, events),
        "photo_people": run_photo_people_text_embeddings(album_ctx, events),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def normalize_steps(payload: Dict[str, Any]) -> Dict[str, bool]:
    default_steps = {"qwen": True, "embeddings": True, "cleanup_temp": False}
    if payload.get("qwen_full_mode", payload.get("full_mode", False)):
        return {**default_steps, "cleanup_temp": bool(payload.get("cleanup_temp", False))}
    return {**default_steps, **(payload.get("steps") or {})}


def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "album_slug" not in payload:
        raise ValueError("Missing input.album_slug")
    if "events" not in payload or not payload["events"]:
        raise ValueError("Missing input.events")

    steps = normalize_steps(payload)
    update_job_status(job_id, "running", "restore_album", "Restoring album context")
    album_ctx = restore_album_context(payload["album_slug"], payload.get("album_name"))

    update_job_status(job_id, "running", "upsert_events", "Restoring event rows")
    db_events = upsert_events(album_ctx, payload["events"])
    results: Dict[str, Any] = {"album": album_ctx, "events": db_events, "steps": {}}

    if steps.get("qwen"):
        update_job_status(job_id, "running", "qwen", "Running Qwen metadata")
        results["steps"]["qwen"] = run_qwen_for_events(album_ctx, db_events)

    if steps.get("embeddings"):
        update_job_status(job_id, "running", "embeddings", "Generating text embeddings")
        results["steps"]["embeddings"] = run_text_embeddings_for_events(album_ctx, db_events)

    if steps.get("cleanup_temp"):
        import shutil
        shutil.rmtree(LOCAL_WORK, ignore_errors=True)
        LOCAL_WORK.mkdir(parents=True, exist_ok=True)
        results["steps"]["cleanup_temp"] = True

    return results


def debug_gpu() -> Dict[str, Any]:
    import torch
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def debug_db_connections() -> Dict[str, Any]:
    rows = db_all(
        """
        SELECT usename, application_name, client_addr::text AS client_addr, state, COUNT(*) AS connections
        FROM pg_stat_activity
        WHERE datname=current_database()
        GROUP BY usename, application_name, client_addr, state
        ORDER BY connections DESC;
        """
    )
    max_conn = db_one("SHOW max_connections;")
    total = db_one("SELECT COUNT(*) AS total_connections FROM pg_stat_activity;")
    return {
        "max_connections": max_conn.get("max_connections") if max_conn else None,
        "total_connections": int(total["total_connections"]) if total else None,
        "connections": rows,
        "db_pool_max_conn": DB_POOL_MAX_CONN,
        "db_application_name": DB_APPLICATION_NAME,
    }


def handler(event):
    started = time.time()
    payload = event.get("input") or event or {}
    job_id = str(payload.get("job_id") or event.get("id") or uuid.uuid4())
    try:
        if payload.get("debug_gpu"):
            return {"ok": True, "job_id": job_id, "debug_gpu": debug_gpu()}
        if payload.get("debug_db_connections"):
            return {"ok": True, "job_id": job_id, "debug_db_connections": debug_db_connections()}

        result = process_album_events(job_id, payload)
        update_job_status(job_id, "completed", "qwen_worker", "Qwen worker completed")
        return {"ok": True, "job_id": job_id, "execution_seconds": round(time.time() - started, 2), "result": result}
    except Exception as e:
        err = {"error": repr(e), "traceback": traceback.format_exc()}
        print("Qwen worker failed:", err, flush=True)
        update_job_status(job_id, "failed", "error", repr(e), err)
        return {"ok": False, "job_id": job_id, "execution_seconds": round(time.time() - started, 2), **err}
    finally:
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
