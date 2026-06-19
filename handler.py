import os
import re
import io
import gc
import json
import time
import uuid
import math
import shutil
import traceback
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values, Json
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

import runpod


# ============================================================
# ENV CONFIG
# ============================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "nikhith-ai-photo-gallery-dev")
RDS_HOST = os.environ.get("RDS_HOST")
RDS_PORT = int(os.environ.get("RDS_PORT", "5432"))
RDS_DB = os.environ.get("RDS_DB")
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

LOCAL_WORK = Path(os.environ.get("LOCAL_WORK", "/tmp/photo-ai-worker"))
LOCAL_WORK.mkdir(parents=True, exist_ok=True)

DELETE_TEMP_AI_INPUT = os.environ.get("DELETE_TEMP_AI_INPUT", "false").lower() == "true"
DELETE_TEMP_ANNOTATED = os.environ.get("DELETE_TEMP_ANNOTATED", "false").lower() == "true"

# Keep false unless you explicitly want to regenerate completed Qwen/search text.
RESET_EXISTING_QWEN = os.environ.get("RESET_EXISTING_QWEN", "false").lower() == "true"

# Compression / AI image settings
AI_INPUT_MAX_SIDE = int(os.environ.get("AI_INPUT_MAX_SIDE", "1600"))
AI_INPUT_WEBP_QUALITY = int(os.environ.get("AI_INPUT_WEBP_QUALITY", "88"))

# Face settings
FACE_DET_SIZE = tuple(
    int(x.strip()) for x in os.environ.get("FACE_DET_SIZE", "640,640").split(",")
)
FACE_DET_CONF_THRESHOLD = float(os.environ.get("FACE_DET_CONF_THRESHOLD", "0.45"))
FACE_CLUSTER_QUALITY_THRESHOLD = float(os.environ.get("FACE_CLUSTER_QUALITY_THRESHOLD", "0.60"))
FACE_CLUSTER_MIN_FACE_SIDE = int(os.environ.get("FACE_CLUSTER_MIN_FACE_SIDE", "32"))
PEOPLE_MATCH_EXISTING_SIM_THRESHOLD = float(os.environ.get("PEOPLE_MATCH_EXISTING_SIM_THRESHOLD", "0.58"))
NEW_FACE_CLUSTER_SIM_THRESHOLD = float(os.environ.get("NEW_FACE_CLUSTER_SIM_THRESHOLD", "0.62"))
DUPLICATE_CANDIDATE_SIM_THRESHOLD = float(os.environ.get("DUPLICATE_CANDIDATE_SIM_THRESHOLD", "0.55"))
# ============================================================
# IMAGE-TO-TEXT PROVIDER
# ============================================================

# Gemma-only image-text worker.
# DB/status/function names still say qwen_* in a few places for backward compatibility
# with your existing schema, but runtime inference is intentionally Gemma only.
#
# Why remove the runtime toggle?
# - One codepath is easier to stabilize and benchmark.
# - The previous Qwen/Gemma toggle made batching/dtype handling harder to reason about.
# - Your current blocker is Gemma; this worker should make Gemma fast and reliable first.
IMAGE_TEXT_MODEL_PROVIDER = "gemma"

# Historical Qwen vars kept only so old env/templates do not crash if they still exist.
# They are not used unless you reintroduce Qwen code intentionally later.
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_IMAGE_MAX_SIDE = int(os.environ.get("QWEN_IMAGE_MAX_SIDE", "448"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "320"))
QWEN_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_INFERENCE_BATCH_SIZE", "4"))

# Gemma settings.
# For A5000 24GB, start with GEMMA_INFERENCE_BATCH_SIZE=2 or 4.
# The code can split batches on CUDA OOM, so you can push this upward safely.
GEMMA_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12B-it")
GEMMA_IMAGE_MAX_SIDE = int(os.environ.get("GEMMA_IMAGE_MAX_SIDE", "448"))
GEMMA_MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "320"))
GEMMA_INFERENCE_BATCH_SIZE = int(os.environ.get("GEMMA_INFERENCE_BATCH_SIZE", "4"))
GEMMA_QUANTIZATION = os.environ.get("GEMMA_QUANTIZATION", "4bit").strip().lower()
GEMMA_ENABLE_THINKING = os.environ.get("GEMMA_ENABLE_THINKING", "false").lower() == "true"
GEMMA_ATTN_IMPLEMENTATION = os.environ.get("GEMMA_ATTN_IMPLEMENTATION", "sdpa")

# Keep false for speed. When true, prints tensor dtypes once per batch.
GEMMA_DEBUG_INPUT_DTYPES = os.environ.get("GEMMA_DEBUG_INPUT_DTYPES", "false").lower() == "true"

# If a large Gemma batch OOMs, split it into smaller batches automatically.
GEMMA_SPLIT_BATCH_ON_OOM = os.environ.get("GEMMA_SPLIT_BATCH_ON_OOM", "true").lower() == "true"

# For speed, do not retry a failed batch one image at a time unless explicitly enabled.
# This prevents the repeated failure loop you saw in logs.
IMAGE_TEXT_RETRY_INDIVIDUAL_ON_BATCH_FAILURE = os.environ.get(
    "IMAGE_TEXT_RETRY_INDIVIDUAL_ON_BATCH_FAILURE",
    "false",
).lower() == "true"

if GEMMA_QUANTIZATION not in {"none", "4bit", "8bit"}:
    raise RuntimeError("GEMMA_QUANTIZATION must be one of: none, 4bit, 8bit")

# Text embedding settings
TEXT_EMBED_MODEL_ID = os.environ.get("TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBED_BATCH_SIZE = int(os.environ.get("TEXT_EMBED_BATCH_SIZE", "64"))

# Image embedding / AI culling settings
IMAGE_EMBED_MODEL_ID = os.environ.get("IMAGE_EMBED_MODEL_ID", "openai/clip-vit-base-patch32")
IMAGE_EMBED_BATCH_SIZE = int(os.environ.get("IMAGE_EMBED_BATCH_SIZE", "16"))
CULLING_VERSION = os.environ.get("CULLING_VERSION", "v1")
CLUSTER_VERSION = os.environ.get("CLUSTER_VERSION", "v1")

# Split-worker strict GPU setting. Keep true so Qwen/CLIP/text embedding do not silently run on CPU.
STRICT_QWEN_GPU = os.environ.get("STRICT_QWEN_GPU", "true").lower() == "true"

os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/runpod-volume/huggingface")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/runpod-volume/huggingface/sentence-transformers")
os.environ.setdefault("TORCH_HOME", "/runpod-volume/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/runpod-volume/cache")

Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".nef", ".cr2", ".arw", ".dng", ".tif", ".tiff"
}

UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_.+",
    re.IGNORECASE,
)

s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)

_FACE_APP = None
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_PROCESS_VISION_INFO = None
_GEMMA_MODEL = None
_GEMMA_PROCESSOR = None
_TEXT_EMBED_MODEL = None
_IMAGE_EMBED_MODEL = None
_IMAGE_EMBED_PROCESSOR = None


# DB connection pooling/retry settings.
# This is intentionally tiny because Runpod workers are long-running processes
# and RDS dev instances can have low max_connections.
_DB_POOL = None
_TABLE_COLUMNS_CACHE: Dict[str, set[str]] = {}
_HAS_TABLE_CACHE: Dict[str, bool] = {}

DB_POOL_MIN_CONN = int(os.environ.get("DB_POOL_MIN_CONN", "1"))
DB_POOL_MAX_CONN = int(os.environ.get("DB_POOL_MAX_CONN", "3"))
DB_CONNECT_RETRIES = int(os.environ.get("DB_CONNECT_RETRIES", "8"))
DB_CONNECT_BASE_SLEEP = float(os.environ.get("DB_CONNECT_BASE_SLEEP", "0.75"))
DB_APPLICATION_NAME = os.environ.get("DB_APPLICATION_NAME", "photo_ai_worker")


# ============================================================
# DB HELPERS
# ============================================================

def assert_env_ready() -> None:
    missing = []
    required = {
        "RDS_HOST": RDS_HOST,
        "RDS_DB": RDS_DB,
        "RDS_USER": RDS_USER,
        "RDS_PASSWORD": RDS_PASSWORD,
        "S3_BUCKET": S3_BUCKET,
    }

    for key, value in required.items():
        if not value:
            missing.append(key)

    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


class PooledDbConnection:
    """
    Small wrapper so existing code can keep calling conn.close().
    close() returns the connection to the pool instead of closing the socket.
    """
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
                # Prevent "idle in transaction" when SELECT helpers return to the pool.
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
            _DB_POOL = SimpleConnectionPool(
                minconn=DB_POOL_MIN_CONN,
                maxconn=DB_POOL_MAX_CONN,
                **_connect_kwargs(),
            )
            print(
                f"DB pool initialized: min={DB_POOL_MIN_CONN}, max={DB_POOL_MAX_CONN}, "
                f"application_name={DB_APPLICATION_NAME}",
                flush=True,
            )
            return _DB_POOL
        except psycopg2.OperationalError as e:
            last_err = e
            sleep_for = min(20.0, DB_CONNECT_BASE_SLEEP * (2 ** attempt))
            print(
                f"DB pool init failed attempt={attempt + 1}/{DB_CONNECT_RETRIES}: {repr(e)}; "
                f"sleeping {sleep_for:.2f}s",
                flush=True,
            )
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
            print(
                f"DB connection failed attempt={attempt + 1}/{DB_CONNECT_RETRIES}: {repr(e)}; "
                f"sleeping {sleep_for:.2f}s",
                flush=True,
            )
            time.sleep(sleep_for)

    raise last_err


def db_one(sql: str, params: Tuple[Any, ...] = ()):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        conn.close()


def db_all(sql: str, params: Tuple[Any, ...] = ()):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def table_columns(table_name: str) -> set[str]:
    cached = _TABLE_COLUMNS_CACHE.get(table_name)
    if cached is not None:
        return cached

    rows = db_all("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s;
    """, (table_name,))

    cols = {r["column_name"] for r in rows}
    _TABLE_COLUMNS_CACHE[table_name] = cols
    return cols


def has_table(table_name: str) -> bool:
    cached = _HAS_TABLE_CACHE.get(table_name)
    if cached is not None:
        return cached

    row = db_one("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema='public'
              AND table_name=%s
        ) AS exists;
    """, (table_name,))

    value = bool(row and row["exists"])
    _HAS_TABLE_CACHE[table_name] = value
    return value


def execute_sql(sql: str, params: Tuple[Any, ...] = ()) -> None:
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    finally:
        conn.close()


def execute_sql_best_effort(sql: str, params: Tuple[Any, ...] = ()) -> bool:
    """
    Use this only for error/status updates where failing to write the status should
    not crash the whole worker.
    """
    try:
        execute_sql(sql, params)
        return True
    except Exception as e:
        print("Best-effort SQL failed:", repr(e), flush=True)
        return False


# ============================================================
# LOGGING / STATUS
# ============================================================

def update_job_status(
    job_id: str,
    status: str,
    step: str,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
):
    payload = {
        "job_id": job_id,
        "status": status,
        "step": step,
        "message": message,
        "extra": extra or {},
    }
    print("[JOB_STATUS]", payload, flush=True)


# ============================================================
# VECTOR HELPERS
# ============================================================

def parse_pg_vector(v: Any) -> np.ndarray:
    if v is None:
        raise ValueError("Cannot parse null vector")

    if isinstance(v, np.ndarray):
        arr = v.astype(np.float32)
    elif isinstance(v, list):
        arr = np.array(v, dtype=np.float32)
    else:
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        arr = np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float32)

    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32)


def vector_to_pg(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def mean_normalized(vectors: List[np.ndarray]) -> np.ndarray:
    X = np.stack(vectors).astype(np.float32)
    c = X.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm
    return c.astype(np.float32)


# ============================================================
# S3 HELPERS
# ============================================================

def is_image_key(key: str) -> bool:
    return Path(key).suffix.lower() in IMAGE_EXTS


def is_generated_original_key(key: str) -> bool:
    return bool(UUID_PREFIX_RE.match(Path(key).name))


def list_s3_objects(prefix: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents", []))

    return objects


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def download_file(key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, key, str(local_path))
    return local_path


def upload_file(local_path: Path, key: str, content_type: str) -> None:
    s3.upload_file(
        str(local_path),
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def delete_s3_prefix(prefix: str) -> int:
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue

        objects = [{"Key": o["Key"]} for o in contents]
        s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects})
        deleted += len(objects)

    return deleted


# ============================================================
# IMAGE HELPERS
# ============================================================

def read_image_any(local_path: Path) -> Image.Image:
    ext = local_path.suffix.lower()

    if ext in {".nef", ".cr2", ".arw", ".dng"}:
        try:
            import rawpy
            with rawpy.imread(str(local_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                )
            return Image.fromarray(rgb).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"RAW image read failed for {local_path}. "
                "Install rawpy/imageio or convert RAW before upload."
            ) from e

    img = Image.open(local_path)
    return ImageOps.exif_transpose(img).convert("RGB")


def save_ai_input_webp(source_local: Path, output_local: Path) -> Tuple[int, int]:
    img = read_image_any(source_local)
    img.thumbnail((AI_INPUT_MAX_SIDE, AI_INPUT_MAX_SIDE), Image.Resampling.LANCZOS)
    output_local.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_local, "WEBP", quality=AI_INPUT_WEBP_QUALITY, method=6)
    return img.size


def image_text_max_side() -> int:
    if IMAGE_TEXT_MODEL_PROVIDER == "gemma":
        return GEMMA_IMAGE_MAX_SIDE
    return QWEN_IMAGE_MAX_SIDE


def make_qwen_image(input_path: Path, max_side: Optional[int] = None) -> Path:
    """
    Historical name kept so the rest of the worker does not need a DB/status rename.
    It prepares the smaller annotated image for either Qwen or Gemma.
    """
    max_side = int(max_side or image_text_max_side())

    img = Image.open(input_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = input_path.parent / (f"{IMAGE_TEXT_MODEL_PROVIDER}_" + input_path.stem + ".jpg")
    img.save(out, "JPEG", quality=82)
    return out



# ============================================================
# ALBUM / EVENT
# ============================================================

def restore_album_context(album_slug: str, album_name: Optional[str] = None) -> Dict[str, Any]:
    album = db_one("""
        SELECT *
        FROM albums
        WHERE slug = %s
        LIMIT 1;
    """, (album_slug,))

    if not album:
        cols = table_columns("albums")

        data = {
            "id": str(uuid.uuid4()),
            "name": album_name or album_slug,
            "slug": album_slug,
            "password_required": False,
            "watermark_enabled": False,
            "is_deleted": False,
        }

        valid = {k: v for k, v in data.items() if k in cols}

        keys = list(valid.keys())
        col_sql = ", ".join(keys)
        val_sql = []
        vals = []

        for k in keys:
            if k == "id":
                val_sql.append("%s::uuid")
            else:
                val_sql.append("%s")
            vals.append(valid[k])

        if "created_at" in cols:
            col_sql += ", created_at"
            val_sql.append("now()")

        if "updated_at" in cols:
            col_sql += ", updated_at"
            val_sql.append("now()")

        sql = f"""
            INSERT INTO albums ({col_sql})
            VALUES ({", ".join(val_sql)})
            RETURNING *;
        """

        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(vals))
                    album = cur.fetchone()
        finally:
            conn.close()

    album_ctx = {
        "album_id": str(album["id"]),
        "album_slug": album["slug"],
        "album_name": album.get("name"),
    }

    print("Restored album:", album_ctx, flush=True)
    return album_ctx


def upsert_events(album_ctx: Dict[str, Any], events: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    album_id = album_ctx["album_id"]
    restored_events: List[Dict[str, Any]] = []

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for event in events:
                    name = event["name"]
                    slug = event["slug"]
                    source_prefix = event["source_prefix"]

                    cur.execute("""
                        INSERT INTO album_events (
                            album_id,
                            name,
                            slug,
                            source_prefix,
                            sort_order,
                            is_deleted,
                            created_at,
                            updated_at
                        )
                        VALUES (
                            %s::uuid,
                            %s,
                            %s,
                            %s,
                            COALESCE(
                                (SELECT MAX(sort_order) + 1 FROM album_events WHERE album_id = %s::uuid),
                                1
                            ),
                            false,
                            now(),
                            now()
                        )
                        ON CONFLICT (album_id, slug)
                        DO UPDATE SET
                            name = EXCLUDED.name,
                            source_prefix = EXCLUDED.source_prefix,
                            is_deleted = false,
                            updated_at = now()
                        RETURNING *;
                    """, (
                        album_id,
                        name,
                        slug,
                        source_prefix,
                        album_id,
                    ))

                    row = cur.fetchone()
                    restored_events.append({
                        "event_id": str(row["id"]),
                        "name": row["name"],
                        "slug": row["slug"],
                        "source_prefix": row.get("source_prefix") or source_prefix,
                    })
    finally:
        conn.close()

    print("Restored events:", restored_events, flush=True)
    return restored_events


# ============================================================
# INGEST
# ============================================================

def get_existing_photo_by_source(album_id: str, event_id: str, source_key: str):
    return db_one("""
        SELECT *
        FROM photos
        WHERE album_id = %s::uuid
          AND album_event_id = %s::uuid
          AND COALESCE(is_deleted, false) = false
          AND (
              source_s3_key = %s
              OR original_s3_key = %s
          )
        LIMIT 1;
    """, (album_id, event_id, source_key, source_key))


def create_photo_row(album_ctx: Dict[str, Any], event: Dict[str, Any], source_key: str, size_bytes: int):
    album_id = album_ctx["album_id"]
    album_slug = album_ctx["album_slug"]
    event_id = event["event_id"]
    event_slug = event["slug"]

    photo_uuid = str(uuid.uuid4())
    file_name = Path(source_key).name

    data = {
        "id": photo_uuid,
        "album_id": album_id,
        "album_event_id": event_id,
        "photo_uuid": photo_uuid,
        "file_name": file_name,
        "original_file_name": file_name,
        "source_s3_key": source_key,
        "original_s3_key": source_key,
        "storage_album_slug": album_slug,
        "storage_event_slug": event_slug,
        "file_size_bytes": size_bytes,
        "is_deleted": False,

        "compression_status": "pending",
        "face_index_status": "pending",
        "qwen_status": "pending",
        "search_index_status": "pending",
        "watermark_status": "skipped",

        "ai_input_s3_key": f"albums/{album_slug}/events/{event_slug}/ai-input/{photo_uuid}.webp",
        "annotated_s3_key": f"albums/{album_slug}/events/{event_slug}/annotated/{photo_uuid}.jpg",

        "clean_preview_s3_key": None,
        "watermarked_preview_s3_key": None,
        "thumbnail_s3_key": None,
    }

    cols = table_columns("photos")
    valid = {k: v for k, v in data.items() if k in cols}

    keys = list(valid.keys())
    col_sql = ", ".join(keys)

    values_sql_parts = []
    values = []

    for k in keys:
        if k in {"id", "album_id", "album_event_id", "photo_uuid"}:
            values_sql_parts.append("%s::uuid")
            values.append(valid[k])
        else:
            values_sql_parts.append("%s")
            values.append(valid[k])

    timestamp_sql = ""
    if "created_at" in cols and "created_at" not in valid:
        col_sql += ", created_at"
        timestamp_sql += ", now()"
    if "updated_at" in cols and "updated_at" not in valid:
        col_sql += ", updated_at"
        timestamp_sql += ", now()"

    sql = f"""
    INSERT INTO photos ({col_sql})
    VALUES ({", ".join(values_sql_parts)}{timestamp_sql})
    RETURNING *;
    """

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(values))
                return cur.fetchone()
    finally:
        conn.close()


def scan_and_ingest_originals(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]

    total_scanned = 0
    total_images = 0
    total_usable = 0
    total_generated_skipped = 0
    created_rows = 0
    skipped_existing = 0
    failed = 0
    per_event = []

    for event in events:
        prefix = event["source_prefix"]
        event_id = event["event_id"]

        objects = list_s3_objects(prefix)
        image_objects = [o for o in objects if is_image_key(o["Key"])]

        # Skip UUID-prefixed generated originals only when there are non-generated originals.
        # If a folder only contains UUID-prefixed keys, we treat them as usable because some existing folders are structured that way.
        generated_objects = [o for o in image_objects if is_generated_original_key(o["Key"])]
        normal_objects = [o for o in image_objects if not is_generated_original_key(o["Key"])]

        if normal_objects:
            usable_objects = normal_objects
        else:
            usable_objects = image_objects
            generated_objects = []

        total_scanned += len(objects)
        total_images += len(image_objects)
        total_usable += len(usable_objects)
        total_generated_skipped += len(generated_objects)

        print(
            f"Scanning {event['name']}: raw={len(objects)}, "
            f"images={len(image_objects)}, usable={len(usable_objects)}, "
            f"generated_to_skip={len(generated_objects)}, prefix={prefix}",
            flush=True,
        )

        event_created = 0
        event_existing = 0
        event_failed = 0

        for obj in usable_objects:
            key = obj["Key"]
            size = int(obj.get("Size") or 0)

            try:
                existing = get_existing_photo_by_source(album_id, event_id, key)
                if existing:
                    skipped_existing += 1
                    event_existing += 1
                    continue

                create_photo_row(album_ctx, event, key, size)
                created_rows += 1
                event_created += 1

            except Exception as e:
                failed += 1
                event_failed += 1
                print("INGEST FAILED:", key, repr(e), flush=True)

        per_event.append({
            "event": event["slug"],
            "source_prefix": prefix,
            "objects": len(objects),
            "image_objects": len(image_objects),
            "usable_images": len(usable_objects),
            "generated_skipped": len(generated_objects),
            "created_rows": event_created,
            "skipped_existing": event_existing,
            "failed": event_failed,
        })

    result = {
        "total_scanned": total_scanned,
        "total_image_objects": total_images,
        "total_usable_images": total_usable,
        "total_generated_skipped": total_generated_skipped,
        "created_rows": created_rows,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "per_event": per_event,
    }

    print("Ingest result:", result, flush=True)
    return result


def validate_s3_sources(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    photos = db_all("""
        SELECT
            id,
            album_event_id,
            file_name,
            source_s3_key,
            original_s3_key
        FROM photos
        WHERE album_id = %s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND COALESCE(is_deleted, false) = false
        ORDER BY created_at;
    """, (album_id, event_ids))

    missing = []

    for photo in photos:
        key = photo.get("original_s3_key") or photo.get("source_s3_key")
        if not key:
            missing.append((photo, key, "missing source/original key"))
            continue

        if not s3_key_exists(key):
            missing.append((photo, key, "S3 HeadObject Not Found"))

    if missing:
        missing_ids = [str(p["id"]) for p, _, _ in missing]

        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE photos
                        SET is_deleted = true,
                            deleted_at = now(),
                            compression_status = 'failed',
                            compression_error = 'Source S3 object missing during serverless preflight validation',
                            updated_at = now()
                        WHERE album_id = %s::uuid
                          AND id = ANY(%s::uuid[]);
                    """, (album_id, missing_ids))
        finally:
            conn.close()

    result = {
        "checked": len(photos),
        "missing_sources_soft_deleted": len(missing),
        "missing_samples": [
            {
                "photo_id": str(p["id"]),
                "file_name": p.get("file_name"),
                "key": key,
                "reason": reason,
            }
            for p, key, reason in missing[:25]
        ],
    }

    print("S3 validation result:", result, flush=True)
    return result


# ============================================================
# COMPRESSION / AI INPUT
# ============================================================

def compress_one_photo(row: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    photo_id = str(row["id"])
    try:
        tmpdir = LOCAL_WORK / "compress" / photo_id
        tmpdir.mkdir(parents=True, exist_ok=True)

        source_key = row.get("original_s3_key") or row.get("source_s3_key")
        if not source_key:
            raise RuntimeError("missing original/source S3 key")

        original_local = tmpdir / Path(source_key).name
        ai_local = tmpdir / "ai.webp"

        download_file(source_key, original_local)
        width, height = save_ai_input_webp(original_local, ai_local)

        ai_key = row.get("ai_input_s3_key")
        if not ai_key:
            album_slug = row["storage_album_slug"]
            event_slug = row["storage_event_slug"]
            photo_uuid = row.get("photo_uuid") or photo_id
            ai_key = f"albums/{album_slug}/events/{event_slug}/ai-input/{photo_uuid}.webp"

        upload_file(ai_local, ai_key, "image/webp")

        cols = table_columns("photos")
        set_parts = [
            "compression_status='completed'",
            "compression_error=NULL",
            "ai_input_s3_key=%s",
            "updated_at=now()"
        ]
        values = [ai_key]

        if "width" in cols:
            set_parts.append("width=%s")
            values.append(width)
        if "height" in cols:
            set_parts.append("height=%s")
            values.append(height)

        values.append(photo_id)

        execute_sql(
            f"""
            UPDATE photos
            SET {", ".join(set_parts)}
            WHERE id=%s::uuid;
            """,
            tuple(values),
        )

        return "ok", photo_id, None

    except Exception as e:
        err = repr(e)
        execute_sql("""
            UPDATE photos
            SET compression_status='failed',
                compression_error=%s,
                updated_at=now()
            WHERE id=%s::uuid;
        """, (err, photo_id))
        return "failed", photo_id, err


def compress_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    rows = db_all("""
        SELECT *
        FROM photos
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND COALESCE(is_deleted,false)=false
          AND compression_status IN ('pending','failed')
        ORDER BY created_at;
    """, (album_id, event_ids))

    ok = 0
    failed = 0
    errors = []

    for row in rows:
        status, photo_id, err = compress_one_photo(row)
        if status == "ok":
            ok += 1
        else:
            failed += 1
            errors.append({"photo_id": photo_id, "error": err})

    result = {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25]}
    print("Compression result:", result, flush=True)
    return result


# ============================================================
# INSIGHTFACE
# ============================================================

def preload_cuda_libs_for_onnxruntime():
    """
    Helps ONNXRuntime load CUDA/cuDNN libraries from pip packages and CUDA image
    before it falls back to incompatible system libraries.
    """
    import os
    import site
    import glob
    import ctypes

    candidate_dirs = []

    for base in site.getsitepackages():
        candidate_dirs.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(candidate_dirs + ([existing] if existing else []))

    preload_names = [
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudart.so.12",
        "libcudnn.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libnvrtc.so.12",
    ]

    loaded = []
    for d in candidate_dirs:
        for name in preload_names:
            path = os.path.join(d, name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(path)
                except Exception as e:
                    print(f"Could not preload {path}: {e}", flush=True)

    print("CUDA preload dirs:", candidate_dirs, flush=True)
    print("CUDA preloaded libs:", loaded, flush=True)


def load_face_app():
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    preload_cuda_libs_for_onnxruntime()

    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = ort.get_available_providers()
    print("ONNXRuntime providers:", available, flush=True)

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider missing. Available providers={available}. "
            "Refusing to run InsightFace on CPU because GPU is expected."
        )

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "cudnn_conv_use_max_workspace": "0"
            }
        ),
        "CPUExecutionProvider"
    ]

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

    _FACE_APP = app
    print("InsightFace loaded on GPU", flush=True)
    return _FACE_APP


def face_index_one_photo(row: Dict[str, Any]) -> Tuple[str, str, int, Optional[str]]:
    face_app = load_face_app()
    photo_id = str(row["id"])

    try:
        tmpdir = LOCAL_WORK / "faces" / photo_id
        tmpdir.mkdir(parents=True, exist_ok=True)

        ai_key = row.get("ai_input_s3_key")
        if not ai_key:
            raise RuntimeError("missing ai_input_s3_key")

        local_img = tmpdir / "ai.webp"
        download_file(ai_key, local_img)

        img = cv2.imread(str(local_img))
        if img is None:
            raise RuntimeError("cv2 could not read ai input")

        faces = face_app.get(img)
        insert_rows = []

        h, w = img.shape[:2]

        for face in faces:
            det = float(face.det_score)
            if det < FACE_DET_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            side = max(0, min(x2 - x1, y2 - y1))
            q = det * side
            emb = vector_to_pg(face.embedding)

            seed = bool(
                det >= FACE_CLUSTER_QUALITY_THRESHOLD
                and side >= FACE_CLUSTER_MIN_FACE_SIDE
            )

            insert_rows.append((
                str(row["album_id"]),
                str(row["album_event_id"]),
                photo_id,
                x1, y1, x2, y2,
                emb,
                det,
                q,
                side,
                seed,
                seed,
            ))

        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM faces WHERE photo_id=%s::uuid", (photo_id,))

                    if insert_rows:
                        execute_values(cur, """
                            INSERT INTO faces(
                                album_id,
                                album_event_id,
                                photo_id,
                                bbox_x1,
                                bbox_y1,
                                bbox_x2,
                                bbox_y2,
                                embedding,
                                detection_confidence,
                                face_quality_score,
                                face_side,
                                is_cluster_seed,
                                is_cover_candidate
                            )
                            VALUES %s
                        """, insert_rows, template="(%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s)")

                    cur.execute("""
                        UPDATE photos
                        SET face_index_status='completed',
                            face_index_error=NULL,
                            updated_at=now()
                        WHERE id=%s::uuid;
                    """, (photo_id,))
        finally:
            conn.close()

        return "ok", photo_id, len(insert_rows), None

    except Exception as e:
        err = repr(e)
        execute_sql("""
            UPDATE photos
            SET face_index_status='failed',
                face_index_error=%s,
                updated_at=now()
            WHERE id=%s::uuid;
        """, (err, photo_id))
        return "failed", photo_id, 0, err


def face_index_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    rows = db_all("""
        SELECT *
        FROM photos
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND compression_status='completed'
          AND face_index_status IN ('pending','failed')
          AND COALESCE(is_deleted,false)=false
        ORDER BY created_at;
    """, (album_id, event_ids))

    ok = 0
    failed = 0
    faces_count = 0
    errors = []

    for row in rows:
        status, photo_id, face_count, err = face_index_one_photo(row)
        if status == "ok":
            ok += 1
            faces_count += face_count
        else:
            failed += 1
            errors.append({"photo_id": photo_id, "error": err})

    result = {
        "rows": len(rows),
        "ok": ok,
        "failed": failed,
        "faces": faces_count,
        "errors": errors[:25],
    }
    print("Face index result:", result, flush=True)
    return result


# ============================================================
# SAFE PEOPLE RECONCILIATION
# ============================================================

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def safe_add_new_people_without_touching_existing_names(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]

    face_rows = db_all("""
        SELECT
            f.id,
            f.album_id,
            f.album_event_id,
            f.photo_id,
            f.embedding,
            f.face_quality_score,
            f.detection_confidence
        FROM faces f
        WHERE f.album_id = %s::uuid
          AND f.person_id IS NULL
        ORDER BY f.face_quality_score DESC NULLS LAST, f.created_at;
    """, (album_id,))

    if not face_rows:
        rebuild_photo_people_base_safe(album_ctx)
        return {
            "unlabeled_faces": 0,
            "assigned_to_existing_people": 0,
            "new_people_created": 0,
            "message": "No unlabeled faces found. Existing people untouched.",
        }

    existing_people = db_all("""
        SELECT
            id,
            person_number,
            display_name,
            default_name,
            centroid_embedding
        FROM people
        WHERE album_id = %s::uuid
          AND COALESCE(is_hidden, false) = false
          AND centroid_embedding IS NOT NULL
        ORDER BY person_number;
    """, (album_id,))

    face_vecs = [parse_pg_vector(r["embedding"]) for r in face_rows]

    assigned_to_existing = []
    remaining_indices = list(range(len(face_rows)))

    if existing_people:
        people_vecs = [parse_pg_vector(p["centroid_embedding"]) for p in existing_people]
        P = np.stack(people_vecs).astype(np.float32)
        F = np.stack(face_vecs).astype(np.float32)

        sims = F @ P.T
        best_people_idx = sims.argmax(axis=1)
        best_scores = sims.max(axis=1)

        still_remaining = []

        for i, score in enumerate(best_scores):
            if float(score) >= PEOPLE_MATCH_EXISTING_SIM_THRESHOLD:
                person = existing_people[int(best_people_idx[i])]
                assigned_to_existing.append((str(person["id"]), str(face_rows[i]["id"])))
            else:
                still_remaining.append(i)

        remaining_indices = still_remaining

    new_clusters: Dict[int, List[int]] = {}

    if remaining_indices:
        if len(remaining_indices) == 1:
            new_clusters[0] = remaining_indices
        else:
            X = np.stack([face_vecs[i] for i in remaining_indices]).astype(np.float32)
            sims = X @ X.T
            dsu = DSU(len(remaining_indices))

            for i in range(len(remaining_indices)):
                for j in range(i + 1, len(remaining_indices)):
                    if float(sims[i, j]) >= NEW_FACE_CLUSTER_SIM_THRESHOLD:
                        dsu.union(i, j)

            temp: Dict[int, List[int]] = {}
            for local_i, original_i in enumerate(remaining_indices):
                root = dsu.find(local_i)
                temp.setdefault(root, []).append(original_i)

            new_clusters = {idx: vals for idx, vals in enumerate(temp.values())}

    conn = get_conn()
    new_people_created = 0

    try:
        with conn:
            with conn.cursor() as cur:
                if assigned_to_existing:
                    execute_values(
                        cur,
                        """
                        UPDATE faces AS f
                        SET person_id = v.person_id::uuid
                        FROM (VALUES %s) AS v(person_id, face_id)
                        WHERE f.id = v.face_id::uuid;
                        """,
                        assigned_to_existing,
                        template="(%s, %s)"
                    )

                cur.execute("""
                    SELECT COALESCE(MAX(person_number), 0) AS max_num
                    FROM people
                    WHERE album_id = %s::uuid;
                """, (album_id,))
                next_num = int(cur.fetchone()["max_num"] or 0) + 1

                for _, face_indices in new_clusters.items():
                    group_faces = [face_rows[i] for i in face_indices]
                    group_vecs = [face_vecs[i] for i in face_indices]

                    centroid = mean_normalized(group_vecs)
                    best_face = sorted(
                        group_faces,
                        key=lambda r: float(r.get("face_quality_score") or 0),
                        reverse=True,
                    )[0]

                    default_name = f"Person {next_num}"

                    cur.execute("""
                        INSERT INTO people(
                            album_id,
                            person_number,
                            default_name,
                            display_name,
                            cover_photo_id,
                            centroid_embedding,
                            face_count,
                            photo_count,
                            occurrence_count,
                            created_at,
                            updated_at
                        )
                        VALUES (
                            %s::uuid,
                            %s,
                            %s,
                            %s,
                            %s::uuid,
                            %s::vector,
                            %s,
                            %s,
                            %s,
                            now(),
                            now()
                        )
                        RETURNING id;
                    """, (
                        album_id,
                        next_num,
                        default_name,
                        default_name,
                        str(best_face["photo_id"]),
                        vector_to_pg(centroid),
                        len(group_faces),
                        len(set(str(f["photo_id"]) for f in group_faces)),
                        len(set(str(f["photo_id"]) for f in group_faces)),
                    ))

                    person_id = str(cur.fetchone()["id"])

                    face_update_rows = [(person_id, str(f["id"])) for f in group_faces]

                    execute_values(
                        cur,
                        """
                        UPDATE faces AS f
                        SET person_id = v.person_id::uuid
                        FROM (VALUES %s) AS v(person_id, face_id)
                        WHERE f.id = v.face_id::uuid;
                        """,
                        face_update_rows,
                        template="(%s, %s)"
                    )

                    next_num += 1
                    new_people_created += 1

                cur.execute("""
                    WITH stats AS (
                        SELECT
                            person_id,
                            COUNT(*) AS face_count,
                            COUNT(DISTINCT photo_id) AS photo_count
                        FROM faces
                        WHERE album_id = %s::uuid
                          AND person_id IS NOT NULL
                        GROUP BY person_id
                    )
                    UPDATE people p
                    SET
                        face_count = stats.face_count,
                        photo_count = stats.photo_count,
                        occurrence_count = stats.photo_count,
                        updated_at = now()
                    FROM stats
                    WHERE p.id = stats.person_id
                      AND p.album_id = %s::uuid;
                """, (album_id, album_id))

    finally:
        conn.close()

    rebuild_photo_people_base_safe(album_ctx)
    crop_and_upload_missing_person_covers(album_ctx)
    build_duplicate_candidates(album_ctx)

    return {
        "unlabeled_faces": len(face_rows),
        "assigned_to_existing_people": len(assigned_to_existing),
        "new_people_created": new_people_created,
        "existing_people_untouched": True,
        "names_preserved": True,
    }


def rebuild_photo_people_base_safe(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM photo_people
                    WHERE album_id = %s::uuid;

                    INSERT INTO photo_people(
                        album_id,
                        album_event_id,
                        photo_id,
                        person_id,
                        person_label,
                        face_ids,
                        co_person_ids,
                        search_text,
                        confidence,
                        created_at,
                        updated_at
                    )
                    WITH base AS (
                        SELECT
                            f.album_id,
                            f.album_event_id,
                            f.photo_id,
                            f.person_id,
                            ARRAY_AGG(f.id) AS face_ids,
                            AVG(f.detection_confidence) AS conf
                        FROM faces f
                        WHERE f.album_id = %s::uuid
                          AND f.person_id IS NOT NULL
                        GROUP BY f.album_id, f.album_event_id, f.photo_id, f.person_id
                    ),
                    enriched AS (
                        SELECT
                            b.*,
                            ARRAY_REMOVE(ARRAY_AGG(o.person_id), b.person_id) AS co_person_ids
                        FROM base b
                        LEFT JOIN base o ON o.photo_id = b.photo_id
                        GROUP BY
                            b.album_id,
                            b.album_event_id,
                            b.photo_id,
                            b.person_id,
                            b.face_ids,
                            b.conf
                    )
                    SELECT
                        e.album_id,
                        e.album_event_id,
                        e.photo_id,
                        e.person_id,
                        COALESCE(NULLIF(pe.display_name, ''), pe.default_name),
                        e.face_ids,
                        e.co_person_ids,
                        COALESCE(NULLIF(pe.display_name, ''), pe.default_name)
                            || ' appears in this photo with '
                            || COALESCE(array_length(e.co_person_ids, 1), 0)::text
                            || ' other people.',
                        e.conf,
                        now(),
                        now()
                    FROM enriched e
                    JOIN people pe ON pe.id = e.person_id;
                """, (album_id, album_id))
    finally:
        conn.close()

    row = db_one("""
        SELECT COUNT(*) AS photo_people_rows
        FROM photo_people
        WHERE album_id = %s::uuid;
    """, (album_id,))

    return dict(row or {})


def crop_and_upload_missing_person_covers(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    album_slug = album_ctx["album_slug"]

    people = db_all("""
        SELECT pe.id AS person_id,
               pe.person_number,
               f.id AS face_id,
               f.photo_id,
               f.bbox_x1,
               f.bbox_y1,
               f.bbox_x2,
               f.bbox_y2,
               p.ai_input_s3_key
        FROM people pe
        JOIN LATERAL (
            SELECT *
            FROM faces f
            WHERE f.person_id = pe.id
            ORDER BY f.face_quality_score DESC NULLS LAST
            LIMIT 1
        ) f ON true
        JOIN photos p ON p.id = f.photo_id
        WHERE pe.album_id = %s::uuid
          AND pe.cover_face_s3_key IS NULL;
    """, (album_id,))

    ok = 0
    failed = 0
    errors = []

    for r in people:
        pid = str(r["person_id"])
        try:
            tmpdir = LOCAL_WORK / "covers" / pid
            tmpdir.mkdir(parents=True, exist_ok=True)

            img_local = tmpdir / "ai.webp"
            download_file(r["ai_input_s3_key"], img_local)

            img = Image.open(img_local).convert("RGB")

            pad = 30
            x1 = max(0, int(r["bbox_x1"]) - pad)
            y1 = max(0, int(r["bbox_y1"]) - pad)
            x2 = min(img.width, int(r["bbox_x2"]) + pad)
            y2 = min(img.height, int(r["bbox_y2"]) + pad)

            crop = img.crop((x1, y1, x2, y2))
            crop = ImageOps.fit(crop, (256, 256), method=Image.Resampling.LANCZOS)

            out = tmpdir / "cover.jpg"
            crop.save(out, "JPEG", quality=90)

            key = f"albums/{album_slug}/faces/{pid}/cover.jpg"
            upload_file(out, key, "image/jpeg")

            execute_sql("""
                UPDATE people
                SET cover_face_s3_key=%s,
                    updated_at=now()
                WHERE id=%s::uuid;
            """, (key, pid))

            ok += 1

        except Exception as e:
            failed += 1
            errors.append({"person_id": pid, "error": repr(e)})

    result = {"missing_covers": len(people), "ok": ok, "failed": failed, "errors": errors[:25]}
    print("Cover crop result:", result, flush=True)
    return result


def ensure_person_merge_candidates_schema() -> None:
    if not has_table("person_merge_candidates"):
        return

    execute_sql("""
        ALTER TABLE person_merge_candidates
        ADD COLUMN IF NOT EXISTS reason TEXT;

        ALTER TABLE person_merge_candidates
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT now();

        CREATE UNIQUE INDEX IF NOT EXISTS uniq_person_merge_candidates_album_pair
        ON person_merge_candidates(album_id, person_a_id, person_b_id);
    """)


def build_duplicate_candidates(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not has_table("person_merge_candidates"):
        return {"skipped": True, "reason": "person_merge_candidates table does not exist"}

    ensure_person_merge_candidates_schema()

    album_id = album_ctx["album_id"]

    people = db_all("""
        SELECT id, person_number, centroid_embedding
        FROM people
        WHERE album_id=%s::uuid
          AND COALESCE(is_hidden,false)=false
          AND centroid_embedding IS NOT NULL;
    """, (album_id,))

    if len(people) < 2:
        return {"candidates": 0}

    X = np.stack([parse_pg_vector(p["centroid_embedding"]) for p in people])
    sims = X @ X.T
    candidates = []

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            sim = float(sims[i, j])
            if sim < DUPLICATE_CANDIDATE_SIM_THRESHOLD:
                continue

            a = str(people[i]["id"])
            b = str(people[j]["id"])

            shared_row = db_one("""
                SELECT COUNT(DISTINCT pp1.photo_id) AS c
                FROM photo_people pp1
                JOIN photo_people pp2 ON pp2.photo_id = pp1.photo_id
                WHERE pp1.person_id=%s::uuid
                  AND pp2.person_id=%s::uuid;
            """, (a, b))

            shared = int(shared_row["c"] or 0)
            reason = "possible duplicate; same-photo conflict" if shared > 0 else "possible duplicate"

            candidates.append((album_id, a, b, sim, shared, "pending", reason))

    if candidates:
        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        INSERT INTO person_merge_candidates(
                            album_id,
                            person_a_id,
                            person_b_id,
                            similarity_score,
                            shared_photo_count,
                            status,
                            reason
                        )
                        VALUES %s
                        ON CONFLICT(album_id, person_a_id, person_b_id)
                        DO UPDATE SET
                            similarity_score=EXCLUDED.similarity_score,
                            shared_photo_count=EXCLUDED.shared_photo_count,
                            status=EXCLUDED.status,
                            reason=EXCLUDED.reason,
                            updated_at=now();
                    """, candidates, template="(%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s)")
        finally:
            conn.close()

    return {"candidates": len(candidates)}



def assert_torch_gpu_ready(component: str):
    import torch

    print(f"{component} torch:", torch.__version__, flush=True)
    print(f"{component} torch cuda:", torch.version.cuda, flush=True)
    print(f"{component} cuda available:", torch.cuda.is_available(), flush=True)
    print(f"{component} device count:", torch.cuda.device_count(), flush=True)

    if torch.cuda.is_available():
        print(f"{component} gpu name:", torch.cuda.get_device_name(0), flush=True)
        return torch.device("cuda")

    if STRICT_QWEN_GPU:
        raise RuntimeError(f"STRICT_QWEN_GPU=true but CUDA is not available for {component}")

    return torch.device("cpu")


# ============================================================
# QWEN
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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    assert_torch_gpu_ready("Qwen")
    print(f"Loading Qwen model: {QWEN_MODEL_ID}", flush=True)

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if STRICT_QWEN_GPU:
        devices = {str(p.device) for p in model.parameters()}
        print("Qwen parameter devices:", sorted(list(devices))[:10], flush=True)
        if not any(d.startswith("cuda") for d in devices):
            raise RuntimeError(f"Qwen model did not load on CUDA. Devices={sorted(list(devices))[:10]}")

    _QWEN_MODEL = model
    _QWEN_PROCESSOR = processor
    _PROCESS_VISION_INFO = process_vision_info

    print("Qwen loaded", flush=True)
    return _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO



def load_gemma():
    global _GEMMA_MODEL, _GEMMA_PROCESSOR

    if _GEMMA_MODEL is not None and _GEMMA_PROCESSOR is not None:
        return _GEMMA_MODEL, _GEMMA_PROCESSOR

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        # Gemma 4 12B model card currently shows this class.
        from transformers import AutoModelForMultimodalLM as GemmaModelClass
    except Exception:
        # Fallback for other recent Transformers builds / smaller variants.
        from transformers import AutoModelForImageTextToText as GemmaModelClass

    assert_torch_gpu_ready("Gemma")
    print(
        f"Loading Gemma model: {GEMMA_MODEL_ID}, "
        f"quantization={GEMMA_QUANTIZATION}, "
        f"attn={GEMMA_ATTN_IMPLEMENTATION}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        GEMMA_MODEL_ID,
        trust_remote_code=True,
    )

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if GEMMA_ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = GEMMA_ATTN_IMPLEMENTATION

    if GEMMA_QUANTIZATION == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif GEMMA_QUANTIZATION == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        # Transformers 5 uses dtype; older builds may still expect torch_dtype.
        model_kwargs["dtype"] = "auto"

    try:
        model = GemmaModelClass.from_pretrained(
            GEMMA_MODEL_ID,
            **model_kwargs,
        )
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = GemmaModelClass.from_pretrained(
            GEMMA_MODEL_ID,
            **model_kwargs,
        )

    model.eval()

    if STRICT_QWEN_GPU:
        devices = {str(p.device) for p in model.parameters()}
        print("Gemma parameter devices:", sorted(list(devices))[:10], flush=True)
        if not any(d.startswith("cuda") for d in devices):
            raise RuntimeError(f"Gemma model did not load on CUDA. Devices={sorted(list(devices))[:10]}")

    _GEMMA_MODEL = model
    _GEMMA_PROCESSOR = processor

    print("Gemma loaded", flush=True)
    return _GEMMA_MODEL, _GEMMA_PROCESSOR


def image_text_inference_batch_size() -> int:
    # Gemma-only runtime. qwen_* DB/status names remain for compatibility.
    return max(1, GEMMA_INFERENCE_BATCH_SIZE)


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

    normalized_people = {}

    for label, value in people_obj.items():
        label = str(label).strip()
        if not label.startswith("Person "):
            continue

        if not isinstance(value, dict):
            value = {"visible_keywords": str(value or "")}

        jc = value.get("jewelry_count") or {}
        if not isinstance(jc, dict):
            jc = {}

        jewelry_count = {
            "bangles": safe_int(jc.get("bangles")),
            "necklace": safe_int(jc.get("necklace")),
            "earrings": safe_int(jc.get("earrings")),
            "rings": safe_int(jc.get("rings")),
            "head_jewelry": safe_int(jc.get("head_jewelry")),
            "other": safe_int(jc.get("other")),
        }

        normalized_people[label] = {
            "visible_keywords": str(value.get("visible_keywords") or "")[:500],
            "jewelry_count": jewelry_count,
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

    merged_search_text = " | ".join([
        caption,
        scene,
        search_text,
        quality_keywords,
    ]).strip()

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


def patch_annotated_keys(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    execute_sql("""
        UPDATE photos
        SET photo_uuid = COALESCE(photo_uuid, gen_random_uuid())
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[]);

        UPDATE photos p
        SET storage_album_slug = COALESCE(p.storage_album_slug, a.slug),
            storage_event_slug = COALESCE(p.storage_event_slug, e.slug),
            updated_at = now()
        FROM albums a
        JOIN album_events e ON e.album_id = a.id
        WHERE p.album_id = a.id
          AND p.album_event_id = e.id
          AND p.album_id = %s::uuid
          AND p.album_event_id = ANY(%s::uuid[]);

        UPDATE photos
        SET annotated_s3_key = 'albums/' || storage_album_slug || '/events/' || storage_event_slug || '/annotated/' || photo_uuid::text || '.jpg',
            updated_at = now()
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND COALESCE(is_deleted,false)=false
          AND (annotated_s3_key IS NULL OR annotated_s3_key='');
    """, (album_id, event_ids, album_id, event_ids, album_id, event_ids))


def mark_no_labeled_face_photos_skipped(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    execute_sql("""
        UPDATE photos p
        SET qwen_status='skipped_no_labeled_faces',
            qwen_error='No labeled faces for person-labeled Qwen metadata',
            updated_at=now()
        WHERE p.album_id=%s::uuid
          AND p.album_event_id = ANY(%s::uuid[])
          AND p.face_index_status='completed'
          AND COALESCE(p.is_deleted,false)=false
          AND p.qwen_status IN ('pending','failed')
          AND NOT EXISTS (
              SELECT 1
              FROM faces f
              WHERE f.photo_id = p.id
                AND f.person_id IS NOT NULL
          );
    """, (album_id, event_ids))


def reset_retryable_qwen_failures(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    execute_sql("""
        UPDATE photos
        SET qwen_status='pending',
            qwen_error=NULL,
            updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND qwen_status='failed'
          AND (
              qwen_error LIKE 'JSONDecodeError%%'
              OR qwen_error LIKE 'ParamValidationError%%'
              OR qwen_error LIKE 'ValueError%%'
              OR qwen_error LIKE 'RuntimeError%%'
          );
    """, (album_id, event_ids))


def maybe_reset_existing_qwen(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    if not RESET_EXISTING_QWEN:
        return

    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    execute_sql("""
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
          AND album_event_id = ANY(%s::uuid[])
          AND COALESCE(is_deleted,false)=false;

        UPDATE photo_people
        SET qwen_description=NULL,
            qwen_json=NULL,
            search_embedding=NULL,
            updated_at=now()
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[]);
    """, (album_id, event_ids, album_id, event_ids))


def annotate_photo(photo: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]]]:
    faces = db_all("""
        SELECT f.*, pe.person_number
        FROM faces f
        JOIN people pe ON pe.id=f.person_id
        WHERE f.photo_id=%s::uuid
          AND f.person_id IS NOT NULL
        ORDER BY pe.person_number;
    """, (str(photo["id"]),))

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

    return make_qwen_image(ann), faces


def fetch_qwen_faces_by_photo(photo_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not photo_ids:
        return {}

    rows = db_all("""
        SELECT f.*, pe.person_number
        FROM faces f
        JOIN people pe ON pe.id=f.person_id
        WHERE f.photo_id = ANY(%s::uuid[])
          AND f.person_id IS NOT NULL
        ORDER BY f.photo_id, pe.person_number;
    """, (photo_ids,))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["photo_id"]), []).append(row)

    return grouped


def fetch_event_map(event_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not event_ids:
        return {}

    rows = db_all("""
        SELECT id, name, slug
        FROM album_events
        WHERE id = ANY(%s::uuid[]);
    """, (event_ids,))

    return {
        str(r["id"]): {"name": r["name"], "slug": r["slug"]}
        for r in rows
    }


def annotate_photo_with_faces(photo: Dict[str, Any], faces: List[Dict[str, Any]]) -> Tuple[Path, List[Dict[str, Any]]]:
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

    return make_qwen_image(ann), faces


def _qwen_describe_batch_impl(image_paths: List[Path]) -> List[Dict[str, Any]]:
    import torch

    model, processor, process_vision_info = load_qwen()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": qwen_prompt()},
            ],
        }
        for path in image_paths
    ]

    texts = [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

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

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], gen)
    ]

    outs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs, gen, trimmed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [normalize_qwen_data(extract_json(o)) for o in outs]




def _gemma_response_to_text(processor: Any, response: Any) -> str:
    try:
        parsed = processor.parse_response(response) if hasattr(processor, "parse_response") else response
    except Exception:
        parsed = response

    if isinstance(parsed, str):
        return parsed

    if isinstance(parsed, dict):
        for key in ("content", "response", "text", "answer"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(parsed, ensure_ascii=False)

    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        if parts:
            return "\n".join(parts)

    return str(parsed or "")


def _model_device_and_dtype(model: Any) -> Tuple[Any, Any]:
    import torch

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        # Gemma on CUDA is fastest/stablest with bf16 activations when supported.
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    return device, model_dtype


def _move_inputs_to_model(inputs: Any, model: Any) -> Dict[str, Any]:
    """
    Move a Gemma processor BatchFeature/dict to the model device without blindly
    casting the whole object.

    This fixes:
      RuntimeError: "LayerNormKernelImpl" not implemented for 'Byte'

    Cause:
      processor.apply_chat_template can return image/pixel tensors as uint8/Byte.
      Calling inputs.to(device) keeps those tensors as Byte, and Gemma eventually
      sends them through LayerNorm, which requires floating point.

    Rules:
      - token ids / positions / masks stay integer or bool
      - pixel/image/video tensors become model dtype, usually bf16/fp16/fp32
      - existing floating tensors become model dtype
    """
    import torch

    device, model_dtype = _model_device_and_dtype(model)

    integer_or_bool_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "cache_position",
        "image_token_mask",
        "cross_attention_mask",
        "pixel_attention_mask",
    }

    # Grid/index tensors must remain integer even though their names mention image/video.
    integer_name_parts = ("mask", "grid", "ids", "position", "cache", "index")
    float_name_parts = ("pixel", "image", "video", "vision")

    def move_one(key: str, value: Any) -> Any:
        if not torch.is_tensor(value):
            return value.to(device) if hasattr(value, "to") else value

        key_l = key.lower()

        if key in integer_or_bool_keys or any(part in key_l for part in integer_name_parts):
            return value.to(device=device)

        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=model_dtype)

        # Important: image-like uint8/byte tensors must become floating point.
        if any(part in key_l for part in float_name_parts):
            return value.to(device=device, dtype=model_dtype)

        # Unknown integer tensor: keep as integer on device.
        return value.to(device=device)

    if isinstance(inputs, dict):
        moved = {k: move_one(k, v) for k, v in inputs.items()}
    elif hasattr(inputs, "items"):
        moved = {k: move_one(k, v) for k, v in inputs.items()}
    else:
        # Last-resort fallback. Avoid dtype cast because this can corrupt token IDs.
        moved = inputs.to(device)

    if GEMMA_DEBUG_INPUT_DTYPES and isinstance(moved, dict):
        print(
            "Gemma input tensor dtypes: " + json.dumps(
                {
                    k: {
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "device": str(v.device),
                    }
                    for k, v in moved.items()
                    if torch.is_tensor(v)
                },
                default=str,
            ),
            flush=True,
        )

    return moved


def _gemma_messages_for_image(image_path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                # Gemma HF examples use url for image paths. Local paths work here.
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": qwen_prompt()},
            ],
        }
    ]


def _gemma_apply_chat_template(processor: Any, conversations: List[List[Dict[str, Any]]]) -> Any:
    template_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "padding": True,
    }

    try:
        return processor.apply_chat_template(
            conversations,
            enable_thinking=GEMMA_ENABLE_THINKING,
            **template_kwargs,
        )
    except TypeError:
        # Some Transformers versions do not accept enable_thinking.
        return processor.apply_chat_template(
            conversations,
            **template_kwargs,
        )


def _is_cuda_oom(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return (
        "cuda out of memory" in text
        or "outofmemoryerror" in text
        or "cublas_status_alloc_failed" in text
        or "cuda error: out of memory" in text
    )


def _gemma_describe_batch_once(image_paths: List[Path]) -> List[Dict[str, Any]]:
    import torch

    model, processor = load_gemma()

    conversations = [_gemma_messages_for_image(path) for path in image_paths]
    inputs = _gemma_apply_chat_template(processor, conversations)
    inputs = _move_inputs_to_model(inputs, model)

    # With padding=True, generate returns the padded prompt plus generated tokens.
    # Trim by the padded input width, not per-row attention length.
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEMMA_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = [out[input_len:] for out in outputs]
    texts = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for text in texts:
        parsed_text = _gemma_response_to_text(processor, text)
        results.append(normalize_qwen_data(extract_json(parsed_text)))

    del inputs, outputs, generated
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _gemma_describe_batch_impl(image_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    True Gemma batch inference.

    Uses the full batch to maximize GPU utilization. If the batch is too large
    for the GPU and GEMMA_SPLIT_BATCH_ON_OOM=true, it recursively splits the
    batch so the job keeps progressing instead of failing the entire run.
    """
    if not image_paths:
        return []

    started = time.time()
    print(
        f"Gemma batch start: images={len(image_paths)}, "
        f"max_new_tokens={GEMMA_MAX_NEW_TOKENS}, "
        f"image_max_side={GEMMA_IMAGE_MAX_SIDE}",
        flush=True,
    )

    try:
        result = _gemma_describe_batch_once(image_paths)
        print(
            f"Gemma batch done: images={len(image_paths)}, seconds={round(time.time() - started, 2)}",
            flush=True,
        )
        return result
    except RuntimeError as e:
        if GEMMA_SPLIT_BATCH_ON_OOM and _is_cuda_oom(e) and len(image_paths) > 1:
            import torch

            print(
                f"Gemma batch OOM; splitting batch: images={len(image_paths)}, error={repr(e)}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mid = len(image_paths) // 2
            return _gemma_describe_batch_impl(image_paths[:mid]) + _gemma_describe_batch_impl(image_paths[mid:])

        raise


def qwen_describe_batch(image_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Backward-compatible function name.

    Runtime is Gemma-only now. DB/status fields still say qwen_* so the rest
    of your existing schema and UI do not need to change.
    """
    print(
        f"Image-to-text provider=gemma, images={len(image_paths)}, batch_size={GEMMA_INFERENCE_BATCH_SIZE}",
        flush=True,
    )
    return _gemma_describe_batch_impl(image_paths)

def get_label_to_id(album_ctx: Dict[str, Any]) -> Dict[str, str]:
    people = db_all("""
        SELECT id, person_number
        FROM people
        WHERE album_id=%s::uuid;
    """, (album_ctx["album_id"],))

    return {
        f"Person {p['person_number']}": str(p["id"])
        for p in people
    }


def save_qwen_photo(
    photo: Dict[str, Any],
    data: Dict[str, Any],
    label_to_id: Dict[str, str],
    event_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    photo_data = data.get("photo", {})
    people_map = data.get("people_map", {})

    event_row = None
    if event_by_id:
        event_row = event_by_id.get(str(photo["album_event_id"]))

    if event_row is None:
        event_row = db_one("""
            SELECT name, slug
            FROM album_events
            WHERE id=%s::uuid
            LIMIT 1;
        """, (str(photo["album_event_id"]),))

    if event_row:
        data["event"] = {
            "name": event_row["name"],
            "slug": event_row["slug"],
        }

    conn = get_conn()

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE photos
                    SET caption=%s,
                        ai_description=%s,
                        search_text=%s,
                        qwen_json=%s,
                        qwen_status='completed',
                        qwen_error=NULL,
                        annotated_s3_key=%s,
                        updated_at=now()
                    WHERE id=%s::uuid;
                """, (
                    photo_data.get("caption"),
                    photo_data.get("detailed_description"),
                    photo_data.get("search_text"),
                    Json(data),
                    photo["annotated_s3_key"],
                    str(photo["id"]),
                ))

                for label, person_data in people_map.items():
                    pid = label_to_id.get(label)
                    if not pid:
                        continue

                    visible_keywords = person_data.get("visible_keywords", "")
                    jewelry_keywords = person_data.get("jewelry_keywords", "")
                    jewelry_count = person_data.get("jewelry_count", {})
                    gaze = person_data.get("camera_gaze", "uncertain")
                    quality_score = person_data.get("photo_quality_score", 0)

                    person_search_text = (
                        f"{label}; {visible_keywords}; "
                        f"jewelry={jewelry_keywords}; "
                        f"jewelry_count={jewelry_count}; "
                        f"camera_gaze={gaze}; "
                        f"photo_quality_score={quality_score}/10"
                    )

                    person_json = {
                        "person_label": label,
                        "description": visible_keywords,
                        "search_text": person_search_text,
                        "jewelry_count": jewelry_count,
                        "jewelry_keywords": jewelry_keywords,
                        "camera_gaze": gaze,
                        "photo_quality_score": quality_score,
                        "confidence": 0.75,
                    }

                    cur.execute("""
                        UPDATE photo_people
                        SET qwen_description=%s,
                            qwen_json=%s,
                            search_text=%s,
                            confidence=%s,
                            updated_at=now()
                        WHERE photo_id=%s::uuid
                          AND person_id=%s::uuid;
                    """, (
                        visible_keywords,
                        Json(person_json),
                        person_search_text,
                        0.75,
                        str(photo["id"]),
                        pid,
                    ))
    finally:
        conn.close()


def mark_qwen_failed(photo: Dict[str, Any], err: Exception) -> None:
    execute_sql_best_effort("""
        UPDATE photos
        SET qwen_status='failed',
            qwen_error=%s,
            updated_at=now()
        WHERE id=%s::uuid;
    """, (repr(err), str(photo["id"])))


def chunks(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_qwen_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    patch_annotated_keys(album_ctx, events)
    maybe_reset_existing_qwen(album_ctx, events)
    mark_no_labeled_face_photos_skipped(album_ctx, events)
    reset_retryable_qwen_failures(album_ctx, events)

    label_to_id = get_label_to_id(album_ctx)

    rows = db_all("""
        SELECT *
        FROM photos
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND face_index_status='completed'
          AND qwen_status IN ('pending','failed')
          AND COALESCE(is_deleted,false)=false
          AND EXISTS (
              SELECT 1
              FROM faces f
              WHERE f.photo_id=photos.id
                AND f.person_id IS NOT NULL
          )
        ORDER BY created_at;
    """, (album_id, event_ids))

    # Reduce DB churn during Qwen: fetch all face labels and event data once.
    photo_ids = [str(r["id"]) for r in rows]
    faces_by_photo_id = fetch_qwen_faces_by_photo(photo_ids)
    event_by_id = fetch_event_map(list({str(r["album_event_id"]) for r in rows}))

    ok = 0
    fail = 0
    errors = []

    for batch in chunks(rows, image_text_inference_batch_size()):
        prepared = []

        for photo in batch:
            try:
                faces = faces_by_photo_id.get(str(photo["id"]), [])
                qwen_img, _faces = annotate_photo_with_faces(photo, faces)
                prepared.append((photo, qwen_img))
            except Exception as e:
                fail += 1
                mark_qwen_failed(photo, e)
                errors.append({"file_name": photo.get("file_name"), "error": repr(e)})

        if not prepared:
            continue

        try:
            datas = qwen_describe_batch([x[1] for x in prepared])

            for (photo, _), data in zip(prepared, datas):
                try:
                    save_qwen_photo(photo, data, label_to_id, event_by_id=event_by_id)
                    ok += 1
                except Exception as e:
                    fail += 1
                    mark_qwen_failed(photo, e)
                    errors.append({"file_name": photo.get("file_name"), "error": repr(e)})

        except Exception as batch_err:
            print("GEMMA BATCH FAILED:", repr(batch_err), flush=True)

            if IMAGE_TEXT_RETRY_INDIVIDUAL_ON_BATCH_FAILURE and len(prepared) > 1:
                print("Retrying Gemma batch individually because IMAGE_TEXT_RETRY_INDIVIDUAL_ON_BATCH_FAILURE=true", flush=True)
                for photo, qwen_img in prepared:
                    try:
                        data = qwen_describe_batch([qwen_img])[0]
                        save_qwen_photo(photo, data, label_to_id, event_by_id=event_by_id)
                        ok += 1
                    except Exception as e:
                        fail += 1
                        mark_qwen_failed(photo, e)
                        errors.append({"file_name": photo.get("file_name"), "error": repr(e)})
            else:
                # Avoid the repeated failure loop from the old code. Mark this batch failed once.
                for photo, _qwen_img in prepared:
                    fail += 1
                    mark_qwen_failed(photo, batch_err)
                    errors.append({"file_name": photo.get("file_name"), "error": repr(batch_err)})

    result = {"rows": len(rows), "ok": ok, "failed": fail, "errors": errors[:25]}
    print("Qwen result:", result, flush=True)
    return result


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def load_text_embed_model():
    global _TEXT_EMBED_MODEL
    if _TEXT_EMBED_MODEL is not None:
        return _TEXT_EMBED_MODEL

    from sentence_transformers import SentenceTransformer
    device = assert_torch_gpu_ready("Text embeddings")
    _TEXT_EMBED_MODEL = SentenceTransformer(TEXT_EMBED_MODEL_ID, device=str(device))
    print(f"Text embedding model loaded: {TEXT_EMBED_MODEL_ID} on {device}", flush=True)
    return _TEXT_EMBED_MODEL


def run_text_embeddings_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    model = load_text_embed_model()

    photo_rows = db_all("""
        SELECT id, search_text
        FROM photos
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND COALESCE(is_deleted,false)=false
          AND search_text IS NOT NULL
          AND search_text <> ''
          AND (
              search_embedding IS NULL
              OR search_index_status IN ('pending','failed')
          )
        ORDER BY created_at;
    """, (album_id, event_ids))

    photo_ok = 0
    photo_failed = 0

    for batch in chunks(photo_rows, TEXT_EMBED_BATCH_SIZE):
        texts = [r["search_text"] for r in batch]
        try:
            embs = model.encode(texts, normalize_embeddings=True)
            rows = [(vector_to_pg(embs[i]), str(batch[i]["id"])) for i in range(len(batch))]

            conn = get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        execute_values(cur, """
                            UPDATE photos AS p
                            SET search_embedding = v.embedding::vector,
                                search_index_status='completed',
                                search_index_error=NULL,
                                updated_at=now()
                            FROM (VALUES %s) AS v(embedding, id)
                            WHERE p.id = v.id::uuid;
                        """, rows, template="(%s,%s)")
            finally:
                conn.close()

            photo_ok += len(batch)

        except Exception as e:
            photo_failed += len(batch)
            ids = [str(r["id"]) for r in batch]
            execute_sql("""
                UPDATE photos
                SET search_index_status='failed',
                    search_index_error=%s,
                    updated_at=now()
                WHERE id = ANY(%s::uuid[]);
            """, (repr(e), ids))

    pp_rows = db_all("""
        SELECT id, search_text
        FROM photo_people
        WHERE album_id=%s::uuid
          AND album_event_id = ANY(%s::uuid[])
          AND search_text IS NOT NULL
          AND search_text <> ''
          AND search_embedding IS NULL
        ORDER BY created_at;
    """, (album_id, event_ids))

    pp_ok = 0
    pp_failed = 0

    for batch in chunks(pp_rows, TEXT_EMBED_BATCH_SIZE):
        texts = [r["search_text"] for r in batch]
        try:
            embs = model.encode(texts, normalize_embeddings=True)
            rows = [(vector_to_pg(embs[i]), str(batch[i]["id"])) for i in range(len(batch))]

            conn = get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        execute_values(cur, """
                            UPDATE photo_people AS pp
                            SET search_embedding = v.embedding::vector,
                                updated_at=now()
                            FROM (VALUES %s) AS v(embedding, id)
                            WHERE pp.id = v.id::uuid;
                        """, rows, template="(%s,%s)")
            finally:
                conn.close()

            pp_ok += len(batch)

        except Exception:
            pp_failed += len(batch)

    result = {
        "photos_embedded": photo_ok,
        "photos_failed": photo_failed,
        "photo_people_embedded": pp_ok,
        "photo_people_failed": pp_failed,
    }
    print("Embedding result:", result, flush=True)
    return result


# ============================================================
# CLEANUP
# ============================================================

def cleanup_temp_s3(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    deleted = []

    for event in events:
        album_slug = album_ctx["album_slug"]
        event_slug = event["slug"]

        if DELETE_TEMP_AI_INPUT:
            prefix = f"albums/{album_slug}/events/{event_slug}/ai-input/"
            count = delete_s3_prefix(prefix)
            deleted.append({"prefix": prefix, "deleted": count})

        if DELETE_TEMP_ANNOTATED:
            prefix = f"albums/{album_slug}/events/{event_slug}/annotated/"
            count = delete_s3_prefix(prefix)
            deleted.append({"prefix": prefix, "deleted": count})

    return {"deleted": deleted}


# ============================================================
# FINAL VERIFY
# ============================================================

def final_verify(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    album_id = album_ctx["album_id"]
    event_ids = [e["event_id"] for e in events]

    event_rows = db_all("""
        SELECT
            e.name,
            e.slug,
            COUNT(p.id) AS photos,
            COUNT(*) FILTER (WHERE p.compression_status = 'completed') AS compressed,
            COUNT(*) FILTER (WHERE p.compression_status = 'pending') AS compression_pending,
            COUNT(*) FILTER (WHERE p.compression_status = 'failed') AS compression_failed,
            COUNT(*) FILTER (WHERE p.face_index_status = 'completed') AS face_indexed,
            COUNT(*) FILTER (WHERE p.face_index_status = 'pending') AS face_pending,
            COUNT(*) FILTER (WHERE p.face_index_status = 'failed') AS face_failed,
            COUNT(*) FILTER (WHERE p.qwen_status = 'completed') AS qwen_completed,
            COUNT(*) FILTER (WHERE p.qwen_status = 'pending') AS qwen_pending,
            COUNT(*) FILTER (WHERE p.qwen_status = 'failed') AS qwen_failed,
            COUNT(*) FILTER (WHERE p.qwen_status = 'skipped_no_labeled_faces') AS qwen_skipped,
            COUNT(*) FILTER (WHERE p.search_index_status = 'completed') AS search_embedded
        FROM album_events e
        LEFT JOIN photos p
          ON p.album_event_id = e.id
         AND COALESCE(p.is_deleted, false) = false
        WHERE e.album_id = %s::uuid
          AND e.id = ANY(%s::uuid[])
        GROUP BY e.name, e.slug
        ORDER BY e.slug;
    """, (album_id, event_ids))

    face_rows = db_all("""
        SELECT
            e.name,
            e.slug,
            COUNT(f.id) AS total_faces,
            COUNT(f.id) FILTER (WHERE f.person_id IS NOT NULL) AS labeled_faces,
            COUNT(f.id) FILTER (WHERE f.person_id IS NULL) AS unlabeled_faces
        FROM album_events e
        LEFT JOIN faces f ON f.album_event_id = e.id
        WHERE e.album_id = %s::uuid
          AND e.id = ANY(%s::uuid[])
        GROUP BY e.name, e.slug
        ORDER BY e.slug;
    """, (album_id, event_ids))

    people_summary = db_one("""
        SELECT
            COUNT(*) AS people_count,
            COUNT(*) FILTER (WHERE cover_face_s3_key IS NULL) AS missing_covers
        FROM people
        WHERE album_id = %s::uuid
          AND COALESCE(is_hidden, false) = false;
    """, (album_id,))

    named_people = db_one("""
        SELECT
            COUNT(*) AS named_people
        FROM people
        WHERE album_id = %s::uuid
          AND COALESCE(is_hidden, false) = false
          AND display_name IS NOT NULL
          AND display_name NOT LIKE 'Person %%';
    """, (album_id,))

    result = {
        "events": [dict(r) for r in event_rows],
        "faces": [dict(r) for r in face_rows],
        "people": dict(people_summary) if people_summary else {},
        "named_people": dict(named_people) if named_people else {},
        "safety": {
            "destructive_people_rebuild_used": False,
            "existing_people_deleted": False,
            "names_preserved": True,
            "auto_merge_duplicates": False,
            "duplicate_candidates_only": True,
        }
    }

    print("Final verify:", result, flush=True)
    return result



# ============================================================
# AI CULLING / BEST PHOTO SELECTION
# ============================================================

def restore_album_only(album_slug: str) -> Dict[str, Any]:
    row = db_one("""
        SELECT id, slug, name, album_type, culling_enabled, culling_config
        FROM albums
        WHERE slug=%s
          AND COALESCE(is_deleted,false)=false
        LIMIT 1;
    """, (album_slug,))
    if not row:
        raise ValueError(f"Album not found: {album_slug}")
    return {
        "album_id": str(row["id"]),
        "album_slug": row["slug"],
        "album_name": row.get("name"),
        "album_type": row.get("album_type") or "general",
        "culling_enabled": bool(row.get("culling_enabled")),
        "culling_config": row.get("culling_config") or {},
    }


def resolve_events_for_album(album_ctx: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    album_id = album_ctx["album_id"]
    event_slugs = payload.get("event_slugs") or payload.get("events_slugs")
    event_slug = payload.get("event_slug")
    if event_slug and not event_slugs:
        event_slugs = [event_slug]

    if payload.get("events"):
        # Existing full pipeline sends source_prefix in events. For culling modes we only need DB event rows.
        slugs = [e.get("slug") for e in payload.get("events", []) if e.get("slug")]
        if slugs and not event_slugs:
            event_slugs = slugs

    if event_slugs:
        rows = db_all("""
            SELECT id, name, slug, source_prefix
            FROM album_events
            WHERE album_id=%s::uuid
              AND slug = ANY(%s)
              AND COALESCE(is_deleted,false)=false
            ORDER BY sort_order, name;
        """, (album_id, event_slugs))
    else:
        rows = db_all("""
            SELECT id, name, slug, source_prefix
            FROM album_events
            WHERE album_id=%s::uuid
              AND COALESCE(is_deleted,false)=false
            ORDER BY sort_order, name;
        """, (album_id,))

    return [
        {
            "event_id": str(r["id"]),
            "name": r["name"],
            "slug": r["slug"],
            "source_prefix": r.get("source_prefix"),
        }
        for r in rows
    ]


def get_qwen_quality(qwen_json: Any) -> Dict[str, Any]:
    if not isinstance(qwen_json, dict):
        return {}
    if "quality" in qwen_json and isinstance(qwen_json["quality"], dict):
        return qwen_json["quality"]
    raw = qwen_json.get("raw") if isinstance(qwen_json.get("raw"), dict) else qwen_json
    return {
        "background_quality": raw.get("background_quality"),
        "frame_clarity": raw.get("frame_clarity"),
        "album_worthy_score": raw.get("album_worthy_score"),
        "camera_gaze_overall": (raw.get("camera_gaze") or {}).get("overall") if isinstance(raw.get("camera_gaze"), dict) else None,
        "album_worthy_reason": raw.get("album_worthy_reason"),
    }


def score01_from_10(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v) / 10.0))
    except Exception:
        return default


def compute_cv_quality(local_path: Path) -> Dict[str, float]:
    img = cv2.imread(str(local_path))
    if img is None:
        raise RuntimeError(f"cv2 could not read image: {local_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = max(0.0, min(1.0, lap_var / 900.0))

    mean = float(gray.mean())
    # Best exposure is around middle gray. This penalizes very dark/bright frames.
    exposure = 1.0 - min(abs(mean - 127.0) / 127.0, 1.0)

    std = float(gray.std())
    contrast = max(0.0, min(1.0, std / 80.0))

    # Cheap noise estimate: residual after small blur. Lower residual is better.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = float(np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32))))
    noise_score = 1.0 - max(0.0, min(1.0, residual / 18.0))

    return {
        "sharpness_score": round(sharpness, 4),
        "exposure_score": round(exposure, 4),
        "contrast_score": round(contrast, 4),
        "noise_score": round(noise_score, 4),
        "motion_blur_score": round(sharpness, 4),
    }


def download_best_available_photo(photo: Dict[str, Any], purpose: str) -> Path:
    photo_id = str(photo["id"])
    key = photo.get("ai_input_s3_key") or photo.get("clean_preview_s3_key") or photo.get("original_s3_key") or photo.get("source_s3_key")
    if not key:
        raise RuntimeError("Photo has no usable S3 key")
    tmpdir = LOCAL_WORK / purpose / photo_id
    tmpdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(key).suffix or ".jpg"
    local = tmpdir / f"input{suffix}"
    download_file(key, local)
    return local


def build_similarity_caption(photo: Dict[str, Any], qwen_json: Any) -> Tuple[str, str]:
    q = qwen_json if isinstance(qwen_json, dict) else {}
    photo_obj = q.get("photo") if isinstance(q.get("photo"), dict) else {}
    quality = get_qwen_quality(q)
    caption = (
        photo.get("caption")
        or photo_obj.get("caption")
        or photo.get("ai_description")
        or photo_obj.get("detailed_description")
        or photo.get("search_text")
        or photo.get("file_name")
        or "photo"
    )
    caption = str(caption)[:700]
    gaze = str(quality.get("camera_gaze_overall") or "uncertain")
    pose_key = "general"
    text = caption.lower()
    if "sitting" in text or "seated" in text:
        pose_key = "seated"
    elif "standing" in text:
        pose_key = "standing"
    elif "close" in text or "portrait" in text:
        pose_key = "portrait"
    elif "group" in text:
        pose_key = "group"
    return f"{caption}; gaze={gaze}; pose={pose_key}", pose_key


def upsert_photo_culling_score(photo: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    photo_id = str(photo["id"])
    album_type = payload.get("album_type") or photo.get("album_type") or "general"
    persona_key = payload.get("persona_key") or album_type or "general"
    scoring_version = payload.get("scoring_version") or CULLING_VERSION

    try:
        local = download_best_available_photo(photo, "culling-score")
        cvq = compute_cv_quality(local)

        q = photo.get("qwen_json") if isinstance(photo.get("qwen_json"), dict) else {}
        quality = get_qwen_quality(q)

        face_row = db_one("""
            SELECT
                COUNT(*) AS face_count,
                AVG(face_quality_score) AS avg_face_quality,
                MAX(face_quality_score) AS max_face_quality,
                AVG(detection_confidence) AS avg_detection_confidence
            FROM faces
            WHERE photo_id=%s::uuid;
        """, (photo_id,)) or {}

        face_count = int(face_row.get("face_count") or 0)
        raw_face_quality = float(face_row.get("max_face_quality") or 0.0)
        # Your current face_quality_score is det_score * face_side, so normalize loosely.
        face_quality = max(0.0, min(1.0, raw_face_quality / 500.0))

        qwen_aesthetic = score01_from_10(quality.get("album_worthy_score"), 0.0)
        composition = score01_from_10(quality.get("background_quality"), 0.5)
        frame_clarity = score01_from_10(quality.get("frame_clarity"), cvq["sharpness_score"])

        gaze = str(quality.get("camera_gaze_overall") or "uncertain").lower()
        gaze_score = 1.0 if gaze == "all" else 0.65 if gaze == "some" else 0.35 if gaze == "none" else 0.5
        eyes_open_score = gaze_score
        smile_score = 0.5
        subject_center_score = 0.6 if face_count else 0.4
        occlusion_penalty = 0.0
        stranger_penalty = 0.0
        defect_penalty = 0.0
        duplicate_penalty = 0.0

        # Persona can be tuned later. For now, wedding/saree reward album-worthy and face/gaze more.
        if persona_key in {"wedding", "saree", "single_portrait", "baby_photoshoot"}:
            persona_score = (qwen_aesthetic * 0.45) + (face_quality * 0.25) + (gaze_score * 0.20) + (composition * 0.10)
        else:
            persona_score = (qwen_aesthetic * 0.35) + (frame_clarity * 0.25) + (composition * 0.20) + (face_quality * 0.20)

        final_score = (
            cvq["sharpness_score"] * 0.18
            + cvq["exposure_score"] * 0.10
            + cvq["contrast_score"] * 0.07
            + cvq["noise_score"] * 0.05
            + face_quality * 0.16
            + eyes_open_score * 0.10
            + gaze_score * 0.08
            + composition * 0.06
            + qwen_aesthetic * 0.12
            + persona_score * 0.08
            - occlusion_penalty
            - stranger_penalty
            - defect_penalty
            - duplicate_penalty
        )
        final_score = max(0.0, min(1.0, final_score))

        similarity_caption, pose_key = build_similarity_caption(photo, q)
        reasons = []
        if cvq["sharpness_score"] > 0.75:
            reasons.append("sharp image")
        if gaze_score >= 0.9:
            reasons.append("subjects looking at camera")
        if qwen_aesthetic >= 0.8:
            reasons.append("high album-worthy AI score")
        if composition >= 0.75:
            reasons.append("good background/composition")
        if not reasons:
            reasons.append("balanced technical and AI quality score")

        quality_json = {
            "cv": cvq,
            "qwen_quality": quality,
            "face_count": face_count,
            "raw_face_quality": raw_face_quality,
            "gaze": gaze,
        }

        execute_sql("""
            INSERT INTO photo_culling_scores(
                photo_id, album_id, album_event_id, album_type, persona_key,
                sharpness_score, exposure_score, contrast_score, noise_score, motion_blur_score,
                composition_score, face_quality_score, eyes_open_score, gaze_score, smile_score,
                subject_center_score, occlusion_penalty, stranger_penalty,
                qwen_aesthetic_score, persona_score, defect_penalty, duplicate_penalty,
                final_score, similarity_caption, pose_key, quality_json, reasons,
                score_status, score_error, scoring_version, created_at, updated_at
            )
            VALUES(
                %s::uuid,%s::uuid,%s::uuid,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                'completed',NULL,%s,now(),now()
            )
            ON CONFLICT(photo_id, scoring_version, persona_key)
            DO UPDATE SET
                album_type=EXCLUDED.album_type,
                sharpness_score=EXCLUDED.sharpness_score,
                exposure_score=EXCLUDED.exposure_score,
                contrast_score=EXCLUDED.contrast_score,
                noise_score=EXCLUDED.noise_score,
                motion_blur_score=EXCLUDED.motion_blur_score,
                composition_score=EXCLUDED.composition_score,
                face_quality_score=EXCLUDED.face_quality_score,
                eyes_open_score=EXCLUDED.eyes_open_score,
                gaze_score=EXCLUDED.gaze_score,
                smile_score=EXCLUDED.smile_score,
                subject_center_score=EXCLUDED.subject_center_score,
                occlusion_penalty=EXCLUDED.occlusion_penalty,
                stranger_penalty=EXCLUDED.stranger_penalty,
                qwen_aesthetic_score=EXCLUDED.qwen_aesthetic_score,
                persona_score=EXCLUDED.persona_score,
                defect_penalty=EXCLUDED.defect_penalty,
                duplicate_penalty=EXCLUDED.duplicate_penalty,
                final_score=EXCLUDED.final_score,
                similarity_caption=EXCLUDED.similarity_caption,
                pose_key=EXCLUDED.pose_key,
                quality_json=EXCLUDED.quality_json,
                reasons=EXCLUDED.reasons,
                score_status='completed',
                score_error=NULL,
                updated_at=now();
        """, (
            photo_id, str(photo["album_id"]), str(photo["album_event_id"]), album_type, persona_key,
            cvq["sharpness_score"], cvq["exposure_score"], cvq["contrast_score"], cvq["noise_score"], cvq["motion_blur_score"],
            composition, face_quality, eyes_open_score, gaze_score, smile_score,
            subject_center_score, occlusion_penalty, stranger_penalty,
            qwen_aesthetic, persona_score, defect_penalty, duplicate_penalty,
            final_score, similarity_caption, pose_key, Json(quality_json), Json(reasons),
            scoring_version,
        ))
        return True, None
    except Exception as e:
        err = repr(e)
        execute_sql_best_effort("""
            INSERT INTO photo_culling_scores(
                photo_id, album_id, album_event_id, album_type, persona_key,
                final_score, score_status, score_error, scoring_version, created_at, updated_at
            )
            VALUES(%s::uuid,%s::uuid,%s::uuid,%s,%s,0,'failed',%s,%s,now(),now())
            ON CONFLICT(photo_id, scoring_version, persona_key)
            DO UPDATE SET score_status='failed', score_error=EXCLUDED.score_error, updated_at=now();
        """, (photo_id, str(photo["album_id"]), str(photo["album_event_id"]), album_type, persona_key, err, scoring_version))
        return False, err


def score_photos(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_ctx = restore_album_only(payload["album_slug"])
    events = resolve_events_for_album(album_ctx, payload)
    event_ids = [e["event_id"] for e in events]
    if not event_ids:
        return {"rows": 0, "ok": 0, "failed": 0, "errors": [], "message": "No events found"}

    persona_key = payload.get("persona_key") or payload.get("album_type") or album_ctx.get("album_type") or "general"
    scoring_version = payload.get("scoring_version") or CULLING_VERSION
    limit = int(payload.get("limit") or payload.get("batch", {}).get("limit") or 100)
    only_unscored = bool(payload.get("only_unscored", payload.get("batch", {}).get("only_unscored", True)))

    where_extra = ""
    params: List[Any] = [album_ctx["album_id"], event_ids]
    if only_unscored:
        where_extra = """
          AND NOT EXISTS (
              SELECT 1 FROM photo_culling_scores s
              WHERE s.photo_id=p.id
                AND s.persona_key=%s
                AND s.scoring_version=%s
                AND s.score_status='completed'
          )
        """
        params.extend([persona_key, scoring_version])
    params.append(limit)

    rows = db_all(f"""
        SELECT p.*, a.album_type
        FROM photos p
        JOIN albums a ON a.id=p.album_id
        WHERE p.album_id=%s::uuid
          AND p.album_event_id = ANY(%s::uuid[])
          AND COALESCE(p.is_deleted,false)=false
          AND p.compression_status='completed'
          {where_extra}
        ORDER BY p.created_at
        LIMIT %s;
    """, tuple(params))

    ok = 0
    failed = 0
    errors = []
    for photo in rows:
        success, err = upsert_photo_culling_score(photo, {**payload, "persona_key": persona_key, "scoring_version": scoring_version})
        if success:
            ok += 1
        else:
            failed += 1
            errors.append({"photo_id": str(photo["id"]), "file_name": photo.get("file_name"), "error": err})

    return {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25], "persona_key": persona_key, "scoring_version": scoring_version}


def load_image_embed_model():
    global _IMAGE_EMBED_MODEL, _IMAGE_EMBED_PROCESSOR
    if _IMAGE_EMBED_MODEL is not None and _IMAGE_EMBED_PROCESSOR is not None:
        return _IMAGE_EMBED_MODEL, _IMAGE_EMBED_PROCESSOR

    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading image embedding model: {IMAGE_EMBED_MODEL_ID}", flush=True)
    processor = CLIPProcessor.from_pretrained(IMAGE_EMBED_MODEL_ID)
    model = CLIPModel.from_pretrained(IMAGE_EMBED_MODEL_ID)
    device = assert_torch_gpu_ready("Image embeddings")
    model.to(device)
    model.eval()
    _IMAGE_EMBED_MODEL = model
    _IMAGE_EMBED_PROCESSOR = processor
    print(f"Image embedding model loaded on {device}", flush=True)
    return model, processor


def image_embed_photos(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    album_ctx = restore_album_only(payload["album_slug"])
    events = resolve_events_for_album(album_ctx, payload)
    event_ids = [e["event_id"] for e in events]
    if not event_ids:
        return {"rows": 0, "ok": 0, "failed": 0, "errors": [], "message": "No events found"}

    model_id = payload.get("embedding_model") or IMAGE_EMBED_MODEL_ID
    limit = int(payload.get("limit") or payload.get("batch", {}).get("limit") or 100)
    only_missing = bool(payload.get("only_missing", payload.get("batch", {}).get("only_missing", True)))

    where_extra = ""
    params: List[Any] = [album_ctx["album_id"], event_ids]
    if only_missing:
        where_extra = """
          AND NOT EXISTS (
              SELECT 1 FROM photo_image_embeddings e
              WHERE e.photo_id=p.id
                AND e.embedding_model=%s
                AND e.embedding IS NOT NULL
          )
        """
        params.append(model_id)
    params.append(limit)

    rows = db_all(f"""
        SELECT p.*
        FROM photos p
        WHERE p.album_id=%s::uuid
          AND p.album_event_id = ANY(%s::uuid[])
          AND COALESCE(p.is_deleted,false)=false
          AND p.compression_status='completed'
          {where_extra}
        ORDER BY p.created_at
        LIMIT %s;
    """, tuple(params))

    model, processor = load_image_embed_model()
    device = next(model.parameters()).device
    ok = 0
    failed = 0
    errors = []

    for batch in chunks(rows, IMAGE_EMBED_BATCH_SIZE):
        prepared = []
        images = []
        for photo in batch:
            try:
                local = download_best_available_photo(photo, "image-embed")
                img = read_image_any(local)
                prepared.append(photo)
                images.append(img)
            except Exception as e:
                failed += 1
                errors.append({"photo_id": str(photo["id"]), "file_name": photo.get("file_name"), "error": repr(e)})

        if not images:
            continue
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.detach().cpu().numpy().astype(np.float32)

            upsert_rows = []
            for i, photo in enumerate(prepared):
                upsert_rows.append((
                    str(photo["id"]), str(photo["album_id"]), str(photo["album_event_id"]),
                    model_id, vector_to_pg(arr[i]), "completed", None,
                ))

            conn = get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        execute_values(cur, """
                            INSERT INTO photo_image_embeddings(
                                photo_id, album_id, album_event_id, embedding_model,
                                embedding, embedding_status, embedding_error, created_at, updated_at
                            ) VALUES %s
                            ON CONFLICT(photo_id, embedding_model)
                            DO UPDATE SET
                                embedding=EXCLUDED.embedding,
                                embedding_status='completed',
                                embedding_error=NULL,
                                updated_at=now();
                        """, upsert_rows, template="(%s::uuid,%s::uuid,%s::uuid,%s,%s::vector,%s,%s,now(),now())")
            finally:
                conn.close()
            ok += len(prepared)
            del inputs, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            failed += len(prepared)
            errors.append({"batch_error": repr(e)})

    return {"rows": len(rows), "ok": ok, "failed": failed, "errors": errors[:25], "embedding_model": model_id}


def fetch_people_sets(photo_ids: List[str]) -> Dict[str, set[str]]:
    if not photo_ids:
        return {}
    rows = db_all("""
        SELECT photo_id, person_id
        FROM photo_people
        WHERE photo_id = ANY(%s::uuid[])
          AND person_id IS NOT NULL;
    """, (photo_ids,))
    out: Dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(str(r["photo_id"]), set()).add(str(r["person_id"]))
    return out


def people_overlap(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def cluster_photos(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_ctx = restore_album_only(payload["album_slug"])
    events = resolve_events_for_album(album_ctx, payload)
    event_ids = [e["event_id"] for e in events]
    if not event_ids:
        return {"clusters": 0, "items": 0, "message": "No events found"}

    persona_key = payload.get("persona_key") or payload.get("album_type") or album_ctx.get("album_type") or "general"
    scoring_version = payload.get("scoring_version") or CULLING_VERSION
    cluster_version = payload.get("cluster_version") or CLUSTER_VERSION
    embedding_model = payload.get("embedding_model") or IMAGE_EMBED_MODEL_ID
    opts = payload.get("options") or {}
    image_threshold = float(opts.get("image_similarity_threshold", 0.90))
    text_threshold = float(opts.get("text_similarity_threshold", 0.84))
    final_threshold = float(opts.get("final_similarity_threshold", 0.88))

    rows = db_all("""
        SELECT
            p.id AS photo_id,
            p.album_id,
            p.album_event_id,
            p.file_name,
            p.search_embedding,
            p.created_at,
            s.final_score,
            s.similarity_caption,
            s.pose_key,
            ie.embedding AS image_embedding
        FROM photos p
        JOIN photo_culling_scores s ON s.photo_id=p.id
           AND s.persona_key=%s
           AND s.scoring_version=%s
           AND s.score_status='completed'
        LEFT JOIN photo_image_embeddings ie ON ie.photo_id=p.id
           AND ie.embedding_model=%s
           AND ie.embedding IS NOT NULL
        WHERE p.album_id=%s::uuid
          AND p.album_event_id = ANY(%s::uuid[])
          AND COALESCE(p.is_deleted,false)=false
        ORDER BY p.album_event_id, p.created_at;
    """, (persona_key, scoring_version, embedding_model, album_ctx["album_id"], event_ids))

    if not rows:
        return {"clusters": 0, "items": 0, "message": "No scored photos found"}

    photo_ids = [str(r["photo_id"]) for r in rows]
    people_by_photo = fetch_people_sets(photo_ids)
    n = len(rows)
    dsu = DSU(n)

    image_vecs: List[Optional[np.ndarray]] = []
    text_vecs: List[Optional[np.ndarray]] = []
    for r in rows:
        image_vecs.append(parse_pg_vector(r["image_embedding"]) if r.get("image_embedding") is not None else None)
        text_vecs.append(parse_pg_vector(r["search_embedding"]) if r.get("search_embedding") is not None else None)

    # O(n^2) is fine for current album sizes. For 50k+ photos, move this to FAISS nearest neighbors.
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid grouping across events unless explicitly allowed.
            if str(rows[i]["album_event_id"]) != str(rows[j]["album_event_id"]) and not opts.get("allow_cross_event_clusters", False):
                continue

            img_sim = float(np.dot(image_vecs[i], image_vecs[j])) if image_vecs[i] is not None and image_vecs[j] is not None else 0.0
            txt_sim = float(np.dot(text_vecs[i], text_vecs[j])) if text_vecs[i] is not None and text_vecs[j] is not None else 0.0
            ppl_sim = people_overlap(people_by_photo.get(str(rows[i]["photo_id"]), set()), people_by_photo.get(str(rows[j]["photo_id"]), set()))
            pose_sim = 1.0 if (rows[i].get("pose_key") or "") == (rows[j].get("pose_key") or "") and rows[i].get("pose_key") else 0.0

            final_sim = (img_sim * 0.55) + (txt_sim * 0.25) + (ppl_sim * 0.15) + (pose_sim * 0.05)
            if img_sim >= image_threshold or (txt_sim >= text_threshold and final_sim >= final_threshold) or final_sim >= final_threshold:
                dsu.union(i, j)

    grouped: Dict[int, List[int]] = {}
    for i in range(n):
        grouped.setdefault(dsu.find(i), []).append(i)

    # Replace previous clusters for this scope/version/persona.
    execute_sql("""
        DELETE FROM photo_similarity_clusters
        WHERE album_id=%s::uuid
          AND cluster_version=%s
          AND persona_key=%s
          AND album_event_id = ANY(%s::uuid[]);
    """, (album_ctx["album_id"], cluster_version, persona_key, event_ids))

    cluster_count = 0
    item_count = 0

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for cluster_num, indices in enumerate(grouped.values(), start=1):
                    cluster_rows = [rows[i] for i in indices]
                    cluster_rows_sorted = sorted(cluster_rows, key=lambda r: float(r.get("final_score") or 0), reverse=True)
                    winner = cluster_rows_sorted[0]
                    rep = cluster_rows_sorted[0]
                    event_id = str(rep["album_event_id"])
                    cluster_key = f"{album_ctx['album_slug']}:{event_id}:{cluster_version}:{cluster_num}"
                    cluster_caption = rep.get("similarity_caption") or rep.get("file_name") or "similar photos"
                    pose_key = rep.get("pose_key") or "general"

                    cur.execute("""
                        INSERT INTO photo_similarity_clusters(
                            album_id, album_event_id, cluster_key, cluster_type, persona_key,
                            representative_photo_id, winner_photo_id, cluster_size,
                            avg_similarity, max_similarity, cluster_caption, pose_key,
                            cluster_status, cluster_version, created_at, updated_at
                        ) VALUES(
                            %s::uuid,%s::uuid,%s,'hybrid_similarity',%s,
                            %s::uuid,%s::uuid,%s,
                            NULL,NULL,%s,%s,
                            'active',%s,now(),now()
                        )
                        RETURNING id;
                    """, (
                        album_ctx["album_id"], event_id, cluster_key, persona_key,
                        str(rep["photo_id"]), str(winner["photo_id"]), len(cluster_rows),
                        cluster_caption, pose_key, cluster_version,
                    ))
                    cluster_id = str(cur.fetchone()["id"])

                    item_rows = []
                    for rank, r in enumerate(cluster_rows_sorted, start=1):
                        item_rows.append((
                            cluster_id, str(r["photo_id"]), str(r["album_id"]), str(r["album_event_id"]),
                            None, None, None, None, None,
                            rank, str(r["photo_id"]) == str(winner["photo_id"]),
                        ))
                    execute_values(cur, """
                        INSERT INTO photo_similarity_cluster_items(
                            cluster_id, photo_id, album_id, album_event_id,
                            image_similarity_score, text_similarity_score, people_overlap_score,
                            timestamp_similarity_score, final_similarity_score,
                            rank_in_cluster, is_cluster_winner, created_at
                        ) VALUES %s;
                    """, item_rows, template="(%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,now())")

                    cluster_count += 1
                    item_count += len(item_rows)
    finally:
        conn.close()

    return {"clusters": cluster_count, "items": item_count, "photos": n, "persona_key": persona_key, "cluster_version": cluster_version}


def select_best_photos(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_ctx = restore_album_only(payload["album_slug"])
    events = resolve_events_for_album(album_ctx, payload)
    event_ids = [e["event_id"] for e in events]
    if not event_ids:
        return {"selected_count": 0, "message": "No events found"}

    limit = int(payload.get("limit") or payload.get("requested_count") or 100)
    album_type = payload.get("album_type") or album_ctx.get("album_type") or "general"
    persona_key = payload.get("persona_key") or album_type or "general"
    scoring_version = payload.get("scoring_version") or CULLING_VERSION
    cluster_version = payload.get("cluster_version") or CLUSTER_VERSION
    opts = payload.get("options") or {}
    max_per_cluster = int(opts.get("max_per_cluster", 1))
    collection_name = payload.get("collection_name") or f"Best {limit} Photos"

    # Prefer cluster winners. If clusters do not exist, fall back to top scores.
    candidate_rows = db_all("""
        WITH clustered AS (
            SELECT
                sci.photo_id,
                sci.cluster_id,
                sci.album_event_id,
                s.final_score,
                s.reasons,
                sci.rank_in_cluster,
                ROW_NUMBER() OVER(PARTITION BY sci.cluster_id ORDER BY s.final_score DESC) AS cluster_rank
            FROM photo_similarity_cluster_items sci
            JOIN photo_similarity_clusters c ON c.id=sci.cluster_id
            JOIN photo_culling_scores s ON s.photo_id=sci.photo_id
              AND s.persona_key=%s
              AND s.scoring_version=%s
              AND s.score_status='completed'
            WHERE sci.album_id=%s::uuid
              AND sci.album_event_id = ANY(%s::uuid[])
              AND c.cluster_version=%s
              AND c.persona_key=%s
        )
        SELECT *
        FROM clustered
        WHERE cluster_rank <= %s
        ORDER BY final_score DESC
        LIMIT %s;
    """, (persona_key, scoring_version, album_ctx["album_id"], event_ids, cluster_version, persona_key, max_per_cluster, limit))

    if not candidate_rows:
        candidate_rows = db_all("""
            SELECT
                s.photo_id,
                NULL::uuid AS cluster_id,
                s.album_event_id,
                s.final_score,
                s.reasons,
                1 AS rank_in_cluster,
                1 AS cluster_rank
            FROM photo_culling_scores s
            JOIN photos p ON p.id=s.photo_id
            WHERE s.album_id=%s::uuid
              AND s.album_event_id = ANY(%s::uuid[])
              AND s.persona_key=%s
              AND s.scoring_version=%s
              AND s.score_status='completed'
              AND COALESCE(p.is_deleted,false)=false
            ORDER BY s.final_score DESC
            LIMIT %s;
        """, (album_ctx["album_id"], event_ids, persona_key, scoring_version, limit))

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO best_photo_collections(
                        album_id, name, collection_type, album_type, persona_key,
                        requested_count, selected_count, selection_mode, status,
                        config, created_by, created_at, updated_at
                    ) VALUES(
                        %s::uuid,%s,'ai_best_picks',%s,%s,
                        %s,%s,%s,'completed',%s,%s,now(),now()
                    )
                    RETURNING id;
                """, (
                    album_ctx["album_id"], collection_name, album_type, persona_key,
                    limit, len(candidate_rows), payload.get("selection_mode", "balanced"),
                    Json(opts), payload.get("created_by"),
                ))
                collection_id = str(cur.fetchone()["id"])

                item_rows = []
                for rank, r in enumerate(candidate_rows, start=1):
                    reasons = r.get("reasons") or []
                    reason_text = ", ".join(reasons[:3]) if isinstance(reasons, list) else str(reasons or "AI selected best photo")
                    item_rows.append((
                        collection_id, str(r["photo_id"]), str(r["cluster_id"]) if r.get("cluster_id") else None,
                        album_ctx["album_id"], str(r["album_event_id"]), rank, float(r.get("final_score") or 0),
                        reason_text, Json({"reasons": reasons, "source": "ai_culling"}),
                    ))

                if item_rows:
                    execute_values(cur, """
                        INSERT INTO best_photo_collection_items(
                            collection_id, photo_id, cluster_id, album_id, album_event_id,
                            rank, score, reason, reason_json, created_at
                        ) VALUES %s;
                    """, item_rows, template="(%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,now())")
    finally:
        conn.close()

    return {
        "collection_id": collection_id,
        "requested_count": limit,
        "selected_count": len(candidate_rows),
        "persona_key": persona_key,
        "collection_name": collection_name,
    }


def best_photos_full(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Repeated calls are safe because each stage supports only_missing/only_unscored behavior.
    score_payload = {**payload, "limit": int(payload.get("score_limit") or payload.get("limit") or 5000), "only_unscored": payload.get("only_unscored", True)}
    embed_payload = {**payload, "limit": int(payload.get("embed_limit") or payload.get("limit") or 5000), "only_missing": payload.get("only_missing", True)}

    update_job_status(job_id, "running", "score_photos", "Scoring photos for culling")
    score_result = score_photos(job_id, score_payload)

    update_job_status(job_id, "running", "image_embed_photos", "Generating image embeddings")
    embed_result = image_embed_photos(job_id, embed_payload)

    update_job_status(job_id, "running", "cluster_photos", "Clustering similar photos")
    cluster_result = cluster_photos(job_id, payload)

    update_job_status(job_id, "running", "select_best_photos", "Selecting best photos")
    select_result = select_best_photos(job_id, payload)

    return {
        "score_photos": score_result,
        "image_embed_photos": embed_result,
        "cluster_photos": cluster_result,
        "select_best_photos": select_result,
    }


def process_culling_mode(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = payload.get("mode")
    if not payload.get("album_slug"):
        raise ValueError("Missing input.album_slug")

    if mode == "score_photos":
        return score_photos(job_id, payload)
    if mode == "image_embed_photos":
        return image_embed_photos(job_id, payload)
    if mode == "cluster_photos":
        return cluster_photos(job_id, payload)
    if mode == "select_best_photos":
        return select_best_photos(job_id, payload)
    if mode == "best_photos_full":
        return best_photos_full(job_id, payload)

    raise ValueError(f"Unsupported culling mode: {mode}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def normalize_steps(payload: Dict[str, Any]) -> Dict[str, bool]:
    if payload.get("full_mode", False) or payload.get("qwen_full_mode", False):
        return {
            "ingest": False,
            "compress": False,
            "face_index": False,
            "safe_people_reconcile": False,
            "rebuild_people": False,
            "qwen": True,
            "embeddings": True,
            "culling": bool(payload.get("culling_enabled", payload.get("run_culling", True))),
            "cleanup_temp": bool(payload.get("cleanup_temp", False)),
        }

    default_steps = {
        "ingest": False,
        "compress": False,
        "face_index": False,
        "safe_people_reconcile": False,
        "rebuild_people": False,
        "qwen": True,
        "embeddings": True,
        "culling": False,
        "cleanup_temp": False,
    }

    supplied = payload.get("steps")
    if supplied:
        merged = {**default_steps, **supplied}
        merged["ingest"] = False
        merged["compress"] = False
        merged["face_index"] = False
        merged["safe_people_reconcile"] = False
        merged["rebuild_people"] = False
        if payload.get("culling_enabled") is not None or payload.get("run_culling") is not None:
            merged["culling"] = bool(payload.get("culling_enabled", payload.get("run_culling")))
        return merged

    return default_steps

def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "album_slug" not in payload:
        raise ValueError("Missing input.album_slug")

    if "events" not in payload or not payload["events"]:
        raise ValueError("Missing input.events")

    album_slug = payload["album_slug"]
    album_name = payload.get("album_name")
    events = payload["events"]

    steps = normalize_steps(payload)

    if steps.get("rebuild_people", False):
        raise RuntimeError(
            "Blocked: destructive rebuild_people is not allowed. "
            "This protects manually renamed people. "
            "Use safe_people_reconcile=true in the Face worker instead."
        )

    update_job_status(job_id, "running", "restore_album", "Restoring album context")
    album_ctx = restore_album_context(album_slug, album_name=album_name)

    update_job_status(job_id, "running", "upsert_events", "Creating/restoring event rows")
    db_events = upsert_events(album_ctx, events)

    results: Dict[str, Any] = {
        "album": album_ctx,
        "events": db_events,
        "steps": {},
    }

    if steps.get("qwen", False):
        update_job_status(job_id, "running", "qwen", "Running Qwen metadata")
        results["steps"]["qwen"] = run_qwen_for_events(album_ctx, db_events)

    if steps.get("embeddings", False):
        update_job_status(job_id, "running", "embeddings", "Generating text embeddings")
        results["steps"]["embeddings"] = run_text_embeddings_for_events(album_ctx, db_events)

    if steps.get("culling", False):
        culling_payload = {
            **payload,
            "album_slug": album_slug,
            "album_type": payload.get("album_type") or album_ctx.get("album_type") or "general",
            "persona_key": payload.get("persona_key") or payload.get("album_type") or album_ctx.get("album_type") or "general",
            "limit": int(payload.get("best_photo_count") or payload.get("limit") or payload.get("requested_count") or 100),
            "requested_count": int(payload.get("best_photo_count") or payload.get("limit") or payload.get("requested_count") or 100),
            "score_limit": int(payload.get("score_limit") or payload.get("culling_score_limit") or 5000),
            "embed_limit": int(payload.get("embed_limit") or payload.get("culling_embed_limit") or 5000),
            "only_unscored": bool(payload.get("only_unscored", False)),
            "only_missing": bool(payload.get("only_missing", False)),
        }
        update_job_status(job_id, "running", "best_photos_full", "Scoring, embedding, clustering, and selecting best photos")
        results["steps"]["best_photos_full"] = best_photos_full(job_id, culling_payload)

    if steps.get("cleanup_temp", False):
        update_job_status(job_id, "running", "cleanup_temp", "Deleting temporary AI folders")
        results["steps"]["cleanup_temp"] = cleanup_temp_s3(album_ctx, db_events)

    update_job_status(job_id, "running", "final_verify", "Verifying final counts")
    results["final_verify"] = final_verify(album_ctx, db_events)

    update_job_status(job_id, "completed", "done", "Qwen worker completed")
    return results

# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(event):
    started = time.time()

    job_id = event.get("id", "local_test")
    payload = event.get("input", {})

    try:
        print("Qwen Worker Start", flush=True)
        print("job_id:", job_id, flush=True)
        print("payload:", payload, flush=True)

        if payload.get("debug_clear_qwen_cache"):
            targets = [
                "/runpod-volume/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct",
                "/runpod-volume/huggingface/models--Qwen--Qwen2.5-VL-3B-Instruct",
                "/models/huggingface/models--Qwen--Qwen2.5-VL-3B-Instruct",
                "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
            ]
            deleted = []
            for t in targets:
                if os.path.exists(t):
                    shutil.rmtree(t, ignore_errors=True)
                    deleted.append(t)
            return {"ok": True, "deleted": deleted}

        if payload.get("debug_gpu"):
            import torch
            return {
                "ok": True,
                "job_id": job_id,
                "debug_gpu": {
                    "strict_qwen_gpu": STRICT_QWEN_GPU,
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "torch_cuda_available": torch.cuda.is_available(),
                    "torch_device_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "cudnn": torch.backends.cudnn.version(),
                },
            }

        if payload.get("debug_db_connections"):
            rows = db_all("""
                SELECT
                  usename,
                  application_name,
                  client_addr::text AS client_addr,
                  state,
                  COUNT(*) AS connections
                FROM pg_stat_activity
                WHERE datname = current_database()
                GROUP BY usename, application_name, client_addr, state
                ORDER BY connections DESC;
            """)
            max_conn = db_one("SHOW max_connections;")
            total = db_one("SELECT COUNT(*) AS total_connections FROM pg_stat_activity;")
            return {
                "ok": True,
                "max_connections": max_conn.get("max_connections") if max_conn else None,
                "total_connections": int(total["total_connections"]) if total else None,
                "connections": rows,
                "db_pool_max_conn": DB_POOL_MAX_CONN,
                "db_application_name": DB_APPLICATION_NAME,
            }

        culling_modes = {
            "score_photos",
            "image_embed_photos",
            "cluster_photos",
            "select_best_photos",
            "best_photos_full",
        }

        if payload.get("mode") in culling_modes:
            result = process_culling_mode(job_id, payload)
            update_job_status(job_id, "completed", payload.get("mode"), "Culling mode completed")
        else:
            result = process_album_events(job_id, payload)

        return {
            "ok": True,
            "job_id": job_id,
            "execution_seconds": round(time.time() - started, 2),
            "result": result,
        }

    except Exception as e:
        err = {
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }

        print("Qwen worker failed:", err, flush=True)
        update_job_status(job_id, "failed", "error", repr(e), err)

        return {
            "ok": False,
            "job_id": job_id,
            "execution_seconds": round(time.time() - started, 2),
            **err,
        }
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
