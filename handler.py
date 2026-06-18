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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
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

# Compression parallelism.
# Compression is S3/network + CPU image encode. GPU is not used here.
# Keep this conservative if your RDS is small; each worker only touches DB at the end.
COMPRESS_MAX_WORKERS = int(os.environ.get("COMPRESS_MAX_WORKERS", "6"))
COMPRESS_LOG_EVERY = int(os.environ.get("COMPRESS_LOG_EVERY", "25"))

# If true, and ai_input_s3_key already exists in S3, mark compression completed without regenerating.
# For a true full rerun, keep false. For faster retries, set true.
COMPRESS_REUSE_EXISTING_AI_INPUT = os.environ.get("COMPRESS_REUSE_EXISTING_AI_INPUT", "false").lower() == "true"

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

# Split-worker / GPU strictness settings.
# Keep strict=true: the worker fails if detector/recognizer are not backed by CUDA.
STRICT_FACE_GPU = os.environ.get("STRICT_FACE_GPU", "true").lower() == "true"
INSIGHTFACE_ALLOWED_MODULES = [
    x.strip()
    for x in os.environ.get("INSIGHTFACE_ALLOWED_MODULES", "detection,recognition").split(",")
    if x.strip()
]

# Qwen endpoint injected by Lambda/RunPod template. QWEN_RUN_URL is preferred;
# QWEN_ENDPOINT_ID is kept as a fallback.
QWEN_ENDPOINT_ID = os.environ.get("QWEN_ENDPOINT_ID")
QWEN_RUN_URL = os.environ.get("QWEN_RUN_URL")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# Qwen settings
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_IMAGE_MAX_SIDE = int(os.environ.get("QWEN_IMAGE_MAX_SIDE", "448"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "320"))
QWEN_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_INFERENCE_BATCH_SIZE", "4"))

# Text embedding settings
TEXT_EMBED_MODEL_ID = os.environ.get("TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBED_BATCH_SIZE = int(os.environ.get("TEXT_EMBED_BATCH_SIZE", "64"))

# Image embedding / AI culling settings
IMAGE_EMBED_MODEL_ID = os.environ.get("IMAGE_EMBED_MODEL_ID", "openai/clip-vit-base-patch32")
IMAGE_EMBED_BATCH_SIZE = int(os.environ.get("IMAGE_EMBED_BATCH_SIZE", "16"))
CULLING_VERSION = os.environ.get("CULLING_VERSION", "v1")
CLUSTER_VERSION = os.environ.get("CLUSTER_VERSION", "v1")

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
    ".jpg", ".jpeg", ".jpe", ".png", ".webp", ".heic", ".heif",
    ".nef", ".cr2", ".arw", ".dng", ".tif", ".tiff",
    ".bmp", ".gif", ".avif", ".jfif"
}

RAW_IMAGE_EXTS = {".nef", ".cr2", ".arw", ".dng"}
HEIF_IMAGE_EXTS = {".heic", ".heif"}

UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_.+",
    re.IGNORECASE,
)

s3 = boto3.client(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    config=Config(
        max_pool_connections=max(32, COMPRESS_MAX_WORKERS * 4),
        retries={"max_attempts": 8, "mode": "standard"},
    ),
)

_FACE_APP = None
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_PROCESS_VISION_INFO = None
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
DB_POOL_MAX_CONN = int(os.environ.get("DB_POOL_MAX_CONN", "8"))
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

    from psycopg2.pool import ThreadedConnectionPool

    last_err = None
    for attempt in range(DB_CONNECT_RETRIES):
        try:
            _DB_POOL = ThreadedConnectionPool(
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

        except Exception as e:
            # Threaded compression can temporarily exhaust the tiny DB pool.
            # Retry instead of failing the photo immediately.
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

GENERATED_S3_PATH_PARTS = (
    "/ai-input/",
    "/annotated/",
    "/thumbnail/",
    "/thumbnails/",
    "/preview/",
    "/previews/",
    "/watermarked/",
    "/faces/",
    "/covers/",
    "/edited/",
    "/exports/",
    "/collage/",
    "/collages/",
)


def normalize_s3_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def force_originals_prefix(album_slug: str, event_slug: str, source_prefix: Optional[str] = None) -> str:
    """
    Always scan originals only.

    Protects us when caller accidentally sends:
      albums/<album>/events/<event>
    instead of:
      albums/<album>/events/<event>/originals/
    """
    raw = normalize_s3_prefix(source_prefix or "")
    if "/originals/" in raw:
        return raw
    return f"albums/{album_slug}/events/{event_slug}/originals/"


def normalize_event_source_prefixes(album_slug: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for event in events:
        item = dict(event)
        slug = item.get("slug")
        if not slug:
            raise ValueError(f"Event is missing slug: {event}")
        item["source_prefix"] = force_originals_prefix(album_slug, slug, item.get("source_prefix"))
        normalized.append(item)
    return normalized


def originals_prefix_for_event(album_ctx: Dict[str, Any], event: Dict[str, Any]) -> str:
    return force_originals_prefix(
        album_ctx["album_slug"],
        event["slug"],
        event.get("source_prefix"),
    )


def is_generated_pipeline_key(key: str) -> bool:
    if not key:
        return True
    normalized = "/" + str(key).strip().lstrip("/")
    return any(part in normalized for part in GENERATED_S3_PATH_PARTS)


def is_image_key(key: str) -> bool:
    # S3 keys under source_prefix/originals are supposed to be photos.
    # Never ingest generated pipeline images as new originals.
    if not key or key.endswith("/"):
        return False

    if is_generated_pipeline_key(key):
        return False

    name = Path(key).name
    if name.startswith(".") or name.lower() in {"thumbs.db", ".ds_store"}:
        return False

    ext = Path(key).suffix.lower()
    if ext in IMAGE_EXTS:
        return True

    # Be liberal only inside originals for odd photographer extensions.
    return "/originals/" in ("/" + str(key).strip().lstrip("/"))

def is_generated_original_key(key: str) -> bool:
    return bool(UUID_PREFIX_RE.match(Path(key).name))



def list_s3_objects(prefix: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    prefix = normalize_s3_prefix(prefix)

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
    """
    Read common photographer image formats by content, not only by extension.

    Supported:
    - JPEG/PNG/WebP/TIFF/BMP/GIF/AVIF if Pillow supports it in the image
    - HEIC/HEIF if pillow-heif is installed
    - RAW files NEF/CR2/ARW/DNG via rawpy
    - Unknown extensions are still attempted with PIL and then OpenCV
    """
    ext = local_path.suffix.lower()

    if ext in RAW_IMAGE_EXTS:
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
                "Install rawpy or convert RAW before upload."
            ) from e

    if ext in HEIF_IMAGE_EXTS:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception as e:
            print(f"HEIF opener not available for {local_path}: {repr(e)}", flush=True)

    try:
        img = Image.open(local_path)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as pil_err:
        # Last fallback: OpenCV can decode some images by content even when the suffix is odd.
        try:
            data = np.fromfile(str(local_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb).convert("RGB")
        except Exception as cv_err:
            raise RuntimeError(
                f"Image read failed for {local_path}. "
                f"PIL error={repr(pil_err)}; OpenCV error={repr(cv_err)}"
            ) from cv_err


def save_ai_input_webp(source_local: Path, output_local: Path) -> Tuple[int, int]:
    img = read_image_any(source_local)
    img.thumbnail((AI_INPUT_MAX_SIDE, AI_INPUT_MAX_SIDE), Image.Resampling.LANCZOS)
    output_local.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_local, "WEBP", quality=AI_INPUT_WEBP_QUALITY, method=6)
    return img.size


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
    total_non_image_skipped = 0
    created_rows = 0
    skipped_existing = 0
    failed = 0
    per_event = []

    for event in events:
        requested_prefix = normalize_s3_prefix(event.get("source_prefix") or "")
        prefix = originals_prefix_for_event(album_ctx, event)
        event_id = event["event_id"]

        objects = list_s3_objects(prefix)
        generated_objects = [o for o in objects if is_generated_pipeline_key(o.get("Key", ""))]
        image_objects = [o for o in objects if is_image_key(o.get("Key", ""))]
        usable_objects = image_objects
        non_image_skipped = max(0, len(objects) - len(generated_objects) - len(image_objects))

        total_scanned += len(objects)
        total_images += len(image_objects)
        total_usable += len(usable_objects)
        total_generated_skipped += len(generated_objects)
        total_non_image_skipped += non_image_skipped

        print(
            f"Scanning {event['name']}: "
            f"requested_prefix={requested_prefix}, actual_prefix={prefix}, "
            f"raw={len(objects)}, images={len(image_objects)}, usable={len(usable_objects)}, "
            f"generated_skipped={len(generated_objects)}, non_image_skipped={non_image_skipped}",
            flush=True,
        )

        event_created = 0
        event_existing = 0
        event_failed = 0

        for obj in usable_objects:
            key = obj["Key"]
            size = int(obj.get("Size") or 0)

            try:
                if is_generated_pipeline_key(key):
                    continue

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
            "requested_prefix": requested_prefix,
            "actual_prefix": prefix,
            "objects": len(objects),
            "image_objects": len(image_objects),
            "usable_images": len(usable_objects),
            "generated_skipped": len(generated_objects),
            "non_image_skipped": non_image_skipped,
            "created_rows": event_created,
            "skipped_existing": event_existing,
            "failed": event_failed,
        })

    result = {
        "total_scanned": total_scanned,
        "total_image_objects": total_images,
        "total_usable_images": total_usable,
        "total_generated_skipped": total_generated_skipped,
        "total_non_image_skipped": total_non_image_skipped,
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

def compress_one_photo(
    row: Dict[str, Any],
    photo_cols: Optional[set[str]] = None,
) -> Tuple[str, str, Optional[str]]:
    photo_id = str(row["id"])

    try:
        tmpdir = LOCAL_WORK / "compress" / photo_id
        tmpdir.mkdir(parents=True, exist_ok=True)

        source_key = row.get("original_s3_key") or row.get("source_s3_key")
        if not source_key:
            raise RuntimeError("missing original/source S3 key")

        ai_key = row.get("ai_input_s3_key")
        if not ai_key:
            album_slug = row["storage_album_slug"]
            event_slug = row["storage_event_slug"]
            photo_uuid = row.get("photo_uuid") or photo_id
            ai_key = f"albums/{album_slug}/events/{event_slug}/ai-input/{photo_uuid}.webp"

        # Optional fast path for retries: do not regenerate if the AI input object already exists.
        if COMPRESS_REUSE_EXISTING_AI_INPUT and ai_key and s3_key_exists(ai_key):
            cols = photo_cols or table_columns("photos")
            set_parts = [
                "compression_status='completed'",
                "compression_error=NULL",
                "ai_input_s3_key=%s",
                "updated_at=now()",
            ]
            values: List[Any] = [ai_key]
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

        suffix = Path(source_key).suffix or ".img"
        original_local = tmpdir / f"original{suffix}"
        ai_local = tmpdir / "ai.webp"

        download_file(source_key, original_local)
        width, height = save_ai_input_webp(original_local, ai_local)
        upload_file(ai_local, ai_key, "image/webp")

        cols = photo_cols or table_columns("photos")
        set_parts = [
            "compression_status='completed'",
            "compression_error=NULL",
            "ai_input_s3_key=%s",
            "updated_at=now()",
        ]
        values: List[Any] = [ai_key]

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
        execute_sql_best_effort("""
            UPDATE photos
            SET compression_status='failed',
                compression_error=%s,
                updated_at=now()
            WHERE id=%s::uuid;
        """, (err, photo_id))
        return "failed", photo_id, err

    finally:
        # Keep /tmp from growing during big albums.
        try:
            shutil.rmtree(LOCAL_WORK / "compress" / photo_id, ignore_errors=True)
        except Exception:
            pass


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

    if not rows:
        result = {
            "rows": 0,
            "ok": 0,
            "failed": 0,
            "errors": [],
            "parallel": True,
            "workers": 0,
        }
        print("Compression result:", result, flush=True)
        return result

    worker_count = max(1, min(COMPRESS_MAX_WORKERS, len(rows)))
    photo_cols = table_columns("photos")

    ok = 0
    failed = 0
    errors = []
    started = time.time()

    print(
        f"Compression parallel start: rows={len(rows)}, workers={worker_count}, "
        f"reuse_existing_ai_input={COMPRESS_REUSE_EXISTING_AI_INPUT}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(compress_one_photo, row, photo_cols) for row in rows]

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                status, photo_id, err = future.result()
            except Exception as e:
                status, photo_id, err = "failed", "unknown", repr(e)

            if status == "ok":
                ok += 1
            else:
                failed += 1
                if len(errors) < 25:
                    errors.append({"photo_id": photo_id, "error": err})

            if COMPRESS_LOG_EVERY > 0 and (i % COMPRESS_LOG_EVERY == 0 or i == len(rows)):
                elapsed = max(0.001, time.time() - started)
                print(
                    f"Compression progress: {i}/{len(rows)} done, ok={ok}, failed={failed}, "
                    f"rate={i / elapsed:.2f} photos/sec",
                    flush=True,
                )

    result = {
        "rows": len(rows),
        "ok": ok,
        "failed": failed,
        "errors": errors[:25],
        "parallel": True,
        "workers": worker_count,
        "seconds": round(time.time() - started, 2),
        "photos_per_second": round(len(rows) / max(0.001, time.time() - started), 3),
    }
    print("Compression result:", result, flush=True)
    return result


# ============================================================
# INSIGHTFACE
# ============================================================

class InsightFaceCudaLoader:
    """
    Strict CUDA loader for InsightFace + ONNXRuntime.

    Why this exists:
    - ort.get_available_providers() can show CUDAExecutionProvider even when CUDA cannot actually load.
    - InsightFace may silently fall back to CPUExecutionProvider.
    - We want to fail early if required CUDA/cuDNN libs are missing.
    """

    REQUIRED_CUDNN_LIBS = [
        "libcudnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
    ]

    OPTIONAL_CUDNN_LIBS = [
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
    ]

    CUDA_LIBS = [
        "libcuda.so.1",
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
        "libnvrtc.so.12",
        "libnvJitLink.so.12",
        "libnvToolsExt.so.1",
    ]

    def __init__(
        self,
        det_size: Tuple[int, int],
        allowed_modules: List[str],
        strict_gpu: bool = True,
    ):
        self.det_size = det_size
        self.allowed_modules = allowed_modules
        self.strict_gpu = strict_gpu

    def find_candidate_lib_dirs(self) -> List[str]:
        import site
        import sysconfig
        import glob

        dirs: List[str] = []

        for base in site.getsitepackages():
            dirs.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))

        user_site = site.getusersitepackages()
        if user_site:
            dirs.extend(glob.glob(os.path.join(user_site, "nvidia", "*", "lib")))

        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            dirs.extend(glob.glob(os.path.join(purelib, "nvidia", "*", "lib")))

        dirs.extend([
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/compat",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/conda/lib",
        ])

        clean: List[str] = []
        seen = set()

        for d in dirs:
            if not d:
                continue
            if not os.path.isdir(d):
                continue
            if d in seen:
                continue
            seen.add(d)
            clean.append(d)

        return clean

    def update_library_path(self, candidate_dirs: List[str]) -> None:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = candidate_dirs + ([existing] if existing else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

        # ONNXRuntime also checks this in some CUDA builds.
        existing_ort = os.environ.get("ORT_CUDA_LIB_PATH", "")
        if not existing_ort:
            cuda_dirs = [
                d for d in candidate_dirs
                if "/nvidia/" in d or "/cuda" in d or "conda" in d
            ]
            if cuda_dirs:
                os.environ["ORT_CUDA_LIB_PATH"] = ":".join(cuda_dirs)

    def find_lib(self, candidate_dirs: List[str], lib_name: str) -> Optional[str]:
        for d in candidate_dirs:
            path = os.path.join(d, lib_name)
            if os.path.exists(path):
                return path
        return None

    def preload_one(self, path: str) -> Tuple[bool, Optional[str]]:
        import ctypes

        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True, None
        except Exception as e:
            return False, repr(e)

    def preload_cuda_libs_for_onnxruntime(self) -> Dict[str, Any]:
        """
        Preload CUDA/cuDNN from pip-installed nvidia packages before InsightFace creates sessions.

        This intentionally fails if required cuDNN 9 split libs are missing.
        """
        candidate_dirs = self.find_candidate_lib_dirs()
        self.update_library_path(candidate_dirs)

        loaded: List[str] = []
        failed: List[Dict[str, str]] = []
        found_required: Dict[str, Optional[str]] = {}

        # Load lower-level CUDA libs first.
        for lib_name in self.CUDA_LIBS:
            path = self.find_lib(candidate_dirs, lib_name)
            if not path:
                continue

            ok, err = self.preload_one(path)
            if ok:
                loaded.append(path)
            else:
                failed.append({"lib": lib_name, "path": path, "error": err or ""})

        # Required cuDNN libs.
        for lib_name in self.REQUIRED_CUDNN_LIBS:
            path = self.find_lib(candidate_dirs, lib_name)
            found_required[lib_name] = path

            if not path:
                continue

            ok, err = self.preload_one(path)
            if ok:
                loaded.append(path)
            else:
                failed.append({"lib": lib_name, "path": path, "error": err or ""})

        # Optional cuDNN libs. Load if present.
        found_optional: Dict[str, Optional[str]] = {}
        for lib_name in self.OPTIONAL_CUDNN_LIBS:
            path = self.find_lib(candidate_dirs, lib_name)
            found_optional[lib_name] = path

            if not path:
                continue

            ok, err = self.preload_one(path)
            if ok:
                loaded.append(path)
            else:
                failed.append({"lib": lib_name, "path": path, "error": err or ""})

        missing_required = [
            name for name, path in found_required.items()
            if not path
        ]

        payload = {
            "candidate_dirs": candidate_dirs,
            "loaded": loaded,
            "failed": failed,
            "found_required_cudnn": found_required,
            "found_optional_cudnn": found_optional,
            "missing_required_cudnn": missing_required,
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "ort_cuda_lib_path": os.environ.get("ORT_CUDA_LIB_PATH", ""),
        }

        print("CUDA preload result:", json.dumps(payload, default=str), flush=True)

        if missing_required:
            raise RuntimeError(
                "Missing required cuDNN libraries for ONNXRuntime CUDAExecutionProvider: "
                f"{missing_required}. "
                "Fix Docker image requirements. Add nvidia-cudnn-cu12, rebuild without cache, "
                "and do not allow CPU fallback."
            )

        hard_failures = [
            f for f in failed
            if f["lib"] in self.REQUIRED_CUDNN_LIBS
        ]

        if hard_failures:
            raise RuntimeError(
                "Required cuDNN libraries were found but failed to preload: "
                f"{hard_failures}"
            )

        return payload

    def print_torch_status(self) -> None:
        import torch

        print("torch:", torch.__version__, flush=True)
        print("torch cuda:", torch.version.cuda, flush=True)
        print("torch cuda available:", torch.cuda.is_available(), flush=True)
        print("torch device count:", torch.cuda.device_count(), flush=True)

        if torch.cuda.is_available():
            print("torch gpu name:", torch.cuda.get_device_name(0), flush=True)
            print("torch gpu capability:", torch.cuda.get_device_capability(0), flush=True)

    def validate_torch_cuda(self) -> None:
        import torch

        if not self.strict_gpu:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "STRICT_FACE_GPU=true but torch.cuda.is_available() is false. "
                "Refusing to run face indexing on CPU."
            )

        try:
            # Real CUDA execution check, not just availability.
            x = torch.randn((256, 256), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            del x, y
        except Exception as e:
            raise RuntimeError(
                "Torch CUDA test failed. Refusing to run face indexing."
            ) from e

    def get_onnxruntime_providers(self) -> List[str]:
        import onnxruntime as ort

        try:
            ort.set_default_logger_severity(2)
        except Exception:
            pass

        available = ort.get_available_providers()
        print("onnxruntime:", ort.__version__, flush=True)
        print("ONNXRuntime device:", ort.get_device(), flush=True)
        print("ONNXRuntime providers:", available, flush=True)
        return available

    def build_provider_list(self) -> List[Any]:
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                    "do_copy_in_default_stream": "1",
                },
            ),

            # CPU is listed second only as fallback for unsupported ops.
            # We still validate detection/recognition sessions start with CUDA.
            "CPUExecutionProvider",
        ]

    def load(self):
        self.preload_cuda_libs_for_onnxruntime()

        self.print_torch_status()
        self.validate_torch_cuda()

        available = self.get_onnxruntime_providers()

        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider missing. Available providers={available}. "
                "Refusing to run InsightFace on CPU because GPU is required."
            )

        from insightface.app import FaceAnalysis

        providers = self.build_provider_list()

        print("InsightFace allowed modules:", self.allowed_modules, flush=True)
        print("InsightFace provider request:", providers, flush=True)

        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=self.allowed_modules,
        )

        app.prepare(ctx_id=0, det_size=self.det_size)

        required_models = set(self.allowed_modules)
        loaded_models = set(app.models.keys())

        print("InsightFace loaded models:", sorted(loaded_models), flush=True)

        missing = required_models - loaded_models
        if missing:
            raise RuntimeError(
                f"InsightFace did not load required modules: {sorted(missing)}"
            )

        provider_report: Dict[str, List[str]] = {}

        for name, model in app.models.items():
            sess = getattr(model, "session", None)
            if not sess:
                print(
                    f"InsightFace model {name} has no ONNX session; skipping provider validation",
                    flush=True,
                )
                continue

            session_providers = sess.get_providers()
            provider_report[name] = session_providers

            print(
                f"InsightFace model {name} providers: {session_providers}",
                flush=True,
            )

            if self.strict_gpu and name in required_models:
                if not session_providers:
                    raise RuntimeError(
                        f"InsightFace required model {name} has no providers."
                    )

                if session_providers[0] != "CUDAExecutionProvider":
                    raise RuntimeError(
                        f"InsightFace required model {name} did not start with CUDA first. "
                        f"Providers={session_providers}. "
                        "This usually means ONNXRuntime CUDA provider could not load cuDNN/CUDA libs."
                    )

        print(
            "InsightFace required models loaded with CUDAExecutionProvider:",
            provider_report,
            flush=True,
        )

        return app


def preload_cuda_libs_for_onnxruntime():
    """
    Backward-compatible wrapper for existing code.
    """
    loader = InsightFaceCudaLoader(
        det_size=FACE_DET_SIZE,
        allowed_modules=INSIGHTFACE_ALLOWED_MODULES,
        strict_gpu=STRICT_FACE_GPU,
    )
    return loader.preload_cuda_libs_for_onnxruntime()


def load_face_app():
    global _FACE_APP

    if _FACE_APP is not None:
        return _FACE_APP

    loader = InsightFaceCudaLoader(
        det_size=FACE_DET_SIZE,
        allowed_modules=INSIGHTFACE_ALLOWED_MODULES,
        strict_gpu=STRICT_FACE_GPU,
    )

    _FACE_APP = loader.load()
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



def _event_ids(events: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [str(e["event_id"]) for e in (events or []) if e.get("event_id")]


def _event_filter_sql(alias: str, event_ids: List[str]) -> str:
    if not event_ids:
        return ""
    return f"AND {alias}.album_event_id = ANY(%s::uuid[])"


def safe_add_new_people_without_touching_existing_names(
    album_ctx: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]] = None,
    build_candidates: bool = False,
    crop_covers: bool = False,
) -> Dict[str, Any]:
    """
    Safe reconciliation.

    - Labels only unlabeled faces.
    - Preserves existing people and display names.
    - For event runs, processes only those event faces.
    - Still matches event faces against existing album-level people.
    - Does not auto-merge duplicates.
    - Expensive cover-crop/duplicate-candidate steps are opt-in.
    """
    started = time.time()
    album_id = album_ctx["album_id"]
    event_ids = _event_ids(events)
    face_event_filter = _event_filter_sql("f", event_ids)

    face_params: List[Any] = [album_id]
    if event_ids:
        face_params.append(event_ids)

    print(
        "safe_people_reconcile start:",
        {
            "album_id": album_id,
            "event_scoped": bool(event_ids),
            "event_ids": event_ids,
            "build_candidates": build_candidates,
            "crop_covers": crop_covers,
        },
        flush=True,
    )

    face_rows = db_all(f"""
        SELECT
            f.id,
            f.album_id,
            f.album_event_id,
            f.photo_id,
            f.embedding,
            f.face_quality_score,
            f.detection_confidence
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.album_id = %s::uuid
          {face_event_filter}
          AND f.person_id IS NULL
          AND f.embedding IS NOT NULL
          AND COALESCE(p.is_deleted, false) = false
        ORDER BY f.face_quality_score DESC NULLS LAST, f.created_at;
    """, tuple(face_params))

    print(
        "safe_people_reconcile loaded unlabeled faces:",
        {"count": len(face_rows), "seconds": round(time.time() - started, 2)},
        flush=True,
    )

    if not face_rows:
        rebuild_result = rebuild_photo_people_base_safe(album_ctx, events)
        return {
            "event_scoped": bool(event_ids),
            "event_ids": event_ids,
            "unlabeled_faces": 0,
            "assigned_to_existing_people": 0,
            "new_people_created": 0,
            "rebuild_photo_people": rebuild_result,
            "message": "No unlabeled faces found. Existing people untouched.",
            "seconds": round(time.time() - started, 2),
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

    print(
        "safe_people_reconcile loaded existing people:",
        {"count": len(existing_people), "seconds": round(time.time() - started, 2)},
        flush=True,
    )

    face_vecs = [parse_pg_vector(r["embedding"]) for r in face_rows]

    assigned_to_existing: List[Tuple[str, str]] = []
    remaining_indices = list(range(len(face_rows)))

    if existing_people:
        people_vecs = [parse_pg_vector(p["centroid_embedding"]) for p in existing_people]
        P = np.stack(people_vecs).astype(np.float32)
        still_remaining: List[int] = []
        batch_size = 2048

        for start_i in range(0, len(face_vecs), batch_size):
            end_i = min(len(face_vecs), start_i + batch_size)
            F = np.stack(face_vecs[start_i:end_i]).astype(np.float32)
            sims = F @ P.T
            best_people_idx = sims.argmax(axis=1)
            best_scores = sims.max(axis=1)

            for local_i, score in enumerate(best_scores):
                global_i = start_i + local_i
                if float(score) >= PEOPLE_MATCH_EXISTING_SIM_THRESHOLD:
                    person = existing_people[int(best_people_idx[local_i])]
                    assigned_to_existing.append((str(person["id"]), str(face_rows[global_i]["id"])))
                else:
                    still_remaining.append(global_i)

            print(
                "safe_people_reconcile match progress:",
                {
                    "processed_faces": end_i,
                    "total_faces": len(face_vecs),
                    "assigned_so_far": len(assigned_to_existing),
                    "remaining_so_far": len(still_remaining),
                    "seconds": round(time.time() - started, 2),
                },
                flush=True,
            )

        remaining_indices = still_remaining

    print(
        "safe_people_reconcile matching complete:",
        {
            "assigned_to_existing": len(assigned_to_existing),
            "remaining_for_new_clusters": len(remaining_indices),
            "seconds": round(time.time() - started, 2),
        },
        flush=True,
    )

    new_clusters: Dict[int, List[int]] = {}

    if remaining_indices:
        if len(remaining_indices) == 1:
            new_clusters[0] = remaining_indices
        else:
            X = np.stack([face_vecs[i] for i in remaining_indices]).astype(np.float32)
            dsu = DSU(len(remaining_indices))
            block = 512

            for start_i in range(0, len(remaining_indices), block):
                end_i = min(len(remaining_indices), start_i + block)
                sims_block = X[start_i:end_i] @ X.T

                for local_i in range(end_i - start_i):
                    i = start_i + local_i
                    for j in range(i + 1, len(remaining_indices)):
                        if float(sims_block[local_i, j]) >= NEW_FACE_CLUSTER_SIM_THRESHOLD:
                            dsu.union(i, j)

                print(
                    "safe_people_reconcile cluster progress:",
                    {
                        "processed_remaining": end_i,
                        "total_remaining": len(remaining_indices),
                        "seconds": round(time.time() - started, 2),
                    },
                    flush=True,
                )

            temp: Dict[int, List[int]] = {}
            for local_i, original_i in enumerate(remaining_indices):
                root = dsu.find(local_i)
                temp.setdefault(root, []).append(original_i)

            new_clusters = {idx: vals for idx, vals in enumerate(temp.values())}

    print(
        "safe_people_reconcile clusters prepared:",
        {"new_clusters": len(new_clusters), "seconds": round(time.time() - started, 2)},
        flush=True,
    )

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
                        template="(%s, %s)",
                    )

                cur.execute("""
                    SELECT COALESCE(MAX(person_number), 0) AS max_num
                    FROM people
                    WHERE album_id = %s::uuid;
                """, (album_id,))
                next_num = int(cur.fetchone()["max_num"] or 0) + 1

                for cluster_i, face_indices in new_clusters.items():
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
                        template="(%s, %s)",
                    )

                    print(
                        "safe_people_reconcile created person:",
                        {
                            "person_number": next_num,
                            "person_id": person_id,
                            "cluster_index": cluster_i,
                            "faces": len(group_faces),
                        },
                        flush=True,
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

    print(
        "safe_people_reconcile DB updates complete:",
        {
            "assigned_to_existing": len(assigned_to_existing),
            "new_people_created": new_people_created,
            "seconds": round(time.time() - started, 2),
        },
        flush=True,
    )

    rebuild_result = rebuild_photo_people_base_safe(album_ctx, events)

    cover_result = None
    if crop_covers:
        cover_result = crop_and_upload_missing_person_covers(album_ctx)

    duplicate_result = None
    if build_candidates:
        duplicate_result = build_duplicate_candidates(album_ctx)

    result = {
        "event_scoped": bool(event_ids),
        "event_ids": event_ids,
        "unlabeled_faces": len(face_rows),
        "assigned_to_existing_people": len(assigned_to_existing),
        "remaining_faces_clustered": len(remaining_indices),
        "new_clusters": len(new_clusters),
        "new_people_created": new_people_created,
        "existing_people_untouched": True,
        "names_preserved": True,
        "rebuild_photo_people": rebuild_result,
        "cover_crop": cover_result,
        "duplicate_candidates": duplicate_result,
        "seconds": round(time.time() - started, 2),
    }

    print("safe_people_reconcile result:", result, flush=True)
    return result


def rebuild_photo_people_base_safe(
    album_ctx: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Rebuild photo_people safely.

    If events are supplied, only rebuild those event rows.
    This avoids deleting/rebuilding the entire album during a one-event run.
    """
    album_id = album_ctx["album_id"]
    event_ids = _event_ids(events)

    delete_event_filter = ""
    face_event_filter = ""
    delete_params: List[Any] = [album_id]
    insert_params: List[Any] = [album_id]

    if event_ids:
        delete_event_filter = "AND album_event_id = ANY(%s::uuid[])"
        face_event_filter = "AND f.album_event_id = ANY(%s::uuid[])"
        delete_params.append(event_ids)
        insert_params.append(event_ids)

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM photo_people
                    WHERE album_id = %s::uuid
                      {delete_event_filter};
                """, tuple(delete_params))

                cur.execute(f"""
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
                            ARRAY_AGG(f.id ORDER BY f.face_quality_score DESC NULLS LAST) AS face_ids,
                            AVG(f.detection_confidence) AS conf
                        FROM faces f
                        JOIN photos p ON p.id = f.photo_id
                        WHERE f.album_id = %s::uuid
                          {face_event_filter}
                          AND f.person_id IS NOT NULL
                          AND COALESCE(p.is_deleted, false) = false
                        GROUP BY f.album_id, f.album_event_id, f.photo_id, f.person_id
                    ),
                    enriched AS (
                        SELECT
                            b.*,
                            ARRAY_REMOVE(ARRAY_AGG(DISTINCT o.person_id), b.person_id) AS co_person_ids
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
                        COALESCE(e.co_person_ids, ARRAY[]::uuid[]),
                        COALESCE(NULLIF(pe.display_name, ''), pe.default_name)
                            || ' appears in this photo with '
                            || COALESCE(array_length(e.co_person_ids, 1), 0)::text
                            || ' other people.',
                        e.conf,
                        now(),
                        now()
                    FROM enriched e
                    JOIN people pe ON pe.id = e.person_id;
                """, tuple(insert_params))
    finally:
        conn.close()

    count_params: List[Any] = [album_id]
    count_event_filter = ""
    if event_ids:
        count_event_filter = "AND album_event_id = ANY(%s::uuid[])"
        count_params.append(event_ids)

    row = db_one(f"""
        SELECT COUNT(*) AS photo_people_rows
        FROM photo_people
        WHERE album_id = %s::uuid
          {count_event_filter};
    """, tuple(count_params))

    result = {
        "event_scoped": bool(event_ids),
        "event_ids": event_ids,
        "photo_people_rows": int(row["photo_people_rows"] or 0) if row else 0,
    }

    print("rebuild_photo_people_base_safe result:", result, flush=True)
    return result

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

    print(f"Loading Qwen model: {QWEN_MODEL_ID}", flush=True)

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


def qwen_describe_batch(image_paths: List[Path]) -> List[Dict[str, Any]]:
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

    for batch in chunks(rows, QWEN_INFERENCE_BATCH_SIZE):
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
            print("QWEN BATCH FAILED, retrying individually:", repr(batch_err), flush=True)

            for photo, qwen_img in prepared:
                try:
                    data = qwen_describe_batch([qwen_img])[0]
                    save_qwen_photo(photo, data, label_to_id, event_by_id=event_by_id)
                    ok += 1
                except Exception as e:
                    fail += 1
                    mark_qwen_failed(photo, e)
                    errors.append({"file_name": photo.get("file_name"), "error": repr(e)})

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
    _TEXT_EMBED_MODEL = SentenceTransformer(TEXT_EMBED_MODEL_ID)
    print(f"Text embedding model loaded: {TEXT_EMBED_MODEL_ID}", flush=True)
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

    import torch
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading image embedding model: {IMAGE_EMBED_MODEL_ID}", flush=True)
    processor = CLIPProcessor.from_pretrained(IMAGE_EMBED_MODEL_ID)
    model = CLIPModel.from_pretrained(IMAGE_EMBED_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
# QWEN ENQUEUE FOR SPLIT WORKER
# ============================================================

def _qwen_run_url() -> str:
    if QWEN_RUN_URL:
        return QWEN_RUN_URL
    if QWEN_ENDPOINT_ID:
        return f"https://api.runpod.ai/v2/{QWEN_ENDPOINT_ID}/run"
    raise RuntimeError("Missing QWEN_RUN_URL or QWEN_ENDPOINT_ID")


def build_qwen_payload_after_face(payload: Dict[str, Any]) -> Dict[str, Any]:
    supplied_steps = payload.get("steps") or {}
    full_mode = bool(payload.get("full_mode", False))

    run_qwen = bool(
        full_mode
        or supplied_steps.get("enqueue_qwen", False)
        or supplied_steps.get("qwen", False)
        or payload.get("enqueue_qwen", False)
        or payload.get("run_qwen", False)
    )

    run_embeddings = bool(
        payload.get("run_embeddings", supplied_steps.get("embeddings", True if run_qwen else False))
    )

    if full_mode:
        run_culling = bool(payload.get("culling_enabled", payload.get("run_culling", True)))
    else:
        run_culling = bool(
            payload.get("culling_enabled", payload.get("run_culling", supplied_steps.get("culling", False)))
        )

    qwen_steps = payload.get("qwen_steps") or {
        "ingest": False,
        "compress": False,
        "face_index": False,
        "safe_people_reconcile": False,
        "rebuild_people": False,
        "qwen": run_qwen,
        "embeddings": run_embeddings,
        "culling": run_culling,
        "cleanup_temp": bool(payload.get("cleanup_temp", supplied_steps.get("cleanup_temp", False))),
    }

    qwen_payload = {
        **payload,
        "album_slug": payload["album_slug"],
        "album_name": payload.get("album_name"),
        "events": payload["events"],
        "full_mode": False,
        "steps": qwen_steps,
        "triggered_by": "photo-face-worker",
    }

    qwen_payload.pop("face_full_mode", None)
    qwen_payload.pop("enqueue_qwen", None)
    return qwen_payload


def enqueue_qwen_after_face(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY missing; cannot enqueue Qwen endpoint")

    qwen_payload = build_qwen_payload_after_face(payload)

    if not qwen_payload.get("steps", {}).get("qwen") and not qwen_payload.get("steps", {}).get("embeddings") and not qwen_payload.get("steps", {}).get("culling"):
        return {
            "enqueued": False,
            "reason": "qwen/embeddings/culling steps are all false",
            "qwen_payload": qwen_payload,
        }

    body = json.dumps({"input": qwen_payload}).encode("utf-8")
    url = _qwen_run_url()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen enqueue failed HTTP {e.code}: {raw}") from e

    print("Qwen enqueued:", data, flush=True)
    return {
        "enqueued": True,
        "run_url": url,
        "qwen_job": data,
        "qwen_payload": qwen_payload,
    }


def forward_culling_mode_to_qwen(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY missing; cannot forward culling mode to Qwen endpoint")

    body = json.dumps({"input": payload}).encode("utf-8")
    url = _qwen_run_url()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen forward failed HTTP {e.code}: {raw}") from e

    return {"forwarded": True, "run_url": url, "qwen_job": data}

# ============================================================
# MAIN PIPELINE
# ============================================================

def normalize_steps(payload: Dict[str, Any]) -> Dict[str, bool]:
    """
    Face worker steps.

    Correct order:
    1. ingest
    2. compress
    3. face_index
    4. safe_people_reconcile
    5. crop_person_covers
    6. enqueue_qwen / Gemma

    Important:
    - Qwen/Gemma must be enqueued after people are reconciled.
      Otherwise the image-text worker sees no labeled people and returns rows=0.
    - Person cover generation is separate from person creation.
      It writes people.cover_face_s3_key.
    """

    full_mode = bool(payload.get("full_mode", False) or payload.get("face_full_mode", False))

    default_steps = {
        "ingest": True,
        "compress": bool(full_mode),
        "face_index": bool(full_mode),
        "safe_people_reconcile": bool(full_mode),
        "crop_person_covers": bool(full_mode),
        "enqueue_qwen": bool(full_mode),

        # Keep old/other pipeline steps present but off in face-worker.
        "rebuild_people": False,
        "qwen": False,
        "embeddings": False,
        "culling": False,
        "cleanup_temp": False,
    }

    supplied = payload.get("steps")
    if isinstance(supplied, dict):
        merged = {**default_steps, **supplied}
    else:
        merged = default_steps

    # Never allow destructive people rebuild from this safe path.
    merged["rebuild_people"] = False

    return {k: bool(v) for k, v in merged.items()}


def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "album_slug" not in payload:
        raise ValueError("Missing input.album_slug")

    if "events" not in payload or not payload["events"]:
        raise ValueError("Missing input.events")

    album_slug = payload["album_slug"]
    album_name = payload.get("album_name")
    events = normalize_event_source_prefixes(album_slug, payload["events"])

    steps = normalize_steps(payload)

    if steps.get("rebuild_people", False):
        raise RuntimeError(
            "Blocked: destructive rebuild_people is not allowed. "
            "This protects manually renamed people. "
            "Use safe_people_reconcile=true instead."
        )

    update_job_status(job_id, "running", "restore_album", "Restoring album context")
    album_ctx = restore_album_context(album_slug, album_name=album_name)

    update_job_status(job_id, "running", "upsert_events", "Creating/restoring event rows")
    db_events = upsert_events(album_ctx, events)

    # Use normalized event prefixes for Qwen/Gemma too, not the raw broad prefix from the request.
    normalized_payload = {**payload, "events": db_events}

    results: Dict[str, Any] = {
        "album": album_ctx,
        "events": db_events,
        "steps": {},
    }

    # 1. Ingest originals from S3 into DB photo rows.
    if steps.get("ingest", True):
        update_job_status(
            job_id,
            "running",
            "ingest",
            "Scanning S3 originals and ingesting photos",
        )
        results["steps"]["ingest"] = scan_and_ingest_originals(album_ctx, db_events)

        update_job_status(
            job_id,
            "running",
            "s3_validation",
            "Validating S3 source objects",
        )
        results["steps"]["s3_validation"] = validate_s3_sources(album_ctx, db_events)

    # 2. Generate AI input images.
    if steps.get("compress", False):
        update_job_status(
            job_id,
            "running",
            "compress",
            "Generating AI input images",
        )
        results["steps"]["compress"] = compress_events(album_ctx, db_events)

    # 3. Detect faces and write face embeddings.
    if steps.get("face_index", False):
        update_job_status(
            job_id,
            "running",
            "face_index",
            "Running InsightFace detection/recognition on GPU",
        )
        results["steps"]["face_index"] = face_index_events(album_ctx, db_events)

    # 4. IMPORTANT: reconcile people before Qwen/Gemma enqueue.
    # Do not crop covers inside this call; we run cover crop as its own explicit step below
    # so existing albums can be backfilled without rerunning reconciliation.
    if steps.get("safe_people_reconcile", False):
        update_job_status(
            job_id,
            "running",
            "safe_people_reconcile",
            "Assigning faces to people before image-text metadata",
        )
        results["steps"]["safe_people_reconcile"] = safe_add_new_people_without_touching_existing_names(
            album_ctx,
            db_events,
            build_candidates=False,
            crop_covers=False,
        )

    # 5. Generate/backfill people cover images.
    # This writes people.cover_face_s3_key.
    if steps.get("crop_person_covers", False):
        update_job_status(
            job_id,
            "running",
            "crop_person_covers",
            "Cropping and uploading missing person cover faces",
        )
        results["steps"]["crop_person_covers"] = crop_and_upload_missing_person_covers(album_ctx)

    # Optional duplicate candidate generation, still off by default.
    if bool(payload.get("build_duplicate_candidates", False)):
        update_job_status(
            job_id,
            "running",
            "build_duplicate_candidates",
            "Building possible duplicate people candidates",
        )
        results["steps"]["build_duplicate_candidates"] = build_duplicate_candidates(album_ctx)

    # 6. Only now enqueue Qwen/Gemma, after faces have person_id and covers can exist.
    if steps.get("enqueue_qwen", False):
        update_job_status(
            job_id,
            "running",
            "enqueue_qwen",
            "Enqueuing image-text endpoint after people reconcile",
        )
        results["steps"]["enqueue_qwen"] = enqueue_qwen_after_face(normalized_payload)

    if steps.get("cleanup_temp", False):
        update_job_status(
            job_id,
            "running",
            "cleanup_temp",
            "Deleting temporary AI folders",
        )
        results["steps"]["cleanup_temp"] = cleanup_temp_s3(album_ctx, db_events)

    update_job_status(job_id, "completed", "face_worker", "Face worker completed")
    return results

# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(event):
    started = time.time()

    job_id = event.get("id", "local_test")
    payload = event.get("input", {})

    try:
        print("Face Worker Start", flush=True)
        print("job_id:", job_id, flush=True)
        print("payload:", payload, flush=True)

        if payload.get("debug_gpu"):
            import torch
            import onnxruntime as ort
            return {
                "ok": True,
                "job_id": job_id,
                "debug_gpu": {
                    "strict_face_gpu": STRICT_FACE_GPU,
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "torch_cuda_available": torch.cuda.is_available(),
                    "torch_device_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "onnxruntime": ort.__version__,
                    "onnxruntime_device": ort.get_device(),
                    "onnxruntime_providers": ort.get_available_providers(),
                    "insightface_allowed_modules": INSIGHTFACE_ALLOWED_MODULES,
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
            result = forward_culling_mode_to_qwen(payload)
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

        print("Face worker failed:", err, flush=True)
        update_job_status(job_id, "failed", "error", repr(e), err)

        return {
            "ok": False,
            "job_id": job_id,
            "execution_seconds": round(time.time() - started, 2),
            **err,
        }
    finally:
        gc.collect()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
