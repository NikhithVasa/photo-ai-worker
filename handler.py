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

# Qwen settings
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_IMAGE_MAX_SIDE = int(os.environ.get("QWEN_IMAGE_MAX_SIDE", "448"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "320"))
QWEN_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_INFERENCE_BATCH_SIZE", "4"))

# Text embedding settings
TEXT_EMBED_MODEL_ID = os.environ.get("TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBED_BATCH_SIZE = int(os.environ.get("TEXT_EMBED_BATCH_SIZE", "64"))

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
_TEXT_EMBED_MODEL = None


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


def get_conn():
    assert_env_ready()
    return psycopg2.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        dbname=RDS_DB,
        user=RDS_USER,
        password=RDS_PASSWORD,
        cursor_factory=RealDictCursor,
    )


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
    rows = db_all("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s;
    """, (table_name,))
    return {r["column_name"] for r in rows}


def has_table(table_name: str) -> bool:
    row = db_one("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema='public'
              AND table_name=%s
        ) AS exists;
    """, (table_name,))
    return bool(row and row["exists"])


def execute_sql(sql: str, params: Tuple[Any, ...] = ()) -> None:
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    finally:
        conn.close()


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

def load_face_app():
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

    _FACE_APP = app
    print("InsightFace loaded", flush=True)
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


def save_qwen_photo(photo: Dict[str, Any], data: Dict[str, Any], label_to_id: Dict[str, str]) -> None:
    photo_data = data.get("photo", {})
    people_map = data.get("people_map", {})

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
    execute_sql("""
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

    ok = 0
    fail = 0
    errors = []

    for batch in chunks(rows, QWEN_INFERENCE_BATCH_SIZE):
        prepared = []

        for photo in batch:
            try:
                qwen_img, _faces = annotate_photo(photo)
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
                    save_qwen_photo(photo, data, label_to_id)
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
                    save_qwen_photo(photo, data, label_to_id)
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
# MAIN PIPELINE
# ============================================================

def normalize_steps(payload: Dict[str, Any]) -> Dict[str, bool]:
    if payload.get("full_mode", False):
        return {
            "ingest": True,
            "compress": True,
            "face_index": True,
            "safe_people_reconcile": True,
            "rebuild_people": False,
            "qwen": True,
            "embeddings": True,
            "cleanup_temp": bool(payload.get("cleanup_temp", False)),
        }

    return payload.get("steps", {
        "ingest": True,
        "compress": False,
        "face_index": False,
        "safe_people_reconcile": False,
        "rebuild_people": False,
        "qwen": False,
        "embeddings": False,
        "cleanup_temp": False,
    })


def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_slug = payload["album_slug"]
    album_name = payload.get("album_name")
    events = payload["events"]

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

    results: Dict[str, Any] = {
        "album": album_ctx,
        "events": db_events,
        "steps": {},
    }

    if steps.get("ingest", True):
        update_job_status(job_id, "running", "ingest", "Scanning S3 and ingesting photos")
        results["steps"]["ingest"] = scan_and_ingest_originals(album_ctx, db_events)

        update_job_status(job_id, "running", "s3_validation", "Validating S3 source objects")
        results["steps"]["s3_validation"] = validate_s3_sources(album_ctx, db_events)

    if steps.get("compress", False):
        update_job_status(job_id, "running", "compress", "Generating AI input images")
        results["steps"]["compress"] = compress_events(album_ctx, db_events)

    if steps.get("face_index", False):
        update_job_status(job_id, "running", "face_index", "Running face detection")
        results["steps"]["face_index"] = face_index_events(album_ctx, db_events)

    if steps.get("safe_people_reconcile", False):
        update_job_status(
            job_id,
            "running",
            "safe_people_reconcile",
            "Assigning new faces without deleting existing people/names"
        )
        results["steps"]["safe_people_reconcile"] = safe_add_new_people_without_touching_existing_names(album_ctx)

    if steps.get("qwen", False):
        update_job_status(job_id, "running", "qwen", "Running Qwen metadata")
        results["steps"]["qwen"] = run_qwen_for_events(album_ctx, db_events)

    if steps.get("embeddings", False):
        update_job_status(job_id, "running", "embeddings", "Generating text embeddings")
        results["steps"]["embeddings"] = run_text_embeddings_for_events(album_ctx, db_events)

    if steps.get("cleanup_temp", False):
        update_job_status(job_id, "running", "cleanup_temp", "Deleting temporary AI folders")
        results["steps"]["cleanup_temp"] = cleanup_temp_s3(album_ctx, db_events)

    update_job_status(job_id, "running", "final_verify", "Verifying final counts")
    results["final_verify"] = final_verify(album_ctx, db_events)

    update_job_status(job_id, "completed", "done", "Pipeline completed")
    return results


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(event):
    started = time.time()

    job_id = event.get("id", "local_test")
    payload = event.get("input", {})

    try:
        print("Worker Start", flush=True)
        print("job_id:", job_id, flush=True)
        print("payload:", payload, flush=True)

        if "album_slug" not in payload:
            raise ValueError("Missing input.album_slug")

        if "events" not in payload or not payload["events"]:
            raise ValueError("Missing input.events")

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

        print("Worker failed:", err, flush=True)
        update_job_status(job_id, "failed", "error", repr(e), err)

        return {
            "ok": False,
            "job_id": job_id,
            "execution_seconds": round(time.time() - started, 2),
            **err,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})