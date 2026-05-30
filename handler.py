import os
import re
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
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

DELETE_TEMP_AI_INPUT = os.environ.get("DELETE_TEMP_AI_INPUT", "false").lower() == "true"
DELETE_TEMP_ANNOTATED = os.environ.get("DELETE_TEMP_ANNOTATED", "false").lower() == "true"

# Safe people matching thresholds.
# Existing people are preserved. New people are added only for unmatched new face clusters.
PEOPLE_MATCH_EXISTING_SIM_THRESHOLD = float(os.environ.get("PEOPLE_MATCH_EXISTING_SIM_THRESHOLD", "0.58"))
NEW_FACE_CLUSTER_SIM_THRESHOLD = float(os.environ.get("NEW_FACE_CLUSTER_SIM_THRESHOLD", "0.62"))

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".nef", ".cr2", ".arw", ".dng", ".tif", ".tiff"
}

UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_.+",
    re.IGNORECASE,
)

s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)


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


# ============================================================
# LOGGING
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
# ALBUM / EVENT
# ============================================================

def restore_album_context(album_slug: str) -> Dict[str, Any]:
    album = db_one("""
        SELECT *
        FROM albums
        WHERE slug = %s
        LIMIT 1;
    """, (album_slug,))

    if not album:
        raise RuntimeError(
            f"Album not found for slug={album_slug}. "
            "Create the album first before running serverless AI."
        )

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
        generated_objects = [o for o in image_objects if is_generated_original_key(o["Key"])]
        usable_objects = [o for o in image_objects if not is_generated_original_key(o["Key"])]

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
# SAFE PEOPLE RECONCILIATION
# This never deletes existing people.
# This preserves manually edited display_name/default_name.
# It only assigns faces with person_id IS NULL.
# If unmatched, it creates new people.
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

    # Faces not yet assigned to any person.
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

    # Cluster remaining unlabeled faces so one new person can get multiple similar faces.
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
                # Assign confident matches to existing people.
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

                # Next person number.
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

                # Refresh counts for all people without touching names.
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

    return {
        "unlabeled_faces": len(face_rows),
        "assigned_to_existing_people": len(assigned_to_existing),
        "new_people_created": new_people_created,
        "existing_people_untouched": True,
        "names_preserved": True,
    }


def rebuild_photo_people_base_safe(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuilds photo_people from faces/person_id.
    Does not delete people. Does not change people display names.
    """
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


# ============================================================
# HEAVY STEPS — BLOCKED UNTIL REAL CODE IS PORTED
# ============================================================

def compress_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    raise NotImplementedError(
        "compress_events is not implemented yet. "
        "Set steps.compress=false until compression code is ported."
    )


def face_index_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    raise NotImplementedError(
        "face_index_events is not implemented yet. "
        "Set steps.face_index=false until InsightFace code is ported."
    )


def run_qwen_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    raise NotImplementedError(
        "run_qwen_for_events is not implemented yet. "
        "Set steps.qwen=false until Qwen code is ported."
    )


def run_text_embeddings_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    raise NotImplementedError(
        "run_text_embeddings_for_events is not implemented yet. "
        "Set steps.embeddings=false until embedding code is ported."
    )


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
            COUNT(*) FILTER (WHERE p.qwen_status = 'skipped_no_labeled_faces') AS qwen_skipped
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
            COUNT(*) FILTER (WHERE f.person_id IS NOT NULL) AS labeled_faces,
            COUNT(*) FILTER (WHERE f.person_id IS NULL) AS unlabeled_faces
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
        }
    }

    print("Final verify:", result, flush=True)
    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_slug = payload["album_slug"]
    events = payload["events"]

    steps = payload.get("steps", {
        "ingest": True,
        "compress": False,
        "face_index": False,
        "safe_people_reconcile": False,
        "rebuild_people": False,
        "qwen": False,
        "embeddings": False,
        "cleanup_temp": False,
    })

    if steps.get("rebuild_people", False):
        raise RuntimeError(
            "Blocked: destructive rebuild_people is not allowed. "
            "This protects manually renamed people. "
            "Use safe_people_reconcile=true after face indexing instead."
        )

    update_job_status(job_id, "running", "restore_album", "Restoring album context")
    album_ctx = restore_album_context(album_slug)

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