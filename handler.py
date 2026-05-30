import os
import time
import traceback
from typing import Any, Dict, List

import runpod


# ============================================================
# ENV CONFIG
# ============================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "nikhith-ai-photo-gallery-dev")

# DB env vars should come from Runpod endpoint secrets/env
RDS_HOST = os.environ.get("RDS_HOST")
RDS_PORT = int(os.environ.get("RDS_PORT", "5432"))
RDS_DB = os.environ.get("RDS_DB")
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

# Optional behavior flags
DELETE_TEMP_AI_INPUT = os.environ.get("DELETE_TEMP_AI_INPUT", "true").lower() == "true"
DELETE_TEMP_ANNOTATED = os.environ.get("DELETE_TEMP_ANNOTATED", "true").lower() == "true"


# ============================================================
# JOB STATUS HELPERS
# Replace these with your real DB status table if you have one.
# ============================================================

def update_job_status(job_id: str, status: str, step: str, message: str = "", extra: Dict[str, Any] | None = None):
    """
    In production, update your processing_jobs / ai_jobs table here.
    For now, this logs progress so Runpod logs show the current step.
    """
    payload = {
        "job_id": job_id,
        "status": status,
        "step": step,
        "message": message,
        "extra": extra or {},
    }
    print("[JOB_STATUS]", payload, flush=True)


# ============================================================
# PIPELINE STUBS
# Replace internals with your notebook-tested code.
# ============================================================

def restore_album_context(album_slug: str) -> Dict[str, Any]:
    """
    Replace with:
      album = db_one("SELECT * FROM albums WHERE slug=%s LIMIT 1", ...)
      album_id = album["id"]
    """
    print(f"Restoring album context for {album_slug}")
    return {
        "album_slug": album_slug,
        "album_id": "REPLACE_WITH_DB_ALBUM_ID",
    }


def upsert_events(album_ctx: Dict[str, Any], events: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Replace with your Cell 2 logic:
      INSERT INTO album_events ... ON CONFLICT(album_id, slug) DO UPDATE
    """
    print("Upserting/restoring events:")
    for e in events:
        print(e)

    return [
        {
            **e,
            "event_id": f"REPLACE_WITH_DB_EVENT_ID_{i}",
        }
        for i, e in enumerate(events)
    ]


def scan_and_ingest_originals(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with:
      - list S3 originals
      - skip UUID-prefixed generated copies
      - ingest_from_s3_prefixes()
      - create photos rows
    """
    print("Scanning S3 originals and ingesting DB photo rows")
    return {
        "ingested": True,
        "events": [e["slug"] for e in events],
    }


def validate_s3_sources(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with your head_object safety check.
    Soft-delete DB rows whose original_s3_key/source_s3_key is missing.
    This avoids the HeadObject 404 surprise.
    """
    print("Validating S3 source keys with head_object")
    return {
        "missing_sources_soft_deleted": 0,
    }


def compress_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with run_compression_jobs, event-scoped.
    Current desired S3 behavior:
      - read originals/
      - write temporary ai-input/
      - do NOT write clean-preview/thumbnail/watermark if you don't need them
    """
    print("Running compression / ai-input generation")
    return {
        "compressed": True,
    }


def face_index_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with:
      - load InsightFace buffalo_l using CUDAExecutionProvider
      - run_face_index_jobs event-scoped
      - insert faces rows
    """
    print("Running InsightFace face indexing on GPU")
    return {
        "face_indexed": True,
    }


def rebuild_people_for_album(album_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace with:
      cluster_people_for_album()
      assign_unlabeled_faces_to_new_people()
      crop_and_upload_missing_person_covers()
      rebuild_photo_people_base()
      build_duplicate_candidates()
    """
    print("Rebuilding people/photo_people for full album")
    return {
        "people_rebuilt": True,
    }


def run_qwen_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with your event-scoped Qwen code:
      - patch annotated keys only for these events
      - mark no-labeled-face photos skipped only for these events
      - annotate ai-input images with Person boxes
      - run Qwen2.5-VL
      - save qwen_json/search_text
    """
    print("Running Qwen only for target events")
    return {
        "qwen_completed": True,
    }


def run_text_embeddings_for_events(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with your Cell 22 text embedding logic, event-scoped.
    """
    print("Running text embeddings for target events")
    return {
        "embeddings_completed": True,
    }


def cleanup_temp_s3(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Delete temporary prefixes after AI finishes:
      albums/{albumSlug}/events/{eventSlug}/ai-input/
      albums/{albumSlug}/events/{eventSlug}/annotated/

    Keep:
      originals/
      albums/{albumSlug}/faces/{personId}/cover.jpg
    """
    deleted = []

    for e in events:
        album_slug = album_ctx["album_slug"]
        event_slug = e["slug"]

        if DELETE_TEMP_AI_INPUT:
            prefix = f"albums/{album_slug}/events/{event_slug}/ai-input/"
            print("Would delete temp prefix:", prefix)
            deleted.append(prefix)

        if DELETE_TEMP_ANNOTATED:
            prefix = f"albums/{album_slug}/events/{event_slug}/annotated/"
            print("Would delete temp prefix:", prefix)
            deleted.append(prefix)

    # Implement actual S3 delete_objects here once you are ready.
    return {
        "temp_prefixes_deleted": deleted,
    }


def final_verify(album_ctx: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace with final DB counts:
      photos, compressed, face_indexed, qwen_completed, qwen_pending,
      qwen_failed, qwen_skipped, people_count, missing_covers, embeddings.
    """
    print("Running final verification")
    return {
        "verified": True,
        "events": [e["slug"] for e in events],
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_album_events(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    album_slug = payload["album_slug"]
    events = payload["events"]

    steps = payload.get("steps", {
        "ingest": True,
        "compress": True,
        "face_index": True,
        "rebuild_people": True,
        "qwen": True,
        "embeddings": True,
        "cleanup_temp": True,
    })

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

    if steps.get("compress", True):
        update_job_status(job_id, "running", "compress", "Generating AI input images")
        results["steps"]["compress"] = compress_events(album_ctx, db_events)

    if steps.get("face_index", True):
        update_job_status(job_id, "running", "face_index", "Running face detection and embeddings")
        results["steps"]["face_index"] = face_index_events(album_ctx, db_events)

    if steps.get("rebuild_people", True):
        update_job_status(job_id, "running", "rebuild_people", "Rebuilding people and photo_people")
        results["steps"]["rebuild_people"] = rebuild_people_for_album(album_ctx)

    if steps.get("qwen", True):
        update_job_status(job_id, "running", "qwen", "Running Qwen metadata")
        results["steps"]["qwen"] = run_qwen_for_events(album_ctx, db_events)

    if steps.get("embeddings", True):
        update_job_status(job_id, "running", "embeddings", "Generating text embeddings")
        results["steps"]["embeddings"] = run_text_embeddings_for_events(album_ctx, db_events)

    if steps.get("cleanup_temp", True):
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
    """
    Runpod sends:
    {
      "input": {
        "album_slug": "...",
        "events": [...]
      }
    }
    """
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
