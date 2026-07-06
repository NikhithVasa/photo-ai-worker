# Photo AI Worker

This repository contains the image-text side of the SaathiDesk media pipeline. It is normally invoked after the face worker has ingested images, generated previews, detected faces, reconciled people, and enqueued the image-text job.

## More Docs

- [Developer Guide](docs/developer-guide.md)
- [Agent Notes](docs/agent-notes.md)

## Two Worker Handlers - Responsibilities

## 1. Face Worker `handler.py`

**Purpose:** prepares photos, detects faces, creates people, and generates face cover images.

**Main steps:**

```text
ingest → compress → face_index → safe_people_reconcile → crop_person_covers → enqueue image-text worker
```

**Writes/updates:**

```text
photos
faces
people
photo_people
people.cover_face_s3_key
person_merge_candidates
```

**Use this worker for:**

```text
Missing face covers
People creation
Face detection
Face embeddings
Person grouping
photo_people rows
```

**Expected logs:**

```text
Face Worker Start
safe_people_reconcile
crop_person_covers
Cover crop result
enqueue_qwen
Face worker completed
```

---

## 2. Image-Text / Gemma-Qwen Worker `handler.py`

**Purpose:** runs AI descriptions/search after faces and people already exist.

**Main steps:**

```text
qwen/Gemma metadata → text embeddings → optional culling/image embeddings
```

**Writes/updates:**

```text
photos.caption
photos.ai_description
photos.search_text
photos.qwen_json
photos.qwen_status
photos.search_embedding
photo_people.qwen_description
photo_people.qwen_json
photo_people.search_text
photo_people.search_embedding
photo_culling_scores
photo_image_embeddings
best_photo_collections
```

**Use this worker for:**

```text
Gemma vs Qwen model selection
Captions
Search metadata
Text embeddings
AI culling / best photos
```

**Expected Gemma logs:**

```text
Qwen Worker Start
Image-to-text provider=gemma
Loading Gemma model
Gemma loaded
Qwen result
Embedding result
Qwen worker completed
```

> Note: names like `Qwen Worker Start`, `qwen_status`, and `qwen_json` are legacy names. The actual model is confirmed by `Image-to-text provider=gemma`.

---

## Simple Mental Model

```text
Face worker      = find people and create face covers
Image-text worker = describe photos and make them searchable
```

## Correct Pipeline

```text
Lambda
  ↓
Face worker
  - ingest
  - compress
  - face_index
  - safe_people_reconcile
  - crop_person_covers
  ↓
Image-text worker
  - Gemma/Qwen captions
  - text embeddings
```

## Debug Mapping

```text
Missing face cover images       → Face worker
People count / grouping issue   → Face worker
Gemma vs Qwen running           → Image-text worker
qwen_completed = 0              → Image-text worker, often caused by Face worker timing
AI captions/search not working  → Image-text worker
```
