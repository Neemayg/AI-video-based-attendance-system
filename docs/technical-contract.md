# Technical Contract & Integration Specifications

This document defines the shared technical contract, data structures, and computer vision stack across **Phase 1-4 (Vision)** and **Phase 5-7 (Attendance Domain Layer)**.

---

## 1. Selected Model & Embedding Specifications

To ensure compatibility across developer modules:

* **Face Embedding Model**: Standard normalized 512-dimensional embedding vector (or 128-D depending on PyTorch/FaceNet/dlib model selected).
* **Vector Type**: `numpy.ndarray` with shape `(512,)` (or `(128,)`) and `dtype=np.float32`.
* **Normalization**: The recognition matcher accepts finite, non-zero embedding vectors and computes cosine similarity internally as the dot product divided by the product of the vector norms. This includes normalization in the cosine calculation.
* **Distance Metric**: **Cosine Similarity** (dot product of L2-normalized vectors), yielding a score range of `[-1.0, 1.0]`.
* **Recognition Threshold**: Thresholds MUST be finite values in the range `[-1.0, 1.0]`.

---

## 2. Storage Data Format

All registered student metadata and embeddings are saved in `data/embeddings/`:

### A. Student Metadata (`data/embeddings/students.json`)
```json
{
  "11024010004": {
    "student_id": "11024010004",
    "name": "Neemay Gupta",
    "created_at": "2026-09-26T00:00:00"
  },
  "11024010010": {
    "student_id": "11024010010",
    "name": "Krish",
    "created_at": "2026-09-26T00:00:00"
  }
}
```

### B. Gallery Embeddings (`data/embeddings/<student_id>.npy`)
Each registered student has a corresponding NumPy array file containing their mean face embedding vector of shape `(512,)`.

---

## 3. Recognition API Contract (`src/recognition/`)

The recognition module MUST expose the following interface signature:

```python
def recognize_face(
    face_embedding: np.ndarray,
    gallery_embeddings: dict[str, np.ndarray],
    gallery_metadata: dict[str, dict],
    threshold: float = 0.65
) -> dict:
    """
    Compares a detected face embedding against the registered gallery.

    Returns:
        dict: {
            "student_id": "11024010004" (or None),
            "name": "Neemay Gupta" (or "Unknown"),
            "score": 0.87, # Cosine similarity score [-1.0 - 1.0]
            "status": "VERIFIED" # "VERIFIED" if score >= threshold else "UNKNOWN"
        }
    """
```

---

## 4. Camera HUD Display Contract (`src/app/`)

The live camera view will overlay bounding boxes and identity labels formatted as follows:

```text
┌─────────────────────────────┐
│                             │
│        FACE BOUNDING BOX    │
│                             │
└─────────────────────────────┘
  Neemay Gupta
  ID: 11024010004
  Match Score: 0.87
  Status: VERIFIED
```

For unknown individuals:
```text
┌─────────────────────────────┐
│                             │
│        FACE BOUNDING BOX    │
│                             │
└─────────────────────────────┘
  Unknown Person
  Match Score: 0.41
  Status: UNKNOWN
```

---

## 5. Attendance Domain Contract (`src/attendance/`)

The attendance engine is strictly partitioned into two logical layers: **Presence** and **Policy**.

### A. Presence Session Engine (`src/attendance/presence.py`)
Responsible for grouping timestamped face and body tracking observations into logical blocks of time. It issues `EntryEvent` and `ExitEvent` objects containing fully assembled `PresenceSession`s representing the exact duration a student was physically inside the camera's monitored zone.

### B. Attendance Policy Engine (`src/attendance/policy.py`)
A pure mathematical domain layer. It takes a closed `PresenceSession` and evaluates it against a `ScheduledPeriod` (from a `Timetable`).
- **Overlap Logic:** Sessions are clipped to the exact bounds of the scheduled period. Multiple sessions in the same period are mathematically merged to prevent double-counting of overlap.
- **Late Threshold:** Assessed solely against the student's *first* entry time relative to the period start.
- **Output:** Produces an `AttendanceRecord` with a deterministic status (`PRESENT`, `PARTIAL`, `ABSENT`, `LATE`) and an explainable english reason string.

---

## 6. Team Ownership Matrix

| Developer | Branch | Module Owned | Responsibilities | Does NOT Own |
|---|---|---|---|---|
| **Krish** | `feature/registration` | `src/registration/` | Student CLI, face sampling, embedding generation, saving `students.json` & `.npy` gallery files | Recognition matching decisions |
| **Swastik** | `feature/recognition` | `src/recognition/` | Embedding vector similarity scoring, thresholding, `recognize_face()` function | Camera/UI code |
| **Neemay Gupta** | `feature/camera-detection` | `src/detection/` & `src/app/` | Webcam feed capture, face box detection, HUD rendering, integrating `recognize_face()` | Recognition algorithm internals |
