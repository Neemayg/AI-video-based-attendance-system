# Technical Contract & Integration Specifications

This document defines the shared technical contract, data structures, and computer vision stack for **Phase 1 (Registration)**, **Phase 2 (Recognition)**, and **Phase 3 (Camera Integration)**.

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

## 5. Team Ownership Matrix

| Developer | Branch | Module Owned | Responsibilities | Does NOT Own |
|---|---|---|---|---|
| **Krish** | `feature/registration` | `src/registration/` | Student CLI, face sampling, embedding generation, saving `students.json` & `.npy` gallery files | Recognition matching decisions |
| **Swastik** | `feature/recognition` | `src/recognition/` | Embedding vector similarity scoring, thresholding, `recognize_face()` function | Camera/UI code |
| **Neemay Gupta** | `feature/camera-detection` | `src/detection/` & `src/app/` | Webcam feed capture, face box detection, HUD rendering, integrating `recognize_face()` | Recognition algorithm internals |
