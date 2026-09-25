# AI Video Attendance System Context

## MVP Scope

The MVP supports one person in the camera frame:

```text
Camera frame
-> single-face detection and alignment
-> 160x160 face tensor
-> 512-D query embedding
-> persisted centroid gallery
-> cosine similarity and threshold decision
-> HUD identity result
```

Do not add multi-person tracking, attendance duration, entry/exit logic, databases, dashboards, anti-spoofing, cross-camera recognition, or LLM features in this phase.

## Module Ownership

- Registration owns student input, face capture, preprocessing, embedding generation, and gallery persistence.
- Detection owns the runtime camera-to-face boundary and returns one bounding box plus one aligned face tensor.
- Recognition owns query embedding consumption, gallery adaptation, cosine similarity, thresholding, and VERIFIED/UNKNOWN classification.
- Application owns the webcam loop, HUD rendering, cleanup, and runtime orchestration.

## Registration Contract

Registration entry point:

```bash
.venv/bin/python -m src.registration.cli
```

The CLI asks for `Student ID:` and `Student Name:`. Existing IDs offer `[A] Cancel` or `[B] Re-register / replace`. Manual capture accepts `SPACE` or `ENTER`; `Q` cancels. Ten valid single-face samples are required.

The registration pipeline uses:

- MTCNN face detection and alignment
- BGR camera frames converted to RGB
- 160x160 aligned tensors
- InceptionResnetV1 pretrained on `vggface2`
- 512-dimensional float32 L2-normalized embeddings
- A normalized per-student centroid
- Frontal-pose and face-quality validation

Gallery files are written under `data/embeddings/`:

```text
data/embeddings/gallery.json
data/embeddings/<student_id>/embeddings.npy
data/embeddings/<student_id>/centroid.npy
```

The loader is:

```python
from src.registration.storage import load_gallery
gallery = load_gallery()
```

It returns metadata and relative file paths. Recognition loads `centroid.npy`, not the complete sample matrix.

`BLUR_THRESHOLD` defaults to `50.0` and can be configured for a camera with:

```bash
BLUR_THRESHOLD=5
```

The current local camera required `BLUR_THRESHOLD=5` during manual registration because its measured face-crop blur variance was approximately 8-12.

## Recognition Contract

Public matcher API:

```python
recognize_face(
    face_embedding,
    gallery_embeddings,
    gallery_metadata,
    threshold=0.65,
) -> dict
```

The result contains:

```python
{
    "student_id": "<id>" or None,
    "name": "<name>" or "Unknown",
    "score": <raw cosine similarity rounded for display>,
    "status": "VERIFIED" or "UNKNOWN",
}
```

Cosine similarity is a score in `[-1.0, 1.0]`, not a probability. Thresholds are finite values in `[-1.0, 1.0]`. The matcher validates finite, non-zero, compatible embeddings and uses numerically stable normalization.

`src/recognition/gallery.py` adapts the persisted registration gallery. `src/recognition/query.py` accepts an already aligned face tensor and does not run detection again.

## Camera Runtime

Launch recognition with:

```bash
BLUR_THRESHOLD=5 RECOGNITION_THRESHOLD=0.65 .venv/bin/python -m src.app
```

The app loads models and the gallery once at startup, processes frames, draws a bounding box and result HUD, and releases the camera and OpenCV windows on exit or error. Press `q` to exit.

A real local gallery currently exists for student ID `100` with name `swastik`. Its samples are shape `(10, 512)` and its centroid is shape `(512,)`; the values are float32, finite, and L2-normalized. The gallery loader successfully reads it.

## Tests and Environment

The project uses the local `.venv` environment. Required packages include NumPy, OpenCV, pytest, Torch, TorchVision, and facenet-pytorch. Model initialization succeeds locally.

The latest full suite result was:

```text
37 passed
```

When model downloads need certificate configuration, use the local certifi bundle:

```bash
SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')
```

## Git State

Work is on `feature/recognition`. Existing changes are intentionally uncommitted and must be preserved. Do not reset, clean, commit, push, or merge without explicit instruction.

The live camera app was launched locally, but the terminal does not report HUD contents. Do not claim a real identity was recognized unless the camera display visibly confirms the expected name, ID, score, and status.
