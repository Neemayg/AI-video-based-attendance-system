"""Face detection and alignment boundary for the camera runtime."""

import os
from dataclasses import dataclass

import numpy as np
import torch

from src.registration.preprocessing import preprocess_face_with_box, extract_faces

MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES", "3"))


@dataclass(frozen=True)
class DetectedFace:
    """A validated face box and its aligned MTCNN tensor or error status."""

    box: np.ndarray
    face_tensor: torch.Tensor | None = None
    error_status: str | None = None


def detect_face(frame: np.ndarray) -> DetectedFace | None:
    """Detect and align one face using the registration preprocessing pipeline."""
    result = preprocess_face_with_box(frame)
    if result is None:
        return None

    box, face_tensor = result
    return DetectedFace(box=box, face_tensor=face_tensor)


def detect_faces(frame: np.ndarray) -> list[DetectedFace]:
    """Detect all faces in the frame and preserve error statuses for unviable constraints."""
    faces = extract_faces(frame)
    # --- Performance: Limit the number of faces to process per frame ---
    # On CPU, processing too many faces can cause severe latency; this ensures a predictable load.
    if len(faces) > MAX_FACES_PER_FRAME:
        faces = faces[:MAX_FACES_PER_FRAME]
    # --- End Cap ---
    detected = []
    for box, tensor, error_status in faces:
        detected.append(DetectedFace(box=box, face_tensor=tensor, error_status=error_status))
    return detected


def initialize_detector() -> None:
    """Load the shared MTCNN model before processing camera frames."""
    from src.registration.preprocessing import get_mtcnn

    get_mtcnn()