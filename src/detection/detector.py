"""Single-face detection and alignment boundary for the camera runtime."""

from dataclasses import dataclass

import numpy as np
import torch

from src.registration.preprocessing import preprocess_face_with_box


@dataclass(frozen=True)
class DetectedFace:
    """A validated face box and its aligned MTCNN tensor."""

    box: np.ndarray
    face_tensor: torch.Tensor


def detect_face(frame: np.ndarray) -> DetectedFace | None:
    """Detect and align one face using the registration preprocessing pipeline."""
    result = preprocess_face_with_box(frame)
    if result is None:
        return None

    box, face_tensor = result
    return DetectedFace(box=box, face_tensor=face_tensor)


def initialize_detector() -> None:
    """Load the shared MTCNN model before processing camera frames."""
    from src.registration.preprocessing import get_mtcnn

    get_mtcnn()