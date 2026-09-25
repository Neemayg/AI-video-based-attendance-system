"""Adapt the persisted registration gallery for the recognition matcher."""

from pathlib import Path

import numpy as np

from .matcher import recognize_face


def load_recognition_gallery() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Load centroid embeddings and names from the registration gallery."""
    from src.registration import config
    from src.registration.storage import load_gallery

    gallery = load_gallery()
    if not isinstance(gallery, dict):
        raise ValueError("registration gallery must be a dictionary")
    if not gallery:
        return {}, {}

    base_dir = Path(config.STORAGE_BASE_DIR).resolve()
    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}

    for student_id, record in gallery.items():
        if not isinstance(student_id, str) or not student_id:
            raise ValueError("gallery student IDs must be non-empty strings")
        if not isinstance(record, dict):
            raise ValueError(f"gallery record for {student_id} must be a dictionary")

        name = record.get("name")
        centroid_file = record.get("centroid_file")
        if not isinstance(name, str) or not name:
            raise ValueError(f"gallery record for {student_id} has an invalid name")
        if not isinstance(centroid_file, str) or not centroid_file:
            raise ValueError(
                f"gallery record for {student_id} has no valid centroid_file"
            )

        centroid_path = (base_dir / centroid_file).resolve()
        try:
            centroid_path.relative_to(base_dir)
        except ValueError as error:
            raise ValueError(
                f"centroid_file for {student_id} must be inside the gallery directory"
            ) from error

        try:
            centroid = np.load(centroid_path, allow_pickle=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"centroid file for {student_id} was not found: {centroid_path}"
            ) from error
        except (OSError, ValueError) as error:
            raise ValueError(
                f"centroid file for {student_id} could not be loaded: {centroid_path}"
            ) from error

        if not isinstance(centroid, np.ndarray) or centroid.shape != (512,):
            raise ValueError(
                f"centroid file for {student_id} must contain a 512-D vector"
            )

        embeddings[student_id] = centroid
        metadata[student_id] = {"name": name}

    return embeddings, metadata


def recognize_persisted_face(
    face_embedding: np.ndarray,
    threshold: float = 0.65,
) -> dict:
    """Recognize a query embedding against the read-only persisted gallery."""
    embeddings, metadata = load_recognition_gallery()
    return recognize_face(face_embedding, embeddings, metadata, threshold)