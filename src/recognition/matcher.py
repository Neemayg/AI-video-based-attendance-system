"""Cosine-similarity face matching against a registered embedding gallery."""

import numpy as np


def _validate_embedding(embedding: np.ndarray, label: str) -> np.ndarray:
    """Return a one-dimensional finite, non-zero embedding."""
    vector = np.asarray(embedding)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional embedding")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")

    scale = np.max(np.abs(vector))
    scaled_norm = np.linalg.norm(vector / scale) if scale else 0.0
    if scale == 0 or not np.isfinite(scaled_norm):
        raise ValueError(f"{label} must have a non-zero norm")
    return vector


def _normalize_embedding(embedding: np.ndarray, label: str) -> np.ndarray:
    """Normalize an embedding without overflowing its norm calculation."""
    vector = _validate_embedding(embedding, label)
    scale = np.max(np.abs(vector))
    scaled_vector = vector / scale
    return scaled_vector / np.linalg.norm(scaled_vector)


def recognize_face(
    face_embedding: np.ndarray,
    gallery_embeddings: dict[str, np.ndarray],
    gallery_metadata: dict[str, dict],
    threshold: float = 0.65,
) -> dict:
    """Match a face embedding and classify the highest-scoring gallery entry."""
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between -1 and 1")

    query = _validate_embedding(face_embedding, "face_embedding")
    if not gallery_embeddings:
        return {
            "student_id": None,
            "name": "Unknown",
            "score": 0.0,
            "status": "UNKNOWN",
        }

    normalized_query = _normalize_embedding(query, "face_embedding")
    best_student_id = None
    best_score = -np.inf

    for student_id, gallery_embedding in sorted(gallery_embeddings.items()):
        gallery_vector = _validate_embedding(
            gallery_embedding, f"gallery embedding for {student_id}"
        )
        if gallery_vector.shape != query.shape:
            raise ValueError(
                f"gallery embedding for {student_id} has incompatible dimensions"
            )

        normalized_gallery = _normalize_embedding(
            gallery_vector, f"gallery embedding for {student_id}"
        )
        score = float(np.dot(normalized_query, normalized_gallery))
        if not np.isfinite(score):
            raise ValueError("cosine similarity score must be finite")
        if score > best_score:
            best_student_id = student_id
            best_score = score

    rounded_score = round(best_score, 2)
    if best_score >= threshold:
        metadata = gallery_metadata.get(best_student_id, {})
        name = metadata.get("name", "Unknown") if isinstance(metadata, dict) else "Unknown"
        return {
            "student_id": best_student_id,
            "name": name or "Unknown",
            "score": rounded_score,
            "status": "VERIFIED",
        }

    return {
        "student_id": None,
        "name": "Unknown",
        "score": rounded_score,
        "status": "UNKNOWN",
    }