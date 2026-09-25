import numpy as np
import pytest

import src.recognition.matcher as matcher
from src.recognition.matcher import recognize_face


def test_known_identity_above_threshold_is_verified():
    result = recognize_face(
        np.array([1.0, 0.0]),
        {"student-1": np.array([1.0, 0.0])},
        {"student-1": {"name": "Ada"}},
    )

    assert result == {
        "student_id": "student-1",
        "name": "Ada",
        "score": 1.0,
        "status": "VERIFIED",
    }


def test_match_below_threshold_is_unknown():
    result = recognize_face(
        np.array([1.0, 0.0]),
        {"student-1": np.array([0.0, 1.0])},
        {"student-1": {"name": "Ada"}},
        threshold=0.65,
    )

    assert result == {
        "student_id": None,
        "name": "Unknown",
        "score": 0.0,
        "status": "UNKNOWN",
    }


def test_empty_gallery_is_unknown():
    assert recognize_face(np.array([1.0, 0.0]), {}, {}) == {
        "student_id": None,
        "name": "Unknown",
        "score": 0.0,
        "status": "UNKNOWN",
    }


def test_highest_scoring_identity_is_selected():
    result = recognize_face(
        np.array([1.0, 0.0]),
        {
            "lower": np.array([0.8, 0.6]),
            "higher": np.array([1.0, 0.1]),
        },
        {
            "lower": {"name": "Lower"},
            "higher": {"name": "Higher"},
        },
    )

    assert result["student_id"] == "higher"
    assert result["name"] == "Higher"
    assert result["status"] == "VERIFIED"


def test_extremely_large_finite_embeddings_are_matched_stably():
    embedding = np.array([1e308, 1e308])

    result = recognize_face(
        embedding,
        {"student-1": embedding},
        {"student-1": {"name": "Ada"}},
    )

    assert result == {
        "student_id": "student-1",
        "name": "Ada",
        "score": 1.0,
        "status": "VERIFIED",
    }


def test_non_finite_similarity_score_is_rejected(monkeypatch):
    monkeypatch.setattr(matcher.np, "dot", lambda *_args: np.nan)

    with pytest.raises(ValueError, match="cosine similarity score must be finite"):
        matcher.recognize_face(
            np.array([1.0, 0.0]),
            {"student-1": np.array([1.0, 0.0])},
            {"student-1": {"name": "Ada"}},
        )


@pytest.mark.parametrize("metadata", [{}, {"student-1": {}}, {"student-1": None}])
def test_missing_metadata_or_name_uses_unknown(metadata):
    result = recognize_face(
        np.array([1.0, 0.0]),
        {"student-1": np.array([1.0, 0.0])},
        metadata,
    )

    assert result["name"] == "Unknown"


@pytest.mark.parametrize(
    "face_embedding, gallery_embeddings, message",
    [
        (np.array([0.0, 0.0]), {}, "face_embedding must have a non-zero norm"),
        (np.array([np.nan, 1.0]), {}, "face_embedding must contain only finite values"),
        (np.array([1.0, 0.0]), {"student-1": np.array([0.0, 0.0])}, "non-zero norm"),
        (np.array([1.0, 0.0]), {"student-1": np.array([np.inf, 1.0])}, "finite values"),
    ],
)
def test_invalid_embeddings_are_rejected(face_embedding, gallery_embeddings, message):
    with pytest.raises(ValueError, match=message):
        recognize_face(face_embedding, gallery_embeddings, {})


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="incompatible dimensions"):
        recognize_face(
            np.array([1.0, 0.0]),
            {"student-1": np.array([1.0, 0.0, 0.0])},
            {},
        )


@pytest.mark.parametrize("threshold", [-1.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        recognize_face(np.array([1.0, 0.0]), {}, {}, threshold=threshold)