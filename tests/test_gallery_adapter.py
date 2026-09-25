import json

import numpy as np
import pytest

from src.recognition.gallery import load_recognition_gallery, recognize_persisted_face


@pytest.fixture
def isolated_gallery(tmp_path, monkeypatch):
    from src.registration import config

    gallery_dir = tmp_path / "embeddings"
    gallery_dir.mkdir()
    monkeypatch.setattr(config, "STORAGE_BASE_DIR", str(gallery_dir))
    monkeypatch.setattr(config, "STORAGE_GALLERY_JSON", str(gallery_dir / "gallery.json"))
    return gallery_dir


def write_gallery(gallery_dir, records):
    (gallery_dir / "gallery.json").write_text(json.dumps(records))


def test_loads_centroids_and_metadata(isolated_gallery):
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    student_dir = isolated_gallery / "101"
    student_dir.mkdir()
    np.save(student_dir / "centroid.npy", centroid)
    write_gallery(
        isolated_gallery,
        {
            "101": {
                "name": "Ada",
                "centroid_file": "101/centroid.npy",
            }
        },
    )

    embeddings, metadata = load_recognition_gallery()

    np.testing.assert_array_equal(embeddings["101"], centroid)
    assert metadata == {"101": {"name": "Ada"}}


def test_known_identity_is_verified_from_persisted_gallery(isolated_gallery):
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    student_dir = isolated_gallery / "101"
    student_dir.mkdir()
    np.save(student_dir / "centroid.npy", centroid)
    write_gallery(
        isolated_gallery,
        {"101": {"name": "Ada", "centroid_file": "101/centroid.npy"}},
    )

    result = recognize_persisted_face(centroid)

    assert result == {
        "student_id": "101",
        "name": "Ada",
        "score": 1.0,
        "status": "VERIFIED",
    }


def test_unknown_identity_and_threshold_are_returned_correctly(isolated_gallery):
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    student_dir = isolated_gallery / "101"
    student_dir.mkdir()
    np.save(student_dir / "centroid.npy", centroid)
    write_gallery(
        isolated_gallery,
        {"101": {"name": "Ada", "centroid_file": "101/centroid.npy"}},
    )

    result = recognize_persisted_face(-centroid, threshold=0.0)

    assert result == {
        "student_id": None,
        "name": "Unknown",
        "score": -1.0,
        "status": "UNKNOWN",
    }


def test_empty_gallery_is_explicitly_supported(isolated_gallery):
    write_gallery(isolated_gallery, {})

    embeddings, metadata = load_recognition_gallery()

    assert embeddings == {}
    assert metadata == {}


@pytest.mark.parametrize(
    "record, error_match",
    [
        ({"name": "Ada"}, "centroid_file"),
        ({"centroid_file": "101/centroid.npy"}, "invalid name"),
        ({"name": "Ada", "centroid_file": "missing.npy"}, "not found"),
    ],
)
def test_invalid_gallery_records_raise_clear_errors(
    isolated_gallery, record, error_match
):
    write_gallery(isolated_gallery, {"101": record})

    with pytest.raises((ValueError, FileNotFoundError), match=error_match):
        load_recognition_gallery()