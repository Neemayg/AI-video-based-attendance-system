import numpy as np
import torch

from src.detection import detector
from src.registration import preprocessing


def test_detection_returns_box_and_aligned_tensor(monkeypatch):
    box = np.array([1.0, 2.0, 30.0, 40.0])
    tensor = torch.zeros(3, 160, 160)
    monkeypatch.setattr(detector, "preprocess_face_with_box", lambda frame: (box, tensor))

    result = detector.detect_face(np.zeros((50, 50, 3), dtype=np.uint8))

    assert result is not None
    np.testing.assert_array_equal(result.box, box)
    assert result.face_tensor is tensor


def test_detection_returns_none_for_rejected_face(monkeypatch):
    monkeypatch.setattr(detector, "preprocess_face_with_box", lambda frame: None)

    assert detector.detect_face(np.zeros((50, 50, 3), dtype=np.uint8)) is None


def test_detect_faces_preserves_error_status(monkeypatch):
    box1 = np.array([1.0, 2.0, 30.0, 40.0])
    box2 = np.array([40.0, 40.0, 60.0, 60.0])
    tensor1 = torch.zeros(3, 160, 160)
    
    def fake_extract_faces(frame):
        return [
            (box1, tensor1, None),
            (box2, None, "FACE_TOO_SMALL")
        ]
        
    monkeypatch.setattr(detector, "extract_faces", fake_extract_faces)
    
    results = detector.detect_faces(np.zeros((50, 50, 3), dtype=np.uint8))
    assert len(results) == 2
    assert results[0].error_status is None
    np.testing.assert_array_equal(results[0].box, box1)
    
    assert results[1].error_status == "FACE_TOO_SMALL"
    assert results[1].face_tensor is None
    np.testing.assert_array_equal(results[1].box, box2)

def test_side_pose_is_rejected_and_frontal_pose_is_allowed():
    frontal_landmarks = np.array(
        [[10.0, 20.0], [30.0, 20.0], [20.0, 30.0], [14.0, 40.0], [26.0, 40.0]]
    )
    side_landmarks = np.array(
        [[10.0, 20.0], [30.0, 20.0], [27.0, 30.0], [15.0, 40.0], [25.0, 40.0]]
    )

    assert preprocessing.validate_frontal_pose(frontal_landmarks)
    assert not preprocessing.validate_frontal_pose(side_landmarks)