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


def test_side_pose_is_rejected_and_frontal_pose_is_allowed():
    frontal_landmarks = np.array(
        [[10.0, 20.0], [30.0, 20.0], [20.0, 30.0], [14.0, 40.0], [26.0, 40.0]]
    )
    side_landmarks = np.array(
        [[10.0, 20.0], [30.0, 20.0], [27.0, 30.0], [15.0, 40.0], [25.0, 40.0]]
    )

    assert preprocessing.validate_frontal_pose(frontal_landmarks)
    assert not preprocessing.validate_frontal_pose(side_landmarks)