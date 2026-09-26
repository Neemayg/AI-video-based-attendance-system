import numpy as np
import pytest

from src.app import main
from src.detection.detector import DetectedFace
from src.detection.person_detector import PersonDetection


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True

    def release(self):
        self.released = True


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FPS = 5

    def __init__(self, capture):
        self.capture = capture
        self.destroyed = False
        self.labels = []

    def VideoCapture(self, index):
        return self.capture

    def rectangle(self, *args):
        pass

    def putText(self, frame, text, *args):
        self.labels.append(text)

    def imshow(self, window, frame):
        pass

    def waitKey(self, delay):
        return ord("q")

    def destroyAllWindows(self):
        self.destroyed = True


def test_camera_initialization_failure_is_reported(monkeypatch):
    capture = FakeCapture(opened=False)
    fake_cv2 = FakeCV2(capture)
    monkeypatch.setattr(main, "cv2", fake_cv2)

    with pytest.raises(RuntimeError, match="Could not open camera"):
        main.run()

    assert capture.released


def test_camera_successfully_releases_resources_and_displays_unknown(monkeypatch):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    capture = FakeCapture(frames=[frame])
    fake_cv2 = FakeCV2(capture)
    detected = DetectedFace(np.array([1, 1, 20, 20]), object())
    unknown = {
        "student_id": None,
        "name": "Unknown",
        "score": 0.2,
        "status": "UNKNOWN",
    }
    monkeypatch.setattr(main, "cv2", fake_cv2)
    monkeypatch.setattr(main, "initialize_models", lambda: None)
    monkeypatch.setattr(main, "load_recognition_gallery", lambda: ({}, {}))
    person_box = PersonDetection(np.array([0, 0, 30, 30]), 0.9)
    monkeypatch.setattr(main, "detect_persons", lambda frame: [person_box])
    monkeypatch.setattr(main, "detect_faces", lambda current_frame: [detected])
    monkeypatch.setattr(main, "generate_embeddings", lambda tensors: np.ones((len(tensors), 512)))
    monkeypatch.setattr(main, "recognize_face", lambda *args: unknown)

    main.run()

    assert capture.released
    assert capture.settings == [(fake_cv2.CAP_PROP_FPS, 60)]
    assert fake_cv2.destroyed
    assert any("Unknown" in label and "UNKNOWN" in label for label in fake_cv2.labels)


def test_camera_supports_multiple_independent_faces(monkeypatch):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    capture = FakeCapture(frames=[frame])
    fake_cv2 = FakeCV2(capture)
    
    # Face 1: valid, verified
    detected1 = DetectedFace(np.array([1, 1, 20, 20]), object())
    verified_result = {
        "student_id": "100",
        "name": "Alice",
        "score": 0.9,
        "status": "VERIFIED",
    }
    
    # Face 2: valid, unknown
    detected2 = DetectedFace(np.array([30, 30, 50, 50]), object())
    unknown_result = {
        "student_id": None,
        "name": "Unknown",
        "score": 0.2,
        "status": "UNKNOWN",
    }
    
    # Face 3: invalid, too small
    detected3 = DetectedFace(np.array([60, 60, 65, 65]), None, "FACE_TOO_SMALL")
    
    def fake_recognize_face(query_embedding, *args, **kwargs):
        if query_embedding is detected1.face_tensor: # match by identity using mock hack
            return verified_result
        return unknown_result

    monkeypatch.setattr(main, "cv2", fake_cv2)
    monkeypatch.setattr(main, "initialize_models", lambda: None)
    monkeypatch.setattr(main, "load_recognition_gallery", lambda: ({}, {}))
    
    person1 = PersonDetection(np.array([0, 0, 30, 30]), 0.9)
    person2 = PersonDetection(np.array([25, 25, 55, 55]), 0.9)
    person3 = PersonDetection(np.array([50, 50, 70, 70]), 0.9)
    monkeypatch.setattr(main, "detect_persons", lambda frame: [person1, person2, person3])
    
    monkeypatch.setattr(main, "detect_faces", lambda current_frame: [detected1, detected2, detected3])
    monkeypatch.setattr(main, "generate_embeddings", lambda tensors: tensors)
    monkeypatch.setattr(main, "recognize_face", fake_recognize_face)

    main.run()

    # Three labels should be drawn independently
    assert any("Alice" in label and "VERIFIED" in label for label in fake_cv2.labels)
    assert any("Unknown" in label and "UNKNOWN" in label for label in fake_cv2.labels)
    assert any("Skipped: FACE_TOO_SMALL" in label for label in fake_cv2.labels)


def test_camera_releases_resources_on_processing_error(monkeypatch):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    capture = FakeCapture(frames=[frame])
    fake_cv2 = FakeCV2(capture)
    monkeypatch.setattr(main, "cv2", fake_cv2)
    monkeypatch.setattr(main, "initialize_models", lambda: None)
    monkeypatch.setattr(main, "load_recognition_gallery", lambda: ({}, {}))
    monkeypatch.setattr(main, "detect_persons", lambda frame: [])
    monkeypatch.setattr(main, "detect_faces", lambda frame: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        main.run()

    assert capture.released
    assert fake_cv2.destroyed