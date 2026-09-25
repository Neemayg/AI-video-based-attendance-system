import numpy as np
import pytest

from src.app import main
from src.detection.detector import DetectedFace


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
    monkeypatch.setattr(main, "detect_face", lambda current_frame: detected)
    monkeypatch.setattr(main, "generate_query_embedding", lambda tensor: np.ones(512))
    monkeypatch.setattr(main, "recognize_face", lambda *args: unknown)

    main.run()

    assert capture.released
    assert capture.settings == [(fake_cv2.CAP_PROP_FPS, 60)]
    assert fake_cv2.destroyed
    assert any("Unknown" in label and "UNKNOWN" in label for label in fake_cv2.labels)


def test_camera_releases_resources_on_processing_error(monkeypatch):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    capture = FakeCapture(frames=[frame])
    fake_cv2 = FakeCV2(capture)
    monkeypatch.setattr(main, "cv2", fake_cv2)
    monkeypatch.setattr(main, "initialize_models", lambda: None)
    monkeypatch.setattr(main, "load_recognition_gallery", lambda: ({}, {}))
    monkeypatch.setattr(main, "detect_face", lambda frame: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        main.run()

    assert capture.released
    assert fake_cv2.destroyed