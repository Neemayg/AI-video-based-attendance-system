import numpy as np
import torch
from unittest.mock import MagicMock, patch
from src.detection.detector import detect_faces, MAX_FACES_PER_FRAME
from src.app.main import run
import src.app.main as main

print(f"MAX_FACES_PER_FRAME config: {MAX_FACES_PER_FRAME}")

# Test 1: Face cap works
boxes = [np.array([float(i*10), float(i*10), float(i*10+20), float(i*10+20)]) for i in range(10)]
def fake_extract(many=True):
    return [(boxes[i], torch.zeros(3, 160, 160), None) for i in range(10)]

with patch("src.detection.detector.extract_faces", side_effect=fake_extract):
    results = detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))
    print(f"Test 1 (Face cap): {len(results)} faces processed (capped at {MAX_FACES_PER_FRAME}). PASS={len(results) == MAX_FACES_PER_FRAME}")

# Test 2: Frame-skipping logic in app
frames_processed = []
def fake_capture():
    cap = MagicMock()
    cap.isOpened = lambda: True
    def make_frames():
        for i in range(7):  # 7 frames, cycle=3 -> detection at frames 0, 3, 6
            frames_processed.append(i)
            yield True, np.zeros((480, 640, 3), dtype=np.uint8)
        while True:
            yield False, None
    gen = make_frames()
    cap.read = lambda: next(gen)
    cap.set = lambda *a: True
    cap.release = MagicMock()
    cap.wait = lambda: ord("q")
    return cap

fake_cv2 = MagicMock()
fake_cv2.CAP_PROP_FPS = 5
fake_cv2.waitKey = lambda d: ord("q")

class FakeResult:
    def __init__(self):
        self.labels = []
        self.destroyed = False
    def __getattr__(self, name):
        # Capture calls like putText, rectangle, imshow
        def method(*args, **kwargs):
            if name == "putText":
                self.labels.append(args[1])
            elif name == "destroyAllWindows":
                self.destroyed = True
        return method

fr = FakeResult()
fr.FONT_HERSHEY_SIMPLEX = 0

with patch.multiple(main, cv2=fake_cv2, initialize_models=lambda: None,
                     load_recognition_gallery=lambda: ({}, {}),
                     detect_faces=lambda f: [], generate_query_embedding=lambda t: None,
                     recognize_face=lambda *a, **k: {}):
    with patch("cv2.VideoCapture", side_effect=fake_capture):
        fake_cv2.destroyAllWindows = fr.destroyAllWindows
        try:
            run()
        except Exception as e:
            print(f"Test 2 (App lifecycle): Exception {e}")
        print(f"Test 2 (Frame skipping): Frames processed = {len(frames_processed)} (detection at 0, 3, 6). PASS={len(frames_processed) == 3}")
