"""Runnable one-person camera recognition application."""

import cv2
import numpy as np

from src.detection.detector import DetectedFace, detect_faces, initialize_detector
from src.recognition.gallery import load_recognition_gallery
from src.recognition.matcher import recognize_face
from src.recognition.query import generate_query_embedding

from . import config


def initialize_models() -> None:
    """Load detector and embedding models once before the frame loop."""
    initialize_detector()
    from src.registration.embedding import get_resnet

    get_resnet()


def _draw_result(frame: np.ndarray, detected: DetectedFace, result: dict) -> None:
    """Draw the detection box and recognition result on a camera frame."""
    x1, y1, x2, y2 = [int(value) for value in detected.box]
    
    if detected.error_status is not None:
        color = (0, 165, 255) # Orange for warning
        label = f"Skipped: {detected.error_status}"
    else:
        color = (0, 255, 0) if result["status"] == "VERIFIED" else (0, 0, 255)
        student_id = result["student_id"] or "-"
        label = (
            f"{result['name']} | ID: {student_id} | "
            f"Score: {result['score']:.2f} | {result['status']}"
        )
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def run(
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.RECOGNITION_THRESHOLD,
) -> None:
    """Run the camera loop until the user presses ``q``."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")
    capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    try:
        initialize_models()
        gallery_embeddings, gallery_metadata = load_recognition_gallery()

        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            detected_list = detect_faces(frame)
            if not detected_list:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                for detected in detected_list:
                    if detected.error_status is not None:
                        # Skip embedding and matching
                        _draw_result(frame, detected, {})
                        continue
                        
                    query_embedding = generate_query_embedding(detected.face_tensor)
                    result = recognize_face(
                        query_embedding,
                        gallery_embeddings,
                        gallery_metadata,
                        threshold,
                    )
                    _draw_result(frame, detected, result)

            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()