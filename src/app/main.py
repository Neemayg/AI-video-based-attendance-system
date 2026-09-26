"""Runnable one-person camera recognition application."""

import cv2
import numpy as np

from src.detection.detector import DetectedFace, detect_faces, initialize_detector
from src.recognition.gallery import load_recognition_gallery
from src.recognition.matcher import recognize_face
from src.registration.embedding import generate_embeddings

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

        # --- Performance: Detection cycle caching ---
        # Running MTCNN face detection every frame is very slow on CPU.
        # The DETECTION_CYCLE runs detection every Nth frame. In between,
        # the previous frame's face positions are reused to generate
        # embeddings/match. Faces are assumed not to move significantly
        # between cached frames (no multi-frame tracking is applied).
        from src.app.config import DETECTION_CYCLE_FRAMES

        frame_counter = 0
        cached_detected = []  # List of (DetectedFace, recognized_result_or_None)

        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            should_detect = (frame_counter % DETECTION_CYCLE_FRAMES == 0)

            if should_detect:
                detected_list = detect_faces(frame)
                cached_detected = []
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
                    valid_faces = [d for d in detected_list if d.error_status is None]
                    if valid_faces:
                        batch_embeddings = generate_embeddings([d.face_tensor for d in valid_faces])
                    
                    valid_face_idx = 0
                    for detected in detected_list:
                        if detected.error_status is not None:
                            _draw_result(frame, detected, {})
                            cached_detected.append((detected, {}))
                            continue

                        query_embedding = batch_embeddings[valid_face_idx]
                        valid_face_idx += 1
                        
                        result = recognize_face(
                            query_embedding,
                            gallery_embeddings,
                            gallery_metadata,
                            threshold,
                        )
                        _draw_result(frame, detected, result)
                        cached_detected.append((detected, result))
            else:
                # --- Reuse previous recognition results ---
                for detected, result in cached_detected:
                    _draw_result(frame, detected, result)

            frame_counter += 1
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()