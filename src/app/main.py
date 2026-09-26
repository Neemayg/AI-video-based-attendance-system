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


def _track_cached_faces(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    cached_detected: list[tuple[DetectedFace, dict]],
) -> list[tuple[DetectedFace, dict]]:
    """Update cached face boxes with optical flow and discard lost tracks."""
    if (
        not cached_detected
        or previous_frame is None
        or current_frame is None
        or current_frame.size == 0
    ):
        return []

    try:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    except (AttributeError, cv2.error, ValueError):
        return []

    tracked = []
    height, width = current_frame.shape[:2]
    for detected, result in cached_detected:
        box = np.asarray(detected.box, dtype=float)
        if (
            box.shape != (4,)
            or not np.all(np.isfinite(box))
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            continue
        x1, y1, x2, y2 = box
        try:
            points = np.array(
                [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32
            ).reshape(-1, 1, 2)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(3, 10, 0.03),
            )
        except (cv2.error, ValueError):
            continue

        if next_points is None or status is None:
            continue
        try:
            next_points = np.asarray(next_points, dtype=float).reshape(-1, 2)
            status = np.asarray(status).reshape(-1).astype(bool)
        except (TypeError, ValueError):
            continue
        if (
            next_points.shape != (4, 2)
            or status.shape != (4,)
            or not np.all(np.isfinite(next_points))
            or status.sum() < 2
        ):
            continue

        displacement = np.median(next_points[status] - points.reshape(-1, 2)[status], axis=0)
        if not np.all(np.isfinite(displacement)):
            continue
        updated_box = np.array(
            [x1 + displacement[0], y1 + displacement[1],
             x2 + displacement[0], y2 + displacement[1]],
            dtype=np.float32,
        )
        updated_box[[0, 2]] = np.clip(updated_box[[0, 2]], 0, width - 1)
        updated_box[[1, 3]] = np.clip(updated_box[[1, 3]], 0, height - 1)
        if updated_box[2] <= updated_box[0] or updated_box[3] <= updated_box[1]:
            continue
        tracked.append(
            (
                DetectedFace(updated_box, detected.face_tensor, detected.error_status),
                result,
            )
        )
    return tracked


def run(
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.RECOGNITION_THRESHOLD,
) -> None:
    """Run the camera loop until the user presses ``q``."""
    detection_cycle_frames = config.validate_detection_cycle_frames()
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
        # the previous frame's recognition results are reused while optical
        # flow updates their boxes. A full detection cycle re-establishes the
        # independent face tracks and recognition results.
        frame_counter = 0
        cached_detected = []  # List of (DetectedFace, recognized_result_or_None)
        previous_frame = None

        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError("Could not read a frame from the camera")

            should_detect = frame_counter % detection_cycle_frames == 0

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
                    for detected in detected_list:
                        if detected.error_status is not None:
                            _draw_result(frame, detected, {})
                            cached_detected.append((detected, {}))
                            continue

                        query_embedding = generate_query_embedding(detected.face_tensor)
                        result = recognize_face(
                            query_embedding,
                            gallery_embeddings,
                            gallery_metadata,
                            threshold,
                        )
                        _draw_result(frame, detected, result)
                        cached_detected.append((detected, result))
            else:
                cached_detected = _track_cached_faces(
                    previous_frame, frame, cached_detected
                )
                for detected, result in cached_detected:
                    _draw_result(frame, detected, result)

            frame_counter += 1
            previous_frame = frame.copy()
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()