"""Runnable one-person camera recognition application."""

import cv2
import numpy as np
from dataclasses import dataclass

from src.detection.detector import DetectedFace, detect_faces, initialize_detector
from src.detection.person import PersonTracker, associate_face_to_person, detect_people
from src.recognition.gallery import load_recognition_gallery
from src.recognition.matcher import recognize_face
from src.recognition.query import generate_query_embedding

from . import config


@dataclass
class _FaceTrack:
    """Internal face continuity state; uncertain tracks are never rendered."""

    detected: DetectedFace
    result: dict
    missed_frames: int = 0
    visible: bool = True


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


def _draw_person_track(frame: np.ndarray, track) -> None:
    """Draw a visible body track, labeling it only after face verification."""
    x1, y1, x2, y2 = [int(value) for value in track.box]
    color = (255, 255, 0)
    label = f"Person #{track.track_id}"
    if track.identity and track.identity.get("status") == "VERIFIED":
        label = f"Person #{track.track_id}: {track.identity['name']}"
        color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, label, (x1, max(25, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


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


def _box_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection-over-union for two finite face boxes."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if (
        left.shape != (4,)
        or right.shape != (4,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return 0.0
    intersection = np.maximum(
        0.0,
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
    )
    intersection_area = float(intersection[0] * intersection[1])
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection_area
    return intersection_area / union if union > 0 else 0.0


def _find_matching_track(box: np.ndarray, tracks: list[_FaceTrack]) -> _FaceTrack | None:
    """Find the nearest recent track for an occluded/error face detection."""
    candidates = [track for track in tracks if track.missed_frames <= config.TRACK_GRACE_FRAMES]
    if not candidates:
        return None
    best = max(candidates, key=lambda track: _box_overlap(box, track.detected.box))
    if _box_overlap(box, best.detected.box) > 0:
        return best

    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        return None
    center = (box[:2] + box[2:]) / 2
    distances = []
    for track in candidates:
        old_box = np.asarray(track.detected.box, dtype=float)
        if old_box.shape != (4,) or not np.all(np.isfinite(old_box)):
            continue
        old_center = (old_box[:2] + old_box[2:]) / 2
        size = max(old_box[2] - old_box[0], old_box[3] - old_box[1], 1.0)
        distances.append((float(np.linalg.norm(center - old_center)), size, track))
    if not distances:
        return None
    distance, size, track = min(distances, key=lambda item: item[0])
    return track if distance <= size * 1.5 else None


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
        tracks: list[_FaceTrack] = []
        person_tracker = PersonTracker()
        person_tracks = []
        previous_frame = None

        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError("Could not read a frame from the camera")

            should_detect = frame_counter % detection_cycle_frames == 0

            if should_detect:
                person_tracks = person_tracker.update(
                    detect_people(frame, config.PERSON_DETECTION_MIN_CONFIDENCE)
                )
                detected_list = detect_faces(frame)
                next_tracks = []
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
                    for track in tracks:
                        track.missed_frames += 1
                        track.visible = False
                        if track.missed_frames <= config.TRACK_GRACE_FRAMES:
                            next_tracks.append(track)
                else:
                    for detected in detected_list:
                        if detected.error_status is not None:
                            _draw_result(frame, detected, {})
                            previous_track = _find_matching_track(detected.box, tracks)
                            result = previous_track.result if previous_track else {}
                            next_tracks.append(_FaceTrack(detected, result))
                            continue

                        query_embedding = generate_query_embedding(detected.face_tensor)
                        result = recognize_face(
                            query_embedding,
                            gallery_embeddings,
                            gallery_metadata,
                            threshold,
                        )
                        person_id = associate_face_to_person(
                            detected.box, person_tracks
                        )
                        if person_id is not None and result.get("status") == "VERIFIED":
                            for person_track in person_tracks:
                                if person_track.track_id == person_id:
                                    person_track.identity = result
                                    break
                        _draw_result(frame, detected, result)
                        next_tracks.append(_FaceTrack(detected, result))
                tracks = next_tracks
                for person_track in person_tracks:
                    if person_track.visible:
                        _draw_person_track(frame, person_track)
            else:
                next_tracks = []
                for track in tracks:
                    if track.visible:
                        updated = _track_cached_faces(
                            previous_frame, frame, [(track.detected, track.result)]
                        )
                        if updated:
                            track.detected, track.result = updated[0]
                            track.missed_frames = 0
                            _draw_result(frame, track.detected, track.result)
                        else:
                            track.missed_frames += 1
                            track.visible = False
                    else:
                        track.missed_frames += 1
                    if track.missed_frames <= config.TRACK_GRACE_FRAMES:
                        next_tracks.append(track)
                tracks = next_tracks
                for person_track in person_tracks:
                    if person_track.visible:
                        _draw_person_track(frame, person_track)

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