"""Runnable one-person camera recognition application."""

import cv2
import numpy as np

from src.detection.detector import DetectedFace, detect_faces, initialize_detector
from src.detection.person_detector import detect_persons, get_person_detector
from src.recognition.gallery import load_recognition_gallery
from src.recognition.matcher import recognize_face
from src.registration.embedding import generate_embeddings

from . import config
from datetime import datetime, timedelta
from src.attendance.config import GRACE_PERIOD_SECONDS
from src.attendance.presence import PresenceSessionEngine
from src.attendance.events import Observation, EntryEvent, ExitEvent
from src.attendance.policy import AttendancePolicyEngine, Timetable, ScheduledPeriod
from src.detection.tracker import CentroidTracker


def initialize_models() -> None:
    """Load detector and embedding models once before the frame loop."""
    initialize_detector()
    get_person_detector()
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
            f"{result['name']} | {result['status']}"
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

def _draw_person_box(frame: np.ndarray, box: np.ndarray, track_id: str):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        frame,
        f"Track: {track_id}",
        (x1, max(25, y1 - 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
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
        cached_persons = [] # List of PersonDetection
        cached_faces = [] # List of (DetectedFace, result, track_id_str)
        cached_active_tracks = {} # track_id -> TrackedPerson

        session_engine = PresenceSessionEngine()
        tracker = CentroidTracker(max_disappeared_seconds=GRACE_PERIOD_SECONDS)
        recent_exits = []  # Store (ExitEvent, expiration_time)

        # Phase 7 Policy Engine setup (Prototype Mock Timetable)
        today = datetime.now()
        mock_period = ScheduledPeriod(
            period_id="mock_p1",
            subject_name="Computer Vision 101",
            room_id="201",
            start_time=today.replace(hour=0, minute=0, second=0, microsecond=0),
            end_time=today.replace(hour=23, minute=59, second=59, microsecond=0)
        )
        timetable = Timetable(periods=[mock_period])
        policy_engine = AttendancePolicyEngine()

        while True:
            current_time = datetime.now()
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            should_detect = (frame_counter % DETECTION_CYCLE_FRAMES == 0)

            if should_detect:
                # 1. Person Detection & Tracking
                person_list = detect_persons(frame)
                cached_persons = person_list
                assigned_track_ids, cached_active_tracks = tracker.update(person_list, current_time)
                
                # 2. Face Detection & Recognition
                detected_list = detect_faces(frame)
                cached_faces = []
                
                if detected_list:
                    valid_faces = [d for d in detected_list if d.error_status is None]
                    if valid_faces:
                        batch_embeddings = generate_embeddings([d.face_tensor for d in valid_faces])
                    
                    valid_face_idx = 0
                    for face_idx, detected in enumerate(detected_list):
                        # Associate face with track
                        fx1, fy1, fx2, fy2 = detected.box
                        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                        
                        associated_track_id = None
                        for p_idx, person in enumerate(person_list):
                            px1, py1, px2, py2 = person.box
                            if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                                associated_track_id = assigned_track_ids[p_idx]
                                break
                        
                        if detected.error_status is not None:
                            cached_faces.append((detected, {}, associated_track_id))
                            continue

                        query_embedding = batch_embeddings[valid_face_idx]
                        valid_face_idx += 1
                        
                        result = recognize_face(
                            query_embedding,
                            gallery_embeddings,
                            gallery_metadata,
                            threshold,
                        )
                        cached_faces.append((detected, result, associated_track_id))
                        
            # --- Rendering ---
            # Draw tracks
            for tid, track in cached_active_tracks.items():
                _draw_person_box(frame, track.box, tid)
                
            # Draw faces
            for detected, result, track_id in cached_faces:
                _draw_result(frame, detected, result)

            # --- Presence Engine Integration ---
            observations = []
            
            # Map of track_id to best observation in this frame
            track_observations = {}
            
            # Create observations for valid recognition frames
            for detected, result, track_id in cached_faces:
                if track_id is not None and result and result.get("status") == "VERIFIED":
                    track_observations[track_id] = Observation(
                        student_id=result["student_id"],
                        name=result["name"],
                        score=result["score"],
                        status=result["status"],
                        timestamp=current_time,
                        track_id=track_id,
                        source="camera",
                        camera_id=str(camera_index),
                        room_id="201" # TODO: Load from config
                    )
            
            # For tracks that are active but not verified
            for tid in cached_active_tracks.keys():
                if tid not in track_observations:
                    track_observations[tid] = Observation(
                        student_id=None,
                        name="Unknown",
                        score=0.0,
                        status="TRACKED",
                        timestamp=current_time,
                        track_id=tid,
                        source="camera",
                        camera_id=str(camera_index),
                        room_id="201"
                    )
                    
            observations = list(track_observations.values())

            events = session_engine.process_observations(observations, current_time)
            
            # Print events to terminal
            for event in events:
                if isinstance(event, EntryEvent):
                    print(f"ENTRY: student_id={event.student_id} timestamp={event.timestamp}")
                elif isinstance(event, ExitEvent):
                    print(f"EXIT: student_id={event.student_id} timestamp={event.timestamp} duration_seconds={event.duration_seconds}")
                    recent_exits.append((event, current_time + timedelta(seconds=5)))
                    
                    # Phase 7: Evaluate closed session against policy
                    if event.closed_session:
                        resolved_period = policy_engine.resolve_period(event.closed_session, timetable)
                        if resolved_period:
                            record = policy_engine.evaluate(
                                student_id=event.student_id,
                                name=event.name,
                                period=resolved_period,
                                sessions=[event.closed_session]
                            )
                            print("\n=== ATTENDANCE RECORD (Phase 7) ===")
                            print(f"Student: {record.name}")
                            print(f"Subject: {record.subject}")
                            print(f"Qualifying Presence: {record.qualifying_presence_seconds // 60}m {record.qualifying_presence_seconds % 60}s")
                            print(f"Status: {record.status}")
                            print(f"Reason: {record.reason}")
                            print("===================================\n")

            # Clean up old exit events from overlay
            recent_exits = [(e, exp) for (e, exp) in recent_exits if exp > current_time]

            # --- Render Attendance HUD ---
            hud_y = 30
            for student_id, session in session_engine.active_sessions.items():
                duration = int((current_time - session.entry_time).total_seconds())
                m, s = divmod(duration, 60)
                lines = [
                    session.name,
                    f"ID: {session.student_id}",
                    "ACTIVE",
                    f"Seen: {session.entry_time.strftime('%H:%M:%S')}",
                    f"Current duration: {m}m {s:02d}s"
                ]
                for line in lines:
                    cv2.putText(frame, line, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    hud_y += 25
                hud_y += 15

            for exit_event, _ in recent_exits:
                m, s = divmod(exit_event.duration_seconds, 60)
                lines = [
                    exit_event.name,
                    "EXIT",
                    exit_event.timestamp.strftime('%H:%M:%S'),
                    f"Duration: {m}m {s:02d}s"
                ]
                for line in lines:
                    cv2.putText(frame, line, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    hud_y += 25
                hud_y += 15

            frame_counter += 1
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()