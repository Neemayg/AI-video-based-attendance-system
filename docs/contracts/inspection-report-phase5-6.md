# Phase 5 & 6 Initial Inspection Report

## A. Current runtime flow
The application (`src/app/main.py`) continuously reads frames from `cv2.VideoCapture`. For performance, it skips detection on most frames using a `DETECTION_CYCLE_FRAMES` interval.
On a detection frame:
1. `detect_faces(frame)` finds all faces.
2. Valid faces are batched through `generate_embeddings()`.
3. Each embedding is matched against the gallery via `recognize_face()`.
4. The resulting `DetectedFace` and recognition `dict` are stored in a `cached_detected` list.
On skipped frames, it simply re-draws the results from `cached_detected`. 

## B. Exact structure of one recognition result
`recognize_face()` returns a flat dictionary:
```python
{
    "student_id": "11024010004",  # or None
    "name": "Nimai Gupta",        # or "Unknown"
    "score": 0.92,                # float, rounded to 2 decimals
    "status": "VERIFIED"          # or "UNKNOWN"
}
```

## C. How multiple recognized faces are currently represented
Multiple faces are represented as a list of independent results in a frame. Specifically, the camera loop generates a list of `(DetectedFace, dict)` tuples (the `cached_detected` list). There is no "global" tracking ID tying a bounding box from Frame 1 to Frame 2. Identity is established purely by the recognition result dictionary.

## D. Where timestamps are currently generated
Currently, there are **no timestamps generated anywhere** in the camera loop. The application processes frames as fast as `cv2` and the GPU allow, with no concept of wall-clock time or video-stream time.

## E. What module should own the new presence/session logic
The user correctly dictated creating a new, isolated domain layer:
```
src/attendance/
    __init__.py
    presence.py
    events.py
    session.py
    config.py
```
This module will be independent of the UI and recognition modules.

## F. Any current limitation that will affect entry/exit
1. **Detection Cycles:** Since the camera caches the recognition results for `N` frames, we will need to decide whether to update the presence engine on *every* camera frame using the cached identities, or only on the frames where active detection happens. Updating only on active detection frames is safer so we don't hallucinate a student who moved out of frame during the cached cycle.
2. **Lack of Timestamps:** We must inject `datetime.now()` (or a time provider) at the top of the camera loop to feed into the Presence engine.
3. **Ghosting/Blinking:** As noted, MTCNN sometimes drops a face for a frame or two. The mandated "grace period" logic in the presence engine will perfectly solve this limitation without needing a complex Kalman filter tracker.
