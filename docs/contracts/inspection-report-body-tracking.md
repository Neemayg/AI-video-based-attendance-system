# Inspection Report: Body-Level Identity Continuity (Phase 6.5 enhancement)

1. **What object currently represents a detected face?**
   `DetectedFace` (in `src/detection/detector.py`). It contains `box` (numpy array), `face_tensor` (aligned face for ResNet), and `error_status`.

2. **What object currently represents a recognition result?**
   A standard Python `dict` returned by `recognize_face` in `src/recognition/matcher.py`. It contains `student_id`, `name`, `score`, and `status`.

3. **What object currently represents a track?**
   There is no explicit `Track` object. The `CentroidTracker` maintains a private dict of tracks, and `tracker.update()` returns a tuple of `(assigned_track_ids, active_tracks)`. The `assigned_track_ids` map 1:1 to the input boxes, and `active_tracks` is a dict of `{track_id: string_track_id}` representing all living tracks.

4. **Whether current detection is face-only or person-level.**
   Currently, detection is purely **face-only**, powered by MTCNN. If a person turns around, MTCNN yields zero boxes, and the person conceptually disappears from the pipeline until they turn back.

5. **Where the current CentroidTracker is called.**
   It is called in `src/app/main.py` directly on the `cached_detected` list. This means it tracks the *face* bounding boxes, not the person bounding boxes.

6. **How track IDs are currently propagated into Observation.**
   In the `main.py` camera loop, `tracker.update(face_boxes)` returns `assigned_track_ids`. These are zipped with the face recognition results to populate the `track_id` field in the `Observation` dataclass. The `Observation` is then sent to the `PresenceSessionEngine`.

7. **Exactly where the current system decides “face disappeared.”**
   MTCNN internally fails to find a face when turned away. The `CentroidTracker` increments `time_missing` for that track. Once `time_missing` exceeds `max_disappeared_seconds` (driven by `GRACE_PERIOD_SECONDS`), the tracker drops the track, and `main.py` stops emitting `TRACKED` observations. The `PresenceSessionEngine` then applies its own timeout and fires an `ExitEvent`.

8. **Whether a suitable person detector already exists in the repository/dependencies.**
   Yes, `torchvision` is available in `requirements.txt`. It ships with several pre-trained object detectors on COCO. `torchvision.models.detection.ssdlite320_mobilenet_v3_large` is highly suitable as a lightweight person detector for CPU/laptop prototypes. It detects full bodies (COCO class 1).

9. **Whether adding a new detector will conflict with the current MTCNN pipeline.**
   Running a person detector + MTCNN sequentially could drop FPS, but we already have `DETECTION_CYCLE_FRAMES` in `main.py` to only run detection periodically. We can run the person detector and MTCNN inside this cycle, keeping performance high. They do not conflict architecturally.
