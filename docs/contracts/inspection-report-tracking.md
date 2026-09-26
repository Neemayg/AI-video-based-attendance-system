# Tracking Inspection Report

1. **Current Detection Output**:
   `detect_faces(frame)` returns a list of `DetectedFace` objects. Each contains a bounding `box`, an optional `face_tensor`, and an `error_status`. There is no tracking ID.

2. **Current Multi-Face Batch Inference**:
   Faces without errors are collected and passed in batch to `generate_embeddings()`. The returned embeddings map 1:1 to the valid faces.

3. **Recognition Result Structure**:
   `recognize_face()` returns a dict with `student_id`, `name`, `score`, and `status` ("VERIFIED" or "UNKNOWN"). This happens inside `main.py` and is currently zipped back with the `DetectedFace` and cached.

4. **Presence Module (`src/attendance/`)**:
   Currently, it consumes `Observation` objects, which just have the `student_id` and `timestamp`. It keys active sessions purely by `student_id`.

5. **Where Tracking Can Be Inserted**:
   Tracking can be inserted inside `src/app/main.py` directly after `detect_faces(frame)`. 
   The tracker will consume the bounding boxes from `DetectedFace` and assign a `track_id` to each. 
   The `cached_detected` list will be modified to store `(DetectedFace, track_id, result)`.
   
   To avoid adding heavy dependencies like OpenCV-contrib trackers or PyTorch ByteTrack (which can be hard to install on Windows), I will implement a lightweight, dependency-free IoU or Centroid-based tracker in `src/detection/tracker.py`. 
   
   The `PresenceSessionEngine` will then be updated to key active sessions by `track_id` (or maintain a `track_id` mapping) so that if recognition yields "UNKNOWN" or face drops, the session remains tied to the `track_id` until the `track_id` times out.
