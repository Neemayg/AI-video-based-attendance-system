# Identity Continuity and Tracking Contract (Phase 6.5)

## A. Detection Input
The camera loop continuously fetches bounding boxes for detected persons using `torchvision` (`ssdlite320_mobilenet_v3_large`). These person-level bounding boxes are fed into the `CentroidTracker` every detection cycle to maintain continuous physical body tracking, independent of face visibility.

## B. Track ID
The `CentroidTracker` assigns a stable `track_id` to each continuous bounding box object over time. This track ID persists even if the face detection model fails to extract a valid face, as long as the object remains relatively stationary or moves continuously.

## C. Identity Association
When a track has a valid, high-confidence face recognition event (`status == "VERIFIED"`) associated with it (calculated by checking if the MTCNN face box lies inside the Person bounding box), the `PresenceSessionEngine` binds the `student_id` to that `track_id` in memory (`track_to_identity`). 
This guarantees that temporary weak recognitions or face drops do not lose the student's identity, as the bounding box of the body remains tracked.

## D. Temporary Face Loss
**Definition:** A track exists, but the face recognition model cannot identify the person (e.g., they turned around, or the frame is noisy).
**Behavior:** The tracker continues to emit the `track_id`. The camera loop emits an `Observation(status="TRACKED", track_id=...)`. The `PresenceSessionEngine` resolves the `track_id` to the persistent `student_id` and keeps the session `ACTIVE`.
No `ExitEvent` is triggered.

## E. Track Loss
**Definition:** The tracker can no longer find a bounding box that matches the track (e.g., the student completely left the camera view).
**Behavior:** The tracker increments its missing counter for that track. Once the missing duration exceeds `max_disappeared_seconds` (which is configured to `ATTENDANCE_GRACE_PERIOD_SECONDS`), the tracker deletes the track.
The `PresenceSessionEngine` will then notice the student is missing and close the session, firing an `ExitEvent`.

## F. Identity Persistence & Safety
One weak recognition (`UNKNOWN`) on a frame does *not* switch the identity of a track. If an observation is `UNKNOWN` but the `track_id` is bound to a verified `student_id`, the engine prefers the verified persistent identity.

## G. Timeout Configuration
Controlled by `ATTENDANCE_GRACE_PERIOD_SECONDS` in `src/attendance/config.py` (defaults to 3 seconds). This now dictates the tracker's `max_disappeared_seconds` timeout.

## H. Re-entry Behavior
If a track is fully lost and times out, the `track_id` is destroyed. The `PresenceSessionEngine` removes the `track_id` from its map and closes the session.
If the student re-enters, the tracker assigns a *new* `track_id`, and once verified, a *new* `EntryEvent` and `PresenceSession` are created.
