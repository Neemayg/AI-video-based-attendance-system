# Presence Engine Contract

## A. Observation input
An observation represents a single frame's output from the recognition engine.
- `student_id`: Unique identifier for the student.
- `name`: Display name.
- `score`: Confidence score of the recognition.
- `status`: Recognition status (must be `VERIFIED` to affect presence).
- `timestamp`: The datetime the frame was processed.

## B. Session state
A session represents a continuous period of presence for a recognized student.
- `session_id`: UUID for the session.
- `student_id`: Student identifier.
- `name`: Student name.
- `entry_time`: The timestamp of the first observation.
- `last_seen_time`: The timestamp of the most recent observation.
- `exit_time`: The timestamp when the session was closed.
- `duration_seconds`: Total time in seconds between entry and exit.
- `status`: `ACTIVE` or `CLOSED`.

## C. Entry event
Triggered exactly once when a student is first verified and a new session opens. Contains `student_id`, `name`, `timestamp` (entry time), and `score`.

## D. Exit event
Triggered exactly once when a session is closed. Contains `student_id`, `name`, `timestamp` (exit time), and `duration_seconds`.

## E. Duration
Calculated as the difference in seconds between `exit_time` and `entry_time`.

## F. Grace-period behavior
Configured via `ATTENDANCE_GRACE_PERIOD_SECONDS` (default: 3). If a student is not observed for this duration, their active session is closed. The exit time is recorded as the time the grace period evaluation resolved.

## G. Multi-student behavior
Each student maintains an independent session state keyed by their `student_id`. A missed detection for Student A does not affect the active session for Student B.

## H. Re-entry behavior
If a student's session closes (e.g., they leave the camera view for longer than the grace period) and they return later, a completely new `PresenceSession` and `EntryEvent` are generated.
