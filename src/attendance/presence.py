import uuid
from datetime import datetime
from typing import List, Dict, Union

from . import config
from .events import Observation, EntryEvent, ExitEvent
from .session import PresenceSession


class PresenceSessionEngine:
    def __init__(self, grace_period_seconds: int = config.GRACE_PERIOD_SECONDS):
        self.grace_period_seconds = grace_period_seconds
        self.active_sessions: Dict[str, PresenceSession] = {}
        self.track_to_identity: Dict[str, dict] = {}

    def process_observations(
        self, observations: List[Observation], current_time: datetime
    ) -> List[Union[EntryEvent, ExitEvent]]:
        """
        Processes a list of recognized observations, updates active presence sessions,
        and returns any triggered Entry or Exit events.
        """
        events: List[Union[EntryEvent, ExitEvent]] = []
        observed_student_ids = set()

        for obs in observations:
            # 1. Identity association
            if obs.track_id is not None:
                # If we have a reliable verification, bind it to the track
                if obs.status == "VERIFIED" and obs.student_id:
                    self.track_to_identity[obs.track_id] = {
                        "student_id": obs.student_id,
                        "name": obs.name
                    }

            # Resolve identity
            student_id = None
            name = None

            if obs.status == "VERIFIED" and obs.student_id:
                student_id = obs.student_id
                name = obs.name
            elif obs.track_id is not None and obs.track_id in self.track_to_identity:
                # Use persistent identity from tracking for temporary face loss / weak rec
                identity = self.track_to_identity[obs.track_id]
                student_id = identity["student_id"]
                name = identity["name"]

            if not student_id:
                continue

            observed_student_ids.add(student_id)

            # Update existing session or open new session
            if student_id in self.active_sessions:
                session = self.active_sessions[student_id]
                session.last_seen_time = current_time
                if obs.track_id is not None:
                    session.track_id = obs.track_id
            else:
                session = PresenceSession(
                    session_id=str(uuid.uuid4()),
                    student_id=student_id,
                    name=name,
                    track_id=obs.track_id,
                    entry_time=current_time,
                    last_seen_time=current_time,
                    exit_time=None,
                    duration_seconds=None,
                    status="ACTIVE",
                    camera_id=obs.camera_id,
                    room_id=obs.room_id,
                )
                self.active_sessions[student_id] = session

                # For EntryEvent score, fallback to 0.0 if not directly provided
                events.append(
                    EntryEvent(
                        student_id=student_id,
                        name=name,
                        timestamp=current_time,
                        score=obs.score if obs.status == "VERIFIED" else 0.0,
                        track_id=obs.track_id,
                        source=obs.source,
                        camera_id=obs.camera_id,
                        room_id=obs.room_id,
                    )
                )

        # 2. Check for expired sessions due to TRACK LOSS (absence beyond grace period)
        expired_ids = []
        for student_id, session in self.active_sessions.items():
            if student_id not in observed_student_ids:
                time_since_last_seen = (
                    current_time - session.last_seen_time
                ).total_seconds()
                
                if time_since_last_seen >= self.grace_period_seconds:
                    expired_ids.append(student_id)

        # 3. Close expired sessions
        for student_id in expired_ids:
            session = self.active_sessions.pop(student_id)
            session.exit_time = current_time
            session.duration_seconds = int(
                (session.exit_time - session.entry_time).total_seconds()
            )
            session.status = "CLOSED"
            
            # Clean up tracking association if it exists
            tracks_to_remove = [tid for tid, ident in self.track_to_identity.items() 
                                if ident["student_id"] == student_id]
            for tid in tracks_to_remove:
                del self.track_to_identity[tid]

            events.append(
                ExitEvent(
                    student_id=session.student_id,
                    name=session.name,
                    timestamp=session.exit_time,
                    duration_seconds=session.duration_seconds,
                    track_id=session.track_id,
                    camera_id=session.camera_id,
                    room_id=session.room_id,
                    closed_session=session
                )
            )

        return events
