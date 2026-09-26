from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PresenceSession:
    session_id: str
    student_id: str
    name: str
    track_id: Optional[str]
    entry_time: datetime
    last_seen_time: datetime
    exit_time: Optional[datetime]
    duration_seconds: Optional[int]
    status: str  # "ACTIVE" or "CLOSED"
    camera_id: Optional[str] = None
    room_id: Optional[str] = None
