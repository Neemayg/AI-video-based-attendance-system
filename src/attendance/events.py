from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import PresenceSession

@dataclass(frozen=True)
class Observation:
    student_id: str
    name: str
    score: float
    status: str
    timestamp: datetime
    track_id: Optional[str] = None
    source: Optional[str] = None
    camera_id: Optional[str] = None
    room_id: Optional[str] = None

@dataclass(frozen=True)
class EntryEvent:
    student_id: str
    name: str
    timestamp: datetime
    score: float
    event_type: str = "ENTRY"
    track_id: Optional[str] = None
    source: Optional[str] = None
    camera_id: Optional[str] = None
    room_id: Optional[str] = None

@dataclass(frozen=True)
class ExitEvent:
    student_id: str
    name: str
    timestamp: datetime
    duration_seconds: int
    event_type: str = "EXIT"
    track_id: Optional[str] = None
    camera_id: Optional[str] = None
    room_id: Optional[str] = None
    closed_session: Optional['PresenceSession'] = None
