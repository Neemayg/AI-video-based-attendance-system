import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

from .session import PresenceSession
from . import config

@dataclass
class ScheduledPeriod:
    period_id: str
    subject_name: str
    room_id: str
    start_time: datetime
    end_time: datetime

@dataclass
class Timetable:
    periods: List[ScheduledPeriod]
    
    def get_period_for_time(self, current_time: datetime, room_id: str) -> Optional[ScheduledPeriod]:
        for period in self.periods:
            if period.room_id == room_id and period.start_time <= current_time <= period.end_time:
                return period
        return None

@dataclass
class AttendanceRecord:
    record_id: str
    student_id: str
    name: str
    period_id: str
    subject: str
    room_id: str
    period_start: datetime
    period_end: datetime
    first_entry_time: Optional[datetime]
    last_exit_time: Optional[datetime]
    qualifying_presence_seconds: int
    status: str
    reason: str
    source_session_ids: List[str]

class AttendancePolicyEngine:
    def __init__(self, 
                 present_threshold: float = config.POLICY_PRESENT_THRESHOLD, 
                 partial_threshold: float = config.POLICY_PARTIAL_THRESHOLD, 
                 late_arrival_threshold_minutes: int = config.POLICY_LATE_THRESHOLD_MINUTES):
        self.present_threshold = present_threshold
        self.partial_threshold = partial_threshold
        self.late_arrival_threshold_minutes = late_arrival_threshold_minutes

    def resolve_period(self, session: PresenceSession, timetable: Timetable) -> Optional[ScheduledPeriod]:
        """Maps a PresenceSession to a specific Timetable Period based on entry time and room."""
        if not session.room_id:
            return None
        
        best_period = None
        max_overlap = 0
        
        for period in timetable.periods:
            if period.room_id != session.room_id:
                continue
            
            # check intersection
            overlap_start = max(session.entry_time, period.start_time)
            session_end = session.exit_time if session.exit_time else session.last_seen_time
            overlap_end = min(session_end, period.end_time)
            
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            if overlap_seconds > max_overlap:
                max_overlap = overlap_seconds
                best_period = period
                
        # If no overlap, fallback to checking if entry_time is inside period (for 0-second sessions)
        if not best_period:
            return timetable.get_period_for_time(session.entry_time, session.room_id)
            
        return best_period

    def calculate_qualifying_presence(self, sessions: List[PresenceSession], period: ScheduledPeriod) -> Tuple[int, Optional[datetime], Optional[datetime]]:
        """Calculates total non-overlapping presence seconds strictly within the period."""
        if not sessions:
            return 0, None, None
            
        # Get intervals constrained by the period
        intervals = []
        for session in sessions:
            start = max(session.entry_time, period.start_time)
            end = min(session.exit_time or session.last_seen_time, period.end_time)
            if start < end:
                intervals.append((start, end))
                
        if not intervals:
            return 0, min(s.entry_time for s in sessions), max(s.exit_time or s.last_seen_time for s in sessions)
            
        # Merge overlapping intervals
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
                
        total_seconds = int(sum((end - start).total_seconds() for start, end in merged))
        
        # Calculate overall first entry and last exit regardless of period boundaries
        first_entry = min(s.entry_time for s in sessions)
        last_exit = max(s.exit_time or s.last_seen_time for s in sessions)
        
        return total_seconds, first_entry, last_exit

    def evaluate(self, student_id: str, name: str, period: ScheduledPeriod, sessions: List[PresenceSession]) -> AttendanceRecord:
        qualifying_seconds, first_entry, last_exit = self.calculate_qualifying_presence(sessions, period)
        
        period_duration = (period.end_time - period.start_time).total_seconds()
        percentage = qualifying_seconds / period_duration if period_duration > 0 else 0
        
        # Policy evaluation
        late_threshold_time = period.start_time + timedelta(minutes=self.late_arrival_threshold_minutes)
        
        status = "REVIEW"
        reason = "Unable to resolve a consistent period/context"
        
        if qualifying_seconds == 0:
            status = "ABSENT"
            reason = "No qualifying presence during the period"
        else:
            is_late = first_entry is not None and first_entry > late_threshold_time
            
            if percentage >= self.present_threshold:
                if is_late:
                    status = "LATE"
                    reason = "Entry occurred after configured late-arrival threshold"
                else:
                    status = "PRESENT"
                    reason = "Qualifying presence requirement satisfied"
            elif percentage >= self.partial_threshold:
                if is_late:
                    status = "LATE"
                    reason = "Entry occurred after configured late-arrival threshold (partial duration)"
                else:
                    status = "PARTIAL"
                    reason = "Qualifying presence below configured present threshold"
            else:
                status = "ABSENT"
                reason = "No qualifying presence during the period"
                
        return AttendanceRecord(
            record_id=str(uuid.uuid4()),
            student_id=student_id,
            name=name,
            period_id=period.period_id,
            subject=period.subject_name,
            room_id=period.room_id,
            period_start=period.start_time,
            period_end=period.end_time,
            first_entry_time=first_entry,
            last_exit_time=last_exit,
            qualifying_presence_seconds=qualifying_seconds,
            status=status,
            reason=reason,
            source_session_ids=[s.session_id for s in sessions]
        )
