import pytest
from datetime import datetime, time, timedelta
from src.attendance.session import PresenceSession
from src.attendance.policy import AttendancePolicyEngine, Timetable, ScheduledPeriod

@pytest.fixture
def timetable():
    return Timetable(periods=[
        ScheduledPeriod(
            period_id="p1", 
            subject_name="DBMS", 
            room_id="201", 
            start_time=datetime(2026, 9, 26, 10, 0), 
            end_time=datetime(2026, 9, 26, 11, 0)
        )
    ])

@pytest.fixture
def engine():
    return AttendancePolicyEngine()

# 1. matching room and time
def test_matching_room_and_time(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 10), last_seen_time=datetime(2026, 9, 26, 10, 50),
        exit_time=datetime(2026, 9, 26, 10, 50), duration_seconds=2400,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = engine.resolve_period(session, timetable)
    assert period is not None
    assert period.subject_name == "DBMS"

# 2. matching time but wrong room
def test_matching_time_wrong_room(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 10), last_seen_time=datetime(2026, 9, 26, 10, 50),
        exit_time=datetime(2026, 9, 26, 10, 50), duration_seconds=2400,
        status="CLOSED", camera_id="1", room_id="999"
    )
    period = engine.resolve_period(session, timetable)
    assert period is None

# 3. no matching period
def test_no_matching_period(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 12, 10), last_seen_time=datetime(2026, 9, 26, 12, 50),
        exit_time=datetime(2026, 9, 26, 12, 50), duration_seconds=2400,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = engine.resolve_period(session, timetable)
    assert period is None

# 4. exact period-start boundary
def test_exact_period_start_boundary(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = engine.resolve_period(session, timetable)
    assert period is not None

# 5. exact period-end boundary
def test_exact_period_end_boundary(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 30), last_seen_time=datetime(2026, 9, 26, 11, 0),
        exit_time=datetime(2026, 9, 26, 11, 0), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = engine.resolve_period(session, timetable)
    assert period is not None

# OVERLAP
# 6. session starts before class
def test_overlap_starts_before_class(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 9, 50), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=2400,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, _, _ = engine.calculate_qualifying_presence([session], period)
    assert duration == 1800 # 10:00 to 10:30

# 7. session ends after class
def test_overlap_ends_after_class(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 30), last_seen_time=datetime(2026, 9, 26, 11, 10),
        exit_time=datetime(2026, 9, 26, 11, 10), duration_seconds=2400,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, _, _ = engine.calculate_qualifying_presence([session], period)
    assert duration == 1800 # 10:30 to 11:00

# 8. session fully covers class
def test_overlap_fully_covers_class(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 9, 0), last_seen_time=datetime(2026, 9, 26, 12, 0),
        exit_time=datetime(2026, 9, 26, 12, 0), duration_seconds=10800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, _, _ = engine.calculate_qualifying_presence([session], period)
    assert duration == 3600 # 10:00 to 11:00

# 9. session outside class
def test_overlap_outside_class(timetable, engine):
    session = PresenceSession(
        session_id="s1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 8, 0), last_seen_time=datetime(2026, 9, 26, 9, 0),
        exit_time=datetime(2026, 9, 26, 9, 0), duration_seconds=3600,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, _, _ = engine.calculate_qualifying_presence([session], period)
    assert duration == 0

# 10. multiple sessions
def test_multiple_sessions(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 10), last_seen_time=datetime(2026, 9, 26, 10, 20),
        exit_time=datetime(2026, 9, 26, 10, 20), duration_seconds=600,
        status="CLOSED", camera_id="0", room_id="201"
    )
    s2 = PresenceSession(
        session_id="2", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 40), last_seen_time=datetime(2026, 9, 26, 10, 50),
        exit_time=datetime(2026, 9, 26, 10, 50), duration_seconds=600,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, first, last = engine.calculate_qualifying_presence([s1, s2], period)
    assert duration == 1200
    assert first == datetime(2026, 9, 26, 10, 10)
    assert last == datetime(2026, 9, 26, 10, 50)

# 11. overlapping sessions without double counting
def test_overlapping_sessions(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    s2 = PresenceSession(
        session_id="2", student_id="123", name="Nimai", track_id="t2",
        entry_time=datetime(2026, 9, 26, 10, 20), last_seen_time=datetime(2026, 9, 26, 10, 50),
        exit_time=datetime(2026, 9, 26, 10, 50), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    period = timetable.periods[0]
    duration, _, _ = engine.calculate_qualifying_presence([s1, s2], period)
    assert duration == 3000 # 10:00 to 10:50

# LATE
# 12. entry before threshold
def test_entry_before_threshold(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 5), last_seen_time=datetime(2026, 9, 26, 10, 55),
        exit_time=datetime(2026, 9, 26, 10, 55), duration_seconds=3000,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "PRESENT"

# 13. entry exactly at threshold
def test_entry_at_threshold(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 10), last_seen_time=datetime(2026, 9, 26, 11, 0),
        exit_time=datetime(2026, 9, 26, 11, 0), duration_seconds=3000,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "PRESENT" # Not late if exactly at threshold (assuming <=)

# 14. entry after threshold
def test_entry_after_threshold(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 11), last_seen_time=datetime(2026, 9, 26, 11, 0),
        exit_time=datetime(2026, 9, 26, 11, 0), duration_seconds=2940,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "LATE"

# POLICY
# 15. PRESENT
def test_policy_present(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 5), last_seen_time=datetime(2026, 9, 26, 10, 55),
        exit_time=datetime(2026, 9, 26, 10, 55), duration_seconds=3000,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "PRESENT"

# 16. PARTIAL
def test_policy_partial(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "PARTIAL"

# 17. ABSENT
def test_policy_absent(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 5),
        exit_time=datetime(2026, 9, 26, 10, 5), duration_seconds=300,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "ABSENT"

# 18. LATE
def test_policy_late(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 11), last_seen_time=datetime(2026, 9, 26, 11, 0),
        exit_time=datetime(2026, 9, 26, 11, 0), duration_seconds=2940,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert record.status == "LATE"

# 19. REVIEW/unmapped case
def test_policy_review_unmapped(timetable, engine):
    s1 = PresenceSession(
        session_id="1", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 5), last_seen_time=datetime(2026, 9, 26, 10, 55),
        exit_time=datetime(2026, 9, 26, 10, 55), duration_seconds=3000,
        status="CLOSED", camera_id="0", room_id="201"
    )
    # If a zero-length period is passed in (which is a data anomaly), it should fall to REVIEW or ABSENT based on division by zero
    bad_period = ScheduledPeriod(
        period_id="p2", subject_name="Anomaly", room_id="201",
        start_time=datetime(2026, 9, 26, 10, 0), end_time=datetime(2026, 9, 26, 10, 0) # 0 seconds
    )
    record = engine.evaluate("123", "Nimai", bad_period, [s1])
    # According to policy, if period duration <= 0, percentage=0 -> ABSENT
    assert record.status == "ABSENT"

# AUDITABILITY
# 20. AttendanceRecord contains source session IDs
def test_record_contains_source_session_ids(timetable, engine):
    s1 = PresenceSession(
        session_id="sid-123", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert "sid-123" in record.source_session_ids

# 21. AttendanceRecord contains explainable reason
def test_record_contains_explainable_reason(timetable, engine):
    s1 = PresenceSession(
        session_id="sid-123", student_id="123", name="Nimai", track_id="t1",
        entry_time=datetime(2026, 9, 26, 10, 0), last_seen_time=datetime(2026, 9, 26, 10, 30),
        exit_time=datetime(2026, 9, 26, 10, 30), duration_seconds=1800,
        status="CLOSED", camera_id="0", room_id="201"
    )
    record = engine.evaluate("123", "Nimai", timetable.periods[0], [s1])
    assert len(record.reason) > 5
