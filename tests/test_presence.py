from datetime import datetime, timedelta
import pytest

from src.attendance.presence import PresenceSessionEngine
from src.attendance.events import Observation, EntryEvent, ExitEvent

def _make_obs(student_id: str, name: str, status: str, dt: datetime, track_id: str = "17") -> Observation:
    return Observation(
        student_id=student_id,
        name=name,
        score=0.99,
        status=status,
        timestamp=dt,
        track_id=track_id
    )

def test_first_verified_observation_opens_session():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    
    obs = _make_obs("1", "Krish", "VERIFIED", t0)
    events = engine.process_observations([obs], t0)
    
    assert len(events) == 1
    assert isinstance(events[0], EntryEvent)
    assert events[0].student_id == "1"
    assert "1" in engine.active_sessions
    assert engine.active_sessions["1"].status == "ACTIVE"

def test_repeated_observations_keep_same_session_active():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=1)
    
    events1 = engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0)], t0)
    events2 = engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t1)], t1)
    
    assert len(events1) == 1
    assert len(events2) == 0  # No new entry event
    assert engine.active_sessions["1"].last_seen_time == t1

def test_unknown_observations_do_not_create_sessions():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    
    obs = _make_obs(None, "Unknown", "UNKNOWN", t0)
    events = engine.process_observations([obs], t0)
    
    assert len(events) == 0
    assert len(engine.active_sessions) == 0

def test_missing_one_frame_does_not_close_session():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=1) # Missing
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0)], t0)
    events = engine.process_observations([], t1) # No faces
    
    assert len(events) == 0
    assert "1" in engine.active_sessions

def test_grace_period_closes_session_and_duration_is_correct():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t_last = datetime(2026, 9, 26, 10, 45, 0)
    t_exit = datetime(2026, 9, 26, 10, 45, 3) # 3 seconds later
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0)], t0)
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t_last)], t_last)
    
    # Process with no observations after 3 seconds
    events = engine.process_observations([], t_exit)
    
    assert len(events) == 1
    assert isinstance(events[0], ExitEvent)
    assert events[0].student_id == "1"
    assert events[0].timestamp == t_exit
    assert events[0].duration_seconds == 43 * 60 + 3 # 43 minutes and 3 seconds
    assert "1" not in engine.active_sessions

def test_re_entry_creates_new_session():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=4) # Closes first session
    t2 = t1 + timedelta(seconds=1) # Re-enter
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0)], t0)
    engine.process_observations([], t1) # Triggers exit
    
    events = engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t2)], t2)
    assert len(events) == 1
    assert isinstance(events[0], EntryEvent)

def test_multiple_students_independent_sessions():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=4)
    
    events0 = engine.process_observations([
        _make_obs("1", "Krish", "VERIFIED", t0, track_id="17"),
        _make_obs("2", "Nimai", "VERIFIED", t0, track_id="21")
    ], t0)
    
    assert len(events0) == 2
    assert "1" in engine.active_sessions
    assert "2" in engine.active_sessions
    
    # Only Krish remains
    events1 = engine.process_observations([
        _make_obs("1", "Krish", "VERIFIED", t1, track_id="17")
    ], t1)
    
    assert len(events1) == 1
    assert isinstance(events1[0], ExitEvent)
    assert events1[0].student_id == "2"
    assert "1" in engine.active_sessions
    assert "2" not in engine.active_sessions

def test_duplicate_observations_same_frame():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    
    events = engine.process_observations([
        _make_obs("1", "Krish", "VERIFIED", t0),
        _make_obs("1", "Krish", "VERIFIED", t0)
    ], t0)
    
    assert len(events) == 1
    assert len(engine.active_sessions) == 1

def test_track_remains_active_while_face_disappears():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=1)
    
    # 1. Face recognized, bind to track 17
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0, track_id="17")], t0)
    assert "1" in engine.active_sessions
    
    # 2. Face not recognized but tracker outputs track 17
    obs = _make_obs(None, "Unknown", "TRACKED", t1, track_id="17")
    events = engine.process_observations([obs], t1)
    
    assert len(events) == 0 # no exit
    assert engine.active_sessions["1"].last_seen_time == t1

def test_face_returns_to_same_track_same_session():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=1)
    t2 = t0 + timedelta(seconds=2)
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0, track_id="17")], t0)
    engine.process_observations([_make_obs(None, "Unknown", "TRACKED", t1, track_id="17")], t1)
    
    events = engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t2, track_id="17")], t2)
    
    assert len(events) == 0
    assert engine.active_sessions["1"].last_seen_time == t2

def test_track_loss_creates_exit_after_timeout():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=4)
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0, track_id="17")], t0)
    
    # Tracker drops track 17, so no observation is passed for it
    events = engine.process_observations([], t1)
    
    assert len(events) == 1
    assert isinstance(events[0], ExitEvent)
    assert events[0].student_id == "1"

def test_one_weak_recognition_does_not_switch_identity():
    engine = PresenceSessionEngine(grace_period_seconds=3)
    t0 = datetime(2026, 9, 26, 10, 2, 0)
    t1 = t0 + timedelta(seconds=1)
    
    engine.process_observations([_make_obs("1", "Krish", "VERIFIED", t0, track_id="17")], t0)
    
    # A noisy frame returns UNKNOWN, but track is 17. The engine resolves it to Krish
    obs = _make_obs(None, "Unknown", "UNKNOWN", t1, track_id="17")
    events = engine.process_observations([obs], t1)
    
    assert len(events) == 0
    assert engine.active_sessions["1"].last_seen_time == t1
