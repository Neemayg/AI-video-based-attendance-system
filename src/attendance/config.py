import os

GRACE_PERIOD_SECONDS = int(os.getenv("ATTENDANCE_GRACE_PERIOD_SECONDS", "3"))

# Phase 7 Policy Configuration (Prototype Defaults)
POLICY_PRESENT_THRESHOLD = float(os.getenv("POLICY_PRESENT_THRESHOLD", "0.75"))
POLICY_PARTIAL_THRESHOLD = float(os.getenv("POLICY_PARTIAL_THRESHOLD", "0.25"))
POLICY_LATE_THRESHOLD_MINUTES = int(os.getenv("POLICY_LATE_THRESHOLD_MINUTES", "10"))

# Hardware context mapping
# In a real app this would be loaded from a DB.
# Format: {"camera_id": "room_id"}
CAMERA_ROOM_MAP = {
    "0": "201",
    "1": "202"
}

def get_room_for_camera(camera_id: str) -> str:
    return CAMERA_ROOM_MAP.get(str(camera_id), "UNKNOWN_ROOM")
