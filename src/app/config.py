"""Runtime settings for the one-face camera MVP."""

import os


CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "60"))
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.65"))
WINDOW_NAME = os.getenv("ATTENDANCE_WINDOW_NAME", "AI Video Attendance")

# --- Performance: Run full face detection every Nth frame to boost FPS ---
# A value of 3 means detection runs once every 3 frames, reusing the prior
# recognition results for the intermediate frames. Increase on slow (CPU) systems.
DETECTION_CYCLE_FRAMES = int(os.getenv("DETECTION_CYCLE_FRAMES", "3"))