import numpy as np
import torch
from src.app.config import DETECTION_CYCLE_FRAMES
print(f"DETECTION_CYCLE_FRAMES = {DETECTION_CYCLE_FRAMES}")

# Simulate the frame-skipping logic exactly as in main.py
frame_counter = 0
frames_where_detection_runs = []

for i in range(10):  # Simulate 10 frames
    should_detect = (frame_counter % DETECTION_CYCLE_FRAMES == 0)
    if should_detect:
        frames_where_detection_runs.append(frame_counter)
    frame_counter += 1

print(f"Detection runs on frames: {frames_where_detection_runs}")
print(f"Expected: every {DETECTION_CYCLE_FRAMES} frames. PASS={len(frames_where_detection_runs) == 4}")
