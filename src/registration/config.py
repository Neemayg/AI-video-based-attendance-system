"""
Configuration for the registration module.
"""

import os

import torch

from typing import NamedTuple


class AnglePrompt(NamedTuple):
    """Guidance prompt shown to the user while capturing a sample."""

    label: str
    instruction: str


CAMERA_INDEX = 0
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "60"))
SAMPLE_TARGET = 10
MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "50.0"))
FRONTAL_EYE_SLOPE_RATIO = 0.25
FRONTAL_FEATURE_RATIO_MIN = 0.25
FRONTAL_FEATURE_RATIO_MAX = 0.75
MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Storage paths
STORAGE_BASE_DIR = os.path.join('data', 'embeddings')
STORAGE_GALLERY_JSON = os.path.join(STORAGE_BASE_DIR, 'gallery.json')

ANGLE_PROMPTS = [
    AnglePrompt(
        label="Frontal",
        instruction="Face the camera directly. Keep your head straight.",
    ),
    AnglePrompt(
        label="Frontal Tilt Down",
        instruction="Tilt your chin down slightly. Forehead visible.",
    ),
    AnglePrompt(
        label="Frontal Tilt Up",
        instruction="Tilt your chin up slightly. Chin visible.",
    ),
    AnglePrompt(
        label="Left Profile",
        instruction="Turn your face ~30 degrees to the left.",
    ),
    AnglePrompt(
        label="Right Profile",
        instruction="Turn your face ~30 degrees to the right.",
    ),
]
