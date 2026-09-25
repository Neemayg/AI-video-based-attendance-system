"""
Configuration for the registration module.
"""
import torch
import os

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
