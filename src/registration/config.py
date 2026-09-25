"""
Configuration for the registration module.
"""
import torch
import os

CAMERA_INDEX = 0
SAMPLE_TARGET = 10
MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60
BLUR_THRESHOLD = 50.0  # Laplacian variance threshold
MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Storage paths
STORAGE_BASE_DIR = os.path.join('data', 'embeddings')
STORAGE_GALLERY_JSON = os.path.join(STORAGE_BASE_DIR, 'gallery.json')
