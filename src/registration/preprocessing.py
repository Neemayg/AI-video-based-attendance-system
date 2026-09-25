"""
Face detection, alignment and validation pipeline using MTCNN.
"""
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from typing import Optional, Tuple

from . import config

_mtcnn = None

def get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=config.MODEL_DEVICE
        )
    return _mtcnn

def validate_quality(frame: np.ndarray, box: np.ndarray) -> bool:
    """Check if the face is large enough and not too blurry."""
    # Check size
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width < config.MIN_FACE_WIDTH or height < config.MIN_FACE_HEIGHT:
        return False
        
    # Check blur on the face crop
    face_crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
    if face_crop.size == 0:
        return False
        
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if variance < config.BLUR_THRESHOLD:
        return False
        
    return True


def validate_frontal_pose(landmarks: np.ndarray) -> bool:
    """Reject strong side poses using MTCNN eye, nose, and mouth landmarks."""
    if landmarks is None or np.asarray(landmarks).shape != (5, 2):
        return False

    left_eye, right_eye, nose, mouth_left, mouth_right = np.asarray(landmarks)
    eye_vector = right_eye - left_eye
    eye_distance = np.linalg.norm(eye_vector)
    if eye_distance == 0 or eye_vector[0] <= 0:
        return False

    if abs(eye_vector[1]) / eye_distance > config.FRONTAL_EYE_SLOPE_RATIO:
        return False

    feature_ratios = [
        (nose[0] - left_eye[0]) / eye_vector[0],
        ((mouth_left[0] + mouth_right[0]) / 2 - left_eye[0]) / eye_vector[0],
    ]
    return all(
        config.FRONTAL_FEATURE_RATIO_MIN <= ratio <= config.FRONTAL_FEATURE_RATIO_MAX
        for ratio in feature_ratios
    )

def preprocess_face_with_box(
    frame: np.ndarray,
) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
    """
    Detects and aligns a single face from a BGR OpenCV frame.
    Returns its box and preprocessed face tensor or None if validation fails.
    """
    mtcnn = get_mtcnn()
    
    # Convert BGR to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
    
    if boxes is None or len(boxes) != 1:
        return None  # Reject 0 or >1 faces
        
    box = boxes[0]
    prob = probs[0]
    
    if prob < 0.90:  # Confident detection
        return None

    if not validate_frontal_pose(landmarks[0]):
        return None
        
    if not validate_quality(frame, box):
        return None
        
    # Reuse the detected box so MTCNN does not detect the same frame twice.
    face_tensor = mtcnn.extract(rgb_frame, boxes, save_path=None)
    
    # Since we are sure there is 1 face, extract returns one aligned tensor.
    if face_tensor is None:
        return None
        
    if face_tensor.ndim == 4:  # batch of faces
        if face_tensor.size(0) != 1:
            return None
        face_tensor = face_tensor[0]
        
    return box, face_tensor


def preprocess_face(frame: np.ndarray) -> Optional[torch.Tensor]:
    """Return one aligned face tensor while preserving the capture API."""
    result = preprocess_face_with_box(frame)
    return result[1] if result is not None else None
