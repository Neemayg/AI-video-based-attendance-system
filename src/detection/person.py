"""Lightweight full-body person detection, tracking, and face association."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

_person_model = None
_person_transform = None


@dataclass(frozen=True)
class PersonDetection:
    """A full-body person box returned by the detector."""

    box: np.ndarray
    confidence: float


@dataclass
class PersonTrack:
    """Stable identity and geometry for one person across detection cycles."""

    track_id: int
    box: np.ndarray
    confidence: float
    missed_frames: int = 0
    visible: bool = True
    identity: dict | None = None


def detect_people(frame: np.ndarray, min_confidence: float = 0.0) -> list[PersonDetection]:
    """Detect full-body people with the lightweight torchvision detector."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    global _person_model, _person_transform
    if _person_model is None:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _person_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        _person_model.eval()
        _person_transform = weights.transforms()

    rgb_frame = frame[:, :, ::-1].copy()
    image = _person_transform(Image.fromarray(rgb_frame))
    with torch.no_grad():
        output = _person_model([image])[0]

    detections = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        confidence = float(score)
        if int(label) == 1 and confidence >= min_confidence:
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append(
                PersonDetection(
                    np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence,
                )
            )
    return detections


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != (4,) or right.shape != (4,):
        return 0.0
    intersection = np.maximum(
        0.0,
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
    )
    area = float(intersection[0] * intersection[1])
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - area
    return area / union if union > 0 else 0.0


class PersonTracker:
    """Greedy geometry tracker with bounded occlusion grace."""

    def __init__(self, max_missed_frames: int = 2, distance_scale: float = 1.5):
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.max_missed_frames = max_missed_frames
        self.distance_scale = distance_scale
        self._next_track_id = 1
        self.tracks: list[PersonTrack] = []

    def update(self, detections: list[PersonDetection]) -> list[PersonTrack]:
        """Match detections to existing tracks and return active tracks."""
        pairs = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                overlap = _iou(track.box, detection.box)
                old_center = (track.box[:2] + track.box[2:]) / 2
                new_center = (detection.box[:2] + detection.box[2:]) / 2
                size = max(track.box[2] - track.box[0], track.box[3] - track.box[1], 1.0)
                distance = float(np.linalg.norm(old_center - new_center))
                if overlap > 0 or distance <= size * self.distance_scale:
                    pairs.append((overlap, -distance, track_index, detection_index))

        assignments = {}
        used_detections = set()
        for _, _, track_index, detection_index in sorted(pairs, reverse=True):
            if track_index in assignments or detection_index in used_detections:
                continue
            assignments[track_index] = detection_index
            used_detections.add(detection_index)

        next_tracks = []
        for track_index, track in enumerate(self.tracks):
            detection_index = assignments.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                track.visible = False
                if track.missed_frames <= self.max_missed_frames:
                    next_tracks.append(track)
                continue
            detection = detections[detection_index]
            track.box = detection.box.copy()
            track.confidence = detection.confidence
            track.missed_frames = 0
            track.visible = True
            next_tracks.append(track)

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            next_tracks.append(
                PersonTrack(
                    self._next_track_id,
                    detection.box.copy(),
                    detection.confidence,
                )
            )
            self._next_track_id += 1
        self.tracks = next_tracks
        return list(self.tracks)


def associate_face_to_person(face_box: np.ndarray, tracks: list[PersonTrack]) -> int | None:
    """Return a person ID only when one visible person contains the face center."""
    face_box = np.asarray(face_box, dtype=float)
    if face_box.shape != (4,) or not np.all(np.isfinite(face_box)):
        return None
    center = (face_box[:2] + face_box[2:]) / 2
    candidates = []
    for track in tracks:
        box = np.asarray(track.box, dtype=float)
        if not track.visible or box.shape != (4,) or not np.all(np.isfinite(box)):
            continue
        if box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]:
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            candidates.append((area, track))
    if len(candidates) != 1:
        return None
    return candidates[0][1].track_id