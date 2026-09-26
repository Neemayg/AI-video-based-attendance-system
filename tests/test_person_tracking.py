import numpy as np
import torch

from src.detection.person import (
    PersonDetection,
    PersonTracker,
    PersonTrack,
    associate_face_to_person,
)
from src.detection import person


def detection(x1, y1, x2, y2, confidence=1.0):
    return PersonDetection(np.array([x1, y1, x2, y2], dtype=np.float32), confidence)


def test_person_track_creation_and_update_preserves_id():
    tracker = PersonTracker(max_missed_frames=2)
    first = tracker.update([detection(0, 0, 40, 80)])[0]
    updated = tracker.update([detection(4, 3, 44, 83)])[0]

    assert updated.track_id == first.track_id
    np.testing.assert_array_equal(updated.box, [4, 3, 44, 83])
    assert updated.visible


def test_person_track_survives_brief_loss_and_reappears():
    tracker = PersonTracker(max_missed_frames=2)
    first = tracker.update([detection(0, 0, 40, 80)])[0]
    hidden = tracker.update([])[0]
    assert hidden.track_id == first.track_id
    assert not hidden.visible

    reappeared = tracker.update([detection(2, 1, 42, 81)])[0]

    assert reappeared.track_id == first.track_id


def test_multiple_people_get_independent_stable_ids():
    tracker = PersonTracker(max_missed_frames=1)
    initial = tracker.update([detection(0, 0, 40, 80), detection(100, 0, 140, 80)])
    updated = tracker.update([detection(102, 1, 142, 81), detection(2, 2, 42, 82)])

    ids_by_x = {int(track.box[0]): track.track_id for track in updated}
    assert ids_by_x[2] == initial[0].track_id
    assert ids_by_x[102] == initial[1].track_id


def test_face_association_requires_one_unambiguous_person():
    tracks = [
        PersonTrack(1, np.array([0, 0, 50, 100], dtype=np.float32), 1.0),
        PersonTrack(2, np.array([60, 0, 110, 100], dtype=np.float32), 1.0),
    ]

    assert associate_face_to_person(np.array([10, 10, 30, 30]), tracks) == 1
    assert associate_face_to_person(np.array([200, 10, 220, 30]), tracks) is None


def test_face_association_does_not_assign_overlapping_ambiguous_tracks():
    tracks = [
        PersonTrack(1, np.array([0, 0, 80, 100], dtype=np.float32), 1.0),
        PersonTrack(2, np.array([40, 0, 120, 100], dtype=np.float32), 1.0),
    ]

    assert associate_face_to_person(np.array([50, 10, 70, 30]), tracks) is None


def test_person_detector_filters_non_person_and_low_confidence(monkeypatch):
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, images):
            return [{
                "boxes": torch.tensor([[1, 2, 31, 82], [40, 2, 70, 82], [80, 2, 110, 82]]),
                "labels": torch.tensor([1, 3, 1]),
                "scores": torch.tensor([0.9, 0.99, 0.2]),
            }]

    monkeypatch.setattr(person, "_person_model", FakeModel())
    monkeypatch.setattr(person, "_person_transform", lambda image: torch.zeros(3, 40, 40))

    detections = person.detect_people(np.zeros((40, 40, 3), dtype=np.uint8), 0.5)

    assert len(detections) == 1
    np.testing.assert_array_equal(detections[0].box, [1, 2, 31, 82])


def test_person_detector_passes_pil_image_to_weights_transform(monkeypatch):
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, images):
            return [{
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,)),
            }]

    received = []
    monkeypatch.setattr(person, "_person_model", FakeModel())
    monkeypatch.setattr(
        person,
        "_person_transform",
        lambda image: received.append(image) or torch.zeros(3, 40, 40),
    )

    person.detect_people(np.zeros((40, 40, 3), dtype=np.uint8))

    assert received[0].__class__.__name__ == "Image"