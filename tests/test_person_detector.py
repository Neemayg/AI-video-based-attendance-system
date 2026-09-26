import numpy as np
import torch
import pytest
from src.detection.person_detector import PersonDetector

class FakeModel:
    def __init__(self, boxes, scores, labels):
        self.boxes = boxes
        self.scores = scores
        self.labels = labels

    def __call__(self, tensor):
        return [{
            'boxes': self.boxes,
            'scores': self.scores,
            'labels': self.labels
        }]

    def to(self, device):
        pass

    def eval(self):
        pass

def test_nms_removes_hand_inside_body(monkeypatch):
    detector = PersonDetector(device='cpu')
    
    # 1: Main body (large), 2: Hand (small, inside body), 3: Non-person (label 2)
    boxes = torch.tensor([
        [10.0, 10.0, 100.0, 200.0],  # Main body
        [20.0, 50.0, 40.0, 80.0],    # Hand
        [0.0, 0.0, 10.0, 10.0]       # Dog
    ])
    scores = torch.tensor([0.9, 0.85, 0.9])
    labels = torch.tensor([1, 1, 2])
    
    detector.model = FakeModel(boxes, scores, labels)
    
    results = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    
    assert len(results) == 1
    np.testing.assert_array_equal(results[0].box, [10.0, 10.0, 100.0, 200.0])

def test_nms_preserves_independent_people(monkeypatch):
    detector = PersonDetector(device='cpu')
    
    boxes = torch.tensor([
        [10.0, 10.0, 100.0, 200.0],
        [150.0, 10.0, 250.0, 200.0]
    ])
    scores = torch.tensor([0.9, 0.9])
    labels = torch.tensor([1, 1])
    
    detector.model = FakeModel(boxes, scores, labels)
    
    results = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert len(results) == 2

def test_nms_preserves_overlapping_but_distinct_people(monkeypatch):
    detector = PersonDetector(device='cpu')
    
    boxes = torch.tensor([
        [10.0, 10.0, 100.0, 200.0],
        [50.0, 10.0, 140.0, 200.0]
    ])
    scores = torch.tensor([0.9, 0.9])
    labels = torch.tensor([1, 1])
    
    detector.model = FakeModel(boxes, scores, labels)
    
    results = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert len(results) == 2
