import torch
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from dataclasses import dataclass
from typing import List

@dataclass
class PersonDetection:
    box: np.ndarray
    confidence: float

class PersonDetector:
    def __init__(self, device='cpu', confidence_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        # Load lightweight pre-trained model
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        # Convert BGR (OpenCV) to RGB
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(tensor)[0]
            
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        persons = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score >= self.confidence_threshold: # COCO class 1 is person
                persons.append(PersonDetection(box=box, confidence=float(score)))
                
        # Filter overlapping boxes (e.g. hands detected as separate persons)
        filtered_persons = []
        for i, p1 in enumerate(persons):
            is_part = False
            box1_area = (p1.box[2] - p1.box[0]) * (p1.box[3] - p1.box[1])
            for j, p2 in enumerate(persons):
                if i == j: continue
                box2_area = (p2.box[2] - p2.box[0]) * (p2.box[3] - p2.box[1])
                
                # Check if p1 is substantially smaller and mostly inside p2
                if box1_area < box2_area:
                    ix1 = max(p1.box[0], p2.box[0])
                    iy1 = max(p1.box[1], p2.box[1])
                    ix2 = min(p1.box[2], p2.box[2])
                    iy2 = min(p1.box[3], p2.box[3])
                    
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / box1_area > 0.6:  # 60% of small box is inside large box
                            is_part = True
                            break
            if not is_part:
                filtered_persons.append(p1)
                
        return filtered_persons

_shared_person_detector = None

def get_person_detector() -> PersonDetector:
    global _shared_person_detector
    if _shared_person_detector is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _shared_person_detector = PersonDetector(device=device, confidence_threshold=0.4)
    return _shared_person_detector

def detect_persons(frame: np.ndarray) -> List[PersonDetection]:
    detector = get_person_detector()
    return detector.detect(frame)
