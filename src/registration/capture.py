"""
Camera lifecycle and sample collection.
"""
import time
import cv2
import torch
from . import config
from .preprocessing import preprocess_face

def capture_face_samples(num_samples: int = config.SAMPLE_TARGET) -> list[torch.Tensor]:
    """
    Opens the webcam and captures `num_samples` valid face tensors.
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
        
    captured_tensors = []
    print("\nStarting camera... Look straight at the camera.")
    
    last_capture_time = 0
    cooldown = 0.5  # seconds between captures to ensure stability/variation
    
    try:
        while len(captured_tensors) < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Render HUD
            display_frame = frame.copy()
            status_text = f"Samples: {len(captured_tensors)}/{num_samples}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Registration", display_frame)
            
            # Allow quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRegistration cancelled by user.")
                break
                
            current_time = time.time()
            if current_time - last_capture_time < cooldown:
                continue
                
            face_tensor = preprocess_face(frame)
            if face_tensor is not None:
                captured_tensors.append(face_tensor)
                last_capture_time = current_time
                print(f"Face detected ✓ [{len(captured_tensors)}/{num_samples}]")
                if len(captured_tensors) == num_samples // 2:
                    print("Now turn slightly left or right...")
                    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    return captured_tensors
