"""
Camera lifecycle and sample collection.
"""
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
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
    captured_tensors = []
    print("\nStarting camera... Look straight at the camera.")
    print("Press SPACE or ENTER to capture a sample. Press Q to cancel.")
    
    try:
        while len(captured_tensors) < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Render HUD
            display_frame = frame.copy()
            status_text = (
                f"Samples: {len(captured_tensors)}/{num_samples} | "
                "SPACE/ENTER: capture | Q: cancel"
            )
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Registration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nRegistration cancelled by user.")
                break
            if key not in (ord(' '), 10, 13):
                continue

            face_tensor = preprocess_face(frame)
            if face_tensor is not None:
                captured_tensors.append(face_tensor)
                print(f"Face detected ✓ [{len(captured_tensors)}/{num_samples}]")
                if len(captured_tensors) == num_samples // 2:
                    print("Keep facing the camera with your whole face visible...")
            else:
                print("No valid face found. Adjust your position and press SPACE or ENTER again.")
                    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    return captured_tensors
