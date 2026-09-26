"""
Camera lifecycle and sample collection with guided angle capture UI.
"""
import cv2
import numpy as np
import torch

from . import config
from .preprocessing import preprocess_face


def _overlay_hud(
    frame: np.ndarray,
    step_label: str,
    instruction: str,
    prompt: str,
    captured: int,
    target: int,
    ready: bool,
) -> np.ndarray:
    """Render an attractive heads-up display on a copy of the frame."""
    display = frame.copy()
    height = display.shape[0]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, height - 140), (display.shape[1], height), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

    cv2.putText(
        display,
        f"Register Face — Step {captured + 1}/{target}: {step_label}",
        (20, height - 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        instruction,
        (20, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
    )

    cv2.putText(
        display,
        prompt,
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if ready else (0, 165, 255),
        2,
    )

    bar_y = height - 20
    bar_width = int((captured / target) * display.shape[1])
    cv2.line(display, (0, bar_y), (display.shape[1], bar_y), (80, 80, 80), 4)
    if bar_width > 0:
        cv2.line(display, (0, bar_y), (bar_width, bar_y), (0, 255, 0), 4)

    return display


def capture_face_samples(num_samples: int = config.SAMPLE_TARGET) -> list[torch.Tensor]:
    """
    Opens the webcam and captures `num_samples` valid face tensors.
    Guides the user through capturing faces from multiple predefined angles.
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    captured_tensors = []
    angles = config.ANGLE_PROMPTS
    num_angles = len(angles)
    current_angle_index = 0

    print("\n=== Guided Face Registration ===")
    print("Position your face according to the on-screen prompts.")
    print("Press SPACE or ENTER to capture a sample. Press Q to cancel.")
    print(f"Total samples needed: {num_samples}")

    try:
        while len(captured_tensors) < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            angle = angles[current_angle_index % num_angles]
            progress = len(captured_tensors) / num_samples
            step_label = angle.label

            face_tensor = preprocess_face(frame)
            ready = face_tensor is not None

            prompt_text = (
                "Face ready. Press SPACE/ENTER to capture."
                if ready
                else "Reposition: no valid face detected."
            )

            display_frame = _overlay_hud(
                frame,
                step_label,
                angle.instruction,
                prompt_text,
                len(captured_tensors),
                num_samples,
                ready,
            )
            cv2.imshow("Face Registration", display_frame)
            cv2.setWindowProperty("Face Registration", cv2.WND_PROP_TOPMOST, 1.0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nRegistration cancelled by user.")
                break
            if key not in (ord(' '), 10, 13):
                continue

            if face_tensor is not None:
                captured_tensors.append(face_tensor)
                current_angle_index = (current_angle_index + 1) % num_angles
                print(
                    f"Captured ✓ [{len(captured_tensors)}/{num_samples}] "
                    f"— Angle: {angle.label}"
                )
            else:
                print(
                    f"No valid face found. Ensure {angle.label}: "
                    "face is clear and well-lit."
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_tensors
