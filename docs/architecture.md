# System Architecture

This document describes the conceptual architecture for the AI-Based Video Attendance System.

---

## 1. Current MVP Architecture

The initial MVP focuses exclusively on single-face identification from a live camera feed.

```
Camera
  │
  ▼
Face Detection
  │
  ▼
Face Embedding
  │
  ▼
Face Recognition
  │
  ▼
Known / Unknown
```

### Component Description (MVP)
1. **Camera**: Captures live video feed (single frame / stream input).
2. **Face Detection**: Detects a single face within the frame and crops region of interest.
3. **Face Embedding**: Extracts feature representation vector for the detected face.
4. **Face Recognition**: Computes similarity distance against registered student embeddings.
5. **Known / Unknown**: Classifies identity and displays match score and status.

---

## 2. Target Future Architecture (PLANNED)

> **Note:** The diagram below represents the **FUTURE** target architecture. These modules are NOT implemented in the initial repository foundation phase.

```
Video Input
  │
  ▼
Detection
  │
  ▼
Tracking
  │
  ▼
Recognition
  │
  ▼
Event Processing
  │
  ▼
Attendance Intelligence
  │
  ▼
Storage / API
  │
  ▼
Dashboard
```

### Target Modules (Future Phases)
- **Video Input**: Multi-camera or video file streams.
- **Detection & Tracking**: Multi-person detection with persistent ID tracking (e.g. ByteTrack).
- **Recognition & Anti-Spoofing**: High-confidence face identification with quality check.
- **Event Processing**: Entry/exit boundary events and duration tracking.
- **Attendance Intelligence**: Automated attendance rules, period classification, and anomaly detection.
- **Storage / API & Dashboard**: Persistent database, REST/WebSocket APIs, and admin monitoring interface.
