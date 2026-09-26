# Development Roadmap

This document outlines the multi-phase development plan for the AI-Based Video Attendance System.

---

## PHASE 0 — Project Foundation
- **Status**: IN PROGRESS
- **Objective**: Establish clean, modular repository structure and developer workflow.
- **Inputs**: Project specifications and folder standards.
- **Processing**: Repository setup, directory tree creation, baseline documentation.
- **Outputs**: Clean foundation codebase ready for parallel feature development.
- **Test Cases**: Directory verification, environment dependency checks.
- **Known Issues**: None.

---

## PHASE 1 — Face Registration
- **Status**: COMPLETE
- **Objective**: Register approximately 5 students and save their embeddings.
- **Inputs**: Student ID, Student Name, face sample images.
- **Processing**: Face embedding extraction and persistent gallery saving.
- **Outputs**: Saved embedding files under `data/embeddings/`.
- **Test Cases**: Functional tests and integrity tests passing.
- **Known Issues**: None.

---

## PHASE 2 — Face Recognition
- **Status**: COMPLETE
- **Objective**: Match input face embedding against gallery embeddings and classify known/unknown.
- **Inputs**: Face embedding vector, gallery embeddings.
- **Processing**: Cosine / Euclidean distance calculation and threshold matching.
- **Outputs**: Matched identity (Name, Student ID), similarity score, VERIFIED/UNKNOWN status.
- **Test Cases**: 100% passing.
- **Known Issues**: None.

---

## PHASE 3 — Live Camera Integration
- **Status**: COMPLETE
- **Objective**: Connect webcam feed to single-face detection and recognition pipeline with HUD display.
- **Inputs**: Live video stream from camera.
- **Processing**: Frame acquisition, single-face bounding box detection, embedding extraction, gallery lookup, UI overlay.
- **Outputs**: Real-time video window with identity bounding box and similarity score.
- **Test Cases**: Tested via `main.py`.
- **Known Issues**: None.

---

## PHASE 4 — Multi-Person Detection & Tracking
- **Status**: COMPLETE
- **Objective**: Detect multiple faces simultaneously and batch inference.
- **Inputs**: High-resolution crowd frames.
- **Processing**: MTCNN multi-face extraction -> batched tensor creation -> ResNet inference.
- **Outputs**: Array of identified faces inside the frame without massive FPS drops.
- **Test Cases**: Passing.
- **Known Issues**: None.

---

## PHASE 5 — Entry / Exit Detection
- **Status**: COMPLETE
- **Objective**: Determine when a student enters or exits the camera field of view.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 6 — Presence Duration
- **Status**: COMPLETE
- **Objective**: Calculate accumulated presence time for tracked students during a session.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

## PHASE 6.5 — Identity Continuity & Tracking
- **Status**: COMPLETE
- **Objective**: Maintain continuous presence sessions during temporary face-loss (e.g., student turning completely around).
- **Inputs**: Camera frames, bounding boxes.
- **Processing**: Lightweight spatial tracking (Centroid tracking). Binds verified identity to a continuous track.
- **Output**: Stable `track_id` mapped to `Observation` events, preventing false EXITS.

---

## PHASE 7A — Period-Level Attendance Policy
- **Status**: COMPLETE
- **Objective**: Mark attendance (Present, Absent, Late, Partial) based on duration thresholds per timetable period.
- **Inputs**: `PresenceSession`, `ScheduledPeriod`, Policy Config.
- **Processing**: Domain logic computing overlap intersections and late thresholds.
- **Outputs**: Explainable `AttendanceRecord`.
- **Test Cases**: 21 tests passing.
- **Known Issues**: None.

---

## PHASE 7B — Data Persistence
- **Status**: NEXT UP
- **Objective**: Persist generated AttendanceRecords and raw PresenceSessions to a database (e.g., PostgreSQL or SQLite).
- **Inputs**: `AttendanceRecord` objects from Phase 7A.
- **Processing**: ORM mapping, transactional saves.
- **Outputs**: Persisted database rows.

## PHASE 8 — Exception & Alert Engine
- **Status**: FUTURE
- **Objective**: Flag spoofing attempts, unrecognized faces, or prolonged absence.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 9 — Dashboard / Application Layer
- **Status**: FUTURE
- **Objective**: Interactive web interface for attendance reports, analytics, and student management.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD
