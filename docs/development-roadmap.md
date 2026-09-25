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
- **Status**: NOT STARTED
- **Objective**: Register approximately 5 students and save their embeddings.
- **Inputs**: Student ID, Student Name, face sample images.
- **Processing**: Face embedding extraction and persistent gallery saving.
- **Outputs**: Saved embedding files under `data/embeddings/`.
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 2 — Face Recognition
- **Status**: NOT STARTED
- **Objective**: Match input face embedding against gallery embeddings and classify known/unknown.
- **Inputs**: Face embedding vector, gallery embeddings.
- **Processing**: Cosine / Euclidean distance calculation and threshold matching.
- **Outputs**: Matched identity (Name, Student ID), similarity score, VERIFIED/UNKNOWN status.
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 3 — Live Camera Integration
- **Status**: NOT STARTED
- **Objective**: Connect webcam feed to single-face detection and recognition pipeline with HUD display.
- **Inputs**: Live video stream from camera.
- **Processing**: Frame acquisition, single-face bounding box detection, embedding extraction, gallery lookup, UI overlay.
- **Outputs**: Real-time video window with identity bounding box and similarity score.
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 4 — Multi-Person Detection & Tracking
- **Status**: FUTURE
- **Objective**: Detect multiple faces simultaneously and track individual identities across frames.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 5 — Entry / Exit Detection
- **Status**: FUTURE
- **Objective**: Determine when a student enters or exits the camera field of view.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 6 — Presence Duration
- **Status**: FUTURE
- **Objective**: Calculate accumulated presence time for tracked students during a session.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

## PHASE 7 — Period-Level Attendance
- **Status**: FUTURE
- **Objective**: Mark attendance (Present, Absent, Late) based on duration thresholds per timetable period.
- **Inputs**: TBD
- **Processing**: TBD
- **Outputs**: TBD
- **Test Cases**: TBD
- **Known Issues**: TBD

---

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
