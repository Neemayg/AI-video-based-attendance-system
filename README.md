# AI-Based Video Attendance System

## Project Overview

The **AI-Based Video Attendance System** is an intelligent visual surveillance solution designed to identify registered students from live video feeds and eventually convert video observations into automated, reliable attendance decisions.

---

## Current MVP

The immediate Minimum Viable Product (MVP) is intentionally focused on core single-face verification:

- **Student Registration**: Register ~5 students with Student ID, Name, and face samples/embeddings.
- **Face Embedding Generation**: Extract numerical feature vectors representing face identity.
- **Single-Face Detection**: Detect a single face from a laptop webcam feed.
- **Face Recognition**: Compare detected embeddings against the registered gallery.
- **Known / Unknown Classification**: Determine if a face matches a registered record.
- **Match / Similarity Score**: Display identity metadata (Name, Student ID, Similarity Score, Status).

> **Important Scope Boundary:**  
> Multi-person tracking, entry/exit detection, attendance duration, period-level attendance, exception handling and dashboard functionality are planned for later phases.

---

## Development Philosophy

The project is developed incrementally following a clear evolutionary pipeline:

```
Registration (COMPLETE)
  ↓
Recognition
  ↓
Detection / Integration
  ↓
Multi-person Tracking
  ↓
Entry / Exit Detection
  ↓
Presence Duration
  ↓
Period Attendance
  ↓
Intelligence Layer
  ↓
Dashboard
```

---

## Team

- **Person 1** — Krish
- **Person 2** — Swastik
- **Person 3** — Neemay Gupta

---

## Repository Structure

```
.
├── README.md                 # Project overview and workflow documentation
├── requirements.txt          # Minimal Python dependencies for foundation
├── .gitignore                # Git ignore configuration
│
├── src/                      # Source code directory
│   ├── registration/         # Student registration & embedding storage
│   ├── recognition/          # Face matching & known/unknown classification
│   ├── detection/            # Webcam feed acquisition & single-face detection
│   ├── attendance/           # Reserved for future attendance logic
│   ├── utils/                # Configuration, logging, and common helpers
│   └── app/                  # Application entry point and live HUD overlay
│
├── data/                     # Local data storage (git ignored contents)
│   ├── students/             # Local student registration records / samples
│   ├── embeddings/           # Saved face embedding gallery
│   └── logs/                 # Event logs
│
├── tests/                    # Automated unit & integration tests
│
└── docs/                     # Project documentation
    ├── architecture.md       # Conceptual current & future architecture
    ├── technical-contract.md # Model, embedding, and module interface specifications
    ├── development-roadmap.md# Phase-by-phase development plan
    └── testing.md            # Test matrix and guidelines
```

### Folder Explanations
- **`src/registration/`**: Student registration, face sample capture and embedding generation.
- **`src/recognition/`**: Face embedding comparison, identity matching and known/unknown classification.
- **`src/detection/`**: Face detection and, later, tracking (multi-person tracking disabled in MVP).
- **`src/attendance/`**: Reserved for future entry/exit, duration and attendance logic.
- **`src/utils/`**: Shared utilities such as configuration, logging, validation, file handling, etc.
- **`src/app/`**: Application entry point and integration layer.
- **`data/students/`**: Local student registration data / face samples.
- **`data/embeddings/`**: Stored face embeddings gallery.
- **`data/logs/`**: Future recognition and attendance event logs.
- **`tests/`**: Unit and integration tests.
- **`docs/`**: Architecture, roadmap and testing documentation.

---

## Development Workflow & Git Collaboration

The repository uses a branch-per-feature workflow to allow parallel development:

```
main
├── feature/registration        (Assigned: Person 1 — Krish)
├── feature/recognition         (Assigned: Person 2 — Swastik)
└── feature/camera-detection    (Assigned: Person 3 — Neemay Gupta)
```

### Workflow Rules:
1. **Branch Creation**: Developers work on dedicated feature branches (`git checkout -b feature/<feature-name>`).
2. **Commit Standard**: Write clear, modular commit messages for logically grouped changes.
3. **Pull Requests**: Changes must be submitted via Pull Requests targeting `main`.
4. **Code Review**: At least one teammate must review and approve the PR before merging.
5. **No Direct Commits to Main**: All feature additions happen in feature branches.

---

## Coding Rules & Guidelines

- **Python Version**: Python 3.11+ compatibility.
- **Modular Code**: Keep functions and modules small, focused, and single-purpose. Avoid giant `main.py` files.
- **No Hardcoding**: Do not hardcode student identities, model outputs, or absolute file paths.
- **Separation of Concerns**: Do not mix attendance logic with face recognition code.
- **Security**: Never commit secrets, credentials, or API keys.
- **No Overengineering**: Build clean, practical prototypes without premature infrastructure (no microservices, Docker, databases, or frontend frameworks until required).
