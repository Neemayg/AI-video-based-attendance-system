# Testing Strategy & Test Matrix

This document outlines the testing matrix for the AI-Based Video Attendance System.

## MVP Test Matrix

| Test Case | Expected Result | Status |
|-----------|------------------|--------|
| Registered student | Correct identity | NOT STARTED |
| Unknown person | UNKNOWN | NOT STARTED |
| Different face angle | Correct identity if confidence is sufficient | NOT STARTED |
| Poor-quality face | Low confidence / review behavior | NOT STARTED |

---

## Test Execution Guidelines
- Unit tests will reside in the `tests/` directory.
- Automated tests will be executed via `pytest`.
- Test data / mock embeddings should be decoupled from real student records.
