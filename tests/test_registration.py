"""
Functional, integrity, and reload tests for Phase 1 (Registration).
"""
import os
import json
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.registration import storage
from src.registration import embedding
from src.registration import config
from src.registration import cli

@pytest.fixture
def mock_face_tensors():
    # Return a batch of 10 fake face tensors (3, 160, 160)
    return [torch.rand(3, 160, 160) for _ in range(10)]

@pytest.fixture
def isolated_storage(tmp_path):
    """Fixture to isolate storage paths to a pytest tmp_path."""
    original_base = config.STORAGE_BASE_DIR
    original_json = config.STORAGE_GALLERY_JSON
    
    test_base = tmp_path / "embeddings"
    test_base.mkdir(parents=True, exist_ok=True)
    
    config.STORAGE_BASE_DIR = str(test_base)
    config.STORAGE_GALLERY_JSON = str(test_base / "gallery.json")
    
    yield test_base
    
    # Restore original paths
    config.STORAGE_BASE_DIR = original_base
    config.STORAGE_GALLERY_JSON = original_json

# --- 17. Embedding Integrity Tests ---
def test_embedding_integrity(mock_face_tensors):
    """Verify embedding batch and centroid generation with L2 normalization."""
    embeddings = embedding.generate_embeddings(mock_face_tensors)
    
    assert embeddings.shape == (10, 512)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    centroid = embedding.create_centroid(embeddings)
    assert centroid.shape == (512,)
    assert centroid.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(centroid), 1.0, rtol=1e-5)

# --- 13. Gallery Reload & Storage Tests ---
def test_gallery_reload_and_storage(mock_face_tensors, isolated_storage):
    """Verify storing embeddings and reloading the gallery metadata."""
    embeddings = embedding.generate_embeddings(mock_face_tensors)
    centroid = embedding.create_centroid(embeddings)
    
    storage.save_registration("999", "Test User", embeddings, centroid)
    
    gallery = storage.load_gallery()
    assert "999" in gallery
    assert gallery["999"]["name"] == "Test User"
    assert gallery["999"]["num_samples"] == 10
    
    emb_path = os.path.join(isolated_storage, "999", "embeddings.npy")
    cent_path = os.path.join(isolated_storage, "999", "centroid.npy")
    assert os.path.exists(emb_path)
    assert os.path.exists(cent_path)

# --- Functional Tests ---
@patch('src.registration.capture.capture_face_samples')
@patch('builtins.input')
def test_valid_registration(mock_input, mock_capture, mock_face_tensors, isolated_storage):
    """1. Valid registration: Test end-to-end CLI execution."""
    mock_input.side_effect = ["101", "Alice"]
    mock_capture.return_value = mock_face_tensors
    
    cli.main()
    
    gallery = storage.load_gallery()
    assert "101" in gallery
    assert gallery["101"]["name"] == "Alice"
    emb_path = os.path.join(isolated_storage, "101", "embeddings.npy")
    assert os.path.exists(emb_path)

@patch('src.registration.capture.capture_face_samples')
@patch('builtins.input')
def test_duplicate_registration_cancel(mock_input, mock_capture, mock_face_tensors, isolated_storage):
    """2. Duplicate registration: MUST NOT silently overwrite."""
    # First registration
    storage.save_registration(
        "102", "Bob", 
        embedding.generate_embeddings(mock_face_tensors), 
        embedding.create_centroid(embedding.generate_embeddings(mock_face_tensors))
    )
    
    # Second registration attempts to overwrite, but user cancels ('A')
    mock_input.side_effect = ["102", "Bob 2", "A"]
    mock_capture.return_value = mock_face_tensors
    
    cli.main()
    
    # Verify data was NOT overwritten
    gallery = storage.load_gallery()
    assert gallery["102"]["name"] == "Bob"  # Still Bob, not Bob 2

@patch('src.registration.capture.capture_face_samples')
@patch('builtins.input')
def test_explicit_overwrite(mock_input, mock_capture, mock_face_tensors, isolated_storage):
    """3. Explicit overwrite: Verify data is intentionally replaced when user chooses [B]."""
    # First registration
    storage.save_registration(
        "103", "Charlie", 
        embedding.generate_embeddings(mock_face_tensors), 
        embedding.create_centroid(embedding.generate_embeddings(mock_face_tensors))
    )
    
    # Overwrite
    mock_input.side_effect = ["103", "Charlie New", "B"]
    mock_capture.return_value = mock_face_tensors
    
    cli.main()
    
    gallery = storage.load_gallery()
    assert gallery["103"]["name"] == "Charlie New"

@patch('builtins.input')
def test_invalid_student_id_name(mock_input, isolated_storage):
    """4. Invalid student ID/name: Should not create corrupt gallery entries."""
    # Empty ID
    mock_input.side_effect = ["", "Valid Name"]
    cli.main()
    
    # Valid ID, empty name
    mock_input.side_effect = ["104", ""]
    cli.main()
    
    gallery = storage.load_gallery()
    assert len(gallery) == 0
    assert not os.path.exists(os.path.join(isolated_storage, "104"))


def test_angle_prompts_contain_multiple_distinct_angles():
    """Verify the registration angle prompts are configured correctly."""
    labels = [prompt.label for prompt in config.ANGLE_PROMPTS]

    assert len(config.ANGLE_PROMPTS) >= 3
    assert "Frontal" in labels
    assert any("Left" in label for label in labels)
    assert any("Right" in label for label in labels)

    for prompt in config.ANGLE_PROMPTS:
        assert isinstance(prompt.label, str) and len(prompt.label) > 0
        assert isinstance(prompt.instruction, str) and len(prompt.instruction) > 10


def test_sample_target_constant():
    """Verify the default sample target is 10."""
    assert config.SAMPLE_TARGET == 10
