"""
Student metadata + embedding persistence.
"""
import os
import json
import numpy as np
from datetime import datetime
from . import config

def load_gallery() -> dict:
    """Loads the gallery metadata."""
    if not os.path.exists(config.STORAGE_GALLERY_JSON):
        return {}
    with open(config.STORAGE_GALLERY_JSON, 'r') as f:
        return json.load(f)

def save_registration(student_id: str, name: str, embeddings: np.ndarray, centroid: np.ndarray) -> None:
    """
    Saves the student embeddings and updates the gallery.
    """
    # Create directory for student
    student_dir = os.path.join(config.STORAGE_BASE_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    
    # File paths
    embeddings_file = os.path.join(student_dir, 'embeddings.npy')
    centroid_file = os.path.join(student_dir, 'centroid.npy')
    
    # Save numpy arrays
    np.save(embeddings_file, embeddings)
    np.save(centroid_file, centroid)
    
    # Update gallery
    gallery = load_gallery()
    
    # Save relative paths for the gallery json
    rel_embeddings_file = f"{student_id}/embeddings.npy"
    rel_centroid_file = f"{student_id}/centroid.npy"
    
    gallery[student_id] = {
        "name": name,
        "embedding_file": rel_embeddings_file,
        "centroid_file": rel_centroid_file,
        "num_samples": embeddings.shape[0],
        "created_at": datetime.now().isoformat()
    }
    
    # Simple save
    with open(config.STORAGE_GALLERY_JSON, 'w') as f:
        json.dump(gallery, f, indent=2)
