"""Query embedding wrapper using the registration representation pipeline."""

import numpy as np
import torch


def generate_query_embedding(face_tensor: torch.Tensor) -> np.ndarray:
    """Generate one 512-D embedding from an aligned face tensor."""
    from src.registration.embedding import generate_embeddings

    embeddings = generate_embeddings([face_tensor])
    if embeddings.shape != (1, 512):
        raise ValueError("query embedding pipeline must return one 512-D embedding")
    return embeddings[0]