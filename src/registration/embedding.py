"""
InceptionResnetV1 model loading and embedding generation.
"""
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from . import config

_resnet = None

def get_resnet() -> InceptionResnetV1:
    global _resnet
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(config.MODEL_DEVICE)
    return _resnet

def generate_embeddings(face_tensors: list[torch.Tensor]) -> np.ndarray:
    """
    Takes a list of preprocessed face tensors and generates a batch of embeddings.
    Returns: [N, 512] float32 array, L2 normalized.
    """
    if not face_tensors:
        return np.array([])
        
    resnet = get_resnet()
    
    # Stack tensors to create a batch [N, 3, 160, 160]
    batch = torch.stack(face_tensors).to(config.MODEL_DEVICE)
    
    with torch.no_grad():
        embeddings = resnet(batch)
        
    embeddings_np = embeddings.cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    # Prevent division by zero
    norms[norms == 0] = 1e-10
    embeddings_np = embeddings_np / norms
    
    return embeddings_np.astype(np.float32)

def create_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Creates a single representative embedding (centroid) from a batch.
    Returns: [512] float32 array, L2 normalized.
    """
    if len(embeddings) == 0:
        return np.array([])
        
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        norm = 1e-10
    centroid = centroid / norm
    return centroid.astype(np.float32)
