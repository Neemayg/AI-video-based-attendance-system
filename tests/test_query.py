import numpy as np
import torch

from src.recognition import query


def test_query_embedding_accepts_tensor_without_running_detection(monkeypatch):
    face_tensor = torch.zeros(3, 160, 160)
    expected = np.ones((1, 512), dtype=np.float32)

    def fake_generate_embeddings(tensors):
        assert tensors == [face_tensor]
        return expected

    monkeypatch.setattr(
        "src.registration.embedding.generate_embeddings", fake_generate_embeddings
    )

    result = query.generate_query_embedding(face_tensor)

    np.testing.assert_array_equal(result, expected[0])