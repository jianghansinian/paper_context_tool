import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from embedding import build_embedding_client, get_embedding, _local_embedding


class TestLocalEmbedding:
    def test_output_shape(self):
        vec = _local_embedding("test text")
        assert vec.shape == (1024,)
        assert vec.dtype == float

    def test_different_inputs_different_outputs(self):
        vec_a = _local_embedding("object detection with CNNs")
        vec_b = _local_embedding("semantic segmentation survey")
        assert not np.allclose(vec_a, vec_b)

    def test_similar_inputs_similar_outputs(self):
        vec_a = _local_embedding("object detection using neural networks")
        vec_b = _local_embedding("object detection with deep learning")
        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        assert cos_sim > 0.2


class TestBuildEmbeddingClient:
    def test_returns_none_when_no_key(self, monkeypatch):
        import embedding as _emb
        monkeypatch.setattr(_emb, "EMBEDDING_API_KEY", "")
        monkeypatch.setenv("ENABLE_LOCAL_EMBEDDING_FALLBACK", "1")
        client = build_embedding_client()
        assert client is None

    def test_creates_client_with_key(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
        client = build_embedding_client(api_key="sk-test-direct")
        assert client is not None
        assert client.api_key == "sk-test-direct"


class TestGetEmbedding:
    def test_falls_back_to_local_when_client_is_none(self):
        vec = get_embedding("test text", client=None)
        assert vec.shape == (1024,)
