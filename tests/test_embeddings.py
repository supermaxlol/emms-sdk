"""Tests for embedding providers."""

import pytest
from emms.core.embeddings import HashEmbedder, cosine_similarity, EmbeddingProvider


class TestHashEmbedder:
    def test_implements_protocol(self):
        e = HashEmbedder()
        assert isinstance(e, EmbeddingProvider)

    def test_dim(self):
        e = HashEmbedder(dim=64)
        assert e.dim == 64

    def test_embed_returns_correct_size(self):
        e = HashEmbedder(dim=128)
        vec = e.embed("hello world")
        assert len(vec) == 128

    def test_embed_is_normalised(self):
        import numpy as np
        e = HashEmbedder(dim=128)
        vec = np.array(e.embed("test sentence with multiple words"))
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_embed_empty_string(self):
        e = HashEmbedder(dim=64)
        vec = e.embed("")
        assert len(vec) == 64
        assert all(v == 0.0 for v in vec)

    def test_deterministic(self):
        e = HashEmbedder()
        v1 = e.embed("same text")
        v2 = e.embed("same text")
        assert v1 == v2

    def test_different_texts_differ(self):
        e = HashEmbedder()
        v1 = e.embed("stock market rose today")
        v2 = e.embed("rainy weather forecast tomorrow")
        assert v1 != v2

    def test_similar_texts_closer(self):
        e = HashEmbedder()
        v_a = e.embed("stock market rose on earnings")
        v_b = e.embed("stock market fell on earnings")
        v_c = e.embed("rainy weather and thunderstorms")

        sim_ab = cosine_similarity(v_a, v_b)
        sim_ac = cosine_similarity(v_a, v_c)
        # Finance sentences should be more similar to each other than to weather
        assert sim_ab > sim_ac

    def test_embed_batch(self):
        e = HashEmbedder(dim=64)
        texts = ["hello", "world", "test"]
        vecs = e.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)
        # Should match individual embeds
        for text, batch_vec in zip(texts, vecs):
            single_vec = e.embed(text)
            assert batch_vec == single_vec


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 1.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0
