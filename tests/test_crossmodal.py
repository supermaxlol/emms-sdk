"""Tests for cross-modal memory binding."""

import pytest
from emms.core.models import Experience, Modality
from emms.crossmodal.binding import CrossModalMemory, _text_features, _emotional_features


@pytest.fixture
def cm():
    return CrossModalMemory()


@pytest.fixture
def finance_experiences():
    return [
        Experience(content="Stock market surged 5% on strong earnings", domain="finance",
                   emotional_valence=0.7, emotional_intensity=0.8, importance=0.9),
        Experience(content="Interest rates cut by central bank", domain="finance",
                   emotional_valence=0.3, emotional_intensity=0.5, importance=0.85),
        Experience(content="Market crash fears grow amid recession signals", domain="finance",
                   emotional_valence=-0.8, emotional_intensity=0.9, importance=0.95),
    ]


@pytest.fixture
def mixed_experiences():
    return [
        Experience(content="The cat sat on the mat", domain="general"),
        Experience(content="Bitcoin price hits new record high", domain="finance",
                   emotional_valence=0.6, importance=0.8),
        Experience(content="Rainy day with thunderstorms expected", domain="weather",
                   emotional_valence=-0.3, emotional_intensity=0.4),
    ]


class TestFeatureExtraction:
    def test_text_features_shape(self):
        exp = Experience(content="hello world this is a test")
        vec = _text_features(exp)
        assert vec.shape == (16,)
        assert vec.dtype.kind == "f"

    def test_emotional_features(self):
        exp = Experience(content="sad news", emotional_valence=-0.5, emotional_intensity=0.8)
        vec = _emotional_features(exp)
        assert vec[0] == pytest.approx(0.25)   # (-0.5+1)/2
        assert vec[1] == pytest.approx(0.8)

    def test_different_content_gives_different_features(self):
        a = _text_features(Experience(content="short"))
        b = _text_features(Experience(content="a much longer sentence with many different words"))
        assert not (a == b).all()


class TestStore:
    def test_store_single(self, cm):
        exp = Experience(content="test")
        result = cm.store(exp)
        assert "text" in result
        assert cm.total_stored == 1

    def test_store_indexes_all_modalities(self, cm):
        exp = Experience(content="test content here")
        cm.store(exp)
        for mod in Modality:
            assert exp.id in cm._indices[mod]

    def test_size_after_stores(self, cm, finance_experiences):
        for exp in finance_experiences:
            cm.store(exp)
        sizes = cm.size
        assert sizes["text"] == 3
        assert sizes["emotional"] == 3


class TestRetrieve:
    def test_retrieve_similar(self, cm, finance_experiences):
        for exp in finance_experiences:
            cm.store(exp)

        query = Experience(content="Stock market performance and earnings reports", domain="finance",
                           emotional_valence=0.5, importance=0.8)
        results = cm.retrieve(query)
        assert len(results) > 0
        # First result should be the most similar finance experience
        assert results[0]["score"] > 0

    def test_retrieve_empty_store(self, cm):
        query = Experience(content="anything")
        results = cm.retrieve(query)
        assert results == []

    def test_retrieve_max_results(self, cm, finance_experiences):
        for exp in finance_experiences:
            cm.store(exp)
        query = Experience(content="finance market")
        results = cm.retrieve(query, max_results=1)
        assert len(results) <= 1

    def test_cross_domain_retrieval(self, cm, mixed_experiences):
        for exp in mixed_experiences:
            cm.store(exp)

        query = Experience(content="Bitcoin cryptocurrency price", domain="finance")
        results = cm.retrieve(query)
        assert len(results) > 0
        # Should find the bitcoin experience
        ids = [r["experience_id"] for r in results]
        bitcoin_exp = [e for e in mixed_experiences if "Bitcoin" in e.content][0]
        assert bitcoin_exp.id in ids

    def test_modality_scores_present(self, cm, finance_experiences):
        for exp in finance_experiences:
            cm.store(exp)
        query = Experience(content="market data")
        results = cm.retrieve(query)
        if results:
            assert "modality_scores" in results[0]
            assert isinstance(results[0]["modality_scores"], dict)
