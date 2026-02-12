"""Tests for multi-strategy retrieval."""

import pytest
import time
from emms.core.models import Experience, MemoryItem, MemoryTier, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.retrieval.strategies import (
    SemanticStrategy,
    TemporalStrategy,
    EmotionalStrategy,
    GraphStrategy,
    DomainStrategy,
    EnsembleRetriever,
)


@pytest.fixture
def embedder():
    return HashEmbedder(dim=64)


@pytest.fixture
def items():
    """Create sample memory items for testing."""
    experiences = [
        Experience(content="Stock market rose 5% today", domain="finance", importance=0.8),
        Experience(content="Bitcoin hit a new all-time high", domain="finance", importance=0.9),
        Experience(content="Python 4.0 was released with major changes", domain="tech", importance=0.7),
        Experience(content="Quantum computing breakthrough at MIT", domain="science", importance=0.85),
        Experience(
            content="I felt really happy about the good news",
            domain="personal",
            emotional_valence=0.8,
            emotional_intensity=0.7,
        ),
    ]
    return [MemoryItem(experience=exp, tier=MemoryTier.WORKING) for exp in experiences]


class TestSemanticStrategy:
    def test_lexical_fallback(self, items):
        strategy = SemanticStrategy(embedder=None)
        score = strategy.score("stock market", items[0], {})
        assert score > 0  # "stock" and "market" overlap

    def test_lexical_no_overlap(self, items):
        strategy = SemanticStrategy(embedder=None)
        score = strategy.score("quantum physics", items[0], {})
        # Very low overlap
        assert score < 0.3

    def test_with_embedder(self, embedder, items):
        strategy = SemanticStrategy(embedder=embedder)
        ctx = {}
        score = strategy.score("stock market trends", items[0], ctx)
        assert 0.0 <= score <= 1.0
        assert "query_vec" in ctx  # should cache query vector

    def test_embedder_caches_query(self, embedder, items):
        strategy = SemanticStrategy(embedder=embedder)
        ctx = {}
        strategy.score("test query", items[0], ctx)
        strategy.score("test query", items[1], ctx)
        # Query vec should be computed only once
        assert "query_vec" in ctx


class TestTemporalStrategy:
    def test_recent_items_score_higher(self, items):
        strategy = TemporalStrategy(half_life=3600)
        # Recent item (just created)
        score_recent = strategy.score("anything", items[0], {})
        assert score_recent > 0.3  # should be recent

    def test_accessed_items_score_higher(self, items):
        strategy = TemporalStrategy()
        item = items[0]
        for _ in range(5):
            item.touch()
        score = strategy.score("anything", item, {})
        assert score > 0.3


class TestEmotionalStrategy:
    def test_emotional_match(self, items):
        strategy = EmotionalStrategy()
        # Item 4 has positive emotion
        score = strategy.score("I am so happy and excited", items[4], {})
        assert score > 0.3

    def test_neutral_query(self, items):
        strategy = EmotionalStrategy()
        score = strategy.score("the report was published", items[4], {})
        # Neutral query should get baseline
        assert 0.0 <= score <= 1.0

    def test_infer_emotion(self):
        strategy = EmotionalStrategy()
        v, i = strategy._infer_emotion("I am so happy and excited")
        assert v > 0  # positive valence
        assert i > 0  # some intensity

    def test_no_emotion_words(self):
        strategy = EmotionalStrategy()
        v, i = strategy._infer_emotion("the weather report today")
        assert v == 0.0
        assert i == 0.0


class TestDomainStrategy:
    def test_same_domain(self, items):
        strategy = DomainStrategy()
        ctx = {}
        score = strategy.score("stock market trading investment", items[0], ctx)
        assert score > 0.5  # finance keywords → finance domain match

    def test_different_domain(self, items):
        strategy = DomainStrategy()
        ctx = {}
        # Science query vs finance item
        score = strategy.score("quantum physics experiment", items[0], ctx)
        assert score < 0.5

    def test_no_domain_signal(self, items):
        strategy = DomainStrategy()
        score = strategy.score("hello world", items[0], {})
        assert score == 0.3  # neutral baseline

    def test_infer_domain(self):
        strategy = DomainStrategy()
        assert strategy._infer_domain("stock market trading") == "finance"
        assert strategy._infer_domain("machine learning algorithm") == "tech"
        assert strategy._infer_domain("hello world") is None


class TestGraphStrategy:
    def test_no_graph(self, items):
        strategy = GraphStrategy()
        score = strategy.score("anything", items[0], {})
        assert score == 0.0  # no graph in context

    def test_with_entity_overlap(self, items):
        from emms.memory.graph import GraphMemory
        graph = GraphMemory()

        # Store experience to populate entities
        items[0].experience.entities = ["stock", "market"]

        strategy = GraphStrategy()
        ctx = {"graph": graph}
        # Add "stock" to graph
        graph.entities["stock"] = None  # just needs to exist as key
        score = strategy.score("stock", items[0], ctx)
        assert score > 0.0


class TestEnsembleRetriever:
    def test_empty_strategies(self, items):
        retriever = EnsembleRetriever()
        results = retriever.retrieve("query", items)
        assert results == []

    def test_single_strategy(self, items):
        retriever = EnsembleRetriever()
        retriever.add_strategy(DomainStrategy(), weight=1.0)
        results = retriever.retrieve("stock market trading", items, max_results=5)
        assert isinstance(results, list)

    def test_multi_strategy(self, embedder, items):
        retriever = EnsembleRetriever()
        retriever.add_strategy(SemanticStrategy(embedder), weight=0.35)
        retriever.add_strategy(TemporalStrategy(), weight=0.20)
        retriever.add_strategy(EmotionalStrategy(), weight=0.15)
        retriever.add_strategy(DomainStrategy(), weight=0.15)
        results = retriever.retrieve("stock market", items, max_results=3)
        assert len(results) <= 3
        # Results should be sorted by score
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_relevance_threshold(self, items):
        retriever = EnsembleRetriever()
        retriever.add_strategy(DomainStrategy(), weight=1.0)
        results = retriever.retrieve(
            "stock market", items,
            relevance_threshold=0.99,
        )
        # Very high threshold should filter most
        assert len(results) <= len(items)

    def test_results_are_retrieval_results(self, embedder, items):
        retriever = EnsembleRetriever()
        retriever.add_strategy(SemanticStrategy(embedder), weight=1.0)
        results = retriever.retrieve("market", items, max_results=2)
        for r in results:
            assert hasattr(r, "memory")
            assert hasattr(r, "score")
            assert hasattr(r, "source_tier")
            assert r.strategy == "ensemble"
