"""Tests for embedding-based retrieval in the hierarchical memory and EMMS."""

import pytest
from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder

# Optional
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")
from emms.storage.chroma import ChromaStore


@pytest.fixture
def embedder():
    return HashEmbedder(dim=128)


@pytest.fixture
def agent_with_embeddings(embedder):
    return EMMS(config=MemoryConfig(working_capacity=7), embedder=embedder)


@pytest.fixture
def agent_with_chroma(embedder):
    chroma = ChromaStore(embedder=embedder, collection_name="test_emms")
    return EMMS(
        config=MemoryConfig(working_capacity=7),
        embedder=embedder,
        vector_store=chroma,
    )


FINANCE_EXPS = [
    Experience(content="Stock market rose 3% on strong earnings", domain="finance", importance=0.9),
    Experience(content="Federal Reserve cut interest rates by 25bps", domain="finance", importance=0.85),
    Experience(content="Bitcoin surged past 100k dollar milestone", domain="finance", importance=0.8),
]

WEATHER_EXPS = [
    Experience(content="Heavy rainfall and flooding in coastal areas", domain="weather", importance=0.6),
    Experience(content="Tornado warning issued for midwest region", domain="weather", importance=0.7),
]

ALL_EXPS = FINANCE_EXPS + WEATHER_EXPS


class TestEmbeddingRetrieval:
    def test_retrieval_uses_embedding_strategy(self, agent_with_embeddings):
        for exp in ALL_EXPS:
            agent_with_embeddings.store(exp)

        results = agent_with_embeddings.retrieve("stock market performance")
        assert len(results) > 0
        assert results[0].strategy == "embedding"

    def test_embedding_retrieval_relevance(self, agent_with_embeddings):
        for exp in ALL_EXPS:
            agent_with_embeddings.store(exp)

        results = agent_with_embeddings.retrieve("interest rates monetary policy")
        assert len(results) > 0
        # Top result should be finance-related
        top_domains = [r.memory.experience.domain for r in results[:3]]
        assert "finance" in top_domains

    def test_cross_domain_discrimination(self, agent_with_embeddings):
        for exp in ALL_EXPS:
            agent_with_embeddings.store(exp)

        finance_results = agent_with_embeddings.retrieve("stock earnings market")
        weather_results = agent_with_embeddings.retrieve("rain flooding tornado")

        finance_domains = [r.memory.experience.domain for r in finance_results]
        weather_domains = [r.memory.experience.domain for r in weather_results]

        # Finance query should mostly return finance
        assert finance_domains.count("finance") >= finance_domains.count("weather")
        # Weather query should mostly return weather
        assert weather_domains.count("weather") >= weather_domains.count("finance")


class TestSemanticRetrieval:
    def test_semantic_via_chroma(self, agent_with_chroma):
        for exp in ALL_EXPS:
            agent_with_chroma.store(exp)

        results = agent_with_chroma.retrieve_semantic("stock market performance")
        assert len(results) > 0
        assert "score" in results[0]
        assert "content" in results[0]

    def test_semantic_returns_relevant(self, agent_with_chroma):
        for exp in ALL_EXPS:
            agent_with_chroma.store(exp)

        results = agent_with_chroma.retrieve_semantic("Federal Reserve rate cut")
        assert len(results) > 0
        # Should find the Fed-related experience near the top
        contents = [r["content"] for r in results[:3]]
        assert any("Federal" in c or "interest" in c for c in contents)

    def test_semantic_with_domain_filter(self, agent_with_chroma):
        for exp in ALL_EXPS:
            agent_with_chroma.store(exp)

        results = agent_with_chroma.retrieve_semantic(
            "severe conditions",
            where={"domain": "weather"},
        )
        for r in results:
            assert r["metadata"]["domain"] == "weather"

    def test_semantic_fallback_without_vectorstore(self, agent_with_embeddings):
        """When no vector store, retrieve_semantic falls back to hierarchical."""
        for exp in ALL_EXPS:
            agent_with_embeddings.store(exp)

        results = agent_with_embeddings.retrieve_semantic("stock market")
        assert len(results) > 0
        # Falls back to dict format
        assert "content" in results[0]

    def test_store_indexes_vector_store(self, agent_with_chroma):
        for exp in ALL_EXPS:
            result = agent_with_chroma.store(exp)
            assert result["vector_indexed"] is True

        assert agent_with_chroma.vector_store.count == len(ALL_EXPS)
