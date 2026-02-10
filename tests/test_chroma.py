"""Tests for ChromaDB vector store backend."""

import pytest

# Skip all tests if chromadb not installed
chromadb = pytest.importorskip("chromadb")

from emms.core.embeddings import HashEmbedder
from emms.storage.chroma import ChromaStore


@pytest.fixture
def store(tmp_path):
    """Isolated ChromaDB store per test (persistent to tmp to avoid shared state)."""
    embedder = HashEmbedder(dim=64)
    return ChromaStore(
        embedder=embedder,
        collection_name="test_collection",
        persist_directory=tmp_path / "chroma_test",
    )


@pytest.fixture
def populated_store(tmp_path):
    """Store with some documents added."""
    embedder = HashEmbedder(dim=64)
    s = ChromaStore(
        embedder=embedder,
        collection_name="populated_test",
        persist_directory=tmp_path / "chroma_pop",
    )
    docs = [
        ("doc1", "Stock market rose 5% on strong earnings", {"domain": "finance"}),
        ("doc2", "Federal Reserve cut interest rates", {"domain": "finance"}),
        ("doc3", "Heavy rain and flooding expected", {"domain": "weather"}),
        ("doc4", "New AI model beats benchmarks", {"domain": "tech"}),
        ("doc5", "Bitcoin surged past 100k milestone", {"domain": "finance"}),
        ("doc6", "Hurricane warning for coastal areas", {"domain": "weather"}),
    ]
    for doc_id, content, meta in docs:
        s.add(doc_id, content, meta)
    return s


class TestAdd:
    def test_add_single(self, store):
        store.add("id1", "test document")
        assert store.count == 1

    def test_add_with_metadata(self, store):
        store.add("id1", "test", {"domain": "finance", "importance": 0.9})
        results = store.query("test", n_results=1)
        assert results[0]["metadata"]["domain"] == "finance"

    def test_add_with_precomputed_embedding(self, store):
        vec = [0.1] * 64
        store.add("id1", "test", embedding=vec)
        assert store.count == 1

    def test_add_batch(self, store):
        ids = ["a", "b", "c"]
        contents = ["alpha", "beta", "gamma"]
        count = store.add_batch(ids, contents)
        assert count == 3
        assert store.count == 3

    def test_upsert_updates(self, store):
        store.add("id1", "original content")
        store.add("id1", "updated content")
        assert store.count == 1
        results = store.query("updated", n_results=1)
        assert "updated" in results[0]["content"]


class TestQuery:
    def test_query_basic(self, populated_store):
        results = populated_store.query("stock market earnings")
        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_query_returns_scores(self, populated_store):
        results = populated_store.query("finance stock market")
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_query_n_results(self, populated_store):
        results = populated_store.query("anything", n_results=2)
        assert len(results) <= 2

    def test_query_with_where_filter(self, populated_store):
        results = populated_store.query(
            "severe conditions",
            n_results=10,
            where={"domain": "weather"},
        )
        for r in results:
            assert r["metadata"]["domain"] == "weather"

    def test_query_relevance_ordering(self, populated_store):
        results = populated_store.query("stock market finance earnings")
        if len(results) >= 2:
            # First result should have highest score
            assert results[0]["score"] >= results[1]["score"]

    def test_query_empty_store(self, store):
        results = store.query("anything")
        assert results == []


class TestDelete:
    def test_delete(self, populated_store):
        initial = populated_store.count
        populated_store.delete(["doc1", "doc2"])
        assert populated_store.count == initial - 2

    def test_reset(self, populated_store):
        assert populated_store.count > 0
        populated_store.reset()
        assert populated_store.count == 0


class TestPersistence:
    def test_persistent_store(self, tmp_path):
        embedder = HashEmbedder(dim=64)

        # Create and add
        store1 = ChromaStore(
            embedder=embedder,
            persist_directory=tmp_path / "chroma_test",
            collection_name="persist_test",
        )
        store1.add("id1", "persistent document", {"key": "value"})
        assert store1.count == 1

        # Reopen from same directory
        store2 = ChromaStore(
            embedder=embedder,
            persist_directory=tmp_path / "chroma_test",
            collection_name="persist_test",
        )
        assert store2.count == 1
        results = store2.query("persistent", n_results=1)
        assert len(results) == 1
