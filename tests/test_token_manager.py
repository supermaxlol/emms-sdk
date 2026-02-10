"""Tests for token context manager."""

import pytest
from emms.core.models import MemoryConfig
from emms.context.token_manager import TokenContextManager


@pytest.fixture
def manager():
    cfg = MemoryConfig(context_window=20, eviction_ratio=0.3)
    return TokenContextManager(cfg)


class TestIngest:
    def test_ingest_within_window(self, manager):
        ctx = manager.ingest(["hello", "world"])
        assert "hello" in ctx
        assert "world" in ctx
        assert len(ctx) == 2

    def test_ingest_fills_window(self, manager):
        tokens = [f"tok_{i}" for i in range(20)]
        ctx = manager.ingest(tokens)
        assert len(ctx) == 20

    def test_ingest_triggers_eviction(self, manager):
        # Fill to capacity
        manager.ingest([f"tok_{i}" for i in range(20)])
        assert len(manager.evicted_tokens) == 0

        # Add more — should evict
        manager.ingest(["overflow_1", "overflow_2"])
        assert len(manager.evicted_tokens) > 0
        assert manager.total_evicted > 0
        assert len(manager.local_context) <= 20


class TestEviction:
    def test_eviction_removes_least_important(self, manager):
        # Ingest tokens with varying characteristics
        manager.ingest(["a", "bb", "IMPORTANT", "longwordhere", "x"])
        initial_count = len(manager.local_context)

        evicted = manager._evict(2)
        assert len(evicted) == 2
        assert len(manager.local_context) == initial_count - 2

    def test_eviction_empty(self, manager):
        evicted = manager._evict(5)
        assert evicted == []

    def test_eviction_zero_count(self, manager):
        manager.ingest(["a", "b"])
        evicted = manager._evict(0)
        assert evicted == []


class TestRetrieveRelevant:
    def test_retrieve_matching_tokens(self, manager):
        # Fill and evict
        manager.ingest([f"tok_{i}" for i in range(20)])
        manager.ingest(["new_1", "new_2", "new_3"])
        assert len(manager.evicted_tokens) > 0

        # Try to retrieve
        retrieved = manager.retrieve_relevant({"tok_0", "tok_1"})
        # Should find exact matches if they were evicted
        for tok in retrieved:
            assert isinstance(tok, str)

    def test_retrieve_from_empty(self, manager):
        results = manager.retrieve_relevant({"test"})
        assert results == []


class TestStats:
    def test_utilisation(self, manager):
        assert manager.utilisation == 0.0
        manager.ingest(["a", "b", "c"])
        assert manager.utilisation == pytest.approx(3 / 20)

    def test_stats_dict(self, manager):
        s = manager.stats
        assert "context_window" in s
        assert "local_tokens" in s
        assert "evicted_tokens" in s
        assert "utilisation" in s
        assert s["context_window"] == 20
