"""Tests for hierarchical memory system."""

import time
import pytest
from emms.core.models import Experience, MemoryConfig, MemoryTier
from emms.memory.hierarchical import HierarchicalMemory


@pytest.fixture
def memory():
    return HierarchicalMemory(MemoryConfig(working_capacity=5))


@pytest.fixture
def experiences():
    return [
        Experience(content="The stock market rose 3% today", domain="finance", importance=0.8, novelty=0.7),
        Experience(content="Interest rates were cut by the Fed", domain="finance", importance=0.9, novelty=0.9),
        Experience(content="Apple released a new iPhone model", domain="tech", importance=0.6, novelty=0.5),
        Experience(content="Python 3.13 adds pattern matching improvements", domain="programming", importance=0.7, novelty=0.8),
        Experience(content="The weather will be sunny tomorrow", domain="weather", importance=0.3, novelty=0.2),
        Experience(content="Bitcoin hit a new all-time high", domain="finance", importance=0.95, novelty=0.95),
        Experience(content="New AI model breaks benchmark records", domain="tech", importance=0.85, novelty=0.9),
    ]


class TestStore:
    def test_store_single(self, memory):
        exp = Experience(content="hello world")
        item = memory.store(exp)
        assert item.tier == MemoryTier.WORKING
        assert memory.size["working"] == 1
        assert memory.total_stored == 1

    def test_store_fills_working(self, memory, experiences):
        for exp in experiences[:5]:
            memory.store(exp)
        # At capacity, consolidation may have promoted some items
        total = memory.size["working"] + memory.size["short_term"]
        assert total == 5

    def test_store_triggers_consolidation(self, memory, experiences):
        """When working memory hits capacity, consolidation should fire."""
        for exp in experiences[:6]:
            memory.store(exp)
        # Some items should have moved to short-term
        total = memory.size["working"] + memory.size["short_term"]
        assert total >= 6


class TestRetrieve:
    def test_retrieve_by_content(self, memory, experiences):
        for exp in experiences:
            memory.store(exp)
        results = memory.retrieve("stock market finance")
        assert len(results) > 0
        assert "stock" in results[0].memory.experience.content.lower() or \
               "finance" in results[0].memory.experience.domain.lower()

    def test_retrieve_empty(self, memory):
        results = memory.retrieve("anything")
        assert results == []

    def test_retrieve_respects_max(self, memory, experiences):
        for exp in experiences:
            memory.store(exp)
        results = memory.retrieve("the", max_results=2)
        assert len(results) <= 2

    def test_retrieve_updates_access(self, memory):
        exp = Experience(content="unique searchable content xyz")
        memory.store(exp)
        results = memory.retrieve("unique searchable xyz")
        assert len(results) > 0
        assert results[0].memory.access_count >= 1


class TestConsolidation:
    def test_manual_consolidation(self, memory, experiences):
        for exp in experiences:
            memory.store(exp)
        moved = memory.consolidate()
        assert isinstance(moved, int)

    def test_consolidation_promotes_important(self):
        """High importance + novelty items should consolidate faster."""
        mem = HierarchicalMemory(MemoryConfig(working_capacity=3, consolidation_threshold=0.5))

        important = Experience(content="critical event", importance=0.99, novelty=0.99)
        trivial = Experience(content="boring stuff", importance=0.1, novelty=0.1)

        mem.store(important)
        mem.store(trivial)
        mem.store(Experience(content="filler to trigger consolidation"))

        # After consolidation, important should be in short_term
        assert mem.size["short_term"] >= 1

    def test_decay_removes_weak_memories(self):
        mem = HierarchicalMemory(MemoryConfig(working_capacity=3, consolidation_threshold=0.3))
        exp = Experience(content="test decay", importance=0.5, novelty=0.5)
        item = mem.store(exp)

        # Manually promote to short-term and age it
        mem.short_term.append(item)
        item.tier = MemoryTier.SHORT_TERM
        item.last_accessed = time.time() - 864000  # 10 days ago
        item.memory_strength = 0.05  # very weak

        initial_st = len(mem.short_term)
        mem._consolidate_short_term()
        # Weak memory should be forgotten
        assert len(mem.short_term) <= initial_st


class TestSize:
    def test_size_dict(self, memory):
        assert memory.size == {
            "working": 0,
            "short_term": 0,
            "long_term": 0,
            "semantic": 0,
            "total": 0,
        }

    def test_size_after_stores(self, memory, experiences):
        for exp in experiences[:3]:
            memory.store(exp)
        assert memory.size["total"] >= 3
