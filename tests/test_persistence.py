"""Tests for memory persistence (save/load state)."""

import pytest
import json
from pathlib import Path

from emms.core.models import Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.memory.hierarchical import HierarchicalMemory
from emms.emms import EMMS


@pytest.fixture
def config():
    return MemoryConfig(working_capacity=5)


@pytest.fixture
def embedder():
    return HashEmbedder(dim=64)


@pytest.fixture
def sample_experiences():
    return [
        Experience(content=f"Experience number {i} about {domain}", domain=domain)
        for i, domain in enumerate([
            "finance", "tech", "science", "weather", "health",
            "finance", "tech", "science", "weather", "health",
        ])
    ]


class TestHierarchicalPersistence:
    def test_save_creates_file(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        mem.store(Experience(content="Test memory", domain="test"))
        path = tmp_path / "memory_state.json"
        mem.save_state(path)
        assert path.exists()

    def test_save_valid_json(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        mem.store(Experience(content="Test memory", domain="test"))
        path = tmp_path / "memory_state.json"
        mem.save_state(path)
        data = json.loads(path.read_text())
        assert "version" in data
        assert "working" in data
        assert "short_term" in data
        assert "long_term" in data
        assert "semantic" in data

    def test_roundtrip_basic(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        for i in range(3):
            mem.store(Experience(content=f"Memory {i}", domain="test"))
        original_size = mem.size["total"]

        path = tmp_path / "state.json"
        mem.save_state(path)

        mem2 = HierarchicalMemory(config)
        mem2.load_state(path)
        assert mem2.size["total"] == original_size

    def test_roundtrip_preserves_content(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        exp = Experience(content="Important financial data XYZ", domain="finance")
        mem.store(exp)
        path = tmp_path / "state.json"
        mem.save_state(path)

        mem2 = HierarchicalMemory(config)
        mem2.load_state(path)
        results = mem2.retrieve("financial data", max_results=5)
        assert len(results) > 0
        assert any("financial" in r.memory.experience.content for r in results)

    def test_roundtrip_with_embeddings(self, tmp_path, config, embedder):
        mem = HierarchicalMemory(config, embedder=embedder)
        mem.store(Experience(content="Stock market analysis", domain="finance"))
        mem.store(Experience(content="Python programming tutorial", domain="tech"))

        path = tmp_path / "state.json"
        mem.save_state(path)

        mem2 = HierarchicalMemory(config, embedder=embedder)
        mem2.load_state(path)
        assert mem2.size["total"] == mem.size["total"]

    def test_roundtrip_after_consolidation(self, tmp_path, config, sample_experiences):
        mem = HierarchicalMemory(config)
        for exp in sample_experiences:
            mem.store(exp)
        mem.consolidate()

        path = tmp_path / "state.json"
        mem.save_state(path)

        mem2 = HierarchicalMemory(config)
        mem2.load_state(path)
        assert mem2.size["total"] == mem.size["total"]
        assert mem2.size["short_term"] == mem.size["short_term"]

    def test_load_nonexistent_file(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        mem.load_state(tmp_path / "nonexistent.json")
        assert mem.size["total"] == 0

    def test_roundtrip_rebuilds_word_index(self, tmp_path, config):
        mem = HierarchicalMemory(config)
        mem.store(Experience(content="Quantum computing breakthrough", domain="science"))
        path = tmp_path / "state.json"
        mem.save_state(path)

        mem2 = HierarchicalMemory(config)
        mem2.load_state(path)
        # Word index should be rebuilt — retrieval should work
        results = mem2.retrieve("quantum", max_results=5)
        assert len(results) > 0


class TestEMMSPersistence:
    def test_emms_save_load(self, tmp_path):
        emms = EMMS(config=MemoryConfig(working_capacity=5))
        emms.store(Experience(content="Important fact to remember", domain="test"))
        memory_path = tmp_path / "memory.json"
        emms.save(memory_path=memory_path)
        assert memory_path.exists()

    def test_emms_load_restores(self, tmp_path):
        emms = EMMS(config=MemoryConfig(working_capacity=5))
        emms.store(Experience(content="Unique test content ABC123", domain="test"))
        memory_path = tmp_path / "memory.json"
        emms.save(memory_path=memory_path)

        emms2 = EMMS(config=MemoryConfig(working_capacity=5))
        emms2.load(memory_path=memory_path)
        results = emms2.retrieve("test content ABC")
        assert len(results) > 0

    def test_emms_save_no_path(self, tmp_path):
        # Save without memory path should just save identity
        emms = EMMS(
            config=MemoryConfig(working_capacity=5),
            identity_path=tmp_path / "identity.json",
        )
        emms.store(Experience(content="Test", domain="test"))
        emms.save()  # no memory_path, should not crash
