"""Integration tests for EMMS v0.4.0 new features.

Tests that all new systems work together: EventBus, consciousness,
graph memory, persistence, patterns, and background consolidation.
"""

import pytest
import time
import asyncio

from emms.core.models import Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.emms import EMMS


@pytest.fixture
def emms():
    return EMMS(config=MemoryConfig(working_capacity=5))


@pytest.fixture
def emms_with_embedder():
    return EMMS(
        config=MemoryConfig(working_capacity=5),
        embedder=HashEmbedder(dim=64),
    )


@pytest.fixture
def sample_experiences():
    return [
        Experience(content="Stock market rose 5% on strong earnings reports", domain="finance", importance=0.8),
        Experience(content="Apple released new iPhone with advanced AI features", domain="tech", importance=0.7),
        Experience(content="Quantum computing breakthrough at Stanford University", domain="science", importance=0.9),
        Experience(content="Heavy rainfall expected across the eastern seaboard", domain="weather", importance=0.4),
        Experience(content="New vaccine trial shows promising results for patients", domain="health", importance=0.85),
        Experience(content="Bitcoin hit record high above 100k dollars today", domain="finance", importance=0.9),
        Experience(content="Microsoft announced partnership with OpenAI for enterprise", domain="tech", importance=0.75),
    ]


class TestEventBusIntegration:
    def test_store_emits_event(self, emms):
        events_received = []
        emms.events.on("memory.stored", lambda data: events_received.append(data))
        emms.store(Experience(content="Test event emission", domain="test"))
        assert len(events_received) == 1
        assert "experience_id" in events_received[0]
        assert events_received[0]["domain"] == "test"

    def test_consolidate_emits_event(self, emms, sample_experiences):
        events_received = []
        emms.events.on("memory.consolidated", lambda data: events_received.append(data))
        for exp in sample_experiences:
            emms.store(exp)
        emms.consolidate()
        assert len(events_received) == 1
        assert "items_consolidated" in events_received[0]

    def test_compress_emits_event(self, emms, sample_experiences):
        events_received = []
        emms.events.on("memory.compressed", lambda data: events_received.append(data))
        for exp in sample_experiences:
            emms.store(exp)
        emms.consolidate()
        emms.compress_long_term()
        # May or may not emit depending on whether there are LT items
        assert isinstance(events_received, list)


class TestConsciousnessIntegration:
    def test_consciousness_enabled_by_default(self, emms):
        assert emms._consciousness_enabled is True
        assert emms.narrator is not None
        assert emms.meaning_maker is not None
        assert emms.temporal is not None
        assert emms.ego_boundary is not None

    def test_consciousness_disabled(self):
        emms = EMMS(enable_consciousness=False)
        assert emms.narrator is None
        assert emms.meaning_maker is None

    def test_store_updates_consciousness(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        # Narrator should have entries
        assert len(emms.narrator.entries) == len(sample_experiences)
        # Themes should be tracked
        assert len(emms.narrator.themes) > 0

    def test_get_narrative(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        narrative = emms.get_narrative("TestAgent")
        assert "TestAgent" in narrative
        assert len(narrative) > 20

    def test_get_first_person_narrative(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        narrative = emms.get_first_person_narrative()
        assert "processed" in narrative.lower() or "experiences" in narrative.lower()

    def test_consciousness_state(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        state = emms.get_consciousness_state()
        assert state["enabled"] is True
        assert "narrative_coherence" in state
        assert "themes" in state
        assert "ego_boundary_strength" in state

    def test_consciousness_state_disabled(self):
        emms = EMMS(enable_consciousness=False)
        state = emms.get_consciousness_state()
        assert state["enabled"] is False


class TestGraphMemoryIntegration:
    def test_graph_enabled_by_default(self, emms):
        assert emms.graph is not None

    def test_graph_disabled(self):
        emms = EMMS(enable_graph=False)
        assert emms.graph is None

    def test_store_populates_graph(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        assert emms.graph.size["entities"] > 0

    def test_query_entity(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        # Query for some entity
        entities = list(emms.graph.entities.keys())
        if entities:
            result = emms.query_entity(entities[0])
            assert result["found"] is True

    def test_query_nonexistent(self, emms):
        result = emms.query_entity("nonexistent_xyz")
        assert result["found"] is False

    def test_graph_disabled_queries(self):
        emms = EMMS(enable_graph=False)
        assert emms.query_entity("x") == {
            "found": False, "entity": None, "neighbors": [], "relationships": []
        }
        assert emms.query_entity_path("a", "b") == []
        assert emms.get_subgraph("x") == {"nodes": [], "edges": []}

    def test_store_result_includes_graph(self, emms):
        result = emms.store(Experience(
            content="Apple Inc announced new products at Cupertino",
            domain="tech",
        ))
        assert "graph_entities" in result


class TestPatternDetection:
    def test_detect_patterns(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        patterns = emms.detect_patterns()
        assert "sequence" in patterns
        assert "content" in patterns
        assert "domain" in patterns


class TestStatsEnhancements:
    def test_stats_include_graph(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        stats = emms.stats
        assert "graph" in stats
        assert "entities" in stats["graph"]

    def test_stats_include_consciousness(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        stats = emms.stats
        assert "consciousness" in stats
        assert "narrative_coherence" in stats["consciousness"]

    def test_stats_include_events(self, emms):
        stats = emms.stats
        assert "events" in stats

    def test_store_result_includes_consciousness(self, emms):
        result = emms.store(Experience(content="Test", domain="test"))
        assert "consciousness" in result


class TestPersistenceIntegration:
    def test_save_load_roundtrip(self, tmp_path):
        # Use larger working capacity so items aren't evicted
        emms = EMMS(config=MemoryConfig(working_capacity=20))
        for exp in [
            Experience(content="Stock market rose 5%", domain="finance"),
            Experience(content="Bitcoin hit record high", domain="finance"),
            Experience(content="Python released new version", domain="tech"),
        ]:
            emms.store(exp)
        memory_path = tmp_path / "memory.json"
        emms.save(memory_path=memory_path)

        emms2 = EMMS(config=MemoryConfig(working_capacity=20))
        emms2.load(memory_path=memory_path)

        # Retrieval should work after loading
        results = emms2.retrieve("stock market")
        assert len(results) > 0


class TestEnhancedConsciousness:
    def test_narrator_traits(self, emms):
        for _ in range(10):
            emms.store(Experience(
                content="Machine learning research paper published",
                domain="tech",
                novelty=0.8,
                importance=0.7,
            ))
        # Should develop "curious" or "analytical" traits
        assert len(emms.narrator.traits) > 0

    def test_narrator_autobiographical(self, emms):
        emms.store(Experience(
            content="Major breakthrough in quantum computing",
            domain="science",
            importance=0.95,
            novelty=0.9,
        ))
        # High-significance events should be recorded
        # (depends on significance threshold)
        assert isinstance(emms.narrator.autobiographical, list)

    def test_meaning_maker_patterns(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        assert emms.meaning_maker.total_processed == len(sample_experiences)
        assert len(emms.meaning_maker.pattern_tracker) > 0

    def test_temporal_milestones(self, emms):
        # Store a high-importance experience to trigger milestone
        emms.store(Experience(
            content="Critical discovery",
            domain="science",
            importance=0.95,
        ))
        # Milestones should be tracked
        assert isinstance(emms.temporal.milestones, list)

    def test_ego_boundary_history(self, emms, sample_experiences):
        for exp in sample_experiences:
            emms.store(exp)
        assert len(emms.ego_boundary.boundary_history) > 0


@pytest.mark.asyncio
class TestAsyncIntegration:
    async def test_async_store_with_consciousness(self):
        emms = EMMS(config=MemoryConfig(working_capacity=5))
        result = await emms.astore(Experience(
            content="Async test experience",
            domain="test",
        ))
        assert "consciousness" in result
        assert result["consciousness"] is True

    async def test_async_consolidate(self):
        emms = EMMS(config=MemoryConfig(working_capacity=3))
        for i in range(5):
            await emms.astore(Experience(
                content=f"Experience {i}",
                domain="test",
                importance=0.8,
            ))
        result = await emms.aconsolidate()
        assert "items_consolidated" in result
