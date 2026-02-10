"""Integration tests for the full EMMS pipeline."""

import time
import pytest
from emms import EMMS, Experience, MemoryConfig


@pytest.fixture
def agent():
    cfg = MemoryConfig(working_capacity=5)
    return EMMS(config=cfg)


@pytest.fixture
def populated_agent(agent):
    """An agent with a mix of experiences already stored."""
    experiences = [
        Experience(content="Apple stock rose 5% after strong quarterly earnings",
                   domain="finance", importance=0.9, novelty=0.8),
        Experience(content="Federal Reserve held interest rates steady",
                   domain="finance", importance=0.85, novelty=0.6),
        Experience(content="New Python 3.13 release improves pattern matching",
                   domain="programming", importance=0.7, novelty=0.9),
        Experience(content="Heavy rainfall expected in coastal areas tomorrow",
                   domain="weather", importance=0.4, novelty=0.3),
        Experience(content="OpenAI announced GPT-5 with reasoning capabilities",
                   domain="tech", importance=0.95, novelty=0.95),
        Experience(content="Bitcoin surged past 100k milestone",
                   domain="finance", importance=0.9, novelty=0.9,
                   emotional_valence=0.7, emotional_intensity=0.8),
        Experience(content="US GDP growth exceeded analyst expectations",
                   domain="finance", importance=0.8, novelty=0.7),
    ]
    for exp in experiences:
        agent.store(exp)
    return agent


class TestFullPipeline:
    def test_store_returns_result(self, agent):
        exp = Experience(content="test event")
        result = agent.store(exp)
        assert "experience_id" in result
        assert "memory_id" in result
        assert "tier" in result
        assert result["elapsed_ms"] >= 0

    def test_store_batch(self, agent):
        exps = [Experience(content=f"event {i}") for i in range(3)]
        results = agent.store_batch(exps)
        assert len(results) == 3

    def test_retrieve_after_store(self, populated_agent):
        results = populated_agent.retrieve("stock market finance earnings")
        assert len(results) > 0
        # Should find finance-related memories
        contents = [r.memory.experience.content.lower() for r in results]
        assert any("stock" in c or "finance" in c or "earnings" in c for c in contents)

    def test_retrieve_different_domain(self, populated_agent):
        results = populated_agent.retrieve("Python programming release")
        assert len(results) > 0

    def test_crossmodal_retrieve(self, populated_agent):
        query = Experience(content="financial market performance",
                          domain="finance", emotional_valence=0.5)
        results = populated_agent.retrieve_crossmodal(query)
        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_episode_detection(self, populated_agent):
        episodes = populated_agent.detect_episodes()
        assert len(episodes) > 0
        total_exps = sum(ep.size for ep in episodes)
        assert total_exps == 7  # all stored experiences

    def test_consolidation(self, populated_agent):
        result = populated_agent.consolidate()
        assert "items_consolidated" in result
        assert "memory_sizes" in result

    def test_stats(self, populated_agent):
        stats = populated_agent.stats
        assert stats["total_stored"] == 7
        assert stats["memory"]["total"] >= 7
        assert stats["identity"]["total_experiences"] == 7
        assert len(stats["identity"]["domains"]) >= 3

    def test_identity_persistence(self, tmp_path):
        path = tmp_path / "ego.json"
        agent = EMMS(identity_path=path)
        agent.store(Experience(content="remember me", domain="test"))
        agent.save()

        # New agent loads the same identity
        agent2 = EMMS(identity_path=path)
        assert agent2.identity.state.total_experiences == 1
        assert agent2.identity.state.session_count == 1


class TestPerformance:
    def test_store_throughput(self, agent):
        """Store 100 experiences and check it's reasonably fast."""
        start = time.time()
        for i in range(100):
            agent.store(Experience(content=f"experience number {i} with some content"))
        elapsed = time.time() - start

        assert elapsed < 5.0  # should be well under 5 seconds
        assert agent.stats["total_stored"] == 100

    def test_retrieve_speed(self, populated_agent):
        """Retrieval should be fast."""
        start = time.time()
        for _ in range(50):
            populated_agent.retrieve("finance stock market")
        elapsed = time.time() - start

        assert elapsed < 2.0  # 50 queries in under 2 seconds


class TestEdgeCases:
    def test_empty_retrieve(self, agent):
        results = agent.retrieve("anything")
        assert results == []

    def test_empty_content(self, agent):
        result = agent.store(Experience(content=""))
        assert result["experience_id"]

    def test_very_long_content(self, agent):
        long_text = "word " * 10000
        result = agent.store(Experience(content=long_text))
        assert result["experience_id"]

    def test_special_characters(self, agent):
        result = agent.store(Experience(content="Hello world special characters @#$% test"))
        assert result["experience_id"]
        # Should store safely and be retrievable
        results = agent.retrieve("Hello world special characters test")
        assert len(results) > 0
