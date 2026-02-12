"""Tests for the agent adapter layer."""

import pytest
import tempfile
from pathlib import Path

from emms.core.models import Experience, MemoryConfig
from emms.adapters.agent import AgentMemory


@pytest.fixture
def agent():
    return AgentMemory(agent_name="TestBot")


@pytest.fixture
def agent_with_storage(tmp_path):
    return AgentMemory(agent_name="PersistBot", storage_dir=tmp_path / "memory")


class TestIngestTurn:
    def test_basic_ingest(self, agent):
        result = agent.ingest_turn(
            user_msg="What is the stock market doing?",
            agent_msg="The market is up 2% today.",
            domain="finance",
        )
        assert "experience_id" in result
        assert "narrative_coherence" in result
        assert "ego_boundary" in result

    def test_ingest_tracks_turns(self, agent):
        agent.ingest_turn(user_msg="Hello", agent_msg="Hi there")
        agent.ingest_turn(user_msg="How are you?", agent_msg="I'm good")
        assert len(agent._turns) == 2

    def test_ingest_updates_emms(self, agent):
        agent.ingest_turn(user_msg="Test message", domain="test")
        assert agent.emms.stats["total_stored"] == 1

    def test_ingest_with_high_importance(self, agent):
        result = agent.ingest_turn(
            user_msg="CRITICAL: Market crash!",
            importance=0.95,
        )
        assert result["ego_boundary"] > 0


class TestBuildContext:
    def test_context_includes_narrative(self, agent):
        agent.ingest_turn(user_msg="Markets are up", domain="finance")
        context = agent.build_context("market performance")
        assert "Identity" in context

    def test_context_includes_memories(self, agent):
        agent.ingest_turn(user_msg="Bitcoin hit 100k today", domain="finance")
        context = agent.build_context("bitcoin price")
        assert "Relevant Memories" in context or "Recent Conversation" in context

    def test_context_includes_recent_turns(self, agent):
        agent.ingest_turn(user_msg="Hello there", agent_msg="Hi!")
        context = agent.build_context("greeting", include_recent_turns=5)
        assert "Hello there" in context

    def test_context_respects_token_limit(self, agent):
        for i in range(20):
            agent.ingest_turn(user_msg=f"Message number {i} about various topics")
        context = agent.build_context("topics", max_tokens=100)
        # Should be bounded
        assert len(context.split()) < 200


class TestRecall:
    def test_recall_returns_results(self, agent):
        agent.ingest_turn(user_msg="Stock market rose 5%", domain="finance")
        results = agent.recall("stock market")
        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]

    def test_recall_empty_memory(self, agent):
        results = agent.recall("anything")
        assert results == []


class TestPreCompactionFlush:
    def test_flush_returns_status(self, agent):
        agent.ingest_turn(user_msg="Important data to save", importance=0.9)
        result = agent.pre_compaction_flush()
        assert "consolidated" in result
        assert result["identity_saved"] is True

    def test_flush_consolidates(self, agent):
        for i in range(10):
            agent.ingest_turn(
                user_msg=f"Important fact {i}",
                importance=0.9,
            )
        result = agent.pre_compaction_flush()
        assert isinstance(result["consolidated"], dict)


class TestMarkdownExport:
    def test_export_returns_markdown(self, agent):
        agent.ingest_turn(user_msg="Test message", domain="test")
        md = agent.export_markdown()
        assert "# TestBot Memory" in md
        assert "## Identity" in md

    def test_export_to_file(self, agent, tmp_path):
        agent.ingest_turn(user_msg="Test message", domain="test")
        path = tmp_path / "MEMORY.md"
        agent.export_markdown(path)
        assert path.exists()
        content = path.read_text()
        assert "TestBot" in content

    def test_export_includes_domains(self, agent):
        agent.ingest_turn(user_msg="Stock news", domain="finance")
        agent.ingest_turn(user_msg="Weather update", domain="weather")
        md = agent.export_markdown()
        assert "finance" in md
        assert "weather" in md


class TestMarkdownImport:
    def test_import_from_file(self, agent, tmp_path):
        md_path = tmp_path / "import.md"
        md_path.write_text(
            "# Memories\n"
            "- [finance] Stock market rose 5% today\n"
            "- [weather] It rained heavily in the afternoon\n"
            "- Short line\n"  # too short, should be skipped
        )
        count = agent.import_markdown(md_path)
        assert count >= 2  # at least the two substantive lines
        assert agent.emms.stats["total_stored"] >= 2

    def test_import_nonexistent(self, agent):
        count = agent.import_markdown("/nonexistent/path.md")
        assert count == 0


class TestSessionManagement:
    def test_save_session(self, agent_with_storage):
        agent_with_storage.ingest_turn(user_msg="Hello")
        agent_with_storage.save_session()
        # Check that turns log was created
        log_path = agent_with_storage.storage_dir / "turns.jsonl"
        assert log_path.exists()

    def test_status_returns_dict(self, agent):
        status = agent.status
        assert "agent_name" in status
        assert status["agent_name"] == "TestBot"
        assert "emms_stats" in status
        assert "narrative_coherence" in status


class TestSentimentEstimation:
    def test_positive_valence(self, agent):
        v = agent._estimate_valence("This is great and wonderful news!")
        assert v > 0

    def test_negative_valence(self, agent):
        v = agent._estimate_valence("This is terrible and awful")
        assert v < 0

    def test_neutral_valence(self, agent):
        v = agent._estimate_valence("The report was published today")
        assert v == 0.0

    def test_intensity_exclamation(self, agent):
        intensity = agent._estimate_intensity("WOW!!! AMAZING!!!")
        assert intensity > 0.3

    def test_novelty_first_message(self, agent):
        novelty = agent._estimate_novelty("Completely new topic here")
        assert novelty >= 0.7
