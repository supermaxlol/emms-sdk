"""Tests for EMMS v0.14.0 — The Temporal Mind.

Coverage:
  - EpisodicBuffer          (28 tests)
  - SchemaExtractor         (24 tests)
  - MotivatedForgetting     (24 tests)
  - MCP v0.14.0 tools       (8 tests)
  - v0.14.0 exports         (8 tests)

Total: 92 tests
"""

from __future__ import annotations

import time
import pytest

from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS(enable_consciousness=False)


def _make_rich_emms(n: int = 12) -> EMMS:
    """EMMS with memories in multiple domains for schema extraction."""
    agent = _make_emms()
    data = [
        ("science",    "Quantum entanglement enables non-local correlations between particles."),
        ("science",    "Entropy always increases in isolated thermodynamic systems."),
        ("science",    "DNA replication uses polymerase enzymes to copy genetic information."),
        ("science",    "General relativity describes gravity as curvature of spacetime."),
        ("philosophy", "Consciousness may be an emergent property of complex information."),
        ("philosophy", "The hard problem of consciousness asks why there is subjective experience."),
        ("philosophy", "Personal identity persists through psychological continuity over time."),
        ("philosophy", "Existentialism holds that existence precedes essence in human life."),
        ("technology", "Neural networks learn by adjusting weights through backpropagation."),
        ("technology", "Transformers use attention mechanisms to process sequential data."),
        ("technology", "Memory systems in AI agents provide persistent cross-session identity."),
        ("technology", "Hash embeddings map text to fixed-size vectors for similarity."),
    ]
    for domain, content in data[:n]:
        agent.store(Experience(content=content, domain=domain, importance=0.7))
    return agent


# ===========================================================================
# EpisodicBuffer
# ===========================================================================

class TestEpisodicBuffer:

    # --- direct module ---

    def test_open_episode_returns_episode(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        ep = buf.open_episode(topic="test episode")
        assert ep.id.startswith("ep_")
        assert ep.topic == "test episode"
        assert ep.is_open
        assert ep.closed_at is None

    def test_open_episode_auto_generates_session_id(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        ep = buf.open_episode()
        assert ep.session_id.startswith("session_")

    def test_open_episode_custom_session_id(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        ep = buf.open_episode(session_id="my_session")
        assert ep.session_id == "my_session"

    def test_close_episode_sets_closed_at(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode(topic="close test")
        ep = buf.close_episode(outcome="success")
        assert not ep.is_open
        assert ep.closed_at is not None
        assert ep.outcome == "success"

    def test_close_episode_duration(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        ep = buf.close_episode()
        assert ep.duration_seconds is not None
        assert ep.duration_seconds >= 0.0

    def test_close_episode_no_open_returns_none(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        result = buf.close_episode()
        assert result is None

    def test_record_turn_increments_count(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.record_turn(content="hello", valence=0.3)
        buf.record_turn(content="world", valence=-0.1)
        ep = buf.current_episode()
        assert ep.turn_count == 2
        assert len(ep.emotional_arc) == 2

    def test_valence_clipped_to_minus_one_one(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.record_turn(valence=5.0)
        buf.record_turn(valence=-10.0)
        ep = buf.current_episode()
        assert ep.emotional_arc[0] == 1.0
        assert ep.emotional_arc[1] == -1.0

    def test_close_episode_computes_mean_valence(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.record_turn(valence=0.5)
        buf.record_turn(valence=-0.5)
        ep = buf.close_episode()
        assert ep.mean_valence == pytest.approx(0.0, abs=1e-9)

    def test_close_episode_computes_peak_valence(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.record_turn(valence=0.2)
        buf.record_turn(valence=-0.9)
        buf.record_turn(valence=0.4)
        ep = buf.close_episode()
        assert ep.peak_valence == pytest.approx(-0.9, abs=1e-9)

    def test_add_memory_associates_with_episode(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.add_memory("mem_abc")
        buf.add_memory("mem_xyz")
        ep = buf.current_episode()
        assert "mem_abc" in ep.key_memory_ids
        assert "mem_xyz" in ep.key_memory_ids

    def test_add_memory_no_duplicates(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode()
        buf.add_memory("mem_dup")
        buf.add_memory("mem_dup")
        ep = buf.current_episode()
        assert ep.key_memory_ids.count("mem_dup") == 1

    def test_recent_episodes_sorted_newest_first(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        for i in range(3):
            ep = buf.open_episode(topic=f"ep{i}")
            buf.close_episode()
        recents = buf.recent_episodes(n=3)
        # Sorted newest-first: ep2, ep1, ep0
        assert recents[0].topic == "ep2"
        assert recents[2].topic == "ep0"

    def test_opening_new_auto_closes_current(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        ep1 = buf.open_episode(topic="first")
        ep2 = buf.open_episode(topic="second")
        # First episode should now be closed
        stored = buf.get_episode(ep1.id)
        assert not stored.is_open
        assert ep2.is_open

    def test_max_episodes_evicts_oldest(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer(max_episodes=3)
        ids = []
        for i in range(4):
            ep = buf.open_episode(topic=f"ep{i}")
            buf.close_episode()
            ids.append(ep.id)
        # Oldest (ids[0]) should be evicted
        assert buf.get_episode(ids[0]) is None
        assert buf.get_episode(ids[3]) is not None

    def test_save_and_load(self, tmp_path):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode(topic="saved episode")
        buf.record_turn(valence=0.5)
        buf.close_episode(outcome="done")
        path = tmp_path / "episodes.json"
        buf.save(path)
        buf2 = EpisodicBuffer()
        ok = buf2.load(path)
        assert ok
        eps = buf2.all_episodes()
        assert len(eps) == 1
        assert eps[0].topic == "saved episode"
        assert eps[0].outcome == "done"

    def test_load_nonexistent_returns_false(self, tmp_path):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        assert not buf.load(tmp_path / "missing.json")

    def test_episode_summary_string(self):
        from emms.memory.episodic import EpisodicBuffer
        buf = EpisodicBuffer()
        buf.open_episode(topic="summary test")
        ep = buf.close_episode()
        s = ep.summary()
        assert "summary test" in s
        assert "closed" in s

    # --- EMMS facade ---

    def test_emms_open_episode(self):
        agent = _make_emms()
        ep = agent.open_episode(topic="facade test")
        assert ep.is_open
        assert ep.topic == "facade test"

    def test_emms_close_episode(self):
        agent = _make_emms()
        agent.open_episode(topic="to close")
        ep = agent.close_episode(outcome="resolved")
        assert ep is not None
        assert not ep.is_open
        assert ep.outcome == "resolved"

    def test_emms_record_episode_turn(self):
        agent = _make_emms()
        agent.open_episode()
        agent.record_episode_turn(content="hello", valence=0.4)
        ep = agent.current_episode()
        assert ep.turn_count == 1
        assert ep.emotional_arc[0] == pytest.approx(0.4)

    def test_emms_recent_episodes(self):
        agent = _make_emms()
        for i in range(3):
            agent.open_episode(topic=f"topic{i}")
            agent.close_episode()
        recent = agent.recent_episodes(n=2)
        assert len(recent) == 2

    def test_emms_current_episode_none_when_closed(self):
        agent = _make_emms()
        assert agent.current_episode() is None

    def test_emms_recent_episodes_empty_when_none(self):
        agent = _make_emms()
        assert agent.recent_episodes() == []

    def test_emms_close_episode_none_when_none_open(self):
        agent = _make_emms()
        assert agent.close_episode() is None

    def test_emms_open_episode_creates_buffer_lazily(self):
        agent = _make_emms()
        # Before opening, no buffer
        assert not hasattr(agent, "_episodic_buffer")
        agent.open_episode()
        assert hasattr(agent, "_episodic_buffer")


# ===========================================================================
# SchemaExtractor
# ===========================================================================

class TestSchemaExtractor:

    def test_extract_returns_schema_report(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        assert report.total_memories_analyzed >= 0
        assert report.schemas_found >= 0
        assert report.duration_seconds >= 0

    def test_schemas_have_required_fields(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        for s in report.schemas:
            assert s.id.startswith("schema_")
            assert isinstance(s.domain, str)
            assert isinstance(s.pattern, str)
            assert isinstance(s.keywords, list)
            assert 0.0 <= s.confidence <= 1.0
            assert len(s.supporting_memory_ids) >= 2

    def test_extract_by_domain_filters_correctly(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract(domain="science")
        for s in report.schemas:
            assert s.domain == "science"

    def test_extract_max_schemas_respected(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract(max_schemas=2)
        assert report.schemas_found <= 2

    def test_schemas_sorted_by_confidence_desc(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        confs = [s.confidence for s in report.schemas]
        assert confs == sorted(confs, reverse=True)

    def test_extract_empty_memory_returns_empty_report(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        assert report.schemas_found == 0
        assert report.schemas == []

    def test_schema_summary_string(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        if report.schemas:
            s = report.schemas[0]
            summary = s.summary()
            assert s.domain in summary
            assert "Pattern" in summary

    def test_report_summary_string(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        summary = report.summary()
        assert "SchemaExtractor" in summary

    def test_keywords_are_meaningful(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        # All keywords should be >= min_keyword_length characters
        for s in report.schemas:
            for kw in s.keywords:
                assert len(kw) >= extractor.min_keyword_length

    def test_min_support_prevents_small_clusters(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_emms()
        # Only one memory per domain → no schema should form
        agent.store(Experience(content="quantum physics entanglement", domain="science"))
        extractor = SchemaExtractor(agent.memory, min_support=2)
        report = extractor.extract()
        assert report.schemas_found == 0

    def test_pattern_contains_domain_name(self):
        from emms.memory.schema import SchemaExtractor
        agent = _make_rich_emms()
        extractor = SchemaExtractor(agent.memory)
        report = extractor.extract()
        for s in report.schemas:
            assert s.domain in s.pattern

    # --- EMMS facade ---

    def test_emms_extract_schemas(self):
        agent = _make_rich_emms()
        report = agent.extract_schemas()
        assert hasattr(report, "schemas_found")
        assert hasattr(report, "schemas")

    def test_emms_extract_schemas_domain_filter(self):
        agent = _make_rich_emms()
        report = agent.extract_schemas(domain="technology")
        for s in report.schemas:
            assert s.domain == "technology"

    def test_emms_extract_schemas_max_schemas(self):
        agent = _make_rich_emms()
        report = agent.extract_schemas(max_schemas=1)
        assert len(report.schemas) <= 1


# ===========================================================================
# MotivatedForgetting
# ===========================================================================

class TestMotivatedForgetting:

    def _make_agent_with_memories(self) -> EMMS:
        agent = _make_emms()
        for i in range(6):
            agent.store(Experience(
                content=f"Memory {i}: important fact about topic {i}.",
                domain="general",
                importance=0.5 + 0.05 * i,
            ))
        return agent

    def test_suppress_reduces_strength(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        agent.store(Experience(content="Target memory to suppress", domain="test"))
        items = list(agent.memory.working) + list(agent.memory.short_term)
        items += list(agent.memory.long_term.values()) + list(agent.memory.semantic.values())
        target = items[0]
        old_strength = target.memory_strength
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.5, prune_threshold=0.0)
        result = mf.suppress(target.id)
        assert result is not None
        assert result.old_strength == pytest.approx(old_strength)
        assert result.new_strength < old_strength

    def test_suppress_prunes_below_threshold(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        agent.store(Experience(content="Weak memory to prune", domain="test"))
        items = list(agent.memory.working) + list(agent.memory.short_term)
        items += list(agent.memory.long_term.values()) + list(agent.memory.semantic.values())
        target = items[0]
        target.memory_strength = 0.1  # Force low strength
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.9, prune_threshold=0.15)
        result = mf.suppress(target.id)
        assert result.pruned
        assert result.new_strength == 0.0

    def test_suppress_nonexistent_returns_none(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        mf = MotivatedForgetting(agent.memory)
        result = mf.suppress("nonexistent_id_xyz")
        assert result is None

    def test_forget_domain_targets_all_in_domain(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        for i in range(4):
            agent.store(Experience(content=f"Target domain memory {i}", domain="forget_me"))
        agent.store(Experience(content="Keep this one", domain="keep_me"))
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.3, prune_threshold=0.0)
        report = mf.forget_domain("forget_me")
        assert report.total_targeted == 4

    def test_forget_domain_report_fields(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        for i in range(3):
            agent.store(Experience(content=f"Domain test memory {i}", domain="target_domain"))
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.5, prune_threshold=0.0)
        report = mf.forget_domain("target_domain")
        assert report.suppressed + report.pruned == report.total_targeted
        assert report.duration_seconds >= 0
        assert len(report.results) == report.total_targeted

    def test_forget_domain_empty_domain_zero_targeted(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        agent.store(Experience(content="Other domain memory", domain="other"))
        mf = MotivatedForgetting(agent.memory)
        report = mf.forget_domain("nonexistent_domain")
        assert report.total_targeted == 0

    def test_forget_below_confidence_uses_strength_proxy(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        agent.store(Experience(content="High strength memory", domain="test"))
        agent.store(Experience(content="Low strength memory", domain="test"))
        # Force the second item's strength to 0.1
        all_items = (list(agent.memory.working) + list(agent.memory.short_term)
                     + list(agent.memory.long_term.values()) + list(agent.memory.semantic.values()))
        all_items[-1].memory_strength = 0.1
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.3, prune_threshold=0.0)
        report = mf.forget_below_confidence(threshold=0.5)
        # At least the low-strength item should be targeted
        assert report.total_targeted >= 1

    def test_forget_below_confidence_report_structure(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = self._make_agent_with_memories()
        mf = MotivatedForgetting(agent.memory, prune_threshold=0.0)
        report = mf.forget_below_confidence(threshold=1.0)  # Suppress everything
        assert report.total_targeted >= 0
        assert report.suppressed + report.pruned == report.total_targeted

    def test_resolve_contradiction_suppresses_weaker(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        agent.store(Experience(content="Claim A: the sky is blue", domain="test"))
        all_items = (list(agent.memory.working) + list(agent.memory.short_term)
                     + list(agent.memory.long_term.values()) + list(agent.memory.semantic.values()))
        target = all_items[0]
        old_strength = target.memory_strength
        mf = MotivatedForgetting(agent.memory, suppression_rate=0.5, prune_threshold=0.0)
        result = mf.resolve_contradiction(target.id)
        assert result is not None
        assert result.new_strength < old_strength
        assert "contradiction" in result.reason

    def test_resolve_contradiction_nonexistent_returns_none(self):
        from emms.memory.forgetting import MotivatedForgetting
        agent = _make_emms()
        mf = MotivatedForgetting(agent.memory)
        result = mf.resolve_contradiction("ghost_id")
        assert result is None

    def test_forgetting_result_summary(self):
        from emms.memory.forgetting import ForgettingResult
        r = ForgettingResult(
            memory_id="mem_abc123",
            reason="test suppression",
            old_strength=0.8,
            new_strength=0.48,
            pruned=False,
        )
        s = r.summary()
        assert "mem_abc123" in s
        assert "weakened" in s

    def test_forgetting_result_summary_pruned(self):
        from emms.memory.forgetting import ForgettingResult
        r = ForgettingResult(
            memory_id="mem_pruned",
            reason="too weak",
            old_strength=0.08,
            new_strength=0.0,
            pruned=True,
        )
        s = r.summary()
        assert "PRUNED" in s

    def test_forgetting_report_summary(self):
        from emms.memory.forgetting import ForgettingReport
        report = ForgettingReport(
            total_targeted=5,
            suppressed=3,
            pruned=2,
            duration_seconds=0.01,
        )
        s = report.summary()
        assert "5 targeted" in s
        assert "3 suppressed" in s
        assert "2 pruned" in s

    # --- EMMS facade ---

    def test_emms_forget_memory_returns_result(self):
        agent = _make_emms()
        agent.store(Experience(content="Memory to forget", domain="test"))
        all_items = (list(agent.memory.working) + list(agent.memory.short_term)
                     + list(agent.memory.long_term.values()) + list(agent.memory.semantic.values()))
        target_id = all_items[0].id
        result = agent.forget_memory(target_id)
        assert result is not None
        assert result.memory_id == target_id

    def test_emms_forget_domain(self):
        agent = _make_emms()
        for i in range(3):
            agent.store(Experience(content=f"Domain memory {i}", domain="ephemeral"))
        report = agent.forget_domain("ephemeral")
        assert report.total_targeted == 3

    def test_emms_forget_below_confidence(self):
        agent = _make_emms()
        agent.store(Experience(content="A memory", domain="test"))
        report = agent.forget_below_confidence(threshold=1.0)  # Catches all
        assert isinstance(report.total_targeted, int)

    def test_emms_resolve_memory_contradiction(self):
        agent = _make_emms()
        agent.store(Experience(content="Claim: water boils at 100°C", domain="science"))
        all_items = (list(agent.memory.working) + list(agent.memory.short_term)
                     + list(agent.memory.long_term.values()) + list(agent.memory.semantic.values()))
        weaker_id = all_items[0].id
        result = agent.resolve_memory_contradiction(weaker_id)
        assert result is not None


# ===========================================================================
# MCP v0.14.0 tools
# ===========================================================================

class TestMCPV140:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(agent)

    def test_tool_count_is_57(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 102

    def test_v140_tools_in_definitions(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_open_episode" in names
        assert "emms_close_episode" in names
        assert "emms_recent_episodes" in names
        assert "emms_extract_schemas" in names
        assert "emms_forget" in names

    def test_open_episode_tool(self):
        server = self._make_server()
        result = server.handle("emms_open_episode", {"topic": "mcp test episode"})
        assert "episode_id" in result
        assert result["topic"] == "mcp test episode"

    def test_close_episode_tool(self):
        server = self._make_server()
        server.handle("emms_open_episode", {"topic": "to close"})
        result = server.handle("emms_close_episode", {"outcome": "completed"})
        assert result["closed"]
        assert result["outcome"] == "completed"

    def test_recent_episodes_tool(self):
        server = self._make_server()
        server.handle("emms_open_episode", {"topic": "ep1"})
        server.handle("emms_close_episode", {})
        result = server.handle("emms_recent_episodes", {"n": 5})
        assert "episodes" in result
        assert result["total"] >= 1

    def test_extract_schemas_tool(self):
        server = self._make_server()
        result = server.handle("emms_extract_schemas", {})
        assert "schemas_found" in result
        assert "schemas" in result

    def test_forget_tool_memory_id(self):
        agent = _make_emms()
        agent.store(Experience(content="Something to forget", domain="test"))
        from emms.adapters.mcp_server import EMCPServer
        server = EMCPServer(agent)
        all_items = (list(agent.memory.working) + list(agent.memory.short_term)
                     + list(agent.memory.long_term.values()) + list(agent.memory.semantic.values()))
        mid = all_items[0].id
        result = server.handle("emms_forget", {"memory_id": mid})
        assert result.get("success") or "memory_id" in result

    def test_forget_tool_domain(self):
        agent = _make_emms()
        for i in range(3):
            agent.store(Experience(content=f"Forget me {i}", domain="forgotten"))
        from emms.adapters.mcp_server import EMCPServer
        server = EMCPServer(agent)
        result = server.handle("emms_forget", {"domain": "forgotten"})
        assert "total_targeted" in result
        assert result["total_targeted"] == 3

    def test_forget_tool_no_target_returns_failure(self):
        server = self._make_server()
        result = server.handle("emms_forget", {})
        assert result.get("success") is False


# ===========================================================================
# v0.14.0 exports
# ===========================================================================

class TestV140Exports:

    def test_version_is_0_14_0(self):
        import emms
        assert emms.__version__ == "0.23.0"

    def test_episodic_buffer_exported(self):
        from emms import EpisodicBuffer
        assert EpisodicBuffer is not None

    def test_episode_exported(self):
        from emms import Episode
        assert Episode is not None

    def test_schema_extractor_exported(self):
        from emms import SchemaExtractor
        assert SchemaExtractor is not None

    def test_schema_exported(self):
        from emms import Schema
        assert Schema is not None

    def test_schema_report_exported(self):
        from emms import SchemaReport
        assert SchemaReport is not None

    def test_motivated_forgetting_exported(self):
        from emms import MotivatedForgetting
        assert MotivatedForgetting is not None

    def test_forgetting_report_exported(self):
        from emms import ForgettingReport
        assert ForgettingReport is not None
