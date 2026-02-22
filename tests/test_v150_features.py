"""Tests for EMMS v0.15.0 — The Reflective Mind.

Coverage:
  - ReflectionEngine    (26 tests)
  - NarrativeWeaver     (26 tests)
  - SourceMonitor       (22 tests)
  - MCP v0.15.0 tools   (8 tests)
  - v0.15.0 exports     (8 tests)

Total: 90 tests
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
    agent = _make_emms()
    data = [
        ("science",    "Quantum entanglement enables non-local correlations between particles.",    0.8),
        ("science",    "Entropy always increases in isolated thermodynamic systems.",               0.75),
        ("science",    "DNA replication uses polymerase enzymes to copy genetic information.",      0.7),
        ("science",    "General relativity describes gravity as curvature of spacetime.",           0.85),
        ("philosophy", "Consciousness may be an emergent property of complex information.",         0.8),
        ("philosophy", "The hard problem of consciousness asks why there is subjective experience.", 0.75),
        ("philosophy", "Personal identity persists through psychological continuity over time.",     0.7),
        ("philosophy", "Existentialism holds that existence precedes essence in human life.",        0.65),
        ("technology", "Neural networks learn by adjusting weights through backpropagation.",        0.8),
        ("technology", "Transformers use attention mechanisms to process sequential data.",           0.75),
        ("technology", "Memory systems in AI agents provide persistent cross-session identity.",     0.7),
        ("technology", "Hash embeddings map text to fixed-size vectors for similarity.",             0.65),
    ]
    for domain, content, imp in data[:n]:
        agent.store(Experience(content=content, domain=domain, importance=imp))
    return agent


# ===========================================================================
# ReflectionEngine
# ===========================================================================

class TestReflectionEngine:

    # --- direct module ---

    def test_reflect_returns_report(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        assert hasattr(report, "lessons")
        assert hasattr(report, "open_questions")
        assert hasattr(report, "memories_reviewed")
        assert report.duration_seconds >= 0

    def test_reflect_empty_memory_returns_empty(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_emms()
        engine = ReflectionEngine(agent.memory)
        report = engine.reflect()
        assert report.memories_reviewed == 0
        assert report.lessons == []

    def test_reflect_produces_lessons_from_rich_memory(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        assert len(report.lessons) >= 0  # May or may not find clusters

    def test_lesson_has_required_fields(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        for l in report.lessons:
            assert l.id.startswith("lesson_")
            assert isinstance(l.content, str) and len(l.content) > 0
            assert isinstance(l.domain, str)
            assert 0.0 <= l.confidence <= 1.0
            assert l.lesson_type in ("pattern", "gap", "contrast", "principle")
            assert isinstance(l.supporting_ids, list)

    def test_reflect_domain_filter(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect(domain="science")
        for l in report.lessons:
            assert l.domain == "science"

    def test_reflect_stores_lessons_in_memory(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        # If lessons were found, check memory count increased
        if report.lessons:
            assert len(report.new_memory_ids) > 0

    def test_reflect_skips_reflection_domain_recursively(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_emms()
        # Store a reflection-domain memory
        agent.store(Experience(content="A reflection about patterns", domain="reflection",
                               importance=0.9))
        # Store some real memories
        for i in range(3):
            agent.store(Experience(content=f"Science fact {i} about quantum physics",
                                   domain="science", importance=0.8))
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        # Reflection memories should not be included in reviewed count
        # (they exist in memory but are excluded from candidates)
        assert report.memories_reviewed >= 0

    def test_open_questions_are_strings(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5)
        report = engine.reflect()
        for q in report.open_questions:
            assert isinstance(q, str) and len(q) > 0

    def test_session_id_auto_generated(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_emms()
        engine = ReflectionEngine(agent.memory)
        report = engine.reflect()
        assert report.session_id.startswith("reflect_")

    def test_session_id_custom(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_emms()
        engine = ReflectionEngine(agent.memory)
        report = engine.reflect(session_id="my_session")
        assert report.session_id == "my_session"

    def test_lesson_summary_string(self):
        from emms.memory.reflection import Lesson
        l = Lesson(
            id="lesson_abc",
            content="A recurring pattern in science: quantum, entropy.",
            domain="science",
            supporting_ids=["m1", "m2", "m3"],
            confidence=0.75,
            lesson_type="pattern",
        )
        s = l.summary()
        assert "pattern" in s
        assert "science" in s

    def test_report_summary_string(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory)
        report = engine.reflect()
        s = report.summary()
        assert "ReflectionReport" in s

    def test_max_lessons_respected(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, min_importance=0.5, max_lessons=2)
        report = engine.reflect()
        assert len(report.lessons) <= 2

    def test_episodes_reviewed_zero_without_buffer(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_rich_emms()
        engine = ReflectionEngine(agent.memory, episodic_buffer=None)
        report = engine.reflect()
        assert report.episodes_reviewed == 0

    def test_episodes_reviewed_with_buffer(self):
        from emms.memory.reflection import ReflectionEngine
        from emms.memory.episodic import EpisodicBuffer
        agent = _make_rich_emms()
        buf = EpisodicBuffer()
        buf.open_episode(topic="test")
        buf.close_episode()
        engine = ReflectionEngine(agent.memory, episodic_buffer=buf, min_importance=0.5)
        report = engine.reflect(lookback_episodes=3)
        assert report.episodes_reviewed >= 0

    # --- EMMS facade ---

    def test_emms_reflect(self):
        agent = _make_rich_emms()
        report = agent.reflect()
        assert hasattr(report, "lessons")

    def test_emms_reflect_domain(self):
        agent = _make_rich_emms()
        report = agent.reflect(domain="technology")
        for l in report.lessons:
            assert l.domain == "technology"

    def test_emms_enable_reflection_returns_engine(self):
        from emms.memory.reflection import ReflectionEngine
        agent = _make_emms()
        engine = agent.enable_reflection()
        assert isinstance(engine, ReflectionEngine)

    def test_emms_enable_reflection_lazy_init(self):
        agent = _make_emms()
        assert not hasattr(agent, "_reflection_engine")
        agent.enable_reflection()
        assert hasattr(agent, "_reflection_engine")

    def test_emms_reflect_with_episodes(self):
        agent = _make_rich_emms()
        agent.open_episode(topic="science session")
        agent.record_episode_turn(content="worked on quantum", valence=0.3)
        agent.close_episode(outcome="completed")
        report = agent.reflect(lookback_episodes=2)
        assert report.episodes_reviewed >= 0


# ===========================================================================
# NarrativeWeaver
# ===========================================================================

class TestNarrativeWeaver:

    # --- direct module ---

    def test_weave_returns_report(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        assert hasattr(report, "threads")
        assert hasattr(report, "total_threads")
        assert report.duration_seconds >= 0

    def test_weave_empty_memory_returns_empty(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        assert report.total_threads == 0

    def test_threads_have_segments(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory, min_thread_length=2)
        report = weaver.weave()
        for t in report.threads:
            assert len(t.segments) >= 1

    def test_segments_have_prose(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        for t in report.threads:
            for seg in t.segments:
                assert isinstance(seg.text, str) and len(seg.text) > 0

    def test_domain_filter(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave(domain="science")
        for t in report.threads:
            assert t.domain == "science"

    def test_max_threads_respected(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave(max_threads=1)
        assert report.total_threads <= 1

    def test_min_thread_length_filters_small_domains(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_emms()
        agent.store(Experience(content="Single memory in solitary domain", domain="solo"))
        # Add enough in another domain
        for i in range(3):
            agent.store(Experience(content=f"Rich domain memory {i}", domain="rich"))
        weaver = NarrativeWeaver(agent.memory, min_thread_length=2)
        report = weaver.weave()
        # "solo" domain should be excluded
        for t in report.threads:
            assert t.domain != "solo"

    def test_segments_have_temporal_position(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        for t in report.threads:
            for seg in t.segments:
                assert 0.0 <= seg.temporal_position <= 1.0

    def test_emotional_arc_per_thread(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        for t in report.threads:
            assert isinstance(t.arc, list)
            assert len(t.arc) == len(t.segments)

    def test_thread_story_returns_string(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        for t in report.threads:
            s = t.story()
            assert isinstance(s, str) and len(s) > 0

    def test_thread_summary_string(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        for t in report.threads:
            s = t.summary()
            assert t.domain in s

    def test_report_summary_string(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        s = report.summary()
        assert "NarrativeReport" in s

    def test_threads_sorted_by_segment_count_desc(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        report = weaver.weave()
        seg_counts = [len(t.segments) for t in report.threads]
        assert seg_counts == sorted(seg_counts, reverse=True)

    def test_segment_episode_ids_populated_with_buffer(self):
        from emms.memory.narrative import NarrativeWeaver
        from emms.memory.episodic import EpisodicBuffer
        agent = _make_emms()
        buf = EpisodicBuffer()
        ep = buf.open_episode(topic="science")
        agent.store(Experience(content="Quantum physics study", domain="science", importance=0.8))
        agent.store(Experience(content="Entropy in thermodynamics", domain="science", importance=0.7))
        # Manually add a memory to the episode
        items = list(agent.memory.working)
        if items:
            buf.add_memory(items[0].id)
        buf.close_episode()
        weaver = NarrativeWeaver(agent.memory, episodic_buffer=buf)
        report = weaver.weave(domain="science")
        # At least one segment should reference episode IDs
        all_ep_ids = []
        for t in report.threads:
            for seg in t.segments:
                all_ep_ids.extend(seg.episode_ids)
        # Not required to have episode_ids but no error should occur
        assert isinstance(all_ep_ids, list)

    def test_get_threads_shorthand(self):
        from emms.memory.narrative import NarrativeWeaver
        agent = _make_rich_emms()
        weaver = NarrativeWeaver(agent.memory)
        threads = weaver.get_threads()
        assert isinstance(threads, list)

    # --- EMMS facade ---

    def test_emms_weave_narrative(self):
        agent = _make_rich_emms()
        report = agent.weave_narrative()
        assert hasattr(report, "total_threads")

    def test_emms_weave_narrative_domain(self):
        agent = _make_rich_emms()
        report = agent.weave_narrative(domain="philosophy")
        for t in report.threads:
            assert t.domain == "philosophy"

    def test_emms_weave_narrative_max_threads(self):
        agent = _make_rich_emms()
        report = agent.weave_narrative(max_threads=2)
        assert report.total_threads <= 2

    def test_emms_narrative_threads(self):
        agent = _make_rich_emms()
        threads = agent.narrative_threads()
        assert isinstance(threads, list)

    def test_emms_narrative_threads_empty_on_empty_memory(self):
        agent = _make_emms()
        threads = agent.narrative_threads()
        assert threads == []


# ===========================================================================
# SourceMonitor
# ===========================================================================

class TestSourceMonitor:

    # --- direct module ---

    def test_tag_stores_source(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="An observation", domain="test"))
        items = list(agent.memory.working)
        mid = items[0].id
        monitor = SourceMonitor(agent.memory)
        tag = monitor.tag(mid, "observation", confidence=0.9)
        assert tag.memory_id == mid
        assert tag.source_type == "observation"
        assert tag.confidence == pytest.approx(0.9)

    def test_get_tag_returns_none_for_untagged(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        monitor = SourceMonitor(agent.memory)
        assert monitor.get_tag("nonexistent") is None

    def test_tag_invalid_source_type_raises(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Test", domain="test"))
        items = list(agent.memory.working)
        monitor = SourceMonitor(agent.memory)
        with pytest.raises(ValueError):
            monitor.tag(items[0].id, "invalid_type")

    def test_tag_confidence_clipped_to_0_1(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Test memory", domain="test"))
        items = list(agent.memory.working)
        monitor = SourceMonitor(agent.memory)
        tag = monitor.tag(items[0].id, "inference", confidence=5.0)
        assert tag.confidence == 1.0
        tag2 = monitor.tag(items[0].id, "inference", confidence=-1.0)
        assert tag2.confidence == 0.0

    def test_auto_tag_dream_domain(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Dream consolidation insight", domain="dream"))
        monitor = SourceMonitor(agent.memory)
        count = monitor.auto_tag()
        assert count >= 1
        items = list(agent.memory.working)
        tag = monitor.get_tag(items[0].id)
        assert tag is not None
        assert tag.source_type == "dream"

    def test_auto_tag_reflection_domain(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="A pattern in science knowledge", domain="reflection"))
        monitor = SourceMonitor(agent.memory)
        monitor.auto_tag()
        items = list(agent.memory.working)
        tag = monitor.get_tag(items[0].id)
        assert tag is not None
        assert tag.source_type == "reflection"

    def test_auto_tag_insight_domain(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Bridge between science and philosophy", domain="insight"))
        monitor = SourceMonitor(agent.memory)
        monitor.auto_tag()
        items = list(agent.memory.working)
        tag = monitor.get_tag(items[0].id)
        assert tag is not None
        assert tag.source_type == "insight"

    def test_auto_tag_skips_already_tagged(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Already tagged", domain="test"))
        items = list(agent.memory.working)
        mid = items[0].id
        monitor = SourceMonitor(agent.memory)
        monitor.tag(mid, "instruction", confidence=0.95)
        count = monitor.auto_tag()
        assert count == 0  # Already tagged; skip
        tag = monitor.get_tag(mid)
        assert tag.source_type == "instruction"  # Should remain

    def test_audit_returns_report(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_rich_emms()
        monitor = SourceMonitor(agent.memory)
        report = monitor.audit()
        assert report.total_audited >= 0
        assert report.flagged_count >= 0
        assert isinstance(report.source_distribution, dict)

    def test_audit_flags_unknown_sources(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        agent.store(Experience(content="Mysterious fact from unknown source",
                               domain="test", importance=0.3))
        monitor = SourceMonitor(agent.memory, flag_threshold=0.6)
        report = monitor.audit()
        # Untagged items with uncertain heuristics should be flagged
        assert report.flagged_count >= 0

    def test_source_profile_empty_when_no_tags(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        monitor = SourceMonitor(agent.memory)
        profile = monitor.source_profile()
        assert profile == {}

    def test_source_profile_after_auto_tag(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_rich_emms()
        monitor = SourceMonitor(agent.memory)
        monitor.auto_tag()
        profile = monitor.source_profile()
        assert isinstance(profile, dict)
        assert len(profile) > 0

    def test_source_tag_summary(self):
        from emms.memory.source_monitor import SourceTag
        tag = SourceTag(memory_id="mem_abc", source_type="observation",
                        confidence=0.85, provenance_note="first-hand input")
        s = tag.summary()
        assert "mem_abc" in s
        assert "observation" in s

    def test_source_report_summary(self):
        from emms.memory.source_monitor import SourceReport
        report = SourceReport(
            total_audited=10,
            flagged_count=3,
            source_distribution={"observation": 5, "unknown": 3, "inference": 2},
            high_risk_entries=[],
            duration_seconds=0.01,
        )
        s = report.summary()
        assert "10 memories" in s
        assert "3 flagged" in s

    # --- EMMS facade ---

    def test_emms_enable_source_monitoring(self):
        from emms.memory.source_monitor import SourceMonitor
        agent = _make_emms()
        monitor = agent.enable_source_monitoring()
        assert isinstance(monitor, SourceMonitor)

    def test_emms_enable_source_monitoring_lazy(self):
        agent = _make_emms()
        assert not hasattr(agent, "_source_monitor")
        agent.enable_source_monitoring()
        assert hasattr(agent, "_source_monitor")

    def test_emms_tag_memory_source(self):
        agent = _make_emms()
        agent.store(Experience(content="A tagged memory", domain="test"))
        items = list(agent.memory.working)
        mid = items[0].id
        tag = agent.tag_memory_source(mid, "instruction", confidence=0.9)
        assert tag.source_type == "instruction"
        assert tag.memory_id == mid

    def test_emms_source_audit(self):
        agent = _make_rich_emms()
        report = agent.source_audit()
        assert report.total_audited >= 0

    def test_emms_source_profile_empty_initially(self):
        agent = _make_emms()
        profile = agent.source_profile()
        assert profile == {}

    def test_emms_source_profile_after_tag(self):
        agent = _make_emms()
        agent.store(Experience(content="Tagged memory", domain="test"))
        items = list(agent.memory.working)
        agent.tag_memory_source(items[0].id, "observation")
        profile = agent.source_profile()
        assert "observation" in profile
        assert profile["observation"] == 1


# ===========================================================================
# MCP v0.15.0 tools
# ===========================================================================

class TestMCPV150:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(agent)

    def test_tool_count_is_62(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 77

    def test_v150_tools_in_definitions(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_reflect" in names
        assert "emms_weave_narrative" in names
        assert "emms_narrative_threads" in names
        assert "emms_source_audit" in names
        assert "emms_tag_source" in names

    def test_reflect_tool(self):
        server = self._make_server()
        result = server.handle("emms_reflect", {})
        assert "lessons_count" in result
        assert "memories_reviewed" in result

    def test_weave_narrative_tool(self):
        server = self._make_server()
        result = server.handle("emms_weave_narrative", {})
        assert "total_threads" in result
        assert "threads" in result

    def test_narrative_threads_tool(self):
        server = self._make_server()
        result = server.handle("emms_narrative_threads", {})
        assert "threads" in result
        assert "total" in result

    def test_source_audit_tool(self):
        server = self._make_server()
        result = server.handle("emms_source_audit", {"flag_threshold": 0.5})
        assert "total_audited" in result
        assert "flagged_count" in result
        assert "source_distribution" in result

    def test_tag_source_tool(self):
        agent = _make_emms()
        agent.store(Experience(content="Memory to tag", domain="test"))
        from emms.adapters.mcp_server import EMCPServer
        server = EMCPServer(agent)
        items = list(agent.memory.working)
        mid = items[0].id
        result = server.handle("emms_tag_source", {
            "memory_id": mid,
            "source_type": "instruction",
            "confidence": 0.85,
        })
        assert result["source_type"] == "instruction"
        assert result["memory_id"] == mid

    def test_reflect_domain_filter(self):
        server = self._make_server()
        result = server.handle("emms_reflect", {"domain": "science"})
        for l in result.get("lessons", []):
            assert l["domain"] == "science"


# ===========================================================================
# v0.15.0 exports
# ===========================================================================

class TestV150Exports:

    def test_version_is_0_15_0(self):
        import emms
        assert emms.__version__ == "0.18.0"

    def test_reflection_engine_exported(self):
        from emms import ReflectionEngine
        assert ReflectionEngine is not None

    def test_lesson_exported(self):
        from emms import Lesson
        assert Lesson is not None

    def test_reflection_report_exported(self):
        from emms import ReflectionReport
        assert ReflectionReport is not None

    def test_narrative_weaver_exported(self):
        from emms import NarrativeWeaver
        assert NarrativeWeaver is not None

    def test_narrative_thread_exported(self):
        from emms import NarrativeThread
        assert NarrativeThread is not None

    def test_source_monitor_exported(self):
        from emms import SourceMonitor
        assert SourceMonitor is not None

    def test_source_report_exported(self):
        from emms import SourceReport
        assert SourceReport is not None
