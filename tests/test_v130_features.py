"""Tests for EMMS v0.13.0 — The Metacognitive Layer.

Coverage:
  - MetacognitionEngine        (26 tests)
  - ProspectiveMemory          (24 tests)
  - ContextualSalienceRetriever(22 tests)
  - MCP v0.13.0 tools          (10 tests)
  - v0.13.0 exports            (10 tests)

Total: 92 tests
"""

from __future__ import annotations

import json
import time
import pytest

from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS(enable_consciousness=False)


def _make_multi_domain(n: int = 12) -> EMMS:
    agent = _make_emms()
    data = [
        ("science",    "Quantum mechanics describes particle behaviour at subatomic scales."),
        ("science",    "DNA contains the genetic blueprint using four nucleotide bases."),
        ("science",    "Thermodynamics governs heat transfer and entropy in physical systems."),
        ("science",    "General relativity links gravity to the curvature of spacetime."),
        ("philosophy", "Consciousness may emerge from complex information processing patterns."),
        ("philosophy", "Personal identity persists through psychological continuity over time."),
        ("philosophy", "Free will debates whether determinism is compatible with agency."),
        ("philosophy", "Existentialism holds that existence precedes essence in human life."),
        ("technology", "Neural networks learn by adjusting weights via backpropagation."),
        ("technology", "Transformers use self-attention to model sequential relationships."),
        ("technology", "EMMS provides persistent cross-session identity for AI agents."),
        ("technology", "Hash embeddings map text to compact fixed-dimensional vectors."),
    ]
    for content, domain in data[:n]:
        agent.store(Experience(
            content=content, domain=domain, importance=0.7,
            emotional_valence=0.1, emotional_intensity=0.5,
        ))
    return agent


def _make_conflicting_emms() -> EMMS:
    """EMMS with memories that have semantic overlap but opposing valence."""
    agent = _make_emms()
    agent.store(Experience(
        content="AI memory systems are wonderful and beneficial for society.",
        domain="tech", importance=0.7, emotional_valence=0.8,
    ))
    agent.store(Experience(
        content="AI memory systems are dangerous and harmful for society.",
        domain="tech", importance=0.7, emotional_valence=-0.8,
    ))
    agent.store(Experience(
        content="Machine learning models are reliable and trustworthy.",
        domain="tech", importance=0.6, emotional_valence=0.7,
    ))
    agent.store(Experience(
        content="Machine learning models are unreliable and untrustworthy.",
        domain="tech", importance=0.6, emotional_valence=-0.7,
    ))
    return agent


# ---------------------------------------------------------------------------
# MetacognitionEngine tests
# ---------------------------------------------------------------------------

class TestMetacognitionImport:
    def test_import_metacognition_engine(self):
        from emms.memory.metacognition import MetacognitionEngine
        assert MetacognitionEngine is not None

    def test_import_memory_confidence(self):
        from emms.memory.metacognition import MemoryConfidence
        assert MemoryConfidence is not None

    def test_import_domain_profile(self):
        from emms.memory.metacognition import DomainProfile
        assert DomainProfile is not None

    def test_import_contradiction_pair(self):
        from emms.memory.metacognition import ContradictionPair
        assert ContradictionPair is not None

    def test_import_metacognition_report(self):
        from emms.memory.metacognition import MetacognitionReport
        assert MetacognitionReport is not None


class TestMetacognitionEngineAssess:
    def test_assess_single_memory_fields(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=4)
        engine = MetacognitionEngine(agent.memory)
        items = engine._collect_all()
        assert items, "no items"
        conf = engine.assess(items[0])
        assert hasattr(conf, "memory_id")
        assert hasattr(conf, "confidence")
        assert hasattr(conf, "strength_factor")
        assert hasattr(conf, "recency_factor")
        assert hasattr(conf, "access_factor")
        assert hasattr(conf, "consolidation_factor")
        assert hasattr(conf, "age_days")

    def test_assess_confidence_in_range(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=4)
        engine = MetacognitionEngine(agent.memory)
        for item in engine._collect_all():
            conf = engine.assess(item)
            assert 0.0 <= conf.confidence <= 1.0

    def test_assess_factors_in_range(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=4)
        engine = MetacognitionEngine(agent.memory)
        for item in engine._collect_all()[:5]:
            conf = engine.assess(item)
            assert 0.0 <= conf.strength_factor <= 1.0
            assert 0.0 <= conf.recency_factor <= 1.0
            assert 0.0 <= conf.access_factor <= 1.0
            assert 0.0 <= conf.consolidation_factor <= 1.0

    def test_assess_all_returns_sorted_descending(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=6)
        engine = MetacognitionEngine(agent.memory)
        all_conf = engine.assess_all()
        assert isinstance(all_conf, list)
        for i in range(len(all_conf) - 1):
            assert all_conf[i].confidence >= all_conf[i + 1].confidence

    def test_assess_all_count_matches_memories(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=6)
        engine = MetacognitionEngine(agent.memory)
        items = engine._collect_all()
        all_conf = engine.assess_all()
        assert len(all_conf) == len(items)

    def test_token_overlap_identical(self):
        from emms.memory.metacognition import MetacognitionEngine
        overlap = MetacognitionEngine._token_overlap("hello world", "hello world")
        assert abs(overlap - 1.0) < 1e-9

    def test_token_overlap_disjoint(self):
        from emms.memory.metacognition import MetacognitionEngine
        overlap = MetacognitionEngine._token_overlap("foo bar", "baz qux")
        assert overlap == 0.0

    def test_token_overlap_partial(self):
        from emms.memory.metacognition import MetacognitionEngine
        overlap = MetacognitionEngine._token_overlap("hello world foo", "hello earth bar")
        assert 0.0 < overlap < 1.0


class TestMetacognitionKnowledgeMap:
    def test_knowledge_map_empty(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_emms()
        engine = MetacognitionEngine(agent.memory)
        assert engine.knowledge_map() == []

    def test_knowledge_map_fields(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=8)
        engine = MetacognitionEngine(agent.memory)
        profiles = engine.knowledge_map()
        assert len(profiles) > 0
        for p in profiles:
            assert hasattr(p, "domain")
            assert hasattr(p, "memory_count")
            assert hasattr(p, "mean_confidence")
            assert hasattr(p, "coverage_score")
            assert hasattr(p, "mean_importance")
            assert hasattr(p, "mean_strength")

    def test_knowledge_map_coverage_sums_to_one(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=12)
        engine = MetacognitionEngine(agent.memory)
        profiles = engine.knowledge_map()
        total_coverage = sum(p.coverage_score for p in profiles)
        assert abs(total_coverage - 1.0) < 0.01

    def test_knowledge_map_three_domains(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=12)
        engine = MetacognitionEngine(agent.memory)
        profiles = engine.knowledge_map()
        domains = {p.domain for p in profiles}
        assert len(domains) >= 3


class TestMetacognitionContradictions:
    def test_find_contradictions_empty(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_emms()
        engine = MetacognitionEngine(agent.memory)
        assert engine.find_contradictions() == []

    def test_find_contradictions_detects_conflicting_valence(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_conflicting_emms()
        engine = MetacognitionEngine(
            agent.memory,
            contradiction_overlap_min=0.2,
            contradiction_valence_min=0.3,
        )
        pairs = engine.find_contradictions()
        assert len(pairs) >= 1

    def test_contradiction_fields(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_conflicting_emms()
        engine = MetacognitionEngine(
            agent.memory,
            contradiction_overlap_min=0.2,
            contradiction_valence_min=0.3,
        )
        pairs = engine.find_contradictions()
        if pairs:
            p = pairs[0]
            assert hasattr(p, "memory_a_id")
            assert hasattr(p, "memory_b_id")
            assert hasattr(p, "semantic_overlap")
            assert hasattr(p, "valence_conflict")
            assert hasattr(p, "contradiction_score")
            assert hasattr(p, "excerpt_a")
            assert hasattr(p, "excerpt_b")

    def test_contradiction_score_is_positive(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_conflicting_emms()
        engine = MetacognitionEngine(
            agent.memory,
            contradiction_overlap_min=0.2,
            contradiction_valence_min=0.3,
        )
        for p in engine.find_contradictions():
            assert p.contradiction_score > 0


class TestMetacognitionReport:
    def test_report_fields(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=8)
        engine = MetacognitionEngine(agent.memory)
        report = engine.report()
        assert hasattr(report, "total_memories")
        assert hasattr(report, "mean_confidence")
        assert hasattr(report, "high_confidence_count")
        assert hasattr(report, "low_confidence_count")
        assert hasattr(report, "domain_profiles")
        assert hasattr(report, "contradictions")
        assert hasattr(report, "knowledge_gaps")
        assert hasattr(report, "recommendations")

    def test_report_summary_returns_string(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=6)
        engine = MetacognitionEngine(agent.memory)
        assert isinstance(engine.report().summary(), str)

    def test_report_recommendations_not_empty(self):
        from emms.memory.metacognition import MetacognitionEngine
        agent = _make_multi_domain(n=6)
        engine = MetacognitionEngine(agent.memory)
        report = engine.report()
        assert len(report.recommendations) > 0


class TestEMMSMetacognitionFacade:
    def test_metacognition_report_facade(self):
        agent = _make_multi_domain(n=6)
        report = agent.metacognition_report()
        assert hasattr(report, "total_memories")
        assert hasattr(report, "mean_confidence")

    def test_knowledge_map_facade(self):
        agent = _make_multi_domain(n=8)
        profiles = agent.knowledge_map()
        assert isinstance(profiles, list)
        assert len(profiles) > 0

    def test_find_contradictions_facade(self):
        agent = _make_conflicting_emms()
        pairs = agent.find_contradictions()
        assert isinstance(pairs, list)

    def test_assess_memory_facade(self):
        agent = _make_multi_domain(n=4)
        items = list(agent.memory.working) + list(agent.memory.short_term)
        if not items:
            pytest.skip("no items")
        conf = agent.assess_memory(items[0].id)
        assert hasattr(conf, "confidence")
        assert 0.0 <= conf.confidence <= 1.0


# ---------------------------------------------------------------------------
# ProspectiveMemory tests
# ---------------------------------------------------------------------------

class TestProspectiveMemoryImport:
    def test_import_prospective_memory(self):
        from emms.memory.prospection import ProspectiveMemory
        assert ProspectiveMemory is not None

    def test_import_intention(self):
        from emms.memory.prospection import Intention
        assert Intention is not None

    def test_import_intention_activation(self):
        from emms.memory.prospection import IntentionActivation
        assert IntentionActivation is not None


class TestProspectiveMemoryBasic:
    def test_intend_creates_intention(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Follow up on memory work", "memory consolidation next session")
        assert intention.id.startswith("int_")
        assert intention.content == "Follow up on memory work"
        assert not intention.fulfilled

    def test_intend_default_priority(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Do something", "when something happens")
        assert intention.priority == 0.5

    def test_intend_custom_priority(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Urgent task", "critical moment", priority=0.9)
        assert abs(intention.priority - 0.9) < 1e-9

    def test_pending_returns_unfulfilled(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        pm.intend("Task A", "context A", priority=0.8)
        pm.intend("Task B", "context B", priority=0.3)
        pending = pm.pending()
        assert len(pending) == 2

    def test_pending_sorted_by_priority(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        pm.intend("Low", "trigger", priority=0.2)
        pm.intend("High", "trigger", priority=0.9)
        pm.intend("Mid", "trigger", priority=0.5)
        pending = pm.pending()
        for i in range(len(pending) - 1):
            assert pending[i].priority >= pending[i + 1].priority

    def test_fulfill_marks_fulfilled(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Task", "trigger")
        result = pm.fulfill(intention.id)
        assert result is True
        assert intention.fulfilled is True
        assert intention.fulfilled_at is not None

    def test_fulfill_unknown_id_returns_false(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        assert pm.fulfill("nonexistent") is False

    def test_fulfilled_not_in_pending(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Task", "trigger")
        pm.fulfill(intention.id)
        pending = pm.pending()
        assert all(i.id != intention.id for i in pending)

    def test_dismiss_removes_intention(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        intention = pm.intend("Temporary", "trigger")
        result = pm.dismiss(intention.id)
        assert result is True
        assert intention.id not in [i.id for i in pm.all_intentions()]

    def test_check_activates_matching_intentions(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory(overlap_threshold=0.1)
        pm.intend("Discuss memory consolidation", "memory consolidation session", priority=0.8)
        pm.intend("Talk about weather", "outdoor activities weather report")
        activations = pm.check("Let us discuss memory and consolidation processes")
        assert len(activations) >= 1
        assert activations[0].intention.content == "Discuss memory consolidation"

    def test_check_does_not_activate_unrelated(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory(overlap_threshold=0.5)
        pm.intend("Discuss quantum physics", "quantum physics particles entanglement")
        activations = pm.check("Let us talk about cooking recipes today")
        assert len(activations) == 0

    def test_check_increments_activation_count(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory(overlap_threshold=0.1)
        intention = pm.intend("Task", "memory processing recall", priority=0.7)
        pm.check("memory processing system")
        assert intention.activation_count >= 1

    def test_check_excludes_fulfilled(self):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory(overlap_threshold=0.1)
        intention = pm.intend("Done task", "memory recall")
        pm.fulfill(intention.id)
        activations = pm.check("memory recall session")
        assert all(a.intention.id != intention.id for a in activations)

    def test_save_and_load(self, tmp_path):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        pm.intend("Persistent task", "trigger for memory", priority=0.7)
        path = tmp_path / "intentions.json"
        pm.save(path)

        pm2 = ProspectiveMemory()
        loaded = pm2.load(path)
        assert loaded is True
        assert len(pm2.pending()) == 1
        assert pm2.pending()[0].content == "Persistent task"

    def test_load_returns_false_for_missing_file(self, tmp_path):
        from emms.memory.prospection import ProspectiveMemory
        pm = ProspectiveMemory()
        result = pm.load(tmp_path / "nonexistent.json")
        assert result is False

    def test_intention_serialization_roundtrip(self):
        from emms.memory.prospection import Intention
        original = Intention(
            id="int_abc123",
            content="Do something",
            trigger_context="when X happens",
            priority=0.75,
            created_at=time.time(),
            activation_count=3,
        )
        restored = Intention.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.content == original.content
        assert abs(restored.priority - original.priority) < 1e-9
        assert restored.activation_count == original.activation_count


class TestEMMSProspectiveFacade:
    def test_enable_prospective_memory_facade(self):
        agent = _make_emms()
        pm = agent.enable_prospective_memory()
        assert pm is not None

    def test_intend_facade(self):
        agent = _make_emms()
        intention = agent.intend("Follow up on X", "when discussing X")
        assert intention.id.startswith("int_")

    def test_check_intentions_facade_empty(self):
        agent = _make_emms()
        results = agent.check_intentions("some context text")
        assert results == []

    def test_pending_intentions_facade(self):
        agent = _make_emms()
        agent.intend("Task A", "trigger A", priority=0.8)
        agent.intend("Task B", "trigger B", priority=0.3)
        pending = agent.pending_intentions()
        assert len(pending) == 2

    def test_fulfill_intention_facade(self):
        agent = _make_emms()
        intention = agent.intend("Task", "trigger")
        result = agent.fulfill_intention(intention.id)
        assert result is True
        pending = agent.pending_intentions()
        assert all(i.id != intention.id for i in pending)


# ---------------------------------------------------------------------------
# ContextualSalienceRetriever tests
# ---------------------------------------------------------------------------

class TestContextualSalienceImport:
    def test_import_contextual_retriever(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        assert ContextualSalienceRetriever is not None

    def test_import_salience_result(self):
        from emms.retrieval.contextual import SalienceResult
        assert SalienceResult is not None


class TestContextualSalienceRetriever:
    def test_retrieve_empty_context(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=6)
        retriever = ContextualSalienceRetriever(agent.memory)
        # No context updated yet → empty
        results = retriever.retrieve()
        assert results == []

    def test_retrieve_after_update(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=6)
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("quantum physics particles")
        results = retriever.retrieve()
        assert isinstance(results, list)

    def test_retrieve_returns_salience_results(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=6)
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("neural networks machine learning AI")
        results = retriever.retrieve()
        for r in results:
            assert hasattr(r, "memory")
            assert hasattr(r, "salience_score")
            assert hasattr(r, "semantic_overlap")
            assert hasattr(r, "importance_factor")
            assert hasattr(r, "recency_factor")
            assert hasattr(r, "affective_resonance")

    def test_retrieve_sorted_by_salience(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=8)
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("memory identity consciousness agent")
        results = retriever.retrieve()
        for i in range(len(results) - 1):
            assert results[i].salience_score >= results[i + 1].salience_score

    def test_retrieve_max_results_respected(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=12)
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("neural networks learning AI agents")
        results = retriever.retrieve(max_results=3)
        assert len(results) <= 3

    def test_context_window_rolling(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=4)
        retriever = ContextualSalienceRetriever(agent.memory, window_size=3)
        for i in range(5):
            retriever.update_context(f"turn {i} content text")
        # Window should hold only last 3
        assert len(retriever._context_window) == 3

    def test_context_valence_mean(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_emms()
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("positive text", valence=0.8)
        retriever.update_context("negative text", valence=-0.4)
        cv = retriever.context_valence
        assert abs(cv - (0.8 + (-0.4)) / 2) < 1e-9

    def test_context_summary_empty(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_emms()
        retriever = ContextualSalienceRetriever(agent.memory)
        assert retriever.context_summary == "(empty context)"

    def test_context_summary_non_empty(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_emms()
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("hello world")
        assert "hello" in retriever.context_summary or "1 turn" in retriever.context_summary

    def test_reset_context_clears_window(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_emms()
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("some text")
        retriever.reset_context()
        assert retriever.retrieve() == []

    def test_salience_scores_in_range(self):
        from emms.retrieval.contextual import ContextualSalienceRetriever
        agent = _make_multi_domain(n=8)
        retriever = ContextualSalienceRetriever(agent.memory)
        retriever.update_context("consciousness identity philosophy")
        for r in retriever.retrieve():
            assert 0.0 <= r.salience_score <= 1.0


class TestEMMSContextualFacade:
    def test_enable_contextual_retrieval_facade(self):
        agent = _make_emms()
        retriever = agent.enable_contextual_retrieval()
        assert retriever is not None

    def test_update_context_facade(self):
        agent = _make_multi_domain(n=4)
        agent.update_context("neural networks learning", valence=0.3)
        results = agent.contextual_retrieve()
        assert isinstance(results, list)

    def test_contextual_retrieve_returns_results(self):
        agent = _make_multi_domain(n=8)
        agent.update_context("quantum physics thermodynamics science")
        results = agent.contextual_retrieve(max_results=5)
        assert len(results) <= 5

    def test_context_summary_facade_before_enable(self):
        agent = _make_emms()
        summary = agent.context_summary()
        assert isinstance(summary, str)

    def test_context_summary_facade_after_update(self):
        agent = _make_emms()
        agent.update_context("hello world test")
        summary = agent.context_summary()
        assert isinstance(summary, str)
        assert summary != "(contextual retrieval not enabled)"


# ---------------------------------------------------------------------------
# MCP v0.13.0 tools
# ---------------------------------------------------------------------------

class TestMCPv130Tools:
    def test_mcp_tool_count(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 57

    def test_mcp_has_metacognition_report(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_metacognition_report" in names

    def test_mcp_has_knowledge_map(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_knowledge_map" in names

    def test_mcp_has_find_contradictions(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_find_contradictions" in names

    def test_mcp_has_intend(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_intend" in names

    def test_mcp_has_check_intentions(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_check_intentions" in names

    def test_mcp_metacognition_report_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n=6)
        server = EMCPServer(agent)
        result = server.handle("emms_metacognition_report", {})
        assert "total_memories" in result
        assert "mean_confidence" in result
        assert "recommendations" in result

    def test_mcp_knowledge_map_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n=6)
        server = EMCPServer(agent)
        result = server.handle("emms_knowledge_map", {})
        assert "domains" in result
        assert "total_domains" in result

    def test_mcp_intend_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_intend", {
            "content": "Review memory coherence",
            "trigger_context": "memory review session",
        })
        assert "intention_id" in result
        assert result["intention_id"].startswith("int_")

    def test_mcp_check_intentions_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms()
        agent.intend("Check memory system", "memory system review")
        server = EMCPServer(agent)
        result = server.handle("emms_check_intentions", {
            "current_context": "memory system review today",
        })
        assert "activated" in result
        assert "total_pending" in result


# ---------------------------------------------------------------------------
# v0.13.0 exports
# ---------------------------------------------------------------------------

class TestV130Exports:
    def test_version(self):
        import emms
        assert emms.__version__ == "0.14.0"

    def test_export_metacognition_engine(self):
        from emms import MetacognitionEngine
        assert MetacognitionEngine is not None

    def test_export_metacognition_report(self):
        from emms import MetacognitionReport
        assert MetacognitionReport is not None

    def test_export_memory_confidence(self):
        from emms import MemoryConfidence
        assert MemoryConfidence is not None

    def test_export_domain_profile(self):
        from emms import DomainProfile
        assert DomainProfile is not None

    def test_export_contradiction_pair(self):
        from emms import ContradictionPair
        assert ContradictionPair is not None

    def test_export_prospective_memory(self):
        from emms import ProspectiveMemory
        assert ProspectiveMemory is not None

    def test_export_intention(self):
        from emms import Intention
        assert Intention is not None

    def test_export_contextual_retriever(self):
        from emms import ContextualSalienceRetriever
        assert ContextualSalienceRetriever is not None

    def test_export_salience_result(self):
        from emms import SalienceResult
        assert SalienceResult is not None
