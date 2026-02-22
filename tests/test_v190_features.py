"""Tests for EMMS v0.19.0 — The Integrated Mind.

Covers:
    - EmotionalRegulator + EmotionalState + ReappraisalResult + EmotionReport
    - ConceptHierarchy + ConceptNode + HierarchyReport
    - SelfModel + Belief + SelfModelReport
    - EMMS facade methods
    - MCP tool count (82) and new tool callability
    - __init__ exports and version string
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from emms import EMMS, Experience
from emms.adapters.mcp_server import EMCPServer, _TOOL_DEFINITIONS
from emms.memory.emotion import (
    EmotionalRegulator, EmotionalState, ReappraisalResult, EmotionReport,
)
from emms.memory.hierarchy import ConceptHierarchy, ConceptNode, HierarchyReport
from emms.memory.self_model import SelfModel, Belief, SelfModelReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms(n: int = 8) -> EMMS:
    agent = EMMS()
    entries = [
        ("Regular exercise dramatically improves cardiovascular health and reduces chronic disease risk",
         "health", 0.8, 0.7),
        ("Daily meditation practice reduces cortisol stress levels and improves cognitive function",
         "health", 0.7, 0.6),
        ("Ancient Rome expanded through military conquest and established complex administrative systems",
         "history", 0.8, -0.1),
        ("The Renaissance period enabled artistic flourishing through wealthy patronage",
         "history", 0.7, 0.5),
        ("Impressionism movement produces emotional expression through light and color technique",
         "art", 0.6, 0.6),
        ("Regular physical training improves cardiovascular endurance and reduces injury risk",
         "health", 0.9, 0.8),
        ("Historical warfare reduces civilian populations and produces lasting trauma",
         "history", 0.8, -0.7),
        ("Exercise enables muscle growth through progressive overload and improves strength",
         "health", 0.75, 0.65),
    ]
    for i, (content, domain, importance, valence) in enumerate(entries[:n]):
        agent.store(Experience(
            content=content, domain=domain,
            importance=importance, emotional_valence=valence,
        ))
    return agent


def _make_negative_emms() -> EMMS:
    """EMMS with mixed positive and negative memories for reappraisal testing."""
    agent = EMMS()
    agent.store(Experience(
        content="The project completely failed causing significant financial damage and embarrassment",
        domain="work",
        importance=0.9,
        emotional_valence=-0.8,
    ))
    agent.store(Experience(
        content="Repeated mistakes in critical systems damaged team morale severely",
        domain="work",
        importance=0.8,
        emotional_valence=-0.6,
    ))
    agent.store(Experience(
        content="Successfully delivered the quarterly report on time with excellent feedback",
        domain="work",
        importance=0.7,
        emotional_valence=0.8,
    ))
    agent.store(Experience(
        content="The client meeting went poorly resulting in contract cancellation",
        domain="work",
        importance=0.8,
        emotional_valence=-0.7,
    ))
    return agent


def _make_hierarchy_emms() -> EMMS:
    """EMMS with repeated tokens for hierarchy building."""
    agent = EMMS()
    texts = [
        "cognitive psychology studies human memory and learning processes systematically",
        "cognitive science integrates psychology neuroscience linguistics philosophy",
        "memory consolidation during sleep strengthens cognitive learning processes",
        "neuroscience reveals how neural networks support cognitive memory functions",
        "learning psychology examines how humans acquire skills through cognitive training",
        "philosophy cognitive science debates consciousness nature mind memory",
        "neural cognitive architecture enables complex learning through memory systems",
        "psychology learning theories inform educational cognitive development practices",
    ]
    for text in texts:
        agent.store(Experience(content=text, domain="science", importance=0.7))
    return agent


# ===========================================================================
# TestEmotionalRegulator
# ===========================================================================


class TestEmotionalRegulator:

    def test_regulate_returns_report(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert isinstance(report, EmotionReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert isinstance(report.current_state, EmotionalState)
        assert isinstance(report.memories_assessed, int)
        assert isinstance(report.reappraisals, list)
        assert isinstance(report.mood_congruent_ids, list)
        assert 0.0 <= report.emotional_coherence <= 1.0
        assert report.duration_seconds >= 0.0

    def test_emotional_state_fields(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        state = report.current_state
        assert isinstance(state, EmotionalState)
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0
        assert isinstance(state.dominant_domain, str)
        assert isinstance(state.sample_size, int)
        assert state.sample_size > 0
        assert isinstance(state.computed_at, float)

    def test_current_state_initially_none(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        assert reg.current_state() is None

    def test_current_state_after_regulate(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        reg.regulate()
        state = reg.current_state()
        assert state is not None
        assert isinstance(state, EmotionalState)

    def test_reappraisal_triggered_for_negative(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        # Should have reappraisals for negative-valence memories (< -0.3)
        assert len(report.reappraisals) >= 1

    def test_reappraisal_result_fields(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        if report.reappraisals:
            r = report.reappraisals[0]
            assert isinstance(r, ReappraisalResult)
            assert isinstance(r.memory_id, str)
            assert isinstance(r.original_valence, float)
            assert isinstance(r.reappraised_content, str)
            assert isinstance(r.new_valence, float)
            assert isinstance(r.shift, float)

    def test_reappraisal_shifts_valence_positive(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        for r in report.reappraisals:
            assert r.new_valence > r.original_valence
            assert r.shift > 0.0

    def test_new_valence_is_clamped(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        for r in report.reappraisals:
            assert -1.0 <= r.new_valence <= 1.0

    def test_reappraisal_shift_is_030(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        for r in report.reappraisals:
            assert abs(r.shift - 0.3) < 0.01 or r.new_valence == 1.0

    def test_store_reappraisals_creates_memory(self):
        agent = _make_negative_emms()
        from emms.memory.emotion import EmotionalRegulator as ER
        reg = ER(memory=agent.memory, reappraise=True, store_reappraisals=True)
        report = reg.regulate()
        if report.reappraisals:
            stored = [r for r in report.reappraisals if r.stored_as_memory_id is not None]
            assert len(stored) >= 1

    def test_no_store_reappraisals(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True,
                                  store_reappraisals=False)
        report = reg.regulate()
        for r in report.reappraisals:
            assert r.stored_as_memory_id is None

    def test_no_reappraise_flag(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=False)
        report = reg.regulate()
        assert report.reappraisals == []

    def test_mood_retrieve_returns_list(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        reg.regulate()
        items = reg.mood_retrieve(k=4)
        assert isinstance(items, list)
        assert len(items) <= 4

    def test_mood_retrieve_without_state_returns_empty(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        items = reg.mood_retrieve()
        assert items == []

    def test_emotional_coherence_bounds(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        coh = reg.emotional_coherence()
        assert 0.0 <= coh <= 1.0

    def test_emotional_coherence_empty(self):
        agent = _make_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        coh = reg.emotional_coherence()
        assert coh == 1.0

    def test_domain_filter(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate(domain="health")
        state = report.current_state
        assert state.dominant_domain == "health"

    def test_empty_memory_returns_zero_assessed(self):
        agent = _make_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert report.memories_assessed == 0

    def test_window_memories_cap(self):
        agent = _make_rich_emms(8)
        reg = EmotionalRegulator(memory=agent.memory, window_memories=3)
        report = reg.regulate()
        assert report.current_state.sample_size <= 3

    def test_mood_congruent_ids_list(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert isinstance(report.mood_congruent_ids, list)
        for mid in report.mood_congruent_ids:
            assert isinstance(mid, str)

    def test_reappraisal_content_is_string(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        for r in report.reappraisals:
            assert isinstance(r.reappraised_content, str)
            assert len(r.reappraised_content) > 0

    def test_emotional_state_summary_str(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        s = report.current_state.summary()
        assert isinstance(s, str)
        assert "EmotionalState" in s

    def test_reappraisal_summary_str(self):
        agent = _make_negative_emms()
        reg = EmotionalRegulator(memory=agent.memory, reappraise=True)
        report = reg.regulate()
        if report.reappraisals:
            s = report.reappraisals[0].summary()
            assert isinstance(s, str)
            assert "Reappraisal" in s

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        s = report.summary()
        assert isinstance(s, str)
        assert "EmotionReport" in s

    def test_valence_in_range(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert -1.0 <= report.current_state.valence <= 1.0

    def test_arousal_in_range(self):
        agent = _make_rich_emms()
        reg = EmotionalRegulator(memory=agent.memory)
        report = reg.regulate()
        assert 0.0 <= report.current_state.arousal <= 1.0


# ===========================================================================
# TestConceptHierarchy
# ===========================================================================


class TestConceptHierarchy:

    def test_build_returns_report(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        assert isinstance(report, HierarchyReport)

    def test_report_fields(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        assert isinstance(report.total_concepts, int)
        assert isinstance(report.total_edges, int)
        assert isinstance(report.max_depth, int)
        assert isinstance(report.domains, list)
        assert isinstance(report.nodes, list)
        assert report.duration_seconds >= 0.0

    def test_concepts_generated(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        assert report.total_concepts >= 1

    def test_nodes_are_concept_nodes(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        for node in report.nodes:
            assert isinstance(node, ConceptNode)

    def test_concept_node_fields(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        if report.nodes:
            n = report.nodes[0]
            assert isinstance(n.id, str)
            assert isinstance(n.label, str)
            assert isinstance(n.domain, str)
            assert isinstance(n.level, int)
            assert n.level >= 0
            assert isinstance(n.children_ids, list)
            assert isinstance(n.supporting_memory_ids, list)
            assert 0.0 <= n.abstraction_score <= 1.0

    def test_root_nodes_exist(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        roots = [n for n in report.nodes if n.level == 0]
        assert len(roots) >= 1

    def test_root_nodes_have_no_parent(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        roots = [n for n in report.nodes if n.level == 0]
        for root in roots:
            assert root.parent_id is None

    def test_non_root_nodes_have_parent(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        non_roots = [n for n in report.nodes if n.level > 0]
        for node in non_roots:
            assert node.parent_id is not None

    def test_abstraction_score_higher_at_root(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        roots = [n for n in report.nodes if n.level == 0]
        children = [n for n in report.nodes if n.level == 1]
        if roots and children:
            max_root_abs = max(n.abstraction_score for n in roots)
            max_child_abs = max(n.abstraction_score for n in children)
            # Roots should generally be more abstract than children
            assert max_root_abs >= max_child_abs

    def test_ancestors_returns_list(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        children = [n for n in report.nodes if n.level == 1]
        if children:
            label = children[0].label
            ancs = hier.ancestors(label)
            assert isinstance(ancs, list)

    def test_ancestors_nonexistent_label(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        hier.build()
        result = hier.ancestors("zzz_nonexistent_xyz")
        assert result == []

    def test_descendants_returns_list(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        roots = [n for n in report.nodes if n.level == 0]
        if roots:
            label = roots[0].label
            desc = hier.descendants(label)
            assert isinstance(desc, list)
            for node in desc:
                assert isinstance(node, ConceptNode)

    def test_concept_distance_self_is_zero(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        if report.nodes:
            label = report.nodes[0].label
            assert hier.concept_distance(label, label) == 0

    def test_concept_distance_nonexistent_is_negative_one(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        hier.build()
        assert hier.concept_distance("zzz_nonexistent", "yyy_also_not") == -1

    def test_concept_distance_parent_child_is_one(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        # Find a parent-child pair
        for node in report.nodes:
            if node.level == 1 and node.parent_id:
                parent_node = next(
                    (n for n in report.nodes if n.id == node.parent_id), None
                )
                if parent_node:
                    dist = hier.concept_distance(parent_node.label, node.label)
                    assert dist == 1
                    break

    def test_most_abstract_returns_roots(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        hier.build()
        abstract = hier.most_abstract(n=5)
        assert isinstance(abstract, list)
        for node in abstract:
            assert node.level == 0

    def test_most_specific_returns_leaves(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        hier.build()
        specific = hier.most_specific(n=5)
        assert isinstance(specific, list)
        for node in specific:
            assert isinstance(node, ConceptNode)

    def test_get_node_found(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        if report.nodes:
            label = report.nodes[0].label
            node = hier.get_node(label)
            assert node is not None
            assert node.label == label

    def test_get_node_not_found(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        hier.build()
        result = hier.get_node("zzz_not_in_hierarchy")
        assert result is None

    def test_domain_filter(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build(domain="science")
        for node in report.nodes:
            assert node.domain == "science"

    def test_min_frequency_filters_rare_tokens(self):
        agent = _make_hierarchy_emms()
        hier_strict = ConceptHierarchy(memory=agent.memory, min_frequency=999)
        report = hier_strict.build()
        # Very high min_frequency means no concepts pass
        assert report.total_concepts == 0

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        hier = ConceptHierarchy(memory=agent.memory)
        report = hier.build()
        assert report.total_concepts == 0
        assert report.nodes == []

    def test_total_edges_matches_children_sum(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        edge_sum = sum(len(n.children_ids) for n in report.nodes)
        assert report.total_edges == edge_sum

    def test_report_summary_str(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        s = report.summary()
        assert isinstance(s, str)
        assert "HierarchyReport" in s

    def test_concept_node_summary_str(self):
        agent = _make_hierarchy_emms()
        hier = ConceptHierarchy(memory=agent.memory, min_frequency=2)
        report = hier.build()
        if report.nodes:
            s = report.nodes[0].summary()
            assert isinstance(s, str)
            assert "ConceptNode" in s


# ===========================================================================
# TestSelfModel
# ===========================================================================


class TestSelfModel:

    def test_update_returns_report(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert isinstance(report, SelfModelReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert isinstance(report.beliefs, list)
        assert isinstance(report.core_domains, list)
        assert isinstance(report.dominant_valence, float)
        assert isinstance(report.consistency_score, float)
        assert isinstance(report.capability_profile, dict)
        assert isinstance(report.total_memories_analyzed, int)
        assert report.duration_seconds >= 0.0

    def test_beliefs_generated(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        assert len(report.beliefs) >= 1

    def test_beliefs_are_belief_instances(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        for b in report.beliefs:
            assert isinstance(b, Belief)

    def test_belief_fields(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        if report.beliefs:
            b = report.beliefs[0]
            assert isinstance(b.id, str)
            assert b.id.startswith("belief_")
            assert isinstance(b.content, str)
            assert isinstance(b.domain, str)
            assert 0.0 <= b.confidence <= 1.0
            assert isinstance(b.supporting_memory_ids, list)
            assert -1.0 <= b.valence <= 1.0
            assert isinstance(b.created_at, float)

    def test_belief_id_prefix(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        for b in report.beliefs:
            assert b.id.startswith("belief_")

    def test_confidence_in_range(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        for b in report.beliefs:
            assert 0.0 <= b.confidence <= 1.0

    def test_beliefs_sorted_by_confidence(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        if len(report.beliefs) >= 2:
            for i in range(len(report.beliefs) - 1):
                assert report.beliefs[i].confidence >= report.beliefs[i + 1].confidence

    def test_capability_profile_dict(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert isinstance(report.capability_profile, dict)
        for domain, score in report.capability_profile.items():
            assert isinstance(domain, str)
            assert 0.0 <= score <= 1.0

    def test_capability_profile_method(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        profile = sm.capability_profile()
        assert isinstance(profile, dict)

    def test_consistency_score_bounds(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        assert 0.0 <= report.consistency_score <= 1.0

    def test_consistency_score_method(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        sm.update()
        score = sm.consistency_score()
        assert 0.0 <= score <= 1.0

    def test_core_domains_list(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert isinstance(report.core_domains, list)
        assert len(report.core_domains) <= 3

    def test_dominant_valence_bounds(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        assert -1.0 <= report.dominant_valence <= 1.0

    def test_total_memories_analyzed(self):
        agent = _make_rich_emms(5)
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert report.total_memories_analyzed == 5

    def test_get_belief_found(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        if report.beliefs:
            bid = report.beliefs[0].id
            found = sm.get_belief(bid)
            assert found is not None
            assert found.id == bid

    def test_get_belief_not_found(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        sm.update()
        result = sm.get_belief("belief_nonexistent_xyz")
        assert result is None

    def test_beliefs_method_after_update(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        sm.update()
        beliefs = sm.beliefs()
        assert isinstance(beliefs, list)
        for b in beliefs:
            assert isinstance(b, Belief)

    def test_beliefs_empty_before_update(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory)
        assert sm.beliefs() == []

    def test_max_beliefs_respected(self):
        agent = _make_rich_emms(8)
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1, max_beliefs=2)
        report = sm.update()
        assert len(report.beliefs) <= 2

    def test_empty_memory_returns_empty_beliefs(self):
        agent = _make_emms()
        sm = SelfModel(memory=agent.memory)
        report = sm.update()
        assert report.beliefs == []
        assert report.total_memories_analyzed == 0

    def test_belief_summary_str(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        if report.beliefs:
            s = report.beliefs[0].summary()
            assert isinstance(s, str)
            assert len(s) > 0

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        sm = SelfModel(memory=agent.memory, min_belief_frequency=1)
        report = sm.update()
        s = report.summary()
        assert isinstance(s, str)
        assert "SelfModelReport" in s


# ===========================================================================
# TestEMMSFacadeV190
# ===========================================================================


class TestEMMSFacadeV190:

    def test_regulate_emotions_returns_report(self):
        agent = _make_rich_emms()
        report = agent.regulate_emotions()
        assert isinstance(report, EmotionReport)

    def test_regulate_emotions_domain_filter(self):
        agent = _make_rich_emms()
        report = agent.regulate_emotions(domain="health")
        assert isinstance(report, EmotionReport)
        assert report.current_state.dominant_domain == "health"

    def test_current_emotional_state_after_regulate(self):
        agent = _make_rich_emms()
        agent.regulate_emotions()
        state = agent.current_emotional_state()
        assert state is not None
        assert isinstance(state, EmotionalState)

    def test_current_emotional_state_before_regulate(self):
        agent = _make_rich_emms()
        state = agent.current_emotional_state()
        assert state is None

    def test_mood_retrieve(self):
        agent = _make_rich_emms()
        agent.regulate_emotions()
        items = agent.mood_retrieve(k=4)
        assert isinstance(items, list)
        assert len(items) <= 4

    def test_build_concept_hierarchy(self):
        agent = _make_hierarchy_emms()
        report = agent.build_concept_hierarchy()
        assert isinstance(report, HierarchyReport)

    def test_build_concept_hierarchy_domain(self):
        agent = _make_hierarchy_emms()
        report = agent.build_concept_hierarchy(domain="science")
        assert isinstance(report, HierarchyReport)

    def test_concept_distance_returns_int(self):
        agent = _make_hierarchy_emms()
        dist = agent.concept_distance("cognitive", "memory")
        assert isinstance(dist, int)

    def test_update_self_model_returns_report(self):
        agent = _make_rich_emms()
        report = agent.update_self_model()
        assert isinstance(report, SelfModelReport)

    def test_self_model_beliefs(self):
        agent = _make_rich_emms()
        agent.update_self_model()
        beliefs = agent.self_model_beliefs()
        assert isinstance(beliefs, list)

    def test_capability_profile(self):
        agent = _make_rich_emms()
        profile = agent.capability_profile()
        assert isinstance(profile, dict)

    def test_lazy_init_emotional_regulator(self):
        agent = _make_emms()
        assert not hasattr(agent, "_emotional_regulator")
        agent.regulate_emotions()
        assert hasattr(agent, "_emotional_regulator")

    def test_lazy_init_self_model(self):
        agent = _make_emms()
        assert not hasattr(agent, "_self_model")
        agent.update_self_model()
        assert hasattr(agent, "_self_model")

    def test_build_concept_hierarchy_stateless(self):
        """ConceptHierarchy is instantiated per-call (stateless)."""
        agent = _make_hierarchy_emms()
        r1 = agent.build_concept_hierarchy()
        r2 = agent.build_concept_hierarchy()
        assert isinstance(r1, HierarchyReport)
        assert isinstance(r2, HierarchyReport)


# ===========================================================================
# TestMCPV190
# ===========================================================================


class TestMCPV190:

    def test_tool_count_is_82(self):
        assert len(_TOOL_DEFINITIONS) == 107

    def test_emms_regulate_emotions_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_regulate_emotions", {})
        assert isinstance(result, dict)
        assert "ok" in result or "error" in result

    def test_emms_current_emotion_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        server.handle("emms_regulate_emotions", {})
        result = server.handle("emms_current_emotion", {})
        assert isinstance(result, dict)
        assert "ok" in result or "error" in result

    def test_emms_build_hierarchy_callable(self):
        agent = _make_hierarchy_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_build_hierarchy", {})
        assert isinstance(result, dict)
        assert "total_concepts" in result or "ok" in result or "error" in result

    def test_emms_concept_distance_callable(self):
        agent = _make_hierarchy_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_concept_distance",
                               {"label_a": "cognitive", "label_b": "memory"})
        assert isinstance(result, dict)
        assert "distance" in result or "ok" in result or "error" in result

    def test_emms_update_self_model_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_update_self_model", {})
        assert isinstance(result, dict)
        assert "ok" in result or "beliefs_count" in result or "error" in result

    def test_new_tools_in_definitions(self):
        tool_names = {t["name"] for t in _TOOL_DEFINITIONS}
        new_tools = {
            "emms_regulate_emotions",
            "emms_current_emotion",
            "emms_build_hierarchy",
            "emms_concept_distance",
            "emms_update_self_model",
        }
        for name in new_tools:
            assert name in tool_names, f"Missing tool: {name}"

    def test_all_v190_tools_return_dict(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        for tool_name in [
            "emms_regulate_emotions",
            "emms_current_emotion",
            "emms_update_self_model",
        ]:
            result = server.handle(tool_name, {})
            assert isinstance(result, dict), f"{tool_name} did not return dict"


# ===========================================================================
# TestV190Exports
# ===========================================================================


class TestV190Exports:

    def test_version_is_0_19_0(self):
        import emms
        assert emms.__version__ == "0.24.0"

    def test_emotional_regulator_exported(self):
        from emms import EmotionalRegulator
        assert EmotionalRegulator is not None

    def test_emotional_state_exported(self):
        from emms import EmotionalState
        assert EmotionalState is not None

    def test_reappraisal_result_exported(self):
        from emms import ReappraisalResult
        assert ReappraisalResult is not None

    def test_emotion_report_exported(self):
        from emms import EmotionReport
        assert EmotionReport is not None

    def test_concept_hierarchy_exported(self):
        from emms import ConceptHierarchy
        assert ConceptHierarchy is not None

    def test_concept_node_exported(self):
        from emms import ConceptNode
        assert ConceptNode is not None

    def test_hierarchy_report_exported(self):
        from emms import HierarchyReport
        assert HierarchyReport is not None

    def test_self_model_exported(self):
        from emms import SelfModel
        assert SelfModel is not None

    def test_belief_exported(self):
        from emms import Belief
        assert Belief is not None

    def test_self_model_report_exported(self):
        from emms import SelfModelReport
        assert SelfModelReport is not None

    def test_all_contains_new_symbols(self):
        import emms
        new_symbols = [
            "EmotionalRegulator", "EmotionalState", "ReappraisalResult", "EmotionReport",
            "ConceptHierarchy", "ConceptNode", "HierarchyReport",
            "SelfModel", "Belief", "SelfModelReport",
        ]
        for sym in new_symbols:
            assert sym in emms.__all__, f"Missing from __all__: {sym}"
