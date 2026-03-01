"""Tests for EMMS v0.20.0 — The Reasoning Mind.

Covers CausalMapper, CounterfactualEngine, SkillDistiller, EMMS façade,
MCP server tools, and public exports.
"""

from __future__ import annotations

import pytest

from emms import EMMS
from emms.core.models import Experience
from emms.memory.causal import CausalMapper, CausalEdge, CausalPath, CausalReport
from emms.memory.counterfactual import (
    CounterfactualEngine, Counterfactual, CounterfactualReport,
)
from emms.memory.skills import SkillDistiller, DistilledSkill, SkillReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_causal_emms() -> EMMS:
    """EMMS pre-loaded with causal-keyword memories."""
    agent = EMMS()
    causal_contents = [
        "Stress causes illness when sustained over time.",
        "Practice enables mastery of complex skills.",
        "Exercise reduces anxiety and improves mood.",
        "Learning increases knowledge and capability.",
        "Poor sleep damages cognitive performance significantly.",
        "Meditation strengthens focus and attention.",
        "High workload triggers burnout in professionals.",
        "Curiosity leads discovery of new ideas.",
        "Stress causes fatigue and reduced productivity.",
        "Exercise enables better sleep quality.",
    ]
    for content in causal_contents:
        agent.store(Experience(content=content, domain="health", importance=0.7))
    return agent


def _make_cf_emms() -> EMMS:
    """EMMS pre-loaded with mixed-valence memories for counterfactuals."""
    agent = EMMS()
    items = [
        ("A project failed because of poor planning.", "work", -0.7),
        ("The meeting went very well today.", "work", 0.8),
        ("I struggled with the difficult problem.", "learning", -0.4),
        ("Team collaboration produced excellent results.", "work", 0.6),
        ("The deadline was missed due to delays.", "work", -0.5),
        ("Training improved performance significantly.", "health", 0.7),
        ("Communication breakdown caused issues.", "work", -0.3),
        ("New approach solved the long-standing problem.", "learning", 0.9),
    ]
    for content, domain, valence in items:
        agent.store(Experience(
            content=content, domain=domain, importance=0.6,
            emotional_valence=valence,
        ))
    return agent


def _make_skill_emms() -> EMMS:
    """EMMS pre-loaded with action-rich memories."""
    agent = EMMS()
    contents = [
        "Practice daily to improve programming skills.",
        "Analyze data carefully to improve decision making.",
        "Build prototypes early to test design assumptions.",
        "Learn from feedback to improve performance.",
        "Practice meditation to reduce stress levels.",
        "Implement tests to build reliable software.",
        "Design systems before writing implementation code.",
        "Practice consistently to develop musical ability.",
        "Analyze patterns to improve predictive accuracy.",
        "Build strong habits to strengthen discipline.",
    ]
    for content in contents:
        agent.store(Experience(content=content, domain="skills", importance=0.7))
    return agent


def _make_rich_emms() -> EMMS:
    """Comprehensive EMMS with causal, valenced, and action memories."""
    agent = _make_causal_emms()
    for content, domain, valence in [
        ("Poor planning causes project failure.", "work", -0.6),
        ("Practice leads improvement in performance.", "work", 0.7),
        ("Stress reduces sleep quality and health.", "health", -0.5),
        ("Collaboration enables better problem solving.", "work", 0.8),
        ("Apply lessons learned to improve outcomes.", "work", 0.5),
        ("Develop skills to increase capability.", "learning", 0.6),
        ("Build systems that solve complex problems.", "learning", 0.7),
        ("Learn principles to apply across domains.", "learning", 0.8),
    ]:
        agent.store(Experience(
            content=content, domain=domain, importance=0.7,
            emotional_valence=valence,
        ))
    return agent


# ===========================================================================
# TestCausalMapper
# ===========================================================================


class TestCausalMapper:

    def test_build_returns_causal_report(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert isinstance(report, CausalReport)

    def test_report_has_required_fields(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert hasattr(report, "total_concepts")
        assert hasattr(report, "total_edges")
        assert hasattr(report, "most_influential")
        assert hasattr(report, "most_affected")
        assert hasattr(report, "edges")
        assert hasattr(report, "duration_seconds")

    def test_report_total_concepts_nonneg(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert report.total_concepts >= 0

    def test_report_total_edges_positive(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert report.total_edges > 0

    def test_edges_are_causal_edge_instances(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        for e in report.edges:
            assert isinstance(e, CausalEdge)

    def test_causal_edge_fields(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert len(report.edges) > 0
        e = report.edges[0]
        assert isinstance(e.source, str)
        assert isinstance(e.target, str)
        assert isinstance(e.relation, str)
        assert isinstance(e.strength, float)
        assert isinstance(e.memory_ids, list)

    def test_edge_strength_in_range(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        for e in report.edges:
            assert 0.0 <= e.strength <= 1.0

    def test_edges_sorted_by_strength_desc(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        strengths = [e.strength for e in report.edges]
        assert strengths == sorted(strengths, reverse=True)

    def test_most_influential_is_list(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert isinstance(report.most_influential, list)

    def test_most_affected_is_list(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert isinstance(report.most_affected, list)

    def test_effects_of_returns_list(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.effects_of("stress")
        assert isinstance(result, list)

    def test_effects_of_returns_causal_edges(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.effects_of("stress")
        for e in result:
            assert isinstance(e, CausalEdge)

    def test_causes_of_returns_list(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.causes_of("illness")
        assert isinstance(result, list)

    def test_causal_path_returns_none_unknown(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.causal_path("zzz_unknown", "yyy_unknown")
        assert result is None

    def test_causal_path_same_node(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.causal_path("stress", "stress")
        # Same source and target returns trivial path
        assert result is not None
        assert isinstance(result, CausalPath)
        assert result.nodes == ["stress"]

    def test_causal_path_fields(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        # Find a connected pair
        report = mapper.build()
        if report.edges:
            e = report.edges[0]
            path = mapper.causal_path(e.source, e.target)
            if path:
                assert isinstance(path.nodes, list)
                assert isinstance(path.edges, list)
                assert isinstance(path.total_strength, float)

    def test_most_influential_method(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        mapper.build()
        result = mapper.most_influential(n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_domain_filter(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report_all = mapper.build()
        report_domain = mapper.build(domain="health")
        # Domain filter should not increase edges
        assert report_domain.total_edges <= report_all.total_edges

    def test_empty_memory(self):
        agent = _make_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert isinstance(report, CausalReport)
        assert report.total_concepts == 0
        assert report.total_edges == 0

    def test_min_strength_filter(self):
        agent = _make_causal_emms()
        mapper_low = CausalMapper(memory=agent.memory, min_strength=0.0)
        mapper_high = CausalMapper(memory=agent.memory, min_strength=0.5)
        report_low = mapper_low.build()
        report_high = mapper_high.build()
        assert report_high.total_edges <= report_low.total_edges

    def test_causal_report_summary_string(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "CausalReport" in summary

    def test_causal_edge_summary_string(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        if report.edges:
            summary = report.edges[0].summary()
            assert isinstance(summary, str)
            assert "CausalEdge" in summary

    def test_causal_path_summary_string(self):
        path = CausalPath(nodes=["a", "b"], edges=[], total_strength=0.5)
        summary = path.summary()
        assert isinstance(summary, str)
        assert "CausalPath" in summary

    def test_duration_nonneg(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        assert report.duration_seconds >= 0.0

    def test_relation_is_keyword(self):
        agent = _make_causal_emms()
        mapper = CausalMapper(memory=agent.memory)
        report = mapper.build()
        valid_keywords = {
            "causes", "enables", "produces", "prevents", "reduces", "increases",
            "requires", "triggers", "inhibits", "leads", "results", "improves",
            "damages", "strengthens", "weakens",
        }
        for e in report.edges:
            assert e.relation in valid_keywords


# ===========================================================================
# TestCounterfactualEngine
# ===========================================================================


class TestCounterfactualEngine:

    def test_generate_returns_report(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert isinstance(report, CounterfactualReport)

    def test_report_has_required_fields(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert hasattr(report, "total_memories_assessed")
        assert hasattr(report, "counterfactuals_generated")
        assert hasattr(report, "counterfactuals")
        assert hasattr(report, "mean_plausibility")
        assert hasattr(report, "duration_seconds")

    def test_total_memories_assessed_positive(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert report.total_memories_assessed > 0

    def test_counterfactuals_generated_nonneg(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert report.counterfactuals_generated >= 0

    def test_counterfactuals_are_counterfactual_instances(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        for c in report.counterfactuals:
            assert isinstance(c, Counterfactual)

    def test_counterfactual_fields(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert len(report.counterfactuals) > 0
        c = report.counterfactuals[0]
        assert isinstance(c.id, str)
        assert isinstance(c.basis_memory_id, str)
        assert isinstance(c.original_content, str)
        assert isinstance(c.counterfactual_content, str)
        assert isinstance(c.direction, str)
        assert isinstance(c.valence_shift, float)
        assert isinstance(c.plausibility, float)
        assert isinstance(c.domain, str)

    def test_id_prefix(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        for c in report.counterfactuals:
            assert c.id.startswith("cf_")

    def test_direction_upward(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate(direction="upward")
        for c in report.counterfactuals:
            assert c.direction == "upward"

    def test_direction_downward(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate(direction="downward")
        for c in report.counterfactuals:
            assert c.direction == "downward"

    def test_direction_both(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate(direction="both")
        directions = {c.direction for c in report.counterfactuals}
        assert len(directions) >= 1  # at least one direction

    def test_upward_valence_shift_positive(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate(direction="upward")
        for c in report.counterfactuals:
            assert c.valence_shift > 0

    def test_downward_valence_shift_negative(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate(direction="downward")
        for c in report.counterfactuals:
            assert c.valence_shift < 0

    def test_plausibility_in_range(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        for c in report.counterfactuals:
            assert 0.0 <= c.plausibility <= 1.0

    def test_mean_plausibility_in_range(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert 0.0 <= report.mean_plausibility <= 1.0

    def test_counterfactuals_sorted_by_abs_valence_shift(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        shifts = [abs(c.valence_shift) for c in report.counterfactuals]
        assert shifts == sorted(shifts, reverse=True)

    def test_for_memory_returns_list(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        engine.generate()
        if engine._counterfactuals:
            cf = engine._counterfactuals[0]
            result = engine.for_memory(cf.basis_memory_id)
            assert isinstance(result, list)
            assert any(c.id == cf.id for c in result)

    def test_upward_method(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        engine.generate()
        result = engine.upward(n=3)
        assert isinstance(result, list)
        for c in result:
            assert c.direction == "upward"

    def test_downward_method(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        engine.generate()
        result = engine.downward(n=3)
        assert isinstance(result, list)
        for c in result:
            assert c.direction == "downward"

    def test_empty_memory(self):
        agent = _make_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert isinstance(report, CounterfactualReport)
        assert report.counterfactuals_generated == 0

    def test_domain_filter(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report_all = engine.generate()
        engine2 = CounterfactualEngine(memory=agent.memory)
        report_domain = engine2.generate(domain="work")
        for c in report_domain.counterfactuals:
            assert c.domain == "work"

    def test_counterfactual_report_summary(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "CounterfactualReport" in summary

    def test_counterfactual_summary(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        if report.counterfactuals:
            summary = report.counterfactuals[0].summary()
            assert isinstance(summary, str)
            assert "Counterfactual" in summary

    def test_duration_nonneg(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory)
        report = engine.generate()
        assert report.duration_seconds >= 0.0

    def test_plausibility_threshold(self):
        agent = _make_cf_emms()
        engine = CounterfactualEngine(memory=agent.memory, plausibility_threshold=0.9)
        report = engine.generate()
        for c in report.counterfactuals:
            assert c.plausibility >= 0.9


# ===========================================================================
# TestSkillDistiller
# ===========================================================================


class TestSkillDistiller:

    def test_distill_returns_skill_report(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert isinstance(report, SkillReport)

    def test_report_has_required_fields(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert hasattr(report, "total_memories_analyzed")
        assert hasattr(report, "skills_distilled")
        assert hasattr(report, "skills")
        assert hasattr(report, "domains_covered")
        assert hasattr(report, "duration_seconds")

    def test_total_memories_analyzed_positive(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert report.total_memories_analyzed > 0

    def test_skills_distilled_positive(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert report.skills_distilled > 0

    def test_skills_are_distilled_skill_instances(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert isinstance(s, DistilledSkill)

    def test_skill_fields(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert len(report.skills) > 0
        s = report.skills[0]
        assert isinstance(s.id, str)
        assert isinstance(s.name, str)
        assert isinstance(s.domain, str)
        assert isinstance(s.description, str)
        assert isinstance(s.preconditions, list)
        assert isinstance(s.outcomes, list)
        assert isinstance(s.confidence, float)
        assert isinstance(s.source_memory_ids, list)

    def test_skill_id_prefix(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert s.id.startswith("skill_")

    def test_confidence_in_range(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert 0.0 <= s.confidence <= 1.0

    def test_skills_sorted_by_confidence_desc(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        confidences = [s.confidence for s in report.skills]
        assert confidences == sorted(confidences, reverse=True)

    def test_preconditions_is_list(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert isinstance(s.preconditions, list)

    def test_outcomes_is_list(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert isinstance(s.outcomes, list)

    def test_domains_covered_is_list(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert isinstance(report.domains_covered, list)

    def test_skills_for_domain(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        distiller.distill()
        result = distiller.skills_for_domain("skills")
        assert isinstance(result, list)
        for s in result:
            assert s.domain == "skills"

    def test_best_skill_returns_none_if_no_skills(self):
        agent = _make_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        result = distiller.best_skill("improve programming")
        assert result is None

    def test_best_skill_returns_skill(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        distiller.distill()
        result = distiller.best_skill("improve programming performance")
        if result is not None:
            assert isinstance(result, DistilledSkill)

    def test_min_skill_frequency_filter(self):
        agent = _make_skill_emms()
        distiller_low = SkillDistiller(
            memory=agent.memory, min_skill_frequency=1, store_skills=False
        )
        distiller_high = SkillDistiller(
            memory=agent.memory, min_skill_frequency=5, store_skills=False
        )
        report_low = distiller_low.distill()
        report_high = distiller_high.distill()
        assert report_high.skills_distilled <= report_low.skills_distilled

    def test_skill_report_summary(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "SkillReport" in summary

    def test_distilled_skill_summary(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        if report.skills:
            summary = report.skills[0].summary()
            assert isinstance(summary, str)
            assert "DistilledSkill" in summary

    def test_domain_filter(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill(domain="skills")
        for s in report.skills:
            assert s.domain == "skills"

    def test_empty_memory(self):
        agent = _make_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert isinstance(report, SkillReport)
        assert report.skills_distilled == 0

    def test_duration_nonneg(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        assert report.duration_seconds >= 0.0

    def test_store_skills_creates_memory(self):
        agent = _make_skill_emms()
        distiller = SkillDistiller(
            memory=agent.memory, store_skills=True, min_skill_frequency=1
        )
        report = distiller.distill()
        # Skills should have been distilled and accumulated in _skills
        assert report.skills_distilled >= 0  # no exception raised
        assert len(distiller._skills) == report.skills_distilled

    def test_skill_name_is_action_token(self):
        from emms.memory.skills import _ACTION_TOKENS
        agent = _make_skill_emms()
        distiller = SkillDistiller(memory=agent.memory, store_skills=False)
        report = distiller.distill()
        for s in report.skills:
            assert s.name in _ACTION_TOKENS


# ===========================================================================
# TestEMMSFacadeV200
# ===========================================================================


class TestEMMSFacadeV200:

    def test_build_causal_map_returns_causal_report(self):
        agent = _make_causal_emms()
        report = agent.build_causal_map()
        assert isinstance(report, CausalReport)

    def test_build_causal_map_domain_filter(self):
        agent = _make_causal_emms()
        report = agent.build_causal_map(domain="health")
        assert isinstance(report, CausalReport)

    def test_effects_of_returns_list(self):
        agent = _make_causal_emms()
        agent.build_causal_map()
        result = agent.effects_of("stress")
        assert isinstance(result, list)

    def test_causes_of_returns_list(self):
        agent = _make_causal_emms()
        agent.build_causal_map()
        result = agent.causes_of("illness")
        assert isinstance(result, list)

    def test_generate_counterfactuals_returns_report(self):
        agent = _make_cf_emms()
        report = agent.generate_counterfactuals()
        assert isinstance(report, CounterfactualReport)

    def test_generate_counterfactuals_direction(self):
        agent = _make_cf_emms()
        report = agent.generate_counterfactuals(direction="upward")
        for c in report.counterfactuals:
            assert c.direction == "upward"

    def test_upward_counterfactuals_returns_list(self):
        agent = _make_cf_emms()
        agent.generate_counterfactuals()
        result = agent.upward_counterfactuals(n=3)
        assert isinstance(result, list)

    def test_downward_counterfactuals_returns_list(self):
        agent = _make_cf_emms()
        agent.generate_counterfactuals()
        result = agent.downward_counterfactuals(n=3)
        assert isinstance(result, list)

    def test_distill_skills_returns_skill_report(self):
        agent = _make_skill_emms()
        report = agent.distill_skills()
        assert isinstance(report, SkillReport)

    def test_distill_skills_domain_filter(self):
        agent = _make_skill_emms()
        report = agent.distill_skills(domain="skills")
        assert isinstance(report, SkillReport)

    def test_best_skill_returns_skill_or_none(self):
        agent = _make_skill_emms()
        agent.distill_skills()
        result = agent.best_skill("improve programming")
        assert result is None or isinstance(result, DistilledSkill)

    def test_lazy_init_causal_mapper(self):
        agent = _make_causal_emms()
        assert not hasattr(agent, "_causal_mapper")
        agent.build_causal_map()
        assert hasattr(agent, "_causal_mapper")

    def test_lazy_init_counterfactual_engine(self):
        agent = _make_cf_emms()
        assert not hasattr(agent, "_counterfactual_engine")
        agent.generate_counterfactuals()
        assert hasattr(agent, "_counterfactual_engine")

    def test_lazy_init_skill_distiller(self):
        agent = _make_skill_emms()
        assert not hasattr(agent, "_skill_distiller")
        agent.distill_skills()
        assert hasattr(agent, "_skill_distiller")

    def test_causal_mapper_reused_on_second_call(self):
        agent = _make_causal_emms()
        agent.build_causal_map()
        mapper1 = agent._causal_mapper
        agent.build_causal_map()
        assert agent._causal_mapper is mapper1


# ===========================================================================
# TestMCPV200
# ===========================================================================


class TestMCPV200:

    def _get_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_87(self):
        server = self._get_server()
        assert len(server.tool_definitions) == 112

    def test_emms_build_causal_map_callable(self):
        server = self._get_server()
        result = server.handle("emms_build_causal_map", {})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_effects_of_callable(self):
        server = self._get_server()
        result = server.handle("emms_effects_of", {"concept": "stress"})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_generate_counterfactuals_callable(self):
        server = self._get_server()
        result = server.handle("emms_generate_counterfactuals", {"direction": "both"})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_distill_skills_callable(self):
        server = self._get_server()
        result = server.handle("emms_distill_skills", {})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_best_skill_callable(self):
        server = self._get_server()
        # First distill
        server.handle("emms_distill_skills", {})
        result = server.handle("emms_best_skill", {"goal_description": "improve performance"})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_new_tools_in_definitions(self):
        server = self._get_server()
        names = {d["name"] for d in server.tool_definitions}
        for name in [
            "emms_build_causal_map",
            "emms_effects_of",
            "emms_generate_counterfactuals",
            "emms_distill_skills",
            "emms_best_skill",
        ]:
            assert name in names


# ===========================================================================
# TestV200Exports
# ===========================================================================


class TestV200Exports:

    def test_version_is_0_20_0(self):
        import emms
        assert emms.__version__ == "0.25.0"

    def test_causal_mapper_exported(self):
        from emms import CausalMapper
        assert CausalMapper is not None

    def test_causal_edge_exported(self):
        from emms import CausalEdge
        assert CausalEdge is not None

    def test_causal_path_exported(self):
        from emms import CausalPath
        assert CausalPath is not None

    def test_causal_report_exported(self):
        from emms import CausalReport
        assert CausalReport is not None

    def test_counterfactual_engine_exported(self):
        from emms import CounterfactualEngine
        assert CounterfactualEngine is not None

    def test_counterfactual_exported(self):
        from emms import Counterfactual
        assert Counterfactual is not None

    def test_counterfactual_report_exported(self):
        from emms import CounterfactualReport
        assert CounterfactualReport is not None

    def test_skill_distiller_exported(self):
        from emms import SkillDistiller
        assert SkillDistiller is not None

    def test_distilled_skill_exported(self):
        from emms import DistilledSkill
        assert DistilledSkill is not None

    def test_skill_report_exported(self):
        from emms import SkillReport
        assert SkillReport is not None

    def test_all_symbols_in_all(self):
        import emms
        for sym in [
            "CausalMapper", "CausalEdge", "CausalPath", "CausalReport",
            "CounterfactualEngine", "Counterfactual", "CounterfactualReport",
            "SkillDistiller", "DistilledSkill", "SkillReport",
        ]:
            assert sym in emms.__all__, f"{sym} not in __all__"
