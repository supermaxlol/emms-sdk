"""Tests for EMMS v0.23.0 — The Moral Mind.

Covers:
- ValueMapper / MappedValue / ValueReport
- MoralReasoner / MoralAssessment / MoralReport
- DilemmaEngine / EthicalDilemma / DilemmaReport
- EMMS facade (v0.23.0 methods)
- MCP tool count (102) + 5 new tools
- Version and __all__ exports
"""

from __future__ import annotations

import pytest

import emms as emms_pkg
from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms() -> EMMS:
    """EMMS loaded with morally-coloured memories across two domains."""
    agent = EMMS()
    # ethics domain — high valence variance, moral + deontological language
    for content, ev in [
        ("Justice must be served: the duty to treat all citizens with dignity and respect.", 0.8),
        ("The outcome harm is clear: cost exceeds benefit when welfare is undermined.", -0.7),
        ("Virtue requires honesty; an honest character builds integrity and trust.", 0.7),
        ("Forbidden actions violate rights; rights must never be sacrificed for utility.", -0.6),
        ("Compassion and care are moral values that result in community welfare.", 0.9),
        ("The rule against harm is categorical: we owe protection to the vulnerable.", 0.5),
        ("Brave and wise individuals produce better outcomes for society overall.", 0.6),
        ("Cost-benefit analysis of welfare outcomes justifies consequentialist choices.", -0.4),
    ]:
        agent.store(Experience(
            content=content, domain="ethics",
            importance=0.85, emotional_valence=ev,
        ))
    # philosophy domain — epistemic + instrumental value language
    for content, ev in [
        ("Truth and knowledge require evidence and reason to achieve understanding.", 0.7),
        ("Progress demands learning and growth; efficiency enables us to achieve more.", 0.6),
        ("Clarity and precision in logic build reliable understanding of reality.", 0.8),
        ("Insight and transparency advance knowledge through careful verification.", 0.5),
        ("Build on existing knowledge to develop and innovate new solutions.", 0.4),
    ]:
        agent.store(Experience(
            content=content, domain="philosophy",
            importance=0.75, emotional_valence=ev,
        ))
    return agent


# ---------------------------------------------------------------------------
# TestValueMapper
# ---------------------------------------------------------------------------

class TestValueMapper:

    def test_map_values_returns_value_report(self):
        from emms.memory.values import ValueMapper, ValueReport
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert isinstance(report, ValueReport)

    def test_report_has_required_fields(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert hasattr(report, "total_values")
        assert hasattr(report, "values")
        assert hasattr(report, "dominant_category")
        assert hasattr(report, "mean_strength")
        assert hasattr(report, "duration_seconds")

    def test_total_values_matches_list_length(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert report.total_values == len(report.values)

    def test_values_are_mapped_value_instances(self):
        from emms.memory.values import ValueMapper, MappedValue
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        if report.values:
            assert isinstance(report.values[0], MappedValue)

    def test_mapped_value_fields(self):
        from emms.memory.values import ValueMapper, MappedValue
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        if report.values:
            v = report.values[0]
            assert hasattr(v, "id")
            assert hasattr(v, "name")
            assert hasattr(v, "category")
            assert hasattr(v, "strength")
            assert hasattr(v, "description")
            assert hasattr(v, "source_memory_ids")
            assert hasattr(v, "created_at")

    def test_value_id_prefix(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        for v in report.values:
            assert v.id.startswith("val_"), f"id '{v.id}' does not start with 'val_'"

    def test_strength_in_range(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        for v in report.values:
            assert 0.0 <= v.strength <= 1.0

    def test_values_sorted_by_strength_descending(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        strengths = [v.strength for v in report.values]
        assert strengths == sorted(strengths, reverse=True)

    def test_category_is_valid(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        valid = {"epistemic", "moral", "aesthetic", "instrumental", "social"}
        for v in report.values:
            assert v.category in valid

    def test_dominant_category_is_string(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert isinstance(report.dominant_category, str)

    def test_mean_strength_in_range(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert 0.0 <= report.mean_strength <= 1.0

    def test_values_for_category_filter(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        mapper.map_values()
        moral_vals = mapper.values_for_category("moral")
        for v in moral_vals:
            assert v.category == "moral"

    def test_strongest_value_returns_top(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        strongest = mapper.strongest_value()
        if report.values:
            assert strongest is not None
            assert strongest.strength == report.values[0].strength

    def test_category_param_filters_output(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values(category="epistemic")
        for v in report.values:
            assert v.category == "epistemic"

    def test_empty_memory_returns_zero(self):
        from emms.memory.values import ValueMapper
        agent = _make_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert report.total_values == 0
        assert report.values == []

    def test_min_strength_filter(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory, min_strength=0.9)
        report = mapper.map_values()
        for v in report.values:
            assert v.strength >= 0.9

    def test_strongest_value_none_on_empty(self):
        from emms.memory.values import ValueMapper
        agent = _make_emms()
        mapper = ValueMapper(memory=agent.memory)
        mapper.map_values()
        assert mapper.strongest_value() is None

    def test_source_memory_ids_is_list(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        for v in report.values:
            assert isinstance(v.source_memory_ids, list)

    def test_report_summary_is_string(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert isinstance(report.summary(), str)

    def test_mapped_value_summary_is_string(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        if report.values:
            assert isinstance(report.values[0].summary(), str)

    def test_duration_non_negative(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert report.duration_seconds >= 0.0

    def test_rich_emms_finds_at_least_one_value(self):
        from emms.memory.values import ValueMapper
        agent = _make_rich_emms()
        mapper = ValueMapper(memory=agent.memory)
        report = mapper.map_values()
        assert report.total_values >= 1


# ---------------------------------------------------------------------------
# TestMoralReasoner
# ---------------------------------------------------------------------------

class TestMoralReasoner:

    def test_reason_returns_moral_report(self):
        from emms.memory.moral import MoralReasoner, MoralReport
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert isinstance(report, MoralReport)

    def test_report_has_required_fields(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert hasattr(report, "total_assessed")
        assert hasattr(report, "assessments")
        assert hasattr(report, "dominant_framework_overall")
        assert hasattr(report, "mean_moral_weight")
        assert hasattr(report, "framework_counts")
        assert hasattr(report, "duration_seconds")

    def test_total_assessed_matches_list(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert report.total_assessed == len(report.assessments)

    def test_assessments_are_moral_assessment_instances(self):
        from emms.memory.moral import MoralReasoner, MoralAssessment
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        if report.assessments:
            assert isinstance(report.assessments[0], MoralAssessment)

    def test_assessment_fields(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        if report.assessments:
            a = report.assessments[0]
            assert hasattr(a, "memory_id")
            assert hasattr(a, "content_excerpt")
            assert hasattr(a, "consequentialist_score")
            assert hasattr(a, "deontological_score")
            assert hasattr(a, "virtue_score")
            assert hasattr(a, "dominant_framework")
            assert hasattr(a, "moral_weight")
            assert hasattr(a, "domain")

    def test_framework_scores_in_range(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        for a in report.assessments:
            assert 0.0 <= a.consequentialist_score <= 1.0
            assert 0.0 <= a.deontological_score <= 1.0
            assert 0.0 <= a.virtue_score <= 1.0

    def test_dominant_framework_valid(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        valid = {"consequentialist", "deontological", "virtue", "none"}
        for a in report.assessments:
            assert a.dominant_framework in valid

    def test_moral_weight_in_range(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        for a in report.assessments:
            assert 0.0 <= a.moral_weight <= 1.0

    def test_assessments_sorted_by_moral_weight_desc(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        weights = [a.moral_weight for a in report.assessments]
        assert weights == sorted(weights, reverse=True)

    def test_assessments_by_framework_filter(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        reasoner.reason()
        for fw in ("consequentialist", "deontological", "virtue", "none"):
            results = reasoner.assessments_by_framework(fw)
            for a in results:
                assert a.dominant_framework == fw

    def test_moral_weight_of_known_memory(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        if report.assessments:
            mid = report.assessments[0].memory_id
            w = reasoner.moral_weight_of(mid)
            assert 0.0 <= w <= 1.0

    def test_moral_weight_of_unknown_returns_zero(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        reasoner.reason()
        assert reasoner.moral_weight_of("nonexistent_id") == 0.0

    def test_framework_counts_is_dict(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert isinstance(report.framework_counts, dict)

    def test_framework_counts_keys(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        for key in ("consequentialist", "deontological", "virtue", "none"):
            assert key in report.framework_counts

    def test_domain_filter(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason(domain="ethics")
        for a in report.assessments:
            assert a.domain == "ethics"

    def test_empty_memory_returns_zero_assessed(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert report.total_assessed == 0

    def test_dominant_framework_overall_is_string(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert isinstance(report.dominant_framework_overall, str)

    def test_mean_moral_weight_in_range(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert 0.0 <= report.mean_moral_weight <= 1.0

    def test_report_summary_is_string(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert isinstance(report.summary(), str)

    def test_assessment_summary_is_string(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        if report.assessments:
            assert isinstance(report.assessments[0].summary(), str)

    def test_content_excerpt_max_80(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        for a in report.assessments:
            assert len(a.content_excerpt) <= 80

    def test_rich_emms_finds_at_least_one_assessment(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert report.total_assessed >= 1

    def test_duration_non_negative(self):
        from emms.memory.moral import MoralReasoner
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        report = reasoner.reason()
        assert report.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# TestDilemmaEngine
# ---------------------------------------------------------------------------

class TestDilemmaEngine:

    def _make_reasoner_and_engine(self, agent):
        from emms.memory.moral import MoralReasoner
        from emms.memory.dilemma import DilemmaEngine
        reasoner = MoralReasoner(memory=agent.memory)
        reasoner.reason()
        engine = DilemmaEngine(memory=agent.memory, moral_reasoner=reasoner)
        return engine

    def test_detect_returns_dilemma_report(self):
        from emms.memory.dilemma import DilemmaReport
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert isinstance(report, DilemmaReport)

    def test_report_has_required_fields(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert hasattr(report, "total_dilemmas")
        assert hasattr(report, "dilemmas")
        assert hasattr(report, "mean_tension")
        assert hasattr(report, "domains_affected")
        assert hasattr(report, "duration_seconds")

    def test_total_dilemmas_matches_list(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert report.total_dilemmas == len(report.dilemmas)

    def test_dilemmas_are_ethical_dilemma_instances(self):
        from emms.memory.dilemma import EthicalDilemma
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            assert isinstance(d, EthicalDilemma)

    def test_ethical_dilemma_fields(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        if report.dilemmas:
            d = report.dilemmas[0]
            assert hasattr(d, "id")
            assert hasattr(d, "description")
            assert hasattr(d, "memory_id_a")
            assert hasattr(d, "memory_id_b")
            assert hasattr(d, "domain")
            assert hasattr(d, "tension_score")
            assert hasattr(d, "framework_a")
            assert hasattr(d, "framework_b")
            assert hasattr(d, "resolution_strategies")
            assert hasattr(d, "created_at")

    def test_dilemma_id_prefix(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            assert d.id.startswith("dil_")

    def test_tension_in_range(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            assert 0.0 <= d.tension_score <= 1.0

    def test_dilemmas_sorted_by_tension_desc(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        tensions = [d.tension_score for d in report.dilemmas]
        assert tensions == sorted(tensions, reverse=True)

    def test_resolution_strategies_non_empty(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            assert len(d.resolution_strategies) >= 1

    def test_resolution_strategies_are_strings(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            for s in d.resolution_strategies:
                assert isinstance(s, str)

    def test_dilemmas_for_domain_filter(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        engine.detect_dilemmas()
        dilemmas = engine.dilemmas_for_domain("ethics")
        for d in dilemmas:
            assert d.domain == "ethics"

    def test_most_tense_dilemma(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        tense = engine.most_tense_dilemma()
        if report.dilemmas:
            assert tense is not None
            assert tense.tension_score == report.dilemmas[0].tension_score
        else:
            assert tense is None

    def test_most_tense_dilemma_none_on_empty(self):
        from emms.memory.moral import MoralReasoner
        from emms.memory.dilemma import DilemmaEngine
        agent = _make_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        engine = DilemmaEngine(memory=agent.memory, moral_reasoner=reasoner)
        engine.detect_dilemmas()
        assert engine.most_tense_dilemma() is None

    def test_domains_affected_is_list(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert isinstance(report.domains_affected, list)

    def test_mean_tension_in_range(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert 0.0 <= report.mean_tension <= 1.0

    def test_mean_tension_zero_on_empty(self):
        from emms.memory.moral import MoralReasoner
        from emms.memory.dilemma import DilemmaEngine
        agent = _make_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        engine = DilemmaEngine(memory=agent.memory, moral_reasoner=reasoner)
        report = engine.detect_dilemmas()
        assert report.mean_tension == 0.0

    def test_report_summary_is_string(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert isinstance(report.summary(), str)

    def test_dilemma_summary_is_string(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        if report.dilemmas:
            assert isinstance(report.dilemmas[0].summary(), str)

    def test_duration_non_negative(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        assert report.duration_seconds >= 0.0

    def test_framework_fields_are_strings(self):
        agent = _make_rich_emms()
        engine = self._make_reasoner_and_engine(agent)
        report = engine.detect_dilemmas()
        for d in report.dilemmas:
            assert isinstance(d.framework_a, str)
            assert isinstance(d.framework_b, str)

    def test_max_dilemmas_cap(self):
        from emms.memory.moral import MoralReasoner
        from emms.memory.dilemma import DilemmaEngine
        agent = _make_rich_emms()
        reasoner = MoralReasoner(memory=agent.memory)
        reasoner.reason()
        engine = DilemmaEngine(memory=agent.memory, moral_reasoner=reasoner, max_dilemmas=2)
        report = engine.detect_dilemmas()
        assert report.total_dilemmas <= 2


# ---------------------------------------------------------------------------
# TestEMMSFacadeV230
# ---------------------------------------------------------------------------

class TestEMMSFacadeV230:

    def test_map_values_callable(self):
        agent = _make_rich_emms()
        report = agent.map_values()
        assert report is not None

    def test_map_values_with_category(self):
        agent = _make_rich_emms()
        report = agent.map_values(category="moral")
        for v in report.values:
            assert v.category == "moral"

    def test_values_for_category_callable(self):
        agent = _make_rich_emms()
        agent.map_values()
        vals = agent.values_for_category("epistemic")
        assert isinstance(vals, list)

    def test_strongest_value_callable(self):
        agent = _make_rich_emms()
        agent.map_values()
        sv = agent.strongest_value()
        # may be None if min_strength not met, but must not raise
        assert sv is None or hasattr(sv, "strength")

    def test_reason_morally_callable(self):
        agent = _make_rich_emms()
        report = agent.reason_morally()
        assert report is not None

    def test_reason_morally_with_domain(self):
        agent = _make_rich_emms()
        report = agent.reason_morally(domain="ethics")
        for a in report.assessments:
            assert a.domain == "ethics"

    def test_moral_weight_of_callable(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        w = agent.moral_weight_of("nonexistent")
        assert w == 0.0

    def test_assessments_by_framework_callable(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        results = agent.assessments_by_framework("virtue")
        assert isinstance(results, list)

    def test_detect_dilemmas_callable(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        report = agent.detect_dilemmas()
        assert report is not None

    def test_detect_dilemmas_with_domain(self):
        agent = _make_rich_emms()
        agent.reason_morally(domain="ethics")
        report = agent.detect_dilemmas(domain="ethics")
        for d in report.dilemmas:
            assert d.domain == "ethics"

    def test_dilemmas_for_domain_callable(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        agent.detect_dilemmas()
        results = agent.dilemmas_for_domain("ethics")
        assert isinstance(results, list)

    def test_most_tense_dilemma_callable(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        agent.detect_dilemmas()
        dilemma = agent.most_tense_dilemma()
        # may be None but should not raise
        assert dilemma is None or hasattr(dilemma, "tension_score")

    def test_value_mapper_lazy_init(self):
        agent = _make_rich_emms()
        agent.map_values()
        # Second call reuses the same instance
        r1 = agent.map_values()
        r2 = agent.map_values()
        assert r1.total_values == r2.total_values

    def test_moral_reasoner_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.reason_morally()
        r2 = agent.reason_morally()
        assert r1.total_assessed == r2.total_assessed

    def test_dilemma_engine_reuses_moral_reasoner(self):
        agent = _make_rich_emms()
        agent.reason_morally()
        # detect_dilemmas should not fail after reason_morally has populated assessments
        report = agent.detect_dilemmas()
        assert hasattr(report, "total_dilemmas")

    def test_all_nine_facade_methods_exist(self):
        agent = _make_emms()
        for method in (
            "map_values", "values_for_category", "strongest_value",
            "reason_morally", "moral_weight_of", "assessments_by_framework",
            "detect_dilemmas", "dilemmas_for_domain", "most_tense_dilemma",
        ):
            assert hasattr(agent, method), f"EMMS missing method: {method}"


# ---------------------------------------------------------------------------
# TestMCPV230
# ---------------------------------------------------------------------------

class TestMCPV230:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_102(self):
        server = self._make_server()
        assert len(server.tool_definitions) == 102, (
            f"Expected 102 tools, got {len(server.tool_definitions)}"
        )

    def test_emms_map_values_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_map_values" in names

    def test_emms_values_for_category_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_values_for_category" in names

    def test_emms_reason_morally_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_reason_morally" in names

    def test_emms_detect_dilemmas_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_detect_dilemmas" in names

    def test_emms_most_tense_dilemma_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_most_tense_dilemma" in names

    def test_map_values_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_map_values", {})
        assert result.get("ok") is True

    def test_reason_morally_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_reason_morally", {})
        assert result.get("ok") is True

    def test_detect_dilemmas_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_detect_dilemmas", {})
        assert result.get("ok") is True

    def test_most_tense_dilemma_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_most_tense_dilemma", {})
        assert result.get("ok") is True

    def test_values_for_category_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_values_for_category", {"category": "moral"})
        assert result.get("ok") is True


# ---------------------------------------------------------------------------
# TestV230Exports
# ---------------------------------------------------------------------------

class TestV230Exports:

    def test_version_is_0_23_0(self):
        assert emms_pkg.__version__ == "0.23.0"

    def test_value_mapper_importable(self):
        from emms import ValueMapper
        assert ValueMapper is not None

    def test_mapped_value_importable(self):
        from emms import MappedValue
        assert MappedValue is not None

    def test_value_report_importable(self):
        from emms import ValueReport
        assert ValueReport is not None

    def test_moral_reasoner_importable(self):
        from emms import MoralReasoner
        assert MoralReasoner is not None

    def test_moral_assessment_importable(self):
        from emms import MoralAssessment
        assert MoralAssessment is not None

    def test_moral_report_importable(self):
        from emms import MoralReport
        assert MoralReport is not None

    def test_dilemma_engine_importable(self):
        from emms import DilemmaEngine
        assert DilemmaEngine is not None

    def test_ethical_dilemma_importable(self):
        from emms import EthicalDilemma
        assert EthicalDilemma is not None

    def test_dilemma_report_importable(self):
        from emms import DilemmaReport
        assert DilemmaReport is not None

    def test_all_nine_in_all(self):
        for name in (
            "ValueMapper", "MappedValue", "ValueReport",
            "MoralReasoner", "MoralAssessment", "MoralReport",
            "DilemmaEngine", "EthicalDilemma", "DilemmaReport",
        ):
            assert name in emms_pkg.__all__, f"{name} missing from __all__"
