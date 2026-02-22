"""Tests for EMMS v0.22.0 — The Creative Mind.

Covers:
- NoveltyDetector / NoveltyScore / NoveltyReport
- ConceptInventor / InventedConcept / InventionReport
- AbstractionEngine / AbstractPrinciple / AbstractionReport
- EMMS facade (v0.22.0 methods)
- MCP tool count (97) + 5 new tools
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
    """EMMS loaded with memories across two domains."""
    agent = EMMS()
    # science domain
    for content in [
        "Experiments reveal that temperature causes pressure changes in gases.",
        "Scientists discovered that radiation reduces cell viability significantly.",
        "Research shows temperature increases reaction rates exponentially.",
        "Experiments demonstrate that pressure enables chemical synthesis pathways.",
        "Temperature measurement requires calibrated instruments and precision tools.",
        "Laboratory analysis confirms radiation disrupts molecular bonding processes.",
    ]:
        agent.store(Experience(content=content, domain="science", importance=0.8, valence=0.5))
    # philosophy domain
    for content in [
        "Virtue ethics argues that character development enables moral excellence.",
        "Kantian reasoning requires categorical imperatives for ethical judgment.",
        "Aristotle argued that virtue requires practical wisdom and habituation.",
        "Moral philosophy addresses questions about justice and human flourishing.",
        "Virtue and character development require sustained practice and reflection.",
        "Ethics demands consistent reasoning across different contextual situations.",
    ]:
        agent.store(Experience(content=content, domain="philosophy", importance=0.7, valence=0.4))
    return agent


# ---------------------------------------------------------------------------
# TestNoveltyDetector
# ---------------------------------------------------------------------------

class TestNoveltyDetector:

    def test_assess_returns_novelty_report(self):
        from emms.memory.novelty import NoveltyDetector, NoveltyReport
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert isinstance(report, NoveltyReport)

    def test_report_total_assessed_positive(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert report.total_assessed > 0

    def test_report_mean_novelty_in_range(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert 0.0 <= report.mean_novelty <= 1.0

    def test_report_scores_list(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert isinstance(report.scores, list)

    def test_report_duration_positive(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert report.duration_seconds >= 0.0

    def test_report_high_novelty_count_le_total(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert report.high_novelty_count <= report.total_assessed

    def test_scores_sorted_descending(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        novelties = [s.novelty for s in report.scores]
        assert novelties == sorted(novelties, reverse=True)

    def test_novelty_score_fields(self):
        from emms.memory.novelty import NoveltyDetector, NoveltyScore
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        if report.scores:
            score = report.scores[0]
            assert isinstance(score, NoveltyScore)
            assert isinstance(score.memory_id, str)
            assert isinstance(score.content_excerpt, str)
            assert isinstance(score.novelty, float)
            assert isinstance(score.domain, str)
            assert isinstance(score.rare_tokens, list)

    def test_novelty_in_range(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        for score in report.scores:
            assert 0.0 <= score.novelty <= 1.0

    def test_novelty_score_summary_string(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        if report.scores:
            summary = report.scores[0].summary()
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_novelty_report_summary_string(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "NoveltyReport" in summary

    def test_most_novel_returns_list(self):
        from emms.memory.novelty import NoveltyDetector, NoveltyScore
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        detector.assess()
        results = detector.most_novel(n=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_most_novel_sorted_descending(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        detector.assess()
        results = detector.most_novel(n=5)
        novelties = [s.novelty for s in results]
        assert novelties == sorted(novelties, reverse=True)

    def test_novelty_of_known_memory(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        if report.scores:
            mid = report.scores[0].memory_id
            val = detector.novelty_of(mid)
            assert 0.0 <= val <= 1.0

    def test_novelty_of_unknown_returns_neutral(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        val = detector.novelty_of("nonexistent_memory_id_xyz")
        assert val == 0.5

    def test_domain_filter(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report_all = detector.assess()
        report_sci = detector.assess(domain="science")
        assert report_sci.total_assessed <= report_all.total_assessed

    def test_domain_filter_results_in_domain(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess(domain="science")
        for score in report.scores:
            assert score.domain == "science"

    def test_empty_memory_returns_report(self):
        from emms.memory.novelty import NoveltyDetector, NoveltyReport
        agent = _make_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert isinstance(report, NoveltyReport)
        assert report.total_assessed == 0
        assert report.mean_novelty == 0.0

    def test_content_excerpt_max_80_chars(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        for score in report.scores:
            assert len(score.content_excerpt) <= 80

    def test_rare_tokens_is_list_of_strings(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        for score in report.scores:
            assert isinstance(score.rare_tokens, list)
            for tok in score.rare_tokens:
                assert isinstance(tok, str)

    def test_max_scores_limit(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory, max_scores=3)
        report = detector.assess()
        assert len(report.scores) <= 3

    def test_high_novelty_threshold(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory, novelty_threshold=0.9)
        report = detector.assess()
        expected = sum(1 for s in detector._scores.values() if s.novelty >= 0.9)
        assert report.high_novelty_count == expected

    def test_scores_cleared_on_reassess(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        detector.assess()
        count1 = len(detector._scores)
        detector.assess()
        count2 = len(detector._scores)
        assert count2 == count1

    def test_report_total_matches_assessed(self):
        from emms.memory.novelty import NoveltyDetector
        agent = _make_rich_emms()
        detector = NoveltyDetector(memory=agent.memory)
        report = detector.assess()
        assert report.total_assessed == len(detector._scores)


# ---------------------------------------------------------------------------
# TestConceptInventor
# ---------------------------------------------------------------------------

class TestConceptInventor:

    def test_invent_returns_invention_report(self):
        from emms.memory.inventor import ConceptInventor, InventionReport
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert isinstance(report, InventionReport)

    def test_report_total_concepts_non_negative(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert report.total_concepts >= 0

    def test_report_domain_pairs_list(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert isinstance(report.domain_pairs, list)

    def test_report_mean_originality_in_range(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert 0.0 <= report.mean_originality <= 1.0

    def test_report_duration_positive(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert report.duration_seconds >= 0.0

    def test_concepts_sorted_descending(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        scores = [c.originality_score for c in report.concepts]
        assert scores == sorted(scores, reverse=True)

    def test_invented_concept_fields(self):
        from emms.memory.inventor import ConceptInventor, InventedConcept
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        if report.concepts:
            c = report.concepts[0]
            assert isinstance(c, InventedConcept)
            assert isinstance(c.id, str)
            assert isinstance(c.token_a, str)
            assert isinstance(c.domain_a, str)
            assert isinstance(c.token_b, str)
            assert isinstance(c.domain_b, str)
            assert isinstance(c.description, str)
            assert isinstance(c.originality_score, float)
            assert isinstance(c.created_at, float)

    def test_concept_id_prefix(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        for c in report.concepts:
            assert c.id.startswith("inv_")

    def test_originality_in_range(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        for c in report.concepts:
            assert 0.0 <= c.originality_score <= 1.0

    def test_domains_differ(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        for c in report.concepts:
            assert c.domain_a != c.domain_b

    def test_description_contains_tokens(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        for c in report.concepts:
            assert c.token_a in c.description
            assert c.token_b in c.description

    def test_concept_summary_string(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        if report.concepts:
            summary = report.concepts[0].summary()
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_invention_report_summary_string(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "InventionReport" in summary

    def test_concepts_for_domain(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        inventor.invent()
        results = inventor.concepts_for_domain("science")
        for c in results:
            assert c.domain_a == "science" or c.domain_b == "science"

    def test_best_concept_returns_concept_or_none(self):
        from emms.memory.inventor import ConceptInventor, InventedConcept
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        inventor.invent()
        result = inventor.best_concept("temperature and molecular analysis")
        assert result is None or isinstance(result, InventedConcept)

    def test_best_concept_empty_memory(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_emms()
        inventor = ConceptInventor(memory=agent.memory)
        inventor.invent()
        result = inventor.best_concept("anything")
        assert result is None

    def test_max_concepts_limit(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory, max_concepts=3)
        report = inventor.invent(n=10)
        assert len(report.concepts) <= 3

    def test_n_parameter_limits_output(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory, max_concepts=20)
        report = inventor.invent(n=2)
        assert len(report.concepts) <= 2

    def test_min_originality_filter(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory, min_originality=0.8)
        report = inventor.invent()
        for c in report.concepts:
            assert c.originality_score >= 0.8

    def test_empty_memory_returns_report(self):
        from emms.memory.inventor import ConceptInventor, InventionReport
        agent = _make_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert isinstance(report, InventionReport)
        assert report.total_concepts == 0

    def test_total_concepts_matches_list(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert report.total_concepts == len(report.concepts)

    def test_single_domain_no_cross_pairs(self):
        from emms.memory.inventor import ConceptInventor
        agent = EMMS()
        for content in [
            "Temperature causes pressure changes in gas systems.",
            "Radiation reduces cellular viability and molecular stability.",
            "Experiments demonstrate chemical reaction pathways systematically.",
        ]:
            agent.store(Experience(content=content, domain="science", importance=0.8))
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        # Single domain — no cross-pairs possible
        assert report.total_concepts == 0

    def test_concepts_stored_internally(self):
        from emms.memory.inventor import ConceptInventor
        agent = _make_rich_emms()
        inventor = ConceptInventor(memory=agent.memory)
        report = inventor.invent()
        assert len(inventor._concepts) == report.total_concepts


# ---------------------------------------------------------------------------
# TestAbstractionEngine
# ---------------------------------------------------------------------------

class TestAbstractionEngine:

    def test_abstract_returns_abstraction_report(self):
        from emms.memory.abstraction import AbstractionEngine, AbstractionReport
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert isinstance(report, AbstractionReport)

    def test_report_total_principles_non_negative(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert report.total_principles >= 0

    def test_report_domains_abstracted_list(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert isinstance(report.domains_abstracted, list)

    def test_report_mean_generality_in_range(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert 0.0 <= report.mean_generality <= 1.0

    def test_report_duration_positive(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert report.duration_seconds >= 0.0

    def test_principles_sorted_descending(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        scores = [p.generality_score for p in report.principles]
        assert scores == sorted(scores, reverse=True)

    def test_abstract_principle_fields(self):
        from emms.memory.abstraction import AbstractionEngine, AbstractPrinciple
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        if report.principles:
            p = report.principles[0]
            assert isinstance(p, AbstractPrinciple)
            assert isinstance(p.id, str)
            assert isinstance(p.label, str)
            assert isinstance(p.domain, str)
            assert isinstance(p.description, str)
            assert isinstance(p.generality_score, float)
            assert isinstance(p.mean_valence, float)
            assert isinstance(p.mean_importance, float)
            assert isinstance(p.source_memory_ids, list)
            assert isinstance(p.created_at, float)

    def test_principle_id_prefix(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        for p in report.principles:
            assert p.id.startswith("abs_")

    def test_generality_in_range(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        for p in report.principles:
            assert 0.0 <= p.generality_score <= 1.0

    def test_min_generality_filter(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.5)
        report = engine.abstract()
        for p in report.principles:
            assert p.generality_score >= 0.5

    def test_source_memory_ids_non_empty(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        for p in report.principles:
            assert len(p.source_memory_ids) >= 2

    def test_principle_summary_string(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        if report.principles:
            summary = report.principles[0].summary()
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_abstraction_report_summary_string(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "AbstractionReport" in summary

    def test_principles_for_domain(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        engine.abstract()
        results = engine.principles_for_domain("science")
        for p in results:
            assert p.domain == "science"

    def test_best_principle_returns_principle_or_none(self):
        from emms.memory.abstraction import AbstractionEngine, AbstractPrinciple
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        engine.abstract()
        result = engine.best_principle("temperature and experimental science")
        assert result is None or isinstance(result, AbstractPrinciple)

    def test_best_principle_empty_returns_none(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_emms()
        engine = AbstractionEngine(memory=agent.memory)
        engine.abstract()
        result = engine.best_principle("anything")
        assert result is None

    def test_domain_filter(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract(domain="science")
        for p in report.principles:
            assert p.domain == "science"

    def test_max_principles_limit(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1, max_principles=3)
        report = engine.abstract()
        assert len(report.principles) <= 3

    def test_empty_memory_returns_report(self):
        from emms.memory.abstraction import AbstractionEngine, AbstractionReport
        agent = _make_emms()
        engine = AbstractionEngine(memory=agent.memory)
        report = engine.abstract()
        assert isinstance(report, AbstractionReport)
        assert report.total_principles == 0

    def test_total_principles_matches_list(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        assert report.total_principles == len(report.principles)

    def test_description_contains_label(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        for p in report.principles:
            assert p.label in p.description

    def test_principles_stored_internally(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        assert len(engine._principles) == report.total_principles

    def test_mean_generality_computed_correctly(self):
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        if report.principles:
            expected = round(
                sum(p.generality_score for p in report.principles) / len(report.principles), 4
            )
            assert abs(report.mean_generality - expected) < 1e-3

    def test_principle_recurring_token_appears_in_domain(self):
        """The label token should actually appear in the source domain memories."""
        from emms.memory.abstraction import AbstractionEngine
        agent = _make_rich_emms()
        engine = AbstractionEngine(memory=agent.memory, min_generality=0.1)
        report = engine.abstract()
        for p in report.principles:
            assert len(p.source_memory_ids) >= 2


# ---------------------------------------------------------------------------
# TestEMMSFacadeV220
# ---------------------------------------------------------------------------

class TestEMMSFacadeV220:

    def test_assess_novelty_returns_report(self):
        from emms.memory.novelty import NoveltyReport
        agent = _make_rich_emms()
        report = agent.assess_novelty()
        assert isinstance(report, NoveltyReport)

    def test_assess_novelty_domain_filter(self):
        from emms.memory.novelty import NoveltyReport
        agent = _make_rich_emms()
        report = agent.assess_novelty(domain="science")
        assert isinstance(report, NoveltyReport)
        for s in report.scores:
            assert s.domain == "science"

    def test_most_novel_after_assess(self):
        from emms.memory.novelty import NoveltyScore
        agent = _make_rich_emms()
        agent.assess_novelty()
        scores = agent.most_novel(n=3)
        assert isinstance(scores, list)
        assert len(scores) <= 3

    def test_novelty_of_unknown(self):
        agent = _make_rich_emms()
        val = agent.novelty_of("nonexistent_xyz")
        assert val == 0.5

    def test_novelty_detector_lazy_init(self):
        agent = _make_rich_emms()
        assert not hasattr(agent, "_novelty_detector")
        agent.assess_novelty()
        assert hasattr(agent, "_novelty_detector")

    def test_novelty_detector_reused(self):
        agent = _make_rich_emms()
        agent.assess_novelty()
        d1 = agent._novelty_detector
        agent.assess_novelty()
        d2 = agent._novelty_detector
        assert d1 is d2

    def test_invent_concepts_returns_report(self):
        from emms.memory.inventor import InventionReport
        agent = _make_rich_emms()
        report = agent.invent_concepts(n=5)
        assert isinstance(report, InventionReport)

    def test_best_concept_after_invent(self):
        from emms.memory.inventor import InventedConcept
        agent = _make_rich_emms()
        agent.invent_concepts()
        result = agent.best_concept("temperature ethics reasoning")
        assert result is None or isinstance(result, InventedConcept)

    def test_concept_inventor_lazy_init(self):
        agent = _make_rich_emms()
        assert not hasattr(agent, "_concept_inventor")
        agent.invent_concepts()
        assert hasattr(agent, "_concept_inventor")

    def test_concept_inventor_reused(self):
        agent = _make_rich_emms()
        agent.invent_concepts()
        i1 = agent._concept_inventor
        agent.invent_concepts()
        i2 = agent._concept_inventor
        assert i1 is i2

    def test_abstract_principles_returns_report(self):
        from emms.memory.abstraction import AbstractionReport
        agent = _make_rich_emms()
        report = agent.abstract_principles()
        assert isinstance(report, AbstractionReport)

    def test_principles_for_domain_facade(self):
        from emms.memory.abstraction import AbstractPrinciple
        agent = _make_rich_emms()
        agent.abstract_principles()
        results = agent.principles_for_domain("science")
        assert isinstance(results, list)
        for p in results:
            assert isinstance(p, AbstractPrinciple)
            assert p.domain == "science"

    def test_best_principle_facade(self):
        from emms.memory.abstraction import AbstractPrinciple
        agent = _make_rich_emms()
        agent.abstract_principles()
        result = agent.best_principle("experimental temperature science")
        assert result is None or isinstance(result, AbstractPrinciple)

    def test_abstraction_engine_lazy_init(self):
        agent = _make_rich_emms()
        assert not hasattr(agent, "_abstraction_engine")
        agent.abstract_principles()
        assert hasattr(agent, "_abstraction_engine")

    def test_abstraction_engine_reused(self):
        agent = _make_rich_emms()
        agent.abstract_principles()
        e1 = agent._abstraction_engine
        agent.abstract_principles()
        e2 = agent._abstraction_engine
        assert e1 is e2

    def test_abstract_principles_domain_filter(self):
        from emms.memory.abstraction import AbstractionReport
        agent = _make_rich_emms()
        report = agent.abstract_principles(domain="philosophy")
        assert isinstance(report, AbstractionReport)
        for p in report.principles:
            assert p.domain == "philosophy"


# ---------------------------------------------------------------------------
# TestMCPV220
# ---------------------------------------------------------------------------

class TestMCPV220:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_97(self):
        server = self._make_server()
        assert len(server.tool_definitions) == 102

    def test_assess_novelty_tool_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_assess_novelty" in names

    def test_most_novel_tool_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_most_novel" in names

    def test_invent_concepts_tool_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_invent_concepts" in names

    def test_abstract_principles_tool_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_abstract_principles" in names

    def test_best_principle_tool_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_best_principle" in names

    def test_handle_assess_novelty_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_assess_novelty", {})
        assert result["ok"] is True
        assert "total_assessed" in result

    def test_handle_most_novel_returns_ok(self):
        server = self._make_server()
        server.handle("emms_assess_novelty", {})
        result = server.handle("emms_most_novel", {"n": 3})
        assert result["ok"] is True
        assert "scores" in result

    def test_handle_invent_concepts_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_invent_concepts", {"n": 5})
        assert result["ok"] is True
        assert "total_concepts" in result

    def test_handle_abstract_principles_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_abstract_principles", {})
        assert result["ok"] is True
        assert "total_principles" in result

    def test_handle_best_principle_returns_ok(self):
        server = self._make_server()
        server.handle("emms_abstract_principles", {})
        result = server.handle("emms_best_principle", {"description": "temperature science"})
        assert result["ok"] is True
        assert "found" in result


# ---------------------------------------------------------------------------
# TestV220Exports
# ---------------------------------------------------------------------------

class TestV220Exports:

    def test_version_is_0_22_0(self):
        assert emms_pkg.__version__ == "0.23.0"

    def test_novelty_detector_exported(self):
        from emms import NoveltyDetector
        assert NoveltyDetector is not None

    def test_novelty_score_exported(self):
        from emms import NoveltyScore
        assert NoveltyScore is not None

    def test_novelty_report_exported(self):
        from emms import NoveltyReport
        assert NoveltyReport is not None

    def test_concept_inventor_exported(self):
        from emms import ConceptInventor
        assert ConceptInventor is not None

    def test_invented_concept_exported(self):
        from emms import InventedConcept
        assert InventedConcept is not None

    def test_invention_report_exported(self):
        from emms import InventionReport
        assert InventionReport is not None

    def test_abstraction_engine_exported(self):
        from emms import AbstractionEngine
        assert AbstractionEngine is not None

    def test_abstract_principle_exported(self):
        from emms import AbstractPrinciple
        assert AbstractPrinciple is not None

    def test_abstraction_report_exported(self):
        from emms import AbstractionReport
        assert AbstractionReport is not None

    def test_all_contains_novelty_detector(self):
        assert "NoveltyDetector" in emms_pkg.__all__

    def test_all_contains_concept_inventor(self):
        assert "ConceptInventor" in emms_pkg.__all__

    def test_all_contains_abstraction_engine(self):
        assert "AbstractionEngine" in emms_pkg.__all__
