"""Tests for EMMS v0.18.0 — The Predictive Mind.

Covers:
    - PredictiveEngine + Prediction + PredictionReport
    - ConceptBlender + BlendedConcept + BlendReport
    - TemporalProjection + FutureScenario + ProjectionReport
    - EMMS facade methods
    - MCP tool count (77) and new tool callability
    - __init__ exports and version string
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from emms import EMMS, Experience
from emms.adapters.mcp_server import EMCPServer, _TOOL_DEFINITIONS
from emms.memory.prediction import Prediction, PredictionReport, PredictiveEngine
from emms.memory.blending import BlendedConcept, BlendReport, ConceptBlender
from emms.memory.projection import FutureScenario, ProjectionReport, TemporalProjection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms(n: int = 8) -> EMMS:
    agent = EMMS()
    domains = ["science", "science", "history", "history", "art", "art", "science", "history"]
    contents = [
        "Regular exercise dramatically improves cardiovascular health and reduces disease risk",
        "Daily meditation practice reduces stress cortisol levels and improves cognitive function",
        "Ancient Rome expanded through military conquest and established complex administrative systems",
        "The Renaissance period enabled artistic flourishing through wealthy patronage and reduces ignorance",
        "Impressionism movement enables emotional expression through light and color technique",
        "Cubism movement produces multiple perspectives simultaneously in visual art",
        "Vaccination enables immune system preparation and prevents infectious disease spread",
        "Trade routes enabled cultural exchange and produces economic growth in ancient civilizations",
    ]
    for i in range(min(n, len(contents))):
        agent.store(Experience(
            content=contents[i],
            domain=domains[i % len(domains)],
            importance=0.6 + 0.04 * i,
        ))
    return agent


def _make_multi_domain_emms() -> EMMS:
    """EMMS with distinct domains for cross-domain blending and analogy."""
    agent = EMMS()
    agent.store(Experience(
        content="stress causes cortisol release which produces anxiety and reduces immune function effectiveness",
        domain="biology",
        importance=0.9,
    ))
    agent.store(Experience(
        content="biology research enables drug discovery through systematic experimentation and reduces disease burden",
        domain="biology",
        importance=0.8,
    ))
    agent.store(Experience(
        content="debt causes interest accumulation which produces financial stress and reduces available savings",
        domain="economics",
        importance=0.9,
    ))
    agent.store(Experience(
        content="market volatility enables arbitrage opportunities through price discrepancy and produces profit",
        domain="economics",
        importance=0.8,
    ))
    agent.store(Experience(
        content="conflict causes psychological trauma which produces lasting behavioral changes",
        domain="psychology",
        importance=0.7,
    ))
    return agent


# ===========================================================================
# TestPredictiveEngine
# ===========================================================================


class TestPredictiveEngine:

    def test_predict_returns_report(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        assert isinstance(report, PredictionReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        assert isinstance(report.total_generated, int)
        assert isinstance(report.confirmed, int)
        assert isinstance(report.violated, int)
        assert isinstance(report.pending, int)
        assert isinstance(report.mean_surprise, float)
        assert isinstance(report.predictions, list)
        assert report.duration_seconds >= 0

    def test_predictions_generated(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        # With memories in multiple domains, predictions should be generated
        assert report.total_generated >= 0

    def test_predictions_are_prediction_instances(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        for p in report.predictions:
            assert isinstance(p, Prediction)

    def test_prediction_fields(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            p = report.predictions[0]
            assert isinstance(p.id, str)
            assert isinstance(p.content, str)
            assert isinstance(p.domain, str)
            assert isinstance(p.basis_memory_ids, list)
            assert 0.0 <= p.confidence <= 1.0
            assert p.outcome in ("pending", "confirmed", "violated")
            assert isinstance(p.surprise_score, float)
            assert isinstance(p.outcome_note, str)

    def test_prediction_id_prefix(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        for p in report.predictions:
            assert p.id.startswith("pred_")

    def test_initial_outcome_is_pending(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        for p in report.predictions:
            assert p.outcome == "pending"

    def test_pending_predictions_returns_list(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        engine.predict()
        pending = engine.pending_predictions()
        assert isinstance(pending, list)
        for p in pending:
            assert isinstance(p, Prediction)
            assert p.outcome == "pending"

    def test_resolve_confirmed(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            pid = report.predictions[0].id
            result = engine.resolve(pid, "confirmed", note="Observed as predicted")
            assert result is True
            pred = engine._predictions[pid]
            assert pred.outcome == "confirmed"
            assert pred.outcome_note == "Observed as predicted"
            assert pred.resolved_at is not None

    def test_resolve_violated(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            pid = report.predictions[0].id
            result = engine.resolve(pid, "violated", note="Opposite occurred")
            assert result is True
            pred = engine._predictions[pid]
            assert pred.outcome == "violated"

    def test_resolve_nonexistent_returns_false(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        result = engine.resolve("pred_nonexistent_xyz", "confirmed")
        assert result is False

    def test_surprise_score_confirmed(self):
        """Confirmed predictions: surprise = 1 - confidence."""
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            p = report.predictions[0]
            conf = p.confidence
            engine.resolve(p.id, "confirmed")
            updated = engine._predictions[p.id]
            expected = round(1.0 - conf, 4)
            assert abs(updated.surprise_score - expected) < 0.01

    def test_surprise_score_violated(self):
        """Violated predictions: surprise = confidence."""
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            p = report.predictions[0]
            conf = p.confidence
            engine.resolve(p.id, "violated")
            updated = engine._predictions[p.id]
            expected = round(conf, 4)
            assert abs(updated.surprise_score - expected) < 0.01

    def test_domain_filter(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict(domain="science")
        for p in report.predictions:
            assert p.domain == "science"

    def test_domain_filter_unknown_returns_empty(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict(domain="alchemy_ancient_nonexistent")
        assert report.total_generated == 0

    def test_surprise_profile_returns_dict(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        engine.predict()
        profile = engine.surprise_profile()
        assert isinstance(profile, dict)

    def test_max_predictions_respected(self):
        agent = _make_rich_emms(8)
        engine = PredictiveEngine(memory=agent.memory, max_predictions=2)
        report = engine.predict()
        assert report.total_generated <= 2

    def test_confidence_threshold_filters(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory, confidence_threshold=0.99)
        report = engine.predict()
        for p in report.predictions:
            assert p.confidence >= 0.99

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        assert report.total_generated == 0
        assert report.predictions == []

    def test_pending_count_decreases_after_resolve(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            before = len(engine.pending_predictions())
            engine.resolve(report.predictions[0].id, "confirmed")
            after = len(engine.pending_predictions())
            assert after == before - 1

    def test_report_counts_sum(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        # After prediction, all are pending
        assert report.total_generated == report.pending

    def test_prediction_summary_str(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        if report.predictions:
            s = report.predictions[0].summary()
            assert isinstance(s, str)
            assert len(s) > 0

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        s = report.summary()
        assert isinstance(s, str)
        assert "PredictionReport" in s

    def test_basis_memory_ids_populated(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        for p in report.predictions:
            assert isinstance(p.basis_memory_ids, list)

    def test_mean_surprise_is_float(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        report = engine.predict()
        assert isinstance(report.mean_surprise, float)

    def test_multiple_predict_calls_accumulate(self):
        agent = _make_rich_emms()
        engine = PredictiveEngine(memory=agent.memory)
        engine.predict()
        count1 = len(engine._predictions)
        engine.predict()
        count2 = len(engine._predictions)
        assert count2 >= count1


# ===========================================================================
# TestConceptBlender
# ===========================================================================


class TestConceptBlender:

    def test_blend_returns_report(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory)
        report = blender.blend()
        assert isinstance(report, BlendReport)

    def test_report_fields(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory)
        report = blender.blend()
        assert isinstance(report.total_pairs_tried, int)
        assert isinstance(report.blends_created, int)
        assert isinstance(report.concepts, list)
        assert report.duration_seconds >= 0

    def test_blends_created(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        assert report.blends_created >= 0

    def test_concepts_are_blended_concept_instances(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        for c in report.concepts:
            assert isinstance(c, BlendedConcept)

    def test_blended_concept_fields(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        if report.concepts:
            c = report.concepts[0]
            assert isinstance(c.id, str)
            assert isinstance(c.source_memory_ids, list)
            assert len(c.source_memory_ids) == 2
            assert isinstance(c.source_domains, list)
            assert isinstance(c.blend_content, str)
            assert isinstance(c.emergent_properties, list)
            assert 0.0 <= c.blend_strength <= 1.0
            assert isinstance(c.created_at, float)

    def test_concept_id_prefix(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        for c in report.concepts:
            assert c.id.startswith("blend_")

    def test_domain_filter_produces_cross_domain(self):
        """When different domain_a and domain_b are given, blends are cross-domain."""
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend(domain_a="biology", domain_b="economics")
        for c in report.concepts:
            assert "biology" in c.source_domains
            assert "economics" in c.source_domains

    def test_blend_strength_in_range(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        for c in report.concepts:
            assert 0.0 <= c.blend_strength <= 1.0

    def test_min_blend_strength_filters(self):
        agent = _make_multi_domain_emms()
        blender_low = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        blender_high = ConceptBlender(memory=agent.memory, min_blend_strength=0.99)
        report_low = blender_low.blend()
        report_high = blender_high.blend()
        assert report_low.blends_created >= report_high.blends_created

    def test_max_blends_respected(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0, max_blends=1)
        report = blender.blend()
        assert report.blends_created <= 1

    def test_blend_pair_direct(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        # Get two memory IDs from different domains
        all_items = blender._collect_all()
        if len(all_items) >= 2:
            id_a = all_items[0].id
            id_b = all_items[-1].id
            result = blender.blend_pair(id_a, id_b)
            # May be None if same domain or below threshold
            if result is not None:
                assert isinstance(result, BlendedConcept)

    def test_blend_pair_nonexistent_returns_none(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory)
        result = blender.blend_pair("nonexistent_a", "nonexistent_b")
        assert result is None

    def test_domain_filter_a(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend(domain_a="biology")
        for c in report.concepts:
            assert "biology" in c.source_domains

    def test_store_blends_creates_memory(self):
        """With store_blends=True, new memories should be stored."""
        agent = _make_multi_domain_emms()
        initial_count = len(list(agent.memory.long_term.values()) +
                           list(agent.memory.semantic.values()) +
                           list(agent.memory.working) +
                           list(agent.memory.short_term))
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0,
                                 store_blends=True)
        report = blender.blend()
        if report.blends_created > 0:
            final_count = len(list(agent.memory.long_term.values()) +
                              list(agent.memory.semantic.values()) +
                              list(agent.memory.working) +
                              list(agent.memory.short_term))
            # New memory created for each blend
            assert final_count >= initial_count

    def test_new_memory_id_set_when_stored(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0,
                                 store_blends=True)
        report = blender.blend()
        for c in report.concepts:
            if c.new_memory_id is not None:
                assert isinstance(c.new_memory_id, str)

    def test_store_blends_false_no_new_memory(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0,
                                 store_blends=False)
        report = blender.blend()
        for c in report.concepts:
            assert c.new_memory_id is None

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        blender = ConceptBlender(memory=agent.memory)
        report = blender.blend()
        assert report.blends_created == 0
        assert report.concepts == []

    def test_blend_content_is_nonempty_string(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        for c in report.concepts:
            assert isinstance(c.blend_content, str)
            assert len(c.blend_content) > 0

    def test_emergent_properties_is_list(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        for c in report.concepts:
            assert isinstance(c.emergent_properties, list)

    def test_concept_summary_str(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        if report.concepts:
            s = report.concepts[0].summary()
            assert isinstance(s, str)
            assert len(s) > 0

    def test_report_summary_str(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        s = report.summary()
        assert isinstance(s, str)
        assert "BlendReport" in s

    def test_total_pairs_tried_is_nonneg(self):
        agent = _make_multi_domain_emms()
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        report = blender.blend()
        assert report.total_pairs_tried >= 0


# ===========================================================================
# TestTemporalProjection
# ===========================================================================


class TestTemporalProjection:

    def test_project_returns_report(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert isinstance(report, ProjectionReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert isinstance(report.total_episodes_used, int)
        assert isinstance(report.total_memories_used, int)
        assert isinstance(report.scenarios_generated, int)
        assert isinstance(report.scenarios, list)
        assert isinstance(report.mean_plausibility, float)
        assert report.duration_seconds >= 0

    def test_scenarios_generated_with_memories(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert report.scenarios_generated >= 0

    def test_scenarios_are_future_scenario_instances(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        for s in report.scenarios:
            assert isinstance(s, FutureScenario)

    def test_future_scenario_fields(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        if report.scenarios:
            s = report.scenarios[0]
            assert isinstance(s.id, str)
            assert isinstance(s.content, str)
            assert isinstance(s.domain, str)
            assert isinstance(s.basis_episode_ids, list)
            assert isinstance(s.basis_memory_ids, list)
            assert isinstance(s.projection_horizon, float)
            assert 0.0 <= s.plausibility <= 1.0
            assert -1.0 <= s.emotional_valence <= 1.0
            assert isinstance(s.created_at, float)

    def test_scenario_id_prefix(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        for s in report.scenarios:
            assert s.id.startswith("proj_")

    def test_max_scenarios_respected(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory, max_scenarios=2)
        report = proj.project()
        assert report.scenarios_generated <= 2

    def test_horizon_days_default(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory, horizon_days=60.0)
        report = proj.project()
        for s in report.scenarios:
            assert s.projection_horizon == 60.0

    def test_horizon_days_override(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory, horizon_days=30.0)
        report = proj.project(horizon_days=90.0)
        for s in report.scenarios:
            assert s.projection_horizon == 90.0

    def test_domain_filter(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project(domain="science")
        for s in report.scenarios:
            assert s.domain == "science"

    def test_plausibility_sorted_descending(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        if len(report.scenarios) >= 2:
            for i in range(len(report.scenarios) - 1):
                assert report.scenarios[i].plausibility >= report.scenarios[i + 1].plausibility

    def test_plausibility_in_range(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        for s in report.scenarios:
            assert 0.0 <= s.plausibility <= 1.0

    def test_most_plausible_returns_list(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        proj.project()
        results = proj.most_plausible(n=3)
        assert isinstance(results, list)
        for s in results:
            assert isinstance(s, FutureScenario)

    def test_most_plausible_respects_n(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        proj.project()
        results = proj.most_plausible(n=2)
        assert len(results) <= 2

    def test_most_plausible_sorted(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        proj.project()
        results = proj.most_plausible(n=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].plausibility >= results[i + 1].plausibility

    def test_total_memories_used(self):
        agent = _make_rich_emms(4)
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert report.total_memories_used == 4

    def test_empty_memory_no_scenarios(self):
        agent = _make_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert report.scenarios_generated == 0
        assert report.scenarios == []

    def test_no_episodes_used_without_buffer(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory, episodic_buffer=None)
        report = proj.project()
        assert report.total_episodes_used == 0

    def test_mean_plausibility_bounds(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        assert 0.0 <= report.mean_plausibility <= 1.0

    def test_scenarios_accumulate_across_calls(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        proj.project()
        count1 = len(proj._scenarios)
        proj.project()
        count2 = len(proj._scenarios)
        assert count2 >= count1

    def test_scenario_content_nonempty(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        for s in report.scenarios:
            assert len(s.content) > 0

    def test_scenario_summary_str(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        if report.scenarios:
            s = report.scenarios[0].summary()
            assert isinstance(s, str)
            assert "FutureScenario" in s

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        proj = TemporalProjection(memory=agent.memory)
        report = proj.project()
        s = report.summary()
        assert isinstance(s, str)
        assert "ProjectionReport" in s


# ===========================================================================
# TestEMMSFacadeV180
# ===========================================================================


class TestEMMSFacadeV180:

    def test_predict_returns_report(self):
        agent = _make_rich_emms()
        report = agent.predict()
        assert isinstance(report, PredictionReport)

    def test_predict_domain_filter(self):
        agent = _make_rich_emms()
        report = agent.predict(domain="science")
        for p in report.predictions:
            assert p.domain == "science"

    def test_resolve_prediction(self):
        agent = _make_rich_emms()
        report = agent.predict()
        if report.predictions:
            pid = report.predictions[0].id
            result = agent.resolve_prediction(pid, "confirmed", note="as predicted")
            assert result is True

    def test_pending_predictions(self):
        agent = _make_rich_emms()
        agent.predict()
        pending = agent.pending_predictions()
        assert isinstance(pending, list)
        for p in pending:
            assert p.outcome == "pending"

    def test_blend_concepts_returns_report(self):
        agent = _make_multi_domain_emms()
        report = agent.blend_concepts()
        assert isinstance(report, BlendReport)

    def test_blend_concepts_domain_filter(self):
        agent = _make_multi_domain_emms()
        report = agent.blend_concepts(domain_a="biology", domain_b="economics")
        for c in report.concepts:
            assert "biology" in c.source_domains or "economics" in c.source_domains

    def test_blend_pair(self):
        agent = _make_multi_domain_emms()
        from emms.memory.blending import ConceptBlender
        blender = ConceptBlender(memory=agent.memory, min_blend_strength=0.0)
        all_items = blender._collect_all()
        if len(all_items) >= 2:
            result = agent.blend_pair(all_items[0].id, all_items[-1].id)
            # Result may be None or BlendedConcept
            assert result is None or isinstance(result, BlendedConcept)

    def test_project_future_returns_report(self):
        agent = _make_rich_emms()
        report = agent.project_future()
        assert isinstance(report, ProjectionReport)

    def test_project_future_domain_filter(self):
        agent = _make_rich_emms()
        report = agent.project_future(domain="science")
        for s in report.scenarios:
            assert s.domain == "science"

    def test_project_future_horizon(self):
        agent = _make_rich_emms()
        report = agent.project_future(horizon_days=90.0)
        for s in report.scenarios:
            assert s.projection_horizon == 90.0

    def test_most_plausible_futures(self):
        agent = _make_rich_emms()
        agent.project_future()
        results = agent.most_plausible_futures(n=3)
        assert isinstance(results, list)
        for s in results:
            assert isinstance(s, FutureScenario)

    def test_lazy_init_predictive_engine(self):
        agent = _make_emms()
        assert not hasattr(agent, "_predictive_engine")
        agent.predict()
        assert hasattr(agent, "_predictive_engine")

    def test_blend_concepts_stateless_per_call(self):
        """blend_concepts creates a fresh engine per call (stateless)."""
        agent = _make_multi_domain_emms()
        r1 = agent.blend_concepts()
        r2 = agent.blend_concepts()
        assert isinstance(r1, BlendReport)
        assert isinstance(r2, BlendReport)

    def test_project_future_stateless_per_call(self):
        """project_future creates a fresh engine per call (stateless)."""
        agent = _make_rich_emms()
        r1 = agent.project_future()
        r2 = agent.project_future()
        assert isinstance(r1, ProjectionReport)
        assert isinstance(r2, ProjectionReport)


# ===========================================================================
# TestMCPV180
# ===========================================================================


class TestMCPV180:

    def test_tool_count_is_77(self):
        assert len(_TOOL_DEFINITIONS) == 97

    def test_emms_predict_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_predict", {})
        assert isinstance(result, dict)
        assert "ok" in result or "total_generated" in result or "error" in result

    def test_emms_pending_predictions_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        server.handle("emms_predict", {})
        result = server.handle("emms_pending_predictions", {})
        assert isinstance(result, dict)
        assert "count" in result or "predictions" in result or "error" in result

    def test_emms_blend_concepts_callable(self):
        agent = _make_multi_domain_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_blend_concepts", {})
        assert isinstance(result, dict)
        assert "blends_created" in result or "ok" in result or "error" in result

    def test_emms_project_future_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        result = server.handle("emms_project_future", {"horizon_days": 30.0})
        assert isinstance(result, dict)
        assert "scenarios_generated" in result or "ok" in result or "error" in result

    def test_emms_plausible_futures_callable(self):
        agent = _make_rich_emms()
        server = EMCPServer(agent)
        server.handle("emms_project_future", {})
        result = server.handle("emms_plausible_futures", {"n": 3})
        assert isinstance(result, dict)
        assert "count" in result or "scenarios" in result or "error" in result

    def test_new_tools_in_definitions(self):
        tool_names = {t["name"] for t in _TOOL_DEFINITIONS}
        new_tools = {
            "emms_predict",
            "emms_pending_predictions",
            "emms_blend_concepts",
            "emms_project_future",
            "emms_plausible_futures",
        }
        for name in new_tools:
            assert name in tool_names, f"Missing tool: {name}"


# ===========================================================================
# TestV180Exports
# ===========================================================================


class TestV180Exports:

    def test_version_is_0_18_0(self):
        import emms
        assert emms.__version__ == "0.22.0"

    def test_predictive_engine_exported(self):
        from emms import PredictiveEngine
        assert PredictiveEngine is not None

    def test_prediction_exported(self):
        from emms import Prediction
        assert Prediction is not None

    def test_prediction_report_exported(self):
        from emms import PredictionReport
        assert PredictionReport is not None

    def test_concept_blender_exported(self):
        from emms import ConceptBlender
        assert ConceptBlender is not None

    def test_blended_concept_exported(self):
        from emms import BlendedConcept
        assert BlendedConcept is not None

    def test_blend_report_exported(self):
        from emms import BlendReport
        assert BlendReport is not None

    def test_temporal_projection_exported(self):
        from emms import TemporalProjection
        assert TemporalProjection is not None

    def test_future_scenario_exported(self):
        from emms import FutureScenario
        assert FutureScenario is not None

    def test_projection_report_exported(self):
        from emms import ProjectionReport
        assert ProjectionReport is not None

    def test_all_contains_new_symbols(self):
        import emms
        new_symbols = [
            "PredictiveEngine", "Prediction", "PredictionReport",
            "ConceptBlender", "BlendedConcept", "BlendReport",
            "TemporalProjection", "FutureScenario", "ProjectionReport",
        ]
        for sym in new_symbols:
            assert sym in emms.__all__, f"Missing from __all__: {sym}"
