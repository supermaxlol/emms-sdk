"""Tests for consciousness-inspired identity enrichment modules."""

import pytest
import time
from emms.core.models import Experience
from emms.identity.consciousness import (
    ContinuousNarrator,
    MeaningMaker,
    TemporalIntegrator,
    EgoBoundaryTracker,
    NarrativeEntry,
)


# ---------------------------------------------------------------------------
# ContinuousNarrator
# ---------------------------------------------------------------------------

class TestContinuousNarrator:
    def test_integrate_returns_entry(self):
        narrator = ContinuousNarrator()
        exp = Experience(content="Market rose 5% today", domain="finance", importance=0.8)
        entry = narrator.integrate(exp)
        assert isinstance(entry, NarrativeEntry)
        assert entry.domain == "finance"

    def test_integrate_updates_themes(self):
        narrator = ContinuousNarrator()
        exp = Experience(content="The stock market analysis shows growth patterns", domain="finance")
        narrator.integrate(exp)
        assert len(narrator.themes) > 0

    def test_coherence_starts_high(self):
        narrator = ContinuousNarrator()
        assert narrator.coherence >= 0.8

    def test_narrative_generation(self):
        narrator = ContinuousNarrator()
        for i in range(5):
            exp = Experience(
                content=f"Financial analysis report number {i} shows market trends",
                domain="finance",
                importance=0.7,
            )
            narrator.integrate(exp)

        narrative = narrator.build_narrative("TestBot")
        assert "TestBot" in narrative
        assert "5 experiences" in narrative
        assert "finance" in narrative.lower() or "financial" in narrative.lower()

    def test_empty_narrative(self):
        narrator = ContinuousNarrator()
        narrative = narrator.build_narrative("EmptyBot")
        assert "EmptyBot" in narrative

    def test_entries_capped(self):
        narrator = ContinuousNarrator(max_entries=10)
        for i in range(20):
            exp = Experience(content=f"Event {i}", domain="test")
            narrator.integrate(exp)
        assert len(narrator.entries) == 10

    def test_significance_includes_emotion(self):
        narrator = ContinuousNarrator()
        exp = Experience(
            content="Terrible market crash",
            domain="finance",
            importance=0.9,
            emotional_valence=-0.8,
            emotional_intensity=0.9,
        )
        entry = narrator.integrate(exp)
        assert entry.significance > 0.3


# ---------------------------------------------------------------------------
# MeaningMaker
# ---------------------------------------------------------------------------

class TestMeaningMaker:
    def test_assess_returns_metrics(self):
        mm = MeaningMaker()
        exp = Experience(content="New quantum computing breakthrough", domain="science")
        result = mm.assess(exp)
        assert "relevance" in result
        assert "learning_potential" in result
        assert "ego_investment" in result
        assert "novel_concepts" in result

    def test_first_experience_has_high_novelty(self):
        mm = MeaningMaker()
        exp = Experience(content="Novel discovery in particle physics", domain="science")
        result = mm.assess(exp)
        assert result["learning_potential"] == 1.0  # all concepts are novel

    def test_repeated_concepts_increase_relevance(self):
        mm = MeaningMaker()
        exp1 = Experience(content="Stock market trading analysis", domain="finance", importance=0.8)
        mm.assess(exp1)
        exp2 = Experience(content="Stock market performance review", domain="finance", importance=0.7)
        result = mm.assess(exp2)
        assert result["relevance"] > 0.0

    def test_value_weights_accumulate(self):
        mm = MeaningMaker()
        for i in range(5):
            exp = Experience(content="Machine learning algorithm", domain="tech", importance=0.9)
            mm.assess(exp)
        assert "machine" in mm.value_weights
        assert mm.value_weights["machine"] > 0


# ---------------------------------------------------------------------------
# TemporalIntegrator
# ---------------------------------------------------------------------------

class TestTemporalIntegrator:
    def test_single_experience_perfect_coherence(self):
        ti = TemporalIntegrator()
        exp = Experience(content="Test experience", domain="test")
        result = ti.update(exp)
        assert result["temporal_coherence"] == 1.0

    def test_same_domain_high_coherence(self):
        ti = TemporalIntegrator()
        for i in range(5):
            exp = Experience(content=f"Finance topic {i}", domain="finance")
            result = ti.update(exp)
        assert result["temporal_coherence"] >= 0.8

    def test_mixed_domains_lower_coherence(self):
        ti = TemporalIntegrator()
        domains = ["finance", "weather", "tech", "science", "health"]
        for d in domains:
            exp = Experience(content=f"Topic in {d}", domain=d)
            result = ti.update(exp)
        assert result["temporal_coherence"] < 0.5

    def test_continuity_score_bounded(self):
        ti = TemporalIntegrator()
        exp = Experience(content="Test", domain="test")
        result = ti.update(exp)
        assert 0.0 <= result["identity_continuity"] <= 1.0


# ---------------------------------------------------------------------------
# EgoBoundaryTracker
# ---------------------------------------------------------------------------

class TestEgoBoundaryTracker:
    def test_self_references_detected(self):
        tracker = EgoBoundaryTracker()
        result = tracker.analyse("I think my analysis shows that I am correct")
        assert result["self_references"] > 0

    def test_other_references_detected(self):
        tracker = EgoBoundaryTracker()
        result = tracker.analyse("You should check their data and compare it with them")
        assert result["other_references"] > 0

    def test_boundary_strength_shifts(self):
        tracker = EgoBoundaryTracker()
        # Self-heavy text
        tracker.analyse("I think I know my position and I believe myself")
        assert tracker.boundary_strength > 0.5

    def test_neutral_text(self):
        tracker = EgoBoundaryTracker()
        result = tracker.analyse("The market data shows growth in technology sector")
        assert result["self_references"] == 0
        assert result["other_references"] == 0

    def test_cumulative_tracking(self):
        tracker = EgoBoundaryTracker()
        tracker.analyse("I think this is good")
        tracker.analyse("I believe my approach works")
        assert tracker.self_count >= 2  # "i" appears in both
