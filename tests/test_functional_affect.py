"""Tests for FunctionalAffect (Gap 3: AGI Roadmap).

Covers:
- AffectState (circumplex model, modulation properties)
- SomaticMarker (context similarity, decision biasing)
- FunctionalAffect (update, modulation, somatic markers, persistence)
- EMMS integration (store pipeline, public wrappers, save/load)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms import EMMS, Experience
from emms.memory.functional_affect import (
    AffectState,
    FunctionalAffect,
    SomaticMarker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


# ---------------------------------------------------------------------------
# AffectState
# ---------------------------------------------------------------------------


class TestAffectState:
    def test_creation_defaults(self):
        s = AffectState()
        assert s.valence == 0.0
        assert s.arousal == 0.3
        assert s.timestamp > 0

    def test_clamping(self):
        s = AffectState(valence=5.0, arousal=-2.0)
        assert s.valence == 1.0
        assert s.arousal == 0.0

    def test_attention_breadth_anxious(self):
        s = AffectState(valence=-0.8, arousal=0.9)
        assert s.attention_breadth < 0.3  # narrow

    def test_attention_breadth_calm(self):
        s = AffectState(valence=0.8, arousal=0.1)
        assert s.attention_breadth > 0.7  # broad

    def test_processing_depth(self):
        high_arousal = AffectState(arousal=0.9)
        low_arousal = AffectState(arousal=0.1)
        assert high_arousal.processing_depth < low_arousal.processing_depth

    def test_risk_tolerance(self):
        positive = AffectState(valence=0.8)
        negative = AffectState(valence=-0.8)
        assert positive.risk_tolerance > negative.risk_tolerance

    def test_novelty_seeking(self):
        curious = AffectState(valence=0.5, arousal=0.5)  # moderate positive
        anxious = AffectState(valence=-0.8, arousal=0.9)
        assert curious.novelty_seeking > anxious.novelty_seeking

    def test_label_quadrants(self):
        assert AffectState(valence=0.5, arousal=0.7).label == "excited"
        assert AffectState(valence=0.5, arousal=0.3).label == "calm"
        assert AffectState(valence=-0.5, arousal=0.7).label == "anxious"
        assert AffectState(valence=-0.5, arousal=0.3).label == "sad"

    def test_to_dict(self):
        s = AffectState(valence=0.5, arousal=0.6)
        d = s.to_dict()
        assert "valence" in d
        assert "attention_breadth" in d
        assert "label" in d

    def test_from_dict(self):
        s = AffectState(valence=0.5, arousal=0.6)
        d = s.to_dict()
        s2 = AffectState.from_dict(d)
        assert s2.valence == s.valence
        assert s2.arousal == s.arousal


# ---------------------------------------------------------------------------
# SomaticMarker
# ---------------------------------------------------------------------------


class TestSomaticMarker:
    def test_creation(self):
        m = SomaticMarker(context_tokens=["risk", "leverage"], valence=-0.7)
        assert m.id.startswith("sm_")
        assert m.valence == -0.7

    def test_similarity(self):
        m = SomaticMarker(context_tokens=["risk", "leverage", "liquidation"])
        assert m.similarity(["risk", "leverage"]) > 0.5
        assert m.similarity(["unrelated", "tokens"]) == 0.0

    def test_decay(self):
        m = SomaticMarker(strength=1.0)
        m.created_at -= 86400 * 30  # 30 days ago
        m.decay()
        assert m.strength < 1.0

    def test_serialization(self):
        m = SomaticMarker(context_tokens=["test"], domain="finance", valence=0.5)
        d = m.to_dict()
        m2 = SomaticMarker.from_dict(d)
        assert m2.context_tokens == m.context_tokens
        assert m2.valence == m.valence


# ---------------------------------------------------------------------------
# FunctionalAffect
# ---------------------------------------------------------------------------


class TestFunctionalAffect:
    def test_initial_state(self):
        fa = FunctionalAffect()
        assert fa.current_state.valence == 0.0
        assert fa.current_state.arousal == 0.3

    def test_update_from_experience_positive(self):
        fa = FunctionalAffect()
        state = fa.update_from_experience(0.8, 0.5, "finance")
        assert state.valence > 0  # shifted positive
        assert fa._total_updates == 1

    def test_update_momentum(self):
        fa = FunctionalAffect(valence_momentum=0.5)
        fa.update_from_experience(1.0)
        v1 = fa.current_state.valence
        fa.update_from_experience(-1.0)
        v2 = fa.current_state.valence
        # Should be somewhere between, not at extremes
        assert v2 < v1
        assert v2 > -1.0

    def test_update_from_surprise(self):
        fa = FunctionalAffect()
        state = fa.update_from_surprise(0.9)
        assert state.arousal > 0.3  # arousal spike
        assert state.valence < 0  # slightly aversive

    def test_update_from_success(self):
        fa = FunctionalAffect()
        state = fa.update_from_success(0.8)
        assert state.valence > 0

    def test_update_from_failure(self):
        fa = FunctionalAffect()
        state = fa.update_from_failure(0.8)
        assert state.valence < 0

    def test_mark_and_consult(self):
        fa = FunctionalAffect()
        fa.mark("high leverage risky position", -0.8, domain="trading")
        result = fa.consult_markers("leverage is risky", "trading")
        assert result is not None
        assert result.valence < 0

    def test_consult_no_markers(self):
        fa = FunctionalAffect()
        result = fa.consult_markers("something new")
        assert result is None

    def test_bias_decision(self):
        fa = FunctionalAffect()
        # Mark "leverage" as bad
        fa.mark("increase leverage position", -0.9, domain="trading")
        # Mark "hedge" as good
        fa.mark("hedge position with options", 0.8, domain="trading")

        options = ["increase leverage", "hedge with options", "do nothing"]
        biased = fa.bias_decision(options, "manage risk", "trading")
        weights = {opt: w for opt, w in biased}

        # Hedge should be preferred over leverage
        assert weights["hedge with options"] > weights["increase leverage"]

    def test_modulate_retrieval_params(self):
        fa = FunctionalAffect()
        fa.update_from_experience(-0.8, 0.9)  # anxious state
        params = fa.modulate_retrieval_params({"max_results": 10})
        # Anxious → narrow attention → fewer results
        assert params["max_results"] <= 10
        assert "_affect_state" in params

    def test_modulate_task_priority(self):
        fa = FunctionalAffect()
        fa._state = AffectState(valence=-0.8, arousal=0.9)  # anxious
        boosted = fa.modulate_task_priority(0.5, "risk")
        normal = fa.modulate_task_priority(0.5, "general")
        assert boosted > normal

    def test_mood_trend(self):
        fa = FunctionalAffect()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            fa.update_from_experience(v)
        trend = fa.mood_trend()
        assert trend["trend"] == "improving"

    def test_emotional_coherence(self):
        fa = FunctionalAffect()
        # Consistent positive → high coherence
        for _ in range(10):
            fa.update_from_experience(0.5)
        assert fa.emotional_coherence() > 0.8

    def test_marker_capacity(self):
        fa = FunctionalAffect(max_markers=10)
        for i in range(20):
            fa.mark(f"context {i} unique tokens here", 0.5)
        assert len(fa._markers) <= 10

    def test_summary(self):
        fa = FunctionalAffect()
        fa.update_from_experience(0.5)
        summary = fa.summary()
        assert "FunctionalAffect" in summary
        assert "1 updates" in summary

    def test_persistence_roundtrip(self):
        fa = FunctionalAffect()
        fa.update_from_experience(0.7, 0.6)
        fa.mark("test context", -0.5, domain="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "affect.json"
            fa.save_state(path)

            fa2 = FunctionalAffect()
            loaded = fa2.load_state(path)
            assert loaded
            assert abs(fa2.current_state.valence - fa.current_state.valence) < 0.001
            assert len(fa2._markers) == 1

    def test_empty_load_returns_false(self):
        fa = FunctionalAffect()
        assert fa.load_state("/nonexistent/path.json") is False


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_affect_state(self):
        agent = _make_emms()
        state = agent.affect_state()
        assert "valence" in state
        assert "label" in state

    def test_affect_update(self):
        agent = _make_emms()
        state = agent.affect_update(0.7, 0.5, "finance")
        assert state["valence"] > 0

    def test_affect_mark_and_consult(self):
        agent = _make_emms()
        marker_id = agent.affect_mark("high risk leverage", -0.8, "trading")
        assert marker_id.startswith("sm_")

        result = agent.affect_consult("leverage risk", "trading")
        assert result is not None
        assert result["valence"] < 0

    def test_affect_bias_decision(self):
        agent = _make_emms()
        agent.affect_mark("bad outcome happened here", -0.9, "trading")
        biased = agent.affect_bias_decision(
            ["safe option", "bad outcome repeat"], "what to do", "trading"
        )
        assert len(biased) == 2

    def test_affect_mood_trend(self):
        agent = _make_emms()
        trend = agent.affect_mood_trend()
        assert "trend" in trend

    def test_affect_summary(self):
        agent = _make_emms()
        summary = agent.affect_summary()
        assert "FunctionalAffect" in summary

    def test_store_updates_affect(self):
        agent = _make_emms()
        # Store with emotional valence
        agent.store(Experience(content="great success", domain="finance",
                              emotional_valence=0.8))
        state = agent.affect_state()
        assert state["valence"] > 0  # should have shifted positive

    def test_save_load_roundtrip(self):
        agent = _make_emms()
        agent.affect_update(0.7, 0.5)
        agent.affect_mark("test", -0.5, "test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = _make_emms()
            agent2.load(str(path))
            state = agent2.affect_state()
            assert state["valence"] > 0  # persisted
