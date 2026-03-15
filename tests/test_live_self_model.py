"""Tests for LiveSelfModel (Gap 6: AGI Roadmap).

Covers:
- LiveBelief incremental update + decay
- CalibrationTracker (Brier score, bias, adjusted confidence)
- LiveSelfModel.update_from_experience
- Drift detection
- Persistence (save/load round-trip)
- EMMS integration (store pipeline, save/load, public wrappers)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from emms import EMMS, Experience
from emms.core.models import MemoryItem, MemoryTier
from emms.memory.live_self_model import (
    CalibrationEntry,
    CalibrationTracker,
    DriftEvent,
    LiveBelief,
    LiveSelfModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_item(content: str = "test content", domain: str = "finance",
               strength: float = 0.7, valence: float = 0.3) -> MemoryItem:
    exp = Experience(content=content, domain=domain, emotional_valence=valence)
    item = MemoryItem(experience=exp)
    item.memory_strength = strength
    return item


# ---------------------------------------------------------------------------
# LiveBelief
# ---------------------------------------------------------------------------


class TestLiveBelief:
    def test_creation_defaults(self):
        b = LiveBelief()
        assert b.id.startswith("lbel_")
        assert b.confidence == 0.0
        assert b.evidence_count == 0

    def test_incremental_update(self):
        b = LiveBelief(content="test belief", domain="finance")
        b.update(0.8, 0.5, "mem_001")
        assert b.evidence_count == 1
        assert b.mean_strength == 0.8
        assert b.mean_valence == 0.5
        assert "mem_001" in b.supporting_memory_ids
        assert b.confidence > 0

        # Second update — running average
        b.update(0.6, 0.3, "mem_002")
        assert b.evidence_count == 2
        assert abs(b.mean_strength - 0.7) < 1e-9
        assert abs(b.mean_valence - 0.4) < 1e-9

    def test_confidence_increases_with_evidence(self):
        b = LiveBelief(content="test", domain="d")
        confs = []
        for i in range(10):
            b.update(0.8, 0.5, f"mem_{i}")
            confs.append(b.confidence)
        # Confidence should generally increase with more evidence
        assert confs[-1] > confs[0]

    def test_decay(self):
        b = LiveBelief(content="old belief", domain="d", confidence=0.8)
        b.last_updated = time.time() - 7200  # 2 hours ago
        b.decay()
        assert b.confidence < 0.8

    def test_supporting_memory_ids_capped(self):
        b = LiveBelief(content="test", domain="d")
        for i in range(15):
            b.update(0.5, 0.0, f"mem_{i}")
        assert len(b.supporting_memory_ids) <= 10

    def test_serialization_roundtrip(self):
        b = LiveBelief(content="test", domain="finance", confidence=0.7)
        b.update(0.8, 0.5, "mem_001")
        d = b.to_dict()
        b2 = LiveBelief.from_dict(d)
        assert b2.content == b.content
        assert b2.evidence_count == b.evidence_count
        assert b2.mean_strength == b.mean_strength


# ---------------------------------------------------------------------------
# CalibrationTracker
# ---------------------------------------------------------------------------


class TestCalibrationTracker:
    def test_empty_brier_score(self):
        ct = CalibrationTracker()
        assert ct.brier_score() == 0.25  # uninformative prior

    def test_perfect_calibration(self):
        ct = CalibrationTracker()
        # Confident and correct
        for i in range(10):
            ct.record(f"p_{i}", "finance", 0.9, True)
        bs = ct.brier_score("finance")
        assert bs < 0.05  # very good

    def test_overconfident_detection(self):
        ct = CalibrationTracker()
        # Say 90% confident but only correct 50% of the time
        for i in range(20):
            ct.record(f"p_{i}", "tech", 0.9, i % 2 == 0)
        bias = ct.calibration_bias("tech")
        assert bias > 0.3  # significantly overconfident

    def test_adjusted_confidence(self):
        ct = CalibrationTracker()
        for i in range(20):
            ct.record(f"p_{i}", "finance", 0.9, i % 2 == 0)
        adj = ct.adjusted_confidence(0.9, "finance")
        assert adj < 0.9  # should be lowered due to overconfidence

    def test_domain_report(self):
        ct = CalibrationTracker()
        ct.record("p1", "finance", 0.8, True)
        ct.record("p2", "tech", 0.6, False)
        report = ct.domain_report()
        assert "finance" in report
        assert "tech" in report
        assert "brier_score" in report["finance"]

    def test_persistence_roundtrip(self):
        ct = CalibrationTracker()
        ct.record("p1", "finance", 0.8, True)
        ct.record("p2", "tech", 0.6, False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ct.save_state(path)

        ct2 = CalibrationTracker()
        ct2.load_state(path)
        assert ct2.brier_score("finance") == ct.brier_score("finance")
        Path(path).unlink()


# ---------------------------------------------------------------------------
# LiveSelfModel
# ---------------------------------------------------------------------------


class TestLiveSelfModel:
    def _make_model(self):
        from emms.memory.hierarchical import HierarchicalMemory
        mem = HierarchicalMemory()
        return LiveSelfModel(memory=mem)

    def test_update_from_experience(self):
        model = self._make_model()
        item = _make_item("deep learning neural networks training", "ml", 0.8, 0.5)
        changes = model.update_from_experience(item)
        assert changes["domain"] == "ml"
        assert model._total_experiences == 1
        assert "ml" in model.capability_vector

    def test_capability_grows_with_experience(self):
        model = self._make_model()
        caps = []
        for i in range(10):
            item = _make_item(f"finance analysis report {i}", "finance", 0.8, 0.3)
            model.update_from_experience(item)
            caps.append(model.capability_vector.get("finance", 0))
        # Capability should increase with more evidence
        assert caps[-1] > caps[0]

    def test_drift_detection(self):
        model = self._make_model(  )
        model.drift_threshold = 0.01  # very sensitive
        item = _make_item("quantum computing breakthrough", "quantum", 0.9, 0.8)
        changes = model.update_from_experience(item)
        # First experience in a new domain should trigger drift
        # (from 0.0 to some positive value)
        drift_events = model.detect_drift()
        assert len(drift_events) >= 1

    def test_beliefs_accumulate(self):
        model = self._make_model()
        for i in range(5):
            item = _make_item(f"blockchain consensus protocol verification {i}", "crypto", 0.7, 0.2)
            model.update_from_experience(item)
        beliefs = model.beliefs()
        assert len(beliefs) > 0

    def test_belief_pruning(self):
        model = self._make_model()
        model.max_beliefs = 5
        # Generate many beliefs across different content
        for i in range(30):
            content = f"unique_token_{i} appears in domain_{i % 3}"
            item = _make_item(content, f"domain_{i % 3}", 0.5, 0.0)
            model.update_from_experience(item)
        assert len(model.live_beliefs) <= model.max_beliefs * 2

    def test_consistency_score(self):
        model = self._make_model()
        # All positive valence → high consistency
        for i in range(5):
            item = _make_item(f"positive experience {i}", "general", 0.8, 0.7)
            model.update_from_experience(item)
        score = model.consistency_score()
        assert 0.0 <= score <= 1.0

    def test_summary_generation(self):
        model = self._make_model()
        item = _make_item("test content for summary", "test", 0.5, 0.0)
        model.update_from_experience(item)
        summary = model.summary()
        assert "LiveSelfModel" in summary
        assert "1 experiences" in summary

    def test_persistence_roundtrip(self):
        model = self._make_model()
        for i in range(5):
            item = _make_item(f"persistence test content {i}", "test", 0.7, 0.3)
            model.update_from_experience(item)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "live_self_model.json"
            model.save_state(path)

            model2 = self._make_model()
            loaded = model2.load_state(path)
            assert loaded is True
            assert model2._total_experiences == model._total_experiences
            assert model2.capability_vector == model.capability_vector
            assert len(model2.live_beliefs) == len(model.live_beliefs)

    def test_empty_load_returns_false(self):
        model = self._make_model()
        assert model.load_state("/nonexistent/path.json") is False


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_store_updates_live_self_model(self):
        agent = _make_emms()
        result = agent.store(Experience(content="financial analysis report", domain="finance"))
        assert "live_self_model_domain" in result
        assert result["live_self_model_domain"] == "finance"

    def test_live_self_model_summary(self):
        agent = _make_emms()
        agent.store(Experience(content="test experience", domain="test"))
        summary = agent.live_self_model_summary()
        assert "LiveSelfModel" in summary
        assert "1 experiences" in summary

    def test_live_beliefs(self):
        agent = _make_emms()
        agent.store(Experience(content="blockchain consensus protocol", domain="crypto"))
        beliefs = agent.live_beliefs()
        assert isinstance(beliefs, list)

    def test_live_capability_profile(self):
        agent = _make_emms()
        agent.store(Experience(content="machine learning model", domain="ml"))
        profile = agent.live_capability_profile()
        assert "ml" in profile

    def test_live_drift_events(self):
        agent = _make_emms()
        events = agent.live_drift_events()
        assert isinstance(events, list)

    def test_live_calibration_report(self):
        agent = _make_emms()
        report = agent.live_calibration_report()
        assert isinstance(report, dict)

    def test_save_load_roundtrip(self):
        agent = _make_emms()
        for i in range(3):
            agent.store(Experience(content=f"roundtrip test {i}", domain="test"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = _make_emms()
            agent2.load(str(path))
            summary = agent2.live_self_model_summary()
            assert "3 experiences" in summary
