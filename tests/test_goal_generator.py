"""Tests for GoalGenerator (Gap 4: AGI Roadmap — Agency)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from emms.memory.goal_generator import GeneratedGoal, GoalGenerator


# ---------------------------------------------------------------------------
# GeneratedGoal
# ---------------------------------------------------------------------------


class TestGeneratedGoal:
    def test_creation(self):
        g = GeneratedGoal(description="Investigate finance", source="curiosity",
                          domain="finance", priority=0.8)
        assert g.source == "curiosity"
        assert g.priority == 0.8

    def test_serialization(self):
        g = GeneratedGoal(description="Test goal", source="maintenance",
                          domain="tech", priority=0.5)
        d = g.to_dict()
        g2 = GeneratedGoal.from_dict(d)
        assert g2.description == g.description
        assert g2.source == g.source


# ---------------------------------------------------------------------------
# GoalGenerator — Individual Sources
# ---------------------------------------------------------------------------


class TestGoalGeneratorSources:
    def test_curiosity_goals(self):
        gg = GoalGenerator()
        goals = gg.generate(
            capability_profile={"finance": 0.2, "tech": 0.9},
            belief_counts={"finance": 10, "tech": 3},
        )
        curiosity = [g for g in goals if g.source == "curiosity"]
        assert len(curiosity) >= 1
        assert any("finance" in g.description for g in curiosity)

    def test_curiosity_no_goals_when_capable(self):
        gg = GoalGenerator()
        goals = gg.generate(
            capability_profile={"finance": 0.8},
            belief_counts={"finance": 10},
        )
        curiosity = [g for g in goals if g.source == "curiosity"]
        assert len(curiosity) == 0

    def test_calibration_goals(self):
        gg = GoalGenerator()
        goals = gg.generate(
            calibration_report={"domain_scores": {"finance": 0.5, "tech": 0.1}},
        )
        cal = [g for g in goals if g.source == "calibration"]
        assert len(cal) >= 1
        assert any("finance" in g.description for g in cal)

    def test_affect_goals(self):
        gg = GoalGenerator()
        goals = gg.generate(
            somatic_markers=[
                {"valence": -0.7, "strength": 0.8, "context_tokens": "leverage risk"},
            ],
        )
        affect = [g for g in goals if g.source == "affect"]
        assert len(affect) >= 1

    def test_maintenance_goals(self):
        gg = GoalGenerator()
        goals = gg.generate(
            memory_domain_counts={"finance": 200, "tech": 50},
        )
        maint = [g for g in goals if g.source == "maintenance"]
        assert len(maint) >= 1
        assert any("finance" in g.description for g in maint)

    def test_obligation_goals(self):
        gg = GoalGenerator()
        old_goal = {
            "description": "Close orphaned positions",
            "created_at": time.time() - 86400 * 5,  # 5 days ago
            "status": "active",
        }
        goals = gg.generate(human_goals=[old_goal])
        oblig = [g for g in goals if g.source == "obligation"]
        assert len(oblig) >= 1


# ---------------------------------------------------------------------------
# GoalGenerator — Deduplication & Suppression
# ---------------------------------------------------------------------------


class TestGoalGeneratorDedup:
    def test_deduplication(self):
        gg = GoalGenerator()
        # Both sources should generate similar finance goals
        goals = gg.generate(
            capability_profile={"finance": 0.2},
            belief_counts={"finance": 10},
            memory_domain_counts={"finance": 200},
        )
        # Should not have near-duplicates
        descs = [g.description for g in goals]
        for i, d1 in enumerate(descs):
            for j, d2 in enumerate(descs):
                if i != j:
                    t1 = set(d1.lower().split())
                    t2 = set(d2.lower().split())
                    union = t1 | t2
                    if union:
                        assert len(t1 & t2) / len(union) <= 0.6

    def test_suppression(self):
        gg = GoalGenerator()
        goals = gg.generate(
            capability_profile={"finance": 0.2},
            belief_counts={"finance": 10},
        )
        assert len(goals) > 0
        gg.suppress(goals[0].description)
        goals2 = gg.generate(
            capability_profile={"finance": 0.2},
            belief_counts={"finance": 10},
        )
        assert goals[0].description not in [g.description for g in goals2]

    def test_unsuppress(self):
        gg = GoalGenerator()
        gg.suppress("test goal")
        assert gg.unsuppress("test goal") is True
        assert gg.unsuppress("nonexistent") is False

    def test_max_goals(self):
        gg = GoalGenerator(max_goals=3)
        goals = gg.generate(
            capability_profile={f"domain_{i}": 0.1 for i in range(10)},
            belief_counts={f"domain_{i}": 10 for i in range(10)},
        )
        assert len(goals) <= 3


# ---------------------------------------------------------------------------
# GoalGenerator — Reporting & Persistence
# ---------------------------------------------------------------------------


class TestGoalGeneratorPersistence:
    def test_generation_log(self):
        gg = GoalGenerator()
        gg.generate(capability_profile={"finance": 0.2}, belief_counts={"finance": 10})
        gg.generate(capability_profile={"tech": 0.3}, belief_counts={"tech": 8})
        log = gg.generation_log()
        assert len(log) == 2

    def test_summary(self):
        gg = GoalGenerator()
        gg.generate(capability_profile={"finance": 0.2}, belief_counts={"finance": 10})
        s = gg.summary()
        assert "GoalGenerator" in s

    def test_save_load_roundtrip(self):
        gg = GoalGenerator()
        gg.generate(capability_profile={"finance": 0.2}, belief_counts={"finance": 10})
        gg.suppress("test suppressed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            gg.save_state(path)

            gg2 = GoalGenerator()
            assert gg2.load_state(path)
            assert len(gg2.goals) == len(gg.goals)
            assert "test suppressed" in gg2.suppressed

    def test_load_nonexistent_returns_false(self):
        gg = GoalGenerator()
        assert gg.load_state("/nonexistent.json") is False
