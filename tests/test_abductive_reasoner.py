"""Tests for AbductiveReasoner (Gap 7: AGI Roadmap — Novel Reasoning)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.abductive_reasoner import AbductiveReasoner, Hypothesis


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class TestHypothesis:
    def test_creation(self):
        h = Hypothesis(id="h1", observation="surprise", explanation="because X",
                       generation_method="inversion")
        assert h.status == "untested"
        assert h.confidence == 0.3

    def test_serialization(self):
        h = Hypothesis(id="h1", observation="obs", explanation="exp",
                       generation_method="analogy", confidence=0.5,
                       evidence_for=["a"], domain="finance")
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        assert h2.id == h.id
        assert h2.evidence_for == ["a"]


# ---------------------------------------------------------------------------
# AbductiveReasoner — Generation Methods
# ---------------------------------------------------------------------------


class TestAbductiveReasonerGeneration:
    def test_inversion_generates_negation(self):
        ar = AbductiveReasoner()
        beliefs = ["Prices always increase in bull markets"]
        hyps = ar.generate_from_surprise("Prices fell unexpectedly",
                                         relevant_beliefs=beliefs)
        inversions = [h for h in hyps if h.generation_method == "inversion"]
        assert len(inversions) >= 1
        assert "never" in inversions[0].explanation.lower() or "decrease" in inversions[0].explanation.lower()

    def test_analogy_finds_cross_domain(self):
        ar = AbductiveReasoner()
        memories = [
            {"domain": "tech", "content": "server latency increases under load"},
            {"domain": "finance", "content": "market volatility increases under stress"},
        ]
        hyps = ar.generate_from_surprise(
            "latency increases during peak hours",
            relevant_memories=memories, domain="ops",
        )
        analogies = [h for h in hyps if h.generation_method == "analogy"]
        assert len(analogies) >= 1

    def test_decomposition_from_failed_prediction(self):
        ar = AbductiveReasoner()
        beliefs = ["Bull market volume drives continued price growth upward",
                    "High volume in bull markets signals strong momentum"]
        prediction = {"content": "Bull market volume growth will drive price momentum"}
        hyps = ar.generate_from_surprise(
            "Market crashed despite high volume growth",
            failed_prediction=prediction,
            relevant_beliefs=beliefs,
        )
        decomps = [h for h in hyps if h.generation_method == "decomposition"]
        assert len(decomps) >= 1

    def test_cross_domain_structural(self):
        ar = AbductiveReasoner()
        memories = [
            {"domain": "biology", "content": "stress causes immune suppression"},
            {"domain": "finance", "content": "normal market conditions"},
        ]
        hyps = ar.generate_from_surprise(
            "high stress causes performance degradation",
            relevant_memories=memories, domain="engineering",
        )
        xdom = [h for h in hyps if h.generation_method == "cross_domain"]
        assert len(xdom) >= 1

    def test_generate_limits_output(self):
        ar = AbductiveReasoner(max_per_observation=3)
        beliefs = [f"Belief {i} about increasing trends" for i in range(10)]
        hyps = ar.generate_from_surprise("Something unexpected",
                                         relevant_beliefs=beliefs)
        assert len(hyps) <= 3

    def test_no_beliefs_no_memories_returns_empty(self):
        ar = AbductiveReasoner()
        hyps = ar.generate_from_surprise("Something unexpected")
        assert len(hyps) == 0


# ---------------------------------------------------------------------------
# AbductiveReasoner — Hypothesis Management
# ---------------------------------------------------------------------------


class TestAbductiveReasonerManagement:
    def test_update_hypothesis_supports(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["Prices increase always"])
        hyp = ar.all_hypotheses[0]
        old_conf = hyp.confidence
        ar.update_hypothesis(hyp.id, "found confirming data", supports=True)
        assert hyp.confidence > old_conf

    def test_update_hypothesis_contradicts(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["Prices increase always"])
        hyp = ar.all_hypotheses[0]
        old_conf = hyp.confidence
        ar.update_hypothesis(hyp.id, "found contradicting data", supports=False)
        assert hyp.confidence < old_conf

    def test_auto_confirm(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["Values are high"])
        hyp = ar.all_hypotheses[0]
        hyp.confidence = 0.65
        for i in range(3):
            ar.update_hypothesis(hyp.id, f"evidence {i}", supports=True)
        assert hyp.status == "confirmed"

    def test_auto_refute(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["Values are high"])
        hyp = ar.all_hypotheses[0]
        for i in range(4):
            ar.update_hypothesis(hyp.id, f"counter {i}", supports=False)
        assert hyp.status == "refuted"

    def test_resolve_hypothesis(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test surprise event",
                                  relevant_beliefs=["Prices always increase in markets"])
        assert len(ar.all_hypotheses) >= 1
        hyp = ar.all_hypotheses[0]
        ar.resolve_hypothesis(hyp.id, "inconclusive")
        assert hyp.status == "inconclusive"

    def test_active_hypotheses_sorted(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test A", relevant_beliefs=["A increases"])
        ar.generate_from_surprise("Test B", relevant_beliefs=["B decreases"])
        # Manually set confidence
        if len(ar.all_hypotheses) >= 2:
            ar.all_hypotheses[0].confidence = 0.8
            ar.all_hypotheses[1].confidence = 0.2
            active = ar.active_hypotheses()
            assert active[0].confidence >= active[-1].confidence

    def test_update_nonexistent_returns_false(self):
        ar = AbductiveReasoner()
        assert ar.update_hypothesis("fake_id", "evidence", True) is False

    def test_hypotheses_by_status(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["X increases"])
        ar.resolve_hypothesis(ar.all_hypotheses[0].id, "confirmed")
        confirmed = ar.hypotheses_by_status("confirmed")
        assert len(confirmed) >= 1


# ---------------------------------------------------------------------------
# AbductiveReasoner — Persistence
# ---------------------------------------------------------------------------


class TestAbductiveReasonerPersistence:
    def test_save_load_roundtrip(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["Values increase"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reasoner.json"
            ar.save_state(path)

            ar2 = AbductiveReasoner()
            assert ar2.load_state(path)
            assert ar2._total_generated == ar._total_generated
            assert len(ar2.all_hypotheses) == len(ar.all_hypotheses)

    def test_load_nonexistent_returns_false(self):
        ar = AbductiveReasoner()
        assert ar.load_state("/nonexistent.json") is False

    def test_summary(self):
        ar = AbductiveReasoner()
        ar.generate_from_surprise("Test", relevant_beliefs=["X increases"])
        s = ar.summary()
        assert "AbductiveReasoner" in s
