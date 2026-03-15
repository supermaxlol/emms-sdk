"""Tests for ReasoningVerifier (Gap 7: AGI Roadmap — Novel Reasoning)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.reasoning_verifier import LogicalIssue, ReasoningVerifier


# ---------------------------------------------------------------------------
# LogicalIssue
# ---------------------------------------------------------------------------


class TestLogicalIssue:
    def test_creation(self):
        i = LogicalIssue(issue_type="circular", location="step_2",
                         description="Restates conclusion", severity="fatal")
        assert i.severity == "fatal"

    def test_serialization(self):
        i = LogicalIssue(issue_type="overgeneralization", location="step_0",
                         description="Claims 'always'", severity="minor",
                         suggested_fix="Qualify")
        d = i.to_dict()
        i2 = LogicalIssue.from_dict(d)
        assert i2.issue_type == i.issue_type
        assert i2.suggested_fix == "Qualify"


# ---------------------------------------------------------------------------
# ReasoningVerifier — Individual Checks
# ---------------------------------------------------------------------------


class TestReasoningVerifierChecks:
    def test_unsupported_premise(self):
        rv = ReasoningVerifier()
        steps = [
            "The quantum flux capacitor enables time travel through dimensional rifts"
        ]
        memory = ["Markets rose today", "Finance is complex"]
        issues = rv.verify_chain(steps, memory_contents=memory)
        unsupported = [i for i in issues if i.issue_type == "unsupported_premise"]
        assert len(unsupported) >= 1

    def test_supported_premise_no_issue(self):
        rv = ReasoningVerifier()
        steps = ["The market trends show rising prices and strong volume"]
        memory = ["Market trends indicate rising prices with strong volume growth"]
        issues = rv.verify_chain(steps, memory_contents=memory)
        unsupported = [i for i in issues if i.issue_type == "unsupported_premise"]
        assert len(unsupported) == 0

    def test_circularity_detected(self):
        rv = ReasoningVerifier()
        steps = ["Markets always rise in bull markets and prices go up"]
        hypothesis = "Markets always rise in bull markets and prices go up"
        issues = rv.verify_chain(steps, hypothesis=hypothesis)
        circular = [i for i in issues if i.issue_type == "circular"]
        assert len(circular) >= 1
        assert circular[0].severity == "fatal"

    def test_non_sequitur_detected(self):
        rv = ReasoningVerifier()
        steps = [
            "The stock market showed strong momentum with rising volume indicators",
            "Photosynthesis converts sunlight into chemical energy in plant cells",
        ]
        issues = rv.verify_chain(steps)
        non_seq = [i for i in issues if i.issue_type == "non_sequitur"]
        assert len(non_seq) >= 1

    def test_connected_steps_no_non_sequitur(self):
        rv = ReasoningVerifier()
        steps = [
            "Market volume increased by thirty percent this quarter",
            "Increased volume often signals strong market momentum ahead",
        ]
        issues = rv.verify_chain(steps)
        non_seq = [i for i in issues if i.issue_type == "non_sequitur"]
        assert len(non_seq) == 0

    def test_overgeneralization_detected(self):
        rv = ReasoningVerifier()
        steps = ["This strategy always works in every market condition"]
        issues = rv.verify_chain(steps)
        overgen = [i for i in issues if i.issue_type == "overgeneralization"]
        assert len(overgen) >= 1

    def test_contradiction_with_belief(self):
        rv = ReasoningVerifier()
        hypothesis = "The market will not crash this quarter and prices will keep rising steadily"
        beliefs = ["The market crash is likely this quarter as prices have been falling steadily"]
        issues = rv.verify_chain([], hypothesis=hypothesis, beliefs=beliefs)
        contradictions = [i for i in issues if i.issue_type == "contradiction"]
        assert len(contradictions) >= 1

    def test_ignored_counterevidence(self):
        rv = ReasoningVerifier()
        steps = ["The data clearly supports our position on market trends"]
        issues = rv.verify_chain(
            steps,
            evidence_against=["Study found opposite trend"],
        )
        ignored = [i for i in issues if i.issue_type == "ignored_counterevidence"]
        assert len(ignored) >= 1

    def test_counterevidence_addressed(self):
        rv = ReasoningVerifier()
        steps = ["However, despite counter-arguments, the data supports this conclusion"]
        issues = rv.verify_chain(
            steps,
            evidence_against=["Counter data exists"],
        )
        ignored = [i for i in issues if i.issue_type == "ignored_counterevidence"]
        assert len(ignored) == 0

    def test_clean_chain_minimal_issues(self):
        rv = ReasoningVerifier()
        steps = [
            "Market data shows steady growth in tech sector",
            "Tech sector growth correlates with increased consumer demand",
            "Based on sector growth data, we expect continued momentum",
        ]
        memory = [
            "Market data from Q3 shows tech sector growing steadily",
            "Consumer demand drives tech sector performance growth",
        ]
        issues = rv.verify_chain(steps, memory_contents=memory)
        fatal = [i for i in issues if i.severity == "fatal"]
        assert len(fatal) == 0


# ---------------------------------------------------------------------------
# ReasoningVerifier — Persistence
# ---------------------------------------------------------------------------


class TestReasoningVerifierPersistence:
    def test_save_load_roundtrip(self):
        rv = ReasoningVerifier()
        rv.verify_chain(["step 1", "step 2"])
        rv.verify_chain(["step 3"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verifier.json"
            rv.save_state(path)

            rv2 = ReasoningVerifier()
            assert rv2.load_state(path)
            assert rv2._total_verifications == 2

    def test_load_nonexistent_returns_false(self):
        rv = ReasoningVerifier()
        assert rv.load_state("/nonexistent.json") is False

    def test_summary(self):
        rv = ReasoningVerifier()
        rv.verify_chain(["test"])
        s = rv.summary()
        assert "ReasoningVerifier" in s
        assert "1 verification" in s

    def test_verification_log(self):
        rv = ReasoningVerifier()
        rv.verify_chain(["step"], hypothesis="test")
        log = rv.verification_log()
        assert len(log) == 1
        assert "hypothesis" in log[0]
