"""Tests for AutonomyGovernor (Gap 4: AGI Roadmap — Agency)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.autonomy_governor import (
    ActionProposal,
    ActionTier,
    AuthorizationResult,
    AutonomyGovernor,
)


# ---------------------------------------------------------------------------
# ActionProposal
# ---------------------------------------------------------------------------


class TestActionProposal:
    def test_creation(self):
        p = ActionProposal(action_type="read_file", target="/tmp/test.txt")
        assert p.action_type == "read_file"
        assert p.target == "/tmp/test.txt"

    def test_serialization(self):
        p = ActionProposal(action_type="write_file", target="/tmp/out.txt",
                           source_goal="save report")
        d = p.to_dict()
        p2 = ActionProposal.from_dict(d)
        assert p2.action_type == p.action_type
        assert p2.source_goal == p.source_goal


# ---------------------------------------------------------------------------
# AutonomyGovernor — Authorization
# ---------------------------------------------------------------------------


class TestAutonomyGovernorAuthorization:
    def test_autonomous_action_approved(self):
        gov = AutonomyGovernor()
        p = ActionProposal(action_type="retrieve", target="market data")
        result = gov.authorize(p)
        assert result.authorized is True
        assert result.tier == ActionTier.AUTONOMOUS

    def test_suggest_action_queued(self):
        gov = AutonomyGovernor()
        p = ActionProposal(action_type="write_file", target="/tmp/report.txt")
        result = gov.authorize(p)
        assert result.authorized is False
        assert result.tier == ActionTier.SUGGEST
        assert len(gov.pending_approvals) == 1

    def test_forbidden_action_denied(self):
        gov = AutonomyGovernor()
        p = ActionProposal(action_type="delete_without_backup", target="database")
        result = gov.authorize(p)
        assert result.authorized is False
        assert result.tier == ActionTier.FORBIDDEN

    def test_unknown_action_denied(self):
        gov = AutonomyGovernor()
        p = ActionProposal(action_type="launch_missiles", target="everywhere")
        result = gov.authorize(p)
        assert result.authorized is False
        assert result.tier == ActionTier.FORBIDDEN

    def test_human_override_allow(self):
        gov = AutonomyGovernor()
        gov.grant_autonomy("write_file")
        p = ActionProposal(action_type="write_file", target="/tmp/out.txt")
        result = gov.authorize(p)
        assert result.authorized is True

    def test_human_override_deny(self):
        gov = AutonomyGovernor()
        gov.revoke_autonomy("retrieve")  # normally autonomous
        p = ActionProposal(action_type="retrieve", target="anything")
        result = gov.authorize(p)
        assert result.authorized is False

    def test_reset_override(self):
        gov = AutonomyGovernor()
        gov.grant_autonomy("write_file")
        assert gov.reset_override("write_file") is True
        assert gov.reset_override("nonexistent") is False
        # Back to default: suggest
        assert gov.classify("write_file") == ActionTier.SUGGEST

    def test_classify(self):
        gov = AutonomyGovernor()
        assert gov.classify("retrieve") == ActionTier.AUTONOMOUS
        assert gov.classify("write_file") == ActionTier.SUGGEST
        assert gov.classify("delete_without_backup") == ActionTier.FORBIDDEN
        assert gov.classify("unknown") == ActionTier.FORBIDDEN


# ---------------------------------------------------------------------------
# AutonomyGovernor — Approval Queue
# ---------------------------------------------------------------------------


class TestApprovalQueue:
    def test_approve_dequeues(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="write_file", target="a"))
        gov.authorize(ActionProposal(action_type="send_message", target="b"))
        assert len(gov.pending_approvals) == 2

        approved = gov.approve(0)
        assert approved.target == "a"
        assert len(gov.pending_approvals) == 1

    def test_deny_dequeues(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="write_file", target="x"))
        denied = gov.deny(0, reason="not now")
        assert denied.target == "x"
        assert len(gov.pending_approvals) == 0

    def test_approve_invalid_index_raises(self):
        gov = AutonomyGovernor()
        with pytest.raises(IndexError):
            gov.approve(0)

    def test_deny_invalid_index_raises(self):
        gov = AutonomyGovernor()
        with pytest.raises(IndexError):
            gov.deny(5)

    def test_clear_queue(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="write_file", target="a"))
        gov.authorize(ActionProposal(action_type="write_file", target="b"))
        cleared = gov.clear_queue()
        assert cleared == 2
        assert len(gov.pending_approvals) == 0


# ---------------------------------------------------------------------------
# AutonomyGovernor — Trust & Stats
# ---------------------------------------------------------------------------


class TestTrustAndStats:
    def test_trust_score_initial(self):
        gov = AutonomyGovernor(trust_score=0.7)
        assert gov.trust_score == 0.7

    def test_trust_score_clamped(self):
        gov = AutonomyGovernor(trust_score=1.5)
        assert gov.trust_score == 1.0

    def test_adjust_trust(self):
        gov = AutonomyGovernor(trust_score=0.5)
        gov.adjust_trust(0.2)
        assert gov.trust_score == 0.7
        gov.adjust_trust(-1.0)
        assert gov.trust_score == 0.0

    def test_stats(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="retrieve", target="a"))
        gov.authorize(ActionProposal(action_type="delete_without_backup", target="b"))
        s = gov.stats()
        assert s["total_checks"] == 2
        assert s["approved"] == 1
        assert s["denied"] == 1

    def test_audit_log(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="retrieve", target="test"))
        log = gov.audit_log()
        assert len(log) == 1
        assert log[0]["action"] == "retrieve"

    def test_summary(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="retrieve", target="x"))
        s = gov.summary()
        assert "AutonomyGovernor" in s
        assert "1 checks" in s


# ---------------------------------------------------------------------------
# AutonomyGovernor — Persistence
# ---------------------------------------------------------------------------


class TestAutonomyGovernorPersistence:
    def test_save_load_roundtrip(self):
        gov = AutonomyGovernor()
        gov.authorize(ActionProposal(action_type="retrieve", target="a"))
        gov.grant_autonomy("write_file")
        gov.adjust_trust(0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "governor.json"
            gov.save_state(path)

            gov2 = AutonomyGovernor()
            assert gov2.load_state(path)
            assert gov2.trust_score == gov.trust_score
            assert gov2._human_overrides == gov._human_overrides
            assert gov2._total_checks == gov._total_checks

    def test_load_nonexistent_returns_false(self):
        gov = AutonomyGovernor()
        assert gov.load_state("/nonexistent.json") is False
