"""Tests for TemporalGrounder (Gap 5: AGI Roadmap — Grounding)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from emms.memory.temporal_grounder import (
    AnchorType,
    TemporalAnchor,
    TemporalGrounder,
)


# ---------------------------------------------------------------------------
# TemporalAnchor
# ---------------------------------------------------------------------------


class TestTemporalAnchor:
    def test_creation(self):
        a = TemporalAnchor(
            name="deploy",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() + 3600,
        )
        assert a.name == "deploy"
        assert a.id.startswith("ta_deploy_")
        assert a.active is True

    def test_time_until_future(self):
        a = TemporalAnchor(
            name="future",
            anchor_type=AnchorType.EVENT,
            timestamp=time.time() + 3600,
        )
        assert a.time_until() > 3500  # roughly 1 hour

    def test_time_until_past(self):
        a = TemporalAnchor(
            name="past",
            anchor_type=AnchorType.EVENT,
            timestamp=time.time() - 100,
        )
        assert a.time_until() < 0

    def test_urgency_past_due(self):
        a = TemporalAnchor(
            name="overdue",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() - 100,
        )
        assert a.urgency() == 1.0

    def test_urgency_far_future(self):
        a = TemporalAnchor(
            name="distant",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() + 86400 * 30,  # 30 days
        )
        assert a.urgency() < 0.2

    def test_urgency_soon(self):
        a = TemporalAnchor(
            name="soon",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() + 120,  # 2 minutes
        )
        assert a.urgency() > 0.7

    def test_is_due(self):
        a = TemporalAnchor(
            name="now",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() - 10,
        )
        assert a.is_due() is True

    def test_is_not_due(self):
        a = TemporalAnchor(
            name="later",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() + 3600,
        )
        assert a.is_due() is False

    def test_advance_recurrence(self):
        now = time.time()
        a = TemporalAnchor(
            name="daily",
            anchor_type=AnchorType.RECURRENCE,
            timestamp=now - 100,  # already past
            recurrence_interval=86400,
        )
        assert a.advance_recurrence() is True
        assert a.timestamp > now  # advanced to future

    def test_advance_non_recurrence_deactivates(self):
        a = TemporalAnchor(
            name="oneshot",
            anchor_type=AnchorType.EVENT,
            timestamp=time.time() - 100,
            recurrence_interval=0,
        )
        assert a.advance_recurrence() is False
        assert a.active is False

    def test_serialization(self):
        a = TemporalAnchor(
            name="test",
            anchor_type=AnchorType.DEADLINE,
            timestamp=time.time() + 3600,
            domain="finance",
            description="Market close",
        )
        d = a.to_dict()
        a2 = TemporalAnchor.from_dict(d)
        assert a2.name == a.name
        assert a2.anchor_type == a.anchor_type
        assert a2.domain == a.domain


# ---------------------------------------------------------------------------
# TemporalGrounder — Anchor Management
# ---------------------------------------------------------------------------


class TestTemporalGrounderAnchors:
    def test_add_anchor(self):
        tg = TemporalGrounder()
        a = tg.add_anchor("test", time.time() + 3600)
        assert a.name == "test"
        assert a.anchor_type == AnchorType.EVENT

    def test_add_deadline(self):
        tg = TemporalGrounder()
        a = tg.add_deadline("release", time.time() + 86400, domain="eng")
        assert a.anchor_type == AnchorType.DEADLINE
        assert a.domain == "eng"

    def test_add_recurrence(self):
        tg = TemporalGrounder()
        a = tg.add_recurrence("standup", time.time() + 3600, 86400)
        assert a.anchor_type == AnchorType.RECURRENCE
        assert a.recurrence_interval == 86400

    def test_get_anchor_by_name(self):
        tg = TemporalGrounder()
        tg.add_anchor("deploy", time.time() + 3600)
        found = tg.get_anchor("deploy")
        assert found is not None
        assert found.name == "deploy"

    def test_remove_anchor(self):
        tg = TemporalGrounder()
        a = tg.add_anchor("temp", time.time() + 100)
        assert tg.remove_anchor(a.id) is True
        assert tg.get_anchor("temp") is None

    def test_remove_nonexistent(self):
        tg = TemporalGrounder()
        assert tg.remove_anchor("fake_id") is False


# ---------------------------------------------------------------------------
# TemporalGrounder — Queries
# ---------------------------------------------------------------------------


class TestTemporalGrounderQueries:
    def test_due_anchors(self):
        tg = TemporalGrounder()
        tg.add_deadline("past", time.time() - 100)
        tg.add_deadline("future", time.time() + 3600)
        due = tg.due_anchors()
        assert len(due) == 1
        assert due[0].name == "past"

    def test_upcoming(self):
        tg = TemporalGrounder()
        tg.add_anchor("soon", time.time() + 100)
        tg.add_anchor("later", time.time() + 3600)
        tg.add_anchor("far", time.time() + 86400 * 30)
        upcoming = tg.upcoming(horizon_seconds=7200)
        names = [a.name for a in upcoming]
        assert "soon" in names
        assert "later" in names
        assert "far" not in names

    def test_upcoming_domain_filter(self):
        tg = TemporalGrounder()
        tg.add_anchor("a", time.time() + 100, domain="eng")
        tg.add_anchor("b", time.time() + 200, domain="finance")
        upcoming = tg.upcoming(domain="eng")
        assert len(upcoming) == 1
        assert upcoming[0].name == "a"

    def test_deadlines_sorted_by_urgency(self):
        tg = TemporalGrounder()
        tg.add_deadline("urgent", time.time() + 60)
        tg.add_deadline("relaxed", time.time() + 86400 * 7)
        deadlines = tg.deadlines()
        assert deadlines[0].name == "urgent"

    def test_tick_fires_due(self):
        tg = TemporalGrounder()
        tg.add_anchor("past_event", time.time() - 10)
        fired = tg.tick()
        assert len(fired) == 1
        assert fired[0].name == "past_event"
        assert fired[0].active is False  # deactivated after firing

    def test_tick_advances_recurrence(self):
        tg = TemporalGrounder()
        now = time.time()
        tg.add_recurrence("daily", now - 10, 86400)
        fired = tg.tick()
        assert len(fired) == 1
        a = tg.get_anchor("daily")
        assert a is not None
        assert a.timestamp > now  # advanced
        assert a.active is True  # still active


# ---------------------------------------------------------------------------
# TemporalGrounder — Temporal Context
# ---------------------------------------------------------------------------


class TestTemporalGrounderContext:
    def test_touch_updates_interaction(self):
        tg = TemporalGrounder()
        old_time = tg._last_interaction
        time.sleep(0.01)
        tg.touch()
        assert tg._last_interaction > old_time
        assert tg._total_interactions == 1

    def test_elapsed_since_last(self):
        tg = TemporalGrounder()
        tg.touch()
        time.sleep(0.01)
        assert tg.elapsed_since_last() > 0

    def test_session_duration(self):
        tg = TemporalGrounder()
        assert tg.session_duration() >= 0

    def test_generate_context(self):
        tg = TemporalGrounder()
        tg.add_deadline("release", time.time() + 3600, description="Ship v1.0")
        ctx = tg.generate_context()
        assert "Current time:" in ctx
        assert "Session duration:" in ctx
        assert "release" in ctx

    def test_generate_context_with_overdue(self):
        tg = TemporalGrounder()
        tg.add_deadline("overdue_task", time.time() - 100)
        ctx = tg.generate_context()
        assert "overdue" in ctx.lower() or "remaining" in ctx.lower()


# ---------------------------------------------------------------------------
# TemporalGrounder — Capacity & Persistence
# ---------------------------------------------------------------------------


class TestTemporalGrounderCapacityPersistence:
    def test_capacity_enforcement(self):
        tg = TemporalGrounder(max_anchors=5)
        for i in range(10):
            tg.add_anchor(f"event_{i}", time.time() + i * 100)
        assert len(tg._anchors) <= 5

    def test_save_load_roundtrip(self):
        tg = TemporalGrounder()
        tg.add_deadline("release", time.time() + 3600, domain="eng")
        tg.add_recurrence("standup", time.time() + 100, 86400)
        tg.touch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "temporal.json"
            tg.save_state(path)

            tg2 = TemporalGrounder()
            assert tg2.load_state(path)
            assert len(tg2._anchors) == 2
            assert tg2._total_interactions == 1

    def test_load_nonexistent_returns_false(self):
        tg = TemporalGrounder()
        assert tg.load_state("/nonexistent.json") is False

    def test_summary(self):
        tg = TemporalGrounder()
        tg.add_deadline("x", time.time() + 3600)
        tg.add_recurrence("y", time.time() + 100, 86400)
        s = tg.summary()
        assert "TemporalGrounder" in s
        assert "1 deadlines" in s
        assert "1 recurrences" in s


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_world_sense_time(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.world_sense_time()
        assert "unix" in result
        assert "iso" in result

    def test_world_sense_system(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.world_sense_system()
        assert "os" in result
        assert "hostname" in result

    def test_world_sense_filesystem(self):
        from emms import EMMS
        agent = EMMS()
        with tempfile.NamedTemporaryFile() as f:
            result = agent.world_sense_filesystem(f.name)
            assert result["exists"] is True

    def test_world_scan(self):
        from emms import EMMS
        agent = EMMS()
        readings = agent.world_scan()
        assert len(readings) >= 2

    def test_reality_check(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.reality_check("The sky is blue")
        assert "status" in result
        assert "confidence_delta" in result

    def test_temporal_add_deadline(self):
        from emms import EMMS
        agent = EMMS()
        anchor = agent.temporal_add_deadline("release", time.time() + 3600, "eng")
        assert "name" in anchor
        assert anchor["name"] == "release"

    def test_temporal_context(self):
        from emms import EMMS
        agent = EMMS()
        ctx = agent.temporal_context()
        assert "Current time:" in ctx

    def test_temporal_tick(self):
        from emms import EMMS
        agent = EMMS()
        agent.temporal_add_deadline("past", time.time() - 10)
        fired = agent.temporal_tick()
        assert len(fired) == 1

    def test_grounding_summary(self):
        from emms import EMMS
        agent = EMMS()
        s = agent.grounding_summary()
        assert "WorldSensor" in s
        assert "TemporalGrounder" in s

    def test_save_load_roundtrip(self):
        from emms import EMMS
        agent = EMMS()
        agent.world_sense_time()
        agent.temporal_add_deadline("test", time.time() + 3600)
        agent.reality_check("test belief")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = EMMS()
            agent2.load(str(path))
            # Verify state persisted
            s = agent2.grounding_summary()
            assert "WorldSensor" in s
