"""Tests for GoalStack and ProspectiveMemory persistence (save/load across restarts)."""

import json
import time
from pathlib import Path

import pytest

from emms.memory.goals import Goal, GoalStack
from emms.memory.prospection import ProspectiveMemory


@pytest.fixture
def tmp_goals_path(tmp_path):
    return tmp_path / "test_goals.json"


class TestGoalSerialization:
    """Goal.to_dict / Goal.from_dict round-trip."""

    def test_round_trip_basic(self):
        g = Goal(
            id="goal_abc123",
            content="Ship v1",
            domain="engineering",
            priority=0.9,
            status="active",
            parent_id=None,
            deadline=None,
            created_at=1000.0,
            resolved_at=None,
            supporting_memory_ids=["mem_1", "mem_2"],
            outcome_note="",
        )
        d = g.to_dict()
        g2 = Goal.from_dict(d)
        assert g2.id == g.id
        assert g2.content == g.content
        assert g2.domain == g.domain
        assert g2.priority == g.priority
        assert g2.status == g.status
        assert g2.parent_id is None
        assert g2.deadline is None
        assert g2.supporting_memory_ids == ["mem_1", "mem_2"]
        assert g2.outcome_note == ""

    def test_round_trip_resolved(self):
        g = Goal(
            id="goal_xyz",
            content="Fix bug",
            domain="ops",
            priority=0.3,
            status="completed",
            parent_id="goal_parent",
            deadline=2000.0,
            created_at=1000.0,
            resolved_at=1500.0,
            supporting_memory_ids=[],
            outcome_note="Deployed fix",
        )
        d = g.to_dict()
        g2 = Goal.from_dict(d)
        assert g2.status == "completed"
        assert g2.resolved_at == 1500.0
        assert g2.parent_id == "goal_parent"
        assert g2.deadline == 2000.0
        assert g2.outcome_note == "Deployed fix"

    def test_json_serializable(self):
        g = Goal(
            id="goal_json",
            content="Test JSON",
            domain="test",
            priority=0.5,
            status="pending",
            parent_id=None,
            deadline=None,
            created_at=time.time(),
            resolved_at=None,
            supporting_memory_ids=[],
            outcome_note="",
        )
        # Must not raise
        json.dumps(g.to_dict())


class TestGoalStackPersistence:
    """GoalStack.save_state / load_state."""

    def test_save_load_empty(self, tmp_goals_path):
        gs = GoalStack()
        gs.save_state(tmp_goals_path)
        assert tmp_goals_path.exists()

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        assert gs2.report().total == 0

    def test_save_load_with_goals(self, tmp_goals_path):
        gs = GoalStack()
        g1 = gs.push("Goal A", domain="eng", priority=0.9)
        g2 = gs.push("Goal B", domain="ops", priority=0.3)
        gs.activate(g1.id)
        gs.complete(g2.id, outcome_note="Done")

        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        report = gs2.report()
        assert report.total == 2
        assert report.active == 1
        assert report.completed == 1

        restored_a = gs2.get(g1.id)
        assert restored_a is not None
        assert restored_a.content == "Goal A"
        assert restored_a.status == "active"
        assert restored_a.priority == 0.9

        restored_b = gs2.get(g2.id)
        assert restored_b is not None
        assert restored_b.status == "completed"
        assert restored_b.outcome_note == "Done"

    def test_save_load_hierarchy(self, tmp_goals_path):
        gs = GoalStack()
        parent = gs.push("Parent goal", priority=1.0)
        child1 = gs.push("Child 1", parent_id=parent.id, priority=0.8)
        child2 = gs.push("Child 2", parent_id=parent.id, priority=0.6)

        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        children = gs2.sub_goals(parent.id)
        assert len(children) == 2
        assert {c.content for c in children} == {"Child 1", "Child 2"}

    def test_load_nonexistent_is_noop(self, tmp_path):
        gs = GoalStack()
        gs.push("Existing goal")
        gs.load_state(tmp_path / "does_not_exist.json")
        # Should not crash, and should not clear existing goals
        assert gs.report().total == 1

    def test_save_overwrites(self, tmp_goals_path):
        gs = GoalStack()
        gs.push("First save")
        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.push("Second save")
        gs2.save_state(tmp_goals_path)

        gs3 = GoalStack()
        gs3.load_state(tmp_goals_path)
        assert gs3.report().total == 1
        assert gs3.report().goals[0].content == "Second save"

    def test_all_statuses_persist(self, tmp_goals_path):
        gs = GoalStack()
        g_pending = gs.push("Pending")
        g_active = gs.push("Active")
        g_done = gs.push("Done")
        g_fail = gs.push("Failed")
        g_abandon = gs.push("Abandoned")

        gs.activate(g_active.id)
        gs.complete(g_done.id, outcome_note="shipped")
        gs.fail(g_fail.id, reason="blocked")
        gs.abandon(g_abandon.id, reason="not needed")

        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        r = gs2.report()
        assert r.pending == 1
        assert r.active == 1
        assert r.completed == 1
        assert r.failed == 1
        assert r.abandoned == 1

    def test_supporting_memory_ids_persist(self, tmp_goals_path):
        gs = GoalStack()
        g = gs.push("With memories", supporting_memory_ids=["m1", "m2", "m3"])
        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        restored = gs2.get(g.id)
        assert restored.supporting_memory_ids == ["m1", "m2", "m3"]

    def test_deadline_persists(self, tmp_goals_path):
        deadline = time.time() + 86400
        gs = GoalStack()
        g = gs.push("Deadline goal", deadline=deadline)
        gs.save_state(tmp_goals_path)

        gs2 = GoalStack()
        gs2.load_state(tmp_goals_path)
        assert gs2.get(g.id).deadline == deadline


class TestIntentionsPersistence:
    """ProspectiveMemory.save / load."""

    def test_save_load_intentions(self, tmp_path):
        path = tmp_path / "intentions.json"
        pm = ProspectiveMemory()
        i1 = pm.intend(
            content="Review PR",
            trigger_context="when code review is mentioned",
            priority=0.9,
        )
        i2 = pm.intend(
            content="Update docs",
            trigger_context="after deploy",
            priority=0.5,
        )
        pm.save(path)

        pm2 = ProspectiveMemory()
        assert pm2.load(path) is True
        pending = pm2.pending()
        assert len(pending) == 2
        contents = {i.content for i in pending}
        assert "Review PR" in contents
        assert "Update docs" in contents

    def test_load_nonexistent(self, tmp_path):
        pm = ProspectiveMemory()
        assert pm.load(tmp_path / "nope.json") is False

    def test_fulfilled_intentions_persist(self, tmp_path):
        path = tmp_path / "intentions.json"
        pm = ProspectiveMemory()
        i = pm.intend(content="Do thing", trigger_context="now", priority=0.5)
        pm.fulfill(i.id)
        pm.save(path)

        pm2 = ProspectiveMemory()
        pm2.load(path)
        assert len(pm2.pending()) == 0
        assert len(pm2.all_intentions()) == 1
        assert pm2.all_intentions()[0].fulfilled is True

    def test_emms_integration(self, tmp_path):
        """Goals + intentions both survive EMMS save/load cycle."""
        from emms import EMMS

        state_path = tmp_path / "emms_state.json"
        e = EMMS()
        g = e.push_goal("Integration test goal", priority=0.8)
        e.intend(content="Check tests", trigger_context="session start", priority=0.7)
        e.save(str(state_path))

        goals_file = tmp_path / "emms_state_goals.json"
        intentions_file = tmp_path / "emms_state_intentions.json"
        assert goals_file.exists()
        assert intentions_file.exists()

        e2 = EMMS()
        e2.load(str(state_path))
        assert e2.goal_report().total == 1
        assert e2.goal_report().goals[0].content == "Integration test goal"
        assert len(e2.pending_intentions()) == 1
        assert e2.pending_intentions()[0].content == "Check tests"
