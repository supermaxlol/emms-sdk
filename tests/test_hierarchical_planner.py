"""Tests for HierarchicalPlanner (Gap 4: AGI Roadmap — Agency)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from emms.memory.hierarchical_planner import (
    HierarchicalPlanner,
    Plan,
    PlanStep,
)
from emms.memory.autonomy_governor import AutonomyGovernor
from emms.memory.goal_generator import GeneratedGoal


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_creation(self):
        s = PlanStep(step_id="step_0", description="Do thing",
                     action_type="retrieve")
        assert s.status == "pending"

    def test_serialization(self):
        s = PlanStep(step_id="step_1", description="Analyze",
                     action_type="analyze", depends_on=["step_0"])
        d = s.to_dict()
        s2 = PlanStep.from_dict(d)
        assert s2.step_id == s.step_id
        assert s2.depends_on == s.depends_on


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_ready_steps_initial(self):
        p = Plan(plan_id="p1", goal_description="test",
                 steps=[
                     PlanStep(step_id="s0", description="first", action_type="retrieve"),
                     PlanStep(step_id="s1", description="second", action_type="analyze",
                              depends_on=["s0"]),
                 ])
        ready = p.ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "s0"

    def test_ready_steps_after_completion(self):
        p = Plan(plan_id="p1", goal_description="test",
                 steps=[
                     PlanStep(step_id="s0", description="first", action_type="retrieve",
                              status="completed"),
                     PlanStep(step_id="s1", description="second", action_type="analyze",
                              depends_on=["s0"]),
                 ])
        ready = p.ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "s1"

    def test_progress(self):
        p = Plan(plan_id="p1", goal_description="test",
                 steps=[
                     PlanStep(step_id="s0", description="a", action_type="retrieve",
                              status="completed"),
                     PlanStep(step_id="s1", description="b", action_type="analyze"),
                 ])
        assert p.progress() == 0.5

    def test_is_complete(self):
        p = Plan(plan_id="p1", goal_description="test",
                 steps=[
                     PlanStep(step_id="s0", description="a", action_type="retrieve",
                              status="completed"),
                     PlanStep(step_id="s1", description="b", action_type="analyze",
                              status="skipped"),
                 ])
        assert p.is_complete() is True

    def test_serialization(self):
        p = Plan(plan_id="p1", goal_description="test goal",
                 steps=[PlanStep(step_id="s0", description="a", action_type="retrieve")])
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        assert p2.plan_id == p.plan_id
        assert len(p2.steps) == 1


# ---------------------------------------------------------------------------
# HierarchicalPlanner — Planning
# ---------------------------------------------------------------------------


class TestHierarchicalPlannerPlanning:
    def test_plan_creates_steps(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Investigate why finance predictions fail", domain="finance")
        assert len(plan.steps) > 0
        assert plan.status == "pending"

    def test_plan_steps_have_dependencies(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Monitor system health")
        for i, step in enumerate(plan.steps):
            if i > 0:
                assert f"step_{i-1}" in step.depends_on

    def test_template_matching_investigate(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Investigate and analyze market trends")
        # Should match "investigate" template
        action_types = [s.action_type for s in plan.steps]
        assert "retrieve" in action_types

    def test_template_matching_consolidate(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Consolidate and deduplicate finance memories")
        action_types = [s.action_type for s in plan.steps]
        assert "consolidate_memory" in action_types or "analyze" in action_types

    def test_template_matching_verify(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Verify prediction calibration for tech domain")
        action_types = [s.action_type for s in plan.steps]
        assert "sense_world" in action_types

    def test_plan_from_goal(self):
        hp = HierarchicalPlanner()
        goal = GeneratedGoal(description="Research new strategy",
                             source="curiosity", domain="trading")
        plan = hp.plan_from_goal(goal)
        assert plan.source == "curiosity"
        assert plan.domain == "trading"

    def test_plan_with_governor_marks_needs_approval(self):
        hp = HierarchicalPlanner()
        gov = AutonomyGovernor()
        # "store_for_human" is a SUGGEST action
        plan = hp.plan("Monitor and report findings", governor=gov)
        suggest_steps = [s for s in plan.steps if s.status == "needs_approval"]
        # At least one step should need approval (store_for_human)
        has_suggest = any(s.action_type in AutonomyGovernor.SUGGEST_ACTIONS
                         for s in plan.steps)
        if has_suggest:
            assert len(suggest_steps) >= 1

    def test_add_custom_template(self):
        hp = HierarchicalPlanner()
        hp.add_template("custom", [
            {"action": "analyze", "desc": "Custom analysis"},
        ])
        assert "custom" in hp._templates


# ---------------------------------------------------------------------------
# HierarchicalPlanner — Execution
# ---------------------------------------------------------------------------


class TestHierarchicalPlannerExecution:
    def test_execute_step(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        step_id = plan.steps[0].step_id
        assert hp.execute_step(pid, step_id, result="done") is True
        assert plan.steps[0].status == "completed"

    def test_execute_step_failure(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        step_id = plan.steps[0].step_id
        hp.execute_step(pid, step_id, error="something broke")
        assert plan.steps[0].status == "failed"

    def test_plan_completes_when_all_done(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        for step in plan.steps:
            hp.execute_step(pid, step.step_id, result="done")
        # Plan should be moved to completed
        assert hp.get_plan(pid) is None  # no longer active
        assert any(p.plan_id == pid for p in hp._completed_plans)

    def test_replan_skips_failed(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        first_step = plan.steps[0].step_id
        hp.execute_step(pid, first_step, error="failed")
        replanned = hp.replan(pid, first_step)
        assert replanned is not None
        assert replanned.status == "replanned"
        # Second step should no longer depend on first
        if len(plan.steps) > 1:
            assert first_step not in plan.steps[1].depends_on

    def test_cancel_plan(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        assert hp.cancel_plan(pid) is True
        assert hp.get_plan(pid) is None

    def test_execute_nonexistent_plan(self):
        hp = HierarchicalPlanner()
        assert hp.execute_step("fake_id", "step_0") is False

    def test_ready_steps_all(self):
        hp = HierarchicalPlanner()
        hp.plan("Plan A")
        hp.plan("Plan B")
        ready = hp.ready_steps_all()
        assert len(ready) >= 2  # at least step_0 from each


# ---------------------------------------------------------------------------
# HierarchicalPlanner — Learning & Queries
# ---------------------------------------------------------------------------


class TestHierarchicalPlannerLearning:
    def test_learn_template_from_success(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan")
        pid = plan.plan_id
        for step in plan.steps:
            hp.execute_step(pid, step.step_id, result="done")
        # Now learn from the completed plan
        completed = [p for p in hp._completed_plans if p.plan_id == pid][0]
        name = hp.learn_template(completed)
        assert name is not None
        assert name in hp._templates

    def test_learn_from_incomplete_returns_none(self):
        plan = Plan(plan_id="p1", goal_description="test", status="executing")
        hp = HierarchicalPlanner()
        assert hp.learn_template(plan) is None

    def test_active_plans_property(self):
        hp = HierarchicalPlanner()
        hp.plan("A")
        hp.plan("B")
        assert len(hp.active_plans) == 2

    def test_summary(self):
        hp = HierarchicalPlanner()
        hp.plan("Test")
        s = hp.summary()
        assert "HierarchicalPlanner" in s
        assert "1 active" in s


# ---------------------------------------------------------------------------
# HierarchicalPlanner — Persistence
# ---------------------------------------------------------------------------


class TestHierarchicalPlannerPersistence:
    def test_save_load_roundtrip(self):
        hp = HierarchicalPlanner()
        plan = hp.plan("Test plan", domain="finance")
        hp.execute_step(plan.plan_id, plan.steps[0].step_id, result="done")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "planner.json"
            hp.save_state(path)

            hp2 = HierarchicalPlanner()
            assert hp2.load_state(path)
            assert hp2._total_plans == hp._total_plans
            assert len(hp2._active_plans) == len(hp._active_plans)

    def test_load_nonexistent_returns_false(self):
        hp = HierarchicalPlanner()
        assert hp.load_state("/nonexistent.json") is False


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_authorize_action(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.authorize_action("retrieve", "market data")
        assert result["authorized"] is True

    def test_authorize_forbidden(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.authorize_action("delete_without_backup", "database")
        assert result["authorized"] is False

    def test_generate_goals(self):
        from emms import EMMS
        agent = EMMS()
        goals = agent.generate_autonomous_goals(
            capability_profile={"finance": 0.2},
            belief_counts={"finance": 10},
        )
        assert isinstance(goals, list)

    def test_plan_goal(self):
        from emms import EMMS
        agent = EMMS()
        plan = agent.plan_goal("Investigate market trends", domain="finance")
        assert "plan_id" in plan
        assert "steps" in plan

    def test_pending_approvals(self):
        from emms import EMMS
        agent = EMMS()
        agent.authorize_action("write_file", "/tmp/test.txt")
        approvals = agent.pending_approvals()
        assert len(approvals) >= 1

    def test_approve_action(self):
        from emms import EMMS
        agent = EMMS()
        agent.authorize_action("write_file", "/tmp/test.txt")
        approved = agent.approve_action(0)
        assert approved["action_type"] == "write_file"

    def test_agency_summary(self):
        from emms import EMMS
        agent = EMMS()
        s = agent.agency_summary()
        assert "AutonomyGovernor" in s
        assert "GoalGenerator" in s
        assert "HierarchicalPlanner" in s

    def test_save_load_roundtrip(self):
        from emms import EMMS
        agent = EMMS()
        agent.authorize_action("retrieve", "test")
        agent.plan_goal("Test plan")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = EMMS()
            agent2.load(str(path))
            s = agent2.agency_summary()
            assert "AutonomyGovernor" in s
