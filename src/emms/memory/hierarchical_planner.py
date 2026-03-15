"""Gap 4 — HierarchicalPlanner: task decomposition + execution.

Decomposes goals into executable plans using template matching and
heuristic decomposition. Each leaf step maps to an action_type
checked by AutonomyGovernor.

NOT using LLM for planning — uses template matching + heuristic
decomposition. Simple but reliable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emms.memory.autonomy_governor import AutonomyGovernor
    from emms.memory.goal_generator import GeneratedGoal


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: str
    description: str
    action_type: str  # maps to AutonomyGovernor action types
    target: str = ""
    parameters: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, skipped, needs_approval
    result: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlanStep:
        return cls(
            step_id=d["step_id"],
            description=d["description"],
            action_type=d["action_type"],
            target=d.get("target", ""),
            parameters=d.get("parameters", {}),
            depends_on=d.get("depends_on", []),
            status=d.get("status", "pending"),
            result=d.get("result", ""),
            error=d.get("error", ""),
        )


@dataclass
class Plan:
    """A hierarchical execution plan for a goal."""
    plan_id: str
    goal_description: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, executing, completed, failed, replanned
    domain: str = "general"
    source: str = "human"  # human or goal source

    def ready_steps(self) -> list[PlanStep]:
        """Steps whose dependencies are all completed."""
        completed_ids = {s.step_id for s in self.steps if s.status == "completed"}
        return [s for s in self.steps
                if s.status == "pending"
                and all(d in completed_ids for d in s.depends_on)]

    def progress(self) -> float:
        """Fraction of steps completed."""
        if not self.steps:
            return 1.0
        return sum(1 for s in self.steps if s.status == "completed") / len(self.steps)

    def is_complete(self) -> bool:
        return all(s.status in ("completed", "skipped") for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal_description": self.goal_description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "status": self.status,
            "domain": self.domain,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(
            plan_id=d["plan_id"],
            goal_description=d["goal_description"],
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            created_at=d.get("created_at", time.time()),
            status=d.get("status", "pending"),
            domain=d.get("domain", "general"),
            source=d.get("source", "human"),
        )


class HierarchicalPlanner:
    """Decomposes goals into executable plans.

    Uses template matching + heuristic decomposition to create
    step-by-step plans. Each step maps to an action type that
    the AutonomyGovernor can authorize.
    """

    def __init__(self, *, max_active: int = 20,
                 max_completed: int = 100) -> None:
        self._templates: dict[str, list[dict]] = self._default_templates()
        self._active_plans: dict[str, Plan] = {}
        self._completed_plans: list[Plan] = []
        self._max_active = max_active
        self._max_completed = max_completed
        self._total_plans = 0

    # -- templates ----------------------------------------------------------

    @staticmethod
    def _default_templates() -> dict[str, list[dict]]:
        return {
            "investigate": [
                {"action": "retrieve", "desc": "Retrieve relevant memories"},
                {"action": "sense_world", "desc": "Check current state"},
                {"action": "analyze", "desc": "Compare memory vs reality"},
                {"action": "store_for_human", "desc": "Report findings"},
            ],
            "consolidate": [
                {"action": "retrieve", "desc": "Retrieve all memories in domain"},
                {"action": "analyze", "desc": "Find duplicates and clusters"},
                {"action": "consolidate_memory", "desc": "Merge similar memories"},
            ],
            "verify": [
                {"action": "retrieve", "desc": "Retrieve belief to verify"},
                {"action": "sense_world", "desc": "Get current data"},
                {"action": "analyze", "desc": "Compare belief vs data"},
                {"action": "update_self_model", "desc": "Update belief confidence"},
            ],
            "monitor": [
                {"action": "sense_world", "desc": "Check system/process state"},
                {"action": "retrieve", "desc": "Retrieve expected state"},
                {"action": "analyze", "desc": "Compare expected vs actual"},
                {"action": "store_for_human", "desc": "Alert if divergence"},
            ],
            "follow_up": [
                {"action": "retrieve", "desc": "Retrieve goal context"},
                {"action": "analyze", "desc": "Assess what's blocking"},
                {"action": "store_for_human", "desc": "Suggest next action"},
            ],
            "research": [
                {"action": "search", "desc": "Search for information"},
                {"action": "retrieve", "desc": "Retrieve related memories"},
                {"action": "analyze", "desc": "Synthesize findings"},
                {"action": "store_for_human", "desc": "Present research summary"},
            ],
        }

    def add_template(self, name: str, steps: list[dict]) -> None:
        """Add a custom plan template."""
        self._templates[name] = steps

    # -- planning -----------------------------------------------------------

    def plan(self, goal_description: str, domain: str = "general",
             source: str = "human",
             governor: AutonomyGovernor | None = None) -> Plan:
        """Create a plan for a goal."""
        self._total_plans += 1

        template_key = self._match_template(goal_description)
        template = self._templates.get(template_key, self._templates["investigate"])

        steps = []
        for i, t in enumerate(template):
            step = PlanStep(
                step_id=f"step_{i}",
                description=f"{t['desc']} for: {goal_description[:50]}",
                action_type=t["action"],
                target=domain,
                parameters={"goal": goal_description, "domain": domain},
                depends_on=[f"step_{i-1}"] if i > 0 else [],
            )
            # Pre-check authorization if governor provided
            if governor:
                from emms.memory.autonomy_governor import ActionProposal
                auth = governor.authorize(ActionProposal(
                    action_type=step.action_type,
                    target=step.target,
                    source_goal=goal_description,
                ))
                if not auth.authorized and auth.tier.value == "suggest":
                    step.status = "needs_approval"
            steps.append(step)

        plan = Plan(
            plan_id=f"plan_{self._total_plans}_{int(time.time()) % 100000}",
            goal_description=goal_description,
            steps=steps,
            domain=domain,
            source=source,
        )

        self._active_plans[plan.plan_id] = plan
        self._enforce_capacity()
        return plan

    def plan_from_goal(self, goal: GeneratedGoal,
                       governor: AutonomyGovernor | None = None) -> Plan:
        """Create a plan from a GeneratedGoal object."""
        return self.plan(
            goal_description=goal.description,
            domain=goal.domain,
            source=goal.source,
            governor=governor,
        )

    def _match_template(self, goal_description: str) -> str:
        """Match goal to plan template by keyword overlap."""
        keywords = {
            "investigate": {"investigate", "understand", "why", "analyze", "curiosity", "explore"},
            "consolidate": {"consolidate", "deduplicate", "clean", "maintenance", "merge", "organize"},
            "verify": {"verify", "check", "calibrate", "test", "prediction", "confirm"},
            "monitor": {"monitor", "watch", "track", "status", "running", "health"},
            "follow_up": {"follow", "stalling", "reminder", "obligation", "overdue", "progress"},
            "research": {"research", "learn", "find", "search", "discover", "study"},
        }
        goal_tokens = set(goal_description.lower().split())
        best_match = "investigate"
        best_score = 0
        for template, kws in keywords.items():
            score = len(goal_tokens & kws)
            if score > best_score:
                best_score = score
                best_match = template
        return best_match

    # -- execution ----------------------------------------------------------

    def execute_step(self, plan_id: str, step_id: str,
                     result: str = "", error: str = "") -> bool:
        """Mark a step as completed or failed. Returns True if found."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return False
        for step in plan.steps:
            if step.step_id == step_id:
                step.status = "completed" if not error else "failed"
                step.result = result
                step.error = error
                break
        else:
            return False

        # Check if plan is complete
        if plan.is_complete():
            plan.status = "completed"
            self._completed_plans.append(plan)
            del self._active_plans[plan_id]
            if len(self._completed_plans) > self._max_completed:
                self._completed_plans = self._completed_plans[-self._max_completed:]
        elif any(s.status == "failed" for s in plan.steps):
            plan.status = "executing"  # still going, with a failure
        else:
            plan.status = "executing"
        return True

    def replan(self, plan_id: str, failed_step_id: str) -> Plan | None:
        """When a step fails, skip it and remove its dependency from later steps."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return None
        for step in plan.steps:
            if step.step_id == failed_step_id:
                step.status = "skipped"
            step.depends_on = [d for d in step.depends_on if d != failed_step_id]
        plan.status = "replanned"
        return plan

    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active plan. Returns True if found."""
        plan = self._active_plans.pop(plan_id, None)
        if plan:
            plan.status = "cancelled"
            self._completed_plans.append(plan)
            return True
        return False

    # -- queries ------------------------------------------------------------

    @property
    def active_plans(self) -> list[Plan]:
        return list(self._active_plans.values())

    def get_plan(self, plan_id: str) -> Plan | None:
        return self._active_plans.get(plan_id)

    def ready_steps_all(self) -> list[tuple[str, PlanStep]]:
        """Return all ready steps across all active plans."""
        result = []
        for pid, plan in self._active_plans.items():
            for step in plan.ready_steps():
                result.append((pid, step))
        return result

    def learn_template(self, plan: Plan) -> str | None:
        """Extract a successful plan's pattern as a new template. Returns template name."""
        if plan.status != "completed":
            return None
        completed_steps = [s for s in plan.steps if s.status == "completed"]
        if not completed_steps:
            return None
        name = f"learned_{self._total_plans}"
        self._templates[name] = [
            {"action": s.action_type, "desc": s.description[:60]}
            for s in completed_steps
        ]
        return name

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        active = len(self._active_plans)
        completed = len(self._completed_plans)
        templates = len(self._templates)
        return (
            f"HierarchicalPlanner: {active} active plans, "
            f"{completed} completed, {templates} templates, "
            f"{self._total_plans} total created"
        )

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_plans": self._total_plans,
            "templates": self._templates,
            "active_plans": {k: v.to_dict() for k, v in self._active_plans.items()},
            "completed_plans": [p.to_dict() for p in self._completed_plans[-self._max_completed:]],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_plans = data.get("total_plans", 0)
            self._templates = data.get("templates", self._default_templates())
            self._active_plans = {
                k: Plan.from_dict(v)
                for k, v in data.get("active_plans", {}).items()
            }
            self._completed_plans = [
                Plan.from_dict(p) for p in data.get("completed_plans", [])
            ]
            return True
        except Exception:
            return False

    # -- internal -----------------------------------------------------------

    def _enforce_capacity(self) -> None:
        if len(self._active_plans) <= self._max_active:
            return
        # Remove oldest completed-status plans first
        to_remove = []
        for pid, plan in sorted(self._active_plans.items(),
                                key=lambda x: x[1].created_at):
            if plan.status in ("completed", "cancelled"):
                to_remove.append(pid)
            if len(self._active_plans) - len(to_remove) <= self._max_active:
                break
        for pid in to_remove:
            self._active_plans.pop(pid, None)
