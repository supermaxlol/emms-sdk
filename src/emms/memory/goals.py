"""GoalStack — hierarchical goal management with lifecycle tracking.

v0.17.0: The Goal-Directed Mind

Intelligent agents don't just react to the present — they pursue future
states. GoalStack models this top-down, goal-directed cognition: goals
are pushed with a priority and optional parent (forming a hierarchy),
moved through a lifecycle (pending → active → completed/failed/abandoned),
and can reference supporting memories that motivated them.

Goal lifecycle
--------------
``pending``    — created, not yet started
``active``     — currently being pursued
``completed``  — successfully achieved (with optional outcome note)
``failed``     — attempted but not achieved (with reason)
``abandoned``  — dropped without attempt (with reason)

Biological analogue: hierarchical task decomposition in the prefrontal
cortex (Koechlin & Summerfield 2007) — nested subgoal structures underlie
complex planning; prefrontal goal maintenance (Miller & Cohen 2001) —
sustained representations of future desired states bias processing across
the rest of the brain.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """A single goal with its full lifecycle state."""

    id: str
    content: str
    domain: str
    priority: float                    # 0..1 — higher = more urgent
    status: str                        # pending | active | completed | failed | abandoned
    parent_id: Optional[str]
    deadline: Optional[float]          # unix timestamp (None = no deadline)
    created_at: float
    resolved_at: Optional[float]       # set when completed/failed/abandoned
    supporting_memory_ids: list[str]
    outcome_note: str                  # rationale / outcome text

    def summary(self) -> str:
        deadline_str = ""
        if self.deadline:
            days = (self.deadline - time.time()) / 86400
            deadline_str = f"  deadline={days:.1f}d"
        return (
            f"Goal [{self.status.upper()}] priority={self.priority:.2f}{deadline_str}\n"
            f"  {self.id[:12]}: {self.content[:80]}"
        )

    def is_resolved(self) -> bool:
        """Return True if this goal has reached a terminal state."""
        return self.status in ("completed", "failed", "abandoned")

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "id": self.id,
            "content": self.content,
            "domain": self.domain,
            "priority": self.priority,
            "status": self.status,
            "parent_id": self.parent_id,
            "deadline": self.deadline,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "supporting_memory_ids": list(self.supporting_memory_ids),
            "outcome_note": self.outcome_note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        """Reconstruct a Goal from a serialized dict."""
        return cls(
            id=d["id"],
            content=d["content"],
            domain=d.get("domain", "general"),
            priority=d.get("priority", 0.5),
            status=d.get("status", "pending"),
            parent_id=d.get("parent_id"),
            deadline=d.get("deadline"),
            created_at=d.get("created_at", time.time()),
            resolved_at=d.get("resolved_at"),
            supporting_memory_ids=list(d.get("supporting_memory_ids", [])),
            outcome_note=d.get("outcome_note", ""),
        )


@dataclass
class GoalReport:
    """Summary statistics for the full goal hierarchy."""

    total: int
    active: int
    completed: int
    failed: int
    pending: int
    abandoned: int
    goals: list[Goal]   # all goals, sorted by priority desc

    def summary(self) -> str:
        lines = [
            f"GoalReport: {self.total} total — "
            f"{self.active} active, {self.pending} pending, "
            f"{self.completed} completed, {self.failed} failed, "
            f"{self.abandoned} abandoned",
        ]
        for g in self.goals[:5]:
            lines.append(f"  [{g.status:9s}] p={g.priority:.2f} {g.content[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GoalStack
# ---------------------------------------------------------------------------


class GoalStack:
    """Hierarchical goal manager.

    Parameters
    ----------
    memory:
        Optional :class:`HierarchicalMemory` instance.  Not required for
        goal management, but stored for future integration.
    """

    def __init__(self, memory: Any = None) -> None:
        self.memory = memory
        self._goals: dict[str, Goal] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        content: str,
        domain: str = "general",
        priority: float = 0.5,
        parent_id: Optional[str] = None,
        deadline: Optional[float] = None,
        supporting_memory_ids: Optional[list[str]] = None,
    ) -> Goal:
        """Create a new goal and add it to the stack.

        Args:
            content:               Goal description.
            domain:                Knowledge domain this goal belongs to.
            priority:              Urgency 0..1 (higher = more urgent).
            parent_id:             ID of a parent goal (sub-goal creation).
            deadline:              Optional unix timestamp deadline.
            supporting_memory_ids: Memory IDs that motivated this goal.

        Returns:
            The newly created :class:`Goal`.
        """
        priority = max(0.0, min(1.0, priority))
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            content=content,
            domain=domain,
            priority=priority,
            status="pending",
            parent_id=parent_id,
            deadline=deadline,
            created_at=time.time(),
            resolved_at=None,
            supporting_memory_ids=list(supporting_memory_ids or []),
            outcome_note="",
        )
        self._goals[goal.id] = goal
        return goal

    def activate(self, goal_id: str) -> bool:
        """Move a pending goal to active status.

        Args:
            goal_id: ID of the goal to activate.

        Returns:
            ``True`` if found and activated; ``False`` otherwise.
        """
        goal = self._goals.get(goal_id)
        if goal is None or goal.status != "pending":
            return False
        goal.status = "active"
        return True

    def complete(self, goal_id: str, outcome_note: str = "") -> bool:
        """Mark a goal as completed.

        Args:
            goal_id:      ID of the goal to complete.
            outcome_note: Optional text describing the outcome.

        Returns:
            ``True`` if found and completed; ``False`` otherwise.
        """
        goal = self._goals.get(goal_id)
        if goal is None or goal.is_resolved():
            return False
        goal.status = "completed"
        goal.resolved_at = time.time()
        goal.outcome_note = outcome_note
        return True

    def fail(self, goal_id: str, reason: str = "") -> bool:
        """Mark a goal as failed.

        Args:
            goal_id: ID of the goal to fail.
            reason:  Optional text describing why it failed.

        Returns:
            ``True`` if found and failed; ``False`` otherwise.
        """
        goal = self._goals.get(goal_id)
        if goal is None or goal.is_resolved():
            return False
        goal.status = "failed"
        goal.resolved_at = time.time()
        goal.outcome_note = reason
        return True

    def abandon(self, goal_id: str, reason: str = "") -> bool:
        """Abandon a goal without attempting it.

        Args:
            goal_id: ID of the goal to abandon.
            reason:  Optional text explaining the abandonment.

        Returns:
            ``True`` if found and abandoned; ``False`` otherwise.
        """
        goal = self._goals.get(goal_id)
        if goal is None or goal.is_resolved():
            return False
        goal.status = "abandoned"
        goal.resolved_at = time.time()
        goal.outcome_note = reason
        return True

    def active_goals(self) -> list[Goal]:
        """Return all active goals sorted by priority descending.

        Returns:
            List of :class:`Goal` with ``status == "active"``.
        """
        return sorted(
            (g for g in self._goals.values() if g.status == "active"),
            key=lambda g: g.priority,
            reverse=True,
        )

    def pending_goals(self) -> list[Goal]:
        """Return all pending goals sorted by priority descending.

        Returns:
            List of :class:`Goal` with ``status == "pending"``.
        """
        return sorted(
            (g for g in self._goals.values() if g.status == "pending"),
            key=lambda g: g.priority,
            reverse=True,
        )

    def sub_goals(self, goal_id: str) -> list[Goal]:
        """Return all direct children of a goal.

        Args:
            goal_id: Parent goal ID.

        Returns:
            List of :class:`Goal` whose ``parent_id`` matches.
        """
        return [g for g in self._goals.values() if g.parent_id == goal_id]

    def get(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a goal by ID.

        Args:
            goal_id: Goal ID to look up.

        Returns:
            The :class:`Goal` or ``None`` if not found.
        """
        return self._goals.get(goal_id)

    def report(self) -> GoalReport:
        """Generate a summary report of the full goal hierarchy.

        Returns:
            :class:`GoalReport` with counts per status.
        """
        all_goals = list(self._goals.values())
        return GoalReport(
            total=len(all_goals),
            active=sum(1 for g in all_goals if g.status == "active"),
            completed=sum(1 for g in all_goals if g.status == "completed"),
            failed=sum(1 for g in all_goals if g.status == "failed"),
            pending=sum(1 for g in all_goals if g.status == "pending"),
            abandoned=sum(1 for g in all_goals if g.status == "abandoned"),
            goals=sorted(all_goals, key=lambda g: g.priority, reverse=True),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Serialize all goals to a JSON file (atomic write)."""
        import os
        import tempfile

        path = Path(path)
        data = {
            "version": "0.17.0",
            "saved_at": time.time(),
            "goals": [g.to_dict() for g in self._goals.values()],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_state(self, path: str | Path) -> None:
        """Restore goals from a JSON file."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._goals.clear()
        for gd in data.get("goals", []):
            goal = Goal.from_dict(gd)
            self._goals[goal.id] = goal
