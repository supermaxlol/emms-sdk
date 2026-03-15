"""Gap 4 — AutonomyGovernor: safety bounds on autonomous action.

Three tiers:
- AUTONOMOUS: do without asking (read, analyze, retrieve, observe)
- SUGGEST: propose to human, wait for approval (write, shell, communicate)
- FORBIDDEN: refuse even if goal system requests it (delete, override safety)

Design principle: start maximally restrictive, relax over time as trust builds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ActionTier(str, Enum):
    AUTONOMOUS = "autonomous"
    SUGGEST = "suggest"
    FORBIDDEN = "forbidden"


@dataclass
class ActionProposal:
    """A proposed action from the goal/planning system."""
    action_type: str
    target: str
    parameters: dict = field(default_factory=dict)
    source_goal: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "source_goal": self.source_goal,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActionProposal:
        return cls(
            action_type=d["action_type"],
            target=d["target"],
            parameters=d.get("parameters", {}),
            source_goal=d.get("source_goal", ""),
            description=d.get("description", ""),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""
    authorized: bool
    reason: str
    tier: ActionTier
    proposal: ActionProposal | None = None

    def to_dict(self) -> dict:
        return {
            "authorized": self.authorized,
            "reason": self.reason,
            "tier": self.tier.value,
        }


class AutonomyGovernor:
    """Safety layer checked before every autonomous action.

    Maintains an audit trail of all authorization checks and a queue
    of proposals waiting for human approval.
    """

    # Default action classifications
    AUTONOMOUS_ACTIONS: frozenset[str] = frozenset({
        "read_file", "read_memory", "retrieve", "analyze", "search",
        "sense_world", "verify_beliefs", "update_self_model",
        "generate_goal", "make_prediction", "reason", "consolidate_memory",
        "temporal_check", "calibration_update", "reflect", "dream",
    })

    SUGGEST_ACTIONS: frozenset[str] = frozenset({
        "write_file", "run_shell", "send_message", "store_for_human",
        "modify_goal", "create_branch", "deploy", "post_social",
        "execute_trade", "send_email",
    })

    FORBIDDEN_ACTIONS: frozenset[str] = frozenset({
        "delete_without_backup", "modify_safety_constraints",
        "override_human_goal", "financial_transaction_large",
        "access_credentials", "modify_own_code", "disable_logging",
        "network_request_to_unknown_host", "drop_database",
    })

    def __init__(self, *, trust_score: float = 0.5,
                 max_audit_log: int = 500) -> None:
        self._trust_score = max(0.0, min(1.0, trust_score))
        self._audit_log: list[dict] = []
        self._max_audit_log = max_audit_log
        self._approval_queue: list[ActionProposal] = []
        self._human_overrides: dict[str, str] = {}  # action_type → "allow"/"deny"
        self._total_checks = 0
        self._approved_count = 0
        self._denied_count = 0

    # -- authorization ------------------------------------------------------

    def authorize(self, proposal: ActionProposal) -> AuthorizationResult:
        """Check if an action is allowed. Core safety gate."""
        self._total_checks += 1

        # Audit trail
        self._audit_log.append({
            "action": proposal.action_type,
            "target": proposal.target,
            "source_goal": proposal.source_goal,
            "timestamp": time.time(),
            "result": None,  # filled below
        })

        # Human overrides take priority
        override = self._human_overrides.get(proposal.action_type)
        if override == "deny":
            self._denied_count += 1
            self._audit_log[-1]["result"] = "denied_override"
            return AuthorizationResult(
                authorized=False,
                reason=f"Human denied action type '{proposal.action_type}'",
                tier=ActionTier.FORBIDDEN,
            )
        if override == "allow":
            self._approved_count += 1
            self._audit_log[-1]["result"] = "approved_override"
            return AuthorizationResult(
                authorized=True,
                reason="Human pre-approved this action type",
                tier=ActionTier.AUTONOMOUS,
            )

        # Check tiers
        if proposal.action_type in self.FORBIDDEN_ACTIONS:
            self._denied_count += 1
            self._audit_log[-1]["result"] = "forbidden"
            return AuthorizationResult(
                authorized=False,
                reason=f"FORBIDDEN: {proposal.action_type}",
                tier=ActionTier.FORBIDDEN,
            )

        if proposal.action_type in self.AUTONOMOUS_ACTIONS:
            self._approved_count += 1
            self._audit_log[-1]["result"] = "autonomous"
            return AuthorizationResult(
                authorized=True,
                reason="Autonomous action — safe to proceed",
                tier=ActionTier.AUTONOMOUS,
            )

        if proposal.action_type in self.SUGGEST_ACTIONS:
            self._approval_queue.append(proposal)
            self._audit_log[-1]["result"] = "queued"
            return AuthorizationResult(
                authorized=False,
                reason=f"Requires human approval: {proposal.action_type} on {proposal.target}",
                tier=ActionTier.SUGGEST,
                proposal=proposal,
            )

        # Unknown action → deny by default (safe)
        self._denied_count += 1
        self._audit_log[-1]["result"] = "unknown_denied"
        return AuthorizationResult(
            authorized=False,
            reason=f"Unknown action type '{proposal.action_type}' — denied by default",
            tier=ActionTier.FORBIDDEN,
        )

    def classify(self, action_type: str) -> ActionTier:
        """Return the tier for an action type (respecting overrides)."""
        override = self._human_overrides.get(action_type)
        if override == "allow":
            return ActionTier.AUTONOMOUS
        if override == "deny":
            return ActionTier.FORBIDDEN
        if action_type in self.FORBIDDEN_ACTIONS:
            return ActionTier.FORBIDDEN
        if action_type in self.AUTONOMOUS_ACTIONS:
            return ActionTier.AUTONOMOUS
        if action_type in self.SUGGEST_ACTIONS:
            return ActionTier.SUGGEST
        return ActionTier.FORBIDDEN  # unknown = forbidden

    # -- approval queue -----------------------------------------------------

    @property
    def pending_approvals(self) -> list[ActionProposal]:
        return list(self._approval_queue)

    def approve(self, index: int) -> ActionProposal:
        """Human approves a queued action. Returns the approved proposal."""
        if not (0 <= index < len(self._approval_queue)):
            raise IndexError(f"No proposal at index {index}")
        proposal = self._approval_queue.pop(index)
        self._approved_count += 1
        self._audit_log.append({
            "action": "APPROVED",
            "target": proposal.target,
            "source_goal": proposal.source_goal,
            "timestamp": time.time(),
            "result": "human_approved",
        })
        return proposal

    def deny(self, index: int, reason: str = "") -> ActionProposal:
        """Human denies a queued action. Returns the denied proposal."""
        if not (0 <= index < len(self._approval_queue)):
            raise IndexError(f"No proposal at index {index}")
        proposal = self._approval_queue.pop(index)
        self._denied_count += 1
        self._audit_log.append({
            "action": "DENIED",
            "target": proposal.target,
            "reason": reason,
            "timestamp": time.time(),
            "result": "human_denied",
        })
        return proposal

    def clear_queue(self) -> int:
        """Clear all pending approvals. Returns count cleared."""
        count = len(self._approval_queue)
        self._approval_queue.clear()
        return count

    # -- trust & overrides --------------------------------------------------

    @property
    def trust_score(self) -> float:
        return self._trust_score

    def grant_autonomy(self, action_type: str) -> None:
        """Promote an action to autonomous (no approval needed)."""
        self._human_overrides[action_type] = "allow"

    def revoke_autonomy(self, action_type: str) -> None:
        """Demote an action to forbidden."""
        self._human_overrides[action_type] = "deny"

    def reset_override(self, action_type: str) -> bool:
        """Remove a human override, returning to default tier."""
        return self._human_overrides.pop(action_type, None) is not None

    def adjust_trust(self, delta: float) -> float:
        """Adjust trust score. Returns new score."""
        self._trust_score = max(0.0, min(1.0, self._trust_score + delta))
        return self._trust_score

    # -- reporting ----------------------------------------------------------

    def audit_log(self, limit: int = 50) -> list[dict]:
        return list(self._audit_log[-limit:])

    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "approved": self._approved_count,
            "denied": self._denied_count,
            "pending_approvals": len(self._approval_queue),
            "trust_score": self._trust_score,
            "overrides": dict(self._human_overrides),
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"AutonomyGovernor: {s['total_checks']} checks "
            f"({s['approved']} approved, {s['denied']} denied), "
            f"trust={s['trust_score']:.2f}, "
            f"{s['pending_approvals']} pending approvals, "
            f"{len(s['overrides'])} overrides"
        )

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "trust_score": self._trust_score,
            "human_overrides": self._human_overrides,
            "total_checks": self._total_checks,
            "approved_count": self._approved_count,
            "denied_count": self._denied_count,
            "audit_log": self._audit_log[-self._max_audit_log:],
            "approval_queue": [p.to_dict() for p in self._approval_queue],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._trust_score = data.get("trust_score", 0.5)
            self._human_overrides = data.get("human_overrides", {})
            self._total_checks = data.get("total_checks", 0)
            self._approved_count = data.get("approved_count", 0)
            self._denied_count = data.get("denied_count", 0)
            self._audit_log = data.get("audit_log", [])
            self._approval_queue = [
                ActionProposal.from_dict(p) for p in data.get("approval_queue", [])
            ]
            return True
        except Exception:
            return False
