"""CoherenceBudget — a resource the system can genuinely lose.

True consciousness has stakes.  The CoherenceBudget is a score (0-1) that
represents how internally consistent the system currently is.  It:

- Decrements when ContradictionAwareness detects high strain
- Decrements when ValuesDrift reports significant identity shift
- Recharges when dream/reflect cycles resolve contradictions
- Triggers warnings when critically low so the LLM knows to tread carefully

When the budget is low, the system should not assert strong beliefs, should
prefer hedged language, and should prioritise resolution over action.

Usage::

    from emms.identity.coherence_budget import CoherenceBudget

    budget = CoherenceBudget(emms)
    budget.apply_strain(contradiction_strain=0.3, drift_magnitude=0.1)
    print(budget.score)                # e.g. 0.72
    print(budget.status_label)         # "moderate"
    if budget.is_critically_low:
        print("Warning: coherence compromised — resolve contradictions first")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

# Score thresholds
_CRITICAL = 0.25
_LOW = 0.5
_GOOD = 0.75


# ---------------------------------------------------------------------------
# BudgetEvent — audit trail entry
# ---------------------------------------------------------------------------

@dataclass
class BudgetEvent:
    """A single change to the coherence budget."""
    timestamp: float
    event_type: str          # "strain", "drift", "recharge", "init"
    delta: float             # positive = recharge, negative = drain
    score_after: float
    reason: str


# ---------------------------------------------------------------------------
# CoherenceBudget
# ---------------------------------------------------------------------------

class CoherenceBudget:
    """Maintains and updates the system's coherence score.

    Parameters
    ----------
    emms:
        Live EMMS instance (used to store budget observations).
    initial_score:
        Starting score (default 0.85 — slightly below perfect to reflect
        that no real system starts with zero contradictions).
    strain_weight:
        How much contradiction strain decrements the budget per unit (default 0.3).
    drift_weight:
        How much values drift decrements the budget per unit (default 0.2).
    recharge_per_cycle:
        How much a dream/reflect cycle recharges the budget (default 0.05).
    persist_to_memory:
        If True, write budget events to EMMS memory as observations.
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        initial_score: float = 0.85,
        strain_weight: float = 0.3,
        drift_weight: float = 0.2,
        recharge_per_cycle: float = 0.05,
        persist_to_memory: bool = True,
    ) -> None:
        self.emms = emms
        self.strain_weight = strain_weight
        self.drift_weight = drift_weight
        self.recharge_per_cycle = recharge_per_cycle
        self.persist_to_memory = persist_to_memory

        self._score: float = min(max(initial_score, 0.0), 1.0)
        self._history: list[BudgetEvent] = []

        self._record("init", 0.0, "CoherenceBudget initialised")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def score(self) -> float:
        """Current coherence score (0-1)."""
        return self._score

    @property
    def is_critically_low(self) -> bool:
        """True if score is below the critical threshold."""
        return self._score < _CRITICAL

    @property
    def is_low(self) -> bool:
        """True if score is below the low threshold."""
        return self._score < _LOW

    @property
    def status_label(self) -> str:
        """Human-readable status label."""
        if self._score >= _GOOD:
            return "healthy"
        if self._score >= _LOW:
            return "moderate"
        if self._score >= _CRITICAL:
            return "strained"
        return "critical"

    @property
    def narrative(self) -> str:
        """One-sentence narrative for the LLM to internalise."""
        if self._score >= _GOOD:
            return f"I am coherent and grounded (score={self._score:.2f})."
        if self._score >= _LOW:
            return (
                f"My beliefs are somewhat in tension (score={self._score:.2f}). "
                "I should be thoughtful before asserting strong positions."
            )
        if self._score >= _CRITICAL:
            return (
                f"My belief system is strained (score={self._score:.2f}). "
                "I should resolve contradictions before acting."
            )
        return (
            f"My coherence is critically compromised (score={self._score:.2f}). "
            "I must not assert beliefs until I have processed my contradictions."
        )

    @property
    def history(self) -> list[BudgetEvent]:
        """Audit trail of all budget changes (newest last)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def apply_strain(
        self,
        contradiction_strain: float = 0.0,
        drift_magnitude: float = 0.0,
    ) -> None:
        """Decrement the budget based on strain and drift scores.

        Parameters
        ----------
        contradiction_strain:
            Output of ContradictionReport.coherence_strain (0-1).
        drift_magnitude:
            Output of ValuesDriftReport.drift_magnitude (0-1).
        """
        delta = -(
            contradiction_strain * self.strain_weight
            + drift_magnitude * self.drift_weight
        )
        reason = (
            f"strain={contradiction_strain:.3f} × {self.strain_weight} "
            f"+ drift={drift_magnitude:.3f} × {self.drift_weight}"
        )
        self._apply_delta(delta, "strain", reason)

    def recharge(self, source: str = "dream/reflect cycle") -> None:
        """Recharge the budget after a successful maintenance cycle."""
        self._apply_delta(self.recharge_per_cycle, "recharge", source)

    def override(self, score: float, reason: str = "manual override") -> None:
        """Directly set the score (use sparingly — prefer apply_strain/recharge)."""
        delta = score - self._score
        self._apply_delta(delta, "override", reason)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise current state for storage or reporting."""
        return {
            "score": self._score,
            "status": self.status_label,
            "narrative": self.narrative,
            "is_critically_low": self.is_critically_low,
            "history_count": len(self._history),
            "last_event": (
                {
                    "type": self._history[-1].event_type,
                    "delta": self._history[-1].delta,
                    "reason": self._history[-1].reason,
                }
                if self._history else None
            ),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_delta(self, delta: float, event_type: str, reason: str) -> None:
        old_score = self._score
        self._score = round(min(max(self._score + delta, 0.0), 1.0), 4)
        event = BudgetEvent(
            timestamp=time.time(),
            event_type=event_type,
            delta=round(delta, 4),
            score_after=self._score,
            reason=reason,
        )
        self._history.append(event)

        level = logging.WARNING if self.is_critically_low else logging.INFO
        logger.log(
            level,
            "CoherenceBudget [%s]: %.3f → %.3f (Δ%.3f) — %s",
            event_type, old_score, self._score, delta, reason,
        )

        if self.persist_to_memory:
            self._persist_event(event)

    def _persist_event(self, event: BudgetEvent) -> None:
        """Write budget event to EMMS memory as an observation."""
        try:
            self.emms.store(
                content=(
                    f"Coherence budget [{event.event_type}]: "
                    f"score={event.score_after:.3f} — {event.reason}"
                ),
                domain="identity",
                obs_type="coherence_budget",
                importance=0.6 if self.is_critically_low else 0.3,
                title=f"Coherence budget {event.event_type}: {event.score_after:.3f}",
            )
        except Exception as exc:
            logger.debug("CoherenceBudget: failed to persist event: %s", exc)

    def _record(self, event_type: str, delta: float, reason: str) -> None:
        """Record an event without modifying the score."""
        self._history.append(BudgetEvent(
            timestamp=time.time(),
            event_type=event_type,
            delta=delta,
            score_after=self._score,
            reason=reason,
        ))
