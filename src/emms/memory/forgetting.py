"""MotivatedForgetting — biologically-inspired active memory suppression.

v0.14.0: The Temporal Mind

Not all forgetting is failure — some is intentional. The brain uses active
suppression (think/no-think paradigm; Anderson & Green 2001) to inhibit
memories associated with negative emotion or goal-conflict. Directed forgetting
(Bjork 1972) demonstrates that a simple instruction to forget changes memory
accessibility measurably.

MotivatedForgetting provides programmatic control over memory suppression:
strength-based decay, domain-level pruning, confidence-threshold eviction,
and contradiction resolution by weakening the less-supported side.

Biological analogue: prefrontal-hippocampal inhibition of unwanted memories
via right DLPFC engagement; memory suppression as a training signal that
makes remaining memories relatively stronger and more accessible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ForgettingResult:
    """Record of a single suppression or pruning action."""

    memory_id: str
    reason: str                 # human-readable explanation
    old_strength: float         # memory_strength before action
    new_strength: float         # memory_strength after action (0 = pruned)
    pruned: bool                # True if item was removed from memory

    def summary(self) -> str:
        action = "PRUNED" if self.pruned else f"weakened {self.old_strength:.3f}→{self.new_strength:.3f}"
        return f"  [{self.memory_id[:12]}] {action} — {self.reason}"


@dataclass
class ForgettingReport:
    """Result of a MotivatedForgetting operation."""

    total_targeted: int         # number of memories considered
    suppressed: int             # memories weakened but not removed
    pruned: int                 # memories removed entirely
    duration_seconds: float
    results: list[ForgettingResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ForgettingReport: {self.total_targeted} targeted, "
            f"{self.suppressed} suppressed, {self.pruned} pruned "
            f"in {self.duration_seconds:.3f}s",
        ]
        for r in self.results[:8]:
            lines.append(r.summary())
        if len(self.results) > 8:
            lines.append(f"  ... and {len(self.results) - 8} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MotivatedForgetting
# ---------------------------------------------------------------------------

class MotivatedForgetting:
    """Provides active, goal-directed memory suppression.

    Operates on the :class:`HierarchicalMemory` tiers directly — memories
    below a pruning threshold are evicted; others are weakened by a
    configurable suppression rate.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    suppression_rate:
        Fraction by which ``memory_strength`` is multiplied during suppression
        (default 0.4, meaning strength drops to 40 % of current value).
    prune_threshold:
        Memories with ``memory_strength`` below this value after suppression
        are removed from the tier entirely (default 0.05).
    """

    def __init__(
        self,
        memory: Any,
        suppression_rate: float = 0.4,
        prune_threshold: float = 0.05,
    ) -> None:
        self.memory = memory
        self.suppression_rate = suppression_rate
        self.prune_threshold = prune_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suppress(self, memory_id: str) -> Optional[ForgettingResult]:
        """Suppress a specific memory by ID.

        Finds the memory across all tiers and reduces its strength by
        ``suppression_rate``.  If the resulting strength falls below
        ``prune_threshold``, the memory is pruned entirely.

        Args:
            memory_id: The ``id`` or ``experience.id`` of the target memory.

        Returns:
            A :class:`ForgettingResult`, or ``None`` if the memory was not found.
        """
        for item in self._collect_all():
            if item.id == memory_id or item.experience.id == memory_id:
                return self._suppress_item(item, self.suppression_rate, "targeted suppression")
        return None

    def forget_domain(
        self,
        domain: str,
        rate: Optional[float] = None,
    ) -> ForgettingReport:
        """Suppress all memories in a domain.

        Args:
            domain: Domain name to target.
            rate:   Suppression rate override (default: instance ``suppression_rate``).

        Returns:
            :class:`ForgettingReport` describing every action taken.
        """
        t0 = time.time()
        r = rate if rate is not None else self.suppression_rate
        targeted = 0
        results: list[ForgettingResult] = []

        for item in self._collect_all():
            if (getattr(item.experience, "domain", None) or "general") == domain:
                targeted += 1
                res = self._suppress_item(item, r, f"domain forget: {domain}")
                results.append(res)

        suppressed = sum(1 for r in results if not r.pruned)
        pruned = sum(1 for r in results if r.pruned)
        return ForgettingReport(
            total_targeted=targeted,
            suppressed=suppressed,
            pruned=pruned,
            duration_seconds=time.time() - t0,
            results=results,
        )

    def forget_below_confidence(
        self,
        threshold: float = 0.3,
        metacognition_engine: Optional[Any] = None,
    ) -> ForgettingReport:
        """Suppress memories whose epistemic confidence is below a threshold.

        If a ``metacognition_engine`` is provided, it is used to assess
        confidence precisely.  Otherwise, ``memory_strength`` is used as a
        proxy.

        Args:
            threshold:            Minimum confidence to keep (default 0.3).
            metacognition_engine: Optional :class:`MetacognitionEngine` for
                                  accurate confidence scoring.

        Returns:
            :class:`ForgettingReport` describing every action taken.
        """
        t0 = time.time()
        results: list[ForgettingResult] = []
        items = self._collect_all()

        if metacognition_engine is not None:
            # Map item.id → confidence
            assessments = {
                mc.memory_id: mc.confidence
                for mc in metacognition_engine.assess_all()
            }
        else:
            assessments = {}

        for item in items:
            if assessments:
                conf = assessments.get(item.id, item.memory_strength)
            else:
                conf = item.memory_strength

            if conf < threshold:
                res = self._suppress_item(
                    item,
                    self.suppression_rate,
                    f"low confidence ({conf:.3f} < {threshold})",
                )
                results.append(res)

        suppressed = sum(1 for r in results if not r.pruned)
        pruned = sum(1 for r in results if r.pruned)
        return ForgettingReport(
            total_targeted=len(results),
            suppressed=suppressed,
            pruned=pruned,
            duration_seconds=time.time() - t0,
            results=results,
        )

    def resolve_contradiction(self, weaker_id: str) -> Optional[ForgettingResult]:
        """Suppress the weaker side of a contradiction.

        Finds the memory by ID and applies a single suppression step.  The
        caller is responsible for identifying which of a contradictory pair is
        weaker (e.g. by using :class:`MetacognitionEngine.find_contradictions`).

        Args:
            weaker_id: Memory ID of the weaker / less-trusted side to suppress.

        Returns:
            :class:`ForgettingResult`, or ``None`` if the memory was not found.
        """
        for item in self._collect_all():
            if item.id == weaker_id or item.experience.id == weaker_id:
                return self._suppress_item(
                    item,
                    self.suppression_rate,
                    "contradiction resolution: weaker side suppressed",
                )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _suppress_item(
        self,
        item: Any,
        rate: float,
        reason: str,
    ) -> ForgettingResult:
        """Apply suppression to one item; prune from tier if below threshold.

        Returns:
            :class:`ForgettingResult` describing the action.
        """
        old_strength = item.memory_strength
        new_strength = old_strength * (1.0 - rate)
        pruned = False

        if new_strength < self.prune_threshold:
            # Remove from whichever tier holds it
            self._prune_from_tiers(item.id)
            new_strength = 0.0
            pruned = True
        else:
            item.memory_strength = new_strength
            # Also decay consolidation score
            if hasattr(item, "consolidation_score"):
                item.consolidation_score = max(
                    0.0, item.consolidation_score * (1.0 - rate * 0.5)
                )

        return ForgettingResult(
            memory_id=item.id,
            reason=reason,
            old_strength=old_strength,
            new_strength=new_strength,
            pruned=pruned,
        )

    def _prune_from_tiers(self, item_id: str) -> None:
        """Remove an item from whichever memory tier contains it."""
        mem = self.memory
        # working and short_term are deques
        for tier in (mem.working, mem.short_term):
            for item in list(tier):
                if item.id == item_id:
                    try:
                        tier.remove(item)
                    except ValueError:
                        pass
                    return
        # long_term and semantic are dicts
        for tier in (mem.long_term, mem.semantic):
            if item_id in tier:
                del tier[item_id]
                return

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
