"""MemoryDecay — Ebbinghaus forgetting curve applied to memory strength.

v0.16.0: The Curious Mind

Not all memories are accessed equally. Some are recalled constantly and stay
vivid; others are never revisited and fade. The Ebbinghaus forgetting curve
models this mathematically: retention R = e^{-t/S} where t is elapsed time
since last access and S is the memory's stability — a measure of how well-
consolidated it is.

MemoryDecay computes and optionally applies this curve to all memories,
updating ``memory_strength`` proportionally to current retention. Stability
increases with each retrieval (``retrieval_boost`` per access), modelling
the spacing effect: repeated retrieval makes memories more durable.

Two modes:
    ``decay()``        — compute only; returns a DecayReport without any
                         modifications (safe to call at any time)
    ``apply_decay()``  — compute and apply; updates ``memory_strength``
                         proportionally; optionally prunes items below
                         ``prune_threshold``

Biological analogue: Ebbinghaus forgetting curve (1885) — the exponential
decay of memory retention over time without rehearsal; the spacing effect
(Cepeda et al. 2006) — distributed practice dramatically slows forgetting.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DecayRecord:
    """Decay computation result for a single memory."""

    memory_id: str
    domain: str
    old_strength: float
    new_strength: float       # 0 if pruned
    retention: float          # R = e^{-t/S}
    stability: float          # S = base_stability + retrieval_boost * access_count
    days_since_access: float
    pruned: bool

    def summary(self) -> str:
        action = "PRUNED" if self.pruned else f"{self.old_strength:.3f}→{self.new_strength:.3f}"
        return (
            f"  [{self.memory_id[:12]}] {action}  "
            f"R={self.retention:.3f}  S={self.stability:.1f}d  "
            f"age={self.days_since_access:.1f}d  [{self.domain}]"
        )


@dataclass
class DecayReport:
    """Result of a MemoryDecay decay() or apply_decay() run."""

    total_processed: int
    decayed: int              # Memories whose strength changed
    pruned: int               # Memories removed (apply_decay only)
    mean_retention: float     # Average retention across all processed
    applied: bool             # True if changes were written to memory
    duration_seconds: float
    records: list[DecayRecord] = field(default_factory=list)  # Top-decayed items

    def summary(self) -> str:
        action = "applied" if self.applied else "simulated"
        lines = [
            f"DecayReport ({action}): {self.total_processed} processed, "
            f"{self.decayed} decayed, {self.pruned} pruned, "
            f"mean_retention={self.mean_retention:.3f} in {self.duration_seconds:.2f}s",
        ]
        for r in self.records[:5]:
            lines.append(r.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryDecay
# ---------------------------------------------------------------------------

class MemoryDecay:
    """Computes and optionally applies Ebbinghaus forgetting to memory strength.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    base_stability:
        Base stability in days for a memory with zero retrievals (default 7).
    retrieval_boost:
        Additional stability days granted per past access (default 2.0).
        A memory retrieved 5 times gets S = base_stability + 5 * retrieval_boost.
    prune_threshold:
        Memories with computed new_strength below this are pruned when
        ``apply_decay(prune=True)`` is used (default 0.05).
    max_records:
        Maximum per-item records to include in the report (default 20).
    """

    def __init__(
        self,
        memory: Any,
        base_stability: float = 7.0,
        retrieval_boost: float = 2.0,
        prune_threshold: float = 0.05,
        max_records: int = 20,
    ) -> None:
        self.memory = memory
        self.base_stability = base_stability
        self.retrieval_boost = retrieval_boost
        self.prune_threshold = prune_threshold
        self.max_records = max_records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decay(self, domain: Optional[str] = None) -> DecayReport:
        """Compute retention for all memories without applying changes.

        Args:
            domain: Restrict to one domain (``None`` = all).

        Returns:
            :class:`DecayReport` with per-memory retention values (read-only).
        """
        return self._run(domain=domain, apply=False, prune=False)

    def apply_decay(
        self,
        domain: Optional[str] = None,
        prune: bool = False,
    ) -> DecayReport:
        """Compute and apply forgetting curve to memory strengths.

        Args:
            domain: Restrict to one domain (``None`` = all).
            prune:  If ``True``, remove memories whose post-decay strength
                    falls below ``prune_threshold``.

        Returns:
            :class:`DecayReport` describing every change made.
        """
        return self._run(domain=domain, apply=True, prune=prune)

    def retention(self, item: Any) -> tuple[float, float]:
        """Compute (retention_R, stability_S) for a single memory item.

        Args:
            item: A :class:`MemoryItem`.

        Returns:
            Tuple ``(R, S)`` where R ∈ [0, 1] and S is stability in days.
        """
        return self._compute_retention(item)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        domain: Optional[str],
        apply: bool,
        prune: bool,
    ) -> DecayReport:
        t0 = time.time()
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        now = time.time()
        records: list[DecayRecord] = []
        total_retention = 0.0
        decayed_count = 0
        pruned_count = 0
        pruned_ids: set[str] = set()

        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            old_strength = item.memory_strength
            R, S = self._compute_retention(item)
            total_retention += R

            new_strength = old_strength * R
            days = (now - (item.last_accessed or item.stored_at)) / 86400

            do_prune = apply and prune and new_strength < self.prune_threshold

            if apply and not do_prune:
                item.memory_strength = max(0.0, new_strength)

            if do_prune:
                pruned_ids.add(item.id)
                pruned_count += 1
                new_strength = 0.0

            changed = abs(new_strength - old_strength) > 1e-6
            if changed:
                decayed_count += 1

            records.append(DecayRecord(
                memory_id=item.id,
                domain=dom,
                old_strength=old_strength,
                new_strength=round(new_strength, 5),
                retention=round(R, 5),
                stability=round(S, 2),
                days_since_access=round(days, 2),
                pruned=do_prune,
            ))

        # Prune items from tiers
        if pruned_ids:
            for tier in (self.memory.working, self.memory.short_term):
                for item in list(tier):
                    if item.id in pruned_ids:
                        try:
                            tier.remove(item)
                        except ValueError:
                            pass
            for tier in (self.memory.long_term, self.memory.semantic):
                for mid in list(pruned_ids & set(tier)):
                    del tier[mid]

        mean_retention = total_retention / max(len(items), 1)

        # Sort records by most-decayed first for report
        records.sort(key=lambda r: r.old_strength - r.new_strength, reverse=True)

        return DecayReport(
            total_processed=len(items),
            decayed=decayed_count,
            pruned=pruned_count,
            mean_retention=round(mean_retention, 4),
            applied=apply,
            duration_seconds=time.time() - t0,
            records=records[: self.max_records],
        )

    def _compute_retention(self, item: Any) -> tuple[float, float]:
        """Compute (R, S) for one memory item."""
        now = time.time()
        last_access = getattr(item, "last_accessed", None) or item.stored_at
        t_days = max(0.0, (now - last_access) / 86400.0)

        access_count = getattr(item, "access_count", 0) or 0

        # Importance-weighted stability: high-importance memories decay slower
        # importance=0 → 1x, importance=0.5 → 2.5x, importance=1.0 → 4x
        _raw_imp = getattr(getattr(item, "experience", None), "importance", None)
        importance = _raw_imp if _raw_imp is not None else 0.5
        importance_factor = 1.0 + importance * 3.0
        stability = (self.base_stability + self.retrieval_boost * access_count) * importance_factor

        # R = e^{-t/S}
        R = math.exp(-t_days / max(stability, 0.1))
        return R, stability

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
