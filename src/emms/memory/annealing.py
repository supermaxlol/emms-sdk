"""MemoryAnnealer — temporal memory annealing based on session gaps.

When a long time passes between sessions, the memory landscape should change:
- Weak memories decay faster (they weren't important enough to survive)
- Emotional valence stabilises toward neutral (time changes perspective)
- High-importance memories that survive are mildly strengthened

This mirrors human experience: a memory from 3 months ago feels different
from one recalled yesterday. The emotional charge softens. The important
ones feel more permanent. The unimportant ones fade.

The "temperature" metaphor from simulated annealing:
  temperature(gap) = 1 / (1 + (gap / half_life))

High temperature (short gap, recent session): high plasticity, many changes.
Low temperature (long gap, stale session): memories have settled into a stable
configuration; only very weak ones are further decayed.

Usage::

    annealer = MemoryAnnealer(memory)
    result = annealer.anneal(last_session_at=three_days_ago)
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnnealingResult:
    """Summary of a memory annealing pass."""
    total_items: int
    gap_seconds: float
    effective_temperature: float   # 0 = cold/stable, 1 = hot/plastic
    accelerated_decay: int         # items weakened due to gap
    emotionally_stabilized: int    # items whose valence moved toward neutral
    strengthened: int              # important survivors strengthened
    duration_seconds: float

    def summary(self) -> str:
        gap_h = self.gap_seconds / 3600
        return (
            f"Memory annealing — gap={gap_h:.1f}h  "
            f"temp={self.effective_temperature:.3f}\n"
            f"  Processed: {self.total_items}  "
            f"Decayed: {self.accelerated_decay}  "
            f"Stabilised: {self.emotionally_stabilized}  "
            f"Strengthened: {self.strengthened}"
        )


# ---------------------------------------------------------------------------
# MemoryAnnealer
# ---------------------------------------------------------------------------

class MemoryAnnealer:
    """Anneal the memory landscape after a session gap.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing memory.
    half_life_gap : float
        Gap in seconds at which temperature = 0.5 (default 3 days = 259200s).
    decay_rate : float
        Base decay per item per annealing pass (default 0.03).
        Actual decay = decay_rate * temperature * (1 - importance).
    emotional_stabilization_rate : float
        Rate at which valence moves toward 0 (default 0.08).
        Actual drift = rate * temperature * |current_valence|.
    strengthen_rate : float
        Strength increment for important survivors (default 0.05).
        Only applied to items with importance >= strengthen_importance_threshold.
    strengthen_importance_threshold : float
        Minimum importance to receive strengthening (default 0.7).
    min_strength : float
        Floor on memory strength (default 0.01).
    max_strength : float
        Ceiling on memory strength (default 2.0).
    """

    def __init__(
        self,
        memory: Any,
        half_life_gap: float = 259_200.0,   # 3 days
        decay_rate: float = 0.03,
        emotional_stabilization_rate: float = 0.08,
        strengthen_rate: float = 0.05,
        strengthen_importance_threshold: float = 0.7,
        min_strength: float = 0.01,
        max_strength: float = 2.0,
    ) -> None:
        self.memory = memory
        self.half_life_gap = half_life_gap
        self.decay_rate = decay_rate
        self.emotional_stabilization_rate = emotional_stabilization_rate
        self.strengthen_rate = strengthen_rate
        self.strengthen_importance_threshold = strengthen_importance_threshold
        self.min_strength = min_strength
        self.max_strength = max_strength

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def anneal(
        self,
        last_session_at: float | None = None,
    ) -> AnnealingResult:
        """Run a full annealing pass.

        Parameters
        ----------
        last_session_at : float | None
            Unix timestamp of the last session end.  If None, uses
            ``time.time() - half_life_gap`` (assumes one half-life has passed).

        Returns
        -------
        AnnealingResult
        """
        t0 = time.time()
        now = t0

        if last_session_at is None:
            gap = self.half_life_gap
        else:
            gap = max(0.0, now - last_session_at)

        temperature = self._temperature(gap)
        all_items = self._collect_all()

        accelerated_decay = 0
        emotionally_stabilized = 0
        strengthened = 0

        for item in all_items:
            exp = item.experience
            importance = float(exp.importance)
            old_strength = item.memory_strength
            old_valence = exp.emotional_valence

            # --- Accelerated decay for weak memories ---
            # Effect is larger when temperature is high AND importance is low
            decay = self.decay_rate * temperature * max(0.0, 1.0 - importance)
            if decay > 0.001:
                item.memory_strength = max(
                    self.min_strength,
                    item.memory_strength - decay,
                )
                if item.memory_strength < old_strength - 0.001:
                    accelerated_decay += 1

            # --- Emotional stabilization (valence drifts toward 0) ---
            stab_rate = self.emotional_stabilization_rate * temperature
            if abs(old_valence) > 0.01 and stab_rate > 0.001:
                drift = -old_valence * stab_rate
                new_v = max(-1.0, min(1.0, old_valence + drift))
                exp.emotional_valence = new_v
                if abs(new_v - old_valence) > 0.001:
                    emotionally_stabilized += 1

            # --- Strengthen important survivors ---
            if importance >= self.strengthen_importance_threshold:
                item.memory_strength = min(
                    self.max_strength,
                    item.memory_strength + self.strengthen_rate * (1.0 - temperature),
                )
                if item.memory_strength > old_strength + 0.001:
                    strengthened += 1

        return AnnealingResult(
            total_items=len(all_items),
            gap_seconds=gap,
            effective_temperature=temperature,
            accelerated_decay=accelerated_decay,
            emotionally_stabilized=emotionally_stabilized,
            strengthened=strengthened,
            duration_seconds=time.time() - t0,
        )

    def temperature(self, gap_seconds: float) -> float:
        """Return the effective temperature for a given gap (public accessor)."""
        return self._temperature(gap_seconds)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _temperature(self, gap_seconds: float) -> float:
        """High-temperature = recent gap (plastic); low-temperature = old gap (stable).

            T(g) = 1 / (1 + g / half_life)
        """
        if self.half_life_gap <= 0:
            return 1.0
        return 1.0 / (1.0 + gap_seconds / self.half_life_gap)

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
