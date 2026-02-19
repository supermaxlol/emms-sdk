"""ReconsolidationEngine — biological memory reconsolidation for EMMS.

When a memory is *retrieved*, it briefly enters a labile (modifiable) state
before re-stabilizing.  This engine models that process:

  * Reinforce:    recall in a confirming context → strength +δ
  * Weaken:       recall in a contradicting context → strength −δ
  * Valence drift: emotional context nudges the stored valence toward the
                   recall context's valence by a small step.
  * Diminishing returns: each recall has less effect than the last
                   (logarithmic attenuation via access_count).

References:
  - Nader, K. et al. (2000). Fear memories require protein synthesis in the
    amygdala for reconsolidation after retrieval. Nature, 406, 722–726.
  - Sara, S. J. (2000). Retrieval and reconsolidation: toward a neurobiology
    of remembering. Learning & Memory, 7(2), 73–84.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from emms.core.models import MemoryItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReconsolidationResult:
    """Result of a single reconsolidation event."""
    memory_id: str
    reconsolidation_type: str      # "reinforce" | "weaken" | "valence_drift" | "none"
    old_strength: float
    new_strength: float
    old_valence: float
    new_valence: float
    delta_strength: float          # new − old (signed)
    delta_valence: float           # new − old (signed)
    recall_count_after: int        # access_count after this event
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"[{self.reconsolidation_type}] {self.memory_id} "
            f"strength {self.old_strength:.3f}→{self.new_strength:.3f} "
            f"valence {self.old_valence:+.3f}→{self.new_valence:+.3f}"
        )


@dataclass
class ReconsolidationReport:
    """Summary of a batch reconsolidation run."""
    total_items: int
    reinforced: int
    weakened: int
    valence_drifted: int
    unchanged: int
    mean_delta_strength: float
    mean_delta_valence: float
    results: list[ReconsolidationResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Reconsolidation report — {self.total_items} items",
            f"  reinforced={self.reinforced}  weakened={self.weakened}  "
            f"valence_drifted={self.valence_drifted}  unchanged={self.unchanged}",
            f"  Δstrength mean={self.mean_delta_strength:+.4f}  "
            f"Δvalence mean={self.mean_delta_valence:+.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReconsolidationEngine:
    """Models biological memory reconsolidation.

    Parameters
    ----------
    reinforce_rate : float
        Base strength increment per confirming recall (default 0.08).
    weaken_rate : float
        Base strength decrement per contradicting recall (default 0.05).
    valence_drift_rate : float
        How quickly stored valence drifts toward context valence (default 0.06).
    min_strength : float
        Floor — strength never drops below this (default 0.01).
    max_strength : float
        Ceiling — strength never exceeds this (default 2.0).
    attenuation_base : float
        Logarithmic attenuation: effect × 1 / log(1 + access_count * base).
        Higher values mean faster diminishing returns (default 1.5).
    """

    def __init__(
        self,
        reinforce_rate: float = 0.08,
        weaken_rate: float = 0.05,
        valence_drift_rate: float = 0.06,
        min_strength: float = 0.01,
        max_strength: float = 2.0,
        attenuation_base: float = 1.5,
    ) -> None:
        import math
        self._math = math
        self.reinforce_rate = reinforce_rate
        self.weaken_rate = weaken_rate
        self.valence_drift_rate = valence_drift_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.attenuation_base = attenuation_base

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reconsolidate(
        self,
        item: MemoryItem,
        context_valence: float | None = None,
        reinforce: bool = True,
    ) -> ReconsolidationResult:
        """Reconsolidate a single memory item after recall.

        Parameters
        ----------
        item : MemoryItem
            The recalled memory.  Its fields are mutated in-place.
        context_valence : float | None
            Emotional valence of the recall context (-1…+1).  When provided,
            stored valence drifts toward this value.
        reinforce : bool
            True → reinforce (confirming recall); False → weaken.

        Returns
        -------
        ReconsolidationResult with before/after values.
        """
        old_strength = item.memory_strength
        old_valence = item.experience.emotional_valence
        recon_type = "none"
        import math

        # Diminishing-returns attenuation
        n = max(1, item.access_count)
        attenuation = 1.0 / math.log(1.0 + n * self.attenuation_base)

        # --- strength change ---
        if reinforce:
            delta_s = self.reinforce_rate * attenuation
            item.memory_strength = min(
                self.max_strength, item.memory_strength + delta_s
            )
            recon_type = "reinforce"
        else:
            delta_s = -self.weaken_rate * attenuation
            item.memory_strength = max(
                self.min_strength, item.memory_strength + delta_s
            )
            recon_type = "weaken"

        # --- valence drift ---
        delta_v = 0.0
        if context_valence is not None:
            # clamp to valid range first
            ctx_v = max(-1.0, min(1.0, context_valence))
            stored_v = item.experience.emotional_valence
            drift = (ctx_v - stored_v) * self.valence_drift_rate * attenuation
            new_v = max(-1.0, min(1.0, stored_v + drift))
            item.experience.emotional_valence = new_v
            delta_v = new_v - stored_v
            if abs(delta_v) > 1e-6:
                recon_type = recon_type + "+valence_drift" if recon_type != "none" else "valence_drift"

        # update access tracking
        item.access_count += 1
        item.last_accessed = time.time()

        return ReconsolidationResult(
            memory_id=item.id,
            reconsolidation_type=recon_type,
            old_strength=old_strength,
            new_strength=item.memory_strength,
            old_valence=old_valence,
            new_valence=item.experience.emotional_valence,
            delta_strength=item.memory_strength - old_strength,
            delta_valence=delta_v,
            recall_count_after=item.access_count,
        )

    def batch_reconsolidate(
        self,
        items: list[MemoryItem],
        context_valence: float | None = None,
        reinforce: bool = True,
    ) -> ReconsolidationReport:
        """Reconsolidate a batch of items.

        Useful after a retrieval round — pass all retrieved items to
        update their reconsolidation state.
        """
        results = [
            self.reconsolidate(item, context_valence=context_valence, reinforce=reinforce)
            for item in items
        ]
        reinforced = sum(1 for r in results if "reinforce" in r.reconsolidation_type)
        weakened = sum(1 for r in results if "weaken" in r.reconsolidation_type)
        valence_drifted = sum(1 for r in results if "valence_drift" in r.reconsolidation_type)
        unchanged = sum(1 for r in results if r.reconsolidation_type == "none")

        mean_ds = (
            sum(r.delta_strength for r in results) / len(results) if results else 0.0
        )
        mean_dv = (
            sum(r.delta_valence for r in results) / len(results) if results else 0.0
        )

        return ReconsolidationReport(
            total_items=len(items),
            reinforced=reinforced,
            weakened=weakened,
            valence_drifted=valence_drifted,
            unchanged=unchanged,
            mean_delta_strength=mean_ds,
            mean_delta_valence=mean_dv,
            results=results,
        )

    def decay_unrecalled(
        self,
        items: list[MemoryItem],
        decay_factor: float = 0.02,
        min_age_seconds: float = 3600.0,
    ) -> ReconsolidationReport:
        """Passively weaken items that have *not* been recalled recently.

        Simulates memory trace decay for items that are not being consolidated
        through use.  Only items older than `min_age_seconds` are decayed.

        Parameters
        ----------
        items : list[MemoryItem]
        decay_factor : float
            Absolute strength reduction per call (default 0.02).
        min_age_seconds : float
            Items more recent than this are not decayed (default 3600 = 1 h).
        """
        now = time.time()
        results: list[ReconsolidationResult] = []
        for item in items:
            age = now - item.last_accessed
            if age < min_age_seconds:
                results.append(ReconsolidationResult(
                    memory_id=item.id,
                    reconsolidation_type="none",
                    old_strength=item.memory_strength,
                    new_strength=item.memory_strength,
                    old_valence=item.experience.emotional_valence,
                    new_valence=item.experience.emotional_valence,
                    delta_strength=0.0,
                    delta_valence=0.0,
                    recall_count_after=item.access_count,
                ))
                continue

            old_s = item.memory_strength
            item.memory_strength = max(self.min_strength, item.memory_strength - decay_factor)
            results.append(ReconsolidationResult(
                memory_id=item.id,
                reconsolidation_type="decay",
                old_strength=old_s,
                new_strength=item.memory_strength,
                old_valence=item.experience.emotional_valence,
                new_valence=item.experience.emotional_valence,
                delta_strength=item.memory_strength - old_s,
                delta_valence=0.0,
                recall_count_after=item.access_count,
            ))

        decayed = sum(1 for r in results if r.reconsolidation_type == "decay")
        mean_ds = sum(r.delta_strength for r in results) / len(results) if results else 0.0
        return ReconsolidationReport(
            total_items=len(items),
            reinforced=0,
            weakened=decayed,
            valence_drifted=0,
            unchanged=len(items) - decayed,
            mean_delta_strength=mean_ds,
            mean_delta_valence=0.0,
            results=results,
        )
