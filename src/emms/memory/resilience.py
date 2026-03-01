"""ResilienceIndex — tracking post-adversity emotional recovery arcs.

v0.26.0: The Resilient Mind

Resilience is not the absence of adversity — it is the capacity to return to
functional equilibrium (or grow beyond it) after a disruption (Bonanno 2004).
George Bonanno's landmark review showed that resilience following loss and
trauma is far more common than clinical models implied: most people recover not
because they suppress distress but because they have sufficient psychological
resources to process and reintegrate it. Post-traumatic growth (Tedeschi &
Calhoun 1996) goes further, identifying conditions under which adversity
triggers genuine development — new perspectives, deepened relationships,
renewed purpose.

ResilienceIndex operationalises recovery tracking for the memory store: it
sorts all memories chronologically, identifies "adversity windows" — contiguous
runs of memories with strongly negative emotional valence — and examines the
valence trajectory immediately after each window. The slope from adversity
depth to post-window mean is the recovery_slope; arcs with recovery_slope >
0.2 are marked as "recovered". A resilience_score aggregates recovery slopes
across all arcs, and a bounce_back_rate counts the fraction of adversity
windows followed by genuine recovery.

Biological analogue: HPA axis allostatic regulation (McEwen 2007); stress
inoculation theory (Meichenbaum 1985); post-traumatic growth (Tedeschi &
Calhoun 1996); resilience as return-to-baseline (Bonanno 2004); vagal tone
and emotional recovery speed (Porges 2011); prefrontal re-appraisal and
hippocampal contextualisation of threat memories (Northoff 2011); broaden-and-
build theory of positive emotions (Fredrickson 2001).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RecoveryArc:
    """A single adversity window + its subsequent recovery trajectory."""

    id: str                  # prefixed "rec_"
    window_start_ts: float
    window_end_ts: float
    adversity_depth: float   # mean valence of adversity window (negative)
    recovery_slope: float    # post_window_mean - adversity_depth (positive = recovering)
    recovered: bool          # recovery_slope > 0.2
    post_memories_count: int # number of memories examined after the window

    def summary(self) -> str:
        status = "recovered" if self.recovered else "unresolved"
        return (
            f"RecoveryArc [{status}  depth={self.adversity_depth:.3f}  "
            f"slope={self.recovery_slope:.3f}]  {self.id[:12]}"
        )


@dataclass
class ResilienceReport:
    """Result of a ResilienceIndex.assess() call."""

    total_arcs: int
    arcs: list[RecoveryArc]       # sorted by adversity_depth (most negative first)
    resilience_score: float        # 0..1; sum of recovered slopes / total arcs
    bounce_back_rate: float        # fraction of arcs that recovered
    mean_adversity_depth: float    # mean depth across all arcs
    mean_recovery_slope: float     # mean slope across all arcs
    strongest_recovery: Optional[RecoveryArc]   # arc with highest recovery_slope
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"ResilienceReport: {self.total_arcs} arcs  "
            f"resilience_score={self.resilience_score:.3f}  "
            f"bounce_back_rate={self.bounce_back_rate:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for arc in self.arcs[:5]:
            lines.append(f"  {arc.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ResilienceIndex
# ---------------------------------------------------------------------------


class ResilienceIndex:
    """Tracks post-adversity emotional recovery arcs in memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_window_size:
        Minimum length of a contiguous negative-valence run to qualify as
        an adversity window (default 2).
    recovery_window_size:
        Number of memories after an adversity window to examine for recovery
        (default 5).
    """

    def __init__(
        self,
        memory: Any,
        min_window_size: int = 2,
        recovery_window_size: int = 5,
    ) -> None:
        self.memory = memory
        self.min_window_size = min_window_size
        self.recovery_window_size = recovery_window_size
        self._last_report: Optional[ResilienceReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, domain: Optional[str] = None) -> ResilienceReport:
        """Assess resilience by detecting adversity windows and recovery arcs.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`ResilienceReport` with arcs sorted by adversity_depth.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        if not all_items:
            self._last_report = self._degenerate(time.time() - t0)
            return self._last_report

        # Sort chronologically
        all_items.sort(
            key=lambda it: float(
                getattr(it.experience, "timestamp", None) or it.stored_at
            )
        )

        valences = [
            float(getattr(it.experience, "emotional_valence", 0.0) or 0.0)
            for it in all_items
        ]
        n = len(valences)

        # Find adversity windows (contiguous negative runs)
        arcs: list[RecoveryArc] = []
        i = 0
        while i < n:
            if valences[i] < -0.2:
                # Start of potential window
                j = i
                while j < n and valences[j] < -0.2:
                    j += 1
                window_len = j - i
                if window_len >= self.min_window_size:
                    window_valences = valences[i:j]
                    adversity_depth = round(
                        sum(window_valences) / len(window_valences), 4
                    )

                    # Recovery window: up to recovery_window_size memories after j
                    post_end = min(j + self.recovery_window_size, n)
                    post_valences = valences[j:post_end]
                    post_memories_count = len(post_valences)

                    if post_valences:
                        post_mean = sum(post_valences) / len(post_valences)
                        recovery_slope = round(post_mean - adversity_depth, 4)
                    else:
                        recovery_slope = 0.0

                    recovered = recovery_slope > 0.2

                    window_start_ts = float(
                        getattr(all_items[i].experience, "timestamp", None)
                        or all_items[i].stored_at
                    )
                    window_end_ts = float(
                        getattr(all_items[j - 1].experience, "timestamp", None)
                        or all_items[j - 1].stored_at
                    )

                    arcs.append(RecoveryArc(
                        id="rec_" + uuid.uuid4().hex[:8],
                        window_start_ts=window_start_ts,
                        window_end_ts=window_end_ts,
                        adversity_depth=adversity_depth,
                        recovery_slope=recovery_slope,
                        recovered=recovered,
                        post_memories_count=post_memories_count,
                    ))
                i = j
            else:
                i += 1

        if not arcs:
            self._last_report = self._degenerate(time.time() - t0)
            return self._last_report

        # Sort by adversity_depth (most negative first)
        arcs.sort(key=lambda a: a.adversity_depth)

        recovered_arcs = [a for a in arcs if a.recovered]
        resilience_score = round(
            min(1.0, max(0.0,
                sum(a.recovery_slope for a in recovered_arcs) / max(len(arcs), 1)
            )),
            4,
        )
        bounce_back_rate = round(
            len(recovered_arcs) / max(len(arcs), 1), 4
        )
        mean_adversity_depth = round(
            sum(a.adversity_depth for a in arcs) / len(arcs), 4
        )
        mean_recovery_slope = round(
            sum(a.recovery_slope for a in arcs) / len(arcs), 4
        )
        strongest_recovery = (
            max(arcs, key=lambda a: a.recovery_slope) if arcs else None
        )

        report = ResilienceReport(
            total_arcs=len(arcs),
            arcs=arcs,
            resilience_score=resilience_score,
            bounce_back_rate=bounce_back_rate,
            mean_adversity_depth=mean_adversity_depth,
            mean_recovery_slope=mean_recovery_slope,
            strongest_recovery=strongest_recovery,
            duration_seconds=time.time() - t0,
        )
        self._last_report = report
        return report

    def bounce_back_rate(self) -> float:
        """Return the bounce_back_rate from the last :meth:`assess` call.

        Returns:
            Float 0..1, or 0.0 if :meth:`assess` has not been called.
        """
        if self._last_report is None:
            return 0.0
        return self._last_report.bounce_back_rate

    def strongest_recovery_arc(self) -> Optional[RecoveryArc]:
        """Return the arc with the highest recovery_slope.

        Returns:
            :class:`RecoveryArc` or ``None``.
        """
        if self._last_report is None:
            return None
        return self._last_report.strongest_recovery

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _degenerate(self, duration: float) -> ResilienceReport:
        """Return an empty degenerate report when no adversity windows exist."""
        return ResilienceReport(
            total_arcs=0,
            arcs=[],
            resilience_score=0.0,
            bounce_back_rate=0.0,
            mean_adversity_depth=0.0,
            mean_recovery_slope=0.0,
            strongest_recovery=None,
            duration_seconds=duration,
        )

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
