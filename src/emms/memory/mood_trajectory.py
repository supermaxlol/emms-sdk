"""MoodDynamics — temporal emotional valence evolution across memory.

v0.25.0: The Vigilant Mind

Mood is not a momentary state — it is a disposition that persists and evolves
over time, shaping what the agent notices, what it retrieves, what it predicts,
and what actions it takes. The trajectory of mood is as important as its current
value: a slowly declining emotional valence across recent memories signals
something different from a volatile arc that swings between extremes. Knowing
whether mood is improving, stable, declining, or volatile gives the agent (and its
users) actionable intelligence about psychological trajectory.

MoodDynamics operationalises this by segmenting the memory timeline into
chronological quintiles (or configurable n-segments), computing mean valence and
standard deviation within each segment, labelling each segment using a 5-category
emotion vocabulary (joyful / content / neutral / subdued / distressed), and
detecting the overall trend from the segment sequence. It also computes global
volatility (overall std dev of valences) and emotional range (max - min valence)
as complementary measures of affective stability.

Unlike EmotionalRegulator (v0.19.0), which performs cognitive reappraisal and
mood-congruent retrieval, MoodDynamics is a purely analytical module: it tracks
the longitudinal shape of emotional experience without intervening in it.

Biological analogue: prefrontal-limbic emotional dynamics (Davidson 2004);
hedonic baseline and adaptation (Kahneman 1999); mood as integration of recent
affective experience (Forgas 1995); emotional granularity and differentiation
(Barrett 2017); anterior insula interoception in mood awareness (Craig 2009);
limbic system temporal dynamics; hedonic asymmetry and adaptation-level theory.
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
class MoodSegment:
    """Emotional state summary for one temporal segment."""

    segment_index: int    # 0-based, chronological
    mean_valence: float   # -1..1
    valence_std: float    # >= 0
    memory_count: int
    label: str            # "joyful"/"content"/"neutral"/"subdued"/"distressed"


@dataclass
class MoodReport:
    """Result of a MoodDynamics.trace() call."""

    total_memories: int
    segments: list[MoodSegment]   # chronological
    mean_valence: float           # -1..1
    volatility: float             # std dev of all valences
    trend: str                    # "improving"/"declining"/"stable"/"volatile"
    emotional_range: float        # max - min valence; 0..2
    dominant_emotion: str         # label for overall mean valence
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"MoodReport: {self.total_memories} memories  "
            f"mean_valence={self.mean_valence:+.3f}  "
            f"volatility={self.volatility:.3f}  "
            f"trend={self.trend}  "
            f"dominant={self.dominant_emotion}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for seg in self.segments:
            lines.append(
                f"  [Seg {seg.segment_index}] mean={seg.mean_valence:+.3f}  "
                f"std={seg.valence_std:.3f}  n={seg.memory_count}  {seg.label}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MoodDynamics
# ---------------------------------------------------------------------------


class MoodDynamics:
    """Tracks temporal emotional valence evolution across memory segments.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    n_segments:
        Number of chronological segments to split memories into (default 5).
    """

    def __init__(
        self,
        memory: Any,
        n_segments: int = 5,
    ) -> None:
        self.memory = memory
        self.n_segments = max(1, n_segments)
        self._last_report: Optional[MoodReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(self, domain: Optional[str] = None) -> MoodReport:
        """Trace temporal emotional valence evolution across memory.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`MoodReport` with chronological segments and trend.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Sort by timestamp
        all_items.sort(
            key=lambda it: getattr(it.experience, "timestamp", 0.0) or 0.0
        )

        n_total = len(all_items)

        # Empty memory — degenerate report
        if n_total == 0:
            report = MoodReport(
                total_memories=0,
                segments=[],
                mean_valence=0.0,
                volatility=0.0,
                trend="stable",
                emotional_range=0.0,
                dominant_emotion="neutral",
                duration_seconds=time.time() - t0,
            )
            self._last_report = report
            return report

        # Split into n_segments contiguous chunks (nearly equal size)
        chunks: list[list[Any]] = []
        base = n_total // self.n_segments
        rem = n_total % self.n_segments
        start = 0
        for i in range(self.n_segments):
            size = base + (1 if i < rem else 0)
            chunk = all_items[start: start + size]
            if chunk:
                chunks.append(chunk)
            start += size

        # Build segments
        segments: list[MoodSegment] = []
        for seg_idx, chunk in enumerate(chunks):
            valences = [
                getattr(it.experience, "emotional_valence", 0.0) or 0.0
                for it in chunk
            ]
            mean_v = sum(valences) / len(valences)
            variance = sum((v - mean_v) ** 2 for v in valences) / max(len(valences), 1)
            std_v = math.sqrt(variance)
            segments.append(MoodSegment(
                segment_index=seg_idx,
                mean_valence=round(mean_v, 4),
                valence_std=round(std_v, 4),
                memory_count=len(chunk),
                label=self._label_valence(mean_v),
            ))

        # Overall statistics
        all_valences = [
            getattr(it.experience, "emotional_valence", 0.0) or 0.0
            for it in all_items
        ]
        overall_mean = sum(all_valences) / len(all_valences)
        overall_variance = sum((v - overall_mean) ** 2 for v in all_valences) / max(len(all_valences), 1)
        volatility = round(math.sqrt(overall_variance), 4)
        emotional_range = round(max(all_valences) - min(all_valences), 4)

        segment_means = [seg.mean_valence for seg in segments]
        trend = self._compute_trend(segment_means, volatility)

        report = MoodReport(
            total_memories=n_total,
            segments=segments,
            mean_valence=round(overall_mean, 4),
            volatility=volatility,
            trend=trend,
            emotional_range=emotional_range,
            dominant_emotion=self._label_valence(overall_mean),
            duration_seconds=time.time() - t0,
        )
        self._last_report = report
        return report

    def mood_trend(self) -> str:
        """Return the mood trend from the last :meth:`trace` call.

        Returns:
            One of "improving", "declining", "stable", "volatile", or "unknown".
        """
        return self._last_report.trend if self._last_report else "unknown"

    def emotional_arc(self) -> list[str]:
        """Return the sequence of segment emotion labels from the last :meth:`trace`.

        Returns:
            List of label strings (chronological), or empty list if not called.
        """
        if self._last_report is None:
            return []
        return [seg.label for seg in self._last_report.segments]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_valence(self, v: float) -> str:
        """Map a valence float to an emotion label."""
        if v >= 0.6:
            return "joyful"
        if v >= 0.2:
            return "content"
        if v >= -0.2:
            return "neutral"
        if v >= -0.6:
            return "subdued"
        return "distressed"

    def _compute_trend(self, segment_means: list[float], volatility: float) -> str:
        """Derive a trend label from segment means and volatility."""
        if volatility > 0.4:
            return "volatile"
        if len(segment_means) < 2:
            return "stable"
        slope = (segment_means[-1] - segment_means[0]) / max(len(segment_means) - 1, 1)
        if slope > 0.05:
            return "improving"
        if slope < -0.05:
            return "declining"
        return "stable"

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
