"""Memory timeline: chronological reconstruction with gap and density analysis.

Provides a structured view of how memories evolved over time, surfacing:

* **TimelineEvent** — a memory item positioned on the timeline with its tier,
  domain, importance, and access metadata.
* **TemporalGap** — a significant time interval between consecutive memories
  where no observations were recorded (potential context switch, sleep, etc.).
* **DensityBucket** — a histogram bucket showing how many memories were stored
  within a fixed time window.
* **TimelineResult** — the assembled timeline with events, gaps, density, and
  summary statistics.

Usage::

    from emms import EMMS, Experience
    from emms.analytics.timeline import MemoryTimeline

    agent = EMMS()
    # … store experiences …
    timeline = MemoryTimeline(agent.memory)
    result = timeline.build(domain="code", since=0.0)
    print(result.summary())
    print(result.export_markdown())
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single memory item positioned on the timeline.

    Attributes
    ----------
    memory_id : str
        Unique ID of the MemoryItem.
    experience_id : str
        ID of the underlying Experience.
    content : str
        Short excerpt (first 120 chars) of the experience content.
    domain : str
        Domain label.
    tier : str
        Memory tier name (working / short_term / long_term / semantic).
    importance : float
        Importance score [0, 1].
    memory_strength : float
        Decay-adjusted memory strength [0, 1].
    stored_at : float
        Unix timestamp when the memory was stored.
    access_count : int
        Number of times the memory has been accessed.
    title : str | None
        Optional title from the experience.
    """
    memory_id: str
    experience_id: str
    content: str
    domain: str
    tier: str
    importance: float
    memory_strength: float
    stored_at: float
    access_count: int
    title: str | None = None


@dataclass
class TemporalGap:
    """A significant time gap between two consecutive timeline events.

    Attributes
    ----------
    start_at : float
        End timestamp of the memory *before* the gap.
    end_at : float
        Start timestamp of the memory *after* the gap.
    duration_seconds : float
        Length of the gap in seconds.
    before_id : str
        memory_id of the last event before the gap.
    after_id : str
        memory_id of the first event after the gap.
    """
    start_at: float
    end_at: float
    duration_seconds: float
    before_id: str
    after_id: str

    @property
    def duration_human(self) -> str:
        """Human-readable duration string."""
        s = self.duration_seconds
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.1f}m"
        if s < 86400:
            return f"{s / 3600:.1f}h"
        return f"{s / 86400:.1f}d"


@dataclass
class DensityBucket:
    """Histogram bucket for memory storage density.

    Attributes
    ----------
    start_at : float
        Bucket start timestamp.
    end_at : float
        Bucket end timestamp.
    count : int
        Number of memories stored in this interval.
    domains : list[str]
        Unique domains observed in this bucket.
    avg_importance : float
        Mean importance of memories in the bucket.
    """
    start_at: float
    end_at: float
    count: int
    domains: list[str] = field(default_factory=list)
    avg_importance: float = 0.0

    @property
    def label(self) -> str:
        """ISO-style label for the bucket start time."""
        import datetime
        try:
            dt = datetime.datetime.utcfromtimestamp(self.start_at)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return f"t={self.start_at:.0f}"


@dataclass
class TimelineResult:
    """Complete timeline result.

    Attributes
    ----------
    events : list[TimelineEvent]
        All events sorted chronologically (oldest first).
    gaps : list[TemporalGap]
        Significant gaps detected between events.
    density : list[DensityBucket]
        Memory density histogram.
    total_memories : int
        Total number of memories on the timeline.
    earliest_at : float
        Timestamp of the oldest event (0.0 if empty).
    latest_at : float
        Timestamp of the newest event (0.0 if empty).
    span_seconds : float
        Total span from earliest to latest event.
    mean_importance : float
        Mean importance across all events.
    domain_counts : dict[str, int]
        How many memories per domain.
    gap_threshold_seconds : float
        The threshold used to classify gaps.
    bucket_size_seconds : float
        The bucket width used for density histogram.
    """
    events: list[TimelineEvent]
    gaps: list[TemporalGap]
    density: list[DensityBucket]
    total_memories: int
    earliest_at: float
    latest_at: float
    span_seconds: float
    mean_importance: float
    domain_counts: dict[str, int]
    gap_threshold_seconds: float
    bucket_size_seconds: float

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.events:
            return "Timeline: empty"
        import datetime
        try:
            earliest_str = datetime.datetime.utcfromtimestamp(self.earliest_at).strftime("%Y-%m-%d")
            latest_str = datetime.datetime.utcfromtimestamp(self.latest_at).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            earliest_str = str(self.earliest_at)
            latest_str = str(self.latest_at)
        return (
            f"Timeline: {self.total_memories} memories "
            f"({earliest_str} → {latest_str}), "
            f"{len(self.gaps)} gap(s), "
            f"mean importance={self.mean_importance:.2f}"
        )

    def export_markdown(self) -> str:
        """Export the timeline as a Markdown document."""
        lines: list[str] = ["# Memory Timeline\n"]
        lines.append(f"**{self.summary()}**\n")

        if self.domain_counts:
            lines.append("## Domains\n")
            for dom, cnt in sorted(self.domain_counts.items(), key=lambda x: -x[1]):
                lines.append(f"- **{dom}**: {cnt} memories")
            lines.append("")

        if self.gaps:
            lines.append(f"## Significant Gaps (>{self.gap_threshold_seconds:.0f}s)\n")
            for gap in self.gaps:
                lines.append(
                    f"- {gap.duration_human} gap between `{gap.before_id[:8]}` "
                    f"and `{gap.after_id[:8]}`"
                )
            lines.append("")

        if self.density:
            lines.append("## Storage Density\n")
            lines.append("| Window | Count | Avg Importance | Domains |")
            lines.append("|--------|-------|---------------|---------|")
            for bucket in self.density:
                dom_str = ", ".join(bucket.domains[:3])
                if len(bucket.domains) > 3:
                    dom_str += f" +{len(bucket.domains) - 3}"
                lines.append(
                    f"| {bucket.label} | {bucket.count} | "
                    f"{bucket.avg_importance:.2f} | {dom_str} |"
                )
            lines.append("")

        if self.events:
            lines.append("## Events (chronological)\n")
            for ev in self.events:
                import datetime
                try:
                    ts_str = datetime.datetime.utcfromtimestamp(ev.stored_at).strftime("%Y-%m-%d %H:%M")
                except (OSError, OverflowError, ValueError):
                    ts_str = f"t={ev.stored_at:.0f}"
                title = ev.title or ev.content[:60]
                lines.append(
                    f"- **[{ts_str}]** `{ev.tier}` | {ev.domain} | "
                    f"imp={ev.importance:.2f} | {title}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryTimeline builder
# ---------------------------------------------------------------------------

class MemoryTimeline:
    """Build chronological memory timelines with gap and density analysis.

    Parameters
    ----------
    memory : HierarchicalMemory
        The hierarchical memory store to analyse.
    gap_threshold_seconds : float
        Minimum gap duration (seconds) to report as a TemporalGap.
        Default is 300 (5 minutes).
    bucket_size_seconds : float
        Time window width (seconds) for density histogram buckets.
        Default is 3600 (1 hour).
    """

    def __init__(
        self,
        memory: Any,  # HierarchicalMemory — avoids circular import
        *,
        gap_threshold_seconds: float = 300.0,
        bucket_size_seconds: float = 3600.0,
    ) -> None:
        self.memory = memory
        self.gap_threshold_seconds = gap_threshold_seconds
        self.bucket_size_seconds = bucket_size_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        domain: str | None = None,
        since: float | None = None,
        until: float | None = None,
        tiers: list[str] | None = None,
        include_expired: bool = False,
    ) -> TimelineResult:
        """Build and return a TimelineResult.

        Args:
            domain: Filter to a single domain (None = all domains).
            since: Only include memories stored after this Unix timestamp.
            until: Only include memories stored before this Unix timestamp.
            tiers: Which tier names to include (None = all).
            include_expired: If False (default), skip expired/superseded memories.

        Returns:
            TimelineResult with events, gaps, density histogram, and statistics.
        """
        items = self._collect(domain, since, until, tiers, include_expired)
        # Sort chronologically
        items.sort(key=lambda item: item.stored_at)

        events = [self._to_event(item) for item in items]
        gaps = self._detect_gaps(events)
        density = self._build_density(events)
        stats = self._compute_stats(events)

        return TimelineResult(
            events=events,
            gaps=gaps,
            density=density,
            total_memories=len(events),
            earliest_at=stats["earliest"],
            latest_at=stats["latest"],
            span_seconds=stats["span"],
            mean_importance=stats["mean_importance"],
            domain_counts=stats["domain_counts"],
            gap_threshold_seconds=self.gap_threshold_seconds,
            bucket_size_seconds=self.bucket_size_seconds,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(
        self,
        domain: str | None,
        since: float | None,
        until: float | None,
        tiers: list[str] | None,
        include_expired: bool,
    ) -> list[Any]:
        """Gather MemoryItems matching the filters."""
        from emms.core.models import MemoryTier

        wanted_tiers: set[str] | None = set(tiers) if tiers else None
        result = []
        for tier_name, store in self.memory._iter_tiers():
            tier_str = tier_name.value if hasattr(tier_name, "value") else str(tier_name)
            if wanted_tiers is not None and tier_str not in wanted_tiers:
                continue
            for item in store:
                if not include_expired and (item.is_expired or item.is_superseded):
                    continue
                if domain is not None and item.experience.domain != domain:
                    continue
                if since is not None and item.stored_at < since:
                    continue
                if until is not None and item.stored_at > until:
                    continue
                result.append(item)
        return result

    @staticmethod
    def _to_event(item: Any) -> TimelineEvent:
        content_excerpt = item.experience.content[:120]
        return TimelineEvent(
            memory_id=item.id,
            experience_id=item.experience.id,
            content=content_excerpt,
            domain=item.experience.domain,
            tier=item.tier.value if hasattr(item.tier, "value") else str(item.tier),
            importance=item.experience.importance,
            memory_strength=item.memory_strength,
            stored_at=item.stored_at,
            access_count=item.access_count,
            title=item.experience.title,
        )

    def _detect_gaps(self, events: list[TimelineEvent]) -> list[TemporalGap]:
        """Find significant time gaps between consecutive events."""
        if len(events) < 2:
            return []
        gaps: list[TemporalGap] = []
        for i in range(len(events) - 1):
            a = events[i]
            b = events[i + 1]
            duration = b.stored_at - a.stored_at
            if duration >= self.gap_threshold_seconds:
                gaps.append(TemporalGap(
                    start_at=a.stored_at,
                    end_at=b.stored_at,
                    duration_seconds=duration,
                    before_id=a.memory_id,
                    after_id=b.memory_id,
                ))
        return gaps

    def _build_density(self, events: list[TimelineEvent]) -> list[DensityBucket]:
        """Build a density histogram with fixed-width time buckets."""
        if not events:
            return []

        t_min = events[0].stored_at
        t_max = events[-1].stored_at
        if t_max == t_min:
            # All events at same timestamp — single bucket
            return [DensityBucket(
                start_at=t_min,
                end_at=t_min + self.bucket_size_seconds,
                count=len(events),
                domains=list({ev.domain for ev in events}),
                avg_importance=sum(ev.importance for ev in events) / len(events),
            )]

        n_buckets = max(1, math.ceil((t_max - t_min) / self.bucket_size_seconds))
        buckets: list[list[TimelineEvent]] = [[] for _ in range(n_buckets)]

        for ev in events:
            idx = min(
                n_buckets - 1,
                int((ev.stored_at - t_min) / self.bucket_size_seconds),
            )
            buckets[idx].append(ev)

        result: list[DensityBucket] = []
        for i, bucket_events in enumerate(buckets):
            start = t_min + i * self.bucket_size_seconds
            end = start + self.bucket_size_seconds
            doms = list({ev.domain for ev in bucket_events})
            avg_imp = (
                sum(ev.importance for ev in bucket_events) / len(bucket_events)
                if bucket_events else 0.0
            )
            result.append(DensityBucket(
                start_at=start,
                end_at=end,
                count=len(bucket_events),
                domains=doms,
                avg_importance=avg_imp,
            ))
        return result

    @staticmethod
    def _compute_stats(events: list[TimelineEvent]) -> dict:
        if not events:
            return {
                "earliest": 0.0,
                "latest": 0.0,
                "span": 0.0,
                "mean_importance": 0.0,
                "domain_counts": {},
            }
        from collections import Counter
        t_min = events[0].stored_at
        t_max = events[-1].stored_at
        mean_imp = sum(ev.importance for ev in events) / len(events)
        domain_counts = dict(Counter(ev.domain for ev in events))
        return {
            "earliest": t_min,
            "latest": t_max,
            "span": t_max - t_min,
            "mean_importance": mean_imp,
            "domain_counts": domain_counts,
        }
