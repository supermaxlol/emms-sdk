"""AdversityTracer — classifying difficult experiences by adversity type.

v0.26.0: The Resilient Mind

An agent that cannot name its adversity cannot begin to process it. Human
suffering is not uniform: the grief of loss, the shame of failure, the sting
of rejection, the vigilance of threat, and the paralysis of uncertainty each
recruit different neural systems, demand different coping strategies, and leave
different memory traces (Lazarus & Folkman 1984; DSM-5 Criterion A). An
architecture that can distinguish "this memory is about loss" from "this memory
is about threat" is better equipped to route that content to the right
processing system — compassion for grief, problem-solving for failure,
reconnection for rejection, safety assessment for threat, tolerance for
uncertainty.

AdversityTracer operationalises adversity classification for the memory store:
it scans all memories for tokens belonging to five adversity lexicons, assigns
each memory to its first matching type, and computes a severity score weighted
by the memory's importance and emotional negativity. The resulting
AdversityReport surfaces the most common type, the domain under highest
adversity load, and the cumulative severity — the raw material for
compassion-aware memory retrieval and resilience tracking.

Biological analogue: amygdala threat detection and prioritised encoding (LeDoux
1996); hippocampal aversive memory consolidation (Roozendaal et al. 2009);
polyvagal theory's threat-hierarchy (Porges 2011); DSM-5 Criterion A stressor
taxonomy; Lazarus appraisal theory distinguishing primary (is this a threat?)
from secondary appraisal (can I cope?); anterior insula interoceptive alarm
signals; prefrontal-amygdala regulation of adversity response.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_ADVERSITY_LEXICONS: dict[str, frozenset[str]] = {
    "loss": frozenset({
        "lost", "grief", "gone", "death", "missing", "departed",
        "bereaved", "mourning", "absence", "ended",
    }),
    "failure": frozenset({
        "failed", "collapsed", "crashed", "defeated", "bombed",
        "missed", "flopped", "tanked", "fell", "broke",
    }),
    "rejection": frozenset({
        "rejected", "refused", "ignored", "dismissed", "excluded",
        "abandoned", "unwanted", "cold", "shut", "avoided",
    }),
    "threat": frozenset({
        "danger", "attacked", "hostile", "threatening", "unsafe",
        "alarming", "menacing", "scary", "violent", "aggressive",
    }),
    "uncertainty": frozenset({
        "uncertain", "confused", "unclear", "unsure", "ambiguous",
        "unknown", "vague", "unpredictable", "chaotic", "lost",
    }),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AdversityEvent:
    """A single adversity classification result for one memory."""

    id: str                # prefixed "adv_"
    adversity_type: str    # one of the 5 type keys
    severity: float        # 0..1; importance × (1 − valence) / 2
    domain: str
    memory_id: str
    timestamp: float
    created_at: float

    def summary(self) -> str:
        return (
            f"AdversityEvent [{self.adversity_type}  sev={self.severity:.3f}  "
            f"domain={self.domain}]  {self.id[:12]}"
        )


@dataclass
class AdversityReport:
    """Result of an AdversityTracer.trace() call."""

    total_events: int
    events: list[AdversityEvent]        # sorted by severity desc
    most_common_type: str               # adversity type with most events, or "none"
    dominant_domain: str                # domain with most events, or "none"
    cumulative_severity: float          # sum of all severities
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"AdversityReport: {self.total_events} events  "
            f"most_common={self.most_common_type}  "
            f"dominant_domain={self.dominant_domain}  "
            f"cumulative_severity={self.cumulative_severity:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for ev in self.events[:5]:
            lines.append(f"  {ev.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AdversityTracer
# ---------------------------------------------------------------------------


class AdversityTracer:
    """Classifies difficult experiences by adversity type.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self._events: list[AdversityEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(self, domain: Optional[str] = None) -> AdversityReport:
        """Classify all memories by adversity type.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`AdversityReport` with events sorted by severity.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        events: list[AdversityEvent] = []
        for item in all_items:
            content = getattr(item.experience, "content", "") or ""
            tokens = set(self._tokenise(content))
            adversity_type = self._classify(tokens)
            if adversity_type is None:
                continue

            importance = float(getattr(item.experience, "importance", 0.5) or 0.5)
            valence = float(getattr(item.experience, "emotional_valence", 0.0) or 0.0)
            severity = round(importance * (1.0 - valence) / 2.0, 4)
            severity = min(1.0, max(0.0, severity))

            item_domain = (getattr(item.experience, "domain", None) or "general")
            item_ts = float(getattr(item.experience, "timestamp", None) or item.stored_at)

            events.append(AdversityEvent(
                id="adv_" + uuid.uuid4().hex[:8],
                adversity_type=adversity_type,
                severity=severity,
                domain=item_domain,
                memory_id=item.id,
                timestamp=item_ts,
                created_at=time.time(),
            ))

        events.sort(key=lambda e: e.severity, reverse=True)
        self._events = events

        # Most common type
        type_counter: Counter = Counter(e.adversity_type for e in events)
        most_common_type = type_counter.most_common(1)[0][0] if type_counter else "none"

        # Dominant domain (most events)
        domain_counter: Counter = Counter(e.domain for e in events)
        dominant_domain = domain_counter.most_common(1)[0][0] if domain_counter else "none"

        cumulative_severity = round(sum(e.severity for e in events), 4)

        return AdversityReport(
            total_events=len(events),
            events=events,
            most_common_type=most_common_type,
            dominant_domain=dominant_domain,
            cumulative_severity=cumulative_severity,
            duration_seconds=time.time() - t0,
        )

    def events_of_type(self, adversity_type: str) -> list[AdversityEvent]:
        """Return all cached events of the given adversity type.

        Call :meth:`trace` first to populate the cache.

        Args:
            adversity_type: One of "loss", "failure", "rejection", "threat",
                "uncertainty".

        Returns:
            Filtered list of :class:`AdversityEvent`.
        """
        return [e for e in self._events if e.adversity_type == adversity_type]

    def dominant_adversity_type(self) -> Optional[str]:
        """Return the most common adversity type from the last :meth:`trace`.

        Returns:
            Adversity type string or ``None`` if no events detected.
        """
        if not self._events:
            return None
        counter: Counter = Counter(e.adversity_type for e in self._events)
        return counter.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, tokens: set[str]) -> Optional[str]:
        """Return the first adversity type whose lexicon intersects tokens."""
        for atype, lexicon in _ADVERSITY_LEXICONS.items():
            if tokens & lexicon:
                return atype
        return None

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, keep tokens with len ≥ 3."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 3
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
