"""NarrativeWeaver — autobiographical narrative construction.

v0.15.0: The Reflective Mind

Individual memories are facts; episodes are bounded experiences; narratives
are coherent stories that connect experiences across time into a meaningful
whole. The NarrativeWeaver transforms the flat memory store into a structured
autobiography — identifying thematic threads per domain, ordering events
chronologically, and generating readable first-person prose for each segment.

The result is a NarrativeReport the agent can use to summarise its history
in conversation, inject into system prompts, or analyse for self-consistency.

Biological analogue: the left-hemisphere "interpreter" (Gazzaniga 1989) that
constantly constructs causal and narrative explanations for experience; the
autobiographical narrative self (Conway & Pleydell-Pearce 2000) — the
long-term self-knowledge structure that gives coherence to identity over time.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Prose templates
# ---------------------------------------------------------------------------

_OPENING_TEMPLATES = [
    "In the area of {domain}, early experiences involved {excerpt}.",
    "The {domain} thread began with {excerpt}.",
    "Initial {domain} work focused on {excerpt}.",
]

_MIDDLE_TEMPLATES = [
    "This developed further: {excerpt}.",
    "Building on earlier work, {excerpt}.",
    "A subsequent {domain} experience involved {excerpt}.",
]

_CLOSING_TEMPLATES = [
    "More recently, {excerpt}.",
    "The most recent {domain} experience involved {excerpt}.",
    "The thread concluded (for now) with {excerpt}.",
]

_SINGLE_TEMPLATES = [
    "In {domain}: {excerpt}.",
    "A single {domain} experience captures: {excerpt}.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NarrativeSegment:
    """A short prose block synthesised from a cluster of memories."""

    id: str
    text: str                     # Narrative prose
    memory_ids: list[str]         # Source memory IDs
    episode_ids: list[str]        # Source episode IDs (may be empty)
    domain: str
    temporal_position: float      # 0 = earliest, 1 = latest in the thread
    emotional_tone: float         # mean valence of contributing memories


@dataclass
class NarrativeThread:
    """A domain-scoped storyline spanning multiple narrative segments."""

    id: str
    theme: str                    # Descriptive theme label
    domain: str
    segments: list[NarrativeSegment]
    span_seconds: float           # Time from first to last memory (0 if single)
    arc: list[float]              # Emotional arc (mean valence per segment)

    def story(self) -> str:
        """Return the full narrative as a single readable string."""
        return " ".join(seg.text for seg in self.segments)

    def summary(self) -> str:
        arc_str = " → ".join(f"{v:+.2f}" for v in self.arc[-5:])
        return (
            f"NarrativeThread [{self.domain}] theme='{self.theme}' "
            f"segments={len(self.segments)} span={self.span_seconds:.0f}s "
            f"arc=[{arc_str}]"
        )


@dataclass
class NarrativeReport:
    """Result of a NarrativeWeaver.weave() call."""

    total_segments: int
    total_threads: int
    threads: list[NarrativeThread]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"NarrativeReport: {self.total_threads} threads, "
            f"{self.total_segments} segments in {self.duration_seconds:.2f}s",
        ]
        for t in self.threads[:5]:
            lines.append(f"  {t.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NarrativeWeaver
# ---------------------------------------------------------------------------

class NarrativeWeaver:
    """Constructs autobiographical narrative threads from stored memories.

    For each domain (or a specified domain) the weaver:

    1. Collects all memories and sorts them chronologically by ``stored_at``.
    2. Divides the timeline into segments of ≤ ``max_segment_memories`` items.
    3. Generates a prose sentence for each segment using position-aware
       templates (opening / middle / closing).
    4. Assembles segments into a :class:`NarrativeThread` with emotional arc.
    5. Returns a :class:`NarrativeReport` sorted by thread length descending.

    Domains with fewer than ``min_thread_length`` memories are skipped.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    episodic_buffer:
        Optional :class:`EpisodicBuffer` used to annotate segments with
        episode IDs when available.
    min_thread_length:
        Minimum number of memories for a domain to produce a thread (default 2).
    max_threads:
        Maximum threads to return (default 8).
    max_segment_memories:
        Maximum memories per prose segment (default 5).
    """

    def __init__(
        self,
        memory: Any,
        episodic_buffer: Optional[Any] = None,
        min_thread_length: int = 2,
        max_threads: int = 8,
        max_segment_memories: int = 5,
    ) -> None:
        self.memory = memory
        self.episodic_buffer = episodic_buffer
        self.min_thread_length = min_thread_length
        self.max_threads = max_threads
        self.max_segment_memories = max_segment_memories

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weave(
        self,
        domain: Optional[str] = None,
        max_threads: Optional[int] = None,
    ) -> NarrativeReport:
        """Weave autobiographical narrative threads from stored memories.

        Args:
            domain:      Restrict to one domain (``None`` = all domains).
            max_threads: Override the instance default.

        Returns:
            :class:`NarrativeReport` with assembled threads.
        """
        t0 = time.time()
        limit = max_threads if max_threads is not None else self.max_threads
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        # Build episode membership map
        ep_map: dict[str, list[str]] = {}  # memory_id → list of episode_ids
        if self.episodic_buffer is not None:
            for ep in self.episodic_buffer.all_episodes():
                for mid in ep.key_memory_ids:
                    ep_map.setdefault(mid, []).append(ep.id)

        threads: list[NarrativeThread] = []
        for dom, dom_items in by_domain.items():
            if len(dom_items) < self.min_thread_length:
                continue
            thread = self._build_thread(dom, dom_items, ep_map)
            if thread:
                threads.append(thread)

        # Sort by number of segments descending, take top-k
        threads.sort(key=lambda t: len(t.segments), reverse=True)
        threads = threads[:limit]

        total_segs = sum(len(t.segments) for t in threads)
        return NarrativeReport(
            total_segments=total_segs,
            total_threads=len(threads),
            threads=threads,
            duration_seconds=time.time() - t0,
        )

    def get_threads(self, domain: Optional[str] = None) -> list[NarrativeThread]:
        """Return narrative threads (shorthand for weave().threads).

        Args:
            domain: Restrict to one domain (``None`` = all).

        Returns:
            List of :class:`NarrativeThread`, longest first.
        """
        return self.weave(domain=domain).threads

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_thread(
        self,
        domain: str,
        items: list[Any],
        ep_map: dict[str, list[str]],
    ) -> Optional[NarrativeThread]:
        """Build a single narrative thread for one domain."""
        # Sort chronologically
        sorted_items = sorted(items, key=lambda it: it.stored_at)
        if not sorted_items:
            return None

        # Divide into segments
        seg_size = self.max_segment_memories
        chunks = [
            sorted_items[i: i + seg_size]
            for i in range(0, len(sorted_items), seg_size)
        ]
        n_chunks = len(chunks)

        segments: list[NarrativeSegment] = []
        arc: list[float] = []

        t_first = sorted_items[0].stored_at
        t_last = sorted_items[-1].stored_at
        span = max(t_last - t_first, 0.0)

        for idx, chunk in enumerate(chunks):
            # temporal position: 0 = first chunk, 1 = last chunk
            t_pos = idx / max(n_chunks - 1, 1)

            mean_v = self._mean_valence(chunk)
            arc.append(mean_v)

            text = self._segment_prose(chunk, domain, idx, n_chunks)
            memory_ids = [it.id for it in chunk]
            episode_ids: list[str] = []
            for mid in memory_ids:
                episode_ids.extend(ep_map.get(mid, []))
            episode_ids = list(dict.fromkeys(episode_ids))  # deduplicate, preserve order

            segments.append(NarrativeSegment(
                id=f"seg_{uuid.uuid4().hex[:8]}",
                text=text,
                memory_ids=memory_ids,
                episode_ids=episode_ids,
                domain=domain,
                temporal_position=round(t_pos, 3),
                emotional_tone=round(mean_v, 3),
            ))

        # Theme = most frequent content word across all items
        theme = self._extract_theme(sorted_items, domain)

        return NarrativeThread(
            id=f"thread_{uuid.uuid4().hex[:8]}",
            theme=theme,
            domain=domain,
            segments=segments,
            span_seconds=span,
            arc=arc,
        )

    def _segment_prose(
        self,
        items: list[Any],
        domain: str,
        idx: int,
        total: int,
    ) -> str:
        """Generate a prose sentence for a segment using position-aware templates."""
        excerpt = self._excerpt(items)
        if total == 1:
            templates = _SINGLE_TEMPLATES
            tmpl = templates[0]
        elif idx == 0:
            templates = _OPENING_TEMPLATES
            tmpl = templates[0]
        elif idx == total - 1:
            templates = _CLOSING_TEMPLATES
            tmpl = templates[0]
        else:
            templates = _MIDDLE_TEMPLATES
            tmpl = templates[min(idx - 1, len(templates) - 1)]
        return tmpl.format(domain=domain, excerpt=excerpt)

    def _excerpt(self, items: list[Any]) -> str:
        """Produce a short excerpt summarising a group of memory items."""
        # Take the highest-importance item's content, truncated
        best = max(items, key=lambda it: getattr(it.experience, "importance", 0.0) or 0.0)
        text = best.experience.content
        if len(text) > 80:
            text = text[:77] + "..."
        return text.rstrip(".")

    def _mean_valence(self, items: list[Any]) -> float:
        valences = [
            getattr(it.experience, "emotional_valence", 0.0) or 0.0
            for it in items
        ]
        if not valences:
            return 0.0
        return sum(valences) / len(valences)

    def _extract_theme(self, items: list[Any], domain: str) -> str:
        """Extract the dominant theme keyword from items."""
        from collections import Counter
        stop = {"memory", "memories", "experience", "agent", "also", "more",
                 "some", "that", "this", "with", "from", "have", "been", "were",
                 "about", "into", "their", "which", "when", "other"}
        counter: Counter = Counter()
        for item in items:
            tokens = item.experience.content.lower().split()
            for t in tokens:
                t = t.strip(".,!?;:\"'()")
                if len(t) >= 5 and t not in stop:
                    counter[t] += 1
        if counter:
            return counter.most_common(1)[0][0]
        return domain

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
