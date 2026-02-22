"""TemporalProjection — episodic future thinking and scenario simulation.

v0.18.0: The Predictive Mind

Memory is not only for the past — it is the engine of the future. Episodic
future thinking (Atance & O'Neill 2001) is the ability to mentally project
oneself forward in time and pre-experience a possible future state. This
capacity draws on the same hippocampal scene-construction mechanism as
episodic recall (Hassabis & Maguire 2007), which is why damage to the
hippocampus impairs both past recall and future imagination equally.

TemporalProjection operationalises this for the memory store: it extracts
recurring patterns from memories (and episodes, if an EpisodicBuffer is
attached), extrapolates them forward across a user-specified horizon, and
generates FutureScenario objects describing plausible future states. Each
scenario carries:

  - A plausibility score (0..1) based on pattern frequency, recency, and
    memory strength
  - An emotional_valence estimate (mean valence of basis memories)
  - A projection_horizon in days
  - The IDs of the basis memories and episodes that motivated it

Two generation paths
--------------------
1. **Episode-based**: if an EpisodicBuffer is attached and has closed episodes,
   the engine extracts their outcome patterns and projects forward.
2. **Memory-based**: always available; uses frequency analysis of high-strength
   memories to identify likely future themes.

Biological analogue: episodic future thinking (Atance & O'Neill 2001);
mental time travel (Tulving 1985); hippocampal scene construction as a general
faculty for imagining novel scenarios (Hassabis & Maguire 2007); the default
mode network as the substrate for past recall and future simulation alike
(Buckner & Carroll 2007).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FutureScenario:
    """A plausible future scenario extrapolated from past memory."""

    id: str
    content: str                    # projected future state description
    domain: str
    basis_episode_ids: list[str]    # episodes that informed this scenario
    basis_memory_ids: list[str]     # memories that informed this scenario
    projection_horizon: float       # days into the future
    plausibility: float             # 0..1 — how likely this scenario is
    emotional_valence: float        # projected emotional tone (-1..1)
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"FutureScenario [{self.domain}]  "
            f"plausibility={self.plausibility:.2f}  "
            f"valence={self.emotional_valence:+.2f}  "
            f"horizon={self.projection_horizon:.0f}d\n"
            f"  {self.id[:12]}: {self.content[:80]}"
        )


@dataclass
class ProjectionReport:
    """Result of a TemporalProjection.project() call."""

    total_episodes_used: int
    total_memories_used: int
    scenarios_generated: int
    scenarios: list[FutureScenario]  # sorted by plausibility desc
    mean_plausibility: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"ProjectionReport: {self.scenarios_generated} scenarios from "
            f"{self.total_memories_used} memories + {self.total_episodes_used} episodes  "
            f"mean_plausibility={self.mean_plausibility:.3f} in {self.duration_seconds:.2f}s",
        ]
        for s in self.scenarios[:5]:
            lines.append(
                f"  [{s.domain}] p={s.plausibility:.2f} v={s.emotional_valence:+.2f}: "
                f"{s.content[:60]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TemporalProjection
# ---------------------------------------------------------------------------


class TemporalProjection:
    """Generates plausible future scenarios by extrapolating from memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    episodic_buffer:
        Optional :class:`EpisodicBuffer` for episode-based projection.
    max_scenarios:
        Maximum :class:`FutureScenario` objects to generate (default 8).
    horizon_days:
        Default projection horizon in days (default 30.0).
    """

    def __init__(
        self,
        memory: Any,
        episodic_buffer: Optional[Any] = None,
        max_scenarios: int = 8,
        horizon_days: float = 30.0,
    ) -> None:
        self.memory = memory
        self.episodic_buffer = episodic_buffer
        self.max_scenarios = max_scenarios
        self.horizon_days = horizon_days
        self._scenarios: list[FutureScenario] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        domain: Optional[str] = None,
        horizon_days: Optional[float] = None,
    ) -> ProjectionReport:
        """Generate plausible future scenarios.

        Args:
            domain:       Restrict to one domain (``None`` = all domains).
            horizon_days: Projection horizon in days (overrides default).

        Returns:
            :class:`ProjectionReport` with scenarios sorted by plausibility.
        """
        t0 = time.time()
        horizon = horizon_days if horizon_days is not None else self.horizon_days
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Gather closed episodes
        episodes = []
        if self.episodic_buffer is not None:
            try:
                episodes = [
                    ep for ep in self.episodic_buffer.episodes
                    if ep.closed_at is not None
                ]
                if domain:
                    episodes = [ep for ep in episodes if ep.domain == domain]
            except Exception:
                episodes = []

        # Generate scenarios from two paths
        all_scenarios: list[FutureScenario] = []
        all_scenarios.extend(
            self._project_from_memories(items, horizon)
        )
        all_scenarios.extend(
            self._project_from_episodes(episodes, items, horizon)
        )

        # Deduplicate by content prefix
        seen: set[str] = set()
        unique: list[FutureScenario] = []
        for sc in all_scenarios:
            key = sc.content[:40]
            if key not in seen:
                seen.add(key)
                unique.append(sc)

        unique.sort(key=lambda s: s.plausibility, reverse=True)
        top = unique[: self.max_scenarios]
        self._scenarios.extend(top)

        mean_p = sum(s.plausibility for s in top) / max(len(top), 1)

        return ProjectionReport(
            total_episodes_used=len(episodes),
            total_memories_used=len(items),
            scenarios_generated=len(top),
            scenarios=top,
            mean_plausibility=round(mean_p, 4),
            duration_seconds=time.time() - t0,
        )

    def most_plausible(self, n: int = 3) -> list[FutureScenario]:
        """Return the n most plausible scenarios generated so far.

        Args:
            n: Number of scenarios to return (default 3).

        Returns:
            List of :class:`FutureScenario` sorted by plausibility descending.
        """
        return sorted(self._scenarios, key=lambda s: s.plausibility, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_from_memories(
        self,
        items: list[Any],
        horizon: float,
    ) -> list[FutureScenario]:
        """Generate scenarios from recurring memory patterns."""
        if not items:
            return []

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        scenarios: list[FutureScenario] = []
        for dom, dom_items in list(by_domain.items())[:4]:
            # Token frequency
            token_counts: Counter = Counter()
            for it in dom_items:
                text = getattr(it.experience, "content", "") or ""
                for w in text.lower().split():
                    tok = w.strip(".,!?;:\"'()")
                    if len(tok) >= 4 and tok not in _STOP_WORDS:
                        token_counts[tok] += 1

            if not token_counts:
                continue

            top_token, top_count = token_counts.most_common(1)[0]
            freq = top_count / len(dom_items)

            # Plausibility from frequency + mean strength
            mean_strength = sum(
                min(1.0, max(0.0, getattr(it, "memory_strength", 0.5)))
                for it in dom_items
            ) / len(dom_items)
            plausibility = round(min(1.0, freq * 0.6 + mean_strength * 0.4), 4)

            # Mean valence
            valences = [
                getattr(it.experience, "emotional_valence", 0.0) or 0.0
                for it in dom_items
            ]
            mean_valence = round(sum(valences) / len(valences), 4)

            basis_ids = [it.id for it in dom_items[:5]]
            content = (
                f"Projection for {dom} over the next {horizon:.0f} days: "
                f"based on {len(dom_items)} memories, themes around \"{top_token}\" "
                f"are likely to continue (frequency={freq:.0%}, "
                f"mean_strength={mean_strength:.2f}). "
                f"Emotional outlook: {'positive' if mean_valence > 0.1 else 'negative' if mean_valence < -0.1 else 'neutral'}."
            )
            scenarios.append(FutureScenario(
                id=f"proj_{uuid.uuid4().hex[:8]}",
                content=content,
                domain=dom,
                basis_episode_ids=[],
                basis_memory_ids=basis_ids,
                projection_horizon=horizon,
                plausibility=plausibility,
                emotional_valence=mean_valence,
            ))

        return scenarios

    def _project_from_episodes(
        self,
        episodes: list[Any],
        items: list[Any],
        horizon: float,
    ) -> list[FutureScenario]:
        """Generate scenarios from closed episode outcomes."""
        if not episodes:
            return []

        scenarios: list[FutureScenario] = []
        for ep in episodes[:4]:
            outcome = getattr(ep, "outcome", "") or ""
            if not outcome:
                continue

            dom = getattr(ep, "domain", "general") or "general"
            ep_id = getattr(ep, "id", "")
            key_ids = getattr(ep, "key_memory_ids", []) or []

            # Plausibility: episodes with strong outcomes are more predictive
            turns = max(1, getattr(ep, "turn_count", 1))
            plausibility = round(min(1.0, 0.5 + 0.05 * min(turns, 10)), 4)

            # Emotional arc
            arc = getattr(ep, "emotional_arc", [])
            mean_valence = round(sum(arc) / len(arc), 4) if arc else 0.0

            content = (
                f"Episode-based projection for {dom} over {horizon:.0f} days: "
                f"past episode (outcome: \"{outcome[:80]}\") suggests this pattern "
                f"will recur. Episode spanned {turns} turns with "
                f"{'positive' if mean_valence > 0.1 else 'negative' if mean_valence < -0.1 else 'neutral'} "
                f"emotional trajectory."
            )
            scenarios.append(FutureScenario(
                id=f"proj_{uuid.uuid4().hex[:8]}",
                content=content,
                domain=dom,
                basis_episode_ids=[ep_id],
                basis_memory_ids=list(key_ids)[:5],
                projection_horizon=horizon,
                plausibility=plausibility,
                emotional_valence=mean_valence,
            ))

        return scenarios

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
