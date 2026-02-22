"""SelfModel — explicit self-representation from accumulated memory.

v0.19.0: The Integrated Mind

Humans do not simply accumulate memories — they construct a *self-model*: an
organised set of beliefs about who they are, what they know, and what they
value. This self-model acts as a schema that organises incoming information,
guides behaviour, and provides a sense of continuity across time (Conway &
Pleydell-Pearce 2000). It is not a fixed entity but a living construction,
constantly updated as new experiences are integrated.

The SelfModel operationalises this for the memory store:

1. **Beliefs**: high-frequency recurring themes per domain become explicit
   belief statements about what the agent "knows" or "holds to be true".
   Confidence is a blend of frequency (how often the theme recurs) and
   memory strength (how strongly consolidated it is).

2. **Capability Profile**: each domain is rated on an expertise scale
   combining mean memory strength (depth) with memory count (breadth),
   normalised via a log scale to reward deep knowledge more than shallow
   accumulation.

3. **Consistency Score**: the degree of emotional agreement across beliefs.
   A self-model with high consistency has coherent, non-contradictory beliefs;
   low consistency signals internal conflict requiring belief revision.

4. **Core Domains**: the three domains with the most accumulated memories —
   the agent's areas of primary engagement.

Biological analogue: self-referential processing in medial prefrontal cortex
(Northoff et al. 2006) — the mPFC preferentially activates for self-related
versus other-related judgements; self-schema theory (Markus 1977) — organised
cognitive structures about the self that guide encoding and retrieval;
autobiographical self (Damasio 1999) — the convergence zone that binds
individual memories into a coherent personal identity; self-consistency
motivation (Lecky 1945) — the drive to maintain coherent self-perception.
"""

from __future__ import annotations

import math
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
class Belief:
    """An inferred belief derived from recurring memory patterns."""

    id: str
    content: str                    # human-readable belief statement
    domain: str
    confidence: float               # 0..1 — freq_ratio × 0.6 + mean_strength × 0.4
    supporting_memory_ids: list[str]
    valence: float                  # mean emotional tone of supporting memories
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"Belief [{self.domain}] conf={self.confidence:.2f} "
            f"v={self.valence:+.2f}: {self.content[:80]}"
        )


@dataclass
class SelfModelReport:
    """Full snapshot of the agent's self-model."""

    beliefs: list[Belief]                       # sorted by confidence desc
    core_domains: list[str]                     # top-3 domains by memory count
    dominant_valence: float                     # mean belief valence
    consistency_score: float                    # 0..1 cross-belief emotional agreement
    capability_profile: dict[str, float]        # domain → expertise level (0..1)
    total_memories_analyzed: int
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"SelfModelReport: {len(self.beliefs)} beliefs  "
            f"core_domains={self.core_domains}  "
            f"dominant_valence={self.dominant_valence:+.3f}  "
            f"consistency={self.consistency_score:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for b in self.beliefs[:5]:
            lines.append(f"  {b.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SelfModel
# ---------------------------------------------------------------------------


class SelfModel:
    """Builds and maintains an explicit self-model from accumulated memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_belief_frequency:
        Minimum number of distinct memories a theme must appear in to become
        a belief (default 2).
    max_beliefs:
        Maximum number of beliefs to maintain in the self-model (default 12).
    """

    def __init__(
        self,
        memory: Any,
        min_belief_frequency: int = 2,
        max_beliefs: int = 12,
    ) -> None:
        self.memory = memory
        self.min_belief_frequency = min_belief_frequency
        self.max_beliefs = max_beliefs
        self._beliefs: list[Belief] = []
        self._last_report: Optional[SelfModelReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> SelfModelReport:
        """Rebuild the self-model from current memory contents.

        Returns:
            :class:`SelfModelReport` reflecting the current state.
        """
        t0 = time.time()
        items = self._collect_all()

        self._beliefs = self._extract_beliefs(items)
        cap = self._compute_capability(items)
        consistency = self._compute_consistency(self._beliefs)

        # Core domains (top-3 by memory count)
        domain_counts: Counter = Counter(
            getattr(it.experience, "domain", None) or "general"
            for it in items
        )
        core_domains = [d for d, _ in domain_counts.most_common(3)]

        dominant_valence = 0.0
        if self._beliefs:
            dominant_valence = round(
                sum(b.valence for b in self._beliefs) / len(self._beliefs), 4
            )

        report = SelfModelReport(
            beliefs=self._beliefs,
            core_domains=core_domains,
            dominant_valence=dominant_valence,
            consistency_score=consistency,
            capability_profile=cap,
            total_memories_analyzed=len(items),
            duration_seconds=time.time() - t0,
        )
        self._last_report = report
        return report

    def beliefs(self) -> list[Belief]:
        """Return the current belief list (empty before first update()).

        Returns:
            List of :class:`Belief` sorted by confidence descending.
        """
        return list(self._beliefs)

    def capability_profile(self) -> dict[str, float]:
        """Return domain → expertise level (0..1).

        Returns:
            Dict mapping domain names to expertise scores.
        """
        items = self._collect_all()
        return self._compute_capability(items)

    def consistency_score(self) -> float:
        """Return the cross-belief emotional consistency (0..1).

        Returns:
            1 − std(belief valences), clamped to 0..1.
        """
        return self._compute_consistency(self._beliefs)

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Return a belief by its ID, or ``None``.

        Args:
            belief_id: The belief ID (prefixed ``belief_``).
        """
        for b in self._beliefs:
            if b.id == belief_id:
                return b
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_beliefs(self, items: list[Any]) -> list[Belief]:
        """Derive beliefs from domain-level memory patterns."""
        if not items:
            return []

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        beliefs: list[Belief] = []
        for dom, dom_items in by_domain.items():
            # Token frequency within domain
            token_counts: Counter = Counter()
            for item in dom_items:
                text = getattr(item.experience, "content", "") or ""
                seen: set[str] = set()
                for w in text.lower().split():
                    tok = w.strip(".,!?;:\"'()")
                    if len(tok) >= 4 and tok not in _STOP_WORDS and tok not in seen:
                        token_counts[tok] += 1
                        seen.add(tok)

            if not token_counts:
                continue

            top_token, top_count = token_counts.most_common(1)[0]
            freq_ratio = top_count / len(dom_items)

            if top_count < self.min_belief_frequency:
                continue

            mean_strength = sum(
                min(1.0, max(0.0, getattr(it, "memory_strength", 0.5)))
                for it in dom_items
            ) / len(dom_items)

            confidence = round(
                min(1.0, freq_ratio * 0.6 + mean_strength * 0.4), 4
            )

            valences = [
                getattr(it.experience, "emotional_valence", 0.0) or 0.0
                for it in dom_items
            ]
            mean_valence = round(sum(valences) / len(valences), 4)

            strength_desc = "strong" if confidence > 0.6 else "developing"
            content = (
                f"In {dom}, a recurring pattern around '{top_token}' suggests: "
                f"{dom} understanding is {strength_desc} "
                f"(confidence={confidence:.0%}, {len(dom_items)} memories)."
            )

            beliefs.append(Belief(
                id=f"belief_{uuid.uuid4().hex[:8]}",
                content=content,
                domain=dom,
                confidence=confidence,
                supporting_memory_ids=[it.id for it in dom_items[:5]],
                valence=mean_valence,
            ))

        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs[: self.max_beliefs]

    def _compute_capability(self, items: list[Any]) -> dict[str, float]:
        """Compute domain expertise as depth × breadth composite."""
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        profile: dict[str, float] = {}
        for dom, dom_items in by_domain.items():
            mean_strength = sum(
                min(1.0, max(0.0, getattr(it, "memory_strength", 0.5)))
                for it in dom_items
            ) / len(dom_items)
            count = len(dom_items)
            # Combine depth (strength) with breadth (count), log-scaled
            expertise = min(1.0, mean_strength * math.log1p(count) / math.log1p(5))
            profile[dom] = round(expertise, 4)

        return profile

    def _compute_consistency(self, beliefs: list[Belief]) -> float:
        """Compute 1 − std(belief valences), clamped to 0..1."""
        if len(beliefs) < 2:
            return 1.0
        valences = [b.valence for b in beliefs]
        mean_v = sum(valences) / len(valences)
        variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
        std_v = math.sqrt(variance)
        return round(max(0.0, min(1.0, 1.0 - std_v)), 4)

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
