"""SelfCompassionGauge — measuring self-kindness vs. self-harshness in memory.

v0.26.0: The Resilient Mind

How an agent speaks about itself in memory — whether it uses language of
understanding and acceptance or language of condemnation and shame — reveals
the internal tone of its self-relationship (Neff 2003). Self-compassion is not
self-indulgence or lowered standards: it is the capacity to hold one's own
suffering, failure, and imperfection with the same warmth one would offer a
good friend. Research shows self-compassion predicts psychological well-being
better than self-esteem, because it is not contingent on success (Neff &
Vonk 2009). A ruthless inner critic, by contrast, amplifies negative affect,
increases rumination, and reduces the adaptive flexibility needed to learn from
mistakes (Gilbert 2010).

SelfCompassionGauge operationalises self-compassion measurement for the memory
store: it scans memory content for tokens belonging to two curated lexicons —
KINDNESS (accepting, affirming, normalising language) and HARSH (condemning,
devaluing, catastrophising language) — and computes a domain-specific
compassion_score and inner_critic_intensity from their relative frequencies.
The resulting SelfCompassionReport identifies where the agent is most and least
compassionate toward itself.

Biological analogue: prefrontal cortex–anterior cingulate self-referential
regulation (Northoff 2011); self-compassion vs. self-criticism neural
dissociation (Longe et al. 2010); shame and the subordinate-self response
(Gilbert 2010); mindful self-compassion as deactivation of threat-processing
loop (Germer & Neff 2013); oxytocin and affiliative self-soothing; parasym-
pathetic activation through self-kind internal speech (Porges 2011).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_KINDNESS_TOKENS: frozenset[str] = frozenset({
    "okay", "allowed", "human", "normal", "forgive", "accept",
    "gentle", "kind", "patient", "understand", "imperfect",
    "natural", "learning", "growth", "enough",
})

_HARSH_TOKENS: frozenset[str] = frozenset({
    "stupid", "pathetic", "worthless", "useless", "weak", "failure",
    "terrible", "awful", "hopeless", "inadequate", "shameful",
    "disgusting", "horrible", "lazy", "idiot",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SelfCompassionProfile:
    """Self-compassion profile for a single domain."""

    domain: str
    compassion_score: float         # 0..1; higher = more self-compassionate
    kindness_count: int
    harsh_count: int
    inner_critic_intensity: float   # 0..1; harsh_count / n_domain
    self_directed_themes: list[str] # top 5 tokens from kindness memories

    def summary(self) -> str:
        return (
            f"SelfCompassionProfile [{self.domain}  "
            f"score={self.compassion_score:.3f}  "
            f"critic={self.inner_critic_intensity:.3f}  "
            f"kind={self.kindness_count}  harsh={self.harsh_count}]"
        )


@dataclass
class SelfCompassionReport:
    """Result of a SelfCompassionGauge.measure() call."""

    total_domains: int
    profiles: list[SelfCompassionProfile]   # sorted by compassion_score desc
    most_compassionate_domain: str          # "none" if empty
    harshest_domain: str                    # "none" if empty
    mean_compassion_score: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"SelfCompassionReport: {self.total_domains} domains  "
            f"most_compassionate={self.most_compassionate_domain}  "
            f"harshest={self.harshest_domain}  "
            f"mean_score={self.mean_compassion_score:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for p in self.profiles[:5]:
            lines.append(f"  {p.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SelfCompassionGauge
# ---------------------------------------------------------------------------


class SelfCompassionGauge:
    """Measures self-kindness vs. self-harshness in accumulated memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_memories:
        Minimum number of memories per domain to form a profile (default 2).
    """

    def __init__(self, memory: Any, min_memories: int = 2) -> None:
        self.memory = memory
        self.min_memories = min_memories
        self._profiles: list[SelfCompassionProfile] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(self, domain: Optional[str] = None) -> SelfCompassionReport:
        """Measure self-compassion across all domains (or a specific one).

        Args:
            domain: Restrict analysis to this domain (``None`` = all).

        Returns:
            :class:`SelfCompassionReport` with profiles sorted by
            compassion_score descending.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain
        by_domain: dict[str, list[Any]] = defaultdict(list)
        for item in all_items:
            d = getattr(item.experience, "domain", None) or "general"
            by_domain[d].append(item)

        profiles: list[SelfCompassionProfile] = []
        for dom, items in by_domain.items():
            if len(items) < self.min_memories:
                continue

            n_domain = len(items)
            kindness_memories: list[Any] = []
            harsh_memories: list[Any] = []

            for item in items:
                content = getattr(item.experience, "content", "") or ""
                tokens = set(self._tokenise(content))
                if tokens & _KINDNESS_TOKENS:
                    kindness_memories.append(item)
                if tokens & _HARSH_TOKENS:
                    harsh_memories.append(item)

            kindness_count = len(kindness_memories)
            harsh_count = len(harsh_memories)

            compassion_score = round(
                min(1.0, max(0.0,
                    (kindness_count - harsh_count + n_domain) / max(2 * n_domain, 1)
                )),
                4,
            )
            inner_critic_intensity = round(
                min(1.0, harsh_count / max(n_domain, 1)),
                4,
            )

            # Top 5 tokens from kindness memories
            token_counter: Counter = Counter()
            for item in kindness_memories:
                content = getattr(item.experience, "content", "") or ""
                for tok in self._tokenise(content):
                    token_counter[tok] += 1
            self_directed_themes = [tok for tok, _ in token_counter.most_common(5)]

            profiles.append(SelfCompassionProfile(
                domain=dom,
                compassion_score=compassion_score,
                kindness_count=kindness_count,
                harsh_count=harsh_count,
                inner_critic_intensity=inner_critic_intensity,
                self_directed_themes=self_directed_themes,
            ))

        profiles.sort(key=lambda p: p.compassion_score, reverse=True)
        self._profiles = profiles

        most_compassionate = profiles[0].domain if profiles else "none"
        harshest = profiles[-1].domain if profiles else "none"
        mean_score = round(
            sum(p.compassion_score for p in profiles) / len(profiles)
            if profiles else 0.0,
            4,
        )

        return SelfCompassionReport(
            total_domains=len(profiles),
            profiles=profiles,
            most_compassionate_domain=most_compassionate,
            harshest_domain=harshest,
            mean_compassion_score=mean_score,
            duration_seconds=time.time() - t0,
        )

    def profile_for_domain(self, domain: str) -> Optional[SelfCompassionProfile]:
        """Return the cached profile for a specific domain, or ``None``.

        Call :meth:`measure` first to populate the cache.

        Args:
            domain: Domain name to look up.

        Returns:
            :class:`SelfCompassionProfile` or ``None``.
        """
        for p in self._profiles:
            if p.domain == domain:
                return p
        return None

    def harshest_domain(self) -> Optional[str]:
        """Return the domain with the lowest compassion_score.

        Returns:
            Domain name or ``None`` if no profiles exist.
        """
        if not self._profiles:
            return None
        return self._profiles[-1].domain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
