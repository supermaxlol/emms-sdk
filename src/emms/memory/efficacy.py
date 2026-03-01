"""SelfEfficacyAssessor — domain-specific confidence from outcome patterns.

v0.25.0: The Vigilant Mind

Self-efficacy is an agent's belief in its own capacity to execute behaviours
necessary to produce specific outcomes (Bandura 1977). It is not global
self-esteem but a domain-specific judgment: an agent may hold high efficacy in
one domain while holding low efficacy in another. Critically, efficacy beliefs are
not fixed — they are updated by mastery experiences (direct successes), vicarious
learning, verbal persuasion, and physiological cues. High efficacy drives
goal persistence; low efficacy drives avoidance. Monitoring these domain-specific
profiles gives the agent insight into where it should invest effort and where it
needs support.

SelfEfficacyAssessor operationalises efficacy measurement from memory content:
it scans for success and failure indicator tokens, computes weighted outcome scores
per domain, derives an efficacy_score normalised to 0..1, and tracks the
trend (improving / declining / stable) by comparing early vs. recent memory halves.
It also extracts recent_themes — the tokens that appear most often in
success-associated memories, giving the agent clues about what skills and
strategies are actually working.

Biological analogue: Bandura self-efficacy theory (1977); prefrontal cortex
confidence representations; frontostriatal circuits in capability appraisal
(Frank et al. 2004); growth mindset and neuroplasticity (Dweck 2006); orbitofrontal
cortex in value-weighted confidence signals (Rangel et al. 2008); developmental
progression from external to internal locus of control (Rotter 1966).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_SUCCESS_TOKENS: frozenset[str] = frozenset({
    "succeed", "achieved", "completed", "mastered", "solved",
    "accomplished", "worked", "improved", "passed", "managed",
    "correct", "effective", "capable", "confident", "ready",
})

_FAILURE_TOKENS: frozenset[str] = frozenset({
    "failed", "wrong", "error", "mistake", "struggle",
    "unable", "difficult", "broken", "ineffective", "poor",
    "lost", "missed", "inadequate", "weak", "confused",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EfficacyProfile:
    """Domain-specific self-efficacy assessment."""

    domain: str
    efficacy_score: float    # 0..1; high = high self-efficacy
    success_count: int       # memories with ≥1 success token
    failure_count: int       # memories with ≥1 failure token
    trending: str            # "improving" / "declining" / "stable"
    recent_themes: list[str] # top 5 tokens from success-indicator memories

    def summary(self) -> str:
        return (
            f"EfficacyProfile [{self.domain}]  "
            f"efficacy={self.efficacy_score:.3f}  "
            f"success={self.success_count}  failure={self.failure_count}  "
            f"trending={self.trending}"
        )


@dataclass
class EfficacyReport:
    """Result of a SelfEfficacyAssessor.assess() call."""

    total_domains: int
    profiles: list[EfficacyProfile]   # sorted by efficacy_score desc
    highest_efficacy_domain: str      # "none" if empty
    lowest_efficacy_domain: str       # "none" if empty
    mean_efficacy: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"EfficacyReport: {self.total_domains} domains  "
            f"highest={self.highest_efficacy_domain}  "
            f"lowest={self.lowest_efficacy_domain}  "
            f"mean_efficacy={self.mean_efficacy:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for p in self.profiles[:5]:
            lines.append(f"  {p.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SelfEfficacyAssessor
# ---------------------------------------------------------------------------


class SelfEfficacyAssessor:
    """Measures domain-specific self-efficacy from outcome language in memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_memories:
        Minimum memories for a domain to be assessed (default 2).
    """

    def __init__(
        self,
        memory: Any,
        min_memories: int = 2,
    ) -> None:
        self.memory = memory
        self.min_memories = min_memories
        self._profiles: list[EfficacyProfile] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, domain: Optional[str] = None) -> EfficacyReport:
        """Assess self-efficacy across all (or one) memory domain(s).

        Args:
            domain: Restrict to this domain (``None`` = all domains).

        Returns:
            :class:`EfficacyReport` with profiles sorted by efficacy_score desc.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain
        domain_items: dict[str, list[Any]] = defaultdict(list)
        for item in all_items:
            dom = getattr(item.experience, "domain", None) or "general"
            domain_items[dom].append(item)

        profiles: list[EfficacyProfile] = []

        for dom, items in domain_items.items():
            if len(items) < self.min_memories:
                continue

            # Sort by timestamp for trending
            items_sorted = sorted(
                items,
                key=lambda it: getattr(it.experience, "timestamp", 0.0) or 0.0,
            )

            def _efficacy_for_group(group: list[Any]) -> tuple[float, float, int, int]:
                """Returns (efficacy_score, success_score, success_count, failure_count)."""
                success_items = []
                failure_items = []
                for it in group:
                    content = getattr(it.experience, "content", "") or ""
                    tokens = {w.strip(".,!?;:\"'()").lower() for w in content.split()}
                    if tokens & _SUCCESS_TOKENS:
                        success_items.append(it)
                    if tokens & _FAILURE_TOKENS:
                        failure_items.append(it)

                success_count = len(success_items)
                failure_count = len(failure_items)

                success_score = (
                    sum(getattr(it.experience, "importance", 0.5) or 0.5 for it in success_items)
                    / max(success_count, 1)
                )
                failure_score = (
                    sum(getattr(it.experience, "importance", 0.5) or 0.5 for it in failure_items)
                    / max(failure_count, 1)
                )

                eff = round(
                    min(1.0, max(0.0, (success_score - failure_score + 1.0) / 2.0)),
                    4,
                )
                return eff, success_score, success_count, failure_count

            efficacy_score, _, success_count, failure_count = _efficacy_for_group(items_sorted)

            # Trending via early/recent halves
            n = len(items_sorted)
            mid = n // 2
            early = items_sorted[:mid]
            recent = items_sorted[mid:]

            if early and recent:
                eff_early, _, _, _ = _efficacy_for_group(early)
                eff_recent, _, _, _ = _efficacy_for_group(recent)
                if eff_recent > eff_early + 0.1:
                    trending = "improving"
                elif eff_recent < eff_early - 0.1:
                    trending = "declining"
                else:
                    trending = "stable"
            else:
                trending = "stable"

            # Recent themes: top 5 tokens from success-indicator memories
            success_counter: Counter = Counter()
            for it in items_sorted:
                content = getattr(it.experience, "content", "") or ""
                tokens_lower = {w.strip(".,!?;:\"'()").lower() for w in content.split()}
                if tokens_lower & _SUCCESS_TOKENS:
                    for tok in self._tokenise(content):
                        success_counter[tok] += 1
            recent_themes = [tok for tok, _ in success_counter.most_common(5)]

            profiles.append(EfficacyProfile(
                domain=dom,
                efficacy_score=efficacy_score,
                success_count=success_count,
                failure_count=failure_count,
                trending=trending,
                recent_themes=recent_themes,
            ))

        profiles.sort(key=lambda p: p.efficacy_score, reverse=True)
        self._profiles = profiles

        highest = profiles[0].domain if profiles else "none"
        lowest = profiles[-1].domain if profiles else "none"
        mean_eff = round(
            sum(p.efficacy_score for p in profiles) / len(profiles)
            if profiles else 0.0,
            4,
        )

        return EfficacyReport(
            total_domains=len(profiles),
            profiles=profiles,
            highest_efficacy_domain=highest,
            lowest_efficacy_domain=lowest,
            mean_efficacy=mean_eff,
            duration_seconds=time.time() - t0,
        )

    def efficacy_for_domain(self, domain: str) -> Optional[EfficacyProfile]:
        """Return the efficacy profile for a specific domain.

        Args:
            domain: Domain name to query.

        Returns:
            :class:`EfficacyProfile`, or ``None`` if not found.
        """
        for p in self._profiles:
            if p.domain == domain:
                return p
        return None

    def highest_efficacy_domain(self) -> Optional[str]:
        """Return the domain name with the highest efficacy score, or ``None``."""
        return self._profiles[0].domain if self._profiles else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 3
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
