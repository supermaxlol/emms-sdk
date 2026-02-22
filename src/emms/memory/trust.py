"""TrustLedger — source credibility scoring from memory patterns.

v0.21.0: The Social Mind

Not all information is equally reliable. Human cognition is remarkably sensitive
to source credibility — we weight claims differently depending on who made them,
how consistent their track record has been, and how emotionally stable their
communications. Source monitoring (Johnson et al. 1993) is the cognitive process
by which we track the origins of our knowledge and calibrate trust accordingly.
This capacity is essential for epistemic hygiene: knowing which sources to rely
on, which to discount, and how to integrate conflicting information.

TrustLedger operationalises source credibility scoring for the memory store. It
groups memories by their domain (treated as the information source), and for each
source computes a composite trust score from three signals: (1) mean importance
— higher-importance memories from a source indicate greater reliability; (2)
valence stability — sources that consistently produce emotionally coherent content
are more trustworthy than those with wildly varying affect; (3) memory count score
— sources with more memories have had more opportunity to demonstrate reliability.
The resulting TrustScore per source enables an agent to weight incoming information
by source credibility and identify the most vs least reliable knowledge domains.

Biological analogue: source monitoring framework (Johnson, Hashtroudi & Lindsay
1993) — the cognitive system tracks the origins of memories; ventromedial prefrontal
cortex in computing the social value and credibility of information sources (Behrens
et al. 2008); the posterior superior temporal sulcus in tracking social reliability
(Todorov 2008); trust as Bayesian belief updating over agent-reliability priors
(Fogg 2003); repetition and consistency as cues to persuasive credibility (Petty
& Cacioppo 1986).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrustScore:
    """Credibility score for a single information source (domain)."""

    source: str                 # domain name
    trust: float                # 0..1 composite credibility score
    memory_count: int           # number of memories from this source
    mean_importance: float      # mean importance of those memories
    valence_stability: float    # 1 − std(valences), clamped 0..1

    def summary(self) -> str:
        return (
            f"TrustScore['{self.source}']  trust={self.trust:.3f}  "
            f"count={self.memory_count}  stability={self.valence_stability:.3f}"
        )


@dataclass
class TrustReport:
    """Result of a TrustLedger.compute_trust() call."""

    scores: list[TrustScore]    # sorted by trust descending
    most_trusted: list[str]     # top-5 source names
    least_trusted: list[str]    # bottom-5 source names
    total_sources: int
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"TrustReport: {self.total_sources} sources  "
            f"in {self.duration_seconds:.2f}s",
            f"  Most trusted: {self.most_trusted[:5]}",
            f"  Least trusted: {self.least_trusted[:5]}",
        ]
        for ts in self.scores[:5]:
            lines.append(f"  {ts.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TrustLedger
# ---------------------------------------------------------------------------


class TrustLedger:
    """Computes and tracks credibility scores per memory source (domain).

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_memories:
        Minimum number of memories from a source to include it in the ledger
        (default 1).
    """

    def __init__(
        self,
        memory: Any,
        min_memories: int = 1,
    ) -> None:
        self.memory = memory
        self.min_memories = min_memories
        self._scores: dict[str, TrustScore] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_trust(self, domain: Optional[str] = None) -> TrustReport:
        """Compute trust scores for all information sources in memory.

        Args:
            domain: If provided, compute trust only for this source domain.

        Returns:
            :class:`TrustReport` with scores sorted by trust descending.
        """
        t0 = time.time()
        items = self._collect_all()

        # Group items by source (domain)
        source_groups: dict[str, list[Any]] = {}
        for item in items:
            src = getattr(item.experience, "domain", None) or "general"
            if domain and src != domain:
                continue
            source_groups.setdefault(src, []).append(item)

        self._scores.clear()
        for src, group in source_groups.items():
            if len(group) < self.min_memories:
                continue
            ts = self._compute_score(src, group)
            self._scores[src] = ts

        scores = sorted(self._scores.values(), key=lambda s: s.trust, reverse=True)
        most_trusted = [s.source for s in scores[:5]]
        least_trusted = [s.source for s in scores[-5:] if s.trust < scores[0].trust] if scores else []

        return TrustReport(
            scores=scores,
            most_trusted=most_trusted,
            least_trusted=least_trusted,
            total_sources=len(scores),
            duration_seconds=time.time() - t0,
        )

    def trust_of(self, source: str) -> float:
        """Return the trust score for a named source.

        Args:
            source: Domain name of the source.

        Returns:
            Trust score 0..1, or 0.5 (neutral) if source is unknown.
        """
        ts = self._scores.get(source)
        return ts.trust if ts is not None else 0.5

    def most_trusted(self, n: int = 5) -> list[TrustScore]:
        """Return the n most trustworthy sources.

        Args:
            n: Number of sources to return (default 5).
        """
        return sorted(self._scores.values(), key=lambda s: s.trust, reverse=True)[:n]

    def least_trusted(self, n: int = 5) -> list[TrustScore]:
        """Return the n least trustworthy sources.

        Args:
            n: Number of sources to return (default 5).
        """
        return sorted(self._scores.values(), key=lambda s: s.trust)[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, source: str, items: list[Any]) -> TrustScore:
        """Compute a TrustScore for a single source from its memory items."""
        importances = [
            min(1.0, max(0.0, getattr(item.experience, "importance", 0.5) or 0.5))
            for item in items
        ]
        valences = [
            getattr(item.experience, "emotional_valence", 0.0) or 0.0
            for item in items
        ]

        mean_imp = sum(importances) / len(importances)
        count = len(items)

        # Valence stability = 1 - std(valences), clamped 0..1
        if len(valences) > 1:
            mean_v = sum(valences) / len(valences)
            variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
            std_v = math.sqrt(variance)
        else:
            std_v = 0.0
        valence_stability = max(0.0, min(1.0, 1.0 - std_v))

        count_score = min(1.0, count / 10)
        trust = round(
            mean_imp * 0.4 + valence_stability * 0.4 + count_score * 0.2, 4
        )

        return TrustScore(
            source=source,
            trust=trust,
            memory_count=count,
            mean_importance=round(mean_imp, 4),
            valence_stability=round(valence_stability, 4),
        )

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
