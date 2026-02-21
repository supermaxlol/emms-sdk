"""MetacognitionEngine — epistemic confidence and self-knowledge assessment.

v0.13.0: The Metacognitive Layer

The MetacognitionEngine gives the agent a model of its own knowledge state.
It assesses how confident the agent should be in each stored memory, identifies
domains where knowledge is sparse or stale, and flags pairs of memories that
appear to contradict each other.

Biological analogue: metacognitive monitoring — the "feeling of knowing", the
tip-of-the-tongue state, knowing what you don't know. Damage to prefrontal
metacognitive circuits produces confabulation: confident assertions about things
the agent has no valid memory of. Healthy metacognition is the guard against this.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfidence:
    """Epistemic confidence assessment for a single memory item."""

    memory_id: str
    confidence: float            # [0,1] overall epistemic confidence
    strength_factor: float       # from item.memory_strength
    recency_factor: float        # from time since last access
    access_factor: float         # from item.access_count
    consolidation_factor: float  # from item.consolidation_score
    age_days: float              # age in days since storage


@dataclass
class DomainProfile:
    """Knowledge profile for a single domain."""

    domain: str
    memory_count: int
    mean_confidence: float
    coverage_score: float        # memory_count / total_memories
    mean_importance: float
    mean_strength: float


@dataclass
class ContradictionPair:
    """Two memories with semantic overlap but conflicting emotional valence."""

    memory_a_id: str
    memory_b_id: str
    semantic_overlap: float      # token overlap score [0,1]
    valence_conflict: float      # |valence_a − valence_b|
    contradiction_score: float   # combined signal
    excerpt_a: str
    excerpt_b: str


@dataclass
class MetacognitionReport:
    """Comprehensive metacognitive self-assessment."""

    total_memories: int
    mean_confidence: float
    high_confidence_count: int   # confidence > threshold_high
    low_confidence_count: int    # confidence < threshold_low
    domain_profiles: list[DomainProfile]
    contradictions: list[ContradictionPair]
    knowledge_gaps: list[str]    # domains with low coverage or confidence
    recommendations: list[str]   # actionable suggestions
    generated_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            f"Metacognition Report ({self.total_memories} memories)",
            f"Mean confidence: {self.mean_confidence:.3f}   "
            f"High: {self.high_confidence_count}   Low: {self.low_confidence_count}",
        ]
        if self.domain_profiles:
            lines.append("Domain profiles:")
            for dp in sorted(self.domain_profiles, key=lambda d: d.mean_confidence, reverse=True)[:5]:
                lines.append(
                    f"  [{dp.domain}] n={dp.memory_count}  "
                    f"conf={dp.mean_confidence:.2f}  "
                    f"imp={dp.mean_importance:.2f}"
                )
        if self.contradictions:
            lines.append(f"Contradictions detected: {len(self.contradictions)}")
            for c in self.contradictions[:2]:
                lines.append(
                    f"  score={c.contradiction_score:.2f}  "
                    f'"{c.excerpt_a[:50]}" ↔ "{c.excerpt_b[:50]}"'
                )
        if self.knowledge_gaps:
            lines.append(f"Knowledge gaps: {', '.join(self.knowledge_gaps[:5])}")
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations[:3]:
                lines.append(f"  • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MetacognitionEngine
# ---------------------------------------------------------------------------

class MetacognitionEngine:
    """Assesses epistemic confidence and knowledge structure across all memories.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    recency_decay:
        Daily exponential decay factor for recency scoring (default 0.05 →
        half-confidence after ~14 days of no access).
    access_saturation:
        Access count at which access_factor saturates to 1.0 (default 5).
    confidence_threshold_high:
        Memories above this confidence are "high confidence" (default 0.65).
    confidence_threshold_low:
        Memories below this are "low confidence" / candidates for gaps
        (default 0.3).
    contradiction_overlap_min:
        Minimum token overlap to consider a pair potentially contradictory
        (default 0.35).
    contradiction_valence_min:
        Minimum |valence_a − valence_b| to flag as contradictory (default 0.5).
    """

    def __init__(
        self,
        memory: Any,
        recency_decay: float = 0.05,
        access_saturation: float = 5.0,
        confidence_threshold_high: float = 0.65,
        confidence_threshold_low: float = 0.3,
        contradiction_overlap_min: float = 0.35,
        contradiction_valence_min: float = 0.5,
    ) -> None:
        self.memory = memory
        self.recency_decay = recency_decay
        self.access_saturation = access_saturation
        self.threshold_high = confidence_threshold_high
        self.threshold_low = confidence_threshold_low
        self.contradiction_overlap_min = contradiction_overlap_min
        self.contradiction_valence_min = contradiction_valence_min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, item: Any) -> MemoryConfidence:
        """Compute epistemic confidence for a single memory item.

        Confidence is a weighted geometric mean of four factors:

        - **strength** — normalised ``memory_strength`` (0..2 → 0..1)
        - **recency** — exponential decay on days since last access
        - **access** — saturating function of ``access_count``
        - **consolidation** — ``consolidation_score`` directly

        Args:
            item: A :class:`MemoryItem` instance.

        Returns:
            :class:`MemoryConfidence` with factor breakdown.
        """
        now = time.time()

        strength_factor = min(1.0, item.memory_strength / 2.0)

        days_since_access = max(0.0, (now - item.last_accessed) / 86400.0)
        recency_factor = math.exp(-self.recency_decay * days_since_access)

        access_factor = min(1.0, item.access_count / max(self.access_saturation, 1.0))

        consolidation_factor = float(item.consolidation_score)

        age_days = max(0.0, (now - item.stored_at) / 86400.0)

        # Weighted geometric mean (weights sum to 1.0)
        w_s, w_r, w_a, w_c = 0.35, 0.25, 0.20, 0.20
        eps = 1e-9
        confidence = math.exp(
            w_s * math.log(strength_factor + eps)
            + w_r * math.log(recency_factor + eps)
            + w_a * math.log(access_factor + eps)
            + w_c * math.log(consolidation_factor + eps)
        )
        confidence = max(0.0, min(1.0, confidence))

        return MemoryConfidence(
            memory_id=item.id,
            confidence=confidence,
            strength_factor=strength_factor,
            recency_factor=recency_factor,
            access_factor=access_factor,
            consolidation_factor=consolidation_factor,
            age_days=age_days,
        )

    def assess_all(self) -> list[MemoryConfidence]:
        """Assess epistemic confidence for every stored memory.

        Returns:
            List of :class:`MemoryConfidence` sorted by confidence descending.
        """
        items = self._collect_all()
        results = [self.assess(item) for item in items]
        results.sort(key=lambda c: c.confidence, reverse=True)
        return results

    def knowledge_map(self) -> list[DomainProfile]:
        """Build a per-domain knowledge profile.

        Returns:
            List of :class:`DomainProfile` sorted by memory_count descending.
        """
        items = self._collect_all()
        if not items:
            return []

        by_domain: dict[str, list[Any]] = defaultdict(list)
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain[dom].append(item)

        profiles: list[DomainProfile] = []
        for dom, dom_items in by_domain.items():
            confidences = [self.assess(it).confidence for it in dom_items]
            profiles.append(
                DomainProfile(
                    domain=dom,
                    memory_count=len(dom_items),
                    mean_confidence=sum(confidences) / len(confidences),
                    coverage_score=len(dom_items) / max(len(items), 1),
                    mean_importance=sum(
                        it.experience.importance for it in dom_items
                    ) / len(dom_items),
                    mean_strength=sum(
                        it.memory_strength for it in dom_items
                    ) / len(dom_items),
                )
            )

        profiles.sort(key=lambda p: p.memory_count, reverse=True)
        return profiles

    def find_contradictions(
        self,
        max_pairs: int = 10,
    ) -> list[ContradictionPair]:
        """Find memory pairs with semantic overlap but conflicting valence.

        Two memories are potentially contradictory if they share significant
        content tokens (implying they're about the same topic) but have
        strongly opposing emotional valence.

        Args:
            max_pairs: Maximum contradiction pairs to return (default 10).

        Returns:
            List of :class:`ContradictionPair` sorted by contradiction_score.
        """
        items = self._collect_all()
        pairs: list[ContradictionPair] = []
        seen: set[frozenset] = set()

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                pair_key: frozenset = frozenset({a.id, b.id})
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                overlap = self._token_overlap(
                    a.experience.content, b.experience.content
                )
                if overlap < self.contradiction_overlap_min:
                    continue

                va = getattr(a.experience, "emotional_valence", 0.0) or 0.0
                vb = getattr(b.experience, "emotional_valence", 0.0) or 0.0
                valence_conflict = abs(va - vb)
                if valence_conflict < self.contradiction_valence_min:
                    continue

                contradiction_score = overlap * valence_conflict
                pairs.append(
                    ContradictionPair(
                        memory_a_id=a.id,
                        memory_b_id=b.id,
                        semantic_overlap=overlap,
                        valence_conflict=valence_conflict,
                        contradiction_score=contradiction_score,
                        excerpt_a=a.experience.content[:80],
                        excerpt_b=b.experience.content[:80],
                    )
                )

        pairs.sort(key=lambda p: p.contradiction_score, reverse=True)
        return pairs[:max_pairs]

    def find_gaps(self) -> list[str]:
        """Identify domains / topics with low confidence or sparse coverage.

        Returns a list of domain names (or "general") where the mean
        confidence is below ``threshold_low`` or coverage is < 5%.

        Returns:
            List of domain name strings.
        """
        profiles = self.knowledge_map()
        if not profiles:
            return []
        gaps = [
            p.domain
            for p in profiles
            if p.mean_confidence < self.threshold_low or p.coverage_score < 0.05
        ]
        return gaps

    def report(self, max_contradictions: int = 5) -> MetacognitionReport:
        """Generate a comprehensive metacognitive self-assessment report.

        Args:
            max_contradictions: Maximum contradiction pairs to include.

        Returns:
            :class:`MetacognitionReport`.
        """
        confidences = self.assess_all()
        domain_profiles = self.knowledge_map()
        contradictions = self.find_contradictions(max_pairs=max_contradictions)
        gaps = self.find_gaps()

        high_count = sum(1 for c in confidences if c.confidence >= self.threshold_high)
        low_count = sum(1 for c in confidences if c.confidence < self.threshold_low)
        mean_conf = (
            sum(c.confidence for c in confidences) / max(len(confidences), 1)
        )

        recommendations: list[str] = []
        if low_count > 0:
            recommendations.append(
                f"Reinforce {low_count} low-confidence memories via spaced repetition."
            )
        if contradictions:
            recommendations.append(
                f"Review {len(contradictions)} contradictory memory pair(s) for resolution."
            )
        if gaps:
            recommendations.append(
                f"Expand knowledge in: {', '.join(gaps[:3])}."
            )
        if not recommendations:
            recommendations.append("Knowledge state is healthy — continue normal operation.")

        return MetacognitionReport(
            total_memories=len(confidences),
            mean_confidence=mean_conf,
            high_confidence_count=high_count,
            low_confidence_count=low_count,
            domain_profiles=domain_profiles,
            contradictions=contradictions,
            knowledge_gaps=gaps,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Jaccard-like token overlap between two texts."""
        toks_a = set(text_a.lower().split())
        toks_b = set(text_b.lower().split())
        if not toks_a or not toks_b:
            return 0.0
        intersection = toks_a & toks_b
        union = toks_a | toks_b
        return len(intersection) / max(len(union), 1)
