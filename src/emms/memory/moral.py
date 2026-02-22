"""MoralReasoner — evaluating experiences through ethical frameworks.

v0.23.0: The Moral Mind

Moral cognition is not a single unified system but the product of multiple
interacting evaluative frameworks. Classical normative ethics identifies three
dominant approaches: consequentialism evaluates actions by their outcomes and
effects on welfare; deontology evaluates actions by adherence to duties,
rights, and rules; virtue ethics evaluates actions by whether they express or
cultivate excellent character. Real moral judgment typically activates more
than one framework, and the tension between them is the source of ethical
dilemmas.

MoralReasoner operationalises multi-framework evaluation for the memory store:
it scans each memory's content for tokens associated with each framework and
computes a framework score proportional to keyword density. The dominant
framework is whichever scores highest. Moral weight integrates the memory's
importance and emotional valence with the peak framework score, producing a
single moral salience measure that captures how morally charged an experience
is, regardless of which framework is primary.

This is distinct from NormExtractor (v0.21.0) — norms are descriptive social
regularities, while moral assessments evaluate individual experiences against
normative frameworks. It is also distinct from ValueMapper (v0.23.0) —
values are stable dispositional orientations, while moral assessments are
per-experience evaluations.

Biological analogue: moral judgment in vmPFC and TPJ (Greene et al. 2001);
dual-process moral cognition — fast emotional vs slow deliberative (Greene
2008); moral foundations theory as domain-general moral grammar (Haidt 2012);
emotional and rule-based moral reasoning as independent systems (Cushman 2013);
ACC in conflict monitoring between competing moral imperatives.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Framework keyword sets
# ---------------------------------------------------------------------------

_CONSEQUENTIALIST: frozenset[str] = frozenset({
    "result", "consequence", "outcome", "benefit", "harm", "cost", "effect",
    "impact", "welfare", "utility", "produce", "leads", "improves", "damages",
    "reduces", "increases", "total", "net", "greater", "best",
})

_DEONTOLOGICAL: frozenset[str] = frozenset({
    "must", "duty", "obligation", "right", "principle", "forbidden", "rule",
    "justice", "owed", "required", "prohibited", "ought", "mandate", "rights",
    "respect", "dignity", "never", "always", "binding", "categorical",
})

_VIRTUE: frozenset[str] = frozenset({
    "honest", "brave", "just", "kind", "wise", "virtuous", "excellence",
    "character", "integrity", "noble", "courageous", "compassion", "generous",
    "humble", "patient", "prudent", "temperate", "loyal", "faithful", "grace",
})

_FRAMEWORK_SETS: dict[str, frozenset[str]] = {
    "consequentialist": _CONSEQUENTIALIST,
    "deontological": _DEONTOLOGICAL,
    "virtue": _VIRTUE,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MoralAssessment:
    """Moral evaluation of a single memory item."""

    memory_id: str
    content_excerpt: str          # first 80 chars of content
    consequentialist_score: float  # 0..1
    deontological_score: float     # 0..1
    virtue_score: float            # 0..1
    dominant_framework: str        # "consequentialist"|"deontological"|"virtue"|"none"
    moral_weight: float            # 0..1
    domain: str

    def summary(self) -> str:
        return (
            f"MoralAssessment [{self.domain}]  weight={self.moral_weight:.3f}  "
            f"dominant={self.dominant_framework}\n"
            f"  C={self.consequentialist_score:.3f}  "
            f"D={self.deontological_score:.3f}  "
            f"V={self.virtue_score:.3f}\n"
            f"  '{self.content_excerpt[:60]}'"
        )


@dataclass
class MoralReport:
    """Result of a MoralReasoner.reason() call."""

    total_assessed: int
    assessments: list[MoralAssessment]   # sorted by moral_weight desc
    dominant_framework_overall: str
    mean_moral_weight: float
    framework_counts: dict[str, int]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"MoralReport: {self.total_assessed} assessed  "
            f"dominant={self.dominant_framework_overall}  "
            f"mean_weight={self.mean_moral_weight:.3f}  "
            f"in {self.duration_seconds:.2f}s",
            f"  Framework counts: {self.framework_counts}",
        ]
        for a in self.assessments[:5]:
            lines.append(f"  {a.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MoralReasoner
# ---------------------------------------------------------------------------


class MoralReasoner:
    """Evaluates memories through three classical ethical frameworks.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_moral_weight:
        Minimum moral weight to include an assessment (default 0.05).
    """

    def __init__(
        self,
        memory: Any,
        min_moral_weight: float = 0.05,
    ) -> None:
        self.memory = memory
        self.min_moral_weight = min_moral_weight
        self._assessments: dict[str, MoralAssessment] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(self, domain: Optional[str] = None) -> MoralReport:
        """Evaluate all memories through ethical frameworks.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`MoralReport` with assessments sorted by moral_weight desc.
        """
        t0 = time.time()
        self._assessments.clear()

        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        assessments: list[MoralAssessment] = []
        for item in items:
            a = self._score_memory(item)
            if a.moral_weight >= self.min_moral_weight:
                self._assessments[item.id] = a
                assessments.append(a)

        assessments.sort(key=lambda a: a.moral_weight, reverse=True)

        # Framework counts
        framework_counts: dict[str, int] = {
            "consequentialist": 0,
            "deontological": 0,
            "virtue": 0,
            "none": 0,
        }
        for a in assessments:
            framework_counts[a.dominant_framework] += 1

        # Dominant overall
        dominant_overall = max(
            (k for k in ("consequentialist", "deontological", "virtue")),
            key=lambda k: framework_counts[k],
            default="none",
        ) if assessments else "none"
        if framework_counts.get(dominant_overall, 0) == 0:
            dominant_overall = "none"

        mean_w = (
            sum(a.moral_weight for a in assessments) / len(assessments)
            if assessments else 0.0
        )

        return MoralReport(
            total_assessed=len(assessments),
            assessments=assessments,
            dominant_framework_overall=dominant_overall,
            mean_moral_weight=round(mean_w, 4),
            framework_counts=framework_counts,
            duration_seconds=time.time() - t0,
        )

    def assessments_by_framework(
        self, framework: str
    ) -> list[MoralAssessment]:
        """Return all assessments dominated by the given framework.

        Args:
            framework: One of consequentialist, deontological, virtue, none.

        Returns:
            List of :class:`MoralAssessment` sorted by moral_weight descending.
        """
        return sorted(
            [a for a in self._assessments.values() if a.dominant_framework == framework],
            key=lambda a: a.moral_weight,
            reverse=True,
        )

    def moral_weight_of(self, memory_id: str) -> float:
        """Return the moral weight of a specific memory.

        Args:
            memory_id: ID of the memory to query.

        Returns:
            Moral weight 0..1, or 0.0 if not assessed.
        """
        if memory_id in self._assessments:
            return self._assessments[memory_id].moral_weight
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_memory(self, item: Any) -> MoralAssessment:
        """Compute moral assessment for a single memory item."""
        content = getattr(item.experience, "content", "") or ""
        importance = getattr(item.experience, "importance", 0.5) or 0.5
        valence = getattr(item.experience, "emotional_valence", 0.0) or 0.0
        domain = getattr(item.experience, "domain", None) or "general"

        words = content.lower().split()
        tokens = [w.strip(".,!?;:\"'()") for w in words]
        n_tokens = max(len(tokens), 1)

        scores: dict[str, float] = {}
        for fw, kw_set in _FRAMEWORK_SETS.items():
            matches = sum(1 for tok in tokens if tok in kw_set)
            scores[fw] = round(matches / n_tokens, 4)

        max_score = max(scores.values()) if scores else 0.0

        # Dominant framework
        if max_score < 0.01:
            dominant = "none"
        else:
            dominant = max(scores, key=scores.__getitem__)

        # Moral weight: importance × |valence| × (max_score + 0.1), clamped 0..1
        moral_weight = min(1.0, importance * abs(valence) * (max_score + 0.1))
        moral_weight = round(moral_weight, 4)

        return MoralAssessment(
            memory_id=item.id,
            content_excerpt=content[:80],
            consequentialist_score=scores.get("consequentialist", 0.0),
            deontological_score=scores.get("deontological", 0.0),
            virtue_score=scores.get("virtue", 0.0),
            dominant_framework=dominant,
            moral_weight=moral_weight,
            domain=domain,
        )

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
