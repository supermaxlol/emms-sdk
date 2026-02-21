"""ImportanceClassifier — automatic importance scoring from content signals.

Infers an importance score (0–1) from six weighted signals without requiring
any external API or ML model.  The result enriches ``Experience.importance``
so that retrieval can rank memories appropriately even when the caller doesn't
supply an explicit score.

Signal breakdown
----------------
1. **entity_density** (0.20) — capitalised-token fraction; named entities matter.
2. **novelty_passthrough** (0.20) — honours the caller's ``experience.novelty``.
3. **emotional_weight** (0.15) — ``|valence| * 0.5 + intensity * 0.5``.
4. **length_score** (0.10) — saturates at ~200 words (very long ≈ substantive).
5. **keyword_score** (0.20) — fraction of high-stakes terms present.
6. **structure_score** (0.15) — title (+0.4), each fact (+0.1), citations (+0.2).

Usage::

    from emms.core.importance import ImportanceClassifier

    clf = ImportanceClassifier()
    experience.importance = clf.score(experience)

Or simply call ``clf.enrich(experience)`` to set the field in-place.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.core.models import Experience

# ---------------------------------------------------------------------------
# High-stakes keyword vocabulary
# ---------------------------------------------------------------------------

_HIGH_STAKES: frozenset[str] = frozenset({
    # Urgency / priority
    "critical", "urgent", "important", "essential", "crucial", "vital",
    "emergency", "alert", "warning", "error", "failure", "bug", "crash",
    # Decisions
    "decision", "decided", "resolved", "agreed", "committed", "approved",
    "rejected", "cancelled", "blocked",
    # Knowledge creation
    "discovered", "learned", "found", "realized", "understood", "identified",
    "determined", "confirmed", "verified", "proven",
    # System / tech
    "security", "vulnerability", "exploit", "breach", "authentication",
    "deadline", "release", "deploy", "rollback", "migration", "breaking",
    # Impact
    "impact", "affect", "change", "update", "fix", "patch", "regression",
    "performance", "memory", "leak", "bottleneck",
})

# Token pattern for capitalised words (rough NER proxy)
_CAP_WORD = re.compile(r'\b[A-Z][a-zA-Z]{2,}\b')
# Sentence splitter
_SENT_SPLIT = re.compile(r'[.!?]+')


class ImportanceClassifier:
    """Scores experience importance from six content-derived signals.

    Parameters
    ----------
    auto_enrich:
        If True, ``enrich()`` always overwrites ``experience.importance``
        even if it was already set to a non-default value.
        Default False — only overwrite if importance == 0.5 (the default).
    weights:
        Override the six signal weights.  Must sum to 1.0.
        Keys: ``entity``, ``novelty``, ``emotional``, ``length``,
        ``keyword``, ``structure``.
    """

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "entity": 0.20,
        "novelty": 0.20,
        "emotional": 0.15,
        "length": 0.10,
        "keyword": 0.20,
        "structure": 0.15,
    }

    def __init__(
        self,
        auto_enrich: bool = False,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.auto_enrich = auto_enrich
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, experience: "Experience") -> float:
        """Return importance score 0–1 derived from six signals."""
        w = self.weights
        text = experience.content

        entity_s = self._entity_score(text)
        novelty_s = float(experience.novelty)
        emotional_s = self._emotional_score(experience)
        length_s = self._length_score(text)
        keyword_s = self._keyword_score(text)
        structure_s = self._structure_score(experience)

        raw = (
            entity_s    * w.get("entity", 0.20)
            + novelty_s * w.get("novelty", 0.20)
            + emotional_s * w.get("emotional", 0.15)
            + length_s  * w.get("length", 0.10)
            + keyword_s * w.get("keyword", 0.20)
            + structure_s * w.get("structure", 0.15)
        )
        return float(min(1.0, max(0.0, raw)))

    def score_breakdown(self, experience: "Experience") -> dict[str, float]:
        """Return per-signal scores for debugging / explanation."""
        text = experience.content
        return {
            "entity":    self._entity_score(text),
            "novelty":   float(experience.novelty),
            "emotional": self._emotional_score(experience),
            "length":    self._length_score(text),
            "keyword":   self._keyword_score(text),
            "structure": self._structure_score(experience),
            "total":     self.score(experience),
        }

    def enrich(self, experience: "Experience") -> "Experience":
        """Set ``experience.importance`` in-place and return the experience.

        Only overwrites if ``auto_enrich=True`` OR the importance is still at
        the default value of 0.5 (meaning the caller didn't customise it).
        """
        if self.auto_enrich or experience.importance == 0.5:
            experience.importance = self.score(experience)
        return experience

    # ------------------------------------------------------------------
    # Signal implementations
    # ------------------------------------------------------------------

    def _entity_score(self, text: str) -> float:
        """Fraction of tokens that are capitalised (proxy for named entities)."""
        tokens = text.split()
        if not tokens:
            return 0.0
        cap_count = sum(1 for t in tokens if t and t[0].isupper() and len(t) > 2)
        # Saturate at ~30 % capitalised — very dense capitalisation ≈ 1.0
        return min(1.0, cap_count / max(len(tokens) * 0.3, 1))

    def _emotional_score(self, experience: "Experience") -> float:
        """Emotional weight from valence + intensity."""
        valence = abs(float(experience.emotional_valence))  # 0–1
        intensity = float(experience.emotional_intensity)   # 0–1
        return min(1.0, valence * 0.5 + intensity * 0.5)

    def _length_score(self, text: str) -> float:
        """Score based on content length — saturates at 200 words."""
        word_count = len(text.split())
        return min(1.0, word_count / 200.0)

    def _keyword_score(self, text: str) -> float:
        """Fraction of high-stakes keywords present in text."""
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        if not words:
            return 0.0
        hits = words & _HIGH_STAKES
        # Normalise: 3+ hits → 1.0 (diminishing returns)
        return min(1.0, len(hits) / 3.0)

    def _structure_score(self, experience: "Experience") -> float:
        """Bonus for structured content: title, facts, citations."""
        score = 0.0
        if experience.title:
            score += 0.4
        score += min(0.4, len(experience.facts) * 0.1)
        if experience.citations:
            score += 0.2
        return min(1.0, score)
