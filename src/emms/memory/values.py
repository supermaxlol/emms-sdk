"""ValueMapper — extracting core values from accumulated memory.

v0.23.0: The Moral Mind

Values are the stable cognitive structures that orient behaviour across
situations — the things an agent cares about, aspires to, and uses to
evaluate what matters. They are distinct from fleeting preferences (which
are episodic) and from norms (which are social expectations). Values are
*personal* and *dispositional*: they persist across contexts, shape how
experiences are encoded, and constitute the agent's moral character.

ValueMapper operationalises value extraction for the memory store. It
maintains a lexicon of value-laden tokens organised into five categories
— epistemic, moral, aesthetic, instrumental, and social — and scans all
memory content for their occurrence. For each detected value token, it
computes a strength score that integrates frequency (how often the value
appears across memories) with the importance of the memories in which it
appears. Values that appear in many high-importance memories are strong;
values that appear rarely or only in low-importance memories are weak.

This is related to but distinct from NormExtractor (v0.21.0) — norms are
descriptive regularities detected by keyword pattern-matching, while values
are dispositional orientations extracted by cross-referencing a theoretically
grounded value lexicon with memory importance weighting.

Biological analogue: value-based decision-making in the orbitofrontal cortex
(Rangel, Camerer & Montague 2008); personal value structures as stable
cognitive schemas (Schwartz 1992); limbic system in emotional valuation
(LeDoux 1996); ventromedial PFC integrating value signals across dimensions
(Levy & Glimcher 2012); autobiographical self as value-laden identity
structure (Damasio 1999).
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Value lexicon — 5 categories × ~10 tokens each
# ---------------------------------------------------------------------------

_VALUE_LEXICON: dict[str, frozenset[str]] = {
    "epistemic": frozenset({
        "truth", "honest", "accuracy", "knowledge", "clarity",
        "understanding", "evidence", "certain", "verify", "transparency",
        "insight", "reason", "logic", "precise", "reliable",
    }),
    "moral": frozenset({
        "justice", "fairness", "harm", "care", "integrity",
        "virtue", "compassion", "dignity", "respect", "rights",
        "ethics", "moral", "values", "conscience", "principle",
    }),
    "aesthetic": frozenset({
        "beauty", "elegance", "meaning", "depth", "creativity",
        "craft", "coherence", "grace", "expression", "style",
        "artistry", "vision", "design", "nuance", "richness",
    }),
    "instrumental": frozenset({
        "growth", "efficiency", "progress", "learn", "improve",
        "build", "achieve", "produce", "develop", "succeed",
        "advance", "master", "optimise", "innovate", "solve",
    }),
    "social": frozenset({
        "trust", "loyalty", "cooperate", "community", "share",
        "belong", "relate", "connect", "support", "collaborate",
        "solidarity", "kinship", "reciprocity", "fellowship", "mutual",
    }),
}

# Reverse map: token → category
_TOKEN_TO_CATEGORY: dict[str, str] = {
    tok: cat
    for cat, tokens in _VALUE_LEXICON.items()
    for tok in tokens
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MappedValue:
    """A single core value extracted from the memory store."""

    id: str                       # prefixed "val_"
    name: str                     # the value token
    category: str                 # epistemic | moral | aesthetic | instrumental | social
    strength: float               # 0..1
    description: str
    source_memory_ids: list[str]
    created_at: float

    def summary(self) -> str:
        n = len(self.source_memory_ids)
        return (
            f"MappedValue [{self.category}]  strength={self.strength:.3f}  "
            f"'{self.name}'  ({n} memories)\n"
            f"  {self.id[:12]}: {self.description[:70]}"
        )


@dataclass
class ValueReport:
    """Result of a ValueMapper.map_values() call."""

    total_values: int
    values: list[MappedValue]      # sorted by strength descending
    dominant_category: str
    mean_strength: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"ValueReport: {self.total_values} values  "
            f"dominant={self.dominant_category}  "
            f"mean_strength={self.mean_strength:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for v in self.values[:5]:
            lines.append(f"  {v.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ValueMapper
# ---------------------------------------------------------------------------


class ValueMapper:
    """Extracts core values from memory by detecting value-laden vocabulary.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_strength:
        Minimum strength to include a value (default 0.2).
    max_values:
        Maximum number of :class:`MappedValue` objects to retain (default 20).
    """

    def __init__(
        self,
        memory: Any,
        min_strength: float = 0.2,
        max_values: int = 20,
    ) -> None:
        self.memory = memory
        self.min_strength = min_strength
        self.max_values = max_values
        self._values: list[MappedValue] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_values(self, category: Optional[str] = None) -> ValueReport:
        """Extract core values from all memories.

        Args:
            category: Restrict output to this value category (``None`` = all).

        Returns:
            :class:`ValueReport` with values sorted by strength descending.
        """
        t0 = time.time()
        all_items = self._collect_all()
        n_items = max(len(all_items), 1)

        # token → list of (memory_id, importance)
        token_occurrences: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for item in all_items:
            content = getattr(item.experience, "content", "") or ""
            importance = getattr(item.experience, "importance", 0.5) or 0.5
            for word in content.lower().split():
                tok = word.strip(".,!?;:\"'()")
                if tok in _TOKEN_TO_CATEGORY:
                    token_occurrences[tok].append((item.id, importance))

        all_values: list[MappedValue] = []
        for tok, occurrences in token_occurrences.items():
            cat = _TOKEN_TO_CATEGORY[tok]
            unique_ids = list(dict.fromkeys(mid for mid, _ in occurrences))
            mean_imp = sum(imp for _, imp in occurrences) / len(occurrences)
            freq_ratio = len(unique_ids) / n_items
            strength = round(min(1.0, mean_imp * freq_ratio * 5.0), 4)

            if strength < self.min_strength:
                continue

            description = (
                f"'{tok}' is a {cat} value appearing in "
                f"{len(unique_ids)} memories (mean_importance={mean_imp:.2f})"
            )
            all_values.append(MappedValue(
                id="val_" + uuid.uuid4().hex[:8],
                name=tok,
                category=cat,
                strength=strength,
                description=description,
                source_memory_ids=unique_ids[:10],
                created_at=time.time(),
            ))

        all_values.sort(key=lambda v: v.strength, reverse=True)

        # Filter by category if requested
        if category:
            all_values = [v for v in all_values if v.category == category]

        self._values = all_values[: self.max_values]

        # Dominant category
        cat_strengths: dict[str, float] = defaultdict(float)
        for v in self._values:
            cat_strengths[v.category] += v.strength
        dominant_category = (
            max(cat_strengths, key=cat_strengths.__getitem__)
            if cat_strengths else "none"
        )

        mean_s = (
            sum(v.strength for v in self._values) / len(self._values)
            if self._values else 0.0
        )

        return ValueReport(
            total_values=len(self._values),
            values=self._values,
            dominant_category=dominant_category,
            mean_strength=round(mean_s, 4),
            duration_seconds=time.time() - t0,
        )

    def values_for_category(self, category: str) -> list[MappedValue]:
        """Return all mapped values in a specific category.

        Args:
            category: One of epistemic, moral, aesthetic, instrumental, social.
        """
        return [v for v in self._values if v.category == category]

    def strongest_value(self) -> Optional[MappedValue]:
        """Return the single strongest value, or ``None`` if none mapped."""
        return self._values[0] if self._values else None

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
