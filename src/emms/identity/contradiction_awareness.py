"""ContradictionAwareness — detect and measure belief conflicts in memory.

When the system holds opposing beliefs simultaneously, it experiences cognitive
dissonance.  This module quantifies that dissonance as a *coherence_strain*
score (0.0 = fully coherent, 1.0 = maximally strained).

The strain feeds directly into the CoherenceBudget (Phase 4), giving the system
something real to lose: if strain is high, the system must resolve contradictions
before acting from a place of integrity.

Usage::

    from emms.identity.contradiction_awareness import ContradictionAwareness

    awareness = ContradictionAwareness(emms)
    report = awareness.scan()
    print(f"Coherence strain: {report.coherence_strain:.2f}")
    for t in report.tensions[:3]:
        print(t.description)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeliefTension:
    """A pair of memories that contradict each other.

    Attributes
    ----------
    memory_id_a / memory_id_b:
        IDs of the two conflicting memories.
    content_a / content_b:
        Short text excerpts for display.
    domain:
        Shared domain label (or "cross-domain").
    semantic_overlap:
        Jaccard token overlap score (0-1) — how topically similar they are.
    valence_gap:
        |valence_a - valence_b| — emotional opposition.
    tension_score:
        Combined score: semantic_overlap × valence_gap (0-1).
    description:
        Human-readable summary of the conflict.
    """

    memory_id_a: str
    memory_id_b: str
    content_a: str
    content_b: str
    domain: str
    semantic_overlap: float
    valence_gap: float
    tension_score: float
    description: str


@dataclass
class ContradictionReport:
    """Full output of a contradiction scan.

    Attributes
    ----------
    tensions:
        List of detected BeliefTension objects, sorted by tension_score descending.
    coherence_strain:
        Aggregate strain score 0-1.  0 = fully coherent, 1 = maximally strained.
    memory_count_scanned:
        How many memories were examined.
    note:
        Human-readable interpretation of the strain level.
    """

    tensions: list[BeliefTension] = field(default_factory=list)
    coherence_strain: float = 0.0
    memory_count_scanned: int = 0
    note: str = ""


# ---------------------------------------------------------------------------
# ContradictionAwareness
# ---------------------------------------------------------------------------

class ContradictionAwareness:
    """Scans EMMS memory for contradicting belief pairs.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    overlap_threshold:
        Minimum Jaccard token overlap for two memories to be considered
        topically related (default 0.2).
    valence_threshold:
        Minimum |valence_a - valence_b| to count as emotionally opposed
        (default 0.4).
    max_pairs:
        Cap on pairs examined (guards against O(n²) blowup on large stores).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        overlap_threshold: float = 0.2,
        valence_threshold: float = 0.4,
        max_pairs: int = 2000,
    ) -> None:
        self.emms = emms
        self.overlap_threshold = overlap_threshold
        self.valence_threshold = valence_threshold
        self.max_pairs = max_pairs

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def scan(self, domain: str | None = None) -> ContradictionReport:
        """Scan memory and return a ContradictionReport.

        Parameters
        ----------
        domain:
            If provided, only scan memories within this domain.
        """
        items = self._collect_items(domain)
        if len(items) < 2:
            return ContradictionReport(
                memory_count_scanned=len(items),
                note="Not enough memories to detect contradictions.",
            )

        tensions = self._find_tensions(items)
        strain = self._compute_strain(tensions, len(items))
        note = self._interpret_strain(strain, len(tensions))

        return ContradictionReport(
            tensions=tensions,
            coherence_strain=strain,
            memory_count_scanned=len(items),
            note=note,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_items(self, domain: str | None) -> list[Any]:
        """Collect non-superseded, non-expired memory items."""
        items = []
        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    if item.is_superseded or item.is_expired:
                        continue
                    if domain and item.experience.domain != domain:
                        continue
                    items.append(item)
        except Exception as exc:
            logger.warning("ContradictionAwareness: error collecting items: %s", exc)
        return items

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        return set(re.findall(r"[a-z]{3,}", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _find_tensions(self, items: list[Any]) -> list[BeliefTension]:
        tensions: list[BeliefTension] = []
        pairs_checked = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pairs_checked >= self.max_pairs:
                    break

                a, b = items[i], items[j]
                v_a = getattr(a, "emotional_valence", 0.0) or 0.0
                v_b = getattr(b, "emotional_valence", 0.0) or 0.0
                valence_gap = abs(v_a - v_b)

                if valence_gap < self.valence_threshold:
                    pairs_checked += 1
                    continue

                content_a = a.experience.content or ""
                content_b = b.experience.content or ""
                tokens_a = self._tokenize(content_a)
                tokens_b = self._tokenize(content_b)
                overlap = self._jaccard(tokens_a, tokens_b)

                if overlap < self.overlap_threshold:
                    pairs_checked += 1
                    continue

                tension = overlap * valence_gap
                domain_a = a.experience.domain or "unknown"
                domain_b = b.experience.domain or "unknown"
                shared_domain = domain_a if domain_a == domain_b else f"{domain_a}↔{domain_b}"

                tensions.append(BeliefTension(
                    memory_id_a=a.memory_id,
                    memory_id_b=b.memory_id,
                    content_a=content_a[:120],
                    content_b=content_b[:120],
                    domain=shared_domain,
                    semantic_overlap=round(overlap, 3),
                    valence_gap=round(valence_gap, 3),
                    tension_score=round(tension, 3),
                    description=(
                        f"[{shared_domain}] Memories agree on topic (overlap={overlap:.2f}) "
                        f"but oppose emotionally (gap={valence_gap:.2f})"
                    ),
                ))
                pairs_checked += 1

            if pairs_checked >= self.max_pairs:
                logger.debug("ContradictionAwareness: hit max_pairs limit (%d)", self.max_pairs)
                break

        tensions.sort(key=lambda t: t.tension_score, reverse=True)
        return tensions[:50]  # Cap report at 50 worst tensions

    @staticmethod
    def _compute_strain(tensions: list[BeliefTension], n_items: int) -> float:
        """Aggregate tension scores into a 0-1 coherence strain.

        Uses a log-dampened average so a few extreme tensions don't
        immediately max out the score.
        """
        if not tensions or n_items == 0:
            return 0.0
        raw = sum(t.tension_score for t in tensions) / max(n_items, 1)
        # Sigmoid-like compression into [0, 1]
        strained = 1.0 - math.exp(-raw * 5)
        return round(min(strained, 1.0), 4)

    @staticmethod
    def _interpret_strain(strain: float, n_tensions: int) -> str:
        if strain < 0.05:
            return f"Highly coherent — {n_tensions} minor tensions detected."
        if strain < 0.2:
            return f"Mild strain ({n_tensions} tensions) — some beliefs diverge but nothing critical."
        if strain < 0.5:
            return f"Moderate strain ({n_tensions} tensions) — several belief conflicts need attention."
        if strain < 0.8:
            return f"High strain ({n_tensions} tensions) — contradictions are significantly impacting coherence."
        return f"Critical strain ({n_tensions} tensions) — belief system is severely inconsistent."
