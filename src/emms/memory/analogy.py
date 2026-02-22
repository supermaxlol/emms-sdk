"""AnalogyEngine — structural analogy detection across memory domains.

v0.17.0: The Goal-Directed Mind

Surface similarity asks "what looks alike?"; structural analogy asks
"what works alike?" — finding deep relational parallels between superficially
different memories. The AnalogyEngine detects cross-domain memory pairs that
share relational structure (causal, temporal, and enabling relations) rather
than topic overlap, synthesises a human-readable insight connecting them, and
optionally stores the insight as a new memory to make the analogy explicit.

Structural similarity
---------------------
Two memories are structurally analogous when they share *relational keywords*
— causal connectives ("causes", "leads", "because"), enabling terms
("enables", "requires", "prevents"), temporal markers ("follows", "produces"),
and quantitative relations ("increases", "reduces", "results"). These words
describe *how* things relate, not *what* they are.

Score: structural_similarity = 0.7 * relational_jaccard + 0.3 * content_jaccard

Cross-domain restriction: only pairs from *different* domains are considered,
because same-domain overlap is handled by SemanticDeduplicator / BeliefReviser.

Biological analogue: Structure Mapping Theory (Gentner 1983) — the hallmark
of analogical reasoning is alignment of relational structure, not surface
features; analogical reminding (Holyoak & Thagard 1995) — structural
similarity drives which past cases are retrieved; analogical transfer is a
primary mechanism of human insight and learning generalisation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Relational keyword vocabulary
# ---------------------------------------------------------------------------

_RELATIONAL_KEYWORDS: frozenset[str] = frozenset({
    "causes", "cause", "caused", "causing",
    "leads", "lead", "leading",
    "because", "since", "therefore", "hence", "thus",
    "prevents", "prevent", "blocked", "blocks",
    "enables", "enable", "allows", "allow",
    "requires", "require", "needs", "need",
    "follows", "follow", "followed", "after", "before",
    "produces", "produce", "generates", "generate",
    "reduces", "reduce", "decreases", "decrease",
    "increases", "increase", "amplifies", "amplify",
    "results", "result", "outcome", "outputs",
    "through", "despite", "without", "unless",
    "triggers", "trigger", "activates", "activate",
    "inhibits", "inhibit", "suppresses", "suppress",
})

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnalogyMapping:
    """Structural mapping between a source and target memory."""

    source_memory_id: str
    target_memory_id: str
    structural_similarity: float        # 0..1
    shared_relations: list[str]         # relational keywords found in both
    source_excerpt: str
    target_excerpt: str

    def summary(self) -> str:
        return (
            f"  {self.source_memory_id[:10]} ↔ {self.target_memory_id[:10]}  "
            f"sim={self.structural_similarity:.3f}  "
            f"relations=[{', '.join(self.shared_relations[:4])}]"
        )


@dataclass
class AnalogyRecord:
    """A detected cross-domain structural analogy."""

    id: str
    source_domain: str
    target_domain: str
    mappings: list[AnalogyMapping]
    analogy_strength: float             # mean structural_similarity over mappings
    insight_content: str
    new_memory_id: Optional[str]        # ID of stored insight memory (if any)
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"AnalogyRecord [{self.source_domain} ↔ {self.target_domain}]  "
            f"strength={self.analogy_strength:.3f}  "
            f"mappings={len(self.mappings)}\n"
            f"  Insight: {self.insight_content[:80]}"
        )


@dataclass
class AnalogyReport:
    """Result of an AnalogyEngine.find_analogies() call."""

    total_pairs_checked: int
    analogies_found: int
    records: list[AnalogyRecord]        # sorted by strength desc
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"AnalogyReport: {self.total_pairs_checked} pairs checked, "
            f"{self.analogies_found} analogies found in {self.duration_seconds:.2f}s",
        ]
        for r in self.records[:5]:
            lines.append(
                f"  [{r.source_domain}↔{r.target_domain}] "
                f"strength={r.analogy_strength:.3f}: {r.insight_content[:60]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AnalogyEngine
# ---------------------------------------------------------------------------


class AnalogyEngine:
    """Detects structural analogies across memory domains.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_structural_similarity:
        Minimum score for a pair to be considered analogous (default 0.25).
    max_analogies:
        Maximum :class:`AnalogyRecord` objects to return (default 6).
    store_insights:
        If ``True``, store each analogy as a new insight memory (default True).
    insight_importance:
        Importance assigned to stored insight memories (default 0.7).
    """

    def __init__(
        self,
        memory: Any,
        min_structural_similarity: float = 0.25,
        max_analogies: int = 6,
        store_insights: bool = True,
        insight_importance: float = 0.7,
    ) -> None:
        self.memory = memory
        self.min_structural_similarity = min_structural_similarity
        self.max_analogies = max_analogies
        self.store_insights = store_insights
        self.insight_importance = insight_importance
        self._records: list[AnalogyRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_analogies(
        self,
        source_domain: Optional[str] = None,
        target_domain: Optional[str] = None,
    ) -> AnalogyReport:
        """Find structural analogies across domains.

        Args:
            source_domain: Restrict source side to this domain (``None`` = all).
            target_domain: Restrict target side to this domain (``None`` = all).

        Returns:
            :class:`AnalogyReport` with found analogies sorted by strength.
        """
        t0 = time.time()
        items = self._collect_all()

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        domains = list(by_domain.keys())
        if len(domains) < 2:
            return AnalogyReport(
                total_pairs_checked=0,
                analogies_found=0,
                records=[],
                duration_seconds=time.time() - t0,
            )

        # Build cross-domain pairs
        checked = 0
        new_records: list[AnalogyRecord] = []

        for i, dom_a in enumerate(domains):
            if source_domain and dom_a != source_domain:
                continue
            items_a = by_domain[dom_a][:10]  # cap per domain

            for dom_b in domains[i + 1:]:
                if target_domain and dom_b != target_domain:
                    continue
                if source_domain and target_domain and dom_a == dom_b:
                    continue
                items_b = by_domain[dom_b][:10]

                # Find best-matching pairs between the two domains
                mappings: list[AnalogyMapping] = []
                for ia in items_a:
                    for ib in items_b:
                        checked += 1
                        sim, relations = self._structural_similarity(ia, ib)
                        if sim >= self.min_structural_similarity:
                            mappings.append(AnalogyMapping(
                                source_memory_id=ia.id,
                                target_memory_id=ib.id,
                                structural_similarity=round(sim, 4),
                                shared_relations=relations,
                                source_excerpt=ia.experience.content[:80],
                                target_excerpt=ib.experience.content[:80],
                            ))

                if not mappings:
                    continue

                # Sort mappings by similarity; take top-3
                mappings.sort(key=lambda m: m.structural_similarity, reverse=True)
                mappings = mappings[:3]

                strength = sum(m.structural_similarity for m in mappings) / len(mappings)
                record = self._build_analogy(dom_a, dom_b, mappings, strength)

                if self.store_insights:
                    record.new_memory_id = self._store_insight(record)

                new_records.append(record)
                self._records.append(record)

                if len(new_records) >= self.max_analogies:
                    break
            if len(new_records) >= self.max_analogies:
                break

        new_records.sort(key=lambda r: r.analogy_strength, reverse=True)

        return AnalogyReport(
            total_pairs_checked=checked,
            analogies_found=len(new_records),
            records=new_records,
            duration_seconds=time.time() - t0,
        )

    def analogies_for(self, memory_id: str) -> list[AnalogyRecord]:
        """Return all recorded analogies involving a specific memory.

        Args:
            memory_id: The memory ID to look up.

        Returns:
            List of :class:`AnalogyRecord` where any mapping references
            the given memory ID, sorted by analogy_strength descending.
        """
        result = []
        for record in self._records:
            for m in record.mappings:
                if m.source_memory_id == memory_id or m.target_memory_id == memory_id:
                    result.append(record)
                    break
        return sorted(result, key=lambda r: r.analogy_strength, reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _structural_similarity(
        self, item_a: Any, item_b: Any
    ) -> tuple[float, list[str]]:
        """Compute structural similarity and shared relational keywords."""
        text_a = getattr(item_a.experience, "content", "") or ""
        text_b = getattr(item_b.experience, "content", "") or ""

        rel_a = self._relational_keywords(text_a)
        rel_b = self._relational_keywords(text_b)
        shared_relations = sorted(rel_a & rel_b)

        rel_jaccard = 0.0
        if rel_a or rel_b:
            union = rel_a | rel_b
            rel_jaccard = len(rel_a & rel_b) / len(union) if union else 0.0

        content_a = self._content_tokens(text_a)
        content_b = self._content_tokens(text_b)
        content_jaccard = 0.0
        if content_a or content_b:
            union_c = content_a | content_b
            content_jaccard = len(content_a & content_b) / len(union_c) if union_c else 0.0

        sim = 0.7 * rel_jaccard + 0.3 * content_jaccard
        return round(sim, 5), shared_relations

    @staticmethod
    def _relational_keywords(text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()")
            for w in text.lower().split()
            if w.strip(".,!?;:\"'()") in _RELATIONAL_KEYWORDS
        }

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()")
            for w in text.lower().split()
            if len(w) >= 4 and w not in _STOP_WORDS and w not in _RELATIONAL_KEYWORDS
        }

    def _build_analogy(
        self,
        dom_a: str,
        dom_b: str,
        mappings: list[AnalogyMapping],
        strength: float,
    ) -> AnalogyRecord:
        """Construct a human-readable AnalogyRecord."""
        best = mappings[0]
        insight_content = (
            f"Structural analogy detected between {dom_a} and {dom_b}: "
            f"both exhibit the relational pattern [{', '.join(best.shared_relations[:4])}]. "
            f"In {dom_a}: \"{best.source_excerpt[:60]}\". "
            f"In {dom_b}: \"{best.target_excerpt[:60]}\". "
            f"This structural parallel may support cross-domain transfer."
        )
        return AnalogyRecord(
            id=f"ana_{uuid.uuid4().hex[:8]}",
            source_domain=dom_a,
            target_domain=dom_b,
            mappings=mappings,
            analogy_strength=round(strength, 4),
            insight_content=insight_content,
            new_memory_id=None,
        )

    def _store_insight(self, record: AnalogyRecord) -> Optional[str]:
        """Store the analogy as an insight memory."""
        new_id: Optional[str] = None
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=record.insight_content,
                domain="insight",
                importance=self.insight_importance,
                title=f"Analogy: {record.source_domain} ↔ {record.target_domain}",
                facts=[
                    f"Source domain: {record.source_domain}",
                    f"Target domain: {record.target_domain}",
                    f"Analogy strength: {record.analogy_strength:.3f}",
                ],
            )
            self.memory.store(exp)
            for item in list(self.memory.working):
                if item.experience.id == exp.id:
                    new_id = item.id
                    break
        except Exception:
            pass
        return new_id

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
