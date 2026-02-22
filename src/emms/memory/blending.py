"""ConceptBlender — conceptual integration of memory pairs into novel blends.

v0.18.0: The Predictive Mind

Creative insight often emerges not from a single idea but from the unexpected
fusion of two. Conceptual blending (Fauconnier & Turner 2002) is the cognitive
mechanism behind metaphor, analogy, and creative synthesis: two mental spaces
(input spaces) are partially projected into a blended space that inherits
selective structure from each and develops emergent properties unique to the
blend.

The ConceptBlender operationalises this for the memory store. It pairs
memories (within or across domains), identifies their shared structure
(overlap tokens) and complementary structure (unique tokens from each side),
synthesises a blend_content string, extracts emergent properties (the novel
combinations), and optionally stores the result as a new insight memory.

Blend strength
--------------
blend_strength = (shared / total_tokens) * coherence
coherence      = min(1, importance_a * importance_b * 4)   — both must be
                 meaningful memories to produce a strong blend

Emergent properties are the unique tokens from each side that don't appear in
the other — these represent structure that is selectively projected from each
input space and creates novelty in the blend.

Biological analogue: conceptual integration theory (Fauconnier & Turner 2002)
— blending is a basic mental operation underlying metaphor, counterfactuals,
humour, and creative thought; frame blending in the construction of novel
meanings (Coulson 2001); creative insight as sudden integration of previously
disconnected structures (Dijksterhuis & Meurs 2006).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BlendedConcept:
    """A novel concept created by blending two memory items."""

    id: str
    source_memory_ids: list[str]    # the two input memories
    source_domains: list[str]
    blend_content: str              # synthesised blend description
    emergent_properties: list[str]  # novel tokens unique to the blend
    blend_strength: float           # 0..1 — coherence of the blend
    new_memory_id: Optional[str]    # ID of stored insight memory (if any)
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        domains_str = " + ".join(self.source_domains)
        return (
            f"BlendedConcept [{domains_str}]  strength={self.blend_strength:.3f}\n"
            f"  {self.id[:12]}: {self.blend_content[:80]}\n"
            f"  Emergent: [{', '.join(self.emergent_properties[:5])}]"
        )


@dataclass
class BlendReport:
    """Result of a ConceptBlender.blend() call."""

    total_pairs_tried: int
    blends_created: int
    concepts: list[BlendedConcept]  # sorted by blend_strength desc
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"BlendReport: {self.total_pairs_tried} pairs tried, "
            f"{self.blends_created} blends created in {self.duration_seconds:.2f}s",
        ]
        for c in self.concepts[:5]:
            domains = " + ".join(c.source_domains)
            lines.append(
                f"  [{domains}] str={c.blend_strength:.3f}: {c.blend_content[:60]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConceptBlender
# ---------------------------------------------------------------------------


class ConceptBlender:
    """Blends pairs of memories into novel conceptual syntheses.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_blend_strength:
        Minimum blend_strength to accept a blend (default 0.15).
    max_blends:
        Maximum :class:`BlendedConcept` objects to generate (default 6).
    store_blends:
        If ``True``, store each blend as a new memory (default True).
    blend_importance:
        Importance of stored blend memories (default 0.6).
    """

    def __init__(
        self,
        memory: Any,
        min_blend_strength: float = 0.15,
        max_blends: int = 6,
        store_blends: bool = True,
        blend_importance: float = 0.6,
    ) -> None:
        self.memory = memory
        self.min_blend_strength = min_blend_strength
        self.max_blends = max_blends
        self.store_blends = store_blends
        self.blend_importance = blend_importance
        self._blends: list[BlendedConcept] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        domain_a: Optional[str] = None,
        domain_b: Optional[str] = None,
    ) -> BlendReport:
        """Generate conceptual blends from memory pairs.

        When both ``domain_a`` and ``domain_b`` are ``None``, blends memories
        across *all* domain pairs. When one is given, uses only memories from
        that domain on one side. When both are given, restricts to only those
        two domains.

        Args:
            domain_a: Source domain for one side (``None`` = any domain).
            domain_b: Source domain for other side (``None`` = any domain).

        Returns:
            :class:`BlendReport` with generated blends sorted by strength.
        """
        t0 = time.time()
        items = self._collect_all()

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        domains = list(by_domain.keys())
        tried = 0
        new_blends: list[BlendedConcept] = []

        # Build candidate pairs
        for i, dom_i in enumerate(domains):
            if domain_a and dom_i != domain_a:
                continue
            items_i = by_domain[dom_i][:8]

            for j, dom_j in enumerate(domains):
                if j <= i and domain_a != domain_b:
                    continue  # avoid duplicates unless same domain requested
                if domain_b and dom_j != domain_b:
                    continue
                items_j = by_domain[dom_j][:8]

                for ia in items_i:
                    for ib in items_j:
                        if ia.id == ib.id:
                            continue
                        tried += 1
                        concept = self._make_blend(ia, ib)
                        if concept is not None and concept.blend_strength >= self.min_blend_strength:
                            if self.store_blends:
                                concept.new_memory_id = self._store_blend(concept)
                            new_blends.append(concept)
                            self._blends.append(concept)
                        if len(new_blends) >= self.max_blends:
                            break
                    if len(new_blends) >= self.max_blends:
                        break
                if len(new_blends) >= self.max_blends:
                    break
            if len(new_blends) >= self.max_blends:
                break

        new_blends.sort(key=lambda c: c.blend_strength, reverse=True)

        return BlendReport(
            total_pairs_tried=tried,
            blends_created=len(new_blends),
            concepts=new_blends,
            duration_seconds=time.time() - t0,
        )

    def blend_pair(
        self,
        memory_id_a: str,
        memory_id_b: str,
    ) -> Optional[BlendedConcept]:
        """Blend a specific pair of memories by ID.

        Args:
            memory_id_a: ID of the first memory.
            memory_id_b: ID of the second memory.

        Returns:
            :class:`BlendedConcept` or ``None`` if either memory not found.
        """
        items = self._collect_all()
        item_a = next((it for it in items if it.id == memory_id_a), None)
        item_b = next((it for it in items if it.id == memory_id_b), None)
        if item_a is None or item_b is None:
            return None

        concept = self._make_blend(item_a, item_b)
        if concept is not None and self.store_blends:
            concept.new_memory_id = self._store_blend(concept)
        return concept

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_blend(self, item_a: Any, item_b: Any) -> Optional[BlendedConcept]:
        """Attempt to blend two memories; return None if below threshold."""
        text_a = getattr(item_a.experience, "content", "") or ""
        text_b = getattr(item_b.experience, "content", "") or ""
        dom_a = getattr(item_a.experience, "domain", None) or "general"
        dom_b = getattr(item_b.experience, "domain", None) or "general"

        tokens_a = self._tokenise(text_a)
        tokens_b = self._tokenise(text_b)

        if not tokens_a or not tokens_b:
            return None

        shared = tokens_a & tokens_b
        unique_a = tokens_a - tokens_b
        unique_b = tokens_b - tokens_a
        total = tokens_a | tokens_b

        shared_ratio = len(shared) / max(len(total), 1)

        imp_a = min(1.0, max(0.0, getattr(item_a, "memory_strength", 0.5)))
        imp_b = min(1.0, max(0.0, getattr(item_b, "memory_strength", 0.5)))
        coherence = min(1.0, imp_a * imp_b * 4)
        blend_strength = round(shared_ratio * coherence + 0.05 * min(len(unique_a), 3) * 0.1, 4)
        blend_strength = min(1.0, blend_strength)

        # Emergent properties: unique tokens from each side (selective projection)
        emergent = sorted(unique_a)[:3] + sorted(unique_b)[:3]

        blend_content = self._generate_blend_text(
            dom_a, dom_b, text_a, text_b, shared, emergent
        )

        return BlendedConcept(
            id=f"blend_{uuid.uuid4().hex[:8]}",
            source_memory_ids=[item_a.id, item_b.id],
            source_domains=[dom_a, dom_b],
            blend_content=blend_content,
            emergent_properties=emergent,
            blend_strength=blend_strength,
            new_memory_id=None,
        )

    @staticmethod
    def _generate_blend_text(
        dom_a: str,
        dom_b: str,
        text_a: str,
        text_b: str,
        shared: set[str],
        emergent: list[str],
    ) -> str:
        shared_str = ", ".join(sorted(shared)[:4]) or "complementary concepts"
        emergent_str = ", ".join(emergent[:4]) or "new structures"
        return (
            f"Conceptual blend of {dom_a} and {dom_b}: both share [{shared_str}]. "
            f"From {dom_a}: \"{text_a[:60]}\". "
            f"From {dom_b}: \"{text_b[:60]}\". "
            f"Emergent blend properties: [{emergent_str}]."
        )

    def _store_blend(self, concept: BlendedConcept) -> Optional[str]:
        """Store the blended concept as a new memory."""
        new_id: Optional[str] = None
        try:
            from emms.core.models import Experience
            domains_str = " + ".join(concept.source_domains)
            exp = Experience(
                content=concept.blend_content,
                domain="insight",
                importance=self.blend_importance,
                title=f"Conceptual blend: {domains_str}",
                facts=[
                    f"Blend strength: {concept.blend_strength:.3f}",
                    f"Emergent: {', '.join(concept.emergent_properties[:3])}",
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

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()")
            for w in text.lower().split()
            if len(w) >= 4 and w not in _STOP_WORDS
        }

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
