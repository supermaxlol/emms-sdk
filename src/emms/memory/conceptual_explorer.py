"""Gap 7 — ConceptualSpaceExplorer: find structural holes and novel combinations.

Three operations:
1. STRUCTURAL HOLES: gaps in the concept graph where a concept SHOULD exist
2. PROPERTY TRANSFER: if A has property P in domain X, try P in domain Y
3. CONCEPTUAL BLENDING: combine concepts from different domains

Uses memory content + domain tags. No embeddings — bag-of-words consistent
with the rest of the system.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StructuralHole:
    """A gap in the concept graph where a concept should exist."""
    concept: str
    present_in: list[str]   # domains where concept exists
    missing_from: str        # domain where it's absent
    related_concept: str     # what exists in the missing domain
    confidence: float = 0.5
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "present_in": self.present_in,
            "missing_from": self.missing_from,
            "related_concept": self.related_concept,
            "confidence": self.confidence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StructuralHole:
        return cls(
            concept=d["concept"],
            present_in=d["present_in"],
            missing_from=d["missing_from"],
            related_concept=d.get("related_concept", ""),
            confidence=d.get("confidence", 0.5),
            description=d.get("description", ""),
        )


@dataclass
class NovelConcept:
    """A newly generated concept from combining existing ones."""
    name: str
    parent_concepts: list[str]
    source_domains: list[str]
    description: str
    novelty_score: float = 0.5
    plausibility_score: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent_concepts": self.parent_concepts,
            "source_domains": self.source_domains,
            "description": self.description,
            "novelty_score": self.novelty_score,
            "plausibility_score": self.plausibility_score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NovelConcept:
        return cls(
            name=d["name"],
            parent_concepts=d["parent_concepts"],
            source_domains=d.get("source_domains", []),
            description=d["description"],
            novelty_score=d.get("novelty_score", 0.5),
            plausibility_score=d.get("plausibility_score", 0.5),
            timestamp=d.get("timestamp", time.time()),
        )


# Stopwords to exclude from concept extraction
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "my", "your",
    "his", "her", "our", "their", "which", "who", "whom", "what",
    "where", "when", "how", "why", "if", "then", "than", "but",
    "and", "or", "not", "no", "with", "from", "for", "to", "of",
    "in", "on", "at", "by", "as", "so", "very", "just", "also",
    "about", "after", "before", "between", "through", "during",
})


class ConceptualSpaceExplorer:
    """Explores and extends the concept space to find novel ideas."""

    def __init__(self, *, max_holes: int = 50,
                 max_concepts: int = 100) -> None:
        self._holes: list[StructuralHole] = []
        self._concepts: list[NovelConcept] = []
        self._max_holes = max_holes
        self._max_concepts = max_concepts
        self._exploration_log: list[dict] = []
        self._total_explorations = 0

    # -- structural holes ---------------------------------------------------

    def find_structural_holes(
        self,
        memories: list[dict],
    ) -> list[StructuralHole]:
        """Find concept combinations that exist in some domains but not others.

        Args:
            memories: list of dicts with 'content' and 'domain' keys.
        """
        self._total_explorations += 1

        # Extract bigram concepts per domain
        domain_concepts: dict[str, set[str]] = {}
        for mem in memories:
            domain = mem.get("domain", "unknown")
            content = mem.get("content", "")
            words = [w for w in content.lower().split() if w not in _STOPWORDS and len(w) > 3]
            concepts = set()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                concepts.add(bigram)
            domain_concepts.setdefault(domain, set()).update(concepts)

        all_domains = set(domain_concepts.keys())
        if len(all_domains) < 2:
            return []

        # Build concept → domains mapping
        concept_domains: dict[str, set[str]] = {}
        for domain, concepts in domain_concepts.items():
            for concept in concepts:
                concept_domains.setdefault(concept, set()).add(domain)

        # Find holes: concepts in 2+ domains but not all
        holes: list[StructuralHole] = []
        for concept, domains in concept_domains.items():
            if not (2 <= len(domains) < len(all_domains)):
                continue
            missing = all_domains - domains
            for missing_domain in missing:
                related = domain_concepts.get(missing_domain, set())
                concept_words = set(concept.split())
                for r in related:
                    if concept_words & set(r.split()):
                        holes.append(StructuralHole(
                            concept=concept,
                            present_in=sorted(domains),
                            missing_from=missing_domain,
                            related_concept=r,
                            confidence=len(domains) / len(all_domains),
                            description=(
                                f"'{concept}' in {sorted(domains)} "
                                f"but not {missing_domain} "
                                f"(related: '{r}')"
                            ),
                        ))
                        break

        unique = self._deduplicate_holes(holes)
        self._holes = sorted(unique, key=lambda h: -h.confidence)[:self._max_holes]

        self._exploration_log.append({
            "timestamp": time.time(),
            "memories_scanned": len(memories),
            "domains": len(all_domains),
            "holes_found": len(self._holes),
        })

        return list(self._holes)

    # -- property transfer --------------------------------------------------

    def property_transfer(
        self,
        source_domain: str,
        target_domain: str,
        memories: list[dict],
    ) -> list[NovelConcept]:
        """Transfer properties from one domain to another."""
        source_props: set[str] = set()
        target_props: set[str] = set()

        for mem in memories:
            domain = mem.get("domain", "unknown")
            content = mem.get("content", "")
            words = {w for w in content.lower().split()
                     if w not in _STOPWORDS and len(w) > 4}
            if domain == source_domain:
                source_props.update(words)
            elif domain == target_domain:
                target_props.update(words)

        transferable = source_props - target_props
        if not transferable:
            return []

        concepts = []
        for prop in sorted(transferable)[:5]:
            concepts.append(NovelConcept(
                name=f"{prop}_in_{target_domain}",
                parent_concepts=[prop],
                source_domains=[source_domain, target_domain],
                description=f"Transfer '{prop}' from {source_domain} to {target_domain}",
                novelty_score=0.6,
                plausibility_score=0.4,
            ))

        self._concepts.extend(concepts)
        if len(self._concepts) > self._max_concepts:
            self._concepts = self._concepts[-self._max_concepts:]
        return concepts

    # -- conceptual blending ------------------------------------------------

    def blend_concepts(
        self,
        concept_a: str,
        domain_a: str,
        concept_b: str,
        domain_b: str,
    ) -> NovelConcept:
        """Create a blend of two concepts from different domains."""
        blend = NovelConcept(
            name=f"{concept_a}+{concept_b}",
            parent_concepts=[concept_a, concept_b],
            source_domains=[domain_a, domain_b],
            description=(
                f"Blend of '{concept_a}' ({domain_a}) and "
                f"'{concept_b}' ({domain_b})"
            ),
            novelty_score=0.7,
            plausibility_score=0.3,
        )
        self._concepts.append(blend)
        if len(self._concepts) > self._max_concepts:
            self._concepts = self._concepts[-self._max_concepts:]
        return blend

    # -- accessors ----------------------------------------------------------

    @property
    def holes(self) -> list[StructuralHole]:
        return list(self._holes)

    @property
    def concepts(self) -> list[NovelConcept]:
        return list(self._concepts)

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        return (
            f"ConceptualSpaceExplorer: {len(self._holes)} holes, "
            f"{len(self._concepts)} generated concepts, "
            f"{self._total_explorations} explorations"
        )

    def exploration_log(self, limit: int = 20) -> list[dict]:
        return list(self._exploration_log[-limit:])

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_explorations": self._total_explorations,
            "holes": [h.to_dict() for h in self._holes],
            "concepts": [c.to_dict() for c in self._concepts[-self._max_concepts:]],
            "exploration_log": self._exploration_log[-50:],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_explorations = data.get("total_explorations", 0)
            self._holes = [StructuralHole.from_dict(h) for h in data.get("holes", [])]
            self._concepts = [NovelConcept.from_dict(c) for c in data.get("concepts", [])]
            self._exploration_log = data.get("exploration_log", [])
            return True
        except Exception:
            return False

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _deduplicate_holes(holes: list[StructuralHole]) -> list[StructuralHole]:
        if not holes:
            return []
        unique = [holes[0]]
        for h in holes[1:]:
            is_dup = False
            for u in unique:
                if (h.missing_from == u.missing_from and
                        h.concept == u.concept):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(h)
        return unique
