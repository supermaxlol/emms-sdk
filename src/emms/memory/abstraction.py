"""AbstractionEngine — lifting episodic memories into abstract principles.

v0.22.0: The Creative Mind

Abstraction is the cognitive capacity to extract a generalised pattern from
specific instances — to move from the concrete episode ("the negotiation
succeeded when we listened first") to the reusable principle ("listening
before speaking improves outcomes"). This capacity transforms episodic memory
into structured knowledge: principles that guide future behaviour even in
novel situations. Without abstraction, an agent accumulates a raw collection
of episodes but cannot extract the generalisable lessons they encode.

AbstractionEngine operationalises this by scanning memory for recurring
tokens within each domain. A token that appears in many memories within a
domain is a *candidate abstract principle*: it represents something that the
agent has repeatedly encountered, thought about, or acted upon. The generality
score (recurring_count / total_domain_memories) quantifies how broadly the
principle applies within the domain. High-generality tokens — appearing in
≥ min_generality fraction of a domain's memories — are elevated to
AbstractPrinciple objects, enriched with mean valence and importance computed
across all memories that contain the token.

Biological analogue: schema abstraction (Bartlett 1932) — repeated experience
consolidates into organised knowledge structures (schemas); prototype theory
(Rosch 1975) — categories are represented by their most typical member;
prefrontal cortex in hierarchical abstraction (Badre & Frank 2012) — the PFC
encodes increasingly abstract task representations at higher levels; analogical
generalisation across instances (Gentner et al. 2009) — structural similarity
enables cross-episode abstraction; hippocampal-neocortical transfer — repeated
experiences gradually shift from episodic to semantic/schematic representation.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AbstractPrinciple:
    """A generalised principle extracted from episodic memories."""

    id: str                       # prefixed "abs_"
    label: str                    # the recurring concept token
    domain: str
    description: str              # template-generated description
    generality_score: float       # 0..1 — fraction of domain memories
    mean_valence: float
    mean_importance: float
    source_memory_ids: list[str]
    created_at: float

    def summary(self) -> str:
        n = len(self.source_memory_ids)
        return (
            f"AbstractPrinciple [{self.domain}]  generality={self.generality_score:.3f}  "
            f"valence={self.mean_valence:.2f}  importance={self.mean_importance:.2f}\n"
            f"  {self.id[:12]}: '{self.label}' in {n} memories"
        )


@dataclass
class AbstractionReport:
    """Result of an AbstractionEngine.abstract() call."""

    total_principles: int
    principles: list[AbstractPrinciple]  # sorted by generality desc
    domains_abstracted: list[str]
    mean_generality: float
    duration_seconds: float

    def summary(self) -> str:
        doms = ", ".join(self.domains_abstracted[:5])
        lines = [
            f"AbstractionReport: {self.total_principles} principles  "
            f"domains=[{doms}]  "
            f"mean_generality={self.mean_generality:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for p in self.principles[:5]:
            lines.append(f"  {p.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AbstractionEngine
# ---------------------------------------------------------------------------


class AbstractionEngine:
    """Lifts specific episodic memories into reusable abstract principles.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_generality:
        Minimum generality score (fraction of domain memories containing the
        token) to qualify as an abstract principle (default 0.3).
    max_principles:
        Maximum number of :class:`AbstractPrinciple` objects to retain
        (default 15).
    """

    def __init__(
        self,
        memory: Any,
        min_generality: float = 0.3,
        max_principles: int = 15,
    ) -> None:
        self.memory = memory
        self.min_generality = min_generality
        self.max_principles = max_principles
        self._principles: list[AbstractPrinciple] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def abstract(self, domain: Optional[str] = None) -> AbstractionReport:
        """Extract abstract principles from memory.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`AbstractionReport` with principles sorted by generality.
        """
        t0 = time.time()
        all_items = self._collect_all()

        # Group items by domain
        domain_items: dict[str, list[Any]] = defaultdict(list)
        for item in all_items:
            dom = getattr(item.experience, "domain", None) or "general"
            domain_items[dom].append(item)

        # Optionally restrict to a single domain
        if domain:
            domain_items = {
                d: items for d, items in domain_items.items() if d == domain
            }

        self._principles = []
        all_principles: list[AbstractPrinciple] = []

        for dom, items in domain_items.items():
            n_domain = len(items)
            if n_domain == 0:
                continue

            # Build token → list[item] index
            token_items: dict[str, list[Any]] = defaultdict(list)
            for item in items:
                content = getattr(item.experience, "content", "") or ""
                tokens = self._tokenise(content)
                for tok in set(tokens):
                    token_items[tok].append(item)

            for tok, tok_items in token_items.items():
                count = len(tok_items)
                if count < 2:  # must appear in at least 2 memories
                    continue
                generality = round(count / n_domain, 4)
                if generality < self.min_generality:
                    continue

                valences = [
                    getattr(it.experience, "valence", 0.0) or 0.0
                    for it in tok_items
                ]
                importances = [
                    getattr(it.experience, "importance", 0.5) or 0.5
                    for it in tok_items
                ]
                mean_valence = round(sum(valences) / len(valences), 4)
                mean_importance = round(sum(importances) / len(importances), 4)

                description = (
                    f"In {dom}: '{tok}' is a recurring principle "
                    f"appearing in {count} memories "
                    f"(generality={generality:.2f})"
                )
                principle = AbstractPrinciple(
                    id="abs_" + uuid.uuid4().hex[:8],
                    label=tok,
                    domain=dom,
                    description=description,
                    generality_score=generality,
                    mean_valence=mean_valence,
                    mean_importance=mean_importance,
                    source_memory_ids=[it.id for it in tok_items],
                    created_at=time.time(),
                )
                all_principles.append(principle)

        # Sort by generality descending, cap
        all_principles.sort(key=lambda p: p.generality_score, reverse=True)
        self._principles = all_principles[: self.max_principles]

        domains_abstracted = list({p.domain for p in self._principles})
        mean_gen = (
            sum(p.generality_score for p in self._principles) / len(self._principles)
            if self._principles else 0.0
        )

        return AbstractionReport(
            total_principles=len(self._principles),
            principles=self._principles,
            domains_abstracted=domains_abstracted,
            mean_generality=round(mean_gen, 4),
            duration_seconds=time.time() - t0,
        )

    def principles_for_domain(self, domain: str) -> list[AbstractPrinciple]:
        """Return principles for a specific domain.

        Args:
            domain: Domain name to filter by.
        """
        return [p for p in self._principles if p.domain == domain]

    def best_principle(self, description: str) -> Optional[AbstractPrinciple]:
        """Return the principle most relevant to a description (Jaccard).

        Args:
            description: Natural-language description to match against.

        Returns:
            Best-matching :class:`AbstractPrinciple`, or ``None`` if none.
        """
        if not self._principles:
            return None
        goal_tokens = set(self._tokenise(description))
        if not goal_tokens:
            return self._principles[0] if self._principles else None

        def jaccard(principle: AbstractPrinciple) -> float:
            principle_tokens = set(
                self._tokenise(principle.description) + [principle.label, principle.domain]
            )
            if not principle_tokens:
                return 0.0
            inter = len(goal_tokens & principle_tokens)
            union = len(goal_tokens | principle_tokens)
            return inter / union if union else 0.0

        return max(self._principles, key=jaccard)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 4
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
