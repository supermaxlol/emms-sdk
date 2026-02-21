"""SchemaExtractor — abstract pattern extraction from concrete memories.

v0.14.0: The Temporal Mind

Individual memories are specific: "I helped debug a consolidation pipeline."
Schemas are abstract: "When debugging algorithmic systems, working backward
from the observed failure mode is the reliable strategy."

The SchemaExtractor finds keywords and concepts that recur across multiple
memories in the same domain, groups memories by those shared patterns, and
synthesises a concise schema description for each group. This is how concrete
experience becomes transferable, abstract knowledge.

Biological analogue: schema theory (Bartlett 1932) — the brain's tendency to
abstract regularities from repeated experiences into generalised knowledge
structures stored separately from individual episodic memories. Semantic
memory (knowing "birds fly") vs episodic memory (remembering seeing a robin).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Stop-words to exclude from schema keywords
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "that", "this", "it", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "my", "i", "me", "we", "our", "you", "he", "she", "they", "them",
    "his", "her", "its", "their", "what", "which", "who", "how", "when",
    "where", "why", "as", "if", "about", "through", "into", "during",
    "before", "after", "above", "between", "each", "more", "also", "not",
    "can", "so", "than", "then", "just", "because", "while", "over",
    "such", "up", "out", "no", "any", "most", "all",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Schema:
    """An abstract pattern distilled from multiple concrete memories."""

    id: str
    domain: str
    pattern: str                     # human-readable schema description
    supporting_memory_ids: list[str] # memories this schema was derived from
    keywords: list[str]              # defining terms
    confidence: float                # support count / max possible (0..1)
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        kws = ", ".join(self.keywords[:6])
        return (
            f"Schema [{self.domain}] conf={self.confidence:.2f}  "
            f"support={len(self.supporting_memory_ids)}\n"
            f"  Pattern: {self.pattern}\n"
            f"  Keywords: {kws}"
        )


@dataclass
class SchemaReport:
    """Result of a SchemaExtractor.extract() run."""

    total_memories_analyzed: int
    schemas_found: int
    schemas: list[Schema]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"SchemaExtractor: {self.schemas_found} schemas from "
            f"{self.total_memories_analyzed} memories in {self.duration_seconds:.2f}s",
        ]
        for s in self.schemas[:5]:
            lines.append(f"  [{s.domain}] conf={s.confidence:.2f}: {s.pattern[:80]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SchemaExtractor
# ---------------------------------------------------------------------------

class SchemaExtractor:
    """Extracts reusable knowledge schemas from repeated memory patterns.

    For each domain (or across all domains) the extractor:

    1. Collects all memories and tokenises their content.
    2. Builds a keyword frequency table (excluding stop-words and very short
       words).
    3. Identifies keywords that appear in at least ``min_support`` distinct
       memories.
    4. Groups memories by their most frequent shared keyword cluster.
    5. Synthesises a concise pattern description for each cluster.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_support:
        Minimum number of memories a keyword must appear in to seed a schema
        (default 2).
    min_keyword_length:
        Minimum character length for a keyword (default 4).
    max_schemas:
        Maximum schemas to return per extract() call (default 12).
    """

    def __init__(
        self,
        memory: Any,
        min_support: int = 2,
        min_keyword_length: int = 4,
        max_schemas: int = 12,
    ) -> None:
        self.memory = memory
        self.min_support = min_support
        self.min_keyword_length = min_keyword_length
        self.max_schemas = max_schemas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        domain: Optional[str] = None,
        max_schemas: Optional[int] = None,
    ) -> SchemaReport:
        """Extract schemas from stored memories.

        Args:
            domain:      If given, restrict to memories in this domain.
                         ``None`` processes all domains separately.
            max_schemas: Override the instance default.

        Returns:
            :class:`SchemaReport` with extracted schemas.
        """
        t0 = time.time()
        limit = max_schemas if max_schemas is not None else self.max_schemas

        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        if not items:
            return SchemaReport(
                total_memories_analyzed=0,
                schemas_found=0,
                schemas=[],
                duration_seconds=time.time() - t0,
            )

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        all_schemas: list[Schema] = []
        for dom, dom_items in by_domain.items():
            schemas = self._extract_for_domain(dom, dom_items)
            all_schemas.extend(schemas)

        # Sort by confidence descending, take top-k
        all_schemas.sort(key=lambda s: s.confidence, reverse=True)
        all_schemas = all_schemas[:limit]

        return SchemaReport(
            total_memories_analyzed=len(items),
            schemas_found=len(all_schemas),
            schemas=all_schemas,
            duration_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_for_domain(self, domain: str, items: list[Any]) -> list[Schema]:
        """Extract schemas from a single domain's memory items."""
        if len(items) < self.min_support:
            return []

        # Build per-item keyword sets
        item_keywords: list[tuple[Any, set[str]]] = []
        for item in items:
            kws = self._keywords(item.experience.content)
            item_keywords.append((item, kws))

        # Count keyword frequency across items
        keyword_freq: Counter = Counter()
        for _, kws in item_keywords:
            for kw in kws:
                keyword_freq[kw] += 1

        # Seed keywords: appear in >= min_support items
        seed_keywords = [
            kw for kw, freq in keyword_freq.most_common(30)
            if freq >= self.min_support
        ]
        if not seed_keywords:
            return []

        # Greedy clustering: assign each item to the seed keyword it contains
        # with the highest frequency
        assigned: set[str] = set()  # item IDs already in a schema
        schemas: list[Schema] = []

        for kw in seed_keywords:
            if len(schemas) >= self.max_schemas:
                break
            # Items containing this keyword
            group = [
                item for item, kws in item_keywords
                if kw in kws and item.id not in assigned
            ]
            if len(group) < self.min_support:
                continue

            # Collect all keywords shared by ≥2 items in this group
            group_kws: Counter = Counter()
            for item in group:
                _, kws = next(
                    (pair for pair in item_keywords if pair[0].id == item.id),
                    (None, set()),
                )
                for k in kws:
                    group_kws[k] += 1
            shared_kws = [k for k, c in group_kws.most_common(8) if c >= 2]
            if not shared_kws:
                shared_kws = [kw]

            pattern = self._generate_pattern(shared_kws, domain)
            confidence = min(1.0, len(group) / max(len(items), 1))

            schemas.append(Schema(
                id=f"schema_{uuid.uuid4().hex[:8]}",
                domain=domain,
                pattern=pattern,
                supporting_memory_ids=[it.id for it in group],
                keywords=shared_kws[:6],
                confidence=confidence,
            ))
            assigned.update(it.id for it in group)

        return schemas

    def _keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        tokens = text.lower().split()
        return {
            t.strip(".,!?;:\"'()[]")
            for t in tokens
            if len(t) >= self.min_keyword_length
            and t not in _STOP_WORDS
            and t.isalpha()
        }

    def _generate_pattern(self, keywords: list[str], domain: str) -> str:
        """Template-based schema pattern description."""
        kw_str = ", ".join(keywords[:4])
        templates = [
            f"In {domain} contexts, recurring themes include: {kw_str}.",
            f"A repeating pattern in {domain}: concepts around {kw_str} co-occur consistently.",
            f"{domain.capitalize()} schema — key elements: {kw_str}; these form a coherent cluster.",
        ]
        # Choose based on keyword count
        idx = min(len(keywords) // 2, len(templates) - 1)
        return templates[idx]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
