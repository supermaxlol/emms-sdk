"""Hierarchical memory: working → short-term → long-term → semantic.

Inspired by Atkinson-Shiffrin with Miller's Law (7±2) for working memory,
exponential decay for short-term, and importance-weighted consolidation.
Supports optional embedding-based retrieval when an EmbeddingProvider is supplied.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Sequence

import numpy as np

from emms.core.models import (
    Experience,
    MemoryConfig,
    MemoryItem,
    MemoryTier,
    RetrievalResult,
)
from emms.core.embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


class HierarchicalMemory:
    """Four-tier memory hierarchy with automatic consolidation.

    If an *embedder* is provided, retrieval uses cosine similarity on
    embeddings (much better quality).  Otherwise falls back to word overlap.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        self.cfg = config or MemoryConfig()
        self.embedder = embedder

        # Tier stores
        self.working: deque[MemoryItem] = deque(maxlen=self.cfg.working_capacity)
        self.short_term: deque[MemoryItem] = deque(maxlen=self.cfg.short_term_capacity)
        self.long_term: dict[str, MemoryItem] = {}
        self.semantic: dict[str, MemoryItem] = {}

        # Embedding cache: experience_id → embedding vector
        self._embeddings: dict[str, list[float]] = {}

        # Stats
        self.total_stored = 0
        self.total_consolidated = 0

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> MemoryItem:
        """Store an experience — always enters working memory first."""
        item = MemoryItem(experience=experience, tier=MemoryTier.WORKING)
        self.working.append(item)
        self.total_stored += 1

        # Compute and cache embedding if embedder available
        if self.embedder is not None:
            if experience.embedding:
                self._embeddings[experience.id] = experience.embedding
            else:
                self._embeddings[experience.id] = self.embedder.embed(
                    experience.content
                )

        # Auto-consolidate if working memory is at capacity
        if len(self.working) >= self.cfg.working_capacity:
            self._consolidate_working()

        return item

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, max_results: int = 10) -> list[RetrievalResult]:
        """Search all tiers, return ranked results.

        Uses embedding cosine similarity when an embedder is available,
        otherwise falls back to word-overlap scoring.
        """
        results: list[RetrievalResult] = []

        # Embed the query once if we have an embedder
        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed(query)

        query_words = set(query.lower().split())

        for tier, store in self._iter_tiers():
            for item in store:
                if query_vec is not None:
                    score = self._embedding_relevance(item, query_vec)
                else:
                    score = self._relevance(item, query_words)

                if score > self.cfg.relevance_threshold:
                    item.touch()
                    results.append(
                        RetrievalResult(
                            memory=item,
                            score=score,
                            source_tier=tier,
                            strategy="embedding" if query_vec else "lexical",
                        )
                    )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    # ------------------------------------------------------------------
    # Consolidation pipeline
    # ------------------------------------------------------------------

    def consolidate(self) -> int:
        """Run a full consolidation pass across all tiers. Returns count moved."""
        moved = 0
        moved += self._consolidate_working()
        moved += self._consolidate_short_term()
        moved += self._consolidate_long_term()
        return moved

    def _consolidate_working(self) -> int:
        """Promote high-value items from working → short-term."""
        moved = 0
        survivors: list[MemoryItem] = []

        for item in list(self.working):
            score = self._consolidation_score(item)
            if score >= self.cfg.consolidation_threshold:
                item.tier = MemoryTier.SHORT_TERM
                item.consolidation_score = score
                self.short_term.append(item)
                moved += 1
                self.total_consolidated += 1
            else:
                survivors.append(item)

        # Rebuild working memory with survivors only
        self.working.clear()
        for s in survivors:
            self.working.append(s)

        return moved

    def _consolidate_short_term(self) -> int:
        """Promote high-value items from short-term → long-term."""
        moved = 0
        survivors: list[MemoryItem] = []

        for item in list(self.short_term):
            item.decay(self.cfg.decay_half_life_seconds)
            score = self._consolidation_score(item)

            if score >= self.cfg.consolidation_threshold and item.access_count >= 2:
                item.tier = MemoryTier.LONG_TERM
                item.consolidation_score = score
                self.long_term[item.id] = item
                moved += 1
                self.total_consolidated += 1
            elif item.memory_strength > 0.1:
                survivors.append(item)
            # else: forgotten

        self.short_term.clear()
        for s in survivors:
            self.short_term.append(s)

        return moved

    def _consolidate_long_term(self) -> int:
        """Promote highly-accessed long-term items → semantic."""
        moved = 0
        to_remove: list[str] = []

        for item_id, item in self.long_term.items():
            item.decay(self.cfg.decay_half_life_seconds * 10)  # slower decay
            if item.access_count >= 5 and item.memory_strength > 0.5:
                item.tier = MemoryTier.SEMANTIC
                self.semantic[item_id] = item
                to_remove.append(item_id)
                moved += 1

        for rid in to_remove:
            del self.long_term[rid]

        return moved

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _relevance(self, item: MemoryItem, query_words: set[str]) -> float:
        """Content-based relevance using word overlap + importance/recency boosts."""
        content_words = set(item.experience.content.lower().split())

        # Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        jaccard = intersection / union if union else 0.0

        # Importance boost (0–0.2)
        importance_boost = item.experience.importance * 0.2

        # Recency boost (0–0.1), exponential decay
        age = item.age
        recency_boost = 0.1 * np.exp(-age / self.cfg.decay_half_life_seconds)

        # Strength boost (0–0.1)
        strength_boost = item.memory_strength * 0.1

        return min(1.0, jaccard + importance_boost + recency_boost + strength_boost)

    def _embedding_relevance(self, item: MemoryItem, query_vec: list[float]) -> float:
        """Cosine similarity + importance/recency boosts."""
        exp_id = item.experience.id
        item_vec = self._embeddings.get(exp_id)
        if item_vec is None:
            # Fallback: embed on-the-fly
            if self.embedder:
                item_vec = self.embedder.embed(item.experience.content)
                self._embeddings[exp_id] = item_vec
            else:
                return 0.0

        sim = cosine_similarity(query_vec, item_vec)
        # Clamp negative similarities to 0
        sim = max(0.0, sim)

        importance_boost = item.experience.importance * 0.15
        age = item.age
        recency_boost = 0.1 * np.exp(-age / self.cfg.decay_half_life_seconds)
        strength_boost = item.memory_strength * 0.05

        return min(1.0, sim * 0.7 + importance_boost + recency_boost + strength_boost)

    def _consolidation_score(self, item: MemoryItem) -> float:
        """Score determining whether an item should be promoted."""
        importance = item.experience.importance
        novelty = item.experience.novelty
        strength = item.memory_strength
        access = min(1.0, item.access_count / 5.0)

        return (
            importance * 0.35
            + novelty * 0.25
            + strength * 0.25
            + access * 0.15
        )

    # ------------------------------------------------------------------
    # Iteration & stats
    # ------------------------------------------------------------------

    def _iter_tiers(self):
        """Yield (tier_enum, iterable_of_items) for each tier."""
        yield MemoryTier.WORKING, list(self.working)
        yield MemoryTier.SHORT_TERM, list(self.short_term)
        yield MemoryTier.LONG_TERM, list(self.long_term.values())
        yield MemoryTier.SEMANTIC, list(self.semantic.values())

    @property
    def size(self) -> dict[str, int]:
        return {
            "working": len(self.working),
            "short_term": len(self.short_term),
            "long_term": len(self.long_term),
            "semantic": len(self.semantic),
            "total": (
                len(self.working)
                + len(self.short_term)
                + len(self.long_term)
                + len(self.semantic)
            ),
        }
