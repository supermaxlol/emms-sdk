"""Hierarchical memory: working → short-term → long-term → semantic.

Inspired by Atkinson-Shiffrin with Miller's Law (7±2) for working memory,
exponential decay for short-term, and importance-weighted consolidation.
Supports optional embedding-based retrieval when an EmbeddingProvider is supplied.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

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


# ---------------------------------------------------------------------------
# VectorIndex — batch cosine similarity via numpy (replaces O(n) per-item)
# ---------------------------------------------------------------------------

class VectorIndex:
    """Numpy-based vector index for fast batch cosine similarity.

    Stores vectors in a dense matrix for efficient batch operations
    instead of computing cosine similarity one-at-a-time.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._matrix: np.ndarray | None = None  # N x dim, L2-normalised rows
        self._dirty = True  # rebuild matrix on next query

        # Buffer for incremental adds before rebuild
        self._buffer: list[tuple[str, np.ndarray]] = []

    def add(self, id: str, vector: list[float]) -> None:
        """Add or update a vector."""
        vec = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if id in self._id_to_idx:
            idx = self._id_to_idx[id]
            if self._matrix is not None and idx < len(self._matrix):
                self._matrix[idx] = vec
                return

        self._buffer.append((id, vec))
        self._dirty = True

    def remove(self, id: str) -> None:
        """Remove a vector by ID."""
        if id in self._id_to_idx:
            idx = self._id_to_idx.pop(id)
            self._ids[idx] = ""  # mark as deleted
            self._dirty = True

    def query(self, vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (id, similarity) pairs."""
        self._rebuild_if_dirty()
        if self._matrix is None or len(self._matrix) == 0:
            return []

        q = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Batch cosine similarity: dot product with normalised matrix
        sims = self._matrix @ q  # (N,)
        top_k = min(k, len(sims))
        indices = np.argpartition(sims, -top_k)[-top_k:]
        indices = indices[np.argsort(sims[indices])[::-1]]

        results = []
        for idx in indices:
            if idx < len(self._ids) and self._ids[idx]:
                results.append((self._ids[idx], float(sims[idx])))
        return results

    def __len__(self) -> int:
        return len(self._id_to_idx)

    def _rebuild_if_dirty(self) -> None:
        """Rebuild the matrix from IDs + buffer."""
        if not self._dirty:
            return

        # Flush buffer
        for id_, vec in self._buffer:
            if id_ not in self._id_to_idx:
                self._id_to_idx[id_] = len(self._ids)
                self._ids.append(id_)

        self._buffer.clear()

        # Remove deleted entries and rebuild
        live_ids = []
        live_vecs = []
        for i, id_ in enumerate(self._ids):
            if id_ and id_ in self._id_to_idx:
                live_ids.append(id_)
                if self._matrix is not None and i < len(self._matrix):
                    live_vecs.append(self._matrix[i])
                else:
                    live_vecs.append(np.zeros(self.dim))

        self._ids = live_ids
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

        if live_vecs:
            self._matrix = np.vstack(live_vecs)
        else:
            self._matrix = None

        self._dirty = False


def _simple_stem(word: str) -> str:
    """Zero-dependency English stemmer (suffix stripping).

    Handles the most common English suffixes. Not perfect, but good enough
    to match "rates"→"rate", "investing"→"invest", "discoveries"→"discover", etc.
    """
    if len(word) <= 3:
        return word
    # Order matters: longest suffixes first
    for suffix, min_stem in [
        ("ational", 4), ("ization", 4), ("fulness", 4),
        ("iveness", 4), ("ousness", 4),
        ("ating", 3), ("ation", 3), ("ments", 3), ("ering", 3),
        ("ities", 3), ("iness", 3),
        ("ment", 3), ("ness", 3), ("ence", 3), ("ance", 3),
        ("ting", 3), ("ling", 3), ("ring", 3), ("ings", 3),
        ("ally", 3), ("ious", 3), ("ical", 3),
        ("ing", 3), ("ies", 3), ("ion", 3), ("ity", 3),
        ("ful", 3), ("ous", 3), ("ive", 3), ("ble", 3),
        ("ant", 3), ("ent", 3), ("ist", 3),
        ("ed", 3), ("er", 3), ("ly", 3), ("al", 3),
        ("es", 3), ("ty", 3),
        ("s", 3),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[:-len(suffix)]
    return word


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

        # Vector index for fast ANN retrieval
        self._vec_index: VectorIndex | None = None
        if embedder is not None:
            self._vec_index = VectorIndex(dim=embedder.dim)

        # Inverted index: word → set of experience IDs (for fast lexical retrieval)
        self._word_index: dict[str, set[str]] = {}

        # Experience ID → MemoryItem lookup (for vector index results)
        self._items_by_exp_id: dict[str, MemoryItem] = {}

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

        # Update inverted index for fast lexical retrieval (with stemming)
        for word in experience.content.lower().split():
            if len(word) >= 2:
                stem = _simple_stem(word)
                if stem not in self._word_index:
                    self._word_index[stem] = set()
                self._word_index[stem].add(experience.id)
                # Also index the original word for exact matches
                if word != stem:
                    if word not in self._word_index:
                        self._word_index[word] = set()
                    self._word_index[word].add(experience.id)

        # Track item by experience ID for fast lookup
        self._items_by_exp_id[experience.id] = item

        # Compute and cache embedding if embedder available
        if self.embedder is not None:
            if experience.embedding:
                self._embeddings[experience.id] = experience.embedding
            else:
                self._embeddings[experience.id] = self.embedder.embed(
                    experience.content
                )
            # Add to vector index
            if self._vec_index is not None:
                self._vec_index.add(experience.id, self._embeddings[experience.id])

        # Auto-consolidate if working memory is at capacity
        if len(self.working) >= self.cfg.working_capacity:
            self._consolidate_working()

        return item

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, max_results: int = 10) -> list[RetrievalResult]:
        """Multi-strategy ensemble retrieval across all tiers.

        Combines multiple scoring signals:
        1. Semantic similarity (embedding or lexical)
        2. Temporal recency boost
        3. Importance-weighted ranking
        4. Access-frequency signal (popular memories surface)
        5. Domain-affinity bonus

        Uses embedding cosine similarity when an embedder is available,
        otherwise falls back to word-overlap scoring.
        """
        results: list[RetrievalResult] = []

        # Embed the query once if we have an embedder
        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed(query)

        query_words = set(query.lower().split())

        # Detect query domain from keywords
        query_domain = self._infer_domain(query_words)

        # Pre-filter using inverted index for lexical retrieval
        # Collect IDs of items that share at least one word/stem with the query
        candidate_ids: set[str] | None = None
        if query_vec is None and self._word_index:
            candidate_ids = set()
            for word in query_words:
                if word in self._word_index:
                    candidate_ids.update(self._word_index[word])
                stem = _simple_stem(word)
                if stem != word and stem in self._word_index:
                    candidate_ids.update(self._word_index[stem])

        for tier, store in self._iter_tiers():
            # Tier weight: higher tiers get a small bonus (semantic > LT > ST > working)
            tier_boost = {
                MemoryTier.SEMANTIC: 0.08,
                MemoryTier.LONG_TERM: 0.04,
                MemoryTier.SHORT_TERM: 0.02,
                MemoryTier.WORKING: 0.0,
            }.get(tier, 0.0)

            for item in store:
                # Skip items that can't match (lexical pre-filter)
                if candidate_ids is not None and item.experience.id not in candidate_ids:
                    continue

                if query_vec is not None:
                    base_score = self._embedding_relevance(item, query_vec)
                else:
                    base_score = self._relevance(item, query_words)

                # Domain affinity: bonus if query and item share a domain
                domain_bonus = 0.0
                if query_domain and item.experience.domain == query_domain:
                    domain_bonus = 0.05

                # Access frequency signal: frequently-accessed memories are likely useful
                access_bonus = min(0.05, item.access_count * 0.01)

                score = min(1.0, base_score + tier_boost + domain_bonus + access_bonus)

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

        # Re-ranking pass: diversify results across domains
        if len(results) > max_results:
            results = self._diversify(results, max_results)

        return results[:max_results]

    def _infer_domain(self, query_words: set[str]) -> str | None:
        """Guess the query domain from keywords."""
        domain_keywords: dict[str, set[str]] = {
            "finance": {"stock", "market", "price", "trading", "investment", "rate", "earnings", "revenue", "profit", "economic", "financial", "crypto", "bitcoin"},
            "tech": {"software", "code", "programming", "algorithm", "computer", "api", "data", "model", "neural", "machine", "learning", "technology"},
            "science": {"research", "study", "experiment", "theory", "quantum", "physics", "biology", "chemistry", "scientific", "discovery"},
            "weather": {"weather", "temperature", "rain", "storm", "climate", "forecast", "wind", "humidity"},
            "health": {"health", "medical", "disease", "treatment", "patient", "clinical", "drug", "therapy", "diagnosis"},
        }
        best_domain = None
        best_overlap = 0
        for domain, keywords in domain_keywords.items():
            overlap = len(query_words & keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain
        return best_domain if best_overlap > 0 else None

    def _diversify(self, results: list[RetrievalResult], max_results: int) -> list[RetrievalResult]:
        """Ensure result diversity across domains and tiers."""
        seen_domains: dict[str, int] = {}
        diverse: list[RetrievalResult] = []
        max_per_domain = max(2, max_results // 2)

        for r in results:
            domain = r.memory.experience.domain
            count = seen_domains.get(domain, 0)
            if count < max_per_domain:
                diverse.append(r)
                seen_domains[domain] = count + 1
            if len(diverse) >= max_results:
                break

        return diverse

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
        """Promote high-value items from short-term → long-term.

        Two promotion paths:
        1. Standard: score >= threshold AND access_count >= 2
        2. Importance bypass: very high importance (>= 0.8) promotes directly
           even without repeated access — critical memories shouldn't be lost.
        """
        moved = 0
        survivors: list[MemoryItem] = []

        for item in list(self.short_term):
            item.decay(self.cfg.decay_half_life_seconds)
            score = self._consolidation_score(item)

            standard_promote = (
                score >= self.cfg.consolidation_threshold
                and item.access_count >= 2
            )
            importance_bypass = item.experience.importance >= 0.8

            if standard_promote or importance_bypass:
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

    # ------------------------------------------------------------------
    # Persistence — save/load full memory state
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        """Serialize full memory state (all tiers + embeddings) to JSON."""
        path = Path(path)

        def _serialize_items(items) -> list[dict]:
            result = []
            for item in items:
                d = item.model_dump()
                # Pydantic serialises enums; ensure string values
                d["tier"] = item.tier.value
                exp = d["experience"]
                # Convert Modality keys to strings
                if "modality_features" in exp:
                    exp["modality_features"] = {
                        (k.value if hasattr(k, "value") else str(k)): v
                        for k, v in exp["modality_features"].items()
                    }
                result.append(d)
            return result

        state = {
            "version": "0.4.0",
            "saved_at": time.time(),
            "working": _serialize_items(self.working),
            "short_term": _serialize_items(self.short_term),
            "long_term": _serialize_items(self.long_term.values()),
            "semantic": _serialize_items(self.semantic.values()),
            "embeddings": {k: v for k, v in self._embeddings.items()},
            "stats": {
                "total_stored": self.total_stored,
                "total_consolidated": self.total_consolidated,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, default=str), encoding="utf-8")
        logger.info("Memory state saved to %s (%d items)", path, self.size["total"])

    def load_state(self, path: Path | str) -> None:
        """Restore full memory state from JSON."""
        path = Path(path)
        if not path.exists():
            logger.warning("No state file at %s", path)
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        def _deserialize_items(items_data: list[dict]) -> list[MemoryItem]:
            result = []
            for d in items_data:
                exp_data = d.pop("experience")
                exp = Experience(**exp_data)
                d["experience"] = exp
                d["tier"] = MemoryTier(d["tier"])
                item = MemoryItem(**d)
                result.append(item)
            return result

        # Restore tiers
        self.working.clear()
        for item in _deserialize_items(data.get("working", [])):
            self.working.append(item)

        self.short_term.clear()
        for item in _deserialize_items(data.get("short_term", [])):
            self.short_term.append(item)

        self.long_term.clear()
        for item in _deserialize_items(data.get("long_term", [])):
            self.long_term[item.id] = item

        self.semantic.clear()
        for item in _deserialize_items(data.get("semantic", [])):
            self.semantic[item.id] = item

        # Restore embeddings
        self._embeddings = data.get("embeddings", {})

        # Rebuild word index, _items_by_exp_id, and VectorIndex from all items
        self._word_index.clear()
        self._items_by_exp_id.clear()
        if self._vec_index is not None:
            self._vec_index = VectorIndex(dim=self._vec_index.dim)

        for _, store in self._iter_tiers():
            for item in store:
                # Rebuild experience ID → item lookup
                self._items_by_exp_id[item.experience.id] = item

                # Rebuild word index
                for word in item.experience.content.lower().split():
                    if len(word) >= 2:
                        stem = _simple_stem(word)
                        if stem not in self._word_index:
                            self._word_index[stem] = set()
                        self._word_index[stem].add(item.experience.id)
                        if word != stem:
                            if word not in self._word_index:
                                self._word_index[word] = set()
                            self._word_index[word].add(item.experience.id)

                # Rebuild VectorIndex from restored embeddings
                if self._vec_index is not None and item.experience.id in self._embeddings:
                    self._vec_index.add(item.experience.id, self._embeddings[item.experience.id])

        # Restore stats
        stats = data.get("stats", {})
        self.total_stored = stats.get("total_stored", 0)
        self.total_consolidated = stats.get("total_consolidated", 0)

        logger.info("Memory state loaded from %s (%d items)", path, self.size["total"])
