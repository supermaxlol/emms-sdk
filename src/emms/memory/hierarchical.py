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
    ObsType,
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
        # Count id_to_idx entries plus buffer items not yet flushed into it.
        buffer_new = sum(1 for id_, _ in self._buffer if id_ not in self._id_to_idx)
        return len(self._id_to_idx) + buffer_new

    def _rebuild_if_dirty(self) -> None:
        """Rebuild the matrix from IDs + buffer."""
        if not self._dirty:
            return

        # Collect buffer vectors BEFORE clearing so they aren't lost.
        # Previously the buffer was flushed into _id_to_idx/_ids but the
        # actual vectors were discarded, causing every newly-added entry to
        # become a zero vector in the rebuilt matrix.
        buffer_vecs: dict[str, np.ndarray] = {}
        for id_, vec in self._buffer:
            if id_ not in self._id_to_idx:
                self._id_to_idx[id_] = len(self._ids)
                self._ids.append(id_)
            buffer_vecs[id_] = vec  # preserve the real vector

        self._buffer.clear()

        # Remove deleted entries and rebuild
        live_ids = []
        live_vecs = []
        for i, id_ in enumerate(self._ids):
            if id_ and id_ in self._id_to_idx:
                live_ids.append(id_)
                if id_ in buffer_vecs:
                    # Freshly added (or updated) vector — use it directly.
                    live_vecs.append(buffer_vecs[id_])
                elif self._matrix is not None and i < len(self._matrix):
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

    **Endless Mode** (biomimetic, claude-mem inspired):
    When ``endless_mode=True``, working memory overflow triggers real-time
    compression instead of silent eviction. Oldest items are merged into a
    single compressed episode and moved to short-term memory, keeping context
    growth O(N) rather than O(N²) over long sessions. This enables ~10–20×
    more memories before context exhaustion.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        endless_mode: bool = False,
        endless_chunk_size: int = 3,
    ):
        self.cfg = config or MemoryConfig()
        self.embedder = embedder

        # Endless Mode settings
        self.endless_mode = endless_mode
        self.endless_chunk_size = endless_chunk_size  # items to compress per overflow
        self._endless_episodes: int = 0  # how many compression cycles have run

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

        # Temporal metadata — set when state is loaded from disk
        self.last_saved_at: float | None = None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def _find_patch_target(self, experience: Experience) -> MemoryItem | None:
        """Find an existing memory that matches the patch_key for update_mode='patch'."""
        match_key = experience.patch_key or experience.title
        if match_key is None:
            return None
        for _, store in self._iter_tiers():
            for item in store:
                if item.is_superseded:
                    continue
                exp = item.experience
                if (exp.patch_key and exp.patch_key == match_key) or (
                    exp.title and exp.title == match_key
                ):
                    return item
        return None

    def store(self, experience: Experience) -> MemoryItem:
        """Store an experience — always enters working memory first.

        When ``experience.update_mode == "patch"``, an existing memory with a
        matching ``patch_key`` or ``title`` is found and marked as superseded
        before the new version is inserted.
        """
        # Handle patch mode: find and supersede old memory before inserting new
        superseded_item: MemoryItem | None = None
        if experience.update_mode == "patch":
            superseded_item = self._find_patch_target(experience)

        item = MemoryItem(experience=experience, tier=MemoryTier.WORKING)
        # Rescue pinned items before deque overflow silently evicts them
        if len(self.working) >= self.cfg.working_capacity:
            self._rescue_pinned_from_working()
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

        # Conflict archival: mark the old version as superseded
        if superseded_item is not None:
            superseded_item.superseded_by = item.id

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

        # Fix 10: Auto-detect supersession — if a new memory in the same domain
        # is semantically similar but factually divergent from an existing memory,
        # mark the older one as superseded_by the new one.
        if superseded_item is None:
            self._auto_detect_supersession(item)

        # Endless mode: compress before overflow instead of evicting
        if self.endless_mode and len(self.working) >= self.cfg.working_capacity:
            self._endless_compress()
        elif len(self.working) >= self.cfg.working_capacity:
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

        tier_boost_map = {
            MemoryTier.SEMANTIC: 0.08,
            MemoryTier.LONG_TERM: 0.04,
            MemoryTier.SHORT_TERM: 0.02,
            MemoryTier.WORKING: 0.0,
        }

        if query_vec is not None and self._vec_index is not None:
            # Fast path: VectorIndex batch cosine similarity — O(k log n) vs O(n)
            # Fetch a broad candidate pool then apply per-item bonuses.
            k = min(max(max_results * 10, 50), len(self._items_by_exp_id) or 1)
            top_matches = self._vec_index.query(query_vec, k=k)
            for exp_id, cos_sim in top_matches:
                item = self._items_by_exp_id.get(exp_id)
                if item is None or item.is_expired or item.is_superseded:
                    continue
                if item.experience.private:
                    continue
                tier_boost = tier_boost_map.get(item.tier, 0.0)
                domain_bonus = 0.05 if (query_domain and item.experience.domain == query_domain) else 0.0
                access_bonus = min(0.05, item.access_count * 0.01)
                staleness_penalty = self._staleness_penalty(item)
                score = min(1.0, cos_sim + tier_boost + domain_bonus + access_bonus - staleness_penalty)
                if score > self.cfg.relevance_threshold:
                    item.touch()
                    results.append(RetrievalResult(
                        memory=item, score=score,
                        source_tier=item.tier, strategy="embedding",
                        staleness_warning=self._staleness_warning(item),
                    ))
        else:
            # Lexical path: pre-filter via inverted index then iterate
            candidate_ids: set[str] | None = None
            if self._word_index:
                candidate_ids = set()
                for word in query_words:
                    if word in self._word_index:
                        candidate_ids.update(self._word_index[word])
                    stem = _simple_stem(word)
                    if stem != word and stem in self._word_index:
                        candidate_ids.update(self._word_index[stem])

            for tier, store in self._iter_tiers():
                tier_boost = tier_boost_map.get(tier, 0.0)
                for item in store:
                    if item.is_expired or item.is_superseded:
                        continue
                    if item.experience.private:
                        continue
                    if candidate_ids is not None and item.experience.id not in candidate_ids:
                        continue
                    base_score = self._relevance(item, query_words)
                    domain_bonus = 0.05 if (query_domain and item.experience.domain == query_domain) else 0.0
                    access_bonus = min(0.05, item.access_count * 0.01)
                    staleness_penalty = self._staleness_penalty(item)
                    score = min(1.0, base_score + tier_boost + domain_bonus + access_bonus - staleness_penalty)
                    if score > self.cfg.relevance_threshold:
                        item.touch()
                        results.append(RetrievalResult(
                            memory=item, score=score,
                            source_tier=tier, strategy="lexical",
                            staleness_warning=self._staleness_warning(item),
                        ))

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

    def _staleness_penalty(self, item: "MemoryItem") -> float:
        """Return a retrieval score penalty for unverified inferences/hearsay.

        Epistemic provenance discount:
        - ``observation`` memories (first-hand): no penalty
        - ``reflection`` memories: small penalty if >7 days unverified
        - ``inference`` memories: moderate penalty if >3 days unverified
        - ``hearsay`` memories: large penalty unless recently verified

        The penalty is capped at 0.15 so stale memories can still surface
        when they are the best match — they just rank lower.
        """
        import math as _math
        exp = item.experience
        epistemic_type = getattr(exp, "epistemic_type", "observation")
        verified_at = getattr(exp, "verified_at", None)

        if epistemic_type == "observation":
            return 0.0

        # Base penalty by type
        base: float
        decay_days: float
        if epistemic_type == "reflection":
            base, decay_days = 0.05, 7.0
        elif epistemic_type == "inference":
            base, decay_days = 0.08, 3.0
        else:  # hearsay
            base, decay_days = 0.12, 1.0

        # Grow the penalty with age since last verification
        ref_time = verified_at if verified_at else item.stored_at
        age_days = (time.time() - ref_time) / 86400.0
        growth = 1.0 - _math.exp(-age_days / max(decay_days, 0.1))
        return round(min(0.15, base * (1.0 + growth)), 4)

    def _staleness_warning(self, item: "MemoryItem") -> "str | None":
        """Fix 11: Generate a human-readable warning for unverified/superseded memories.

        Returns None if the memory is healthy (verified observation), otherwise
        a short warning string visible to the reasoning agent at retrieval time.
        """
        exp = item.experience
        epistemic_type = getattr(exp, "epistemic_type", "observation")
        verified_at = getattr(exp, "verified_at", None)

        # Superseded memories get the strongest warning
        if item.superseded_by is not None:
            superseder = self._items_by_exp_id.get(item.superseded_by)
            if superseder:
                snippet = superseder.experience.content[:60]
                return f"\u26a0 Superseded by [{snippet}...]. This may be outdated."
            return "\u26a0 Superseded by a newer memory. This may be outdated."

        # Observations are trusted
        if epistemic_type == "observation":
            return None

        # Unverified hearsay/inference
        if epistemic_type in ("hearsay", "inference") and verified_at is None:
            return f"\u26a0 Never verified ({epistemic_type}). Treat as unconfirmed."

        # Stale verification (>3 days since last check)
        if verified_at is not None:
            days_since = (time.time() - verified_at) / 86400.0
            if days_since > 3.0 and epistemic_type in ("hearsay", "inference"):
                return f"\u26a0 Last verified {days_since:.0f} days ago ({epistemic_type}). May be stale."

        return None

    def _auto_detect_supersession(self, new_item: "MemoryItem") -> None:
        """Fix 10: At store time, detect if new memory supersedes an existing one.

        Searches same-domain memories for high semantic similarity (>0.75).
        When found, checks for content divergence via word-level Jaccard distance.
        If the topic is similar but content differs, marks the older memory as
        superseded — it stays in the store (history matters) but retrieval
        treats superseded memories with a hard discount.
        """
        new_exp = new_item.experience
        new_emb = self._embeddings.get(new_exp.id)
        if new_emb is None:
            return  # no embedding — can't do similarity check

        from emms.core.embeddings import cosine_similarity

        _SIMILARITY_THRESHOLD = 0.75
        _DIVERGENCE_THRESHOLD = 0.40  # word-level Jaccard distance must be > this

        new_words = set(new_exp.content.lower().split())

        for _tier, store in self._iter_tiers():
            for item in store:
                if item.id == new_item.id or item.is_superseded:
                    continue
                if item.experience.domain != new_exp.domain:
                    continue

                old_emb = self._embeddings.get(item.experience.id)
                if old_emb is None:
                    continue

                sim = cosine_similarity(new_emb, old_emb)
                if sim < _SIMILARITY_THRESHOLD:
                    continue

                # High similarity — check if content actually differs
                old_words = set(item.experience.content.lower().split())
                if not old_words or not new_words:
                    continue
                jaccard = len(old_words & new_words) / len(old_words | new_words)
                divergence = 1.0 - jaccard

                if divergence > _DIVERGENCE_THRESHOLD:
                    # Same topic, different content → supersede the older one
                    item.superseded_by = new_item.id

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

    def _rescue_pinned_from_working(self) -> None:
        """Move pinned items directly to long_term before deque overflow evicts them."""
        rescued = [item for item in self.working if item.pinned]
        if not rescued:
            return
        for item in rescued:
            self.working.remove(item)
            item.tier = MemoryTier.LONG_TERM
            item.consolidation_score = max(item.consolidation_score, 0.9)
            self.long_term[item.id] = item
            # Preserve index entries
            self._items_by_exp_id[item.experience.id] = item
        logger.debug("Rescued %d pinned items from working memory overflow", len(rescued))

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

    def _endless_compress(self) -> None:
        """Biomimetic real-time compression for Endless Mode.

        When working memory is full, instead of evicting the oldest items,
        compress the oldest `endless_chunk_size` items into a single merged
        episode and push that to short-term memory. This keeps context growth
        O(N) rather than O(N²) — the hippocampal consolidation analogy.

        The compressed episode carries:
        - A concatenated narrative summary of all source items
        - The highest importance and emotional intensity of the group
        - All source session_ids (first one wins for the episode)
        - obs_type = CHANGE (reflects that this is a synthesis)
        """
        if len(self.working) < self.endless_chunk_size:
            return

        # Take the oldest chunk items from the left of the deque, skipping pinned
        chunk: list[MemoryItem] = []
        skipped_pinned: list[MemoryItem] = []
        for _ in range(min(self.endless_chunk_size, len(self.working))):
            item = self.working.popleft()
            if item.pinned:
                skipped_pinned.append(item)
            else:
                chunk.append(item)
        # Re-insert pinned items at the front
        for item in reversed(skipped_pinned):
            self.working.appendleft(item)

        if not chunk:
            return

        # Build merged summary
        summaries = [f"[{i+1}] {item.experience.content}" for i, item in enumerate(chunk)]
        merged_content = (
            f"[Compressed episode #{self._endless_episodes + 1} | "
            f"{len(chunk)} memories] " + " | ".join(summaries)
        )

        # Aggregate metadata
        max_importance = max(i.experience.importance for i in chunk)
        max_intensity = max(i.experience.emotional_intensity for i in chunk)
        avg_valence = sum(i.experience.emotional_valence for i in chunk) / len(chunk)
        domains = list({i.experience.domain for i in chunk})
        session_id = chunk[0].experience.session_id
        all_entities = list({e for i in chunk for e in i.experience.entities})

        # Build merged experience
        from emms.core.models import ObsType
        compressed_exp = Experience(
            content=merged_content,
            domain=domains[0] if len(domains) == 1 else "multi",
            importance=max_importance,
            emotional_intensity=max_intensity,
            emotional_valence=avg_valence,
            novelty=max(i.experience.novelty for i in chunk),
            session_id=session_id,
            obs_type=ObsType.CHANGE,
            entities=all_entities,
            metadata={
                "compressed": True,
                "source_count": len(chunk),
                "source_ids": [i.experience.id for i in chunk],
                "episode_index": self._endless_episodes,
            },
        )

        episode_item = MemoryItem(
            experience=compressed_exp,
            tier=MemoryTier.SHORT_TERM,
            memory_strength=max(i.memory_strength for i in chunk),
            consolidation_score=max_importance,
        )
        self.short_term.append(episode_item)
        self._items_by_exp_id[compressed_exp.id] = episode_item
        self._endless_episodes += 1
        self.total_consolidated += len(chunk)

        logger.debug(
            "Endless Mode: compressed %d items into episode #%d",
            len(chunk), self._endless_episodes,
        )

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
            elif item.pinned:
                # Pinned items always promote to long_term, never forgotten
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
        """Content-based relevance using BM25 + importance/recency boosts.

        Replaces plain Jaccard with BM25 (k1=1.5, b=0.75) using the
        word-index for approximate document-frequency stats. BM25 handles
        term-frequency saturation and document-length normalization, giving
        meaningfully better ranking than Jaccard on natural-language queries.
        """
        content = item.experience.content.lower()
        content_words = content.split()
        doc_len = len(content_words)

        # BM25 parameters (standard TREC values)
        k1 = 1.5
        b = 0.75
        # Approximate average document length from word index size
        n_docs = max(1, len(self._items_by_exp_id))
        avg_dl = max(1, sum(
            len(i.experience.content.split())
            for i in self._items_by_exp_id.values()
        ) / n_docs) if n_docs <= 200 else 30  # cap expensive scan at 200 docs

        bm25_score = 0.0
        for qw in query_words:
            tf = content_words.count(qw)
            if tf == 0:
                stem = _simple_stem(qw)
                tf = content_words.count(stem)
            if tf == 0:
                continue
            # Document frequency from word index
            df = len(self._word_index.get(qw, set()) | self._word_index.get(_simple_stem(qw), set()))
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
            bm25_score += idf * tf_norm

        # Normalize to [0, 1] with a soft cap
        n_terms = max(1, len(query_words))
        normalized = min(1.0, bm25_score / (n_terms * 3.0))

        # Importance boost (0–0.2)
        importance_boost = item.experience.importance * 0.2

        # Recency boost (0–0.1), exponential decay
        age = item.age
        recency_boost = 0.1 * np.exp(-age / self.cfg.decay_half_life_seconds)

        # Strength boost (0–0.1)
        strength_boost = item.memory_strength * 0.1

        return min(1.0, normalized + importance_boost + recency_boost + strength_boost)

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

        # Collect only live experience IDs to prune orphan embeddings.
        # Orphans accumulate when memories are evicted/consolidated but their
        # embedding vectors remain in _embeddings. Without pruning, the dict
        # grows unboundedly and inflates save/load time.
        live_exp_ids: set[str] = set()
        for _, store in self._iter_tiers():
            for item in store:
                live_exp_ids.add(item.experience.id)

        state = {
            "version": "0.4.0",
            "saved_at": time.time(),
            "working": _serialize_items(self.working),
            "short_term": _serialize_items(self.short_term),
            "long_term": _serialize_items(self.long_term.values()),
            "semantic": _serialize_items(self.semantic.values()),
            "embeddings": {k: v for k, v in self._embeddings.items() if k in live_exp_ids},
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

        # Capture when this state was last saved (for elapsed-time awareness)
        self.last_saved_at = data.get("saved_at")

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

                # Rebuild VectorIndex from restored embeddings.
                # Guard against dimension mismatch (e.g. state saved with
                # HashEmbedder 128-dim then loaded with SentenceTransformer
                # 384-dim).  Re-embed stale vectors using the current embedder.
                if self._vec_index is not None and item.experience.id in self._embeddings:
                    stored_emb = self._embeddings[item.experience.id]
                    if len(stored_emb) == self._vec_index.dim:
                        self._vec_index.add(item.experience.id, stored_emb)
                    elif self.embedder is not None:
                        new_emb = self.embedder.embed(item.experience.content)
                        self._embeddings[item.experience.id] = new_emb
                        self._vec_index.add(item.experience.id, new_emb)
                    # else: no embedder, no vec index entry (lexical fallback)

        # Restore stats
        stats = data.get("stats", {})
        self.total_stored = stats.get("total_stored", 0)
        self.total_consolidated = stats.get("total_consolidated", 0)

        logger.info("Memory state loaded from %s (%d items)", path, self.size["total"])

    # ------------------------------------------------------------------
    # Progressive disclosure helpers (claude-mem inspired)
    # ------------------------------------------------------------------

    def get_timeline(
        self,
        session_id: str | None = None,
        limit: int = 50,
        include_private: bool = False,
    ) -> list[MemoryItem]:
        """Layer 2 of progressive disclosure: chronological memory timeline.

        Returns all items sorted by timestamp (oldest first), optionally
        filtered to a single session. Mirrors claude-mem's timeline() tool
        which provides chronological context around search results.

        Args:
            session_id: If provided, only return items from this session.
            limit: Maximum items to return.
            include_private: If False (default), skip private experiences.
        """
        all_items: list[MemoryItem] = []
        for _, store in self._iter_tiers():
            for item in store:
                if not include_private and item.experience.private:
                    continue
                if session_id is not None and item.experience.session_id != session_id:
                    continue
                all_items.append(item)

        all_items.sort(key=lambda i: i.experience.timestamp)
        return all_items[:limit]

    def get_sessions(self) -> list[str]:
        """Return all distinct session IDs stored in memory (excluding None)."""
        seen: set[str] = set()
        for _, store in self._iter_tiers():
            for item in store:
                sid = item.experience.session_id
                if sid is not None:
                    seen.add(sid)
        return sorted(seen)

    def search_by_file(self, file_path: str) -> list[MemoryItem]:
        """Find all memories that reference a specific file path.

        Searches both ``files_read`` and ``files_modified`` on each stored
        experience.  Returns results sorted newest-first by timestamp.

        Args:
            file_path: Exact or partial file path to search for.

        Returns:
            MemoryItems whose experience references *file_path*.
        """
        results: list[MemoryItem] = []
        for _, store in self._iter_tiers():
            for item in store:
                exp = item.experience
                # Support both exact list membership and substring matching
                in_read = any(file_path in f for f in exp.files_read)
                in_modified = any(file_path in f for f in exp.files_modified)
                if in_read or in_modified:
                    results.append(item)
        results.sort(key=lambda i: i.experience.timestamp, reverse=True)
        return results

    def retrieve_filtered(
        self,
        query: str,
        max_results: int = 10,
        *,
        namespace: str | None = None,
        obs_type: "ObsType | None" = None,
        domain: str | None = None,
        session_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
        sort_by: str = "relevance",
        concept_tags: "list | None" = None,
        include_superseded: bool = False,
        include_expired: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve with structured pre-filters applied before scoring.

        Provides fine-grained control over which memories are considered,
        combining the full scoring pipeline with explicit field filters.

        Args:
            query: Natural-language search query.
            max_results: Maximum results to return.
            namespace: Only return memories in this namespace (e.g. ``"project-x"``).
            obs_type: Filter to a specific observation type (e.g. ``ObsType.BUGFIX``).
            domain: Only return memories matching this domain string.
            session_id: Restrict to memories from a specific session.
            since: Only include memories stored after this Unix timestamp.
            until: Only include memories stored before this Unix timestamp.
            min_confidence: Only include memories with confidence ≥ this value.
            min_importance: Only include memories with importance ≥ this value.
            sort_by: How to rank results when ``query`` is empty.
                ``"relevance"`` (default) sorts by importance × confidence.
                ``"recency"`` sorts by stored_at descending (most recent first).
                ``"importance"`` sorts by importance descending.
                Ignored when a non-empty query is provided (semantic/lexical score ranks).
            include_superseded: If False (default), skip superseded memories.
            include_expired: If False (default), skip TTL-expired memories.

        Returns:
            Scored and filtered RetrievalResult list.
        """
        # Build a filtered candidate pool first, then run normal scoring
        candidate_items: list[MemoryItem] = []
        for _, store in self._iter_tiers():
            for item in store:
                exp = item.experience
                # Lifecycle filters
                if not include_expired and item.is_expired:
                    continue
                if not include_superseded and item.is_superseded:
                    continue
                if item.experience.private:
                    continue
                # Field filters
                if namespace is not None and exp.namespace != namespace:
                    continue
                if obs_type is not None and exp.obs_type != obs_type:
                    continue
                if domain is not None and exp.domain != domain:
                    continue
                if session_id is not None and exp.session_id != session_id:
                    continue
                if since is not None and exp.timestamp < since:
                    continue
                if until is not None and exp.timestamp > until:
                    continue
                if min_confidence is not None and exp.confidence < min_confidence:
                    continue
                if min_importance is not None and exp.importance < min_importance:
                    continue
                if concept_tags is not None and not any(t in exp.concept_tags for t in concept_tags):
                    continue
                candidate_items.append(item)

        if not candidate_items:
            return []

        # Filter-only mode: empty query → skip semantic scoring, order by sort_by
        if not query.strip():
            if sort_by == "recency":
                candidate_items.sort(key=lambda i: i.experience.timestamp, reverse=True)
                score_fn = lambda i: i.experience.timestamp  # noqa: E731
                strategy = "filtered+recency"
            elif sort_by == "importance":
                candidate_items.sort(key=lambda i: i.experience.importance, reverse=True)
                score_fn = lambda i: i.experience.importance  # noqa: E731
                strategy = "filtered+importance"
            else:  # "relevance" (default)
                candidate_items.sort(
                    key=lambda i: i.experience.importance * i.experience.confidence,
                    reverse=True,
                )
                score_fn = lambda i: i.experience.importance * i.experience.confidence  # noqa: E731
                strategy = "filtered+relevance"
            return [
                RetrievalResult(
                    memory=item,
                    score=score_fn(item),
                    source_tier=item.tier,
                    strategy=strategy,
                )
                for item in candidate_items[:max_results]
            ]

        # Score candidates using the full retrieval pipeline
        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed(query)
        query_words = set(query.lower().split())
        query_domain = self._infer_domain(query_words)

        results: list[RetrievalResult] = []
        for item in candidate_items:
            tier = item.tier
            tier_boost = {
                MemoryTier.SEMANTIC: 0.08,
                MemoryTier.LONG_TERM: 0.04,
                MemoryTier.SHORT_TERM: 0.02,
                MemoryTier.WORKING: 0.0,
            }.get(tier, 0.0)

            if query_vec is not None:
                base_score = self._embedding_relevance(item, query_vec)
            else:
                base_score = self._relevance(item, query_words)

            domain_bonus = 0.05 if (query_domain and item.experience.domain == query_domain) else 0.0
            access_bonus = min(0.05, item.access_count * 0.01)
            # Confidence scaling: low-confidence memories get a penalty
            confidence_scale = 0.5 + 0.5 * item.experience.confidence
            score = min(1.0, (base_score + tier_boost + domain_bonus + access_bonus) * confidence_scale)

            if score > self.cfg.relevance_threshold:
                item.touch()
                results.append(RetrievalResult(
                    memory=item,
                    score=score,
                    source_tier=tier,
                    strategy="filtered+embedding" if query_vec else "filtered+lexical",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    async def stream_retrieve(
        self,
        query: str,
        max_results: int = 10,
    ):
        """Async generator that yields RetrievalResult items tier-by-tier.

        Results are emitted as they are scored, from highest-priority tier
        (semantic) to lowest (working). Within each tier results are yielded
        in descending score order.

        Cooperative multitasking is maintained with ``asyncio.sleep(0)``
        between tier boundaries so other tasks can run.

        Usage::

            async for result in memory.stream_retrieve("machine learning"):
                print(result.memory.experience.content, result.score)
        """
        import asyncio as _asyncio

        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed(query)
        query_words = set(query.lower().split())
        query_domain = self._infer_domain(query_words)

        emitted = 0
        for tier, store in self._iter_tiers():
            if emitted >= max_results:
                break

            tier_boost = {
                MemoryTier.SEMANTIC: 0.08,
                MemoryTier.LONG_TERM: 0.04,
                MemoryTier.SHORT_TERM: 0.02,
                MemoryTier.WORKING: 0.0,
            }.get(tier, 0.0)

            tier_results: list[RetrievalResult] = []
            for item in store:
                if item.is_expired or item.is_superseded:
                    continue
                if query_vec is not None:
                    base_score = self._embedding_relevance(item, query_vec)
                else:
                    base_score = self._relevance(item, query_words)
                domain_bonus = 0.05 if (query_domain and item.experience.domain == query_domain) else 0.0
                access_bonus = min(0.05, item.access_count * 0.01)
                score = min(1.0, base_score + tier_boost + domain_bonus + access_bonus)
                if score > self.cfg.relevance_threshold:
                    item.touch()
                    tier_results.append(RetrievalResult(
                        memory=item,
                        score=score,
                        source_tier=tier,
                        strategy="stream+embedding" if query_vec else "stream+lexical",
                    ))

            tier_results.sort(key=lambda r: r.score, reverse=True)
            for r in tier_results:
                if emitted >= max_results:
                    return
                yield r
                emitted += 1

            # Cooperative yield between tiers
            await _asyncio.sleep(0)

    def upvote(self, memory_id: str, boost: float = 0.1) -> bool:
        """Strengthen a memory via positive user feedback.

        Raises the memory's strength by *boost* (capped at 1.0) and records
        an access. Useful for reinforcement: "this retrieval was helpful".

        Args:
            memory_id: The ``MemoryItem.id`` to upvote.
            boost: Strength increment (default 0.1).

        Returns:
            True if the memory was found and updated, False otherwise.
        """
        for _, store in self._iter_tiers():
            for item in store:
                if item.id == memory_id or item.experience.id == memory_id:
                    item.memory_strength = min(1.0, item.memory_strength + boost)
                    item.touch()
                    logger.debug("Upvoted %s → strength=%.3f", memory_id, item.memory_strength)
                    return True
        return False

    def downvote(self, memory_id: str, decay: float = 0.2) -> bool:
        """Weaken a memory via negative user feedback.

        Reduces the memory's strength by *decay* (floored at 0.0).
        Used when a retrieved memory was irrelevant or incorrect.

        Args:
            memory_id: The ``MemoryItem.id`` to downvote.
            decay: Strength reduction (default 0.2).

        Returns:
            True if the memory was found and updated, False otherwise.
        """
        for _, store in self._iter_tiers():
            for item in store:
                if item.id == memory_id or item.experience.id == memory_id:
                    item.memory_strength = max(0.0, item.memory_strength - decay)
                    logger.debug("Downvoted %s → strength=%.3f", memory_id, item.memory_strength)
                    return True
        return False

    def export_markdown(
        self,
        path: "Path | str",
        include_private: bool = False,
        namespace: str | None = None,
    ) -> int:
        """Export memories as a structured, human-readable Markdown document.

        Groups memories by domain and tier. Each memory is rendered with its
        title, facts, key metadata, and content — suitable for human review,
        version control diffing, or feeding to an LLM as context.

        Args:
            path: Destination .md file path.
            include_private: If False (default), skip private experiences.
            namespace: If provided, only export memories from this namespace.

        Returns:
            Number of memories exported.
        """
        import datetime as _dt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all items, apply filters
        all_items: list[MemoryItem] = []
        for _, store in self._iter_tiers():
            for item in store:
                if not include_private and item.experience.private:
                    continue
                if namespace is not None and item.experience.namespace != namespace:
                    continue
                all_items.append(item)

        # Sort by timestamp (oldest first)
        all_items.sort(key=lambda i: i.experience.timestamp)

        # Group by domain
        by_domain: dict[str, list[MemoryItem]] = {}
        for item in all_items:
            d = item.experience.domain
            if d not in by_domain:
                by_domain[d] = []
            by_domain[d].append(item)

        lines: list[str] = [
            "# EMMS Memory Export",
            "",
            f"> Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"> Total memories: {len(all_items)}  ",
            f"> Domains: {', '.join(sorted(by_domain.keys()))}",
            "",
        ]

        for domain, items in sorted(by_domain.items()):
            lines += [f"## {domain.capitalize()} ({len(items)} memories)", ""]
            for item in items:
                exp = item.experience
                dt = _dt.datetime.fromtimestamp(exp.timestamp).strftime("%Y-%m-%d %H:%M")
                tier_label = item.tier.value.replace("_", " ").title()
                obs = f" · {exp.obs_type.value}" if exp.obs_type else ""
                conf = f" · conf={exp.confidence:.2f}" if exp.confidence < 1.0 else ""

                title = exp.title or exp.content[:60].rstrip()
                lines += [
                    f"### {title}",
                    f"*{dt} · {tier_label}{obs}{conf} · strength={item.memory_strength:.2f}*",
                    "",
                ]

                if exp.facts:
                    for fact in exp.facts:
                        lines.append(f"- {fact}")
                    lines.append("")

                # Content (truncated if very long)
                content = exp.content
                if len(content) > 500:
                    content = content[:497] + "…"
                lines += [f"> {content}", ""]

                if exp.files_read:
                    lines.append(f"**Files read:** `{'`, `'.join(exp.files_read)}`  ")
                if exp.files_modified:
                    lines.append(f"**Files modified:** `{'`, `'.join(exp.files_modified)}`  ")
                if exp.files_read or exp.files_modified:
                    lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown export: %d memories → %s", len(all_items), path)
        return len(all_items)

    def export_jsonl(
        self,
        path: "Path | str",
        include_private: bool = False,
    ) -> int:
        """Export all memories as newline-delimited JSON (JSONL format).

        claude-mem stores observations in JSONL for human readability and
        version-control friendliness. Each line is one MemoryItem serialised
        to JSON — easy to grep, diff, and process with standard tools.

        Args:
            path: Destination file path.
            include_private: If False (default), private experiences are omitted.

        Returns:
            Number of items written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as fh:
            for _, store in self._iter_tiers():
                for item in store:
                    if not include_private and item.experience.private:
                        continue
                    fh.write(item.model_dump_json() + "\n")
                    count += 1
        logger.info("Exported %d memories to %s", count, path)
        return count

    def import_jsonl(self, path: "Path | str") -> int:
        """Import memories from a JSONL file previously exported by export_jsonl().

        Skips items whose experience IDs already exist in memory.

        Returns:
            Number of items imported.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("JSONL file not found: %s", path)
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = MemoryItem.model_validate_json(line)
                if item.experience.id in self._items_by_exp_id:
                    continue  # already loaded
                # Re-store the experience so indexes are rebuilt correctly
                self.store(item.experience)
                count += 1
        logger.info("Imported %d memories from %s", path, count)
        return count
