"""Multi-strategy ensemble retrieval for EMMS.

Ported from the monolithic EMMS.py — five complementary retrieval strategies
combined via weighted ensemble scoring.

Strategies:
    1. SemanticStrategy   — embedding cosine similarity
    2. TemporalStrategy   — recency + access pattern scoring
    3. EmotionalStrategy  — emotional resonance matching
    4. GraphStrategy      — entity/relationship overlap
    5. DomainStrategy     — domain affinity scoring

Usage::

    retriever = EnsembleRetriever()
    retriever.add_strategy(SemanticStrategy(embedder), weight=0.35)
    retriever.add_strategy(TemporalStrategy(), weight=0.20)
    results = retriever.retrieve(query, items, max_results=10)
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from emms.core.models import (
    CompactResult,
    ConceptTag,
    Experience,
    MemoryItem,
    MemoryTier,
    ObsType,
    RetrievalResult,
)
from emms.core.embeddings import EmbeddingProvider, cosine_similarity


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RetrievalStrategy(Protocol):
    """Interface for a single retrieval scoring strategy."""

    name: str

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        """Return relevance score in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class SemanticStrategy:
    """Embedding cosine similarity scoring."""

    name = "semantic"

    # Max embedding entries to keep per retriever instance.
    # 384-dim float64 ≈ 3 KB each → 2 000 entries ≈ 6 MB ceiling.
    _CACHE_MAX = 2_000

    def __init__(self, embedder: EmbeddingProvider | None = None):
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        if self.embedder is None:
            return self._lexical_fallback(query, item)

        # Get query vector (cached in context)
        query_vec = context.get("query_vec")
        if query_vec is None:
            query_vec = self.embedder.embed(query)
            context["query_vec"] = query_vec

        # Get item vector
        item_id = item.experience.id
        item_vec = context.get("embeddings", {}).get(item_id)
        if item_vec is None:
            item_vec = self._cache.get(item_id)
        if item_vec is None:
            item_vec = self.embedder.embed(item.experience.content)
            # Evict oldest entry when cap reached (insertion-ordered dict)
            if len(self._cache) >= self._CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[item_id] = item_vec

        sim = cosine_similarity(query_vec, item_vec)
        return max(0.0, sim)

    def _lexical_fallback(self, query: str, item: MemoryItem) -> float:
        """Word overlap when no embedder is available."""
        qw = set(query.lower().split())
        cw = set(item.experience.content.lower().split())
        inter = len(qw & cw)
        union = len(qw | cw)
        return inter / union if union else 0.0


class TemporalStrategy:
    """Recency and access-pattern scoring."""

    name = "temporal"

    def __init__(self, half_life: float = 86_400.0):
        self.half_life = half_life

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        now = time.time()
        age = now - item.stored_at

        # Recency: exponential decay
        recency = math.exp(-0.693 * age / self.half_life)

        # Access frequency: popular memories surface
        access_score = min(1.0, item.access_count / 10.0)

        # Memory strength
        strength = item.memory_strength

        return recency * 0.5 + access_score * 0.25 + strength * 0.25


class EmotionalStrategy:
    """Emotional resonance matching."""

    name = "emotional"

    # Keywords mapped to (valence, intensity) hints
    _EMOTION_WORDS: dict[str, tuple[float, float]] = {
        "happy": (0.8, 0.6), "sad": (-0.7, 0.6), "angry": (-0.8, 0.9),
        "excited": (0.9, 0.8), "fear": (-0.6, 0.8), "surprise": (0.3, 0.7),
        "love": (0.9, 0.7), "hate": (-0.9, 0.8), "anxious": (-0.5, 0.7),
        "calm": (0.3, 0.2), "joy": (0.9, 0.7), "grief": (-0.8, 0.9),
        "hope": (0.6, 0.5), "despair": (-0.9, 0.8), "proud": (0.7, 0.6),
        "ashamed": (-0.6, 0.7), "grateful": (0.8, 0.5), "jealous": (-0.5, 0.6),
        "content": (0.5, 0.3), "frustrated": (-0.6, 0.7),
        "crisis": (-0.5, 0.8), "success": (0.8, 0.6), "failure": (-0.7, 0.7),
        "growth": (0.6, 0.5), "decline": (-0.5, 0.5), "breakthrough": (0.8, 0.8),
    }

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        # Infer query emotional tone
        q_valence, q_intensity = self._infer_emotion(query)
        i_valence = item.experience.emotional_valence
        i_intensity = item.experience.emotional_intensity

        # Emotional resonance = similarity in emotional space
        valence_match = 1.0 - abs(q_valence - i_valence) / 2.0
        intensity_match = 1.0 - abs(q_intensity - i_intensity)

        # If query has no emotional signal, return neutral
        if q_intensity < 0.1:
            return 0.3  # neutral baseline

        return valence_match * 0.6 + intensity_match * 0.4

    def _infer_emotion(self, text: str) -> tuple[float, float]:
        """Infer emotional valence and intensity from text."""
        words = text.lower().split()
        valences, intensities = [], []
        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if clean in self._EMOTION_WORDS:
                v, i = self._EMOTION_WORDS[clean]
                valences.append(v)
                intensities.append(i)
        if not valences:
            return 0.0, 0.0
        return float(np.mean(valences)), float(np.mean(intensities))


class GraphStrategy:
    """Entity/relationship overlap scoring."""

    name = "graph"

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        graph = context.get("graph")
        if graph is None:
            return 0.0

        # Extract query entities
        query_entities = context.get("query_entities")
        if query_entities is None:
            query_entities = set()
            for word in query.split():
                clean = re.sub(r'[^\w]', '', word)
                if clean.lower() in graph.entities:
                    query_entities.add(clean.lower())
            context["query_entities"] = query_entities

        if not query_entities:
            return 0.0

        # Check entity overlap with item
        item_entities = {e.lower() for e in item.experience.entities}
        if not item_entities:
            return 0.0

        overlap = len(query_entities & item_entities)
        union = len(query_entities | item_entities)

        return overlap / union if union else 0.0


class ImportanceStrategy:
    """LangMem-inspired: weight retrieval by memory importance + strength.

    LangMem states: "Memory relevance is more than just semantic similarity.
    Recall should combine similarity with importance of the memory, as well
    as the memory's strength."  This strategy scores on those two dimensions
    alone so it can be blended into any ensemble.
    """

    name = "importance"

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        imp = item.experience.importance  # [0, 1]
        strength = item.memory_strength   # [0, 1], decays over time
        return imp * 0.6 + strength * 0.4


class DomainStrategy:
    """Domain affinity scoring."""

    name = "domain"

    _DOMAIN_KEYWORDS: dict[str, set[str]] = {
        "finance": {"stock", "market", "price", "trading", "investment", "rate", "earnings", "revenue", "profit", "economic", "financial", "crypto", "bitcoin", "bank"},
        "tech": {"software", "code", "programming", "algorithm", "computer", "api", "data", "model", "neural", "machine", "learning", "technology", "cloud", "server"},
        "science": {"research", "study", "experiment", "theory", "quantum", "physics", "biology", "chemistry", "scientific", "discovery", "hypothesis"},
        "weather": {"weather", "temperature", "rain", "storm", "climate", "forecast", "wind", "humidity", "snow"},
        "health": {"health", "medical", "disease", "treatment", "patient", "clinical", "drug", "therapy", "diagnosis", "vaccine"},
    }

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        query_domain = context.get("query_domain")
        if query_domain is None:
            query_domain = self._infer_domain(query)
            context["query_domain"] = query_domain

        if query_domain is None:
            return 0.3  # no domain signal

        if item.experience.domain == query_domain:
            return 1.0
        return 0.2

    def _infer_domain(self, text: str) -> str | None:
        words = set(text.lower().split())
        best, best_overlap = None, 0
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            overlap = len(words & keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best = domain
        return best if best_overlap > 0 else None


# ---------------------------------------------------------------------------
# Ensemble retriever
# ---------------------------------------------------------------------------

class EnsembleRetriever:
    """Combines multiple strategies with configurable weights.

    Usage::

        retriever = EnsembleRetriever()
        retriever.add_strategy(SemanticStrategy(embedder), weight=0.35)
        retriever.add_strategy(TemporalStrategy(), weight=0.20)
        retriever.add_strategy(EmotionalStrategy(), weight=0.15)
        retriever.add_strategy(GraphStrategy(), weight=0.15)
        retriever.add_strategy(DomainStrategy(), weight=0.15)
        results = retriever.retrieve("market trends", items, max_results=10)
    """

    def __init__(self) -> None:
        self.strategies: list[tuple[RetrievalStrategy, float]] = []

    def add_strategy(self, strategy: RetrievalStrategy, weight: float = 1.0) -> None:
        """Add a strategy with a weight."""
        self.strategies.append((strategy, weight))

    def retrieve(
        self,
        query: str,
        items: Sequence[MemoryItem],
        max_results: int = 10,
        context: dict[str, Any] | None = None,
        relevance_threshold: float = 0.3,
    ) -> list[RetrievalResult]:
        """Score all items using the ensemble and return top results."""
        if not self.strategies:
            return []

        ctx = context or {}
        total_weight = sum(w for _, w in self.strategies)
        if total_weight == 0:
            return []

        scored: list[tuple[float, MemoryItem, dict[str, float]]] = []

        for item in items:
            strategy_scores: dict[str, float] = {}
            combined = 0.0

            for strategy, weight in self.strategies:
                s = strategy.score(query, item, ctx)
                strategy_scores[strategy.name] = s
                combined += s * weight

            combined /= total_weight
            combined = min(1.0, combined)

            if combined >= relevance_threshold:
                scored.append((combined, item, strategy_scores))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievalResult] = []
        for score, item, strategy_scores in scored[:max_results]:
            item.touch()
            # Build human-readable explanation from top contributing strategies
            top = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation = " + ".join(f"{k}={v:.2f}" for k, v in top)
            results.append(RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="ensemble",
                strategy_scores=strategy_scores,
                explanation=explanation,
            ))

        return results

    def search_compact(
        self,
        query: str,
        items: Sequence[MemoryItem],
        max_results: int = 20,
        context: dict[str, Any] | None = None,
        relevance_threshold: float = 0.3,
        snippet_length: int = 120,
        obs_type: ObsType | None = None,
        concept_tags: list[ConceptTag] | None = None,
    ) -> list[CompactResult]:
        """Layer 1 of progressive disclosure: return a compact index of matches.

        Inspired by claude-mem's 3-layer retrieval pattern. Each CompactResult
        costs ~50-80 tokens — scan many memories cheaply here, then call
        get_full() only for the IDs you actually need (Layer 3).

        Args:
            obs_type: If set, only return memories with this observation type.
            concept_tags: If set, only return memories carrying ALL listed tags.
            Private experiences (experience.private=True) are never returned.
        """
        # filter private, then optionally by obs_type and concept_tags
        visible: list[MemoryItem] = []
        for item in items:
            exp = item.experience
            if exp.private:
                continue
            if obs_type is not None and exp.obs_type != obs_type:
                continue
            if concept_tags:
                item_tags = set(exp.concept_tags)
                if not all(t in item_tags for t in concept_tags):
                    continue
            visible.append(item)

        full_results = self.retrieve(
            query, visible, max_results=max_results,
            context=context, relevance_threshold=relevance_threshold,
        )
        compact: list[CompactResult] = []
        for r in full_results:
            exp = r.memory.experience

            # Prefer title + first fact for compact snippet; fall back to raw content
            if exp.title:
                detail = exp.facts[0] if exp.facts else exp.content[:80]
                snippet = f"{exp.title} | {detail}"
                if len(snippet) > snippet_length + 40:
                    snippet = snippet[:snippet_length + 37] + "…"
            else:
                snippet = exp.content[:snippet_length]
                if len(exp.content) > snippet_length:
                    snippet += "…"

            # Approximate token cost (words * 1.3 accounts for subword tokens)
            token_estimate = int(len(exp.content.split()) * 1.3)

            compact.append(CompactResult(
                id=r.memory.id,
                snippet=snippet,
                domain=exp.domain,
                score=round(r.score, 3),
                tier=r.source_tier,
                session_id=exp.session_id,
                timestamp=exp.timestamp,
                obs_type=exp.obs_type,
                concept_tags=list(exp.concept_tags),
                token_estimate=token_estimate,
            ))
        return compact

    def get_full(
        self,
        ids: list[str],
        items: Sequence[MemoryItem],
    ) -> list[MemoryItem]:
        """Layer 3 of progressive disclosure: fetch full MemoryItems by ID.

        Use after search_compact() to retrieve only the memories you need.
        Private memories are excluded regardless of ID.
        """
        id_set = set(ids)
        return [
            item for item in items
            if item.id in id_set and not item.experience.private
        ]

    # ------------------------------------------------------------------
    # Factory presets
    # ------------------------------------------------------------------

    @classmethod
    def from_balanced(
        cls,
        embedder: EmbeddingProvider | None = None,
    ) -> "EnsembleRetriever":
        """Create a retriever with claude-mem's recommended weighting + LangMem importance.

        Weights:
            60% — SemanticStrategy    (meaning-based, highest signal)
            20% — TemporalStrategy    (recency + access frequency)
            10% — ImportanceStrategy  (LangMem: importance × memory_strength)
            10% — DomainStrategy      (subject-area affinity)
        """
        retriever = cls()
        retriever.add_strategy(SemanticStrategy(embedder=embedder), weight=0.60)
        retriever.add_strategy(TemporalStrategy(), weight=0.20)
        retriever.add_strategy(ImportanceStrategy(), weight=0.10)
        retriever.add_strategy(DomainStrategy(), weight=0.10)
        return retriever

    @classmethod
    def from_identity(
        cls,
        embedder: EmbeddingProvider | None = None,
    ) -> "EnsembleRetriever":
        """Create a retriever tuned for identity/consciousness workloads.

        Weights:
            30% — SemanticStrategy    (content similarity)
            20% — TemporalStrategy    (narrative arc)
            15% — ImportanceStrategy  (LangMem: importance × strength)
            15% — EmotionalStrategy   (emotionally salient memories)
            10% — GraphStrategy       (entity/relationship coherence)
            10% — DomainStrategy      (domain consistency)
        """
        retriever = cls()
        retriever.add_strategy(SemanticStrategy(embedder=embedder), weight=0.30)
        retriever.add_strategy(TemporalStrategy(), weight=0.20)
        retriever.add_strategy(ImportanceStrategy(), weight=0.15)
        retriever.add_strategy(EmotionalStrategy(), weight=0.15)
        retriever.add_strategy(GraphStrategy(), weight=0.10)
        retriever.add_strategy(DomainStrategy(), weight=0.10)
        return retriever


# ---------------------------------------------------------------------------
# ChromaSemanticStrategy — semantic retrieval via ChromaDB (optional)
# ---------------------------------------------------------------------------

try:
    from emms.storage.chroma import ChromaStore, _HAS_CHROMA  # type: ignore[attr-defined]
except ImportError:
    _HAS_CHROMA = False
    ChromaStore = None  # type: ignore[assignment,misc]


class ChromaSemanticStrategy:
    """SemanticStrategy backed by ChromaDB for high-fidelity vector search.

    Requires: pip install chromadb

    This strategy uses ChromaStore's HNSW index instead of the in-process
    VectorIndex, giving much better recall on large memory stores (>10k items).
    Falls back to lexical overlap if chromadb is not installed.

    Usage::

        store = ChromaStore(embedder=HashEmbedder(), persist_directory="~/.emms/chroma")
        retriever = EnsembleRetriever.from_balanced(embedder=HashEmbedder())
        # Replace the semantic strategy:
        retriever.strategies[0] = (ChromaSemanticStrategy(store), 0.70)
    """

    name = "chroma_semantic"

    def __init__(self, chroma_store: "ChromaStore | None" = None):
        if not _HAS_CHROMA:
            logger.warning(
                "chromadb not installed — ChromaSemanticStrategy falls back to "
                "lexical overlap. Run: pip install chromadb"
            )
        self._store = chroma_store
        self._query_cache: dict[str, list[dict]] = {}

    def score(
        self,
        query: str,
        item: MemoryItem,
        context: dict[str, Any],
    ) -> float:
        if self._store is None or not _HAS_CHROMA:
            # lexical fallback
            qw = set(query.lower().split())
            cw = set(item.experience.content.lower().split())
            inter = len(qw & cw)
            union = len(qw | cw)
            return inter / union if union else 0.0

        # Batch query once per unique query string per retrieve() call
        chroma_results = context.get("_chroma_results")
        if chroma_results is None:
            chroma_results = {
                r["id"]: r["score"]
                for r in self._store.query(query, n_results=200)
            }
            context["_chroma_results"] = chroma_results

        return chroma_results.get(item.experience.id, 0.0)
