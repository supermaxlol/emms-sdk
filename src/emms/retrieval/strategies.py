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
    Experience,
    MemoryItem,
    MemoryTier,
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
            results.append(RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="ensemble",
            ))

        return results
