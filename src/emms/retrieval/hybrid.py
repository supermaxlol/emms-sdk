"""Hybrid retrieval: BM25 lexical + embedding cosine fused via Reciprocal Rank Fusion.

This module provides zero-dependency hybrid search that combines two complementary
retrieval signals:

1. **BM25** (Robertson et al., 1994) — term-frequency/inverse-document-frequency
   lexical matching with k1 and b saturation parameters.
2. **Embedding cosine similarity** — dense vector semantic matching via the
   EMMS EmbeddingProvider interface (falls back to hash embeddings if none is set).

The two ranked lists are fused via **Reciprocal Rank Fusion** (Cormack et al., 2009):

    RRF(d, k=60) = Σ  1 / (k + rank_i(d))

RRF is rank-position-based, making it robust to differences in scale between BM25
and cosine scores without any normalisation.

Usage::

    from emms import EMMS, Experience
    from emms.retrieval.hybrid import HybridRetriever

    agent = EMMS()
    agent.store(Experience(content="Python decorators tutorial", domain="code"))
    agent.store(Experience(content="Advanced metaclass patterns", domain="code"))

    retriever = HybridRetriever(agent.memory)
    results = retriever.retrieve("python decorators", max_results=5)
    for r in results:
        print(r.score, r.memory.experience.content)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from emms.core.models import MemoryItem, MemoryTier, RetrievalResult


# ---------------------------------------------------------------------------
# BM25 implementation (pure Python, zero dependencies)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lower-case word tokeniser."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25:
    """Sparse BM25 over a fixed corpus of MemoryItems."""

    def __init__(
        self,
        items: list[MemoryItem],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.items = items
        self.k1 = k1
        self.b = b
        self._build(items)

    def _build(self, items: list[MemoryItem]) -> None:
        """Pre-compute IDF and per-document term frequencies."""
        self._docs: list[list[str]] = []
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []

        for item in items:
            tokens = _tokenise(item.experience.content)
            self._docs.append(tokens)
            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            self._tf.append(dict(tf))
            self._dl.append(len(tokens))

        self._n = len(items)
        self._avgdl = sum(self._dl) / max(1, self._n)

        # IDF: log( (N - df + 0.5) / (df + 0.5) + 1 ) — smoothed Robertson IDF
        df: dict[str, int] = defaultdict(int)
        for tf_map in self._tf:
            for tok in tf_map:
                df[tok] += 1
        self._idf: dict[str, float] = {}
        for tok, freq in df.items():
            self._idf[tok] = math.log(
                (self._n - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def scores(self, query: str) -> list[float]:
        """Return BM25 scores for all documents."""
        q_tokens = _tokenise(query)
        result: list[float] = []
        for idx in range(self._n):
            score = 0.0
            dl = self._dl[idx]
            tf_map = self._tf[idx]
            for tok in q_tokens:
                if tok not in tf_map:
                    continue
                idf = self._idf.get(tok, 0.0)
                tf = tf_map[tok]
                numer = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * numer / denom
            result.append(score)
        return result


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: list[list[int]],
    rrf_k: float = 60.0,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists via RRF.

    Args:
        ranked_lists: Each inner list is a list of item indices sorted
                      best-first (index 0 = rank 1).
        rrf_k: RRF smoothing constant (default 60, standard literature value).

    Returns:
        List of (item_index, rrf_score) sorted descending by score.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# HybridSearchResult
# ---------------------------------------------------------------------------

@dataclass
class HybridSearchResult:
    """A single result from hybrid (BM25 + embedding) retrieval.

    Attributes
    ----------
    memory : MemoryItem
        The retrieved memory.
    score : float
        Fused RRF score (sum of reciprocal ranks across both channels).
    bm25_rank : int
        Rank in the BM25-only channel (1-based; 0 = not in BM25 results).
    embedding_rank : int
        Rank in the embedding-only channel (1-based; 0 = not found).
    bm25_score : float
        Raw BM25 score.
    embedding_score : float
        Raw cosine similarity in [0, 1].
    """
    memory: MemoryItem
    score: float
    bm25_rank: int = 0
    embedding_rank: int = 0
    bm25_score: float = 0.0
    embedding_score: float = 0.0

    def to_retrieval_result(self) -> RetrievalResult:
        """Convert to standard RetrievalResult for interoperability."""
        return RetrievalResult(
            memory=self.memory,
            score=self.score,
            source_tier=self.memory.tier,
            strategy="hybrid_rrf",
            strategy_scores={
                "bm25_rank": float(self.bm25_rank),
                "embedding_rank": float(self.embedding_rank),
                "bm25_score": self.bm25_score,
                "embedding_score": self.embedding_score,
            },
            explanation=(
                f"hybrid_rrf: BM25 rank={self.bm25_rank}, "
                f"emb rank={self.embedding_rank}, "
                f"RRF={self.score:.4f}"
            ),
        )


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid BM25 + embedding retriever with Reciprocal Rank Fusion.

    Retrieves from all four memory tiers (working, short_term, long_term,
    semantic) and fuses the two signals via RRF.

    Parameters
    ----------
    memory : HierarchicalMemory
        The hierarchical memory store to search.
    bm25_k1 : float
        BM25 term saturation parameter (default 1.5).
    bm25_b : float
        BM25 length normalisation parameter (default 0.75).
    rrf_k : float
        RRF smoothing constant (default 60).
    embedder : EmbeddingProvider | None
        Optional embedding provider.  When None, falls back to BM25-only
        ranking (embedding channel still participates with all-zero scores).
    """

    def __init__(
        self,
        memory: Any,  # HierarchicalMemory — avoid circular import
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: float = 60.0,
        embedder: Any | None = None,
    ) -> None:
        self.memory = memory
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        tiers: list[MemoryTier] | None = None,
        min_score: float = 0.0,
    ) -> list[HybridSearchResult]:
        """Run hybrid retrieval and return fused results.

        Args:
            query: Natural-language search query.
            max_results: Maximum results to return.
            tiers: Which tiers to search (defaults to all four).
            min_score: Minimum RRF score to include a result.

        Returns:
            List of HybridSearchResult sorted by descending RRF score.
        """
        items = self._collect_items(tiers)
        if not items:
            return []

        bm25_ranked, bm25_raw = self._rank_bm25(query, items)
        emb_ranked, emb_raw = self._rank_embedding(query, items)

        fused = _rrf_fuse([bm25_ranked, emb_ranked], rrf_k=self.rrf_k)

        # Build rank lookup tables (1-based)
        bm25_pos = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
        emb_pos = {idx: rank + 1 for rank, idx in enumerate(emb_ranked)}

        results: list[HybridSearchResult] = []
        for idx, rrf_score in fused:
            if rrf_score < min_score:
                break
            results.append(HybridSearchResult(
                memory=items[idx],
                score=rrf_score,
                bm25_rank=bm25_pos.get(idx, 0),
                embedding_rank=emb_pos.get(idx, 0),
                bm25_score=bm25_raw[idx],
                embedding_score=emb_raw[idx],
            ))
            if len(results) >= max_results:
                break

        return results

    def retrieve_as_retrieval_results(
        self,
        query: str,
        max_results: int = 10,
        tiers: list[MemoryTier] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Same as retrieve() but returns standard RetrievalResult objects."""
        return [
            r.to_retrieval_result()
            for r in self.retrieve(query, max_results, tiers, min_score)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_items(self, tiers: list[MemoryTier] | None) -> list[MemoryItem]:
        """Gather MemoryItems from requested tiers (expired/superseded excluded)."""
        wanted = set(tiers) if tiers else set(MemoryTier)
        items: list[MemoryItem] = []
        for tier_name, store in self.memory._iter_tiers():
            tier_enum = MemoryTier(tier_name) if isinstance(tier_name, str) else tier_name
            if tier_enum not in wanted:
                continue
            for item in store:
                if item.is_expired or item.is_superseded:
                    continue
                items.append(item)
        return items

    def _rank_bm25(
        self, query: str, items: list[MemoryItem]
    ) -> tuple[list[int], list[float]]:
        """Return (sorted_indices, raw_scores) for BM25 ranking."""
        bm25 = _BM25(items, k1=self.bm25_k1, b=self.bm25_b)
        raw = bm25.scores(query)
        sorted_indices = sorted(range(len(items)), key=lambda i: raw[i], reverse=True)
        return sorted_indices, raw

    def _rank_embedding(
        self, query: str, items: list[MemoryItem]
    ) -> tuple[list[int], list[float]]:
        """Return (sorted_indices, raw_scores) for embedding cosine ranking."""
        embedder = self.embedder or getattr(self.memory, "embedder", None)
        if embedder is None:
            # No embedder — all items tied at 0; preserve original order
            raw = [0.0] * len(items)
            return list(range(len(items))), raw

        from emms.core.embeddings import cosine_similarity

        q_emb = embedder.embed(query)
        raw: list[float] = []
        for item in items:
            if item.embedding is not None:
                sim = cosine_similarity(q_emb, item.embedding)
            elif hasattr(item.experience, "embedding") and item.experience.embedding is not None:
                sim = cosine_similarity(q_emb, item.experience.embedding)
            else:
                # compute on-the-fly
                i_emb = embedder.embed(item.experience.content)
                sim = cosine_similarity(q_emb, i_emb)
            # cosine_similarity returns a float in [-1, 1] — clamp to [0, 1]
            raw.append(max(0.0, float(sim)))

        sorted_indices = sorted(range(len(items)), key=lambda i: raw[i], reverse=True)
        return sorted_indices, raw
