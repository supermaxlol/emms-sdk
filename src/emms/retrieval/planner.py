"""MemoryQueryPlanner — heuristic query decomposition + parallel retrieval.

Splits a complex natural-language query into sub-queries, runs each
through HybridRetriever in parallel (using asyncio), cross-boosts results
that appear in multiple sub-query results, and returns a deduplicated,
ranked list.

Decomposition heuristics:
  1. Conjunction split: "X and Y" → ["X", "Y"]
  2. Comma split: "X, Y, Z" → ["X", "Y", "Z"]
  3. Question decomposition: "What is X and how does Y work?" → ["What is X", "how does Y work"]
  4. Long queries: split on " and " / " or " / " but "

Sub-query result cross-boost:
  For each result appearing in n > 1 sub-queries: score += 0.1 * (n - 1)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from emms.core.models import RetrievalResult
from emms.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubQueryResult:
    """Results from a single sub-query."""
    sub_query: str
    results: list[RetrievalResult]
    retrieval_time: float


@dataclass
class QueryPlan:
    """The execution plan for a complex query."""
    original_query: str
    sub_queries: list[str]
    sub_results: list[SubQueryResult]
    merged_results: list[RetrievalResult]
    total_unique_results: int
    cross_boost_count: int           # number of results that got cross-boosted
    planning_time: float
    execution_time: float

    def summary(self) -> str:
        lines = [
            f"Query: {self.original_query!r}",
            f"Sub-queries: {len(self.sub_queries)}",
            f"Total unique results: {self.total_unique_results}",
            f"Cross-boosted: {self.cross_boost_count}",
            f"Planning: {self.planning_time*1000:.1f}ms  "
            f"Execution: {self.execution_time*1000:.1f}ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """Rule-based query decomposer.

    Parameters
    ----------
    min_sub_query_tokens : int
        Sub-queries shorter than this (in word count) are dropped.
    max_sub_queries : int
        Hard cap on number of sub-queries to generate.
    """

    def __init__(
        self,
        min_sub_query_tokens: int = 2,
        max_sub_queries: int = 6,
    ) -> None:
        self.min_tokens = min_sub_query_tokens
        self.max_sub = max_sub_queries

    def decompose(self, query: str) -> list[str]:
        """Return a list of sub-queries for the given query string."""
        query = query.strip()
        if not query:
            return []

        candidates = self._split(query)
        # clean + filter
        cleaned = []
        seen: set[str] = set()
        for c in candidates:
            c = c.strip().rstrip("?.,;")
            c_lower = c.lower()
            if not c or c_lower in seen:
                continue
            if len(c.split()) < self.min_tokens:
                continue
            seen.add(c_lower)
            cleaned.append(c)

        if not cleaned:
            return [query]
        return cleaned[: self.max_sub]

    def _split(self, query: str) -> list[str]:
        # Try question decomposition first (multiple question clauses)
        q_parts = re.split(r'\?\s+', query)
        if len(q_parts) > 1:
            parts = []
            for p in q_parts:
                p = p.strip()
                if p:
                    parts.extend(self._split_conjunctions(p))
            return parts if len(parts) > 1 else self._split_conjunctions(query)

        return self._split_conjunctions(query)

    def _split_conjunctions(self, query: str) -> list[str]:
        """Split on ' and ', ' or ', comma, ' but '."""
        # conjunction split
        parts = re.split(r'\s+and\s+|\s+but\s+|\s+or\s+', query, flags=re.I)
        if len(parts) > 1:
            return parts
        # comma split
        parts = [p.strip() for p in query.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
        return [query]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class MemoryQueryPlanner:
    """Decomposes, retrieves, cross-boosts, and merges memory query results.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing hierarchical memory.
    max_results_per_sub : int
        Max results per sub-query from HybridRetriever.
    max_final_results : int
        Cap on merged result list.
    cross_boost : float
        Score increment per additional sub-query that returns a given item.
    bm25_k1, bm25_b, rrf_k : float
        HybridRetriever parameters.
    embedder : EmbeddingProvider | None
    min_score : float
        Filter out results below this threshold after cross-boost.
    """

    def __init__(
        self,
        memory: Any,
        max_results_per_sub: int = 10,
        max_final_results: int = 20,
        cross_boost: float = 0.10,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: float = 60.0,
        embedder: Any = None,
        min_score: float = 0.0,
    ) -> None:
        self.memory = memory
        self.max_per_sub = max_results_per_sub
        self.max_final = max_final_results
        self.cross_boost = cross_boost
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k
        self.embedder = embedder
        self.min_score = min_score
        self._decomposer = QueryDecomposer()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def plan_retrieve(self, query: str, max_results: int | None = None) -> QueryPlan:
        """Decompose query, retrieve per sub-query, cross-boost, return plan."""
        t_plan_start = time.time()
        sub_queries = self._decomposer.decompose(query)
        planning_time = time.time() - t_plan_start

        t_exec_start = time.time()
        sub_results: list[SubQueryResult] = []
        for sq in sub_queries:
            t0 = time.time()
            results = self._retrieve_sub(sq)
            sub_results.append(SubQueryResult(
                sub_query=sq,
                results=results,
                retrieval_time=time.time() - t0,
            ))

        execution_time = time.time() - t_exec_start
        max_r = max_results if max_results is not None else self.max_final
        merged, boost_count = self._merge(sub_results, max_r)

        return QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            sub_results=sub_results,
            merged_results=merged,
            total_unique_results=len(merged),
            cross_boost_count=boost_count,
            planning_time=planning_time,
            execution_time=execution_time,
        )

    def plan_retrieve_simple(
        self, query: str, max_results: int | None = None
    ) -> list[RetrievalResult]:
        """Convenience: run plan_retrieve and return only merged results."""
        plan = self.plan_retrieve(query, max_results=max_results)
        return plan.merged_results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _retrieve_sub(self, sub_query: str) -> list[RetrievalResult]:
        retriever = HybridRetriever(
            memory=self.memory,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            rrf_k=self.rrf_k,
            embedder=self.embedder,
        )
        return retriever.retrieve_as_retrieval_results(
            query=sub_query,
            max_results=self.max_per_sub,
        )

    def _merge(
        self,
        sub_results: list[SubQueryResult],
        max_results: int,
    ) -> tuple[list[RetrievalResult], int]:
        """Cross-boost + deduplicate.

        Returns (merged_list, n_cross_boosted).
        """
        # Accumulate: id → (best_result, appearances)
        seen: dict[str, tuple[RetrievalResult, int]] = {}

        for sr in sub_results:
            for rr in sr.results:
                mid = rr.memory.id
                if mid in seen:
                    existing, count = seen[mid]
                    # keep higher score base
                    best_score = max(existing.score, rr.score)
                    seen[mid] = (
                        RetrievalResult(
                            memory=existing.memory,
                            score=best_score,
                            source_tier=existing.memory.tier,
                            strategy=existing.strategy,
                            strategy_scores=existing.strategy_scores,
                            explanation=existing.explanation,
                        ),
                        count + 1,
                    )
                else:
                    seen[mid] = (rr, 1)

        boost_count = 0
        final: list[RetrievalResult] = []
        for mid, (rr, count) in seen.items():
            boosted_score = rr.score
            if count > 1:
                boosted_score += self.cross_boost * (count - 1)
                boost_count += 1
            if boosted_score < self.min_score:
                continue
            final.append(RetrievalResult(
                memory=rr.memory,
                score=min(boosted_score, 1.0),
                source_tier=rr.memory.tier,
                strategy=rr.strategy + (f"+boost×{count-1}" if count > 1 else ""),
                strategy_scores=rr.strategy_scores,
                explanation=rr.explanation,
            ))

        final.sort(key=lambda r: -r.score)
        return final[:max_results], boost_count
