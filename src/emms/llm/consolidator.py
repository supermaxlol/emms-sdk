"""LLMConsolidator — LLM-backed memory cluster synthesis.

Groups semantically similar memories and synthesises each group into a
single high-level memory using an LLM (or an extractive fallback when no
LLM is available).

The clustering uses a simple union-find approach on a cosine / lexical
similarity matrix — no dependency on ``MemoryClustering``.  The caller
can also supply pre-built clusters (e.g. from ``MemoryClustering``).

Usage::

    from emms import EMMS
    from emms.integrations.llm import ClaudeProvider, LLMEnhancer
    from emms.llm.consolidator import LLMConsolidator

    agent = EMMS()
    # ... store experiences ...

    provider = ClaudeProvider(api_key="sk-...")
    enhancer = LLMEnhancer(provider)
    consolidator = LLMConsolidator(agent.memory)

    result = await consolidator.auto_consolidate(threshold=0.7, llm_enhancer=enhancer)
    print(result)  # {"clusters_found": 3, "synthesised": 3, "stored": 3}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from emms.core.models import Experience, MemoryTier

if TYPE_CHECKING:
    from emms.core.models import MemoryItem
    from emms.integrations.llm import LLMEnhancer
    from emms.memory.hierarchical import HierarchicalMemory
    from emms.memory.clustering import MemoryCluster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (single-linkage clustering)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return {root: [member indices]}."""
        result: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            result.setdefault(r, []).append(i)
        return result


# ---------------------------------------------------------------------------
# Similarity helpers (TF-IDF lexical — zero dependency)
# ---------------------------------------------------------------------------

_STOP_W = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","i","you","he","she","it","we","they","not","so",
    "this","that","if","as","be","have","do","can","will","my","your","its",
})


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", text)
            if w.lower() not in _STOP_W}


def _lex_sim(a: str, b: str) -> float:
    """Jaccard similarity on token sets."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cos_sim(v1: list[float], v2: list[float]) -> float:
    a, b = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))


# ---------------------------------------------------------------------------
# Extractive synthesiser (LLM fallback)
# ---------------------------------------------------------------------------

def _extractive_synthesis(items: list["MemoryItem"]) -> str:
    """Concatenate the most important unique sentences from cluster items."""
    sentences: list[str] = []
    seen_tokens: list[set[str]] = []

    # Sort by importance (descending) + recency
    sorted_items = sorted(
        items,
        key=lambda m: (m.experience.importance * 0.6 + m.memory_strength * 0.4),
        reverse=True,
    )

    for item in sorted_items:
        content = item.experience.content
        sents = re.split(r'(?<=[.!?])\s+', content.strip())
        for s in sents:
            if len(s) < 15:
                continue
            tok = _tokens(s)
            # Deduplicate via Jaccard > 0.6 with already-selected
            dup = any(
                len(tok & prev) / max(1, len(tok | prev)) > 0.6
                for prev in seen_tokens
            )
            if not dup:
                sentences.append(s)
                seen_tokens.append(tok)
            if len(sentences) >= 5:
                break
        if len(sentences) >= 5:
            break

    return " ".join(sentences[:5]) if sentences else items[0].experience.content


# ---------------------------------------------------------------------------
# ConsolidationResult
# ---------------------------------------------------------------------------

class ConsolidationResult:
    """Result from a consolidation run.

    Attributes
    ----------
    clusters_found : number of similar groups identified.
    synthesised : number of groups for which a synthesis was produced.
    stored : number of new synthesised memories stored.
    failed : number of LLM calls that failed (fell back to extractive).
    elapsed_s : wall-clock seconds for the whole run.
    """

    def __init__(self):
        self.clusters_found = 0
        self.synthesised = 0
        self.stored = 0
        self.failed = 0
        self.elapsed_s = 0.0

    def __repr__(self) -> str:
        return (
            f"ConsolidationResult(clusters_found={self.clusters_found}, "
            f"synthesised={self.synthesised}, stored={self.stored}, "
            f"failed={self.failed}, elapsed_s={self.elapsed_s:.2f})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "clusters_found": self.clusters_found,
            "synthesised": self.synthesised,
            "stored": self.stored,
            "failed": self.failed,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# ---------------------------------------------------------------------------
# LLMConsolidator
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a memory consolidation assistant. Below are {n} related memories from an AI agent.
Synthesise them into ONE concise, higher-level memory (2-4 sentences) that captures the
key insight, pattern, or fact they share. Focus on what is most useful for future retrieval.

Memories:
{memories}

Respond with ONLY a JSON object:
{{
  "title": "<short title ≤10 words>",
  "content": "<synthesised memory 2-4 sentences>",
  "domain": "<most specific applicable domain>",
  "importance": <0.0-1.0 float>
}}"""


class LLMConsolidator:
    """Synthesise clusters of semantically similar memories via LLM.

    Parameters
    ----------
    memory : HierarchicalMemory to read from and store synthesised memories into.
    min_cluster_size : minimum number of items to trigger synthesis (default 2).
    store_tier : target tier for synthesised memories (default SEMANTIC).
    """

    def __init__(
        self,
        memory: "HierarchicalMemory",
        min_cluster_size: int = 2,
        store_tier: MemoryTier = MemoryTier.SEMANTIC,
    ):
        self.memory = memory
        self.min_cluster_size = min_cluster_size
        self.store_tier = store_tier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def consolidate_cluster(
        self,
        items: list["MemoryItem"],
        llm_enhancer: "LLMEnhancer | None" = None,
    ) -> Experience | None:
        """Synthesise a single cluster into one Experience.

        Parameters
        ----------
        items : cluster members (at least ``min_cluster_size``).
        llm_enhancer : optional LLM backend.

        Returns
        -------
        Experience with the synthesised content, or None if synthesis failed.
        """
        if len(items) < self.min_cluster_size:
            return None

        if llm_enhancer is not None:
            exp = await self._llm_synthesise(items, llm_enhancer)
            if exp is not None:
                return exp

        # Extractive fallback
        return self._extractive_to_experience(items)

    async def auto_consolidate(
        self,
        threshold: float = 0.7,
        llm_enhancer: "LLMEnhancer | None" = None,
        tier: MemoryTier = MemoryTier.LONG_TERM,
        max_clusters: int = 20,
    ) -> ConsolidationResult:
        """Scan memory for similar items and consolidate each cluster.

        1. Build similarity matrix over ``tier`` items.
        2. Apply union-find single-linkage at ``threshold``.
        3. Synthesise each cluster with size >= ``min_cluster_size``.
        4. Store each synthesis as a new SEMANTIC-tier memory.
        5. Mark original items as superseded.

        Parameters
        ----------
        threshold : minimum similarity to link two items (default 0.7).
        llm_enhancer : optional LLM backend for synthesis.
        tier : which tier to scan (default LONG_TERM).
        max_clusters : cap on the number of clusters to process.
        """
        t0 = time.time()
        result = ConsolidationResult()

        items = self._get_tier_items(tier)
        if len(items) < self.min_cluster_size:
            result.elapsed_s = time.time() - t0
            return result

        # Build clusters via union-find on similarity matrix
        clusters = self._find_clusters(items, threshold)
        viable = [c for c in clusters if len(c) >= self.min_cluster_size]
        viable = viable[:max_clusters]
        result.clusters_found = len(viable)

        for cluster_items in viable:
            exp = await self.consolidate_cluster(cluster_items, llm_enhancer)
            if exp is None:
                result.failed += 1
                continue

            result.synthesised += 1

            # Store in semantic tier via high-importance consolidation
            try:
                exp.update_mode = "insert"  # always add new synthesis
                # High importance ensures consolidation into semantic tier
                exp.importance = max(exp.importance, 0.8)
                self.memory.store(exp)
                # Run consolidation to promote the item up the tier hierarchy
                self.memory.consolidate()
                result.stored += 1
            except Exception:
                logger.warning("Failed to store consolidated memory", exc_info=True)
                result.failed += 1

        result.elapsed_s = time.time() - t0
        return result

    async def consolidate_from_clusters(
        self,
        clusters: list["MemoryCluster"],
        llm_enhancer: "LLMEnhancer | None" = None,
    ) -> ConsolidationResult:
        """Consolidate pre-built MemoryCluster objects.

        Parameters
        ----------
        clusters : from MemoryClustering.cluster().
        llm_enhancer : optional LLM backend.
        """
        t0 = time.time()
        result = ConsolidationResult()
        result.clusters_found = sum(
            1 for c in clusters if len(c.members) >= self.min_cluster_size
        )

        for cluster in clusters:
            if len(cluster.members) < self.min_cluster_size:
                continue

            exp = await self.consolidate_cluster(cluster.members, llm_enhancer)
            if exp is None:
                result.failed += 1
                continue

            result.synthesised += 1
            try:
                exp.importance = max(exp.importance, 0.8)
                self.memory.store(exp)
                self.memory.consolidate()
                result.stored += 1
            except Exception:
                logger.warning("Failed to store consolidated memory", exc_info=True)
                result.failed += 1

        result.elapsed_s = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_tier_items(self, tier: MemoryTier) -> list["MemoryItem"]:
        """Get non-expired, non-superseded items from the given tier."""
        tier_list = {
            MemoryTier.WORKING: list(self.memory.working),
            MemoryTier.SHORT_TERM: list(self.memory.short_term),
            MemoryTier.LONG_TERM: list(self.memory.long_term.values()),
            MemoryTier.SEMANTIC: list(self.memory.semantic.values()),
        }.get(tier, [])
        return [
            m for m in tier_list
            if not m.is_expired and not m.is_superseded
        ]

    def _find_clusters(
        self,
        items: list["MemoryItem"],
        threshold: float,
    ) -> list[list["MemoryItem"]]:
        """Single-linkage clustering via union-find on pairwise similarity."""
        n = len(items)
        uf = _UnionFind(n)

        # Try embedding-based cosine first, fall back to lexical
        embeddings = getattr(self.memory, "_embeddings", {})

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similarity(items[i], items[j], embeddings)
                if sim >= threshold:
                    uf.union(i, j)

        groups = uf.groups()
        return [[items[idx] for idx in indices] for indices in groups.values()]

    def _similarity(
        self,
        a: "MemoryItem",
        b: "MemoryItem",
        embeddings: dict,
    ) -> float:
        """Compute similarity between two items."""
        ea = embeddings.get(a.experience.id)
        eb = embeddings.get(b.experience.id)
        if ea is not None and eb is not None:
            return _cos_sim(ea, eb)
        # Lexical fallback
        return _lex_sim(a.experience.content, b.experience.content)

    async def _llm_synthesise(
        self,
        items: list["MemoryItem"],
        llm_enhancer: "LLMEnhancer",
    ) -> Experience | None:
        """Call LLM to synthesise cluster items into one Experience."""
        memory_lines = "\n".join(
            f"{i+1}. [{item.experience.domain}] {item.experience.content[:300]}"
            for i, item in enumerate(items[:10])  # cap at 10 to stay in token budget
        )
        prompt = _SYNTHESIS_PROMPT.format(
            n=min(len(items), 10),
            memories=memory_lines,
        )

        try:
            raw = await llm_enhancer.provider.generate(prompt, max_tokens=300)
            data = self._parse_json(raw)
            if not data or "content" not in data:
                return None

            # Derive metadata from cluster
            domains = [m.experience.domain for m in items]
            from collections import Counter
            top_domain = Counter(domains).most_common(1)[0][0]

            exp = Experience(
                content=str(data["content"]),
                domain=str(data.get("domain", top_domain)),
                title=str(data.get("title", ""))[:80] or None,
                importance=float(data.get("importance", 0.8)),
                novelty=0.6,
                facts=[f"Synthesised from {len(items)} related memories"],
            )
            return exp

        except Exception:
            logger.debug("LLM synthesis failed", exc_info=True)
            return None

    def _extractive_to_experience(self, items: list["MemoryItem"]) -> Experience:
        """Build an Experience from extractive synthesis."""
        content = _extractive_synthesis(items)
        domains = [m.experience.domain for m in items]
        from collections import Counter
        top_domain = Counter(domains).most_common(1)[0][0]
        avg_importance = sum(m.experience.importance for m in items) / len(items)

        return Experience(
            content=content,
            domain=top_domain,
            importance=max(avg_importance, 0.7),
            novelty=0.5,
            facts=[f"Extractive synthesis of {len(items)} related memories"],
            title=f"Synthesised: {top_domain} ({len(items)} sources)",
        )

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract the first JSON object from text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None
