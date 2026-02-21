"""AssociativeRetriever — retrieval via spreading activation on the AssociationGraph.

v0.12.0: The Associative Mind

Classic retrieval (BM25, embedding cosine) finds memories that *match* a query.
Associative retrieval finds memories that are *connected* to memories that match
— capturing the indirect, contextual associations that surface relevant but not
obviously similar memories.

Biological analogue: priming — recalling one memory raises the accessibility of
associated memories even when the cue does not directly reference them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from emms.memory.association import AssociationGraph, ActivationResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssociativeResult:
    """A memory reached via spreading activation."""

    memory: Any              # the MemoryItem
    activation_score: float  # accumulated activation
    steps_from_seed: int     # hop count from nearest seed
    path: list[str] = field(default_factory=list)  # memory IDs on the path


# ---------------------------------------------------------------------------
# AssociativeRetriever
# ---------------------------------------------------------------------------

class AssociativeRetriever:
    """Retrieves memories via spreading activation on an :class:`AssociationGraph`.

    Usage pattern:

    1. Point at a :class:`HierarchicalMemory` and optionally a pre-built graph.
    2. Call :meth:`retrieve` with seed memory IDs — activation spreads and top
       activating non-seed memories are returned.
    3. Or call :meth:`retrieve_by_query` to auto-select seeds from a text query
       before spreading.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    association_graph:
        Optional pre-built :class:`AssociationGraph`. A new one is built and
        cached on first use if not provided.
    semantic_blend:
        Reserved for future hybrid scoring (currently unused).
    """

    def __init__(
        self,
        memory: Any,
        association_graph: AssociationGraph | None = None,
        semantic_blend: float = 0.3,
    ) -> None:
        self.memory = memory
        self.graph = association_graph
        self.semantic_blend = semantic_blend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        seed_ids: list[str],
        max_results: int = 10,
        steps: int = 3,
        decay: float = 0.5,
    ) -> list[AssociativeResult]:
        """Retrieve memories via spreading activation from seed memory IDs.

        Args:
            seed_ids:    Memory IDs to initialise activation from.
            max_results: Maximum results to return (default 10).
            steps:       Activation hop depth (default 3).
            decay:       Activation decay factor per hop (default 0.5).

        Returns:
            :class:`AssociativeResult` list sorted by activation descending.
        """
        if not seed_ids:
            return []

        graph = self._get_graph()
        activations = graph.spreading_activation(
            seed_ids, decay=decay, steps=steps
        )
        item_map = self._item_map()

        results: list[AssociativeResult] = []
        for ar in activations[:max_results]:
            item = item_map.get(ar.memory_id)
            if item is None:
                continue
            results.append(
                AssociativeResult(
                    memory=item,
                    activation_score=ar.activation,
                    steps_from_seed=ar.steps_from_seed,
                    path=ar.path,
                )
            )
        return results

    def retrieve_by_query(
        self,
        query: str,
        seed_count: int = 3,
        max_results: int = 10,
        steps: int = 3,
        decay: float = 0.5,
        rebuild_graph: bool = False,
    ) -> list[AssociativeResult]:
        """Find seed memories for *query*, then spread activation.

        Args:
            query:         Text query used to select seed memories.
            seed_count:    Number of seed memories to start from (default 3).
            max_results:   Maximum results after activation (default 10).
            steps:         Activation hop depth (default 3).
            decay:         Activation decay per hop (default 0.5).
            rebuild_graph: Rebuild the association graph before retrieval.

        Returns:
            :class:`AssociativeResult` list sorted by activation descending.
        """
        graph = self._get_graph()
        if rebuild_graph:
            graph.auto_associate()

        seed_ids = self._bm25_seeds(query, k=seed_count)
        if not seed_ids:
            return []

        return self.retrieve(seed_ids, max_results=max_results, steps=steps, decay=decay)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_graph(self) -> AssociationGraph:
        """Return the association graph, building it on first call."""
        if self.graph is None:
            self.graph = AssociationGraph(self.memory)
            self.graph.auto_associate()
        return self.graph

    def _item_map(self) -> dict[str, Any]:
        return {item.id: item for item in self._collect_all()}

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items

    def _bm25_seeds(self, query: str, k: int = 3) -> list[str]:
        """Token-overlap seed selection (BM25-lite)."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return []

        items = self._collect_all()
        scored: list[tuple[float, str]] = []
        for item in items:
            text_tokens = set(item.experience.content.lower().split())
            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                score = overlap / (len(text_tokens) + 1)
                scored.append((score, item.id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:k]]
