"""AssociationGraph — explicit memory-to-memory association graph.

v0.12.0: The Associative Mind

Memories do not exist in isolation. Each is embedded in a web of connections
to related memories: semantic neighbours, temporal co-occurrences, emotional
resonances, domain siblings. The AssociationGraph makes these connections
explicit so they can be queried, traversed, and used to spread activation.

Biological analogue: Hebbian associative networks — neurons that fire together
wire together. Recall of one memory primes related ones.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssociationEdge:
    """A directed edge in the association graph."""

    source_id: str
    target_id: str
    edge_type: str   # "semantic" | "temporal" | "affective" | "domain" | "explicit"
    weight: float    # 0.0 – 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class ActivationResult:
    """A memory reached via spreading activation."""

    memory_id: str
    activation: float        # accumulated activation score
    steps_from_seed: int     # hop count from nearest seed
    path: list[str] = field(default_factory=list)


@dataclass
class AssociationStats:
    """Graph-level statistics."""

    total_nodes: int
    total_edges: int
    mean_degree: float
    mean_edge_weight: float
    most_connected_id: Optional[str]
    edge_type_counts: dict[str, int]

    def summary(self) -> str:
        lines = [
            f"Nodes: {self.total_nodes}   Edges: {self.total_edges}",
            f"Mean degree: {self.mean_degree:.2f}   Mean weight: {self.mean_edge_weight:.3f}",
        ]
        if self.most_connected_id:
            lines.append(f"Most connected: {self.most_connected_id}")
        if self.edge_type_counts:
            counts_str = "  ".join(
                f"{k}={v}" for k, v in sorted(self.edge_type_counts.items())
            )
            lines.append(f"Edge types: {counts_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AssociationGraph
# ---------------------------------------------------------------------------

class AssociationGraph:
    """Explicit association graph over MemoryItems.

    Edges are stored bidirectionally (undirected semantics, directed storage).
    Edge types:

      - "semantic"  — cosine similarity of stored embeddings
      - "temporal"  — items stored within ``temporal_window`` seconds
      - "affective" — emotional valence within ``affective_tolerance``
      - "domain"    — same experience domain string
      - "explicit"  — manually added via :meth:`associate`

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance to draw items from.
    semantic_threshold:
        Minimum cosine similarity to form a semantic edge (default 0.5).
    temporal_window:
        Maximum seconds between ``stored_at`` timestamps for a temporal edge
        (default 300 = 5 minutes).
    affective_tolerance:
        Maximum |valence_a − valence_b| for an affective edge (default 0.3).
    """

    def __init__(
        self,
        memory: Any,
        semantic_threshold: float = 0.5,
        temporal_window: float = 300.0,
        affective_tolerance: float = 0.3,
    ) -> None:
        self.memory = memory
        self.semantic_threshold = semantic_threshold
        self.temporal_window = temporal_window
        self.affective_tolerance = affective_tolerance
        # Adjacency list: memory_id → list[AssociationEdge leaving that node]
        self._adj: dict[str, list[AssociationEdge]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def associate(
        self,
        id_a: str,
        id_b: str,
        edge_type: str = "explicit",
        weight: float = 0.8,
    ) -> AssociationEdge:
        """Manually add a bidirectional association between two memories.

        Args:
            id_a:       Source memory ID.
            id_b:       Target memory ID.
            edge_type:  Type label (default "explicit").
            weight:     Edge weight 0–1 (default 0.8).

        Returns:
            The A→B :class:`AssociationEdge`.
        """
        edge_ab = AssociationEdge(id_a, id_b, edge_type, max(0.0, min(1.0, weight)))
        edge_ba = AssociationEdge(id_b, id_a, edge_type, max(0.0, min(1.0, weight)))
        self._adj.setdefault(id_a, []).append(edge_ab)
        self._adj.setdefault(id_b, []).append(edge_ba)
        return edge_ab

    def ensure_bidirectional(self) -> int:
        """Ensure every A→B edge has a corresponding B→A reverse edge.

        ``associate()`` already creates bidirectional edges, but edges added
        via other paths (entity co-occurrence in graph.py, direct _adj writes)
        may be one-directional. Call this periodically to keep the graph
        consistent for spreading activation.

        Returns:
            Number of reverse edges added.
        """
        added = 0
        for source_id, edges in list(self._adj.items()):
            for edge in edges:
                target_edges = self._adj.get(edge.target_id, [])
                has_reverse = any(e.target_id == source_id for e in target_edges)
                if not has_reverse:
                    reverse = AssociationEdge(
                        edge.target_id, source_id, edge.edge_type, edge.weight
                    )
                    self._adj.setdefault(edge.target_id, []).append(reverse)
                    added += 1
        return added

    def auto_associate(self, items: list[Any] | None = None) -> int:
        """Automatically build edges from all stored memory items.

        For each pair (a, b) the strongest signal wins. Existing edges are
        cleared and rebuilt. Call this after adding new memories.

        Args:
            items: Override the memory items to process (for testing).

        Returns:
            Number of undirected edges added.
        """
        if items is None:
            items = self._collect_all()
        # Reset graph
        self._adj.clear()

        if len(items) < 2:
            return 0

        added = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                best_weight = 0.0
                best_type = "semantic"

                # 1. Semantic similarity — HIGHEST PRIORITY.
                # When real embeddings exist and similarity meets threshold,
                # semantic wins unconditionally over affective/temporal/domain.
                # Previously, default emotional_valence=0.0 gave affective
                # weight=1.0 on every pair, permanently blocking semantic edges.
                emb_a = self.memory._embeddings.get(a.experience.id)
                emb_b = self.memory._embeddings.get(b.experience.id)
                semantic_set = False
                if emb_a is not None and emb_b is not None:
                    sim = self._cosine_sim(emb_a, emb_b)
                    if sim >= self.semantic_threshold:
                        best_weight, best_type = sim, "semantic"
                        semantic_set = True

                # 2-4: Only evaluate fallback signals when no semantic edge was set
                if not semantic_set:
                    # 2. Temporal proximity
                    gap = abs(a.stored_at - b.stored_at)
                    if gap <= self.temporal_window:
                        w = 1.0 - gap / max(self.temporal_window, 1e-9)
                        if w > best_weight:
                            best_weight, best_type = w, "temporal"

                    # 3. Affective similarity (emotional valence)
                    va = getattr(a.experience, "emotional_valence", 0.0) or 0.0
                    vb = getattr(b.experience, "emotional_valence", 0.0) or 0.0
                    if abs(va - vb) <= self.affective_tolerance:
                        w = 1.0 - abs(va - vb) / max(self.affective_tolerance, 1e-9)
                        if w > best_weight:
                            best_weight, best_type = w, "affective"

                    # 4. Domain match
                    dom_a = getattr(a.experience, "domain", None)
                    dom_b = getattr(b.experience, "domain", None)
                    if dom_a and dom_b and dom_a == dom_b:
                        if 0.6 > best_weight:
                            best_weight, best_type = 0.6, "domain"

                if best_weight > 0.0:
                    self.associate(a.id, b.id, edge_type=best_type, weight=best_weight)
                    added += 1

        return added

    def spreading_activation(
        self,
        seed_ids: list[str],
        decay: float = 0.5,
        steps: int = 3,
        initial: float = 1.0,
        min_activation: float = 0.01,
    ) -> list[ActivationResult]:
        """BFS spreading activation from a set of seed nodes.

        Activation starts at ``initial`` on each seed and decays by
        ``edge.weight * decay`` per hop. Nodes accumulate the maximum
        activation received across all paths.

        Args:
            seed_ids:        Memory IDs to initialise activation.
            decay:           Decay factor per hop (default 0.5).
            steps:           Maximum hop depth (default 3).
            initial:         Starting activation on seeds (default 1.0).
            min_activation:  Prune paths below this threshold (default 0.01).

        Returns:
            :class:`ActivationResult` list sorted by activation descending,
            excluding seed nodes.
        """
        activation: dict[str, float] = {}
        paths: dict[str, list[str]] = {}
        step_map: dict[str, int] = {}

        for sid in seed_ids:
            activation[sid] = initial
            paths[sid] = [sid]
            step_map[sid] = 0

        frontier = list(seed_ids)
        for step in range(1, steps + 1):
            next_frontier: list[str] = []
            seen_this_step: set[str] = set()
            for node in frontier:
                act = activation.get(node, 0.0)
                for edge in self._adj.get(node, []):
                    spread = act * edge.weight * decay
                    if spread < min_activation:
                        continue
                    tid = edge.target_id
                    if spread > activation.get(tid, 0.0):
                        activation[tid] = spread
                        paths[tid] = paths.get(node, [node]) + [tid]
                        step_map[tid] = step
                        if tid not in seen_this_step:
                            seen_this_step.add(tid)
                            next_frontier.append(tid)
            frontier = next_frontier
            if not frontier:
                break

        seed_set = set(seed_ids)
        results = [
            ActivationResult(
                memory_id=mid,
                activation=act,
                steps_from_seed=step_map.get(mid, 0),
                path=paths.get(mid, [mid]),
            )
            for mid, act in activation.items()
            if mid not in seed_set
        ]
        results.sort(key=lambda r: r.activation, reverse=True)
        return results

    def neighbors(
        self,
        memory_id: str,
        min_weight: float = 0.0,
    ) -> list[AssociationEdge]:
        """Return all edges leaving *memory_id*, optionally filtered by weight.

        Returns:
            Edges sorted by weight descending.
        """
        edges = self._adj.get(memory_id, [])
        if min_weight > 0.0:
            edges = [e for e in edges if e.weight >= min_weight]
        return sorted(edges, key=lambda e: e.weight, reverse=True)

    def strongest_path(self, id_a: str, id_b: str) -> list[str]:
        """Find the path from *id_a* to *id_b* that maximises product of edge weights.

        Uses Dijkstra with ``−log(weight)`` as the distance metric so that
        maximising the weight product is equivalent to minimising total cost.

        Returns:
            List of memory IDs on the path (inclusive of endpoints), or ``[]``
            if no path exists.
        """
        import heapq

        dist: dict[str, float] = {id_a: 0.0}
        prev: dict[str, str | None] = {id_a: None}
        heap: list[tuple[float, str]] = [(0.0, id_a)]

        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, float("inf")):
                continue
            if node == id_b:
                break
            for edge in self._adj.get(node, []):
                cost = d - math.log(max(edge.weight, 1e-9))
                tid = edge.target_id
                if cost < dist.get(tid, float("inf")):
                    dist[tid] = cost
                    prev[tid] = node
                    heapq.heappush(heap, (cost, tid))

        if id_b not in dist:
            return []

        path: list[str] = []
        cur: str | None = id_b
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def stats(self) -> AssociationStats:
        """Return summary statistics for the current graph state."""
        nodes = set(self._adj.keys())
        # Undirected: count each edge once (source < target)
        unique_edges = [
            e
            for edges in self._adj.values()
            for e in edges
            if e.source_id < e.target_id
        ]
        edge_type_counts: dict[str, int] = {}
        for e in unique_edges:
            edge_type_counts[e.edge_type] = edge_type_counts.get(e.edge_type, 0) + 1

        degrees = {n: len(edges) for n, edges in self._adj.items()}
        mean_degree = sum(degrees.values()) / max(len(degrees), 1)
        mean_weight = (
            sum(e.weight for e in unique_edges) / max(len(unique_edges), 1)
        )
        most_connected = (
            max(degrees, key=lambda n: degrees[n]) if degrees else None
        )

        return AssociationStats(
            total_nodes=len(nodes),
            total_edges=len(unique_edges),
            mean_degree=mean_degree,
            mean_edge_weight=mean_weight,
            most_connected_id=most_connected,
            edge_type_counts=edge_type_counts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return dot / (na * nb)
