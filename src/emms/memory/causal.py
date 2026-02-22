"""CausalMapper — directed causal graph extraction from memory.

v0.20.0: The Reasoning Mind

Humans do not merely observe associations — they infer *causes*. When we see
stress followed by illness, we do not just correlate them; we construct a
causal model: "stress weakens the immune system, which causes illness." This
causal understanding is the engine of prediction, intervention, and control.
Without it, an agent can learn what tends to co-occur but cannot reason about
what would happen if the world were changed.

CausalMapper operationalises causal extraction for the memory store: it scans
memory content for relational keywords (causes, enables, produces, prevents,
reduces, increases, requires, triggers, inhibits, leads, results, improves,
damages, strengthens, weakens), extracts source→target concept pairs around
each keyword, and builds a directed causal graph. Edges are weighted by their
co-occurrence frequency, and the graph supports forward inference (effects_of),
backward inference (causes_of), and shortest-path causal chaining (causal_path).

Biological analogue: causal model theory (Pearl 2000) — human cognition is
fundamentally causal inference, not mere association; hippocampal-PFC circuits
implement causal reasoning by integrating episodic memories into generative
models (Kumaran et al. 2016); the brain builds causal generative models rather
than simple statistical associations (Tenenbaum, Kemp, Griffiths & Goodman
2011); anterior PFC represents abstract causal relations across domains.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_CAUSAL_KEYWORDS: frozenset[str] = frozenset({
    "causes", "enables", "produces", "prevents", "reduces", "increases",
    "requires", "triggers", "inhibits", "leads", "results", "improves",
    "damages", "strengthens", "weakens",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CausalEdge:
    """A directed causal relation between two concept tokens."""

    source: str             # cause concept
    target: str             # effect concept
    relation: str           # the causal keyword connecting them
    strength: float         # 0..1 — frequency-based weight
    memory_ids: list[str]   # supporting memories

    def summary(self) -> str:
        return (
            f"CausalEdge: '{self.source}' —[{self.relation}]→ '{self.target}'  "
            f"strength={self.strength:.3f}  memories={len(self.memory_ids)}"
        )


@dataclass
class CausalPath:
    """A directed path through the causal graph."""

    nodes: list[str]            # sequence of concept tokens
    edges: list[CausalEdge]
    total_strength: float       # mean edge strength along path

    def summary(self) -> str:
        arrow = " → ".join(self.nodes)
        return (
            f"CausalPath [{len(self.edges)} hops  "
            f"strength={self.total_strength:.3f}]: {arrow}"
        )


@dataclass
class CausalReport:
    """Result of a CausalMapper.build() call."""

    total_concepts: int
    total_edges: int
    most_influential: list[str]   # top concepts by out-degree
    most_affected: list[str]      # top concepts by in-degree
    edges: list[CausalEdge]       # sorted by strength desc
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"CausalReport: {self.total_concepts} concepts  "
            f"{self.total_edges} edges  "
            f"in {self.duration_seconds:.2f}s",
            f"  Most influential: {self.most_influential[:5]}",
            f"  Most affected: {self.most_affected[:5]}",
        ]
        for e in self.edges[:5]:
            lines.append(f"  {e.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CausalMapper
# ---------------------------------------------------------------------------


class CausalMapper:
    """Builds a directed causal graph from memory content.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_strength:
        Minimum edge strength to include in the graph (default 0.1).
    max_concepts:
        Maximum number of unique concept nodes (default 30).
    """

    def __init__(
        self,
        memory: Any,
        min_strength: float = 0.01,
        max_concepts: int = 30,
    ) -> None:
        self.memory = memory
        self.min_strength = min_strength
        self.max_concepts = max_concepts
        # source → target → edge
        self._graph: dict[str, dict[str, CausalEdge]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, domain: Optional[str] = None) -> CausalReport:
        """Extract a causal graph from accumulated memories.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`CausalReport` with concept nodes, edges, and rankings.
        """
        t0 = time.time()
        self._graph.clear()

        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        if not items:
            return CausalReport(
                total_concepts=0, total_edges=0,
                most_influential=[], most_affected=[],
                edges=[], duration_seconds=time.time() - t0,
            )

        n_items = len(items)
        relations = self._extract_relations(items)

        # Count (source, relation, target) tuples
        edge_counts: Counter = Counter(
            (src, rel, tgt) for src, rel, tgt, _ in relations
        )
        # Map (src, tgt) → list of memory_ids
        edge_mems: dict[tuple[str, str], list[str]] = {}
        edge_rels: dict[tuple[str, str], str] = {}
        for src, rel, tgt, mid in relations:
            edge_mems.setdefault((src, tgt), [])
            if mid not in edge_mems[(src, tgt)]:
                edge_mems[(src, tgt)].append(mid)
            edge_rels[(src, tgt)] = rel

        # Build graph
        for (src, tgt), mids in edge_mems.items():
            # Pick the most common relation for this src→tgt pair
            rel = edge_rels[(src, tgt)]
            strength = round(len(mids) / n_items, 4)
            if strength < self.min_strength:
                continue
            if src not in self._graph:
                self._graph[src] = {}
            self._graph[src][tgt] = CausalEdge(
                source=src,
                target=tgt,
                relation=rel,
                strength=strength,
                memory_ids=mids[:5],
            )

        # Collect all edges
        all_edges = [
            edge
            for out_edges in self._graph.values()
            for edge in out_edges.values()
        ]
        all_edges.sort(key=lambda e: e.strength, reverse=True)

        # Rankings
        out_degree: Counter = Counter()
        in_degree: Counter = Counter()
        for src, out_edges in self._graph.items():
            out_degree[src] += len(out_edges)
            for tgt in out_edges:
                in_degree[tgt] += 1

        concepts = set(self._graph.keys()) | {
            tgt for out in self._graph.values() for tgt in out
        }

        return CausalReport(
            total_concepts=len(concepts),
            total_edges=len(all_edges),
            most_influential=[c for c, _ in out_degree.most_common(5)],
            most_affected=[c for c, _ in in_degree.most_common(5)],
            edges=all_edges,
            duration_seconds=time.time() - t0,
        )

    def causes_of(self, concept: str) -> list[CausalEdge]:
        """Return all causal edges whose target is ``concept``.

        Args:
            concept: The effect concept to query.

        Returns:
            List of :class:`CausalEdge` sorted by strength descending.
        """
        result = []
        for out_edges in self._graph.values():
            if concept in out_edges:
                result.append(out_edges[concept])
        return sorted(result, key=lambda e: e.strength, reverse=True)

    def effects_of(self, concept: str) -> list[CausalEdge]:
        """Return all causal edges whose source is ``concept``.

        Args:
            concept: The cause concept to query.

        Returns:
            List of :class:`CausalEdge` sorted by strength descending.
        """
        out_edges = self._graph.get(concept, {})
        return sorted(out_edges.values(), key=lambda e: e.strength, reverse=True)

    def causal_path(
        self, source: str, target: str
    ) -> Optional[CausalPath]:
        """Find the shortest causal path from source to target.

        Args:
            source: Starting concept.
            target: Goal concept.

        Returns:
            :class:`CausalPath` or ``None`` if no path exists.
        """
        if source == target:
            return CausalPath(nodes=[source], edges=[], total_strength=1.0)
        if source not in self._graph:
            return None

        # BFS
        queue: deque[list[str]] = deque([[source]])
        visited: set[str] = {source}
        while queue:
            path_nodes = queue.popleft()
            current = path_nodes[-1]
            for tgt, edge in self._graph.get(current, {}).items():
                new_path = path_nodes + [tgt]
                if tgt == target:
                    # Reconstruct edges
                    path_edges: list[CausalEdge] = []
                    for i in range(len(new_path) - 1):
                        src_node = new_path[i]
                        tgt_node = new_path[i + 1]
                        path_edges.append(self._graph[src_node][tgt_node])
                    mean_str = (
                        sum(e.strength for e in path_edges) / len(path_edges)
                        if path_edges else 0.0
                    )
                    return CausalPath(
                        nodes=new_path,
                        edges=path_edges,
                        total_strength=round(mean_str, 4),
                    )
                if tgt not in visited:
                    visited.add(tgt)
                    queue.append(new_path)
        return None

    def most_influential(self, n: int = 5) -> list[str]:
        """Return the n concepts with highest out-degree (most causal impact).

        Args:
            n: Number of concepts to return.
        """
        out_degree: Counter = Counter(
            {src: len(out) for src, out in self._graph.items()}
        )
        return [c for c, _ in out_degree.most_common(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_relations(
        self, items: list[Any]
    ) -> list[tuple[str, str, str, str]]:
        """Extract (source, relation, target, memory_id) tuples from items."""
        relations: list[tuple[str, str, str, str]] = []
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            words = text.lower().split()
            for i, word in enumerate(words):
                kw = word.strip(".,!?;:\"'()")
                if kw not in _CAUSAL_KEYWORDS:
                    continue
                # Source: scan backward for first non-stop meaningful token
                src = None
                for j in range(i - 1, max(i - 5, -1), -1):
                    tok = words[j].strip(".,!?;:\"'()")
                    if len(tok) >= 3 and tok not in _STOP_WORDS and tok not in _CAUSAL_KEYWORDS:
                        src = tok
                        break
                # Target: scan forward for first non-stop meaningful token
                tgt = None
                for j in range(i + 1, min(i + 5, len(words))):
                    tok = words[j].strip(".,!?;:\"'()")
                    if len(tok) >= 3 and tok not in _STOP_WORDS and tok not in _CAUSAL_KEYWORDS:
                        tgt = tok
                        break
                if src and tgt and src != tgt:
                    relations.append((src, kw, tgt, item.id))
        return relations

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
