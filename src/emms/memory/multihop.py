"""Multi-hop graph reasoning over the EMMS knowledge graph.

Enables reasoning across entity chains that span multiple relationship hops,
discovering indirect connections that single-hop queries miss.

Key capabilities:

* **Multi-hop BFS** — breadth-first traversal from a seed entity up to
  ``max_hops`` steps, collecting all reachable entities with path details.
* **Path-strength scoring** — each HopPath's strength is the product of edge
  strengths along the path (multiplicative decay), rewarding short, strong chains.
* **Betweenness bridging** — approximate betweenness centrality for identifying
  hub entities that connect many pairs (useful for concept bridging queries).
* **DOT export** — `MultiHopResult.to_dot()` emits a Graphviz DOT string of
  the sub-graph explored during the query.

Usage::

    from emms import EMMS, Experience
    from emms.memory.multihop import MultiHopGraphReasoner

    agent = EMMS()
    agent.store(Experience(content="Alice works with Bob at Acme Corp", domain="hr"))
    agent.store(Experience(content="Bob leads the Python team at Acme Corp", domain="hr"))

    reasoner = MultiHopGraphReasoner(agent.graph)
    result = reasoner.query("Alice", max_hops=3, max_results=10)

    print(result.summary())
    for path in result.paths:
        print(" → ".join(path.entities), f"(strength={path.strength:.3f})")

    print(result.to_dot())
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HopPath:
    """A single multi-hop path from the seed to a target entity.

    Attributes
    ----------
    entities : list[str]
        Ordered entity names along the path (seed first, target last).
        Lower-cased for consistency with graph keys; display names can be
        recovered from ``GraphMemory.entities``.
    relations : list[str]
        Relationship types on each hop (len = len(entities) - 1).
    strength : float
        Product of edge strengths along the path (∈ (0, 1]).
        Shorter, stronger paths score higher.
    hops : int
        Number of hops (= len(entities) - 1).
    """
    entities: list[str]
    relations: list[str]
    strength: float
    hops: int

    @property
    def seed(self) -> str:
        """First entity in the path (the seed)."""
        return self.entities[0] if self.entities else ""

    @property
    def target(self) -> str:
        """Last entity in the path (the destination)."""
        return self.entities[-1] if self.entities else ""


@dataclass
class ReachableEntity:
    """An entity reachable from the seed, with aggregated path info.

    Attributes
    ----------
    name : str
        Lower-cased entity name.
    display_name : str
        Original capitalised name from the entity registry.
    entity_type : str
        Entity type (person, org, concept, etc.).
    importance : float
        Importance from the entity registry.
    min_hops : int
        Shortest hop distance from the seed.
    best_path : HopPath
        The path with the highest strength score.
    all_paths : list[HopPath]
        All paths found to this entity (up to a capped count).
    """
    name: str
    display_name: str
    entity_type: str
    importance: float
    min_hops: int
    best_path: HopPath
    all_paths: list[HopPath] = field(default_factory=list)


@dataclass
class MultiHopResult:
    """Result of a multi-hop graph reasoning query.

    Attributes
    ----------
    seed : str
        The query seed entity (lower-cased).
    reachable : list[ReachableEntity]
        All reachable entities, sorted by best_path.strength descending.
    paths : list[HopPath]
        All discovered paths, sorted by strength descending.
    bridging_entities : list[tuple[str, float]]
        Top hub entities by approximate betweenness (name, score).
    total_entities_explored : int
        Count of graph nodes visited during BFS.
    max_hops_used : int
        The max_hops parameter that produced these results.
    """
    seed: str
    reachable: list[ReachableEntity]
    paths: list[HopPath]
    bridging_entities: list[tuple[str, float]]
    total_entities_explored: int
    max_hops_used: int

    def summary(self) -> str:
        """One-line summary of the result."""
        return (
            f"MultiHop({self.seed!r}, max_hops={self.max_hops_used}): "
            f"{len(self.reachable)} reachable entities, "
            f"{len(self.paths)} paths, "
            f"{len(self.bridging_entities)} bridging hubs"
        )

    def to_dot(
        self,
        title: str | None = None,
        max_nodes: int = 50,
        min_strength: float = 0.0,
    ) -> str:
        """Export the explored sub-graph as a Graphviz DOT string.

        Args:
            title: Graph title (default: "MultiHop: {seed}").
            max_nodes: Maximum nodes to include (highest-strength paths first).
            min_strength: Exclude edges with strength below this threshold.

        Returns:
            Graphviz DOT source string.
        """
        label = title or f"MultiHop: {self.seed}"
        lines = [
            f'digraph "{label}" {{',
            "  rankdir=LR;",
            f'  label="{label}";',
            "  fontsize=12;",
            '  node [shape=ellipse fontsize=10];',
            '  edge [fontsize=9];',
        ]

        # Collect edges from paths (sorted by strength, take top max_nodes)
        edges: dict[tuple[str, str], tuple[float, str]] = {}  # (src, tgt) → (strength, rel)
        for path in sorted(self.paths, key=lambda p: p.strength, reverse=True):
            for i in range(len(path.entities) - 1):
                src = path.entities[i]
                tgt = path.entities[i + 1]
                rel = path.relations[i] if i < len(path.relations) else "related_to"
                edge_str = path.strength
                if (src, tgt) not in edges or edges[(src, tgt)][0] < edge_str:
                    edges[(src, tgt)] = (edge_str, rel)
            if len(edges) >= max_nodes:
                break

        # Collect unique nodes
        node_names: set[str] = set()
        node_names.add(self.seed)
        for (src, tgt), (strength, _) in edges.items():
            if strength >= min_strength:
                node_names.add(src)
                node_names.add(tgt)

        # Emit nodes
        for name in sorted(node_names):
            colour = "#ff9966" if name == self.seed else "#99ccff"
            lines.append(
                f'  "{name}" [style=filled fillcolor="{colour}"];'
            )

        # Emit edges
        for (src, tgt), (strength, rel) in edges.items():
            if strength < min_strength:
                continue
            lines.append(
                f'  "{src}" -> "{tgt}" '
                f'[label="{rel}" penwidth={max(0.5, strength * 3):.1f}];'
            )

        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MultiHopGraphReasoner
# ---------------------------------------------------------------------------

class MultiHopGraphReasoner:
    """Multi-hop BFS reasoning over a GraphMemory knowledge graph.

    Parameters
    ----------
    graph : GraphMemory
        The graph memory instance to reason over.
    max_paths_per_entity : int
        Cap on the number of paths stored per reachable entity (default 5).
    """

    def __init__(
        self,
        graph: Any,  # GraphMemory — avoids circular import
        *,
        max_paths_per_entity: int = 5,
    ) -> None:
        self.graph = graph
        self.max_paths_per_entity = max_paths_per_entity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        seed: str,
        *,
        max_hops: int = 3,
        max_results: int = 20,
        min_strength: float = 0.0,
    ) -> MultiHopResult:
        """Run a multi-hop BFS query from a seed entity.

        Args:
            seed: Seed entity name (case-insensitive).
            max_hops: Maximum hop depth for BFS.
            max_results: Maximum reachable entities to return.
            min_strength: Skip paths with product strength below this.

        Returns:
            MultiHopResult with reachable entities, paths, and bridging hubs.
        """
        seed_key = seed.lower()

        if not self.graph or seed_key not in self.graph._adj:
            return MultiHopResult(
                seed=seed_key,
                reachable=[],
                paths=[],
                bridging_entities=[],
                total_entities_explored=0,
                max_hops_used=max_hops,
            )

        reachable_map, all_paths, explored = self._bfs(
            seed_key, max_hops, min_strength
        )

        # Build ReachableEntity objects, sorted by best strength
        reachable: list[ReachableEntity] = []
        for entity_key, paths in reachable_map.items():
            best = max(paths, key=lambda p: p.strength)
            entity_obj = self.graph.entities.get(entity_key)
            display_name = entity_obj.name if entity_obj else entity_key.title()
            entity_type = entity_obj.entity_type if entity_obj else "concept"
            importance = entity_obj.importance if entity_obj else 0.5
            reachable.append(ReachableEntity(
                name=entity_key,
                display_name=display_name,
                entity_type=entity_type,
                importance=importance,
                min_hops=min(p.hops for p in paths),
                best_path=best,
                all_paths=sorted(paths, key=lambda p: p.strength, reverse=True),
            ))

        reachable.sort(key=lambda r: r.best_path.strength, reverse=True)
        reachable = reachable[:max_results]

        all_paths_sorted = sorted(all_paths, key=lambda p: p.strength, reverse=True)

        bridging = self._compute_bridging(all_paths)

        return MultiHopResult(
            seed=seed_key,
            reachable=reachable,
            paths=all_paths_sorted,
            bridging_entities=bridging,
            total_entities_explored=explored,
            max_hops_used=max_hops,
        )

    def find_connection(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> HopPath | None:
        """Find the strongest path between two entities.

        Args:
            source: Source entity name.
            target: Target entity name.
            max_hops: Maximum hops to search.

        Returns:
            The strongest HopPath, or None if no path found.
        """
        result = self.query(source, max_hops=max_hops)
        target_key = target.lower()
        for re_ in result.reachable:
            if re_.name == target_key:
                return re_.best_path
        return None

    # ------------------------------------------------------------------
    # BFS internals
    # ------------------------------------------------------------------

    def _bfs(
        self,
        seed_key: str,
        max_hops: int,
        min_strength: float,
    ) -> tuple[dict[str, list[HopPath]], list[HopPath], int]:
        """Run BFS from seed_key up to max_hops, collecting all paths.

        Returns
        -------
        (reachable_map, all_paths, explored_count)
        """
        # BFS queue entries: (current_entity, path_entities, path_relations, path_strength)
        queue: deque[tuple[str, list[str], list[str], float]] = deque()
        queue.append((seed_key, [seed_key], [], 1.0))

        # best_strength[entity] — track best strength seen so we prune weak branches
        best_strength: dict[str, float] = {seed_key: 1.0}

        reachable_map: dict[str, list[HopPath]] = {}
        all_paths: list[HopPath] = []
        explored: set[str] = {seed_key}

        while queue:
            current, path_ents, path_rels, strength = queue.popleft()

            hop_count = len(path_ents) - 1

            if hop_count >= max_hops:
                continue

            for neighbor in self.graph._adj.get(current, set()):
                if neighbor in path_ents:
                    # Avoid cycles within a single path
                    continue

                explored.add(neighbor)

                # Get edge strength
                edge_strength = self._edge_strength(current, neighbor)
                new_strength = strength * edge_strength

                if new_strength < min_strength:
                    continue

                new_path_ents = path_ents + [neighbor]
                rel_type = self._edge_rel_type(current, neighbor)
                new_path_rels = path_rels + [rel_type]

                path = HopPath(
                    entities=new_path_ents,
                    relations=new_path_rels,
                    strength=new_strength,
                    hops=len(new_path_ents) - 1,
                )

                # Record this path for the neighbor
                if neighbor not in reachable_map:
                    reachable_map[neighbor] = []
                if len(reachable_map[neighbor]) < self.max_paths_per_entity:
                    reachable_map[neighbor].append(path)

                all_paths.append(path)

                # Prune: only enqueue if we found a stronger path to this node
                if new_strength > best_strength.get(neighbor, 0.0):
                    best_strength[neighbor] = new_strength
                    queue.append((neighbor, new_path_ents, new_path_rels, new_strength))

        return reachable_map, all_paths, len(explored)

    def _edge_strength(self, src: str, tgt: str) -> float:
        """Look up the strength of the (src, tgt) edge."""
        rel = self.graph._rel_index.get((src, tgt)) or self.graph._rel_index.get((tgt, src))
        if rel is not None:
            return rel.strength
        return 0.5  # default

    def _edge_rel_type(self, src: str, tgt: str) -> str:
        """Look up the relation type of the (src, tgt) edge."""
        rel = self.graph._rel_index.get((src, tgt)) or self.graph._rel_index.get((tgt, src))
        if rel is not None:
            return rel.relation_type
        return "related_to"

    # ------------------------------------------------------------------
    # Bridging / betweenness approximation
    # ------------------------------------------------------------------

    def _compute_bridging(
        self,
        paths: list[HopPath],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Approximate betweenness centrality from discovered paths.

        For each internal node (not seed, not target) across all paths,
        tally how many paths pass through it.  Normalise by the path's
        strength so that high-quality connections count more.

        Returns
        -------
        List of (entity_name, bridging_score) sorted descending, top_n.
        """
        scores: dict[str, float] = {}
        for path in paths:
            # Internal nodes: all except first and last
            for entity in path.entities[1:-1]:
                scores[entity] = scores.get(entity, 0.0) + path.strength

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]
