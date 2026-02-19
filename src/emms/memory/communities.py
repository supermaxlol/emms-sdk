"""GraphCommunityDetection — Label Propagation Algorithm over GraphMemory.

Detects communities (topic clusters) in the entity-relationship graph using
a weighted Label Propagation Algorithm (LPA).  No external dependencies.

Algorithm (Raghavan et al. 2007, weighted variant):
  1. Assign each entity a unique label.
  2. Iterate: for each entity in random order, assign the most frequent
     label among its weighted neighbours.
  3. Repeat until convergence (no label changes) or max_iter reached.

Modularity Q is computed to evaluate partition quality.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Community:
    """A single detected community."""
    community_id: int
    label: int                        # internal LPA label
    entities: list[str]               # entity names
    size: int = 0
    total_internal_strength: float = 0.0
    avg_importance: float = 0.0
    dominant_types: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.size = len(self.entities)


@dataclass
class CommunityResult:
    """Output of community detection."""
    communities: list[Community]
    modularity: float                 # Modularity Q ∈ [-1, 1]
    total_entities: int
    total_edges: int
    num_communities: int
    converged: bool
    iterations_used: int
    bridge_entities: list[str]        # entities connecting ≥2 communities

    def summary(self) -> str:
        lines = [
            f"Communities: {self.num_communities}",
            f"Entities: {self.total_entities}",
            f"Edges: {self.total_edges}",
            f"Modularity Q: {self.modularity:.4f}",
            f"Converged: {self.converged} ({self.iterations_used} iters)",
        ]
        if self.bridge_entities:
            lines.append(f"Bridge entities: {', '.join(self.bridge_entities[:5])}")
        return "\n".join(lines)

    def get_community_for_entity(self, entity_name: str) -> Community | None:
        for c in self.communities:
            if entity_name in c.entities:
                return c
        return None

    def export_markdown(self) -> str:
        lines = [
            "# Graph Communities\n",
            f"**Modularity Q:** {self.modularity:.4f}  |  "
            f"**Communities:** {self.num_communities}  |  "
            f"**Entities:** {self.total_entities}\n",
        ]
        for c in sorted(self.communities, key=lambda x: -x.size):
            lines.append(f"## Community {c.community_id} ({c.size} entities)")
            if c.dominant_types:
                types_str = ", ".join(
                    f"{t}:{n}" for t, n in
                    sorted(c.dominant_types.items(), key=lambda x: -x[1])
                )
                lines.append(f"*Types:* {types_str}")
            lines.append(", ".join(c.entities[:20]))
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class GraphCommunityDetector:
    """Label Propagation community detector for EMMS GraphMemory.

    Parameters
    ----------
    max_iter : int
        Maximum LPA iterations before stopping.
    seed : int | None
        Random seed for reproducible label update order.
    min_community_size : int
        Merge communities smaller than this into an "other" bucket.
    """

    def __init__(
        self,
        max_iter: int = 100,
        seed: int | None = 42,
        min_community_size: int = 1,
    ) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.min_community_size = min_community_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, graph: Any) -> CommunityResult:
        """Run LPA on a GraphMemory instance and return CommunityResult."""
        # Build local adjacency from GraphMemory internals
        # graph._adj  : dict[str, set[str]]
        # graph._rel_index : dict[tuple[str,str], Relationship]
        # graph._entities  : dict[str, Entity]

        entities: dict[str, Any] = getattr(graph, "_entities", {})
        adj: dict[str, set[str]] = getattr(graph, "_adj", {})
        rel_index: dict[tuple[str, str], Any] = getattr(graph, "_rel_index", {})

        if not entities:
            return CommunityResult(
                communities=[],
                modularity=0.0,
                total_entities=0,
                total_edges=0,
                num_communities=0,
                converged=True,
                iterations_used=0,
                bridge_entities=[],
            )

        names = list(entities.keys())
        total_edges = sum(len(nbrs) for nbrs in adj.values()) // 2

        # ------------------------------------------------------------------
        # Helper: edge weight
        # ------------------------------------------------------------------
        def _weight(u: str, v: str) -> float:
            rel = rel_index.get((u, v)) or rel_index.get((v, u))
            return rel.strength if rel is not None else 0.5

        # ------------------------------------------------------------------
        # Compute total graph strength (sum of all edge weights × 2)
        # ------------------------------------------------------------------
        total_strength = 0.0
        for u in names:
            for v in adj.get(u, set()):
                if u < v:
                    total_strength += _weight(u, v)
        total_strength_2 = 2.0 * total_strength if total_strength > 0 else 1.0

        # ------------------------------------------------------------------
        # Node strengths (sum of edge weights)
        # ------------------------------------------------------------------
        node_strength: dict[str, float] = {}
        for u in names:
            node_strength[u] = sum(_weight(u, v) for v in adj.get(u, set()))

        # ------------------------------------------------------------------
        # LPA
        # ------------------------------------------------------------------
        rng = random.Random(self.seed)
        labels: dict[str, int] = {name: i for i, name in enumerate(names)}
        converged = False
        iters = 0

        for iters in range(1, self.max_iter + 1):
            order = names[:]
            rng.shuffle(order)
            changed = 0
            for node in order:
                neighbours = adj.get(node, set())
                if not neighbours:
                    continue
                # count weighted label frequencies among neighbours
                freq: dict[int, float] = {}
                for nb in neighbours:
                    lbl = labels[nb]
                    freq[lbl] = freq.get(lbl, 0.0) + _weight(node, nb)
                best_label = max(freq, key=lambda lbl: (freq[lbl], -lbl))
                if best_label != labels[node]:
                    labels[node] = best_label
                    changed += 1
            if changed == 0:
                converged = True
                break

        # ------------------------------------------------------------------
        # Build Community objects
        # ------------------------------------------------------------------
        label_to_nodes: dict[int, list[str]] = {}
        for name, lbl in labels.items():
            label_to_nodes.setdefault(lbl, []).append(name)

        # Merge tiny communities into label=-1 "other"
        final_communities: list[Community] = []
        cid = 0
        for lbl, members in sorted(label_to_nodes.items(), key=lambda x: -len(x[1])):
            if len(members) < self.min_community_size:
                # merge into first community if any exist
                if final_communities:
                    final_communities[0].entities.extend(members)
                    final_communities[0].size += len(members)
                continue
            # compute internal strength
            internal_strength = 0.0
            for u in members:
                for v in adj.get(u, set()):
                    if v in set(members) and u < v:
                        internal_strength += _weight(u, v)
            # avg importance
            importances = [entities[m].importance for m in members if m in entities]
            avg_imp = sum(importances) / len(importances) if importances else 0.5
            # dominant entity types
            type_counts: dict[str, int] = {}
            for m in members:
                etype = getattr(entities.get(m), "entity_type", "concept")
                type_counts[etype] = type_counts.get(etype, 0) + 1
            final_communities.append(Community(
                community_id=cid,
                label=lbl,
                entities=members,
                total_internal_strength=internal_strength,
                avg_importance=avg_imp,
                dominant_types=type_counts,
            ))
            cid += 1

        # ------------------------------------------------------------------
        # Modularity Q
        # ------------------------------------------------------------------
        Q = 0.0
        for comm in final_communities:
            member_set = set(comm.entities)
            for u in comm.entities:
                for v in comm.entities:
                    if u == v:
                        continue
                    actual = _weight(u, v) if v in adj.get(u, set()) else 0.0
                    expected = node_strength.get(u, 0.0) * node_strength.get(v, 0.0) / total_strength_2
                    Q += actual - expected
        Q /= total_strength_2 if total_strength_2 > 0 else 1.0

        # ------------------------------------------------------------------
        # Bridge entities (connected to ≥ 2 distinct communities)
        # ------------------------------------------------------------------
        entity_to_community: dict[str, int] = {}
        for comm in final_communities:
            for name in comm.entities:
                entity_to_community[name] = comm.community_id

        bridges: list[str] = []
        for name in names:
            connected_communities: set[int] = set()
            for nb in adj.get(name, set()):
                cid_nb = entity_to_community.get(nb, -1)
                if cid_nb != entity_to_community.get(name, -1):
                    connected_communities.add(cid_nb)
            if len(connected_communities) >= 2:
                bridges.append(name)

        return CommunityResult(
            communities=final_communities,
            modularity=Q,
            total_entities=len(names),
            total_edges=total_edges,
            num_communities=len(final_communities),
            converged=converged,
            iterations_used=iters,
            bridge_entities=bridges,
        )
