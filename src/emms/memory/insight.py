"""InsightEngine — cross-domain insight generation from association bridges.

v0.12.0: The Associative Mind

When two memories from different domains are strongly connected in the
association graph, they form a "bridge" — a structural hint that a shared
underlying pattern exists. The InsightEngine traverses these bridges and
synthesises new insight memories that make the connection explicit.

Biological analogue: Default Mode Network activity during rest (and REM sleep)
— the brain spontaneously binds distant concepts, producing the "aha!" moments
of creative insight and analogical reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from emms.memory.association import AssociationGraph


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InsightBridge:
    """A cross-domain association bridge that generated an insight."""

    memory_a_id: str
    memory_b_id: str
    domain_a: str
    domain_b: str
    bridge_weight: float
    insight_content: str
    new_memory_id: Optional[str] = None


@dataclass
class InsightReport:
    """Result of a single InsightEngine.discover() run."""

    session_id: Optional[str]
    started_at: float
    duration_seconds: float
    bridges_found: int
    insights_generated: int
    new_memory_ids: list[str]
    bridges: list[InsightBridge]

    def summary(self) -> str:
        lines = [
            f"InsightEngine run in {self.duration_seconds:.2f}s",
            f"Bridges found: {self.bridges_found}   "
            f"Insights generated: {self.insights_generated}",
        ]
        for b in self.bridges[:5]:
            preview = b.insight_content[:85]
            lines.append(
                f"  [{b.domain_a}↔{b.domain_b}] "
                f"w={b.bridge_weight:.2f}  {preview}..."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InsightEngine
# ---------------------------------------------------------------------------

class InsightEngine:
    """Generates new insight memories from cross-domain association bridges.

    The engine:

    1. Builds (or uses a pre-built) :class:`AssociationGraph`.
    2. Walks all edges looking for cross-domain pairs with weight ≥
       ``min_bridge_weight``.
    3. For the top-*k* bridges (by weight) it synthesises a concise insight
       memory and stores it in the hierarchical memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance to store insight memories in.
    association_graph:
        Optional pre-built :class:`AssociationGraph`. If ``None``, one is
        built on each ``discover()`` call.
    min_bridge_weight:
        Minimum edge weight to qualify as a bridge (default 0.45).
    max_insights:
        Maximum number of insight memories to generate per run (default 8).
    insight_importance:
        Importance assigned to synthesised insight memories (default 0.72).
    cross_domain_only:
        If ``True`` (default), only bridges between *different* domains are
        used. Set ``False`` to also generate within-domain insights.
    """

    def __init__(
        self,
        memory: Any,
        association_graph: AssociationGraph | None = None,
        min_bridge_weight: float = 0.45,
        max_insights: int = 8,
        insight_importance: float = 0.72,
        cross_domain_only: bool = True,
    ) -> None:
        self.memory = memory
        self.graph = association_graph
        self.min_bridge_weight = min_bridge_weight
        self.max_insights = max_insights
        self.insight_importance = insight_importance
        self.cross_domain_only = cross_domain_only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        session_id: Optional[str] = None,
        rebuild_graph: bool = True,
    ) -> InsightReport:
        """Find cross-domain bridges and store insight memories.

        Args:
            session_id:    Label attached to the report.
            rebuild_graph: Call ``auto_associate()`` before searching.
                           Set to ``False`` if the graph was pre-built.

        Returns:
            :class:`InsightReport` detailing bridges found and memories created.
        """
        t0 = time.time()

        graph = self.graph
        if graph is None:
            graph = AssociationGraph(self.memory)
        if rebuild_graph:
            graph.auto_associate()

        items = self._collect_all()
        item_map: dict[str, Any] = {it.id: it for it in items}

        bridges: list[InsightBridge] = []
        seen_pairs: set[frozenset] = set()

        for mid, edges in graph._adj.items():
            if mid not in item_map:
                continue
            item_a = item_map[mid]
            dom_a = getattr(item_a.experience, "domain", None) or "general"

            for edge in edges:
                tid = edge.target_id
                if tid not in item_map:
                    continue
                pair: frozenset = frozenset({mid, tid})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if edge.weight < self.min_bridge_weight:
                    continue

                item_b = item_map[tid]
                dom_b = getattr(item_b.experience, "domain", None) or "general"

                if self.cross_domain_only and dom_a == dom_b:
                    continue

                content = self._generate_insight(item_a, item_b, edge.weight)
                bridges.append(
                    InsightBridge(
                        memory_a_id=mid,
                        memory_b_id=tid,
                        domain_a=dom_a,
                        domain_b=dom_b,
                        bridge_weight=edge.weight,
                        insight_content=content,
                    )
                )

        # Take the top-k by bridge weight
        bridges.sort(key=lambda b: b.bridge_weight, reverse=True)
        bridges = bridges[: self.max_insights]

        # Store insight memories
        from emms.core.models import Experience

        new_ids: list[str] = []
        for bridge in bridges:
            try:
                exp = Experience(
                    content=bridge.insight_content,
                    domain="insight",
                    importance=self.insight_importance,
                )
                item = self.memory.store(exp)
                if item is not None and hasattr(item, "id"):
                    bridge.new_memory_id = item.id
                    new_ids.append(item.id)
            except Exception:
                pass

        return InsightReport(
            session_id=session_id,
            started_at=t0,
            duration_seconds=time.time() - t0,
            bridges_found=len(bridges),
            insights_generated=len(new_ids),
            new_memory_ids=new_ids,
            bridges=bridges,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_insight(self, item_a: Any, item_b: Any, weight: float) -> str:
        """Template-based insight synthesis from two memory items."""
        dom_a = getattr(item_a.experience, "domain", "general")
        dom_b = getattr(item_b.experience, "domain", "general")
        exc_a = item_a.experience.content[:75].rstrip(" .,")
        exc_b = item_b.experience.content[:75].rstrip(" .,")
        connector = "strongly connects with" if weight > 0.7 else "unexpectedly resonates with"
        return (
            f"Cross-domain insight [{dom_a}↔{dom_b}]: "
            f'"{exc_a}..." {connector} "{exc_b}..." — '
            f"suggesting a shared underlying pattern across domains."
        )

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
