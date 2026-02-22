"""ConceptHierarchy — taxonomic knowledge organisation from memory.

v0.19.0: The Integrated Mind

Human semantic memory is not a flat list — it is organised into hierarchical
taxonomies where abstract concepts subsume more specific ones. "Animal" is a
parent of "dog", which is a parent of "golden retriever". This hierarchical
structure enables rapid generalisation (all dogs have four legs), efficient
retrieval via categorical cues, and structured inference. Without it, every
concept is equally close to every other — the mind becomes a bag of words
rather than an organised knowledge system.

ConceptHierarchy extracts this taxonomic structure from the token statistics
of accumulated memories:

1. **Candidates**: tokens appearing in ≥ min_frequency distinct memories
   become candidate concepts.

2. **Level 0 (Roots)**: the top 1/3 most frequent candidates become abstract
   root concepts — they appear broadly across many memories.

3. **Level 1 (Children)**: each remaining candidate is assigned to the most-
   frequent root it co-occurs with in the same memory.

4. **Level 2+ (Grandchildren)**: candidates not yet placed are assigned to
   the most-frequent level-1 node they co-occur with, up to max_depth.

5. **Concept distance**: BFS traversal through the parent/children graph.

This gives a meaningful 2–3 level taxonomy purely from co-occurrence
statistics — no embeddings required.

Biological analogue: hierarchical semantic network (Collins & Quillian 1969)
— the "teachable language comprehender" demonstrated that humans verify
"A canary can fly" faster than "A canary has skin" because fewer links
separate CANARY→FLY than CANARY→ANIMAL→HAS-SKIN; prototype theory (Rosch
1975) — categories organised around typical members with graded membership;
taxonomic organisation in semantic memory (Rogers & McClelland 2004) —
distributed representations exhibit hierarchical property inheritance.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConceptNode:
    """A node in the concept taxonomy."""

    id: str
    label: str                      # the concept word
    domain: str                     # most common domain among supporting memories
    level: int                      # 0 = root (most abstract)
    parent_id: Optional[str]        # None for roots
    children_ids: list[str]
    supporting_memory_ids: list[str]
    abstraction_score: float        # freq / total_items → high = abstract

    def summary(self) -> str:
        indent = "  " * self.level
        return (
            f"{indent}ConceptNode[L{self.level}] '{self.label}'  "
            f"domain={self.domain!r}  "
            f"abstraction={self.abstraction_score:.3f}  "
            f"children={len(self.children_ids)}"
        )


@dataclass
class HierarchyReport:
    """Result of a ConceptHierarchy.build() call."""

    total_concepts: int
    total_edges: int
    max_depth: int
    domains: list[str]
    nodes: list[ConceptNode]        # sorted by level asc, then freq desc
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"HierarchyReport: {self.total_concepts} concepts  "
            f"{self.total_edges} edges  "
            f"max_depth={self.max_depth}  "
            f"domains={self.domains}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for node in self.nodes[:8]:
            lines.append(f"  {node.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConceptHierarchy
# ---------------------------------------------------------------------------


class ConceptHierarchy:
    """Builds a taxonomic concept hierarchy from memory token statistics.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_frequency:
        Minimum number of distinct memories a token must appear in to be
        included as a concept node (default 2).
    max_depth:
        Maximum hierarchy depth (0 = roots only, default 3).
    max_concepts:
        Maximum concept nodes across the whole tree (default 40).
    """

    def __init__(
        self,
        memory: Any,
        min_frequency: int = 2,
        max_depth: int = 3,
        max_concepts: int = 40,
    ) -> None:
        self.memory = memory
        self.min_frequency = min_frequency
        self.max_depth = max_depth
        self.max_concepts = max_concepts
        self._nodes: dict[str, ConceptNode] = {}   # label → node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, domain: Optional[str] = None) -> HierarchyReport:
        """Build the concept hierarchy from accumulated memories.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`HierarchyReport` with all concept nodes.
        """
        t0 = time.time()
        self._nodes.clear()

        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        if not items:
            return HierarchyReport(
                total_concepts=0, total_edges=0, max_depth=0,
                domains=[], nodes=[], duration_seconds=time.time() - t0,
            )

        freq_counter = self._extract_candidates(items)
        self._build_tree(freq_counter, items)

        # Build sorted node list
        nodes = sorted(
            self._nodes.values(),
            key=lambda n: (n.level, -n.abstraction_score),
        )

        total_edges = sum(len(n.children_ids) for n in nodes)
        max_depth_found = max((n.level for n in nodes), default=0)
        domains = sorted({n.domain for n in nodes})

        return HierarchyReport(
            total_concepts=len(nodes),
            total_edges=total_edges,
            max_depth=max_depth_found,
            domains=domains,
            nodes=nodes,
            duration_seconds=time.time() - t0,
        )

    def ancestors(self, label: str) -> list[ConceptNode]:
        """Return all ancestors of a concept, root first.

        Args:
            label: The concept label to look up.

        Returns:
            List of :class:`ConceptNode` from root to parent of ``label``.
        """
        node = self._nodes.get(label)
        if node is None:
            return []
        chain: list[ConceptNode] = []
        current = node
        while current.parent_id is not None:
            parent = next(
                (n for n in self._nodes.values() if n.id == current.parent_id),
                None,
            )
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return list(reversed(chain))

    def descendants(self, label: str) -> list[ConceptNode]:
        """Return all descendants of a concept in BFS order.

        Args:
            label: The concept label to look up.

        Returns:
            List of :class:`ConceptNode` in breadth-first order.
        """
        start = self._nodes.get(label)
        if start is None:
            return []
        result: list[ConceptNode] = []
        queue: deque[ConceptNode] = deque()
        queue.append(start)
        visited: set[str] = {start.id}
        while queue:
            node = queue.popleft()
            for child_id in node.children_ids:
                child = next(
                    (n for n in self._nodes.values() if n.id == child_id),
                    None,
                )
                if child and child.id not in visited:
                    visited.add(child.id)
                    result.append(child)
                    queue.append(child)
        return result

    def concept_distance(self, label_a: str, label_b: str) -> int:
        """Compute the shortest path distance between two concepts.

        Args:
            label_a: First concept label.
            label_b: Second concept label.

        Returns:
            Number of hops, or ``-1`` if no path exists.
        """
        if label_a == label_b:
            return 0
        node_a = self._nodes.get(label_a)
        node_b = self._nodes.get(label_b)
        if node_a is None or node_b is None:
            return -1

        # Build adjacency: parent ↔ children
        def neighbors(node: ConceptNode) -> list[str]:
            ids: list[str] = list(node.children_ids)
            if node.parent_id:
                ids.append(node.parent_id)
            return ids

        id_map = {n.id: n for n in self._nodes.values()}
        queue: deque[tuple[str, int]] = deque([(node_a.id, 0)])
        visited: set[str] = {node_a.id}
        while queue:
            current_id, dist = queue.popleft()
            current = id_map.get(current_id)
            if current is None:
                continue
            for nid in neighbors(current):
                if nid == node_b.id:
                    return dist + 1
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, dist + 1))
        return -1

    def most_abstract(self, n: int = 5) -> list[ConceptNode]:
        """Return the n most abstract (level 0) concepts by abstraction score.

        Args:
            n: Number of concepts to return.

        Returns:
            List of :class:`ConceptNode` at level 0 sorted by abstraction descending.
        """
        roots = [nd for nd in self._nodes.values() if nd.level == 0]
        return sorted(roots, key=lambda nd: nd.abstraction_score, reverse=True)[:n]

    def most_specific(self, n: int = 5) -> list[ConceptNode]:
        """Return the n most specific (deepest level, leaf) concepts.

        Args:
            n: Number of concepts to return.

        Returns:
            List of deepest :class:`ConceptNode` sorted by abstraction ascending.
        """
        if not self._nodes:
            return []
        max_level = max(nd.level for nd in self._nodes.values())
        leaves = [nd for nd in self._nodes.values()
                  if nd.level == max_level and not nd.children_ids]
        if not leaves:
            leaves = [nd for nd in self._nodes.values() if not nd.children_ids]
        return sorted(leaves, key=lambda nd: nd.abstraction_score)[:n]

    def get_node(self, label: str) -> Optional[ConceptNode]:
        """Return the concept node for a label, or ``None``.

        Args:
            label: The concept word to look up.
        """
        return self._nodes.get(label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_candidates(self, items: list[Any]) -> Counter:
        """Count token frequency across all memory contents."""
        freq: Counter = Counter()
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            seen = set()
            for w in text.lower().split():
                tok = w.strip(".,!?;:\"'()")
                if len(tok) >= 4 and tok not in _STOP_WORDS and tok not in seen:
                    freq[tok] += 1
                    seen.add(tok)
        return freq

    def _build_tree(self, freq_counter: Counter, items: list[Any]) -> None:
        """Populate self._nodes with a taxonomic hierarchy."""
        n_items = max(len(items), 1)

        # Filter to min_frequency
        candidates = {
            tok: count
            for tok, count in freq_counter.items()
            if count >= self.min_frequency
        }
        if not candidates:
            return

        # Sort by frequency descending
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        sorted_cands = sorted_cands[: self.max_concepts]

        n_roots = max(1, len(sorted_cands) // 3)
        root_labels = [tok for tok, _ in sorted_cands[:n_roots]]

        # Pre-compute co-occurrence sets: tok → set of memory IDs it appears in
        mem_sets: dict[str, set[str]] = {}
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            for w in text.lower().split():
                tok = w.strip(".,!?;:\"'()")
                if tok in candidates:
                    mem_sets.setdefault(tok, set()).add(item.id)

        # Dominant domain per token
        def dominant_domain(tok: str) -> str:
            tok_ids = mem_sets.get(tok, set())
            domain_ctr: Counter = Counter()
            for item in items:
                if item.id in tok_ids:
                    d = getattr(item.experience, "domain", None) or "general"
                    domain_ctr[d] += 1
            if domain_ctr:
                return domain_ctr.most_common(1)[0][0]
            return "general"

        def supporting_ids(tok: str) -> list[str]:
            return list(mem_sets.get(tok, set()))[:8]

        def make_node(label: str, level: int, parent_id: Optional[str]) -> ConceptNode:
            freq = candidates[label]
            return ConceptNode(
                id=f"concept_{uuid.uuid4().hex[:8]}",
                label=label,
                domain=dominant_domain(label),
                level=level,
                parent_id=parent_id,
                children_ids=[],
                supporting_memory_ids=supporting_ids(label),
                abstraction_score=round(freq / n_items, 4),
            )

        # Create roots
        for label in root_labels:
            self._nodes[label] = make_node(label, 0, None)

        if self.max_depth == 0:
            return

        # Level 1: assign remaining candidates to most-frequent co-occurring root
        remaining = [tok for tok, _ in sorted_cands[n_roots:]]
        level_1_labels: list[str] = []

        for tok in remaining:
            if len(self._nodes) >= self.max_concepts:
                break
            tok_mems = mem_sets.get(tok, set())
            best_root: Optional[str] = None
            best_overlap = 0
            for root_label in root_labels:
                root_mems = mem_sets.get(root_label, set())
                overlap = len(tok_mems & root_mems)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_root = root_label
            if best_root is not None and best_overlap > 0:
                parent_node = self._nodes[best_root]
                child = make_node(tok, 1, parent_node.id)
                self._nodes[tok] = child
                parent_node.children_ids.append(child.id)
                level_1_labels.append(tok)

        if self.max_depth < 2:
            return

        # Level 2+: assign still-unplaced candidates to level-1 nodes
        placed = set(root_labels) | set(level_1_labels)
        level_2_remaining = [tok for tok, _ in sorted_cands if tok not in placed]
        current_level_labels = level_1_labels

        for depth in range(2, self.max_depth + 1):
            if not current_level_labels or not level_2_remaining:
                break
            next_level_labels: list[str] = []
            for tok in list(level_2_remaining):
                if len(self._nodes) >= self.max_concepts:
                    break
                tok_mems = mem_sets.get(tok, set())
                best_parent: Optional[str] = None
                best_overlap = 0
                for parent_label in current_level_labels:
                    parent_mems = mem_sets.get(parent_label, set())
                    overlap = len(tok_mems & parent_mems)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent_label
                if best_parent is not None and best_overlap > 0:
                    parent_node = self._nodes[best_parent]
                    child = make_node(tok, depth, parent_node.id)
                    self._nodes[tok] = child
                    parent_node.children_ids.append(child.id)
                    next_level_labels.append(tok)
                    level_2_remaining.remove(tok)
            current_level_labels = next_level_labels

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
