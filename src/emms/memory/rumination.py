"""RuminationDetector — identifying repetitive intrusive thought patterns.

v0.25.0: The Vigilant Mind

An agent that cannot notice when it is circling the same painful territory is
trapped: unable to distinguish genuine re-evaluation from compulsive repetition.
Rumination is the cognitive loop — the return, again and again, to memories with
overlapping themes and negative valence, without resolution. In humans, it is a
major risk factor for depression (Nolen-Hoeksema 2000) and an obstacle to
adaptive coping (Watkins 2008). Detecting it is the first step to breaking it.

RuminationDetector operationalises rumination detection for the memory store: it
computes Jaccard token similarity between all pairs of memories and uses
union-find clustering to group memories that repeatedly revisit the same thematic
territory. Clusters with high token overlap and negative emotional valence receive
high rumination_scores. For each cluster, the detector extracts the top theme
tokens and generates a resolution hint — a prompt to consider alternative
framings.

Biological analogue: default mode network self-referential processing (Buckner
et al. 2008); maladaptive rumination vs. adaptive reflection (Nolen-Hoeksema
2000); repetitive negative thought (Watkins 2008); anterior cingulate cortex in
conflict monitoring and self-correction; mindfulness and cognitive defusion as
debiasing strategies; perseverative cognition hypothesis (Brosschot et al. 2006).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
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
class RuminationCluster:
    """A cluster of memories revisiting the same thematic territory."""

    id: str                      # prefixed "rum_"
    domain: str
    cluster_size: int
    rumination_score: float      # 0..1; high = strong repetitive/negative pattern
    mean_negativity: float       # 0..1; (1 - mean_valence) / 2
    theme_tokens: list[str]      # top 5 recurring tokens
    memory_ids: list[str]        # up to 10
    resolution_hint: str
    created_at: float

    def summary(self) -> str:
        return (
            f"RuminationCluster [score={self.rumination_score:.3f}  "
            f"size={self.cluster_size}  domain={self.domain}]\n"
            f"  {self.id[:12]}: themes={self.theme_tokens[:3]}  "
            f"hint={self.resolution_hint[:60]}"
        )


@dataclass
class RuminationReport:
    """Result of a RuminationDetector.detect() call."""

    total_clusters: int
    clusters: list[RuminationCluster]   # sorted by rumination_score desc
    most_ruminative_domain: str         # domain of top cluster, or "none"
    overall_rumination_score: float     # mean rumination_score, 0 if empty
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"RuminationReport: {self.total_clusters} clusters  "
            f"most_ruminative_domain={self.most_ruminative_domain}  "
            f"overall_score={self.overall_rumination_score:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for c in self.clusters[:5]:
            lines.append(f"  {c.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RuminationDetector
# ---------------------------------------------------------------------------


class RuminationDetector:
    """Detects repetitive intrusive thought clusters in memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_cluster_size:
        Minimum number of memories to form a cluster (default 2).
    similarity_threshold:
        Jaccard threshold for linking two memories (default 0.15).
    max_clusters:
        Maximum number of :class:`RuminationCluster` objects to retain (default 10).
    """

    def __init__(
        self,
        memory: Any,
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.15,
        max_clusters: int = 10,
    ) -> None:
        self.memory = memory
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self._clusters: list[RuminationCluster] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, domain: Optional[str] = None) -> RuminationReport:
        """Detect rumination clusters in accumulated memory.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`RuminationReport` with clusters sorted by rumination_score.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        n_total = len(all_items)

        # Build token sets
        token_sets: list[set[str]] = [
            set(self._tokenise(getattr(it.experience, "content", "") or ""))
            for it in all_items
        ]

        # Union-Find clustering
        raw_groups = self._build_clusters(all_items, token_sets)

        clusters: list[RuminationCluster] = []
        for group in raw_groups:
            if len(group) < self.min_cluster_size:
                continue

            cluster_size = len(group)
            valences = [
                getattr(it.experience, "emotional_valence", 0.0) or 0.0
                for it in group
            ]
            mean_negativity = round(
                sum((1.0 - v) / 2.0 for v in valences) / cluster_size, 4
            )
            rumination_score = round(
                min(1.0, (cluster_size / max(n_total, 1)) * (1.0 + mean_negativity)),
                4,
            )

            # Theme tokens: top 5 across cluster
            token_counter: Counter = Counter()
            for it in group:
                content = getattr(it.experience, "content", "") or ""
                for tok in self._tokenise(content):
                    token_counter[tok] += 1
            theme_tokens = [tok for tok, _ in token_counter.most_common(5)]

            top_theme = theme_tokens[0] if theme_tokens else "this theme"
            resolution_hint = (
                f"Consider reframing '{top_theme}' by exploring alternative "
                f"perspectives or outcomes."
            )

            domain_val = (
                getattr(group[0].experience, "domain", None) or "general"
            )
            memory_ids = [it.id for it in group[:10]]

            clusters.append(RuminationCluster(
                id="rum_" + uuid.uuid4().hex[:8],
                domain=domain_val,
                cluster_size=cluster_size,
                rumination_score=rumination_score,
                mean_negativity=mean_negativity,
                theme_tokens=theme_tokens,
                memory_ids=memory_ids,
                resolution_hint=resolution_hint,
                created_at=time.time(),
            ))

        clusters.sort(key=lambda c: c.rumination_score, reverse=True)
        self._clusters = clusters[: self.max_clusters]

        most_ruminative_domain = (
            self._clusters[0].domain if self._clusters else "none"
        )
        overall_score = round(
            sum(c.rumination_score for c in self._clusters) / len(self._clusters)
            if self._clusters else 0.0,
            4,
        )

        return RuminationReport(
            total_clusters=len(self._clusters),
            clusters=self._clusters,
            most_ruminative_domain=most_ruminative_domain,
            overall_rumination_score=overall_score,
            duration_seconds=time.time() - t0,
        )

    def rumination_themes(self) -> list[str]:
        """Return the union of theme_tokens across all detected clusters.

        Returns:
            Deduplicated list of recurring theme token strings.
        """
        seen: set[str] = set()
        result: list[str] = []
        for c in self._clusters:
            for tok in c.theme_tokens:
                if tok not in seen:
                    seen.add(tok)
                    result.append(tok)
        return result

    def most_ruminative_cluster(self) -> Optional[RuminationCluster]:
        """Return the cluster with the highest rumination_score, or ``None``."""
        return self._clusters[0] if self._clusters else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clusters(
        self, items: list[Any], token_sets: list[set[str]]
    ) -> list[list[Any]]:
        """Union-Find clustering of items by Jaccard token overlap."""
        n = len(items)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if not token_sets[i] or not token_sets[j]:
                    continue
                if self._jaccard(token_sets[i], token_sets[j]) >= self.similarity_threshold:
                    union(i, j)

        groups: dict[int, list[Any]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(items[i])

        return list(groups.values())

    def _jaccard(self, set_a: set[str], set_b: set[str]) -> float:
        """Jaccard similarity of two token sets."""
        union_size = len(set_a | set_b)
        if union_size == 0:
            return 0.0
        return len(set_a & set_b) / union_size

    def _tokenise(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 4
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
