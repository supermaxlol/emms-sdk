"""AffectiveRetriever — retrieve memories by emotional proximity.

Memories carry two emotional dimensions (from Experience):
  * emotional_valence   ∈ [-1.0, +1.0]  — positive ↔ negative sentiment
  * emotional_intensity ∈ [ 0.0,  1.0]  — calm ↔ intense

This retriever finds memories whose emotional signature is close to a target
emotional state, optionally blending in semantic (BM25) matching.

Emotional proximity is computed as:
    proximity = 1 - sqrt((v_diff² + i_diff²) / 2)
where v_diff = |target_valence - stored_valence| and similarly for intensity.
The result is in [0, 1]; higher = emotionally closer.

The final score blends semantic and emotional signals:
    score = semantic_blend * semantic_score + (1 - semantic_blend) * proximity

When no semantic query is given, semantic_blend is set to 0 automatically so
only the emotional proximity drives ranking.

Usage::

    from emms import EMMS, Experience
    from emms.retrieval.affective import AffectiveRetriever

    mem = EMMS()
    mem.store(Experience(content="deep grief", emotional_valence=-0.9, emotional_intensity=0.8))
    mem.store(Experience(content="mild joy", emotional_valence=0.3, emotional_intensity=0.2))

    retriever = AffectiveRetriever(mem.memory)
    results = retriever.retrieve(target_valence=-0.7, target_intensity=0.7)
    for r in results:
        print(r.memory.experience.content, r.emotional_proximity)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AffectiveResult:
    """A single result from an affective memory query."""
    memory: Any              # MemoryItem
    score: float             # combined final score [0, 1]
    valence_distance: float  # |target_v - stored_v|
    intensity_distance: float  # |target_i - stored_i|
    emotional_proximity: float  # 1 - sqrt((v_diff² + i_diff²) / 2)


@dataclass
class EmotionalLandscape:
    """High-level summary of the emotional distribution of all memories."""
    total_memories: int
    mean_valence: float
    mean_intensity: float
    valence_std: float
    intensity_std: float
    # histogram buckets: {"-1.0_-0.5": 3, "-0.5_0.0": 7, ...}
    valence_histogram: dict[str, int]
    intensity_histogram: dict[str, int]
    # top-3 emotional poles
    most_positive: list[str]   # memory IDs with highest valence
    most_negative: list[str]   # memory IDs with lowest valence
    most_intense: list[str]    # memory IDs with highest intensity

    def summary(self) -> str:
        lines = [
            f"Emotional Landscape — {self.total_memories} memories",
            f"  valence:   mean={self.mean_valence:+.3f}  std={self.valence_std:.3f}",
            f"  intensity: mean={self.mean_intensity:.3f}  std={self.intensity_std:.3f}",
            f"  most positive IDs:  {self.most_positive}",
            f"  most negative IDs:  {self.most_negative}",
            f"  most intense IDs:   {self.most_intense}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simple BM25-lite for semantic blending
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(
    items: list[Any],
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return a BM25 score for each item.  Returns zeros if query is empty."""
    q_tokens = _tokenise(query)
    if not q_tokens:
        return [0.0] * len(items)

    corpus = [_tokenise(it.experience.content) for it in items]
    N = len(corpus)
    avgdl = sum(len(d) for d in corpus) / max(N, 1)

    # df
    df: dict[str, int] = defaultdict(int)
    for doc in corpus:
        for t in set(doc):
            df[t] += 1

    scores = []
    for doc in corpus:
        tf: dict[str, int] = defaultdict(int)
        for t in doc:
            tf[t] += 1
        dl = len(doc)
        score = 0.0
        for t in q_tokens:
            if t not in df:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            tf_norm = tf[t] * (k1 + 1.0) / (tf[t] + k1 * (1 - b + b * dl / max(avgdl, 1)))
            score += idf * tf_norm
        scores.append(score)

    # normalise to [0, 1]
    max_s = max(scores) if scores else 0.0
    if max_s > 0:
        scores = [s / max_s for s in scores]
    return scores


# ---------------------------------------------------------------------------
# AffectiveRetriever
# ---------------------------------------------------------------------------

class AffectiveRetriever:
    """Retrieve memories by emotional proximity.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing memory.
    semantic_blend : float
        Weight of the semantic (BM25) score in the final blend (default 0.4).
        Set to 0.0 for pure emotional retrieval.
    """

    def __init__(
        self,
        memory: Any,
        semantic_blend: float = 0.4,
    ) -> None:
        self.memory = memory
        self.semantic_blend = max(0.0, min(1.0, semantic_blend))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str = "",
        target_valence: float | None = None,
        target_intensity: float | None = None,
        max_results: int = 10,
        valence_tolerance: float = 2.0,
        intensity_tolerance: float = 2.0,
    ) -> list[AffectiveResult]:
        """Retrieve memories emotionally close to the target state.

        Parameters
        ----------
        query : str
            Optional semantic query to blend with emotional proximity.
        target_valence : float | None
            Target emotional valence (-1…+1).  If None, valence proximity
            is not considered and only intensity is used.
        target_intensity : float | None
            Target emotional intensity (0…1).  If None, intensity proximity
            is not considered and only valence is used.
        max_results : int
            Maximum number of results to return.
        valence_tolerance : float
            Memories with |valence_diff| > tolerance are excluded.
            Default 2.0 (i.e. no filtering for the valid range [-1,1]).
        intensity_tolerance : float
            Similarly for intensity.  Default 2.0.

        Returns
        -------
        list[AffectiveResult] sorted by score descending.
        """
        items = self._collect_all()
        if not items:
            return []

        # Semantic scores
        blend = self.semantic_blend if query.strip() else 0.0
        sem_scores = _bm25_scores(items, query) if blend > 0 else [0.0] * len(items)

        results: list[AffectiveResult] = []
        for item, sem_score in zip(items, sem_scores):
            exp = item.experience
            v = exp.emotional_valence
            i = exp.emotional_intensity

            v_diff = abs((target_valence or 0.0) - v) if target_valence is not None else 0.0
            i_diff = abs((target_intensity or 0.0) - i) if target_intensity is not None else 0.0

            # Tolerance filtering
            if target_valence is not None and v_diff > valence_tolerance:
                continue
            if target_intensity is not None and i_diff > intensity_tolerance:
                continue

            # Proximity: [0, 1]; 1 = perfect match
            if target_valence is not None and target_intensity is not None:
                proximity = 1.0 - math.sqrt((v_diff ** 2 + i_diff ** 2) / 2.0)
            elif target_valence is not None:
                proximity = 1.0 - v_diff / 2.0  # max diff is 2.0 on [-1,1]
            elif target_intensity is not None:
                proximity = 1.0 - i_diff  # max diff is 1.0 on [0,1]
            else:
                proximity = 0.5  # no target — neutral proximity

            # Final blended score
            score = blend * sem_score + (1.0 - blend) * proximity
            score = max(0.0, min(1.0, score))

            results.append(AffectiveResult(
                memory=item,
                score=score,
                valence_distance=v_diff,
                intensity_distance=i_diff,
                emotional_proximity=proximity,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:max_results]

    def retrieve_similar_feeling(
        self,
        reference_memory_id: str,
        max_results: int = 10,
    ) -> list[AffectiveResult]:
        """Retrieve memories with similar emotional signature to a reference.

        Finds the reference memory by its MemoryItem ID, then retrieves
        other memories with similar valence and intensity.

        Parameters
        ----------
        reference_memory_id : str
            The MemoryItem ID of the reference memory.
        max_results : int
            Maximum results (reference itself is excluded).
        """
        items = self._collect_all()
        ref = next((it for it in items if it.id == reference_memory_id), None)
        if ref is None:
            return []

        target_v = ref.experience.emotional_valence
        target_i = ref.experience.emotional_intensity
        all_results = self.retrieve(
            query="",
            target_valence=target_v,
            target_intensity=target_i,
            max_results=max_results + 1,
        )
        # Exclude reference itself
        return [r for r in all_results if r.memory.id != reference_memory_id][:max_results]

    def emotional_landscape(self) -> EmotionalLandscape:
        """Summarise the emotional distribution across all memories."""
        items = self._collect_all()
        if not items:
            return EmotionalLandscape(
                total_memories=0,
                mean_valence=0.0,
                mean_intensity=0.0,
                valence_std=0.0,
                intensity_std=0.0,
                valence_histogram={},
                intensity_histogram={},
                most_positive=[],
                most_negative=[],
                most_intense=[],
            )

        valences = [it.experience.emotional_valence for it in items]
        intensities = [it.experience.emotional_intensity for it in items]
        n = len(items)

        mean_v = sum(valences) / n
        mean_i = sum(intensities) / n

        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in valences) / n)
        std_i = math.sqrt(sum((i - mean_i) ** 2 for i in intensities) / n)

        # Histograms
        v_hist = self._histogram(valences, bins=[(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.01)])
        i_hist = self._histogram(intensities, bins=[(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)])

        # Top emotional extremes (by ID)
        sorted_by_v = sorted(items, key=lambda it: -it.experience.emotional_valence)
        sorted_by_neg_v = sorted(items, key=lambda it: it.experience.emotional_valence)
        sorted_by_i = sorted(items, key=lambda it: -it.experience.emotional_intensity)

        return EmotionalLandscape(
            total_memories=n,
            mean_valence=mean_v,
            mean_intensity=mean_i,
            valence_std=std_v,
            intensity_std=std_i,
            valence_histogram=v_hist,
            intensity_histogram=i_hist,
            most_positive=[it.id for it in sorted_by_v[:3]],
            most_negative=[it.id for it in sorted_by_neg_v[:3]],
            most_intense=[it.id for it in sorted_by_i[:3]],
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        """Flatten all tiers of hierarchical memory."""
        items: list[Any] = []
        for tier_store in (self.memory.working, self.memory.short_term):
            items.extend(tier_store)
        for tier_store in (self.memory.long_term, self.memory.semantic):
            items.extend(tier_store.values())
        return items

    def _histogram(
        self,
        values: list[float],
        bins: list[tuple[float, float]],
    ) -> dict[str, int]:
        hist: dict[str, int] = {}
        for lo, hi in bins:
            key = f"{lo:.2f}_{hi:.2f}"
            hist[key] = sum(1 for v in values if lo <= v < hi)
        return hist
