"""Graph-theoretic episode boundary detection.

Detects natural "episode" boundaries in a stream of experiences using
a similarity graph and modularity-based community detection.
This is one of the novel contributions from the original EMMS research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from emms.core.models import Experience

logger = logging.getLogger(__name__)

# Optional heavy dependency — degrade gracefully
try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


@dataclass
class Episode:
    """A detected episode (contiguous segment of related experiences)."""

    episode_id: int
    experiences: list[Experience] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    coherence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def size(self) -> int:
        return len(self.experiences)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def _domain_sim(a: Experience, b: Experience) -> float:
    return 1.0 if a.domain == b.domain else 0.3


def _temporal_sim(a: Experience, b: Experience, half_life: float = 300.0) -> float:
    """Exponential proximity in time (default half-life = 5 min)."""
    dt = abs(a.timestamp - b.timestamp)
    return float(np.exp(-0.693 * dt / half_life))


def _experience_similarity(a: Experience, b: Experience) -> float:
    """Combined similarity: content + domain + temporal."""
    content = _word_overlap(a.content, b.content)
    domain = _domain_sim(a, b)
    temporal = _temporal_sim(a, b)
    return content * 0.5 + domain * 0.2 + temporal * 0.3


# ---------------------------------------------------------------------------
# Boundary detector
# ---------------------------------------------------------------------------

class EpisodeBoundaryDetector:
    """Detect episode boundaries in a stream of experiences.

    Uses a graph where nodes = experiences and edges = pairwise similarity.
    Communities in the graph correspond to episodes.

    Falls back to a simple surprise-based heuristic if networkx is not
    installed.
    """

    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self._buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Add an experience to the detection buffer."""
        self._buffer.append(experience)

    def detect(self) -> list[Episode]:
        """Run boundary detection on buffered experiences."""
        if len(self._buffer) < 2:
            if self._buffer:
                ep = Episode(
                    episode_id=0,
                    experiences=list(self._buffer),
                    start_time=self._buffer[0].timestamp,
                    end_time=self._buffer[-1].timestamp,
                    coherence=1.0,
                )
                return [ep]
            return []

        if _HAS_NX:
            return self._detect_graph()
        return self._detect_heuristic()

    def clear(self) -> None:
        self._buffer.clear()

    # -- graph-based (preferred) ------------------------------------------

    def _detect_graph(self) -> list[Episode]:
        G = nx.Graph()
        for i, exp in enumerate(self._buffer):
            G.add_node(i, experience=exp)

        # Add edges above threshold
        for i in range(len(self._buffer)):
            for j in range(i + 1, len(self._buffer)):
                sim = _experience_similarity(self._buffer[i], self._buffer[j])
                if sim >= self.similarity_threshold:
                    G.add_edge(i, j, weight=sim)

        # Community detection via greedy modularity
        if G.number_of_edges() == 0:
            # No edges — every experience is its own episode
            return self._one_per_experience()

        communities = nx.community.greedy_modularity_communities(G)
        episodes: list[Episode] = []

        for eid, community in enumerate(communities):
            nodes = sorted(community)
            exps = [self._buffer[n] for n in nodes]
            timestamps = [e.timestamp for e in exps]

            # Intra-community coherence
            sims = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    sims.append(
                        _experience_similarity(exps[i], exps[j])
                    )
            coherence = float(np.mean(sims)) if sims else 1.0

            episodes.append(
                Episode(
                    episode_id=eid,
                    experiences=exps,
                    start_time=min(timestamps),
                    end_time=max(timestamps),
                    coherence=coherence,
                )
            )

        episodes.sort(key=lambda e: e.start_time)
        return episodes

    # -- heuristic fallback -----------------------------------------------

    def _detect_heuristic(self) -> list[Episode]:
        """Simple surprise-based segmentation without networkx."""
        episodes: list[Episode] = []
        current_exps: list[Experience] = [self._buffer[0]]

        for exp in self._buffer[1:]:
            prev = current_exps[-1]
            sim = _experience_similarity(prev, exp)

            if sim < self.similarity_threshold:
                # Boundary detected
                episodes.append(self._make_episode(len(episodes), current_exps))
                current_exps = [exp]
            else:
                current_exps.append(exp)

        if current_exps:
            episodes.append(self._make_episode(len(episodes), current_exps))

        return episodes

    # -- helpers -----------------------------------------------------------

    def _make_episode(self, eid: int, exps: list[Experience]) -> Episode:
        timestamps = [e.timestamp for e in exps]
        return Episode(
            episode_id=eid,
            experiences=exps,
            start_time=min(timestamps),
            end_time=max(timestamps),
            coherence=1.0,
        )

    def _one_per_experience(self) -> list[Episode]:
        return [
            Episode(
                episode_id=i,
                experiences=[exp],
                start_time=exp.timestamp,
                end_time=exp.timestamp,
                coherence=1.0,
            )
            for i, exp in enumerate(self._buffer)
        ]
