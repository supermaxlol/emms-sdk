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

# Optional heavy dependencies — degrade gracefully
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    from sklearn.cluster import SpectralClustering
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


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

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        algorithm: str = "auto",
    ):
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm  # "auto", "graph", "spectral", "conductance", "heuristic", "multi"
        self._buffer: list[Experience] = []
        self._last_metrics: dict[str, float] = {}

    def add(self, experience: Experience) -> None:
        """Add an experience to the detection buffer."""
        self._buffer.append(experience)

    def detect(self) -> list[Episode]:
        """Run boundary detection on buffered experiences.

        Algorithm selection:
        - "auto": picks best available (graph > spectral > heuristic)
        - "graph": greedy modularity communities (requires networkx)
        - "spectral": spectral clustering (requires scikit-learn)
        - "conductance": minimize graph conductance (requires networkx)
        - "multi": run all available, pick best by metrics
        - "heuristic": surprise-based, no dependencies
        """
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

        algo = self.algorithm

        if algo == "multi":
            return self._detect_multi_algorithm()
        elif algo == "spectral" and _HAS_SKLEARN:
            return self._detect_spectral()
        elif algo == "conductance" and _HAS_NX:
            return self._detect_conductance()
        elif algo == "graph" and _HAS_NX:
            return self._detect_graph()
        elif algo == "heuristic":
            return self._detect_heuristic()
        else:  # auto
            if _HAS_NX:
                return self._detect_graph()
            return self._detect_heuristic()

    @property
    def metrics(self) -> dict[str, float]:
        """Metrics from the last detection run."""
        return dict(self._last_metrics)

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

    # -- spectral clustering -----------------------------------------------

    def _detect_spectral(self) -> list[Episode]:
        """Episode detection via spectral clustering (requires scikit-learn)."""
        n = len(self._buffer)

        # Build affinity matrix
        affinity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = _experience_similarity(self._buffer[i], self._buffer[j])
                affinity[i, j] = sim
                affinity[j, i] = sim
            affinity[i, i] = 1.0

        # Determine optimal k using eigenvalue gap heuristic
        k = self._estimate_k_spectral(affinity, max_k=min(n // 2, 10))

        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
            )
            labels = sc.fit_predict(affinity)
        except Exception:
            logger.warning("Spectral clustering failed, falling back to heuristic")
            return self._detect_heuristic()

        # Build episodes from cluster labels
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        episodes = []
        for eid, (_, nodes) in enumerate(sorted(clusters.items())):
            nodes.sort()
            exps = [self._buffer[n] for n in nodes]
            timestamps = [e.timestamp for e in exps]
            coherence = self._compute_cluster_coherence(nodes)
            episodes.append(Episode(
                episode_id=eid,
                experiences=exps,
                start_time=min(timestamps),
                end_time=max(timestamps),
                coherence=coherence,
            ))

        episodes.sort(key=lambda e: e.start_time)
        self._last_metrics = self._calculate_boundary_metrics(episodes)
        return episodes

    def _estimate_k_spectral(self, affinity: np.ndarray, max_k: int = 10) -> int:
        """Estimate number of clusters from eigenvalue gaps."""
        try:
            eigenvalues = np.linalg.eigvalsh(affinity)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # Find largest gap
            gaps = np.diff(eigenvalues[:max_k])
            k = int(np.argmin(gaps) + 2)  # +2 because diff reduces by 1, and we want at least 2
            return max(2, min(k, max_k))
        except Exception:
            return min(3, max_k)

    # -- conductance optimisation ------------------------------------------

    def _detect_conductance(self) -> list[Episode]:
        """Episode detection by minimising graph conductance (requires networkx)."""
        G = nx.Graph()
        for i, exp in enumerate(self._buffer):
            G.add_node(i, experience=exp)

        for i in range(len(self._buffer)):
            for j in range(i + 1, len(self._buffer)):
                sim = _experience_similarity(self._buffer[i], self._buffer[j])
                if sim >= self.similarity_threshold:
                    G.add_edge(i, j, weight=sim)

        if G.number_of_edges() == 0:
            return self._one_per_experience()

        # Start with greedy modularity communities
        communities = list(nx.community.greedy_modularity_communities(G))

        # Refine by conductance: try moving boundary nodes
        communities = self._refine_by_conductance(G, communities)

        episodes = []
        for eid, community in enumerate(communities):
            nodes = sorted(community)
            exps = [self._buffer[n] for n in nodes]
            timestamps = [e.timestamp for e in exps]
            coherence = self._compute_cluster_coherence(nodes)
            episodes.append(Episode(
                episode_id=eid, experiences=exps,
                start_time=min(timestamps), end_time=max(timestamps),
                coherence=coherence,
            ))

        episodes.sort(key=lambda e: e.start_time)
        self._last_metrics = self._calculate_boundary_metrics(episodes)
        return episodes

    def _refine_by_conductance(self, G, communities: list) -> list[set]:
        """Refine communities by minimising conductance."""
        communities = [set(c) for c in communities]
        improved = True
        max_iterations = 10

        for _ in range(max_iterations):
            if not improved:
                break
            improved = False

            for node in G.nodes():
                current_comm_idx = None
                for idx, comm in enumerate(communities):
                    if node in comm:
                        current_comm_idx = idx
                        break
                if current_comm_idx is None:
                    continue

                current_conductance = self._conductance(G, communities[current_comm_idx])
                best_idx = current_comm_idx
                best_conductance = current_conductance

                for idx, comm in enumerate(communities):
                    if idx == current_comm_idx:
                        continue
                    # Try moving node
                    new_comm = comm | {node}
                    new_cond = self._conductance(G, new_comm)
                    if new_cond < best_conductance:
                        best_conductance = new_cond
                        best_idx = idx

                if best_idx != current_comm_idx:
                    communities[current_comm_idx].discard(node)
                    communities[best_idx].add(node)
                    improved = True

        # Remove empty communities
        return [c for c in communities if c]

    def _conductance(self, G, community: set) -> float:
        """Compute conductance of a community."""
        if not community or len(community) == len(G.nodes()):
            return 1.0
        cut = sum(
            G[u][v].get("weight", 1.0)
            for u in community for v in G.neighbors(u)
            if v not in community
        )
        vol = sum(
            G[u][v].get("weight", 1.0)
            for u in community for v in G.neighbors(u)
        )
        if vol == 0:
            return 1.0
        return cut / vol

    # -- multi-algorithm ---------------------------------------------------

    def _detect_multi_algorithm(self) -> list[Episode]:
        """Run all available algorithms and pick the best by metrics."""
        candidates: list[tuple[str, list[Episode]]] = []

        # Always available
        candidates.append(("heuristic", self._detect_heuristic()))

        if _HAS_NX:
            candidates.append(("graph", self._detect_graph()))
            candidates.append(("conductance", self._detect_conductance()))

        if _HAS_SKLEARN:
            candidates.append(("spectral", self._detect_spectral()))

        # Pick the one with highest average coherence
        best_name = "heuristic"
        best_episodes = candidates[0][1]
        best_score = 0.0

        for name, episodes in candidates:
            if not episodes:
                continue
            avg_coherence = float(np.mean([e.coherence for e in episodes]))
            # Prefer more episodes (better segmentation) but not too many
            size_penalty = abs(len(episodes) - len(self._buffer) / 5) * 0.01
            score = avg_coherence - size_penalty
            if score > best_score:
                best_score = score
                best_episodes = episodes
                best_name = name

        logger.info("Multi-algorithm: selected %s (score=%.3f)", best_name, best_score)
        self._last_metrics = self._calculate_boundary_metrics(best_episodes)
        self._last_metrics["selected_algorithm"] = hash(best_name) % 100 / 100  # encode as float
        return best_episodes

    # -- metrics -----------------------------------------------------------

    def _compute_cluster_coherence(self, nodes: list[int]) -> float:
        """Compute average pairwise similarity within a cluster."""
        if len(nodes) < 2:
            return 1.0
        sims = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sims.append(_experience_similarity(
                    self._buffer[nodes[i]], self._buffer[nodes[j]]
                ))
        return float(np.mean(sims)) if sims else 1.0

    def _calculate_boundary_metrics(self, episodes: list[Episode]) -> dict[str, float]:
        """Calculate quality metrics for a set of detected episodes."""
        if not episodes:
            return {"avg_coherence": 0.0, "episode_count": 0.0}

        coherences = [e.coherence for e in episodes]
        sizes = [e.size for e in episodes]

        return {
            "avg_coherence": float(np.mean(coherences)),
            "min_coherence": float(np.min(coherences)),
            "max_coherence": float(np.max(coherences)),
            "episode_count": float(len(episodes)),
            "avg_episode_size": float(np.mean(sizes)),
            "size_std": float(np.std(sizes)) if len(sizes) > 1 else 0.0,
        }

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
