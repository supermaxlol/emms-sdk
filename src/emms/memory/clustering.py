"""MemoryCluster — pure-Python k-means clustering over memory items.

Groups memories into semantic clusters using:
- embedding vectors when available (L2-normalized cosine approximation)
- TF-IDF bag-of-words vectors as fallback (no ML dependency)

k-means++ initialization is used for stable, quality-aware seeding.
An elbow method (``auto_k=True``) selects the best k automatically.

Usage::

    from emms import EMMS
    from emms.memory.clustering import MemoryClustering

    agent = EMMS()
    # ... store experiences ...

    clustering = MemoryClustering()
    clusters = clustering.cluster(agent.memory.long_term, k=5)

    for c in clusters:
        print(c.label, "—", [m.experience.content[:40] for m in c.members])
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from emms.core.models import MemoryItem


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MemoryCluster:
    """One cluster produced by MemoryClustering.

    Attributes
    ----------
    id : cluster index (0-based).
    members : list of MemoryItem objects in this cluster.
    centroid : numpy array — mean vector of all member vectors.
    label : auto-generated label from the most frequent domain + top tokens.
    inertia : sum of squared distances from centroid (lower = tighter cluster).
    """

    id: int
    members: list["MemoryItem"] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(1))
    label: str = ""
    inertia: float = 0.0


# ---------------------------------------------------------------------------
# TF-IDF vectorizer (zero-dependency)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","can","could",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "this","that","these","those","not","so","if","as","up","out","into","its",
    "their","our","my","your","his","its","also","than","then","when","where",
})


def _tokenize(text: str) -> list[str]:
    return [
        w.lower() for w in re.findall(r"[A-Za-z][a-z]+", text)
        if len(w) > 2 and w.lower() not in _STOP_WORDS
    ]


def _build_tfidf(texts: list[str]) -> np.ndarray:
    """Build an N×V TF-IDF matrix from raw texts (L2-normalised rows)."""
    N = len(texts)
    tokenized = [_tokenize(t) for t in texts]

    # Build vocabulary
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    # Only keep terms that appear in at least 2 documents (or at least 1 if N<4)
    min_df = 2 if N >= 4 else 1
    vocab = sorted(w for w, c in df.items() if c >= min_df)
    if not vocab:
        vocab = sorted(df.keys())[:50]

    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}

    matrix = np.zeros((N, V), dtype=np.float64)
    for n, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        total = max(1, len(tokens))
        for w, cnt in counts.items():
            if w in w2i:
                tf = cnt / total
                idf = math.log((N + 1) / (df[w] + 1)) + 1.0
                matrix[n, w2i[w]] = tf * idf

    # L2-normalise rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    return matrix, vocab


# ---------------------------------------------------------------------------
# k-means++ helpers
# ---------------------------------------------------------------------------

def _kmeans_plus_init(X: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    """k-means++ centroid initialisation.

    Returns a (k, D) array of initial centroids.
    """
    N = len(X)
    first = rng.randrange(N)
    centroids = [X[first].copy()]

    for _ in range(k - 1):
        # For each point, compute min squared distance to existing centroids
        dists = np.min(
            np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]),
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-12)
        cum = np.cumsum(probs)
        r = rng.random()
        idx = int(np.searchsorted(cum, r))
        idx = min(idx, N - 1)
        centroids.append(X[idx].copy())

    return np.vstack(centroids)


def _run_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run k-means with k-means++ init.

    Returns (labels, centroids, inertia).
    """
    rng = random.Random(seed)
    centroids = _kmeans_plus_init(X, k, rng)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        # Assignment
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)

        # Update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        # L2-normalise centroids (cosine-space k-means)
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_centroids /= norms

        if np.linalg.norm(new_centroids - centroids) < tol:
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    # Inertia
    inertia = sum(
        float(np.sum((X[labels == j] - centroids[j]) ** 2))
        for j in range(k)
        if (labels == j).any()
    )
    return labels, centroids, inertia


def _elbow_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    seed: int = 42,
) -> int:
    """Select k using the elbow method (max second derivative of inertia)."""
    k_max = min(k_max, len(X) - 1)
    k_min = max(2, k_min)
    if k_min >= k_max:
        return k_min

    inertias: list[float] = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        _, _, inertia = _run_kmeans(X, k, seed=seed)
        inertias.append(inertia)

    if len(inertias) < 3:
        return ks[0]

    # Second differences (curvature)
    d1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    d2 = [d1[i] - d1[i + 1] for i in range(len(d1) - 1)]
    best_idx = int(np.argmax(d2)) + 1  # +1 because d2 is offset by 1
    return ks[best_idx]


# ---------------------------------------------------------------------------
# Auto-label
# ---------------------------------------------------------------------------

def _cluster_label(members: list["MemoryItem"]) -> str:
    """Generate a short human-readable label for a cluster."""
    domains: Counter[str] = Counter()
    tokens: Counter[str] = Counter()

    for item in members:
        exp = item.experience
        domains[exp.domain] += 1
        tokens.update(_tokenize(exp.content))

    domain = domains.most_common(1)[0][0] if domains else "general"
    top_tokens = [w for w, _ in tokens.most_common(3)]
    if top_tokens:
        return f"[{domain}] {', '.join(top_tokens)}"
    return f"[{domain}]"


# ---------------------------------------------------------------------------
# MemoryClustering
# ---------------------------------------------------------------------------

class MemoryClustering:
    """Group memory items into semantic clusters.

    Prefers embedding vectors; falls back to TF-IDF if no embeddings present.

    Parameters
    ----------
    seed : random seed for reproducibility.
    max_iter : maximum k-means iterations.
    """

    def __init__(self, seed: int = 42, max_iter: int = 100):
        self.seed = seed
        self.max_iter = max_iter

    def cluster(
        self,
        items: list["MemoryItem"],
        k: int | None = None,
        auto_k: bool = False,
        k_min: int = 2,
        k_max: int = 10,
    ) -> list[MemoryCluster]:
        """Cluster memory items.

        Parameters
        ----------
        items : list of MemoryItem to cluster.
        k : number of clusters (required unless ``auto_k=True``).
        auto_k : if True, ignore ``k`` and select the best k via elbow method.
        k_min, k_max : search bounds for ``auto_k``.

        Returns
        -------
        list[MemoryCluster] sorted by cluster ID.
        """
        if not items:
            return []

        # Build feature matrix
        X, valid_items = self._build_matrix(items)

        if len(valid_items) < 2:
            # Can't cluster fewer than 2 items
            single = MemoryCluster(
                id=0,
                members=list(valid_items),
                centroid=X[0] if len(X) else np.zeros(1),
                label=_cluster_label(list(valid_items)),
                inertia=0.0,
            )
            return [single]

        if auto_k:
            k = _elbow_k(X, k_min=k_min, k_max=k_max, seed=self.seed)
        elif k is None:
            raise ValueError("Either k or auto_k=True must be provided.")

        k = max(1, min(k, len(valid_items)))

        labels, centroids, total_inertia = _run_kmeans(
            X, k=k, max_iter=self.max_iter, seed=self.seed
        )

        # Build cluster objects
        cluster_map: dict[int, list[tuple[int, "MemoryItem"]]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[int(label)].append((idx, valid_items[idx]))

        clusters: list[MemoryCluster] = []
        for cid in sorted(cluster_map.keys()):
            cluster_items = [pair[1] for pair in cluster_map[cid]]
            member_vecs = X[labels == cid]
            per_inertia = float(np.sum(
                np.sum((member_vecs - centroids[cid]) ** 2, axis=1)
            ))
            c = MemoryCluster(
                id=cid,
                members=cluster_items,
                centroid=centroids[cid].copy(),
                label=_cluster_label(cluster_items),
                inertia=per_inertia,
            )
            clusters.append(c)

        return clusters

    def _build_matrix(
        self, items: list["MemoryItem"]
    ) -> tuple[np.ndarray, list["MemoryItem"]]:
        """Build an (N, D) L2-normalised feature matrix.

        Uses embedding vectors if present on the first item; falls back to
        TF-IDF on content text.
        """
        # Check if embedding vectors are available
        # (embeddings are stored externally in HierarchicalMemory._embeddings)
        # We use TF-IDF as the zero-dependency default; embeddings can be
        # injected by calling cluster_with_embeddings() below.
        texts = [item.experience.content for item in items]
        X, _ = _build_tfidf(texts)
        return X, list(items)

    def cluster_with_embeddings(
        self,
        items: list["MemoryItem"],
        embeddings: dict[str, list[float]],  # exp_id → embedding vector
        k: int | None = None,
        auto_k: bool = False,
        k_min: int = 2,
        k_max: int = 10,
    ) -> list[MemoryCluster]:
        """Cluster using pre-computed embedding vectors.

        Parameters
        ----------
        items : list of MemoryItem.
        embeddings : dict mapping ``experience.id`` → embedding vector.
            Items without an entry fall back to zero vectors.
        k, auto_k, k_min, k_max : same as ``cluster()``.
        """
        if not items:
            return []

        dim = max(len(v) for v in embeddings.values()) if embeddings else 8
        X_rows: list[np.ndarray] = []
        valid_items: list["MemoryItem"] = []

        for item in items:
            vec = embeddings.get(item.experience.id)
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float64)
                n = np.linalg.norm(arr)
                if n > 0:
                    arr = arr / n
                X_rows.append(arr)
                valid_items.append(item)

        if len(valid_items) < 2:
            # Fall back to TF-IDF on all items
            return self.cluster(items, k=k, auto_k=auto_k, k_min=k_min, k_max=k_max)

        X = np.vstack(X_rows)

        if auto_k:
            k = _elbow_k(X, k_min=k_min, k_max=k_max, seed=self.seed)
        elif k is None:
            raise ValueError("Either k or auto_k=True must be provided.")

        k = max(1, min(k, len(valid_items)))
        labels, centroids, _ = _run_kmeans(X, k=k, max_iter=self.max_iter, seed=self.seed)

        cluster_map: dict[int, list["MemoryItem"]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[int(label)].append(valid_items[idx])

        clusters: list[MemoryCluster] = []
        for cid in sorted(cluster_map.keys()):
            cluster_items = cluster_map[cid]
            member_vecs = X[labels == cid]
            per_inertia = float(np.sum(
                np.sum((member_vecs - centroids[cid]) ** 2, axis=1)
            ))
            c = MemoryCluster(
                id=cid,
                members=cluster_items,
                centroid=centroids[cid].copy(),
                label=_cluster_label(cluster_items),
                inertia=per_inertia,
            )
            clusters.append(c)

        return clusters
