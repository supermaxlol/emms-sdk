"""Embedding providers for EMMS.

Abstracts away the embedding model so users can swap between:
  - HashEmbedder:   zero-dependency, deterministic (default)
  - SentenceTransformerEmbedder: high-quality, requires sentence-transformers
  - Any callable that maps str → list[float]
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into a fixed-size float vector."""

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple strings (default: serial)."""
        ...


# ---------------------------------------------------------------------------
# HashEmbedder — zero-dependency, deterministic, surprisingly decent
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic embedding via locality-sensitive hashing.

    Uses multiple hash functions with sinusoidal projections to create
    a dense vector that preserves rough textual similarity.
    Zero external dependencies — ships with numpy only.
    """

    def __init__(self, dim: int = 128, ngram_range: tuple[int, int] = (2, 4)):
        self._dim = dim
        self._ngram_range = ngram_range

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        text_lower = text.lower().strip()
        if not text_lower:
            return vec.tolist()

        # Character n-gram hashing
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i : i + n]
                h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                idx = h % self._dim
                # Use sin/cos projections for smoother distribution
                vec[idx] += math.cos(h * 0.0001)
                vec[(idx + 1) % self._dim] += math.sin(h * 0.0001)

        # Word-level hashing with multiple projections per word
        # More projections = better discrimination between documents
        words = [w for w in text_lower.split() if len(w) >= 2]
        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            # 4 projections per word (was 2) for better coverage
            for offset in (0, 3, 7, 13):
                idx = (h + offset * 31) % self._dim
                vec[idx] += 1.0 / (1 + offset * 0.3)

        # Word bigram hashing (captures 2-word phrases)
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 0.8
            vec[(idx + 5) % self._dim] += 0.4

        # L2-normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — high-quality, requires extra install
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Wraps sentence-transformers for high-quality semantic embeddings.

    Requires: pip install sentence-transformers
    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), convert_to_numpy=True)
        return vecs.tolist()


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))
