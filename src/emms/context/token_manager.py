"""Token-level context management for LLM context windows.

Implements intelligent eviction so that the most relevant tokens stay in
context while less important ones are moved to episodic storage and can be
retrieved later.  Inspired by EM-LLM's attention-sink approach.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Sequence

from emms.core.models import MemoryConfig

logger = logging.getLogger(__name__)


class TokenContextManager:
    """Manages an LLM's finite context window with smart eviction/retrieval."""

    def __init__(self, config: MemoryConfig | None = None):
        cfg = config or MemoryConfig()
        self.context_window = cfg.context_window
        self.eviction_ratio = cfg.eviction_ratio

        # Three-tier token system (EM-LLM style)
        self.initial_tokens: list[str] = []      # Attention sinks (first N)
        self.local_context: list[str] = []        # Active context window
        self.evicted_tokens: list[str] = []       # Episodic storage

        # Per-token metadata
        self._importance: dict[str, float] = defaultdict(lambda: 0.5)
        self._frequency: dict[str, int] = defaultdict(int)

        # Stats
        self.total_evicted = 0
        self.total_retrieved = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, tokens: Sequence[str]) -> list[str]:
        """Add tokens to context, evicting if necessary. Returns the full context."""
        needed = len(self.local_context) + len(tokens)

        if needed > self.context_window:
            overflow = needed - self.context_window
            self._evict(overflow)

        self.local_context.extend(tokens)
        self._update_metadata(tokens)
        return self.get_context()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self, count: int) -> list[str]:
        """Evict *count* lowest-scoring tokens from local context."""
        if not self.local_context or count <= 0:
            return []

        scored = []
        for i, tok in enumerate(self.local_context):
            recency = (len(self.local_context) - i) / len(self.local_context)
            importance = self._importance[tok]
            frequency = min(1.0, self._frequency[tok] / 10)
            score = recency * 0.4 + importance * 0.4 + frequency * 0.2
            scored.append((score, i, tok))

        scored.sort(key=lambda x: x[0])

        evicted_indices: set[int] = set()
        evicted: list[str] = []
        for score, idx, tok in scored[:count]:
            evicted.append(tok)
            evicted_indices.add(idx)

        self.evicted_tokens.extend(evicted)
        self.local_context = [
            t for i, t in enumerate(self.local_context) if i not in evicted_indices
        ]
        self.total_evicted += len(evicted)
        return evicted

    # ------------------------------------------------------------------
    # Retrieval from evicted storage
    # ------------------------------------------------------------------

    def retrieve_relevant(self, keywords: set[str], max_tokens: int = 200) -> list[str]:
        """Pull tokens back from evicted storage that match *keywords*."""
        if not self.evicted_tokens or not keywords:
            return []

        keywords_lower = {k.lower() for k in keywords}
        results: list[tuple[float, str]] = []

        for tok in self.evicted_tokens:
            tok_lower = tok.lower()
            if tok_lower in keywords_lower:
                results.append((1.0, tok))
            else:
                # Partial character overlap
                best = max(
                    (
                        len(set(tok_lower) & set(kw)) / max(len(tok_lower), len(kw))
                        for kw in keywords_lower
                        if len(kw) > 2
                    ),
                    default=0.0,
                )
                if best > 0.6:
                    results.append((best, tok))

        results.sort(reverse=True)
        retrieved = [tok for _, tok in results[:max_tokens]]
        self.total_retrieved += len(retrieved)
        return retrieved

    # ------------------------------------------------------------------
    # Context access
    # ------------------------------------------------------------------

    def get_context(self) -> list[str]:
        """Return the full token context (initial + local)."""
        return self.initial_tokens + self.local_context

    @property
    def utilisation(self) -> float:
        """Fraction of context window currently used."""
        return len(self.local_context) / self.context_window if self.context_window else 0

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "context_window": self.context_window,
            "local_tokens": len(self.local_context),
            "evicted_tokens": len(self.evicted_tokens),
            "utilisation": self.utilisation,
            "total_evicted": self.total_evicted,
            "total_retrieved": self.total_retrieved,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_metadata(self, tokens: Sequence[str]) -> None:
        for tok in tokens:
            self._frequency[tok] += 1
            self._importance[tok] = self._token_importance(tok)

    @staticmethod
    def _token_importance(token: str) -> float:
        length = min(1.0, len(token) / 10)
        if token.isupper():
            kind = 0.9
        elif token.isdigit():
            kind = 0.8
        elif len(token) > 8:
            kind = 0.9
        else:
            kind = 0.5
        return (length + kind) / 2
