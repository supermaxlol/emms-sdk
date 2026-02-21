"""ContextualSalienceRetriever — dynamic memory spotlight on current context.

v0.13.0: The Metacognitive Layer

Classic retrieval answers: "which memories match this query?"
Contextual salience answers: "which memories are most relevant to what is
happening *right now* — the accumulated flow of this conversation?"

The retriever maintains a rolling context window of recent text. As each turn
arrives, the window updates. Retrieval scores memories on four axes:

  - **Semantic overlap** with accumulated context tokens
  - **Importance** of the memory item
  - **Recency** (recently stored memories are more plastic / salient)
  - **Affective resonance** (emotional match between memory valence and
    the mean valence of recent context)

Biological analogue: context-dependent memory — the well-documented phenomenon
where recall is best when the retrieval context matches the encoding context
(Godden & Baddeley 1975). The current conversational mood, topic, and emotional
tone prime a specific subset of memories.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SalienceResult:
    """A memory that is salient to the current conversational context."""

    memory: Any              # the MemoryItem
    salience_score: float    # combined salience [0,1]
    semantic_overlap: float  # token overlap with context window
    importance_factor: float # normalised experience importance
    recency_factor: float    # how recently the item was stored
    affective_resonance: float  # emotional match with context valence


# ---------------------------------------------------------------------------
# ContextualSalienceRetriever
# ---------------------------------------------------------------------------

class ContextualSalienceRetriever:
    """Retrieves memories by salience to the current rolling context window.

    Call :meth:`update_context` after each conversational turn to keep the
    context window current. Then call :meth:`retrieve` to get the memories
    most relevant to the ongoing conversation.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    window_size:
        Number of recent text snippets to keep in the context window
        (default 6).
    semantic_weight:
        Weight of semantic overlap in the combined score (default 0.35).
    importance_weight:
        Weight of memory importance (default 0.30).
    recency_weight:
        Weight of storage recency (default 0.25).
    affective_weight:
        Weight of affective resonance (default 0.10).
    recency_half_life_days:
        Days at which recency factor decays to 0.5 (default 7).
    """

    def __init__(
        self,
        memory: Any,
        window_size: int = 6,
        semantic_weight: float = 0.35,
        importance_weight: float = 0.30,
        recency_weight: float = 0.25,
        affective_weight: float = 0.10,
        recency_half_life_days: float = 7.0,
    ) -> None:
        self.memory = memory
        self.window_size = window_size
        self.semantic_weight = semantic_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.affective_weight = affective_weight
        self.recency_half_life_days = recency_half_life_days

        self._context_window: deque[str] = deque(maxlen=window_size)
        self._valence_window: deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_context(self, text: str, valence: float = 0.0) -> None:
        """Add a new text snippet to the rolling context window.

        Args:
            text:    The text to add (e.g. user message + agent reply).
            valence: Estimated emotional valence of this snippet (default 0.0).
        """
        self._context_window.append(text)
        self._valence_window.append(max(-1.0, min(1.0, valence)))

    def retrieve(self, max_results: int = 10) -> list[SalienceResult]:
        """Retrieve memories most salient to the current context window.

        Args:
            max_results: Maximum memories to return (default 10).

        Returns:
            List of :class:`SalienceResult` sorted by salience_score descending.
            Returns empty list if the context window is empty.
        """
        if not self._context_window:
            return []

        context_tokens = self._context_tokens()
        ctx_valence = self.context_valence

        items = self._collect_all()
        if not items:
            return []

        now = time.time()
        recency_decay = math.log(2) / max(self.recency_half_life_days * 86400, 1)

        results: list[SalienceResult] = []
        for item in items:
            sr = self._score(item, context_tokens, ctx_valence, now, recency_decay)
            if sr.salience_score > 0.0:
                results.append(sr)

        results.sort(key=lambda r: r.salience_score, reverse=True)
        return results[:max_results]

    @property
    def context_summary(self) -> str:
        """A brief summary of the current context window."""
        if not self._context_window:
            return "(empty context)"
        joined = " | ".join(str(t)[:60] for t in self._context_window)
        return f"[{len(self._context_window)} turns] {joined}"

    @property
    def context_valence(self) -> float:
        """Mean emotional valence of the current context window."""
        if not self._valence_window:
            return 0.0
        return sum(self._valence_window) / len(self._valence_window)

    @property
    def context_tokens(self) -> set[str]:
        """Token set of the accumulated context window."""
        return self._context_tokens()

    def reset_context(self) -> None:
        """Clear the context window."""
        self._context_window.clear()
        self._valence_window.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(
        self,
        item: Any,
        context_tokens: set[str],
        ctx_valence: float,
        now: float,
        recency_decay: float,
    ) -> SalienceResult:
        # 1. Semantic overlap (Jaccard against context token set)
        item_tokens = set(item.experience.content.lower().split())
        if context_tokens and item_tokens:
            intersection = context_tokens & item_tokens
            union = context_tokens | item_tokens
            semantic_overlap = len(intersection) / max(len(union), 1)
        else:
            semantic_overlap = 0.0

        # 2. Importance factor (already 0..1)
        importance_factor = float(item.experience.importance)

        # 3. Recency factor (exponential decay from stored_at)
        age_seconds = max(0.0, now - item.stored_at)
        recency_factor = math.exp(-recency_decay * age_seconds)

        # 4. Affective resonance
        item_valence = getattr(item.experience, "emotional_valence", 0.0) or 0.0
        valence_diff = abs(item_valence - ctx_valence)
        affective_resonance = max(0.0, 1.0 - valence_diff)

        # Combined weighted score
        salience_score = (
            self.semantic_weight * semantic_overlap
            + self.importance_weight * importance_factor
            + self.recency_weight * recency_factor
            + self.affective_weight * affective_resonance
        )

        return SalienceResult(
            memory=item,
            salience_score=salience_score,
            semantic_overlap=semantic_overlap,
            importance_factor=importance_factor,
            recency_factor=recency_factor,
            affective_resonance=affective_resonance,
        )

    def _context_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for text in self._context_window:
            tokens.update(str(text).lower().split())
        return tokens

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
