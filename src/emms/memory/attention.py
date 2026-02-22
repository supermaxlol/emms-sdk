"""AttentionFilter — selective attention modulating memory retrieval.

v0.17.0: The Goal-Directed Mind

Not all memories are equally relevant in every moment. The AttentionFilter
models a dynamic attentional spotlight: a set of active goals, topics, and
keywords that currently hold the agent's focus. When retrieving memories,
each item is scored by attentional relevance — how much it overlaps with
the spotlight — and the top-k most attended items are returned.

Attention score composition:
    attention = 0.40 * goal_relevance
              + 0.30 * importance
              + 0.20 * keyword_overlap
              + 0.10 * recency_score

``goal_relevance``  — token Jaccard between item content and the content
                       of all active goals in the attached GoalStack
``importance``      — memory_strength (already 0..2, clamped to 0..1)
``keyword_overlap`` — Jaccard between item tokens and spotlight keywords
``recency_score``   — 1 / (1 + age_in_days) for short-lived spotlight focus

Biological analogue: spotlight model of attention (Posner 1980) — selective
attention operates as a spatiotemporal spotlight, enhancing processing of
attended stimuli; biased competition (Desimone & Duncan 1995) — goal
representations in prefrontal cortex provide top-down bias signals that
modulate competition between memory traces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AttentionResult:
    """Attention scoring for a single memory item."""

    memory_id: str
    content_excerpt: str
    domain: str
    attention_score: float   # composite 0..1
    goal_relevance: float    # component scores
    importance: float
    recency_score: float
    keyword_overlap: float

    def summary(self) -> str:
        return (
            f"  [{self.memory_id[:12]}] attn={self.attention_score:.3f}  "
            f"goal={self.goal_relevance:.2f}  kw={self.keyword_overlap:.2f}  "
            f"[{self.domain}] {self.content_excerpt[:50]}"
        )


@dataclass
class AttentionReport:
    """Result of an AttentionFilter.spotlight_retrieve() call."""

    spotlight_keywords: list[str]
    spotlight_goal_ids: list[str]
    items_scored: int
    results: list[AttentionResult]     # top-k, sorted by attention_score desc
    top_domain: str
    duration_seconds: float

    def summary(self) -> str:
        kw_str = ", ".join(self.spotlight_keywords[:8]) or "(none)"
        lines = [
            f"AttentionReport: {self.items_scored} scored, "
            f"{len(self.results)} returned  "
            f"top_domain={self.top_domain}  "
            f"spotlight=[{kw_str}]",
        ]
        for r in self.results[:5]:
            lines.append(r.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AttentionFilter
# ---------------------------------------------------------------------------


class AttentionFilter:
    """Selective attention spotlight for memory retrieval.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    goal_stack:
        Optional :class:`GoalStack` whose active goals inform the spotlight.
    """

    def __init__(
        self,
        memory: Any,
        goal_stack: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.goal_stack = goal_stack
        self._spotlight_keywords: set[str] = set()
        self._spotlight_goal_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_spotlight(
        self,
        text: Optional[str] = None,
        goal_ids: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
    ) -> None:
        """Expand the attentional spotlight with new content.

        Args:
            text:      Free text — tokens extracted and added to spotlight.
            goal_ids:  Goal IDs from an attached GoalStack to track.
            keywords:  Explicit keyword list to add.
        """
        if text:
            self._spotlight_keywords.update(self._tokenise(text))
        if keywords:
            self._spotlight_keywords.update(
                w.lower().strip(".,!?;:\"'()") for w in keywords
            )
        if goal_ids:
            self._spotlight_goal_ids.update(goal_ids)

    def spotlight_retrieve(self, k: int = 8) -> AttentionReport:
        """Return the k most attention-relevant memories.

        Integrates active goal content from the attached GoalStack (if any)
        into the spotlight before scoring.

        Args:
            k: Maximum number of results to return.

        Returns:
            :class:`AttentionReport` with scored results.
        """
        t0 = time.time()
        items = self._collect_all()

        # Pull in active goal content
        goal_keywords: set[str] = set()
        active_goal_ids: list[str] = list(self._spotlight_goal_ids)
        if self.goal_stack is not None:
            for g in self.goal_stack.active_goals():
                goal_keywords.update(self._tokenise(g.content))
                if g.id not in active_goal_ids:
                    active_goal_ids.append(g.id)

        combined_keywords = self._spotlight_keywords | goal_keywords

        results: list[AttentionResult] = []
        for item in items:
            results.append(self._score_item(item, combined_keywords, goal_keywords))

        results.sort(key=lambda r: r.attention_score, reverse=True)
        top_k = results[:k]

        # Determine top domain
        if top_k:
            from collections import Counter
            top_domain = Counter(r.domain for r in top_k).most_common(1)[0][0]
        else:
            top_domain = ""

        return AttentionReport(
            spotlight_keywords=sorted(combined_keywords)[:20],
            spotlight_goal_ids=active_goal_ids,
            items_scored=len(items),
            results=top_k,
            top_domain=top_domain,
            duration_seconds=time.time() - t0,
        )

    def attention_profile(self) -> dict[str, float]:
        """Return mean attention scores per domain.

        Returns:
            Dict mapping domain → mean attention score (0..1).
        """
        items = self._collect_all()
        if not items:
            return {}

        goal_keywords: set[str] = set()
        if self.goal_stack is not None:
            for g in self.goal_stack.active_goals():
                goal_keywords.update(self._tokenise(g.content))
        combined = self._spotlight_keywords | goal_keywords

        from collections import defaultdict
        domain_scores: dict[str, list[float]] = defaultdict(list)
        for item in items:
            r = self._score_item(item, combined, goal_keywords)
            dom = getattr(item.experience, "domain", None) or "general"
            domain_scores[dom].append(r.attention_score)

        return {
            dom: round(sum(scores) / len(scores), 4)
            for dom, scores in domain_scores.items()
        }

    def clear_spotlight(self) -> None:
        """Reset the attentional spotlight to empty."""
        self._spotlight_keywords.clear()
        self._spotlight_goal_ids.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_item(
        self,
        item: Any,
        combined_keywords: set[str],
        goal_keywords: set[str],
    ) -> AttentionResult:
        """Compute the full attention score for one memory item."""
        now = time.time()
        content = getattr(item.experience, "content", "") or ""
        dom = getattr(item.experience, "domain", None) or "general"

        # Goal relevance — overlap with active goal tokens
        item_tokens = self._tokenise(content)
        goal_relevance = self._jaccard(item_tokens, goal_keywords) if goal_keywords else 0.0

        # Importance — memory_strength clamped to [0, 1]
        importance = min(1.0, max(0.0, getattr(item, "memory_strength", 0.5)))

        # Keyword overlap — with the full spotlight
        keyword_overlap = self._jaccard(item_tokens, combined_keywords) if combined_keywords else 0.0

        # Recency — exponential decay: 1 / (1 + age_days)
        last_access = getattr(item, "last_accessed", None) or getattr(item, "stored_at", now)
        age_days = max(0.0, (now - last_access) / 86400.0)
        recency_score = 1.0 / (1.0 + age_days)

        attention_score = (
            0.40 * goal_relevance
            + 0.30 * importance
            + 0.20 * keyword_overlap
            + 0.10 * recency_score
        )
        attention_score = round(min(1.0, attention_score), 5)

        return AttentionResult(
            memory_id=item.id,
            content_excerpt=content[:80],
            domain=dom,
            attention_score=attention_score,
            goal_relevance=round(goal_relevance, 4),
            importance=round(importance, 4),
            recency_score=round(recency_score, 4),
            keyword_overlap=round(keyword_overlap, 4),
        )

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()")
            for w in text.lower().split()
            if len(w) >= 4 and w not in _STOP_WORDS
        }

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
