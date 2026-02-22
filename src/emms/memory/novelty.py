"""NoveltyDetector — scoring memory novelty against the corpus centroid.

v0.22.0: The Creative Mind

Novelty detection is the cognitive capacity to recognise when something is
*surprising* — when an experience deviates from what the mind has come to
expect. This capacity is fundamental to learning: surprising events capture
attention, trigger deeper encoding, and signal that the world contains
information that existing knowledge does not yet explain. Without novelty
detection, an agent accumulates experience indiscriminately, unable to
distinguish the unusual from the routine.

NoveltyDetector operationalises this for the memory store. It constructs a
global token-frequency map across all accumulated memories — the *corpus
centroid*, representing what is typical. For each memory, it then scores
novelty as the fraction of its tokens that are rare in the corpus (appearing
in fewer than a threshold number of memories). Memories whose tokens are
mostly common score near zero; memories whose tokens are largely unique to
themselves score near one. The resulting NoveltyScore objects reveal which
experiences were most surprising relative to the agent's prior knowledge.

This is related to but distinct from curiosity (v0.16.0, which drives
exploration of unknown *topics*) — novelty detection operates *post-hoc*
on stored memories rather than prospectively on information gaps.

Biological analogue: hippocampal novelty detection — the CA1 region responds
specifically to novel stimuli by comparing incoming patterns against stored
representations (Kumaran & Maguire 2007); the dopaminergic prediction-error
signal (Schultz 1998) — dopamine neurons fire in response to unexpected
rewards, encoding the surprise of an outcome relative to prediction; the locus
coeruleus-norepinephrine system mediates arousal responses to novelty; the
orbitofrontal cortex encodes novelty-driven exploration value (Dayan & Daw
2008); novelty-P300 ERP component reflecting cortical surprise detection.
"""

from __future__ import annotations

import time
from collections import Counter
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
class NoveltyScore:
    """Novelty score for a single memory item."""

    memory_id: str
    content_excerpt: str    # first 80 characters of content
    novelty: float          # 0..1 — fraction of tokens that are rare
    domain: str
    rare_tokens: list[str]  # tokens with below-threshold corpus frequency

    def summary(self) -> str:
        rare = ", ".join(self.rare_tokens[:5]) or "—"
        return (
            f"NoveltyScore [{self.domain}]  novelty={self.novelty:.3f}  "
            f"rare=[{rare}]\n"
            f"  {self.memory_id[:12]}: {self.content_excerpt[:60]}"
        )


@dataclass
class NoveltyReport:
    """Result of a NoveltyDetector.assess() call."""

    total_assessed: int
    high_novelty_count: int     # scores above novelty_threshold
    mean_novelty: float
    scores: list[NoveltyScore]  # sorted by novelty descending
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"NoveltyReport: {self.total_assessed} memories assessed  "
            f"{self.high_novelty_count} high-novelty  "
            f"mean={self.mean_novelty:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for s in self.scores[:5]:
            lines.append(f"  {s.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NoveltyDetector
# ---------------------------------------------------------------------------


class NoveltyDetector:
    """Scores each memory by how unusual it is relative to the corpus centroid.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    novelty_threshold:
        Novelty score above which a memory is considered "high novelty"
        (default 0.6).
    max_scores:
        Maximum number of :class:`NoveltyScore` objects to retain in the
        report (default 50).
    """

    def __init__(
        self,
        memory: Any,
        novelty_threshold: float = 0.6,
        max_scores: int = 50,
    ) -> None:
        self.memory = memory
        self.novelty_threshold = novelty_threshold
        self.max_scores = max_scores
        self._scores: dict[str, NoveltyScore] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, domain: Optional[str] = None) -> NoveltyReport:
        """Score all memories for novelty against the full corpus.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`NoveltyReport` with scores sorted by novelty descending.
        """
        t0 = time.time()
        all_items = self._collect_all()

        # Build global token-frequency counter from ALL items (not filtered)
        global_token_doc_freq: Counter = Counter()
        for item in all_items:
            content = getattr(item.experience, "content", "") or ""
            tokens = self._tokenise(content)
            for tok in set(tokens):  # count once per document
                global_token_doc_freq[tok] += 1

        n_total = max(len(all_items), 1)
        # Rarity threshold: a token is "rare" if it appears in < threshold docs
        rarity_threshold = max(2, int(n_total * 0.10))

        # Now score the target items (optionally filtered by domain)
        target_items = all_items
        if domain:
            target_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        self._scores.clear()
        all_scores: list[NoveltyScore] = []
        for item in target_items:
            ns = self._score_item(item, global_token_doc_freq, rarity_threshold)
            self._scores[item.id] = ns
            all_scores.append(ns)

        all_scores.sort(key=lambda s: s.novelty, reverse=True)
        top_scores = all_scores[: self.max_scores]

        high_count = sum(1 for s in all_scores if s.novelty >= self.novelty_threshold)
        mean_n = (
            sum(s.novelty for s in all_scores) / len(all_scores)
            if all_scores else 0.0
        )

        return NoveltyReport(
            total_assessed=len(all_scores),
            high_novelty_count=high_count,
            mean_novelty=round(mean_n, 4),
            scores=top_scores,
            duration_seconds=time.time() - t0,
        )

    def most_novel(self, n: int = 5) -> list[NoveltyScore]:
        """Return the n most novel memories from the last assessment.

        Args:
            n: Number of scores to return (default 5).
        """
        return sorted(self._scores.values(), key=lambda s: s.novelty, reverse=True)[:n]

    def novelty_of(self, memory_id: str) -> float:
        """Return the novelty score for a specific memory.

        Args:
            memory_id: ID of the memory to query.

        Returns:
            Novelty score 0..1, or 0.5 (neutral) if not yet assessed.
        """
        ns = self._scores.get(memory_id)
        return ns.novelty if ns is not None else 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_item(
        self,
        item: Any,
        freq_counter: Counter,
        rarity_threshold: int,
    ) -> NoveltyScore:
        """Compute a NoveltyScore for a single memory item."""
        content = getattr(item.experience, "content", "") or ""
        dom = getattr(item.experience, "domain", None) or "general"
        tokens = self._tokenise(content)

        if not tokens:
            return NoveltyScore(
                memory_id=item.id,
                content_excerpt=content[:80],
                novelty=0.0,
                domain=dom,
                rare_tokens=[],
            )

        rare = [t for t in tokens if freq_counter.get(t, 0) < rarity_threshold]
        novelty = round(len(rare) / len(tokens), 4)

        return NoveltyScore(
            memory_id=item.id,
            content_excerpt=content[:80],
            novelty=novelty,
            domain=dom,
            rare_tokens=list(dict.fromkeys(rare))[:8],  # ordered unique
        )

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
