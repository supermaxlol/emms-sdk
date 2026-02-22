"""EmotionalRegulator — emotion regulation via cognitive reappraisal.

v0.19.0: The Integrated Mind

Emotions are not noise — they are information. But raw emotional states can
distort cognition if unregulated. Cognitive reappraisal (Gross 1998) is the
most adaptive regulation strategy: reinterpreting the meaning of an event to
alter its emotional impact. Unlike suppression, reappraisal changes the
emotional trajectory at the source — it generates a new, growth-oriented
framing of negative experiences.

The EmotionalRegulator operationalises this for the memory store:

1. **Emotional State**: computes the agent's current mood valence (mean of
   recent memory valences) and arousal (standard deviation — high variance =
   turbulent affect; low variance = stable mood).

2. **Cognitive Reappraisal**: for memories with strongly negative valence
   (< −0.3), generates an alternative framing and optionally stores it as a
   new "reappraisal" domain memory. The new_valence is shifted +0.3 toward
   positive (clamped to −1..1).

3. **Mood-congruent Retrieval**: returns memories whose valence is closest to
   the current emotional state — mood congruency is a powerful retrieval bias
   in human memory (Bower 1981).

4. **Emotional Coherence**: measures valence consistency across all memories
   (1 − std). High coherence = stable, integrated affect; low coherence =
   conflicted or mixed emotional state.

Biological analogue: process model of emotion regulation (Gross 1998);
cognitive reappraisal as the most adaptive regulation strategy (Aldao et al.
2010); mood-congruent memory effects (Bower 1981); amygdala–prefrontal cortex
interaction in affect regulation (Ochsner & Gross 2005); affective valence as
a primary dimension of emotional experience (Russell 1980).
"""

from __future__ import annotations

import math
import time
import uuid
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
class EmotionalState:
    """The agent's current emotional state computed from recent memories."""

    valence: float        # -1..1 mean mood
    arousal: float        # 0..1 emotional intensity (stdev of recent valences)
    dominant_domain: str  # domain contributing most to current mood
    sample_size: int      # number of memories used to compute state
    computed_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        tone = "positive" if self.valence > 0.1 else "negative" if self.valence < -0.1 else "neutral"
        return (
            f"EmotionalState: valence={self.valence:+.3f} ({tone})  "
            f"arousal={self.arousal:.3f}  dominant_domain={self.dominant_domain!r}  "
            f"sample={self.sample_size}"
        )


@dataclass
class ReappraisalResult:
    """Result of cognitive reappraisal applied to one memory."""

    memory_id: str
    original_valence: float
    reappraised_content: str       # the alternative framing
    new_valence: float             # post-reappraisal valence
    shift: float                   # new_valence − original_valence
    stored_as_memory_id: Optional[str] = None  # ID if persisted as a new memory

    def summary(self) -> str:
        return (
            f"Reappraisal [{self.memory_id[:12]}]  "
            f"valence {self.original_valence:+.3f} → {self.new_valence:+.3f}  "
            f"shift={self.shift:+.3f}"
        )


@dataclass
class EmotionReport:
    """Full emotion regulation report from EmotionalRegulator.regulate()."""

    current_state: EmotionalState
    memories_assessed: int
    reappraisals: list[ReappraisalResult]
    mood_congruent_ids: list[str]   # memory IDs most resonant with current mood
    emotional_coherence: float       # 0..1 valence consistency
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"EmotionReport: {self.memories_assessed} memories assessed  "
            f"{len(self.reappraisals)} reappraisals  "
            f"coherence={self.emotional_coherence:.3f}  "
            f"in {self.duration_seconds:.2f}s",
            f"  {self.current_state.summary()}",
        ]
        if self.reappraisals:
            lines.append(
                f"  Latest reappraisal: {self.reappraisals[-1].reappraised_content[:80]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EmotionalRegulator
# ---------------------------------------------------------------------------


class EmotionalRegulator:
    """Tracks emotional state and applies cognitive reappraisal.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    window_memories:
        Number of most-recent memories used to compute emotional state (default 20).
    reappraise:
        If ``True``, generate reappraisals for strongly negative memories (default True).
    store_reappraisals:
        If ``True``, persist each reappraisal as a new memory (default True).
    reappraisal_importance:
        Importance assigned to stored reappraisal memories (default 0.6).
    """

    def __init__(
        self,
        memory: Any,
        window_memories: int = 20,
        reappraise: bool = True,
        store_reappraisals: bool = True,
        reappraisal_importance: float = 0.6,
    ) -> None:
        self.memory = memory
        self.window_memories = window_memories
        self.reappraise = reappraise
        self.store_reappraisals = store_reappraisals
        self.reappraisal_importance = reappraisal_importance
        self._current_state: Optional[EmotionalState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def regulate(self, domain: Optional[str] = None) -> EmotionReport:
        """Assess emotional state and apply regulation strategies.

        Args:
            domain: Restrict assessment to this domain (``None`` = all domains).

        Returns:
            :class:`EmotionReport` with state, reappraisals, and mood-congruent IDs.
        """
        t0 = time.time()
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Sort by stored_at descending for window computation
        items_sorted = sorted(
            items,
            key=lambda x: getattr(x.experience, "stored_at", 0.0) or 0.0,
            reverse=True,
        )
        window = items_sorted[: self.window_memories]

        state = self._compute_state(window if window else items_sorted[:1])
        self._current_state = state

        # Reappraisals for negative-valence items
        reappraisals: list[ReappraisalResult] = []
        if self.reappraise and window:
            for item in window:
                v = getattr(item.experience, "emotional_valence", 0.0) or 0.0
                if v < -0.3:
                    result = self._reappraise(item, state)
                    if result is not None:
                        reappraisals.append(result)

        # Mood-congruent IDs (top-8 closest to current mood valence)
        all_items = self._collect_all()
        mood_ids = self._mood_congruent(all_items, state.valence, k=8)

        # Emotional coherence across all memories
        coherence = self.emotional_coherence()

        return EmotionReport(
            current_state=state,
            memories_assessed=len(items),
            reappraisals=reappraisals,
            mood_congruent_ids=mood_ids,
            emotional_coherence=coherence,
            duration_seconds=time.time() - t0,
        )

    def current_state(self) -> Optional[EmotionalState]:
        """Return the most recently computed emotional state, or ``None``."""
        return self._current_state

    def mood_retrieve(self, k: int = 8) -> list[Any]:
        """Return the k memories most resonant with the current emotional state.

        Args:
            k: Number of items to return (default 8).

        Returns:
            List of memory items sorted by mood-congruence descending.
        """
        state = self._current_state
        if state is None:
            return []
        items = self._collect_all()
        return sorted(
            items,
            key=lambda x: abs(
                (getattr(x.experience, "emotional_valence", 0.0) or 0.0)
                - state.valence
            ),
        )[:k]

    def emotional_coherence(self) -> float:
        """Compute valence coherence across all memories (0..1).

        Returns:
            ``1 − std(valences)`` clamped to 0..1.
        """
        items = self._collect_all()
        if not items:
            return 1.0
        valences = [
            getattr(it.experience, "emotional_valence", 0.0) or 0.0
            for it in items
        ]
        if len(valences) < 2:
            return 1.0
        mean_v = sum(valences) / len(valences)
        variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
        std_v = math.sqrt(variance)
        return round(max(0.0, min(1.0, 1.0 - std_v)), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_state(self, window: list[Any]) -> EmotionalState:
        """Compute EmotionalState from a list of memory items."""
        if not window:
            return EmotionalState(
                valence=0.0, arousal=0.0,
                dominant_domain="general", sample_size=0,
            )

        valences = [
            getattr(it.experience, "emotional_valence", 0.0) or 0.0
            for it in window
        ]
        mean_v = round(sum(valences) / len(valences), 4)

        # Arousal = population std of valences in window, clamped 0..1
        if len(valences) >= 2:
            variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
            arousal = round(min(1.0, math.sqrt(variance)), 4)
        else:
            arousal = 0.0

        # Dominant domain = mode of domains in window
        domain_counts: Counter = Counter(
            getattr(it.experience, "domain", None) or "general"
            for it in window
        )
        dominant = domain_counts.most_common(1)[0][0]

        return EmotionalState(
            valence=mean_v,
            arousal=arousal,
            dominant_domain=dominant,
            sample_size=len(window),
        )

    def _reappraise(
        self,
        item: Any,
        state: EmotionalState,
    ) -> Optional[ReappraisalResult]:
        """Generate a cognitive reappraisal for a negative-valence memory."""
        content = getattr(item.experience, "content", "") or ""
        original_v = getattr(item.experience, "emotional_valence", 0.0) or 0.0
        domain = getattr(item.experience, "domain", None) or "general"

        # Find the most salient token in this memory
        tokens = [
            w.strip(".,!?;:\"'()") for w in content.lower().split()
            if len(w.strip(".,!?;:\"'()")) >= 4
            and w.strip(".,!?;:\"'()") not in _STOP_WORDS
        ]
        top_token = tokens[0] if tokens else "this experience"

        reappraised = (
            f"Reappraising: [{content[:60]}] — viewed from a growth perspective, "
            f"this pattern around '{top_token}' may indicate opportunity for "
            f"development and deeper understanding in the {domain} domain."
        )
        new_v = round(min(1.0, original_v + 0.3), 4)
        shift = round(new_v - original_v, 4)

        stored_id: Optional[str] = None
        if self.store_reappraisals:
            stored_id = self._store_reappraisal(reappraised, domain, new_v)

        return ReappraisalResult(
            memory_id=item.id,
            original_valence=original_v,
            reappraised_content=reappraised,
            new_valence=new_v,
            shift=shift,
            stored_as_memory_id=stored_id,
        )

    def _store_reappraisal(
        self, content: str, domain: str, valence: float
    ) -> Optional[str]:
        """Store a reappraisal as a new memory; return its ID."""
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=content,
                domain="reappraisal",
                importance=self.reappraisal_importance,
                emotional_valence=valence,
            )
            item = self.memory.store(exp)
            return getattr(item, "id", None)
        except Exception:
            return None

    def _mood_congruent(
        self, items: list[Any], target_valence: float, k: int
    ) -> list[str]:
        """Return top-k memory IDs most resonant with target_valence."""
        ranked = sorted(
            items,
            key=lambda x: abs(
                (getattr(x.experience, "emotional_valence", 0.0) or 0.0)
                - target_valence
            ),
        )
        return [it.id for it in ranked[:k]]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
