"""Spaced Repetition System (SRS) for EMMS — SM-2 algorithm.

Implements the SM-2 spaced repetition algorithm (Wozniak, 1990) to schedule
periodic memory reviews, prioritising under-reviewed and difficult memories.

Memory items are enrolled into the SRS by calling ``SpacedRepetitionSystem.enroll()``.
After each review, the caller provides a quality score (0–5) and the system
updates the review schedule using the SM-2 easiness-factor update:

    EF' = max(1.3, EF + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

where q ≥ 3 means a successful recall and q < 3 means a lapse (reset).

Integration with EMMS:

* ``MemoryItem.srs_enrolled``      — bool, whether enrolled in SRS
* ``MemoryItem.srs_next_review``   — float (Unix timestamp), when next due
* ``MemoryItem.srs_interval_days`` — float, current interval in days

Usage::

    from emms import EMMS
    from emms.memory.spaced_repetition import SpacedRepetitionSystem

    agent = EMMS()
    srs = SpacedRepetitionSystem(agent.memory)

    # Enrol a specific memory:
    srs.enroll("mem_abc123")

    # Mark a review (quality 0-5):
    srs.record_review("mem_abc123", quality=4)

    # Get items due for review now:
    due = srs.get_due_items()

Or via EMMS façade:

    agent.srs_enroll("mem_abc123")
    agent.srs_record_review("mem_abc123", quality=4)
    due = agent.srs_due()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from emms.memory.hierarchical import HierarchicalMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SM-2 constants
# ---------------------------------------------------------------------------

_INITIAL_EF: float = 2.5          # starting easiness factor
_MIN_EF: float = 1.3              # floor for easiness factor
_INITIAL_INTERVAL_DAYS: float = 1.0


# ---------------------------------------------------------------------------
# SRSCard — persisted per-memory SRS state
# ---------------------------------------------------------------------------

class SRSCard(BaseModel):
    """SM-2 scheduling card for a single memory item."""

    memory_id: str

    # SM-2 state
    repetitions: int = 0                        # number of successful reviews
    easiness_factor: float = _INITIAL_EF        # EF — drives interval growth
    interval_days: float = _INITIAL_INTERVAL_DAYS  # days until next review
    next_review: float = Field(default_factory=time.time)  # Unix timestamp

    # Audit
    last_reviewed: float | None = None
    total_reviews: int = 0
    lapses: int = 0                              # quality < 3 events

    @property
    def is_due(self) -> bool:
        """True if this card is due for review."""
        return time.time() >= self.next_review

    @property
    def overdue_days(self) -> float:
        """How many days overdue this card is (negative if not yet due)."""
        return (time.time() - self.next_review) / 86400.0


# ---------------------------------------------------------------------------
# SpacedRepetitionSystem
# ---------------------------------------------------------------------------

class SpacedRepetitionSystem:
    """SM-2 spaced repetition engine for EMMS memory items.

    Parameters
    ----------
    memory:
        The ``HierarchicalMemory`` to look up memory items in.
    """

    def __init__(self, memory: "HierarchicalMemory") -> None:
        self.memory = memory
        self._cards: dict[str, SRSCard] = {}   # memory_id → SRSCard

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, memory_id: str, start_due_now: bool = True) -> SRSCard | None:
        """Enrol a memory item in the SRS.

        Args:
            memory_id: The ``MemoryItem.id`` to enrol.
            start_due_now: If True, set next_review to now so the first review
                happens immediately.

        Returns:
            The new ``SRSCard``, or ``None`` if the memory_id isn't found.
        """
        item = self._find_item(memory_id)
        if item is None:
            logger.warning("SRS: memory_id %r not found", memory_id)
            return None

        if memory_id in self._cards:
            logger.debug("SRS: %r already enrolled", memory_id)
            return self._cards[memory_id]

        card = SRSCard(
            memory_id=memory_id,
            next_review=time.time() if start_due_now else time.time() + 86400.0,
        )
        self._cards[memory_id] = card

        # Mark on MemoryItem too (if fields exist)
        _set_srs_fields(item, enrolled=True, next_review=card.next_review, interval=card.interval_days)
        logger.info("SRS: enrolled %r", memory_id)
        return card

    def enroll_all(self) -> int:
        """Enrol all non-superseded, non-expired memory items.

        Returns:
            Number of items newly enrolled.
        """
        count = 0
        for _, store in self.memory._iter_tiers():
            for item in store:
                if item.is_expired or item.is_superseded:
                    continue
                if item.id not in self._cards:
                    self.enroll(item.id, start_due_now=False)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Review recording
    # ------------------------------------------------------------------

    def record_review(self, memory_id: str, quality: int) -> SRSCard | None:
        """Record a review outcome and update the SM-2 schedule.

        Args:
            memory_id: The memory item reviewed.
            quality: Recall quality 0–5:
                5 — perfect response
                4 — correct with slight hesitation
                3 — correct with difficulty
                2 — incorrect; the correct answer seems easy after recall
                1 — incorrect; the correct answer feels difficult
                0 — complete blackout

        Returns:
            Updated ``SRSCard``, or None if not enrolled.
        """
        card = self._cards.get(memory_id)
        if card is None:
            # Auto-enrol on first review
            card = self.enroll(memory_id)
            if card is None:
                return None

        quality = max(0, min(5, quality))
        card.total_reviews += 1
        card.last_reviewed = time.time()

        if quality >= 3:
            # Successful recall — advance schedule
            if card.repetitions == 0:
                card.interval_days = 1.0
            elif card.repetitions == 1:
                card.interval_days = 6.0
            else:
                card.interval_days = round(card.interval_days * card.easiness_factor, 2)
            card.repetitions += 1
        else:
            # Lapse — reset to beginning
            card.lapses += 1
            card.repetitions = 0
            card.interval_days = 1.0

        # Update easiness factor (clamped to MIN_EF)
        card.easiness_factor = max(
            _MIN_EF,
            card.easiness_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02),
        )
        card.next_review = time.time() + card.interval_days * 86400.0

        # Sync to MemoryItem
        item = self._find_item(memory_id)
        if item is not None:
            _set_srs_fields(item, enrolled=True, next_review=card.next_review, interval=card.interval_days)
            # Boost memory_strength on good recall, decay on lapse
            if quality >= 3:
                item.memory_strength = min(1.0, item.memory_strength + 0.05)
            else:
                item.memory_strength = max(0.0, item.memory_strength - 0.1)
            item.touch()

        logger.debug(
            "SRS: reviewed %r q=%d  next_review_in=%.1fd  EF=%.2f",
            memory_id, quality, card.interval_days, card.easiness_factor,
        )
        return card

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_due_items(self, max_items: int = 50) -> list[SRSCard]:
        """Return cards due for review, sorted most-overdue first.

        Args:
            max_items: Maximum number of cards to return.
        """
        due = [card for card in self._cards.values() if card.is_due]
        due.sort(key=lambda c: c.overdue_days, reverse=True)
        return due[:max_items]

    def get_card(self, memory_id: str) -> SRSCard | None:
        """Return the SRS card for a given memory ID."""
        return self._cards.get(memory_id)

    @property
    def stats(self) -> dict[str, Any]:
        """Return SRS statistics."""
        total = len(self._cards)
        due_count = sum(1 for c in self._cards.values() if c.is_due)
        if total:
            avg_ef = sum(c.easiness_factor for c in self._cards.values()) / total
            avg_interval = sum(c.interval_days for c in self._cards.values()) / total
        else:
            avg_ef = _INITIAL_EF
            avg_interval = 0.0
        return {
            "enrolled": total,
            "due_now": due_count,
            "avg_easiness_factor": round(avg_ef, 3),
            "avg_interval_days": round(avg_interval, 1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        """Persist all SRS cards to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": "0.6.0",
            "saved_at": time.time(),
            "cards": {mid: card.model_dump() for mid, card in self._cards.items()},
        }
        path.write_text(json.dumps(state, default=str), encoding="utf-8")
        logger.info("SRS state saved to %s (%d cards)", path, len(self._cards))

    def load_state(self, path: Path | str) -> None:
        """Restore SRS cards from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("No SRS state file at %s", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._cards = {
            mid: SRSCard(**card_data)
            for mid, card_data in data.get("cards", {}).items()
        }
        logger.info("SRS state loaded from %s (%d cards)", path, len(self._cards))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _find_item(self, memory_id: str):
        """Locate a MemoryItem by ID across all tiers."""
        for _, store in self.memory._iter_tiers():
            for item in store:
                if item.id == memory_id:
                    return item
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_srs_fields(item: Any, *, enrolled: bool, next_review: float, interval: float) -> None:
    """Set SRS fields on a MemoryItem if they exist (duck-typed)."""
    if hasattr(item, "srs_enrolled"):
        item.srs_enrolled = enrolled
    if hasattr(item, "srs_next_review"):
        item.srs_next_review = next_review
    if hasattr(item, "srs_interval_days"):
        item.srs_interval_days = interval
