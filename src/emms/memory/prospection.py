"""ProspectiveMemory — future-oriented intention storage with context triggering.

v0.13.0: The Metacognitive Layer

Prospective memory is memory for the future: "remember to do X when Y happens."
Unlike retrospective memory (recalling the past), prospective memory holds
intentions and activates them when the right context arrives.

Without this module the agent's sessions are entirely reactive — it responds
to what's asked but has no mechanism for carrying forward plans, follow-up
commitments, or self-directed goals. With ProspectiveMemory, the agent can
intend things between sessions and execute those intentions when the moment
arises.

Biological analogue: event-based prospective memory (brush teeth *when* you
see the toothbrush) and time-based prospective memory (call doctor *at 3pm*).
Mediated by rostral prefrontal cortex — the same region implicated in self-
reflective thought and planning.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Intention:
    """A single future-oriented intention with a trigger condition."""

    id: str
    content: str             # what the agent plans to do
    trigger_context: str     # textual description of when to trigger
    priority: float          # 0..1 (higher = more urgent)
    created_at: float
    fulfilled: bool = False
    fulfilled_at: Optional[float] = None
    activation_count: int = 0    # times this intention was triggered

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "trigger_context": self.trigger_context,
            "priority": self.priority,
            "created_at": self.created_at,
            "fulfilled": self.fulfilled,
            "fulfilled_at": self.fulfilled_at,
            "activation_count": self.activation_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intention":
        return cls(
            id=d["id"],
            content=d["content"],
            trigger_context=d["trigger_context"],
            priority=float(d.get("priority", 0.5)),
            created_at=float(d.get("created_at", time.time())),
            fulfilled=bool(d.get("fulfilled", False)),
            fulfilled_at=d.get("fulfilled_at"),
            activation_count=int(d.get("activation_count", 0)),
        )


@dataclass
class IntentionActivation:
    """An intention that has been activated by the current context."""

    intention: Intention
    activation_score: float  # how well current context matched trigger
    trigger_overlap: float   # Jaccard token overlap
    days_pending: float      # days since the intention was created


# ---------------------------------------------------------------------------
# ProspectiveMemory
# ---------------------------------------------------------------------------

class ProspectiveMemory:
    """Stores and manages future-oriented intentions.

    Intentions are stored in-memory and can be persisted to JSON via
    :meth:`save` / :meth:`load`. Each intention has a ``trigger_context``
    description; when :meth:`check` is called with the current context text,
    intentions whose trigger matches above ``overlap_threshold`` are returned
    as :class:`IntentionActivation` objects.

    Parameters
    ----------
    overlap_threshold:
        Minimum Jaccard token overlap for an intention to activate
        (default 0.15).
    max_intentions:
        Maximum number of stored intentions (oldest are evicted, default 50).
    """

    def __init__(
        self,
        overlap_threshold: float = 0.15,
        max_intentions: int = 50,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.max_intentions = max_intentions
        self._intentions: dict[str, Intention] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intend(
        self,
        content: str,
        trigger_context: str,
        priority: float = 0.5,
    ) -> Intention:
        """Store a new intention.

        If the store is full (``max_intentions`` reached) the lowest-priority
        unfulfilled intention is evicted first.

        Args:
            content:         What the agent intends to do.
            trigger_context: Textual description of when to trigger.
            priority:        Urgency 0–1 (default 0.5).

        Returns:
            The created :class:`Intention`.
        """
        # Evict if at capacity
        unfulfilled = [i for i in self._intentions.values() if not i.fulfilled]
        if len(unfulfilled) >= self.max_intentions:
            lowest = min(unfulfilled, key=lambda i: i.priority)
            del self._intentions[lowest.id]

        intention = Intention(
            id=f"int_{uuid.uuid4().hex[:10]}",
            content=content,
            trigger_context=trigger_context,
            priority=max(0.0, min(1.0, priority)),
            created_at=time.time(),
        )
        self._intentions[intention.id] = intention
        return intention

    def check(self, current_context: str) -> list[IntentionActivation]:
        """Check which unfulfilled intentions are activated by the current context.

        Args:
            current_context: Text representing the current conversational context.

        Returns:
            List of :class:`IntentionActivation` sorted by activation_score
            descending.
        """
        activations: list[IntentionActivation] = []
        now = time.time()

        for intention in self._intentions.values():
            if intention.fulfilled:
                continue
            overlap = self._token_overlap(
                current_context, intention.trigger_context
            )
            if overlap < self.overlap_threshold:
                continue

            # Score = overlap × priority — higher priority intentions surface first
            activation_score = overlap * (0.5 + 0.5 * intention.priority)
            days_pending = (now - intention.created_at) / 86400.0

            intention.activation_count += 1
            activations.append(
                IntentionActivation(
                    intention=intention,
                    activation_score=activation_score,
                    trigger_overlap=overlap,
                    days_pending=days_pending,
                )
            )

        activations.sort(key=lambda a: a.activation_score, reverse=True)
        return activations

    def fulfill(self, intention_id: str) -> bool:
        """Mark an intention as fulfilled.

        Args:
            intention_id: The ``id`` of the intention to fulfill.

        Returns:
            ``True`` if the intention was found and marked, ``False`` otherwise.
        """
        intention = self._intentions.get(intention_id)
        if intention is None:
            return False
        intention.fulfilled = True
        intention.fulfilled_at = time.time()
        return True

    def dismiss(self, intention_id: str) -> bool:
        """Remove an intention without fulfilling it.

        Args:
            intention_id: The ``id`` of the intention to remove.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        if intention_id in self._intentions:
            del self._intentions[intention_id]
            return True
        return False

    def pending(self) -> list[Intention]:
        """Return all unfulfilled intentions sorted by priority descending.

        Returns:
            List of unfulfilled :class:`Intention` objects.
        """
        return sorted(
            (i for i in self._intentions.values() if not i.fulfilled),
            key=lambda i: i.priority,
            reverse=True,
        )

    def all_intentions(self) -> list[Intention]:
        """Return all intentions (fulfilled and unfulfilled).

        Returns:
            List of :class:`Intention` sorted by created_at descending.
        """
        return sorted(
            self._intentions.values(),
            key=lambda i: i.created_at,
            reverse=True,
        )

    def save(self, path: str | Path) -> None:
        """Persist all intentions to a JSON file (atomic write).

        Args:
            path: File path to write to.
        """
        import os
        import tempfile

        path = Path(path)
        data = [i.to_dict() for i in self._intentions.values()]
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, path: str | Path) -> bool:
        """Load intentions from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            ``True`` if loaded successfully, ``False`` if file not found.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self._intentions = {
                d["id"]: Intention.from_dict(d) for d in data
            }
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Jaccard token overlap between two strings."""
        toks_a = set(text_a.lower().split())
        toks_b = set(text_b.lower().split())
        if not toks_a or not toks_b:
            return 0.0
        intersection = toks_a & toks_b
        union = toks_a | toks_b
        return len(intersection) / max(len(union), 1)
