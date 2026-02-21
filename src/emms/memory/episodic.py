"""EpisodicBuffer — structured session-as-episode storage.

v0.14.0: The Temporal Mind

Individual memories are atomic facts. Episodes are bounded experiences — a
conversation, a problem-solving session, a creative burst — with a beginning,
middle, and end. The EpisodicBuffer stores these episodes as first-class
objects, giving the agent a structured timeline of its own history.

Without episodic memory the agent lives in an eternal present: every recalled
fact is equally accessible and timeless. With it, the agent can say "in my
session about consciousness I had a turning point when..." — experience
becomes narrative.

Biological analogue: hippocampal episodic memory — the "mental time travel"
system that allows humans to re-experience specific events in their context
(Tulving 1972). Damage produces anterograde amnesia: facts can still be
learned but episodes cannot be formed.
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
class Episode:
    """A bounded, temporally structured experience."""

    id: str
    session_id: str
    topic: str                      # what this episode was about
    opened_at: float                # Unix timestamp
    closed_at: Optional[float]      # None if still open
    emotional_arc: list[float]      # valence per recorded turn
    key_memory_ids: list[str]       # important memories from this episode
    turn_count: int                 # number of turns recorded
    outcome: str                    # brief resolution description
    peak_valence: float             # highest |valence| reached
    mean_valence: float             # average valence

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration in seconds, or None if still open."""
        if self.closed_at is None:
            return None
        return self.closed_at - self.opened_at

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def summary(self) -> str:
        status = "OPEN" if self.is_open else f"closed ({self.duration_seconds:.0f}s)"
        arc_str = ""
        if self.emotional_arc:
            arc_vals = " → ".join(f"{v:+.2f}" for v in self.emotional_arc[-5:])
            arc_str = f"  arc: {arc_vals}"
        return (
            f"Episode {self.id[:12]}  [{self.topic[:40]}]  "
            f"turns={self.turn_count}  status={status}"
            f"{arc_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "topic": self.topic,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "emotional_arc": self.emotional_arc,
            "key_memory_ids": self.key_memory_ids,
            "turn_count": self.turn_count,
            "outcome": self.outcome,
            "peak_valence": self.peak_valence,
            "mean_valence": self.mean_valence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        return cls(
            id=d["id"],
            session_id=d.get("session_id", ""),
            topic=d.get("topic", ""),
            opened_at=float(d.get("opened_at", time.time())),
            closed_at=d.get("closed_at"),
            emotional_arc=list(d.get("emotional_arc", [])),
            key_memory_ids=list(d.get("key_memory_ids", [])),
            turn_count=int(d.get("turn_count", 0)),
            outcome=d.get("outcome", ""),
            peak_valence=float(d.get("peak_valence", 0.0)),
            mean_valence=float(d.get("mean_valence", 0.0)),
        )


# ---------------------------------------------------------------------------
# EpisodicBuffer
# ---------------------------------------------------------------------------

class EpisodicBuffer:
    """Stores sessions as structured, bounded episodes.

    Episodes are separate from the hierarchical memory — they are meta-records
    about *experiences*, not the experiences themselves. Each episode tracks:

    - Topic / theme
    - Temporal boundaries (opened_at, closed_at)
    - Emotional arc (valence per turn)
    - Key memory IDs referenced during the episode
    - Turn count and outcome description

    Parameters
    ----------
    max_episodes:
        Maximum number of episodes to keep (oldest are evicted, default 100).
    """

    def __init__(self, max_episodes: int = 100) -> None:
        self.max_episodes = max_episodes
        self._episodes: dict[str, Episode] = {}
        self._current_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_episode(
        self,
        session_id: Optional[str] = None,
        topic: str = "",
    ) -> Episode:
        """Open a new episode, closing any currently open one first.

        Args:
            session_id: Optional session label (auto-generated if omitted).
            topic:      What this episode is about.

        Returns:
            The newly created :class:`Episode`.
        """
        # Auto-close any open episode
        if self._current_id is not None:
            self.close_episode()

        if session_id is None:
            session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"

        episode = Episode(
            id=f"ep_{uuid.uuid4().hex[:10]}",
            session_id=session_id,
            topic=topic,
            opened_at=time.time(),
            closed_at=None,
            emotional_arc=[],
            key_memory_ids=[],
            turn_count=0,
            outcome="",
            peak_valence=0.0,
            mean_valence=0.0,
        )

        # Evict oldest if at capacity
        if len(self._episodes) >= self.max_episodes:
            oldest_id = min(
                self._episodes, key=lambda eid: self._episodes[eid].opened_at
            )
            del self._episodes[oldest_id]

        self._episodes[episode.id] = episode
        self._current_id = episode.id
        return episode

    def close_episode(
        self,
        episode_id: Optional[str] = None,
        outcome: str = "",
    ) -> Optional[Episode]:
        """Close an episode, computing final statistics.

        Args:
            episode_id: ID of the episode to close.  Defaults to the currently
                open episode.
            outcome:    Brief description of how the episode resolved.

        Returns:
            The closed :class:`Episode`, or ``None`` if not found.
        """
        eid = episode_id or self._current_id
        if eid is None:
            return None
        episode = self._episodes.get(eid)
        if episode is None:
            return None

        episode.closed_at = time.time()
        episode.outcome = outcome

        arc = episode.emotional_arc
        if arc:
            episode.mean_valence = sum(arc) / len(arc)
            episode.peak_valence = max(arc, key=abs)
        else:
            episode.mean_valence = 0.0
            episode.peak_valence = 0.0

        if self._current_id == eid:
            self._current_id = None

        return episode

    def record_turn(
        self,
        episode_id: Optional[str] = None,
        content: str = "",
        valence: float = 0.0,
    ) -> None:
        """Record a turn within an episode.

        Args:
            episode_id: Episode to record to.  Defaults to current.
            content:    Turn text (used only to update turn_count here).
            valence:    Emotional valence of this turn.
        """
        eid = episode_id or self._current_id
        if eid is None:
            return
        episode = self._episodes.get(eid)
        if episode is None or not episode.is_open:
            return
        episode.turn_count += 1
        episode.emotional_arc.append(max(-1.0, min(1.0, valence)))

    def add_memory(
        self,
        memory_id: str,
        episode_id: Optional[str] = None,
    ) -> None:
        """Associate a memory ID with an episode.

        Args:
            memory_id:  Memory ID to add.
            episode_id: Episode to add to.  Defaults to current.
        """
        eid = episode_id or self._current_id
        if eid is None:
            return
        episode = self._episodes.get(eid)
        if episode is None:
            return
        if memory_id not in episode.key_memory_ids:
            episode.key_memory_ids.append(memory_id)

    def current_episode(self) -> Optional[Episode]:
        """Return the currently open episode, or ``None``."""
        if self._current_id is None:
            return None
        return self._episodes.get(self._current_id)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Return an episode by ID, or ``None`` if not found."""
        return self._episodes.get(episode_id)

    def recent_episodes(self, n: int = 10) -> list[Episode]:
        """Return the *n* most recent episodes (by opened_at) descending.

        Args:
            n: Number of episodes to return (default 10).

        Returns:
            List of :class:`Episode` sorted newest first.
        """
        all_ep = sorted(
            self._episodes.values(),
            key=lambda e: e.opened_at,
            reverse=True,
        )
        return all_ep[:n]

    def all_episodes(self) -> list[Episode]:
        """Return all episodes sorted oldest first."""
        return sorted(self._episodes.values(), key=lambda e: e.opened_at)

    def save(self, path: str | Path) -> None:
        """Persist episodes to a JSON file."""
        data = [e.to_dict() for e in self.all_episodes()]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> bool:
        """Load episodes from a JSON file.

        Returns:
            ``True`` if loaded successfully, ``False`` if file not found.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self._episodes = {d["id"]: Episode.from_dict(d) for d in data}
            # Restore current episode pointer if any is still open
            open_eps = [e for e in self._episodes.values() if e.is_open]
            self._current_id = open_eps[-1].id if open_eps else None
            return True
        except Exception:
            return False
