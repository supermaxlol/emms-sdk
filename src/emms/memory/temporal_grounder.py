"""Gap 5 — TemporalGrounder: real-time temporal awareness.

Anchors EMMS to wall-clock time with:
- **TemporalAnchor**: named reference points (events, deadlines, recurrences)
- **Deadline tracking**: time remaining, urgency scoring
- **Recurrences**: repeating events (daily standup, weekly review, etc.)
- **Elapsed-time awareness**: how long since last interaction, session duration
- **Temporal context generation**: text snippets that orient the agent in time
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AnchorType(str, Enum):
    EVENT = "event"           # past or future event
    DEADLINE = "deadline"     # must be done by this time
    RECURRENCE = "recurrence" # repeating event


@dataclass
class TemporalAnchor:
    """A named point or interval on the timeline."""
    name: str
    anchor_type: AnchorType
    timestamp: float  # Unix epoch — when this anchor fires
    domain: str = "general"
    description: str = ""
    # Recurrence: interval in seconds (0 = one-shot)
    recurrence_interval: float = 0.0
    # Track completions for recurrences
    last_fired: float = 0.0
    active: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"ta_{self.name}_{int(self.timestamp)}"

    def time_until(self) -> float:
        """Seconds until this anchor fires. Negative = past."""
        return self.timestamp - time.time()

    def urgency(self) -> float:
        """0..1 urgency score. 1.0 = past due or imminent."""
        remaining = self.time_until()
        if remaining <= 0:
            return 1.0
        if remaining > 86400 * 7:  # > 7 days
            return 0.1
        if remaining > 86400:  # > 1 day
            return 0.3
        if remaining > 3600:  # > 1 hour
            return 0.6
        if remaining > 300:  # > 5 minutes
            return 0.8
        return 0.95

    def is_due(self) -> bool:
        """True if this anchor has passed or is imminent (< 60s)."""
        return self.time_until() <= 60

    def advance_recurrence(self) -> bool:
        """If recurrent, advance to next firing. Returns True if advanced."""
        if self.recurrence_interval <= 0:
            self.active = False
            return False
        self.last_fired = time.time()
        while self.timestamp <= time.time():
            self.timestamp += self.recurrence_interval
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "anchor_type": self.anchor_type.value,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "description": self.description,
            "recurrence_interval": self.recurrence_interval,
            "last_fired": self.last_fired,
            "active": self.active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemporalAnchor:
        return cls(
            name=d["name"],
            anchor_type=AnchorType(d["anchor_type"]),
            timestamp=d["timestamp"],
            domain=d.get("domain", "general"),
            description=d.get("description", ""),
            recurrence_interval=d.get("recurrence_interval", 0.0),
            last_fired=d.get("last_fired", 0.0),
            active=d.get("active", True),
            metadata=d.get("metadata", {}),
        )


class TemporalGrounder:
    """Keeps EMMS anchored in real time.

    Maintains a set of temporal anchors (deadlines, events, recurrences)
    and generates temporal context strings that can be injected into
    prompts or reasoning chains.
    """

    def __init__(self, *, max_anchors: int = 200) -> None:
        self._anchors: dict[str, TemporalAnchor] = {}
        self._max_anchors = max_anchors
        self._session_start: float = time.time()
        self._last_interaction: float = time.time()
        self._total_interactions: int = 0

    # -- anchor management --------------------------------------------------

    def add_anchor(self, name: str, timestamp: float,
                   anchor_type: AnchorType = AnchorType.EVENT,
                   domain: str = "general",
                   description: str = "",
                   recurrence_interval: float = 0.0) -> TemporalAnchor:
        """Create and register a temporal anchor."""
        anchor = TemporalAnchor(
            name=name,
            anchor_type=anchor_type,
            timestamp=timestamp,
            domain=domain,
            description=description,
            recurrence_interval=recurrence_interval,
        )
        self._anchors[anchor.id] = anchor
        self._enforce_capacity()
        return anchor

    def add_deadline(self, name: str, deadline: float,
                     domain: str = "general",
                     description: str = "") -> TemporalAnchor:
        """Convenience: add a deadline anchor."""
        return self.add_anchor(
            name=name,
            timestamp=deadline,
            anchor_type=AnchorType.DEADLINE,
            domain=domain,
            description=description,
        )

    def add_recurrence(self, name: str, first_fire: float,
                       interval_seconds: float,
                       domain: str = "general",
                       description: str = "") -> TemporalAnchor:
        """Convenience: add a recurring event."""
        return self.add_anchor(
            name=name,
            timestamp=first_fire,
            anchor_type=AnchorType.RECURRENCE,
            domain=domain,
            description=description,
            recurrence_interval=interval_seconds,
        )

    def remove_anchor(self, anchor_id: str) -> bool:
        """Remove an anchor by ID. Returns True if found."""
        return self._anchors.pop(anchor_id, None) is not None

    def get_anchor(self, name: str) -> TemporalAnchor | None:
        """Find an anchor by name (linear scan)."""
        for a in self._anchors.values():
            if a.name == name:
                return a
        return None

    # -- queries ------------------------------------------------------------

    def due_anchors(self) -> list[TemporalAnchor]:
        """Return all anchors that are past due, sorted by urgency desc."""
        due = [a for a in self._anchors.values() if a.active and a.is_due()]
        return sorted(due, key=lambda a: a.urgency(), reverse=True)

    def upcoming(self, horizon_seconds: float = 86400,
                 domain: str | None = None) -> list[TemporalAnchor]:
        """Return anchors firing within the horizon, sorted chronologically."""
        now = time.time()
        cutoff = now + horizon_seconds
        out = []
        for a in self._anchors.values():
            if not a.active:
                continue
            if domain and a.domain != domain:
                continue
            if a.timestamp <= cutoff:
                out.append(a)
        return sorted(out, key=lambda a: a.timestamp)

    def deadlines(self, domain: str | None = None) -> list[TemporalAnchor]:
        """Return active deadlines sorted by urgency (most urgent first)."""
        out = []
        for a in self._anchors.values():
            if a.anchor_type != AnchorType.DEADLINE or not a.active:
                continue
            if domain and a.domain != domain:
                continue
            out.append(a)
        return sorted(out, key=lambda a: a.urgency(), reverse=True)

    # -- tick (advance recurrences, collect due) ----------------------------

    def tick(self) -> list[TemporalAnchor]:
        """Advance time: fire due anchors, advance recurrences.

        Returns list of anchors that just fired.
        """
        fired = []
        for a in list(self._anchors.values()):
            if not a.active:
                continue
            if a.is_due():
                fired.append(a)
                if a.recurrence_interval > 0:
                    a.advance_recurrence()
                else:
                    a.active = False
        return fired

    # -- temporal context generation ----------------------------------------

    def touch(self) -> None:
        """Mark an interaction (for elapsed-time awareness)."""
        self._last_interaction = time.time()
        self._total_interactions += 1

    def elapsed_since_last(self) -> float:
        """Seconds since last interaction."""
        return time.time() - self._last_interaction

    def session_duration(self) -> float:
        """Seconds since session start."""
        return time.time() - self._session_start

    def generate_context(self, domain: str | None = None) -> str:
        """Generate a temporal context string for prompt injection.

        Includes: current time, session duration, upcoming deadlines,
        time since last interaction.
        """
        import datetime as _dt
        now = _dt.datetime.now()
        parts = [
            f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})",
            f"Session duration: {self._format_duration(self.session_duration())}",
        ]

        elapsed = self.elapsed_since_last()
        if elapsed > 60:
            parts.append(f"Time since last interaction: {self._format_duration(elapsed)}")

        # Upcoming deadlines
        deadlines = self.deadlines(domain)[:3]
        if deadlines:
            parts.append("Upcoming deadlines:")
            for d in deadlines:
                remaining = d.time_until()
                urgency = d.urgency()
                marker = "!!!" if urgency > 0.8 else ("!" if urgency > 0.5 else "")
                parts.append(
                    f"  {marker}{d.name}: {self._format_duration(abs(remaining))} "
                    f"{'overdue' if remaining < 0 else 'remaining'}"
                )

        # Due recurrences
        due = [a for a in self.due_anchors() if a.anchor_type == AnchorType.RECURRENCE]
        if due[:2]:
            parts.append("Due now:")
            for a in due[:2]:
                parts.append(f"  - {a.name}: {a.description or 'no description'}")

        return "\n".join(parts)

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "session_start": self._session_start,
            "last_interaction": self._last_interaction,
            "total_interactions": self._total_interactions,
            "anchors": [a.to_dict() for a in self._anchors.values()],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._session_start = data.get("session_start", time.time())
            self._last_interaction = data.get("last_interaction", time.time())
            self._total_interactions = data.get("total_interactions", 0)
            self._anchors.clear()
            for ad in data.get("anchors", []):
                a = TemporalAnchor.from_dict(ad)
                self._anchors[a.id] = a
            return True
        except Exception:
            return False

    def summary(self) -> str:
        active = sum(1 for a in self._anchors.values() if a.active)
        deadlines_count = sum(1 for a in self._anchors.values()
                              if a.active and a.anchor_type == AnchorType.DEADLINE)
        recurrences_count = sum(1 for a in self._anchors.values()
                                if a.active and a.anchor_type == AnchorType.RECURRENCE)
        return (
            f"TemporalGrounder: {active} active anchors "
            f"({deadlines_count} deadlines, {recurrences_count} recurrences), "
            f"{self._total_interactions} interactions, "
            f"session {self._format_duration(self.session_duration())}"
        )

    # -- internal -----------------------------------------------------------

    def _enforce_capacity(self) -> None:
        """Remove oldest inactive anchors if over capacity."""
        if len(self._anchors) <= self._max_anchors:
            return
        # Remove inactive first
        inactive = [aid for aid, a in self._anchors.items() if not a.active]
        for aid in inactive:
            del self._anchors[aid]
            if len(self._anchors) <= self._max_anchors:
                return
        # If still over, remove oldest events (not deadlines or recurrences)
        events = sorted(
            [(aid, a) for aid, a in self._anchors.items()
             if a.anchor_type == AnchorType.EVENT],
            key=lambda x: x[1].timestamp,
        )
        for aid, _ in events:
            del self._anchors[aid]
            if len(self._anchors) <= self._max_anchors:
                return

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.0f}m"
        if seconds < 86400:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m}m"
        d = int(seconds // 86400)
        h = int((seconds % 86400) // 3600)
        return f"{d}d{h}h"
