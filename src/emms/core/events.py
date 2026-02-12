"""Lightweight pub/sub event bus for EMMS inter-component communication.

Components can emit and subscribe to named events so the system
reacts cohesively without tight coupling.

Standard events:
    memory.stored        — Experience stored in hierarchical memory
    memory.consolidated  — Consolidation pass completed
    memory.compressed    — Long-term memories compressed
    episode.detected     — New episode boundary found
    identity.updated     — Identity state changed
    graph.updated        — Graph memory entity/relationship added
    threshold.reached    — Capacity or scoring threshold hit
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any]], None]


class EventBus:
    """Simple synchronous pub/sub event bus.

    Usage::

        bus = EventBus()
        bus.on("memory.stored", lambda data: print(data))
        bus.emit("memory.stored", {"id": "exp_abc123"})
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callback]] = defaultdict(list)
        self._history: list[tuple[str, dict[str, Any]]] = []
        self._max_history: int = 100

    # ── Subscribe ──────────────────────────────────────────────────────

    def on(self, event: str, callback: Callback) -> None:
        """Register a callback for *event*."""
        self._listeners[event].append(callback)

    def off(self, event: str, callback: Callback) -> None:
        """Remove a specific callback for *event*."""
        try:
            self._listeners[event].remove(callback)
        except ValueError:
            pass

    def once(self, event: str, callback: Callback) -> None:
        """Register a callback that fires only once."""
        def _wrapper(data: dict[str, Any]) -> None:
            callback(data)
            self.off(event, _wrapper)
        self.on(event, _wrapper)

    # ── Emit ───────────────────────────────────────────────────────────

    def emit(self, event: str, data: dict[str, Any] | None = None) -> int:
        """Emit *event* with optional *data*. Returns listener count."""
        data = data or {}
        listeners = self._listeners.get(event, [])

        # Record in history (bounded)
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append((event, data))

        for cb in list(listeners):  # copy to allow mutation during iteration
            try:
                cb(data)
            except Exception:
                logger.exception("Error in event listener for %r", event)

        return len(listeners)

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def history(self) -> list[tuple[str, dict[str, Any]]]:
        """Recent event history (last *max_history* events)."""
        return list(self._history)

    @property
    def listener_counts(self) -> dict[str, int]:
        """Number of listeners per event."""
        return {k: len(v) for k, v in self._listeners.items() if v}

    def clear(self) -> None:
        """Remove all listeners and history."""
        self._listeners.clear()
        self._history.clear()
