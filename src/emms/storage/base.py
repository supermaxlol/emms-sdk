"""Storage backends for persisting memory state.

Provides a simple protocol and two implementations:
  - InMemoryStore  (for tests / ephemeral use)
  - JSONFileStore  (for persistence across sessions)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    """Minimal protocol for memory persistence."""

    def save(self, key: str, data: Any) -> None: ...
    def load(self, key: str) -> Any | None: ...
    def keys(self) -> list[str]: ...


class InMemoryStore:
    """Ephemeral in-memory store (good for tests)."""

    def __init__(self):
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> None:
        self._data[key] = data

    def load(self, key: str) -> Any | None:
        return self._data.get(key)

    def keys(self) -> list[str]:
        return list(self._data.keys())


class JSONFileStore:
    """Persist each key as a separate JSON file under a directory."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        path = self._dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def load(self, key: str) -> Any | None:
        path = self._dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning("Failed to load %s: %s", key, e)
        return None

    def keys(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]
