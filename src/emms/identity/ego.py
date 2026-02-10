"""Persistent identity / digital ego.

Maintains a coherent self-model across sessions.  Reframed from
"consciousness" to "persistent agent identity" — same core algorithms,
grounded terminology.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from emms.core.models import Experience

logger = logging.getLogger(__name__)


class IdentityState(BaseModel):
    """Serialisable snapshot of the agent's identity."""

    ego_id: str = Field(default_factory=lambda: f"ego_{uuid.uuid4().hex[:8]}")
    name: str = "EMMS Agent"
    personality: str = "Analytical, thorough, memory-focused"
    core_beliefs: list[str] = Field(default_factory=lambda: [
        "Memory is the foundation of coherent behaviour",
        "I learn and grow from every interaction",
    ])
    identity_coherence: float = 0.9
    narrative: str = ""
    autobiographical: list[dict[str, Any]] = Field(default_factory=list)
    session_count: int = 0
    total_experiences: int = 0
    domains_seen: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class PersistentIdentity:
    """Manages a persistent agent identity across sessions.

    Stores identity to a JSON file so it survives restarts.
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._path = Path(storage_path) if storage_path else None
        self.state = self._load()

    # ------------------------------------------------------------------
    # Integrate an experience into the identity narrative
    # ------------------------------------------------------------------

    def integrate(self, experience: Experience) -> dict[str, Any]:
        """Update the identity model with a new experience."""
        self.state.total_experiences += 1
        self.state.updated_at = time.time()

        if experience.domain not in self.state.domains_seen:
            self.state.domains_seen.append(experience.domain)

        # Autobiographical entry (keep last 100)
        entry = {
            "experience_id": experience.id,
            "domain": experience.domain,
            "summary": experience.content[:120],
            "timestamp": experience.timestamp,
            "importance": experience.importance,
        }
        self.state.autobiographical.append(entry)
        if len(self.state.autobiographical) > 100:
            self.state.autobiographical = self.state.autobiographical[-100:]

        # Update narrative
        self.state.narrative = self._build_narrative()

        return {
            "identity_coherence": self.state.identity_coherence,
            "total_experiences": self.state.total_experiences,
            "domains": self.state.domains_seen,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.state.model_dump_json(indent=2))

    def _load(self) -> IdentityState:
        if self._path and self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                state = IdentityState(**data)
                state.session_count += 1
                return state
            except Exception as e:
                logger.warning("Failed to load identity: %s — starting fresh", e)
        return IdentityState()

    # ------------------------------------------------------------------
    # Narrative builder
    # ------------------------------------------------------------------

    def _build_narrative(self) -> str:
        s = self.state
        recent = s.autobiographical[-3:] if s.autobiographical else []
        recent_summaries = "; ".join(e["summary"] for e in recent)
        return (
            f"I am {s.name}. I have processed {s.total_experiences} experiences "
            f"across {len(s.domains_seen)} domains over {s.session_count} sessions. "
            f"Recent activity: {recent_summaries}"
        )
