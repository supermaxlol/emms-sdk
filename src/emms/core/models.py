"""Core data models for EMMS.

All data flows through these types. Keep them lean — no behaviour, just structure.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"


class Modality(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    EMOTIONAL = "emotional"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MemoryConfig(BaseModel):
    """Tunable knobs for the whole EMMS pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_capacity: int = Field(default=7, description="Miller's law: 7±2")
    short_term_capacity: int = 50
    long_term_capacity: int = 10_000
    context_window: int = 32_000
    eviction_ratio: float = 0.3
    consolidation_threshold: float = 0.7
    decay_half_life_seconds: float = 86_400.0  # 1 day
    relevance_threshold: float = 0.3
    modalities: list[Modality] = Field(
        default_factory=lambda: list(Modality),
    )


# ---------------------------------------------------------------------------
# Experience — the atomic unit entering the system
# ---------------------------------------------------------------------------

class Experience(BaseModel):
    """A single experience to be memorised.

    This replaces the old SensorimotorExperience dataclass.
    Only *content* and *domain* are required — everything else has sane defaults.
    """

    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:12]}")
    content: str
    domain: str = "general"
    timestamp: float = Field(default_factory=time.time)

    # Optional rich features
    embedding: list[float] | None = None
    emotional_valence: float = 0.0  # -1 negative … +1 positive
    emotional_intensity: float = 0.0  # 0 calm … 1 intense
    novelty: float = 0.5
    importance: float = 0.5
    modality_features: dict[Modality, list[float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Graph memory fields (populated by GraphMemory)
    entities: list[str] = Field(default_factory=list)
    relationships: list[dict[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# MemoryItem — an experience after it enters the memory system
# ---------------------------------------------------------------------------

class MemoryItem(BaseModel):
    """Wrapper around an Experience once it lives inside a memory tier."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    experience: Experience
    tier: MemoryTier = MemoryTier.WORKING

    # Lifecycle metadata
    stored_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    memory_strength: float = 1.0
    consolidation_score: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- helpers ----------------------------------------------------------

    def touch(self) -> None:
        """Record an access (retrieval)."""
        self.last_accessed = time.time()
        self.access_count += 1

    def decay(self, half_life: float = 86_400.0) -> float:
        """Apply exponential decay and return new strength."""
        elapsed = time.time() - self.last_accessed
        self.memory_strength *= np.exp(-0.693 * elapsed / half_life)
        return self.memory_strength

    @property
    def age(self) -> float:
        return time.time() - self.stored_at


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """A single result from a memory query."""

    memory: MemoryItem
    score: float
    source_tier: MemoryTier
    strategy: str = "default"
