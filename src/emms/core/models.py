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


class ObsType(str, Enum):
    """Discrete observation type (claude-mem inspired).

    Categorises what kind of event this memory records so retrieval can
    filter semantically (e.g. "show me all bugfix memories").
    """
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    CHANGE = "change"
    DISCOVERY = "discovery"
    DECISION = "decision"


class ConceptTag(str, Enum):
    """Knowledge-type tag for an observation (claude-mem inspired).

    Tags explain *how* to interpret the memory — the epistemological role
    of what was learned, beyond what domain it belongs to.
    """
    HOW_IT_WORKS = "how-it-works"
    WHY_IT_EXISTS = "why-it-exists"
    WHAT_CHANGED = "what-changed"
    PROBLEM_SOLUTION = "problem-solution"
    GOTCHA = "gotcha"
    PATTERN = "pattern"
    TRADE_OFF = "trade-off"


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
    consolidation_threshold: float = 0.55
    decay_half_life_seconds: float = 86_400.0  # 1 day
    relevance_threshold: float = 0.3
    modalities: list[Modality] = Field(
        default_factory=lambda: list(Modality),
    )

    # v0.6.0: SemanticDeduplicator thresholds
    dedup_cosine_threshold: float = 0.92   # cosine similarity to flag as near-dup
    dedup_lexical_threshold: float = 0.85  # lexical similarity fallback threshold
    enable_auto_dedup: bool = False        # if True, run dedup after each consolidation


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

    # claude-mem inspired: session tracking and privacy
    session_id: str | None = None  # groups memories by conversation session
    private: bool = False          # exclude from retrieval / export when True

    # claude-mem inspired: observation classification
    obs_type: ObsType | None = None                      # what kind of event this is
    concept_tags: list[ConceptTag] = Field(default_factory=list)  # how to interpret it

    # LangMem inspired: update mode and conflict tracking
    update_mode: str = "insert"       # "insert" (append new) | "patch" (update matching)
    patch_key: str | None = None      # when update_mode="patch", match on this key (default: title)

    # GitHub Copilot inspired: citation-based validation
    citations: list[str] = Field(default_factory=list)  # mem_ids this memory cites/validates

    # LangMem inspired: namespace scoping (project / repo / agent isolation)
    namespace: str = "default"     # partition key — only retrieve within same namespace

    # Confidence scoring: how certain are we that this memory is accurate?
    confidence: float = 1.0        # 0 (uncertain) … 1 (fully verified)

    # claude-mem inspired: rich structured content fields
    title: str | None = None           # short title ≤10 words (used in compact index)
    subtitle: str | None = None        # one-sentence explanation ≤24 words
    facts: list[str] = Field(default_factory=list)           # discrete factual bullet points
    files_read: list[str] = Field(default_factory=list)      # files read during this event
    files_modified: list[str] = Field(default_factory=list)  # files created or edited

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

    # Copilot-inspired: TTL with refresh-on-use + conflict archival
    expires_at: float | None = None       # None = no hard expiry; use TTL helper to set
    superseded_by: str | None = None      # mem_id of the newer memory that replaced this one

    # v0.6.0: Spaced Repetition System fields
    srs_enrolled: bool = False            # whether enrolled in SRS review schedule
    srs_next_review: float | None = None  # Unix timestamp of next review (None = not scheduled)
    srs_interval_days: float = 1.0        # current SM-2 interval in days

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- helpers ----------------------------------------------------------

    def touch(self, ttl_seconds: float | None = None) -> None:
        """Record an access (retrieval). Optionally refresh TTL on use (Copilot pattern)."""
        self.last_accessed = time.time()
        self.access_count += 1
        if ttl_seconds is not None and self.expires_at is not None:
            # Refresh expiry — important memories self-renew
            self.expires_at = time.time() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        """True if this memory has passed its hard TTL expiry."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_superseded(self) -> bool:
        """True if a newer memory has replaced this one."""
        return self.superseded_by is not None

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
    strategy_scores: dict[str, float] = Field(default_factory=dict)  # per-strategy breakdown
    explanation: str = ""  # human-readable scoring explanation


class CompactResult(BaseModel):
    """Compact index entry for progressive disclosure retrieval (claude-mem pattern).

    Layer 1 of 3: ~50-80 tokens per result. Use to scan many memories cheaply,
    then call get_full() only for the IDs you actually need.
    """

    id: str
    snippet: str           # title + first fact, or first 120 chars of content
    domain: str
    score: float
    tier: MemoryTier
    session_id: str | None = None
    timestamp: float
    obs_type: ObsType | None = None
    concept_tags: list[ConceptTag] = Field(default_factory=list)
    token_estimate: int | None = None  # ≈ token cost to retrieve full content
    namespace: str = "default"
    confidence: float = 1.0


class SessionSummary(BaseModel):
    """Structured summary of a single session (claude-mem inspired).

    Mirrors claude-mem's session table schema: request/investigated/learned/
    completed/next_steps — giving the memory system a high-level narrative
    of what happened in each session, retrievable without loading all memories.
    """

    session_id: str
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None

    # Structured narrative fields
    request: str = ""          # what the user asked for this session
    investigated: str = ""     # what was explored / looked at
    learned: str = ""          # key insights / discoveries made
    completed: str = ""        # what was actually finished
    next_steps: str = ""       # outstanding items / follow-up work

    # Optional stats
    memory_count: int = 0      # how many memories were stored this session
    obs_types: dict[str, int] = Field(default_factory=dict)  # type → count

    def close(self, ended_at: float | None = None) -> None:
        """Mark session as ended."""
        self.ended_at = ended_at or time.time()

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at
