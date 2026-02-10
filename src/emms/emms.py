"""EMMS — the top-level orchestrator.

Ties together hierarchical memory, cross-modal binding, episode detection,
token context management, persistent identity, and optional vector store
into a single interface.

Usage:
    from emms import EMMS, Experience

    agent = EMMS()
    agent.store(Experience(content="The market rose 2% today", domain="finance"))
    results = agent.retrieve("market trends")

With embeddings + ChromaDB:
    from emms.core.embeddings import HashEmbedder
    from emms.storage.chroma import ChromaStore

    embedder = HashEmbedder(dim=128)
    agent = EMMS(embedder=embedder, vector_store=ChromaStore(embedder=embedder))
    agent.store(Experience(content="The market rose 2%", domain="finance"))
    results = agent.retrieve_semantic("stock market trends")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

from emms.core.embeddings import EmbeddingProvider
from emms.core.models import (
    Experience,
    MemoryConfig,
    MemoryItem,
    MemoryTier,
    Modality,
    RetrievalResult,
)
from emms.context.token_manager import TokenContextManager
from emms.crossmodal.binding import CrossModalMemory
from emms.episodes.boundary import EpisodeBoundaryDetector, Episode
from emms.identity.ego import PersistentIdentity
from emms.memory.hierarchical import HierarchicalMemory

logger = logging.getLogger(__name__)


class EMMS:
    """Enhanced Memory Management System.

    One object = one agent's complete memory.

    Parameters
    ----------
    config : MemoryConfig, optional
    identity_path : path to persist identity JSON
    embedder : EmbeddingProvider, optional
        When supplied, enables embedding-based retrieval in the hierarchical
        memory and the ``retrieve_semantic`` method.
    vector_store : object with add/query, optional
        When supplied (e.g. ChromaStore), experiences are also indexed in
        the vector store for fast ANN search via ``retrieve_semantic``.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        identity_path: str | Path | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: Any | None = None,
    ):
        self.cfg = config or MemoryConfig()
        self.embedder = embedder
        self.vector_store = vector_store

        # Sub-systems
        self.memory = HierarchicalMemory(self.cfg, embedder=embedder)
        self.crossmodal = CrossModalMemory(self.cfg.modalities)
        self.episodes = EpisodeBoundaryDetector()
        self.tokens = TokenContextManager(self.cfg)
        self.identity = PersistentIdentity(identity_path)

        # Pipeline stats
        self._store_count = 0
        self._retrieve_count = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> dict[str, Any]:
        """Full pipeline: hierarchy + cross-modal + vector store + episodes + identity."""
        t0 = time.time()

        # 1. Hierarchical memory (also computes embedding if embedder set)
        mem_item = self.memory.store(experience)

        # 2. Cross-modal indexing
        cm_result = self.crossmodal.store(experience)

        # 3. Vector store (ChromaDB etc.)
        if self.vector_store is not None:
            embedding = self.memory._embeddings.get(experience.id)
            self.vector_store.add(
                id=experience.id,
                content=experience.content,
                metadata={
                    "domain": experience.domain,
                    "importance": experience.importance,
                    "novelty": experience.novelty,
                    "memory_id": mem_item.id,
                },
                embedding=embedding,
            )

        # 4. Episode buffer
        self.episodes.add(experience)

        # 5. Identity integration
        id_result = self.identity.integrate(experience)

        self._store_count += 1
        elapsed = time.time() - t0

        return {
            "experience_id": experience.id,
            "memory_id": mem_item.id,
            "tier": mem_item.tier.value,
            "crossmodal_modalities": list(cm_result.keys()),
            "identity_experiences": id_result["total_experiences"],
            "vector_indexed": self.vector_store is not None,
            "elapsed_ms": round(elapsed * 1000, 2),
        }

    def store_batch(self, experiences: Sequence[Experience]) -> list[dict[str, Any]]:
        """Store multiple experiences."""
        return [self.store(exp) for exp in experiences]

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, max_results: int = 10) -> list[RetrievalResult]:
        """Retrieve from hierarchical memory."""
        self._retrieve_count += 1
        return self.memory.retrieve(query, max_results)

    def retrieve_semantic(
        self,
        query: str,
        max_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic retrieval via the vector store (ChromaDB).

        Falls back to embedding-based hierarchical retrieval if no
        vector store is configured, or to lexical retrieval if no
        embedder is configured either.
        """
        if self.vector_store is not None:
            self._retrieve_count += 1
            return self.vector_store.query(
                query_text=query,
                n_results=max_results,
                where=where,
            )
        # Fallback: hierarchical retrieval (will use embeddings if available)
        return [
            {"id": r.memory.experience.id, "content": r.memory.experience.content,
             "score": r.score, "metadata": {"tier": r.source_tier.value}}
            for r in self.retrieve(query, max_results)
        ]

    def retrieve_crossmodal(
        self,
        query: Experience,
        max_results: int = 10,
    ) -> list[dict]:
        """Retrieve via cross-modal similarity."""
        return self.crossmodal.retrieve(query, max_results=max_results)

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def detect_episodes(self) -> list[Episode]:
        """Run episode boundary detection on buffered experiences."""
        return self.episodes.detect()

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Run memory consolidation across all tiers."""
        moved = self.memory.consolidate()
        return {
            "items_consolidated": moved,
            "memory_sizes": self.memory.size,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save identity to disk."""
        self.identity.save()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_stored": self._store_count,
            "total_retrieved": self._retrieve_count,
            "memory": self.memory.size,
            "crossmodal": self.crossmodal.size,
            "tokens": self.tokens.stats,
            "identity": {
                "total_experiences": self.identity.state.total_experiences,
                "domains": self.identity.state.domains_seen,
                "sessions": self.identity.state.session_count,
            },
            "throughput_per_sec": round(
                self._store_count / uptime, 1
            ) if uptime > 0 else 0,
        }
