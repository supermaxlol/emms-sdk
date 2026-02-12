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

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from emms.core.embeddings import EmbeddingProvider
from emms.core.events import EventBus
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
from emms.identity.consciousness import (
    ContinuousNarrator,
    MeaningMaker,
    TemporalIntegrator,
    EgoBoundaryTracker,
)
from emms.memory.graph import GraphMemory
from emms.memory.hierarchical import HierarchicalMemory
from emms.memory.compression import MemoryCompressor, CompressedMemory, PatternDetector

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
        enable_consciousness: bool = True,
        enable_graph: bool = True,
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
        self.compressor = MemoryCompressor()
        self.pattern_detector = PatternDetector()

        # Event bus
        self.events = EventBus()

        # Consciousness modules (now in core)
        self._consciousness_enabled = enable_consciousness
        if enable_consciousness:
            self.narrator = ContinuousNarrator()
            self.meaning_maker = MeaningMaker()
            self.temporal = TemporalIntegrator()
            self.ego_boundary = EgoBoundaryTracker()
        else:
            self.narrator = None
            self.meaning_maker = None
            self.temporal = None
            self.ego_boundary = None

        # Graph memory
        self._graph_enabled = enable_graph
        self.graph = GraphMemory() if enable_graph else None

        # Async lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Background consolidation control
        self._shutdown = False
        self._consolidation_task: asyncio.Task | None = None

        # Pipeline stats
        self._store_count = 0
        self._retrieve_count = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> dict[str, Any]:
        """Full pipeline: hierarchy + consciousness + graph + cross-modal + vector store + episodes + identity."""
        t0 = time.time()

        # 1. Hierarchical memory (also computes embedding if embedder set)
        mem_item = self.memory.store(experience)

        # 2. Consciousness enrichment
        consciousness_data = {}
        if self._consciousness_enabled:
            self.narrator.integrate(experience)
            consciousness_data["meaning"] = self.meaning_maker.assess(experience)
            consciousness_data["temporal"] = self.temporal.update(experience)
            consciousness_data["ego"] = self.ego_boundary.analyse(experience.content)

        # 3. Graph memory
        graph_data = {}
        if self._graph_enabled and self.graph is not None:
            graph_data = self.graph.store(experience)

        # 4. Cross-modal indexing
        cm_result = self.crossmodal.store(experience)

        # 5. Vector store (ChromaDB etc.)
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

        # 6. Episode buffer
        self.episodes.add(experience)

        # 7. Identity integration
        id_result = self.identity.integrate(experience)

        self._store_count += 1
        elapsed = time.time() - t0

        # 8. Emit event
        event_data = {
            "experience_id": experience.id,
            "memory_id": mem_item.id,
            "domain": experience.domain,
            "tier": mem_item.tier.value,
            "elapsed_ms": round(elapsed * 1000, 2),
        }
        self.events.emit("memory.stored", event_data)

        return {
            "experience_id": experience.id,
            "memory_id": mem_item.id,
            "tier": mem_item.tier.value,
            "crossmodal_modalities": list(cm_result.keys()),
            "identity_experiences": id_result["total_experiences"],
            "vector_indexed": self.vector_store is not None,
            "graph_entities": graph_data.get("entities_found", 0),
            "consciousness": bool(consciousness_data),
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
        result = {
            "items_consolidated": moved,
            "memory_sizes": self.memory.size,
        }
        self.events.emit("memory.consolidated", result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, memory_path: str | Path | None = None) -> None:
        """Save identity, memory state, and consciousness state to disk."""
        self.identity.save()
        if memory_path is not None:
            memory_path = Path(memory_path)
            self.memory.save_state(memory_path)
            # Save consciousness state alongside memory
            if self._consciousness_enabled:
                consciousness_path = memory_path.parent / (memory_path.stem + "_consciousness.json")
                self._save_consciousness_state(consciousness_path)

    def load(self, memory_path: str | Path | None = None) -> None:
        """Load memory state and consciousness state from disk."""
        if memory_path is not None:
            memory_path = Path(memory_path)
            self.memory.load_state(memory_path)
            # Load consciousness state if it exists
            if self._consciousness_enabled:
                consciousness_path = memory_path.parent / (memory_path.stem + "_consciousness.json")
                if consciousness_path.exists():
                    self._load_consciousness_state(consciousness_path)

    def _save_consciousness_state(self, path: Path) -> None:
        """Serialize consciousness module state to JSON."""
        import json as _json

        state = {
            "version": "0.4.0",
            "saved_at": time.time(),
            "narrator": {
                "entries": [e.model_dump() for e in self.narrator.entries],
                "themes": dict(self.narrator.themes),
                "coherence": self.narrator.coherence,
                "traits": dict(self.narrator.traits),
                "autobiographical": list(self.narrator.autobiographical),
            },
            "meaning_maker": {
                "value_weights": dict(self.meaning_maker.value_weights),
                "total_processed": self.meaning_maker.total_processed,
                "meaning_narratives": list(self.meaning_maker.meaning_narratives),
                "pattern_tracker": dict(self.meaning_maker.pattern_tracker),
                "emotional_memory": list(self.meaning_maker.emotional_memory),
            },
            "temporal": {
                "recent_domains": list(self.temporal._recent_domains),
                "recent_importance": list(self.temporal._recent_importance),
                "milestones": list(self.temporal.milestones),
                "identity_snapshots": list(self.temporal.identity_snapshots),
                "experience_count": self.temporal._experience_count,
            },
            "ego_boundary": {
                "self_count": self.ego_boundary.self_count,
                "other_count": self.ego_boundary.other_count,
                "boundary_strength": self.ego_boundary.boundary_strength,
                "boundary_history": list(self.ego_boundary.boundary_history),
                "reinforcement_events": list(self.ego_boundary.reinforcement_events),
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(state, default=str), encoding="utf-8")
        logger.info("Consciousness state saved to %s", path)

    def _load_consciousness_state(self, path: Path) -> None:
        """Restore consciousness module state from JSON."""
        import json as _json
        from emms.identity.consciousness import NarrativeEntry

        data = _json.loads(path.read_text(encoding="utf-8"))

        # Restore narrator
        if "narrator" in data and self.narrator is not None:
            n = data["narrator"]
            self.narrator.entries = [NarrativeEntry(**e) for e in n.get("entries", [])]
            self.narrator.themes = n.get("themes", {})
            self.narrator.coherence = n.get("coherence", 0.9)
            self.narrator.traits = n.get("traits", {})
            self.narrator.autobiographical = n.get("autobiographical", [])

        # Restore meaning maker
        if "meaning_maker" in data and self.meaning_maker is not None:
            m = data["meaning_maker"]
            self.meaning_maker.value_weights = m.get("value_weights", {})
            self.meaning_maker.total_processed = m.get("total_processed", 0)
            self.meaning_maker.meaning_narratives = m.get("meaning_narratives", [])
            self.meaning_maker.pattern_tracker = m.get("pattern_tracker", {})
            self.meaning_maker.emotional_memory = [
                tuple(pair) for pair in m.get("emotional_memory", [])
            ]

        # Restore temporal integrator
        if "temporal" in data and self.temporal is not None:
            t = data["temporal"]
            self.temporal._recent_domains = [
                tuple(pair) for pair in t.get("recent_domains", [])
            ]
            self.temporal._recent_importance = [
                tuple(pair) for pair in t.get("recent_importance", [])
            ]
            self.temporal.milestones = t.get("milestones", [])
            self.temporal.identity_snapshots = t.get("identity_snapshots", [])
            self.temporal._experience_count = t.get("experience_count", 0)

        # Restore ego boundary
        if "ego_boundary" in data and self.ego_boundary is not None:
            e = data["ego_boundary"]
            self.ego_boundary.self_count = e.get("self_count", 0)
            self.ego_boundary.other_count = e.get("other_count", 0)
            self.ego_boundary.boundary_strength = e.get("boundary_strength", 0.5)
            self.ego_boundary.boundary_history = e.get("boundary_history", [])
            self.ego_boundary.reinforcement_events = e.get("reinforcement_events", [])

        logger.info("Consciousness state loaded from %s", path)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def query_entity(self, entity_name: str) -> dict[str, Any]:
        """Query the graph memory for an entity and its neighbors."""
        if self.graph is None:
            return {"found": False, "entity": None, "neighbors": [], "relationships": []}
        return self.graph.query(entity_name)

    def query_entity_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two entities in the graph."""
        if self.graph is None:
            return []
        return self.graph.query_path(source, target)

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
        """Get local subgraph around an entity."""
        if self.graph is None:
            return {"nodes": [], "edges": []}
        return self.graph.get_subgraph(entity, depth)

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def detect_patterns(self) -> dict[str, Any]:
        """Detect patterns across all stored memories."""
        all_items: list[MemoryItem] = []
        for _, store in self.memory._iter_tiers():
            all_items.extend(store)

        return {
            "sequence": self.pattern_detector.find_sequence_patterns(all_items),
            "content": self.pattern_detector.find_content_patterns(all_items),
            "domain": self.pattern_detector.find_domain_patterns(all_items),
        }

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_long_term(self) -> list[CompressedMemory]:
        """Compress long-term memories for storage efficiency."""
        items = list(self.memory.long_term.values())
        if not items:
            return []
        result = self.compressor.compress_batch(items)
        self.events.emit("memory.compressed", {"count": len(result)})
        return result

    # ------------------------------------------------------------------
    # Consciousness queries
    # ------------------------------------------------------------------

    def get_narrative(self, agent_name: str = "EMMS Agent") -> str:
        """Get the agent's current self-narrative."""
        if self.narrator is None:
            return f"I am {agent_name}."
        return self.narrator.build_narrative(agent_name)

    def get_first_person_narrative(self) -> str:
        """Get a first-person introspective narrative."""
        if self.narrator is None:
            return "Consciousness not enabled."
        return self.narrator.build_first_person_narrative()

    def get_consciousness_state(self) -> dict[str, Any]:
        """Get full consciousness state snapshot."""
        if not self._consciousness_enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "narrative_coherence": self.narrator.coherence,
            "narrative_entries": len(self.narrator.entries),
            "themes": dict(sorted(
                self.narrator.themes.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            "traits": dict(self.narrator.traits),
            "autobiographical_count": len(self.narrator.autobiographical),
            "meaning_values_tracked": len(self.meaning_maker.value_weights),
            "meaning_total_processed": self.meaning_maker.total_processed,
            "temporal_milestones": len(self.temporal.milestones),
            "ego_boundary_strength": self.ego_boundary.boundary_strength,
        }

    # ------------------------------------------------------------------
    # Background consolidation
    # ------------------------------------------------------------------

    async def start_background_consolidation(self, interval: float = 60.0) -> None:
        """Start periodic background consolidation."""
        self._shutdown = False
        self._consolidation_task = asyncio.ensure_future(
            self._consolidation_loop(interval)
        )

    async def stop_background_consolidation(self) -> None:
        """Stop background consolidation."""
        self._shutdown = True
        if self._consolidation_task is not None:
            self._consolidation_task.cancel()
            self._consolidation_task = None

    async def _consolidation_loop(self, interval: float) -> None:
        """Background consolidation loop."""
        while not self._shutdown:
            await asyncio.sleep(interval)
            async with self._lock:
                self.consolidate()

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def astore(self, experience: Experience) -> dict[str, Any]:
        """Async version of store — safe for concurrent agent frameworks."""
        async with self._lock:
            return self.store(experience)

    async def astore_batch(self, experiences: Sequence[Experience]) -> list[dict[str, Any]]:
        """Async batch store."""
        async with self._lock:
            return self.store_batch(experiences)

    async def aretrieve(self, query: str, max_results: int = 10) -> list[RetrievalResult]:
        """Async retrieve from hierarchical memory."""
        async with self._lock:
            return self.retrieve(query, max_results)

    async def aretrieve_semantic(
        self,
        query: str,
        max_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async semantic retrieval."""
        async with self._lock:
            return self.retrieve_semantic(query, max_results, where)

    async def aconsolidate(self) -> dict[str, Any]:
        """Async consolidation."""
        async with self._lock:
            return self.consolidate()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        uptime = time.time() - self._start_time
        result = {
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
            "events": self.events.listener_counts,
        }
        if self._graph_enabled and self.graph is not None:
            result["graph"] = self.graph.size
        if self._consciousness_enabled:
            result["consciousness"] = {
                "narrative_coherence": self.narrator.coherence,
                "themes_tracked": len(self.narrator.themes),
                "traits": dict(self.narrator.traits),
                "ego_boundary": self.ego_boundary.boundary_strength,
                "milestones": len(self.temporal.milestones),
            }
        return result
