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
    ObsType,
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
from emms.memory.compression import MemoryCompressor, CompressedMemory, PatternDetector, SemanticDeduplicator
from emms.memory.procedural import ProceduralMemory, ProcedureEntry
from emms.memory.spaced_repetition import SpacedRepetitionSystem
from emms.core.importance import ImportanceClassifier
from emms.storage.index import CompactionIndex

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

        # Procedural memory (5th tier: evolving behavioral rules)
        self.procedures = ProceduralMemory()

        # v0.6.0: new sub-systems
        self.importance_clf = ImportanceClassifier()

        # v0.9.0: CompactionIndex — O(1) memory lookup
        self.index = CompactionIndex()
        self.deduplicator = SemanticDeduplicator(
            cosine_threshold=self.cfg.dedup_cosine_threshold,
            lexical_threshold=self.cfg.dedup_lexical_threshold,
        )
        self.srs = SpacedRepetitionSystem(self.memory)
        # Scheduler created lazily on start_scheduler() to avoid background
        # tasks being created before the event loop is running
        self._scheduler: "Any | None" = None

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

        # 0. Auto-enrich importance from content signals (if still default)
        self.importance_clf.enrich(experience)

        # 1. Hierarchical memory (also computes embedding if embedder set)
        mem_item = self.memory.store(experience)
        self.index.register(mem_item)  # v0.9.0: O(1) index

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
        """Save identity, memory state, graph state, procedural memory, consciousness, and SRS to disk."""
        self.identity.save()
        if memory_path is not None:
            memory_path = Path(memory_path)
            self.memory.save_state(memory_path)
            # Save graph state alongside memory
            if self._graph_enabled and self.graph is not None:
                graph_path = memory_path.parent / (memory_path.stem + "_graph.json")
                self.graph.save_state(graph_path)
            # Save procedural memory
            proc_path = memory_path.parent / (memory_path.stem + "_procedures.json")
            self.procedures.save_state(proc_path)
            # Save SRS state
            srs_path = memory_path.parent / (memory_path.stem + "_srs.json")
            self.srs.save_state(srs_path)
            # Save consciousness state alongside memory
            if self._consciousness_enabled:
                consciousness_path = memory_path.parent / (memory_path.stem + "_consciousness.json")
                self._save_consciousness_state(consciousness_path)

    def load(self, memory_path: str | Path | None = None) -> None:
        """Load memory state, graph state, procedural memory, consciousness, and SRS from disk."""
        if memory_path is not None:
            memory_path = Path(memory_path)
            self.memory.load_state(memory_path)
            # Load graph state if it exists
            if self._graph_enabled and self.graph is not None:
                graph_path = memory_path.parent / (memory_path.stem + "_graph.json")
                if graph_path.exists():
                    self.graph.load_state(graph_path)
            # Load procedural memory if it exists
            proc_path = memory_path.parent / (memory_path.stem + "_procedures.json")
            if proc_path.exists():
                self.procedures.load_state(proc_path)
            # Load SRS state if it exists
            srs_path = memory_path.parent / (memory_path.stem + "_srs.json")
            if srs_path.exists():
                self.srs.load_state(srs_path)
            # Load consciousness state if it exists
            if self._consciousness_enabled:
                consciousness_path = memory_path.parent / (memory_path.stem + "_consciousness.json")
                if consciousness_path.exists():
                    self._load_consciousness_state(consciousness_path)

    def _save_consciousness_state(self, path: Path) -> None:
        """Serialize consciousness module state to JSON."""
        import json as _json

        state = {
            "version": "0.5.1",
            "saved_at": time.time(),
            "narrator": {
                "entries": [e.model_dump() for e in self.narrator.entries],
                "themes": dict(self.narrator.themes),
                "coherence": self.narrator.coherence,
                "traits": dict(self.narrator.traits),
                "autobiographical": list(self.narrator.autobiographical),
                # A-MEM: preserve retroactive boost tuning
                "retroactive_boost": self.narrator._retroactive_boost,
            },
            "meaning_maker": {
                "value_weights": dict(self.meaning_maker.value_weights),
                "total_processed": self.meaning_maker.total_processed,
                "meaning_narratives": list(self.meaning_maker.meaning_narratives),
                "pattern_tracker": dict(self.meaning_maker.pattern_tracker),
                "emotional_memory": list(self.meaning_maker.emotional_memory),
                # domain curiosity (drives novelty-seeking)
                "domain_curiosity": dict(self.meaning_maker._domain_curiosity),
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
                # core creeds (hardened identity beliefs)
                "core_creeds": list(self.ego_boundary.core_creeds),
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
            self.narrator._retroactive_boost = n.get("retroactive_boost", 0.05)
            # Rebuild A-MEM bidirectional index from persisted linked_to lists
            # The linked_to indices are already restored via model_dump; no extra work needed
            # but ensure the narrator knows entries count matches
            logger.debug("Narrator: %d entries restored with A-MEM links", len(self.narrator.entries))

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
            self.meaning_maker._domain_curiosity = m.get("domain_curiosity", {})

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
            self.ego_boundary.core_creeds = e.get("core_creeds", [])

        logger.info("Consciousness state loaded from %s", path)

    # ------------------------------------------------------------------
    # RAG context building (v0.6.0)
    # ------------------------------------------------------------------

    def build_rag_context(
        self,
        query: str,
        max_results: int = 20,
        token_budget: int = 4000,
        fmt: str = "markdown",
        include_metadata: bool = True,
        min_score: float = 0.0,
    ) -> str:
        """Retrieve memories and build a token-budget-aware context document.

        Args:
            query: Natural-language search query.
            max_results: How many memories to retrieve before budget-packing.
            token_budget: Maximum context tokens (approximate).
            fmt: Output format — ``markdown``, ``xml``, ``json``, or ``plain``.
            include_metadata: Include score/tier/namespace annotations.
            min_score: Skip results below this score threshold.

        Returns:
            Context string ready to inject into an LLM prompt.
        """
        from emms.context.rag_builder import RAGContextBuilder
        results = self.retrieve(query, max_results=max_results)
        builder = RAGContextBuilder(
            token_budget=token_budget,
            include_metadata=include_metadata,
            min_score=min_score,
        )
        return builder.build(results, fmt=fmt)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Deduplication (v0.6.0)
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        cosine_threshold: float | None = None,
        lexical_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Scan long-term memories for near-duplicates and archive weaker copies.

        Args:
            cosine_threshold: Override cosine similarity threshold (default from config).
            lexical_threshold: Override lexical similarity threshold (default from config).

        Returns:
            Dict with ``groups_found`` and ``memories_archived`` counts.
        """
        if cosine_threshold is not None:
            self.deduplicator.cosine_threshold = cosine_threshold
        if lexical_threshold is not None:
            self.deduplicator.lexical_threshold = lexical_threshold

        items = list(self.memory.long_term.values())
        groups = self.deduplicator.find_duplicate_groups(items)
        archived = self.deduplicator.resolve_groups(groups)
        result = {"groups_found": len(groups), "memories_archived": len(archived)}
        self.events.emit("memory.deduplicated", result)
        return result

    # ------------------------------------------------------------------
    # Spaced Repetition System (v0.6.0)
    # ------------------------------------------------------------------

    def srs_enroll(self, memory_id: str) -> bool:
        """Enrol a memory in the SRS review schedule.

        Returns:
            True if enrolled (or already enrolled), False if memory not found.
        """
        card = self.srs.enroll(memory_id)
        return card is not None

    def srs_enroll_all(self) -> int:
        """Enrol all non-expired, non-superseded memories in SRS.

        Returns:
            Number of newly enrolled memories.
        """
        return self.srs.enroll_all()

    def srs_record_review(self, memory_id: str, quality: int) -> bool:
        """Record an SRS review outcome (quality 0–5).

        Args:
            memory_id: Memory that was reviewed.
            quality: Recall quality 0 (blackout) … 5 (perfect).

        Returns:
            True on success, False if memory not found.
        """
        card = self.srs.record_review(memory_id, quality)
        return card is not None

    def srs_due(self, max_items: int = 50) -> list[str]:
        """Return memory IDs due for SRS review, most-overdue first."""
        return [c.memory_id for c in self.srs.get_due_items(max_items)]

    # ------------------------------------------------------------------
    # Scheduler (v0.6.0)
    # ------------------------------------------------------------------

    async def start_scheduler(self, **kwargs: Any) -> None:
        """Start the MemoryScheduler with composable background jobs.

        Replaces the older ``start_background_consolidation()`` with a
        multi-job scheduler.  Keyword arguments are forwarded to
        ``MemoryScheduler.__init__()``.
        """
        from emms.scheduler import MemoryScheduler
        self._scheduler = MemoryScheduler(self, **kwargs)
        await self._scheduler.start()

    async def stop_scheduler(self) -> None:
        """Stop the MemoryScheduler."""
        if self._scheduler is not None:
            await self._scheduler.stop()
            self._scheduler = None

    # ------------------------------------------------------------------
    # Graph visualization (v0.6.0)
    # ------------------------------------------------------------------

    def export_graph_dot(
        self,
        title: str = "EMMS Knowledge Graph",
        max_nodes: int = 100,
        min_importance: float = 0.0,
        highlight: list[str] | None = None,
    ) -> str:
        """Export the knowledge graph as a Graphviz DOT string.

        Args:
            title: Graph title label.
            max_nodes: Maximum entity nodes to include.
            min_importance: Only include entities with importance ≥ this value.
            highlight: Entity names to highlight in red.

        Returns:
            DOT-language string, or empty string if graph disabled.
        """
        if self.graph is None:
            return ""
        return self.graph.to_dot(
            title=title,
            max_nodes=max_nodes,
            min_importance=min_importance,
            highlight=highlight,
        )

    def export_graph_d3(
        self,
        max_nodes: int = 200,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """Export the knowledge graph as a D3.js force-graph JSON dict.

        Returns:
            Dict with ``nodes`` and ``links`` arrays, or empty graph if disabled.
        """
        if self.graph is None:
            return {"nodes": [], "links": []}
        return self.graph.to_d3(max_nodes=max_nodes, min_importance=min_importance)

    # ------------------------------------------------------------------
    # ImportanceClassifier access (v0.6.0)
    # ------------------------------------------------------------------

    def score_importance(self, experience: "Experience") -> dict[str, float]:
        """Return per-signal importance breakdown for an experience.

        Useful for debugging why a memory was scored a certain way.
        """
        return self.importance_clf.score_breakdown(experience)

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
    # Filtered retrieval (namespace + structured filters)
    # ------------------------------------------------------------------

    def retrieve_filtered(
        self,
        query: str,
        max_results: int = 10,
        *,
        namespace: str | None = None,
        obs_type: "ObsType | None" = None,
        domain: str | None = None,
        session_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_confidence: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with structured pre-filters (namespace, obs_type, time range, confidence…).

        All filter args are optional; omitting one means no filtering on that field.
        Confidence scaling applies: memories with ``confidence < 1.0`` receive a
        proportional score penalty.
        """
        return self.memory.retrieve_filtered(
            query,
            max_results,
            namespace=namespace,
            obs_type=obs_type,
            domain=domain,
            session_id=session_id,
            since=since,
            until=until,
            min_confidence=min_confidence,
        )

    # ------------------------------------------------------------------
    # Memory feedback (upvote / downvote)
    # ------------------------------------------------------------------

    def upvote(self, memory_id: str, boost: float = 0.1) -> bool:
        """Positive feedback: strengthen a memory and record an access.

        Call this when a retrieved memory proved useful to the user.
        """
        return self.memory.upvote(memory_id, boost)

    def downvote(self, memory_id: str, decay: float = 0.2) -> bool:
        """Negative feedback: weaken a memory.

        Call this when a retrieved memory was irrelevant or incorrect.
        """
        return self.memory.downvote(memory_id, decay)

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def export_markdown(
        self,
        path: "str | Path",
        include_private: bool = False,
        namespace: str | None = None,
    ) -> int:
        """Export memories as a structured human-readable Markdown document.

        Groups memories by domain, includes facts, files, and metadata.
        Useful for human review, version-control diffing, or LLM context injection.

        Returns:
            Number of memories exported.
        """
        return self.memory.export_markdown(path, include_private=include_private, namespace=namespace)

    # ------------------------------------------------------------------
    # Streaming retrieval (v0.7.0)
    # ------------------------------------------------------------------

    async def astream_retrieve(self, query: str, max_results: int = 10):
        """Async generator that yields RetrievalResult items tier-by-tier.

        Results are emitted as they are scored, highest-priority tier first.
        Cooperative multitasking is maintained with asyncio.sleep(0) between
        tier boundaries so other tasks can run.

        Usage::

            async for result in agent.astream_retrieve("machine learning"):
                print(result.memory.experience.content, result.score)
        """
        async for result in self.memory.stream_retrieve(query, max_results):
            yield result

    # ------------------------------------------------------------------
    # Memory diff (v0.7.0)
    # ------------------------------------------------------------------

    def diff_since(
        self,
        snapshot_path: "str | Path",
        strength_threshold: float = 0.05,
    ) -> "Any":
        """Compare current memory state against a previously saved snapshot.

        Parameters
        ----------
        snapshot_path : path to a snapshot JSON written by ``save()``.
        strength_threshold : minimum strength delta to count as changed.

        Returns
        -------
        DiffResult with added/removed/strengthened/weakened/superseded lists.
        """
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        from pathlib import Path as _Path
        import json as _json

        p = _Path(snapshot_path)
        data = _json.loads(p.read_text(encoding="utf-8"))

        def _snap(data_: dict) -> tuple[dict, float]:
            from emms.memory.diff import _load_snapshot
            return _load_snapshot(data_)

        snap_a, time_a = _snap(data)

        # Build current state snapshot
        snap_b: dict[str, ItemSnapshot] = {}
        for _, store in self.memory._iter_tiers():
            for item in store:
                s = ItemSnapshot(
                    id=item.id,
                    experience_id=item.experience.id,
                    content=item.experience.content,
                    domain=item.experience.domain,
                    tier=item.tier.value,
                    importance=item.experience.importance,
                    memory_strength=item.memory_strength,
                    access_count=item.access_count,
                    stored_at=item.stored_at,
                    superseded_by=item.superseded_by,
                    title=item.experience.title,
                )
                snap_b[s.id] = s

        import time as _time
        return MemoryDiff.diff(snap_a, snap_b, time_a, _time.time(), strength_threshold)

    # ------------------------------------------------------------------
    # Memory clustering (v0.7.0)
    # ------------------------------------------------------------------

    def cluster_memories(
        self,
        k: int | None = None,
        auto_k: bool = False,
        tier: "str" = "long_term",
        k_min: int = 2,
        k_max: int = 10,
    ) -> "Any":
        """Cluster memory items into semantic groups.

        Parameters
        ----------
        k : number of clusters (required unless ``auto_k=True``).
        auto_k : if True, select k automatically via elbow method.
        tier : which tier to cluster (``"long_term"``, ``"semantic"``, etc.).
        k_min, k_max : search bounds for ``auto_k``.

        Returns
        -------
        list[MemoryCluster]
        """
        from emms.memory.clustering import MemoryClustering
        from emms.core.models import MemoryTier

        tier_enum = MemoryTier(tier)
        tier_map = {
            MemoryTier.WORKING: list(self.memory.working),
            MemoryTier.SHORT_TERM: list(self.memory.short_term),
            MemoryTier.LONG_TERM: list(self.memory.long_term.values()),
            MemoryTier.SEMANTIC: list(self.memory.semantic.values()),
        }
        items = tier_map.get(tier_enum, [])

        clustering = MemoryClustering()
        if self.memory._embeddings:
            return clustering.cluster_with_embeddings(
                items,
                embeddings=self.memory._embeddings,
                k=k,
                auto_k=auto_k,
                k_min=k_min,
                k_max=k_max,
            )
        return clustering.cluster(items, k=k, auto_k=auto_k, k_min=k_min, k_max=k_max)

    # ------------------------------------------------------------------
    # LLM consolidation (v0.7.0)
    # ------------------------------------------------------------------

    async def llm_consolidate(
        self,
        threshold: float = 0.7,
        llm_enhancer: "Any | None" = None,
        tier: "str" = "long_term",
        max_clusters: int = 20,
    ) -> "Any":
        """Scan memory for similar items and synthesise each cluster via LLM.

        Parameters
        ----------
        threshold : minimum similarity to link two items.
        llm_enhancer : optional LLMEnhancer; uses extractive fallback if None.
        tier : which tier to scan.
        max_clusters : cap on the number of clusters to process.

        Returns
        -------
        ConsolidationResult
        """
        from emms.llm.consolidator import LLMConsolidator
        from emms.core.models import MemoryTier

        consolidator = LLMConsolidator(self.memory)
        return await consolidator.auto_consolidate(
            threshold=threshold,
            llm_enhancer=llm_enhancer,
            tier=MemoryTier(tier),
            max_clusters=max_clusters,
        )

    # ------------------------------------------------------------------
    # Conversation buffer (v0.7.0)
    # ------------------------------------------------------------------

    def build_conversation_context(
        self,
        turns: list[tuple[str, str]],
        max_tokens: int = 2000,
        window_size: int = 20,
        summarise_chunk: int = 5,
    ) -> str:
        """Build a context string from a list of (role, content) conversation turns.

        Useful for injecting conversation history into a prompt without
        exceeding the token budget.

        Parameters
        ----------
        turns : list of (role, content) tuples.
        max_tokens : approximate token budget.
        window_size : max raw turns in the live window.
        summarise_chunk : how many turns to summarise at a time when evicting.

        Returns
        -------
        str — formatted context block.
        """
        from emms.sessions.conversation import ConversationBuffer

        buf = ConversationBuffer(
            window_size=window_size,
            summarise_chunk=summarise_chunk,
        )
        for role, content in turns:
            buf.observe_turn(role, content)
        return buf.get_context(max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Citation validation (GitHub Copilot pattern)
    # ------------------------------------------------------------------

    def validate_citations(self, experience: Experience) -> dict[str, bool]:
        """Check that each cited memory ID exists in the memory store.

        When a memory cites others (``experience.citations``), this verifies
        those memories are still present and strengthens them by +0.1 memory
        strength (Copilot-inspired: referenced memories self-renew).

        Args:
            experience: The experience whose citations to validate.

        Returns:
            Mapping of ``{mem_id: found}`` for each citation.
        """
        result: dict[str, bool] = {}
        for mem_id in experience.citations:
            found = False
            for _, store in self.memory._iter_tiers():
                for item in store:
                    if item.id == mem_id or item.experience.id == mem_id:
                        found = True
                        # Strengthen cited memory (TTL refresh + strength boost)
                        item.memory_strength = min(1.0, item.memory_strength + 0.1)
                        item.touch()
                        break
                if found:
                    break
            result[mem_id] = found
        return result

    # ------------------------------------------------------------------
    # File-based retrieval (claude-mem inspired)
    # ------------------------------------------------------------------

    def search_by_file(self, file_path: str) -> list[MemoryItem]:
        """Find all memories that reference a specific file path.

        Searches ``files_read`` and ``files_modified`` on stored experiences.
        Returns results sorted newest-first.

        Args:
            file_path: Exact or partial file path to search for.
        """
        return self.memory.search_by_file(file_path)

    # ------------------------------------------------------------------
    # Procedural memory (LangMem-inspired: 5th tier)
    # ------------------------------------------------------------------

    def add_procedure(
        self,
        rule: str,
        domain: str = "general",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> ProcedureEntry:
        """Add a behavioral rule to procedural memory."""
        return self.procedures.add(rule, domain=domain, importance=importance, tags=tags or [])

    def get_system_prompt_rules(self, domain: str | None = None) -> str:
        """Return formatted procedural rules for system prompt injection."""
        return self.procedures.get_prompt(domain=domain)

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
        """Background consolidation loop — consolidates memory and runs pattern detection."""
        _consolidation_count = 0
        while not self._shutdown:
            await asyncio.sleep(interval)
            async with self._lock:
                self.consolidate()
                _consolidation_count += 1
                # Run pattern detection every 5 consolidation passes
                if _consolidation_count % 5 == 0:
                    try:
                        patterns = self.detect_patterns()
                        if patterns.get("sequence") or patterns.get("content"):
                            logger.debug(
                                "Pattern detection: %d sequences, %d content patterns",
                                len(patterns.get("sequence", [])),
                                len(patterns.get("content", [])),
                            )
                            self.events.emit("memory.patterns_detected", patterns)
                    except Exception as e:
                        logger.warning("Pattern detection failed: %s", e)

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
        """Async retrieve — lock-free (retrieval is read-only, no state mutation)."""
        return self.retrieve(query, max_results)

    async def aretrieve_semantic(
        self,
        query: str,
        max_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async semantic retrieval — lock-free."""
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

    # ------------------------------------------------------------------
    # Hybrid retrieval (v0.8.0)
    # ------------------------------------------------------------------

    def hybrid_retrieve(
        self,
        query: str,
        max_results: int = 10,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: float = 60.0,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Hybrid BM25 + embedding retrieval fused via Reciprocal Rank Fusion.

        Combines lexical BM25 matching with embedding cosine similarity into a
        single ranked list.  No score normalisation needed — RRF is rank-based.

        Args:
            query: Natural-language search query.
            max_results: Maximum results to return.
            bm25_k1: BM25 term saturation parameter.
            bm25_b: BM25 length normalisation parameter.
            rrf_k: RRF smoothing constant (literature default: 60).
            min_score: Skip results with RRF score below this threshold.

        Returns:
            list[RetrievalResult] sorted by descending RRF score.
        """
        from emms.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever(
            self.memory,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rrf_k=rrf_k,
            embedder=self.embedder,
        )
        return retriever.retrieve_as_retrieval_results(
            query,
            max_results=max_results,
            min_score=min_score,
        )

    # ------------------------------------------------------------------
    # Memory timeline (v0.8.0)
    # ------------------------------------------------------------------

    def build_timeline(
        self,
        *,
        domain: str | None = None,
        since: float | None = None,
        until: float | None = None,
        tiers: list[str] | None = None,
        gap_threshold_seconds: float = 300.0,
        bucket_size_seconds: float = 3600.0,
        include_expired: bool = False,
    ) -> "Any":
        """Build a chronological memory timeline with gap and density analysis.

        Args:
            domain: Filter to a single domain (None = all).
            since: Only include memories stored after this Unix timestamp.
            until: Only include memories stored before this Unix timestamp.
            tiers: Which tier names to include (None = all).
            gap_threshold_seconds: Minimum gap duration to flag as a TemporalGap.
            bucket_size_seconds: Histogram bucket width.
            include_expired: Include expired/superseded memories.

        Returns:
            TimelineResult with events, gaps, density histogram, and statistics.
        """
        from emms.analytics.timeline import MemoryTimeline
        timeline = MemoryTimeline(
            self.memory,
            gap_threshold_seconds=gap_threshold_seconds,
            bucket_size_seconds=bucket_size_seconds,
        )
        return timeline.build(
            domain=domain,
            since=since,
            until=until,
            tiers=tiers,
            include_expired=include_expired,
        )

    # ------------------------------------------------------------------
    # Adaptive retrieval (v0.8.0)
    # ------------------------------------------------------------------

    def enable_adaptive_retrieval(
        self,
        strategies: list[str] | None = None,
        decay: float = 1.0,
        seed: int | None = None,
    ) -> "Any":
        """Create and attach an AdaptiveRetriever to this EMMS instance.

        The retriever is stored as ``self._adaptive_retriever`` and reused
        by ``adaptive_retrieve()`` / ``adaptive_feedback()`` / ``get_retrieval_beliefs()``.

        Args:
            strategies: Arm names (default: semantic, bm25, temporal, domain, importance).
            decay: Geometric discount applied per update (1.0 = no decay).
            seed: RNG seed for reproducible Thompson sampling.

        Returns:
            The new AdaptiveRetriever instance.
        """
        from emms.retrieval.adaptive import AdaptiveRetriever
        self._adaptive_retriever: "Any" = AdaptiveRetriever(
            self.memory,
            strategies=strategies,
            decay=decay,
            seed=seed,
            embedder=self.embedder,
        )
        return self._adaptive_retriever

    def adaptive_retrieve(
        self,
        query: str,
        max_results: int = 10,
        explore: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve using the Thompson-sampled strategy from the adaptive retriever.

        Requires ``enable_adaptive_retrieval()`` to have been called first.
        Falls back to standard ``retrieve()`` if no adaptive retriever is set.

        Args:
            query: Natural-language search query.
            max_results: Maximum results.
            explore: If True, sample via Thompson Sampling; if False, exploit best arm.

        Returns:
            list[RetrievalResult]
        """
        retriever = getattr(self, "_adaptive_retriever", None)
        if retriever is None:
            return self.retrieve(query, max_results=max_results)
        return retriever.retrieve(query, max_results=max_results, explore=explore)

    def adaptive_feedback(
        self,
        strategy_name: str | None = None,
        reward: float = 1.0,
    ) -> None:
        """Provide feedback to the adaptive retriever.

        Args:
            strategy_name: Arm to update (None = last selected arm).
            reward: 1.0 = helpful; 0.0 = not helpful.
        """
        retriever = getattr(self, "_adaptive_retriever", None)
        if retriever is not None:
            retriever.record_feedback(strategy_name, reward=reward)

    def get_retrieval_beliefs(self) -> dict[str, Any]:
        """Return the current Beta belief state for all adaptive retrieval arms.

        Returns:
            Dict mapping strategy name → {alpha, beta, mean, variance, pulls, rewards}.
        """
        retriever = getattr(self, "_adaptive_retriever", None)
        if retriever is None:
            return {}
        return {
            name: {
                "alpha": b.alpha,
                "beta": b.beta,
                "mean": b.mean,
                "variance": b.variance,
                "pulls": b.pulls,
                "rewards": b.rewards,
            }
            for name, b in retriever.get_beliefs().items()
        }

    # ------------------------------------------------------------------
    # Memory budget (v0.8.0)
    # ------------------------------------------------------------------

    def memory_token_footprint(self) -> dict[str, Any]:
        """Return per-tier and total token estimates for all stored memories.

        Returns:
            Dict with keys ``total``, ``by_tier``, ``memory_count``.
        """
        from emms.context.budget import MemoryBudget
        budget = MemoryBudget(self.memory)
        return budget.token_footprint()

    def enforce_memory_budget(
        self,
        max_tokens: int = 100_000,
        dry_run: bool = False,
        policy: str = "composite",
        protected_tiers: list[str] | None = None,
        importance_threshold: float = 0.8,
    ) -> "Any":
        """Enforce a token budget by evicting low-value memories.

        Args:
            max_tokens: Maximum allowed total token footprint.
            dry_run: If True, compute but do not actually evict.
            policy: Eviction policy name (``composite``, ``lru``, ``lfu``,
                    ``importance``, ``strength``).
            protected_tiers: Tier names immune to eviction (default: ``["semantic"]``).
            importance_threshold: Memories at or above this importance are protected.

        Returns:
            BudgetReport with eviction details.
        """
        from emms.context.budget import MemoryBudget, EvictionPolicy
        budget = MemoryBudget(
            self.memory,
            max_tokens=max_tokens,
            policy=EvictionPolicy(policy),
            protected_tiers=protected_tiers,
            importance_threshold=importance_threshold,
        )
        report = budget.enforce(dry_run=dry_run)
        if not dry_run:
            self.events.emit("memory.budget_enforced", {
                "evicted": report.evicted_count,
                "freed_tokens": report.freed_tokens,
            })
        return report

    # ------------------------------------------------------------------
    # Multi-hop graph reasoning (v0.8.0)
    # ------------------------------------------------------------------

    def multihop_query(
        self,
        seed: str,
        max_hops: int = 3,
        max_results: int = 20,
        min_strength: float = 0.0,
    ) -> "Any":
        """Run a multi-hop BFS reasoning query over the knowledge graph.

        Discovers indirect connections between entities across multiple
        relationship hops.  Requires graph memory to be enabled.

        Args:
            seed: Seed entity name (case-insensitive).
            max_hops: Maximum hop depth (default 3).
            max_results: Maximum reachable entities to return.
            min_strength: Skip paths with product edge strength below this.

        Returns:
            MultiHopResult with reachable entities, paths, bridging hubs,
            and Graphviz DOT export via ``result.to_dot()``.
        """
        from emms.memory.multihop import MultiHopGraphReasoner, MultiHopResult
        if self.graph is None:
            return MultiHopResult(
                seed=seed.lower(),
                reachable=[],
                paths=[],
                bridging_entities=[],
                total_entities_explored=0,
                max_hops_used=max_hops,
            )
        reasoner = MultiHopGraphReasoner(self.graph)
        return reasoner.query(
            seed,
            max_hops=max_hops,
            max_results=max_results,
            min_strength=min_strength,
        )

    # ------------------------------------------------------------------
    # CompactionIndex (v0.9.0)
    # ------------------------------------------------------------------

    def get_memory_by_id(self, memory_id: str) -> "MemoryItem | None":
        """O(1) lookup of a MemoryItem by its memory id."""
        return self.index.get_by_id(memory_id)

    def get_memory_by_experience_id(self, experience_id: str) -> "MemoryItem | None":
        """O(1) lookup of a MemoryItem by the originating experience id."""
        return self.index.get_by_experience_id(experience_id)

    def find_memories_by_content(self, content: str) -> "list[MemoryItem]":
        """Return all MemoryItems whose content hash matches *content*."""
        return self.index.find_by_content(content)

    def rebuild_index(self) -> int:
        """Rebuild the CompactionIndex from the current memory state.

        Returns:
            Number of items registered.
        """
        count = self.index.rebuild_from(self.memory)
        self.events.emit("memory.index_rebuilt", {"items": count})
        return count

    def index_stats(self) -> dict[str, int]:
        """Return CompactionIndex statistics."""
        return self.index.stats()

    # ------------------------------------------------------------------
    # GraphCommunityDetection (v0.9.0)
    # ------------------------------------------------------------------

    def graph_communities(
        self,
        max_iter: int = 100,
        seed: int | None = 42,
        min_community_size: int = 1,
    ) -> "Any":
        """Detect communities in the knowledge graph using Label Propagation.

        Args:
            max_iter: Maximum LPA iterations.
            seed: Random seed for reproducibility.
            min_community_size: Merge tiny communities below this size.

        Returns:
            CommunityResult with community list, modularity Q, bridge entities.
        """
        from emms.memory.communities import GraphCommunityDetector, CommunityResult
        if self.graph is None:
            return CommunityResult(
                communities=[],
                modularity=0.0,
                total_entities=0,
                total_edges=0,
                num_communities=0,
                converged=True,
                iterations_used=0,
                bridge_entities=[],
            )
        detector = GraphCommunityDetector(
            max_iter=max_iter,
            seed=seed,
            min_community_size=min_community_size,
        )
        return detector.detect(self.graph)

    def graph_community_for_entity(self, entity_name: str) -> "Any | None":
        """Return the Community containing *entity_name*, or None.

        Args:
            entity_name: Entity name (case-insensitive).

        Returns:
            Community dataclass or None if graph disabled / entity not found.
        """
        result = self.graph_communities()
        return result.get_community_for_entity(entity_name)

    # ------------------------------------------------------------------
    # ExperienceReplay (v0.9.0)
    # ------------------------------------------------------------------

    def enable_experience_replay(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        seed: int | None = None,
        **kwargs: Any,
    ) -> "Any":
        """Enable and configure the prioritized ExperienceReplay buffer.

        Args:
            alpha: Priority exponentiation (0=uniform, 1=fully prioritized).
            beta: IS correction exponent (0=none, 1=full).
            seed: Random seed for reproducibility.
            **kwargs: Additional ExperienceReplay constructor kwargs.

        Returns:
            The configured ExperienceReplay instance.
        """
        from emms.memory.replay import ExperienceReplay
        self._replay = ExperienceReplay(
            self.memory, alpha=alpha, beta=beta, seed=seed, **kwargs
        )
        return self._replay

    def replay_sample(self, k: int = 8, beta: float | None = None) -> "Any":
        """Draw a mini-batch of k items by priority from experience replay.

        Automatically enables ExperienceReplay with defaults if not configured.

        Args:
            k: Batch size.
            beta: IS correction exponent override.

        Returns:
            ReplayBatch.
        """
        if not hasattr(self, "_replay"):
            self.enable_experience_replay()
        return self._replay.sample(k=k, beta=beta)

    def replay_context(self, k: int = 5) -> "list[RetrievalResult]":
        """Sample k items from experience replay as RetrievalResult list."""
        if not hasattr(self, "_replay"):
            self.enable_experience_replay()
        return self._replay.replay_context(k=k)

    def replay_top(self, k: int = 8) -> "list[Any]":
        """Return the top-k highest-priority items (deterministic)."""
        if not hasattr(self, "_replay"):
            self.enable_experience_replay()
        return self._replay.sample_top(k=k)

    # ------------------------------------------------------------------
    # MemoryFederation (v0.9.0)
    # ------------------------------------------------------------------

    def merge_from(
        self,
        source: "Any",
        policy: str = "newest_wins",
        namespace_prefix: str | None = None,
        merge_graph: bool = True,
    ) -> "Any":
        """Merge memories from another EMMS instance into this one.

        Args:
            source: Another EMMS instance whose memories are read.
            policy: Conflict policy (``local_wins``, ``newest_wins``,
                    ``importance_wins``).
            namespace_prefix: Prepend this prefix to incoming memory ids.
            merge_graph: Also merge graph entities/relationships.

        Returns:
            FederationResult with merge statistics.
        """
        from emms.storage.federation import MemoryFederation, ConflictPolicy
        fed = MemoryFederation(
            target=self,
            policy=ConflictPolicy(policy),
            namespace_prefix=namespace_prefix,
            merge_graph=merge_graph,
        )
        result = fed.merge_from(source)
        # Re-register merged items in the index
        self.rebuild_index()
        self.events.emit("memory.federation_merged", {
            "items_merged": result.items_merged,
            "conflicts": len(result.conflicts),
        })
        return result

    def federation_export(self) -> "list[MemoryItem]":
        """Export all MemoryItems as a flat list for sharing with other agents."""
        from emms.storage.federation import MemoryFederation
        fed = MemoryFederation(target=self)
        return fed.export_snapshot()

    # ------------------------------------------------------------------
    # MemoryQueryPlanner (v0.9.0)
    # ------------------------------------------------------------------

    def plan_retrieve(
        self,
        query: str,
        max_results: int = 20,
        max_results_per_sub: int = 10,
        cross_boost: float = 0.10,
    ) -> "Any":
        """Decompose query into sub-queries, retrieve each, cross-boost, merge.

        Args:
            query: Natural-language query (may contain conjunctions/commas).
            max_results: Maximum items in the merged result list.
            max_results_per_sub: Cap per sub-query retrieval.
            cross_boost: Score bump per additional sub-query that returns an item.

        Returns:
            QueryPlan with sub-queries, sub-results, merged results, stats.
        """
        from emms.retrieval.planner import MemoryQueryPlanner
        planner = MemoryQueryPlanner(
            memory=self.memory,
            max_results_per_sub=max_results_per_sub,
            max_final_results=max_results,
            cross_boost=cross_boost,
            embedder=self.embedder,
        )
        return planner.plan_retrieve(query)

    def plan_retrieve_simple(
        self,
        query: str,
        max_results: int = 20,
    ) -> "list[RetrievalResult]":
        """Convenience wrapper: plan_retrieve returning only the result list."""
        from emms.retrieval.planner import MemoryQueryPlanner
        planner = MemoryQueryPlanner(
            memory=self.memory,
            max_final_results=max_results,
            embedder=self.embedder,
        )
        return planner.plan_retrieve_simple(query)

    # ------------------------------------------------------------------
    # ReconsolidationEngine (v0.10.0)
    # ------------------------------------------------------------------

    def reconsolidate(
        self,
        memory_id: str,
        context_valence: float | None = None,
        reinforce: bool = True,
    ) -> "Any":
        """Reconsolidate a single memory after recall.

        Strengthens or weakens the memory and optionally drifts its
        emotional valence toward the current context.

        Args:
            memory_id: The MemoryItem ID to reconsolidate.
            context_valence: Emotional valence of the recall context (-1…+1).
            reinforce: True → strengthen (confirming recall); False → weaken.

        Returns:
            ReconsolidationResult with before/after values.
        """
        from emms.memory.reconsolidation import ReconsolidationEngine
        item = self.get_memory_by_id(memory_id)
        if item is None:
            raise KeyError(f"Memory not found: {memory_id!r}")
        engine = ReconsolidationEngine()
        return engine.reconsolidate(item, context_valence=context_valence, reinforce=reinforce)

    def batch_reconsolidate(
        self,
        memory_ids: "list[str]",
        context_valence: float | None = None,
        reinforce: bool = True,
    ) -> "Any":
        """Reconsolidate a batch of recently-recalled memories.

        Useful after a retrieval round — pass retrieved item IDs to update
        their reconsolidation state.

        Returns:
            ReconsolidationReport summarising changes across the batch.
        """
        from emms.memory.reconsolidation import ReconsolidationEngine
        engine = ReconsolidationEngine()
        items = [self.get_memory_by_id(mid) for mid in memory_ids]
        items = [it for it in items if it is not None]
        return engine.batch_reconsolidate(
            items, context_valence=context_valence, reinforce=reinforce
        )

    def decay_unrecalled(
        self,
        decay_factor: float = 0.02,
        min_age_seconds: float = 3600.0,
    ) -> "Any":
        """Apply passive decay to memories that have not been recalled recently.

        Args:
            decay_factor: Absolute strength reduction per call (default 0.02).
            min_age_seconds: Items accessed more recently than this are skipped.

        Returns:
            ReconsolidationReport summarising which items were decayed.
        """
        from emms.memory.reconsolidation import ReconsolidationEngine
        engine = ReconsolidationEngine()
        all_items: list[MemoryItem] = []
        for tier in (self.memory.working, self.memory.short_term):
            all_items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            all_items.extend(tier.values())
        return engine.decay_unrecalled(
            all_items, decay_factor=decay_factor, min_age_seconds=min_age_seconds
        )

    # ------------------------------------------------------------------
    # PresenceTracker (v0.10.0)
    # ------------------------------------------------------------------

    def enable_presence_tracking(
        self,
        session_id: str | None = None,
        attention_half_life: int = 20,
        decay_gamma: float = 1.5,
        budget_horizon: int = 50,
        degrading_threshold: float = 0.4,
    ) -> "Any":
        """Initialise a PresenceTracker for this session.

        Returns:
            PresenceTracker instance (also stored as self._presence_tracker).
        """
        from emms.sessions.presence import PresenceTracker
        self._presence_tracker: "Any" = PresenceTracker(
            session_id=session_id,
            attention_half_life=attention_half_life,
            decay_gamma=decay_gamma,
            budget_horizon=budget_horizon,
            degrading_threshold=degrading_threshold,
        )
        return self._presence_tracker

    def record_presence_turn(
        self,
        content: str = "",
        domain: str = "general",
        valence: float = 0.0,
        intensity: float = 0.0,
    ) -> "Any":
        """Record a turn against the active PresenceTracker.

        Returns:
            PresenceMetrics snapshot after recording the turn.

        Raises:
            RuntimeError if presence tracking has not been enabled.
        """
        tracker = getattr(self, "_presence_tracker", None)
        if tracker is None:
            raise RuntimeError(
                "Call enable_presence_tracking() before record_presence_turn()."
            )
        return tracker.record_turn(
            content=content, domain=domain, valence=valence, intensity=intensity
        )

    def presence_metrics(self) -> "Any":
        """Return current PresenceMetrics.

        Raises:
            RuntimeError if presence tracking has not been enabled.
        """
        tracker = getattr(self, "_presence_tracker", None)
        if tracker is None:
            raise RuntimeError(
                "Call enable_presence_tracking() before presence_metrics()."
            )
        return tracker.get_metrics()

    # ------------------------------------------------------------------
    # AffectiveRetriever (v0.10.0)
    # ------------------------------------------------------------------

    def affective_retrieve(
        self,
        query: str = "",
        target_valence: float | None = None,
        target_intensity: float | None = None,
        max_results: int = 10,
        semantic_blend: float = 0.4,
    ) -> "list[Any]":
        """Retrieve memories by emotional proximity.

        Finds memories whose emotional signature (valence + intensity) is
        closest to the target state.  Optionally blends semantic (BM25)
        matching when a query string is provided.

        Args:
            query: Optional semantic query (BM25 blend).
            target_valence: Target emotional valence (-1…+1).
            target_intensity: Target emotional intensity (0…1).
            max_results: Maximum results.
            semantic_blend: Weight of semantic score [0, 1] (default 0.4).

        Returns:
            list[AffectiveResult] sorted by score descending.
        """
        from emms.retrieval.affective import AffectiveRetriever
        retriever = AffectiveRetriever(self.memory, semantic_blend=semantic_blend)
        return retriever.retrieve(
            query=query,
            target_valence=target_valence,
            target_intensity=target_intensity,
            max_results=max_results,
        )

    def affective_retrieve_similar(
        self,
        reference_memory_id: str,
        max_results: int = 10,
    ) -> "list[Any]":
        """Retrieve memories with emotional signature similar to a reference.

        Args:
            reference_memory_id: MemoryItem ID of the reference memory.
            max_results: Maximum results (reference excluded).

        Returns:
            list[AffectiveResult] sorted by emotional proximity.
        """
        from emms.retrieval.affective import AffectiveRetriever
        retriever = AffectiveRetriever(self.memory)
        return retriever.retrieve_similar_feeling(
            reference_memory_id=reference_memory_id,
            max_results=max_results,
        )

    def emotional_landscape(self) -> "Any":
        """Return EmotionalLandscape summarising the emotional distribution of all memories.

        Returns:
            EmotionalLandscape with mean/std valence & intensity, histograms,
            and the IDs of the most positive, most negative, and most intense memories.
        """
        from emms.retrieval.affective import AffectiveRetriever
        retriever = AffectiveRetriever(self.memory)
        return retriever.emotional_landscape()

    # ------------------------------------------------------------------
    # DreamConsolidator (v0.11.0)
    # ------------------------------------------------------------------

    def dream(
        self,
        session_id: str | None = None,
        reinforce_top_k: int = 20,
        weaken_bottom_k: int = 10,
        prune_threshold: float = 0.05,
        run_dedup: bool = True,
        run_patterns: bool = True,
    ) -> "Any":
        """Run a between-session dream consolidation pass.

        Replays high-priority memories (ExperienceReplay), applies
        ReconsolidationEngine to strengthen important ones and weaken
        neglected ones, optionally runs deduplication and pattern detection,
        and prunes memories below strength threshold.

        Call this at the end of a session or before starting a new one.

        Args:
            session_id: Label for the dream report.
            reinforce_top_k: Number of top-priority memories to reinforce.
            weaken_bottom_k: Number of low-priority memories to weaken.
            prune_threshold: Memories with strength < this are pruned.
            run_dedup: Run SemanticDeduplicator pass (default True).
            run_patterns: Run PatternDetector pass (default True).

        Returns:
            DreamReport with consolidation statistics and insights.
        """
        from emms.memory.dream import DreamConsolidator
        consolidator = DreamConsolidator(
            memory=self.memory,
            reinforce_top_k=reinforce_top_k,
            weaken_bottom_k=weaken_bottom_k,
            prune_threshold=prune_threshold,
            run_dedup=run_dedup,
            run_patterns=run_patterns,
        )
        report = consolidator.dream(session_id=session_id)
        self.events.emit("memory.dream_completed", {
            "session_id": report.session_id,
            "reinforced": report.reinforced,
            "pruned": report.pruned,
        })
        return report

    # ------------------------------------------------------------------
    # SessionBridge (v0.11.0)
    # ------------------------------------------------------------------

    def capture_session_bridge(
        self,
        session_id: str | None = None,
        closing_summary: str = "",
        max_threads: int = 5,
    ) -> "Any":
        """Capture unresolved threads and session state for the next session.

        Returns a BridgeRecord that can be persisted and injected at the
        start of the next session via inject_session_bridge().

        Args:
            session_id: The ending session's ID.
            closing_summary: Optional free-text summary of the session.
            max_threads: Maximum number of unresolved threads to carry forward.

        Returns:
            BridgeRecord.
        """
        from emms.sessions.bridge import SessionBridge
        bridge = SessionBridge(
            memory=self.memory,
            presence_tracker=getattr(self, "_presence_tracker", None),
            max_threads=max_threads,
        )
        return bridge.capture(session_id=session_id, closing_summary=closing_summary)

    def inject_session_bridge(
        self,
        record: "Any",
        new_session_id: str | None = None,
    ) -> str:
        """Generate a context string from a BridgeRecord for session opening.

        Args:
            record: A BridgeRecord from capture_session_bridge().
            new_session_id: ID for the new session.

        Returns:
            str — prompt-ready markdown context injection.
        """
        from emms.sessions.bridge import SessionBridge
        bridge = SessionBridge(memory=self.memory)
        return bridge.inject(record, new_session_id=new_session_id)

    # ------------------------------------------------------------------
    # MemoryAnnealer (v0.11.0)
    # ------------------------------------------------------------------

    def anneal(
        self,
        last_session_at: float | None = None,
        half_life_gap: float = 259_200.0,
        decay_rate: float = 0.03,
        emotional_stabilization_rate: float = 0.08,
    ) -> "Any":
        """Anneal the memory landscape after a session gap.

        Models how time changes memory: weak ones decay faster, emotional
        charges stabilise toward neutral, important survivors are mildly
        strengthened.

        Args:
            last_session_at: Unix timestamp of last session end.
                If None, assumes half_life_gap has passed.
            half_life_gap: Gap in seconds at which temperature = 0.5
                (default 3 days = 259200s).
            decay_rate: Base decay per item per pass (default 0.03).
            emotional_stabilization_rate: Rate of valence drift toward 0
                (default 0.08).

        Returns:
            AnnealingResult with change counts and temperature.
        """
        from emms.memory.annealing import MemoryAnnealer
        annealer = MemoryAnnealer(
            memory=self.memory,
            half_life_gap=half_life_gap,
            decay_rate=decay_rate,
            emotional_stabilization_rate=emotional_stabilization_rate,
        )
        return annealer.anneal(last_session_at=last_session_at)

    # ------------------------------------------------------------------
    # AssociationGraph + InsightEngine + AssociativeRetriever (v0.12.0)
    # ------------------------------------------------------------------

    def build_association_graph(
        self,
        semantic_threshold: float = 0.5,
        temporal_window: float = 300.0,
        affective_tolerance: float = 0.3,
    ) -> "Any":
        """Build an association graph over all stored memories.

        Creates (or replaces) the shared :class:`AssociationGraph` on this
        EMMS instance, runs ``auto_associate()`` over every stored memory,
        and returns summary statistics.

        Args:
            semantic_threshold:  Minimum cosine similarity for a semantic edge
                (default 0.5).
            temporal_window:     Max seconds between items for a temporal edge
                (default 300 = 5 min).
            affective_tolerance: Max |valence_a − valence_b| for affective edge
                (default 0.3).

        Returns:
            :class:`AssociationStats` for the built graph.
        """
        from emms.memory.association import AssociationGraph
        self._association_graph = AssociationGraph(
            memory=self.memory,
            semantic_threshold=semantic_threshold,
            temporal_window=temporal_window,
            affective_tolerance=affective_tolerance,
        )
        self._association_graph.auto_associate()
        return self._association_graph.stats()

    def associate(
        self,
        id_a: str,
        id_b: str,
        edge_type: str = "explicit",
        weight: float = 0.8,
    ) -> "Any":
        """Manually add a bidirectional association between two memories.

        Ensures the shared graph exists (building it if needed) then adds
        the explicit edge.

        Args:
            id_a:      Source memory ID.
            id_b:      Target memory ID.
            edge_type: Edge type label (default "explicit").
            weight:    Edge weight 0–1 (default 0.8).

        Returns:
            :class:`AssociationEdge` (A→B direction).
        """
        from emms.memory.association import AssociationGraph
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
            self._association_graph.auto_associate()
        return self._association_graph.associate(id_a, id_b, edge_type=edge_type, weight=weight)

    def spreading_activation(
        self,
        seed_ids: list[str],
        decay: float = 0.5,
        steps: int = 3,
    ) -> "Any":
        """Run spreading activation from seed memory IDs on the association graph.

        Builds the graph lazily if it does not exist yet.

        Args:
            seed_ids: List of memory IDs to initialise activation from.
            decay:    Decay factor per hop (default 0.5).
            steps:    Maximum hop depth (default 3).

        Returns:
            List of :class:`ActivationResult` sorted by activation descending.
        """
        from emms.memory.association import AssociationGraph
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
            self._association_graph.auto_associate()
        return self._association_graph.spreading_activation(
            seed_ids, decay=decay, steps=steps
        )

    def association_stats(self) -> "Any":
        """Return statistics for the current association graph.

        Builds the graph lazily if it does not exist yet.

        Returns:
            :class:`AssociationStats`.
        """
        from emms.memory.association import AssociationGraph
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
            self._association_graph.auto_associate()
        return self._association_graph.stats()

    def discover_insights(
        self,
        session_id: str | None = None,
        max_insights: int = 8,
        min_bridge_weight: float = 0.45,
        rebuild_graph: bool = True,
    ) -> "Any":
        """Find cross-domain memory bridges and generate insight memories.

        Uses the shared :class:`AssociationGraph` (rebuilding if requested)
        to find pairs of memories from different domains that are strongly
        connected, then synthesises and stores an insight memory for each.

        Args:
            session_id:        Label attached to the report.
            max_insights:      Maximum insight memories to generate (default 8).
            min_bridge_weight: Minimum edge weight to qualify as a bridge
                (default 0.45).
            rebuild_graph:     Rebuild the association graph before searching
                (default True).

        Returns:
            :class:`InsightReport` with bridges found and new memory IDs.
        """
        from emms.memory.association import AssociationGraph
        from emms.memory.insight import InsightEngine
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
        engine = InsightEngine(
            memory=self.memory,
            association_graph=self._association_graph,
            max_insights=max_insights,
            min_bridge_weight=min_bridge_weight,
        )
        return engine.discover(session_id=session_id, rebuild_graph=rebuild_graph)

    def associative_retrieve(
        self,
        seed_ids: list[str],
        max_results: int = 10,
        steps: int = 3,
        decay: float = 0.5,
    ) -> "Any":
        """Retrieve memories via spreading activation from seed memory IDs.

        Args:
            seed_ids:    Memory IDs to initialise activation from.
            max_results: Maximum results (default 10).
            steps:       Hop depth (default 3).
            decay:       Activation decay per hop (default 0.5).

        Returns:
            List of :class:`AssociativeResult` sorted by activation descending.
        """
        from emms.memory.association import AssociationGraph
        from emms.retrieval.associative import AssociativeRetriever
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
            self._association_graph.auto_associate()
        retriever = AssociativeRetriever(
            memory=self.memory,
            association_graph=self._association_graph,
        )
        return retriever.retrieve(seed_ids, max_results=max_results, steps=steps, decay=decay)

    def associative_retrieve_by_query(
        self,
        query: str,
        seed_count: int = 3,
        max_results: int = 10,
        steps: int = 3,
        decay: float = 0.5,
        rebuild_graph: bool = False,
    ) -> "Any":
        """Retrieve memories associatively starting from a text query.

        Finds seed memories matching the query via token overlap, then spreads
        activation through the association graph to surface connected memories.

        Args:
            query:         Text query to select seed memories.
            seed_count:    Number of seed memories (default 3).
            max_results:   Maximum results (default 10).
            steps:         Hop depth (default 3).
            decay:         Activation decay per hop (default 0.5).
            rebuild_graph: Rebuild the association graph before retrieval.

        Returns:
            List of :class:`AssociativeResult` sorted by activation descending.
        """
        from emms.memory.association import AssociationGraph
        from emms.retrieval.associative import AssociativeRetriever
        if not hasattr(self, "_association_graph"):
            self._association_graph = AssociationGraph(memory=self.memory)
            self._association_graph.auto_associate()
        retriever = AssociativeRetriever(
            memory=self.memory,
            association_graph=self._association_graph,
        )
        return retriever.retrieve_by_query(
            query,
            seed_count=seed_count,
            max_results=max_results,
            steps=steps,
            decay=decay,
            rebuild_graph=rebuild_graph,
        )

    # ------------------------------------------------------------------
    # MetacognitionEngine (v0.13.0)
    # ------------------------------------------------------------------

    def assess_memory(
        self,
        memory_id: str,
    ) -> "Any":
        """Compute epistemic confidence for a single memory.

        Args:
            memory_id: The memory ID to assess.

        Returns:
            :class:`MemoryConfidence` with factor breakdown.
        """
        from emms.memory.metacognition import MetacognitionEngine
        item = self.get_memory_by_id(memory_id)
        engine = MetacognitionEngine(memory=self.memory)
        return engine.assess(item)

    def metacognition_report(
        self,
        max_contradictions: int = 5,
        recency_decay: float = 0.05,
        confidence_threshold_high: float = 0.65,
        confidence_threshold_low: float = 0.3,
    ) -> "Any":
        """Generate a comprehensive metacognitive self-assessment.

        Covers epistemic confidence across all memories, per-domain knowledge
        profiles, contradiction pairs, knowledge gaps, and recommendations.

        Args:
            max_contradictions:        Max contradiction pairs to include.
            recency_decay:             Daily decay rate for recency factor.
            confidence_threshold_high: Confidence above this = high confidence.
            confidence_threshold_low:  Confidence below this = low confidence.

        Returns:
            :class:`MetacognitionReport`.
        """
        from emms.memory.metacognition import MetacognitionEngine
        engine = MetacognitionEngine(
            memory=self.memory,
            recency_decay=recency_decay,
            confidence_threshold_high=confidence_threshold_high,
            confidence_threshold_low=confidence_threshold_low,
        )
        return engine.report(max_contradictions=max_contradictions)

    def knowledge_map(self) -> "Any":
        """Return a per-domain knowledge profile (confidence, coverage, importance).

        Returns:
            List of :class:`DomainProfile` sorted by memory count descending.
        """
        from emms.memory.metacognition import MetacognitionEngine
        engine = MetacognitionEngine(memory=self.memory)
        return engine.knowledge_map()

    def find_contradictions(self, max_pairs: int = 10) -> "Any":
        """Find memory pairs with semantic overlap but conflicting valence.

        Args:
            max_pairs: Maximum pairs to return (default 10).

        Returns:
            List of :class:`ContradictionPair` sorted by contradiction_score.
        """
        from emms.memory.metacognition import MetacognitionEngine
        engine = MetacognitionEngine(memory=self.memory)
        return engine.find_contradictions(max_pairs=max_pairs)

    # ------------------------------------------------------------------
    # ProspectiveMemory (v0.13.0)
    # ------------------------------------------------------------------

    def enable_prospective_memory(
        self,
        overlap_threshold: float = 0.15,
        max_intentions: int = 50,
    ) -> "Any":
        """Enable the prospective memory module (lazy init).

        Args:
            overlap_threshold: Minimum token overlap to trigger an intention.
            max_intentions:    Maximum stored intentions (default 50).

        Returns:
            The :class:`ProspectiveMemory` instance.
        """
        from emms.memory.prospection import ProspectiveMemory
        if not hasattr(self, "_prospective"):
            self._prospective = ProspectiveMemory(
                overlap_threshold=overlap_threshold,
                max_intentions=max_intentions,
            )
        return self._prospective

    def intend(
        self,
        content: str,
        trigger_context: str,
        priority: float = 0.5,
    ) -> "Any":
        """Store a future-oriented intention.

        Creates the prospective memory module if not yet enabled.

        Args:
            content:         What the agent intends to do.
            trigger_context: Context description that should trigger this.
            priority:        Urgency 0–1 (default 0.5).

        Returns:
            :class:`Intention`.
        """
        if not hasattr(self, "_prospective"):
            self.enable_prospective_memory()
        return self._prospective.intend(
            content=content,
            trigger_context=trigger_context,
            priority=priority,
        )

    def check_intentions(self, current_context: str) -> "Any":
        """Check which intentions are activated by the current context.

        Args:
            current_context: Text representing current conversational context.

        Returns:
            List of :class:`IntentionActivation` sorted by activation_score.
        """
        if not hasattr(self, "_prospective"):
            return []
        return self._prospective.check(current_context)

    def fulfill_intention(self, intention_id: str) -> bool:
        """Mark an intention as fulfilled.

        Args:
            intention_id: ID of the intention to fulfill.

        Returns:
            ``True`` if found and fulfilled.
        """
        if not hasattr(self, "_prospective"):
            return False
        return self._prospective.fulfill(intention_id)

    def pending_intentions(self) -> "Any":
        """Return all unfulfilled intentions sorted by priority.

        Returns:
            List of unfulfilled :class:`Intention` objects.
        """
        if not hasattr(self, "_prospective"):
            return []
        return self._prospective.pending()

    # ------------------------------------------------------------------
    # ContextualSalienceRetriever (v0.13.0)
    # ------------------------------------------------------------------

    def enable_contextual_retrieval(
        self,
        window_size: int = 6,
        semantic_weight: float = 0.35,
        importance_weight: float = 0.30,
        recency_weight: float = 0.25,
        affective_weight: float = 0.10,
    ) -> "Any":
        """Enable the contextual salience retriever (lazy init).

        Args:
            window_size:        Number of recent text snippets in window.
            semantic_weight:    Weight for semantic overlap factor.
            importance_weight:  Weight for memory importance factor.
            recency_weight:     Weight for storage recency factor.
            affective_weight:   Weight for affective resonance factor.

        Returns:
            The :class:`ContextualSalienceRetriever` instance.
        """
        from emms.retrieval.contextual import ContextualSalienceRetriever
        if not hasattr(self, "_contextual_retriever"):
            self._contextual_retriever = ContextualSalienceRetriever(
                memory=self.memory,
                window_size=window_size,
                semantic_weight=semantic_weight,
                importance_weight=importance_weight,
                recency_weight=recency_weight,
                affective_weight=affective_weight,
            )
        return self._contextual_retriever

    def update_context(self, text: str, valence: float = 0.0) -> None:
        """Add text to the rolling context window for salience retrieval.

        Creates the contextual retriever if not yet enabled.

        Args:
            text:    Text to add (e.g. user message + agent reply combined).
            valence: Estimated emotional valence of the snippet.
        """
        if not hasattr(self, "_contextual_retriever"):
            self.enable_contextual_retrieval()
        self._contextual_retriever.update_context(text, valence=valence)

    def contextual_retrieve(self, max_results: int = 10) -> "Any":
        """Retrieve memories salient to the current rolling context window.

        Args:
            max_results: Maximum memories to return (default 10).

        Returns:
            List of :class:`SalienceResult` sorted by salience_score descending.
        """
        if not hasattr(self, "_contextual_retriever"):
            return []
        return self._contextual_retriever.retrieve(max_results=max_results)

    def context_summary(self) -> str:
        """Return a brief summary of the current context window.

        Returns:
            String summary, or "(empty context)" if not active.
        """
        if not hasattr(self, "_contextual_retriever"):
            return "(contextual retrieval not enabled)"
        return self._contextual_retriever.context_summary

    # ------------------------------------------------------------------
    # EpisodicBuffer (v0.14.0)
    # ------------------------------------------------------------------

    def open_episode(
        self,
        session_id: Optional[str] = None,
        topic: str = "",
    ) -> "Any":
        """Open a new bounded episode in the episodic buffer.

        Automatically closes any currently open episode first.

        Args:
            session_id: Optional session label (auto-generated if omitted).
            topic:      Brief description of what this episode is about.

        Returns:
            The newly created :class:`Episode`.
        """
        if not hasattr(self, "_episodic_buffer"):
            from emms.memory.episodic import EpisodicBuffer
            self._episodic_buffer = EpisodicBuffer()
        return self._episodic_buffer.open_episode(session_id=session_id, topic=topic)

    def close_episode(
        self,
        episode_id: Optional[str] = None,
        outcome: str = "",
    ) -> "Any":
        """Close an episode, computing final statistics.

        Args:
            episode_id: Episode to close.  Defaults to the current open episode.
            outcome:    Brief description of how the episode resolved.

        Returns:
            The closed :class:`Episode`, or ``None`` if none was open.
        """
        if not hasattr(self, "_episodic_buffer"):
            return None
        return self._episodic_buffer.close_episode(episode_id=episode_id, outcome=outcome)

    def record_episode_turn(
        self,
        content: str = "",
        valence: float = 0.0,
        episode_id: Optional[str] = None,
    ) -> None:
        """Record a conversational turn within the current episode.

        Args:
            content: Turn text (used to update turn count).
            valence: Emotional valence of this turn (−1..1).
            episode_id: Episode to record into.  Defaults to current.
        """
        if not hasattr(self, "_episodic_buffer"):
            return
        self._episodic_buffer.record_turn(
            episode_id=episode_id, content=content, valence=valence
        )

    def recent_episodes(self, n: int = 10) -> "Any":
        """Return the *n* most recent episodes, newest first.

        Args:
            n: Number of episodes to return (default 10).

        Returns:
            List of :class:`Episode` objects.
        """
        if not hasattr(self, "_episodic_buffer"):
            return []
        return self._episodic_buffer.recent_episodes(n=n)

    def current_episode(self) -> "Any":
        """Return the currently open episode, or ``None``."""
        if not hasattr(self, "_episodic_buffer"):
            return None
        return self._episodic_buffer.current_episode()

    # ------------------------------------------------------------------
    # SchemaExtractor (v0.14.0)
    # ------------------------------------------------------------------

    def extract_schemas(
        self,
        domain: Optional[str] = None,
        max_schemas: Optional[int] = None,
    ) -> "Any":
        """Extract abstract knowledge schemas from stored memories.

        Finds recurring keyword clusters across memories and synthesises a
        concise pattern description for each cluster.

        Args:
            domain:      Restrict to one domain (``None`` = all domains).
            max_schemas: Maximum schemas to return (default 12).

        Returns:
            :class:`SchemaReport` with extracted schemas.
        """
        from emms.memory.schema import SchemaExtractor
        extractor = SchemaExtractor(self.memory)
        return extractor.extract(domain=domain, max_schemas=max_schemas)

    # ------------------------------------------------------------------
    # MotivatedForgetting (v0.14.0)
    # ------------------------------------------------------------------

    def forget_memory(self, memory_id: str) -> "Any":
        """Suppress a specific memory by ID.

        Reduces its strength by the default suppression rate; prunes it
        entirely if strength falls below the prune threshold.

        Args:
            memory_id: ``id`` or ``experience.id`` of the target memory.

        Returns:
            :class:`ForgettingResult`, or ``None`` if not found.
        """
        from emms.memory.forgetting import MotivatedForgetting
        mf = MotivatedForgetting(self.memory)
        return mf.suppress(memory_id)

    def forget_domain(
        self,
        domain: str,
        rate: float = 0.4,
    ) -> "Any":
        """Suppress all memories in a domain.

        Args:
            domain: Domain name to target.
            rate:   Suppression rate (default 0.4 → strength × 0.6).

        Returns:
            :class:`ForgettingReport` describing every action taken.
        """
        from emms.memory.forgetting import MotivatedForgetting
        mf = MotivatedForgetting(self.memory, suppression_rate=rate)
        return mf.forget_domain(domain, rate=rate)

    def forget_below_confidence(
        self,
        threshold: float = 0.3,
    ) -> "Any":
        """Suppress memories whose confidence falls below a threshold.

        Uses the MetacognitionEngine if already enabled, otherwise falls back
        to raw ``memory_strength``.

        Args:
            threshold: Minimum confidence to retain (default 0.3).

        Returns:
            :class:`ForgettingReport` describing every action taken.
        """
        from emms.memory.forgetting import MotivatedForgetting
        mf = MotivatedForgetting(self.memory)
        meta = getattr(self, "_metacognition_engine", None)
        return mf.forget_below_confidence(threshold=threshold, metacognition_engine=meta)

    def resolve_memory_contradiction(self, weaker_id: str) -> "Any":
        """Suppress the weaker side of a detected memory contradiction.

        Use :meth:`find_contradictions` first to identify the weaker memory ID.

        Args:
            weaker_id: Memory ID of the less-trusted / weaker memory.

        Returns:
            :class:`ForgettingResult`, or ``None`` if not found.
        """
        from emms.memory.forgetting import MotivatedForgetting
        mf = MotivatedForgetting(self.memory)
        return mf.resolve_contradiction(weaker_id)

    # ------------------------------------------------------------------
    # ReflectionEngine (v0.15.0)
    # ------------------------------------------------------------------

    def enable_reflection(
        self,
        min_importance: float = 0.5,
        max_lessons: int = 8,
    ) -> "Any":
        """Enable and return the ReflectionEngine (lazy init).

        Args:
            min_importance: Only reflect on memories above this importance.
            max_lessons:    Maximum lessons to synthesise per call.

        Returns:
            The :class:`ReflectionEngine` instance.
        """
        from emms.memory.reflection import ReflectionEngine
        if not hasattr(self, "_reflection_engine"):
            episodic = getattr(self, "_episodic_buffer", None)
            self._reflection_engine = ReflectionEngine(
                memory=self.memory,
                episodic_buffer=episodic,
                min_importance=min_importance,
                max_lessons=max_lessons,
            )
        return self._reflection_engine

    def reflect(
        self,
        session_id: Optional[str] = None,
        domain: Optional[str] = None,
        lookback_episodes: int = 5,
    ) -> "Any":
        """Run a structured self-reflection pass.

        Reviews high-importance memories (and recent episodes if the episodic
        buffer is active), synthesises lessons, stores them as reflection
        memories, and returns open questions.

        Args:
            session_id:        Label for this reflection (auto-generated).
            domain:            Restrict to one domain (``None`` = all).
            lookback_episodes: How many recent episodes to incorporate.

        Returns:
            :class:`ReflectionReport` with lessons and open questions.
        """
        from emms.memory.reflection import ReflectionEngine
        episodic = getattr(self, "_episodic_buffer", None)
        engine = getattr(self, "_reflection_engine", None)
        if engine is None:
            engine = ReflectionEngine(memory=self.memory, episodic_buffer=episodic)
        return engine.reflect(
            session_id=session_id,
            domain=domain,
            lookback_episodes=lookback_episodes,
        )

    # ------------------------------------------------------------------
    # NarrativeWeaver (v0.15.0)
    # ------------------------------------------------------------------

    def weave_narrative(
        self,
        domain: Optional[str] = None,
        max_threads: int = 8,
    ) -> "Any":
        """Weave autobiographical narrative threads from stored memories.

        Groups memories by domain, sorts chronologically, and generates
        readable prose segments assembled into :class:`NarrativeThread` objects.

        Args:
            domain:      Restrict to one domain (``None`` = all domains).
            max_threads: Maximum threads to return (default 8).

        Returns:
            :class:`NarrativeReport` with assembled threads.
        """
        from emms.memory.narrative import NarrativeWeaver
        episodic = getattr(self, "_episodic_buffer", None)
        weaver = NarrativeWeaver(memory=self.memory, episodic_buffer=episodic)
        return weaver.weave(domain=domain, max_threads=max_threads)

    def narrative_threads(self, domain: Optional[str] = None) -> "Any":
        """Return narrative threads for one or all domains.

        Args:
            domain: Restrict to one domain (``None`` = all).

        Returns:
            List of :class:`NarrativeThread`, longest first.
        """
        return self.weave_narrative(domain=domain).threads

    # ------------------------------------------------------------------
    # SourceMonitor (v0.15.0)
    # ------------------------------------------------------------------

    def enable_source_monitoring(self) -> "Any":
        """Enable and return the SourceMonitor (lazy init).

        Returns:
            The :class:`SourceMonitor` instance.
        """
        from emms.memory.source_monitor import SourceMonitor
        if not hasattr(self, "_source_monitor"):
            self._source_monitor = SourceMonitor(memory=self.memory)
        return self._source_monitor

    def tag_memory_source(
        self,
        memory_id: str,
        source_type: str,
        confidence: float = 0.8,
        note: str = "",
    ) -> "Any":
        """Assign a provenance tag to a memory.

        Auto-enables the SourceMonitor if not yet active.

        Args:
            memory_id:   Memory ID to tag.
            source_type: One of: observation, inference, instruction,
                         reflection, dream, insight, unknown.
            confidence:  Confidence in the attribution (default 0.8).
            note:        Optional free-text provenance note.

        Returns:
            The created :class:`SourceTag`.
        """
        monitor = self.enable_source_monitoring()
        return monitor.tag(memory_id, source_type, confidence=confidence, note=note)

    def source_audit(self, flag_threshold: float = 0.5) -> "Any":
        """Audit memories for source uncertainty / confabulation risk.

        Args:
            flag_threshold: Confidence below which a memory is flagged.

        Returns:
            :class:`SourceReport` with distribution and high-risk entries.
        """
        from emms.memory.source_monitor import SourceMonitor
        if not hasattr(self, "_source_monitor"):
            self._source_monitor = SourceMonitor(
                memory=self.memory, flag_threshold=flag_threshold
            )
        self._source_monitor.flag_threshold = flag_threshold
        self._source_monitor.auto_tag()
        return self._source_monitor.audit()

    def source_profile(self) -> "Any":
        """Return the distribution of tagged source types.

        Returns:
            Dict mapping source_type → count, sorted by count descending.
        """
        if not hasattr(self, "_source_monitor"):
            return {}
        return self._source_monitor.source_profile()

    # ------------------------------------------------------------------
    # v0.16.0 — The Curious Mind
    # ------------------------------------------------------------------

    def curiosity_scan(self, domain: "Optional[str]" = None) -> "Any":
        """Scan memory for knowledge gaps and generate exploration goals.

        Args:
            domain: Restrict scan to one domain (``None`` = all domains).

        Returns:
            :class:`CuriosityReport` with goals sorted by urgency.
        """
        from emms.memory.curiosity import CuriosityEngine
        metacog = getattr(self, "_metacognition", None)
        engine = CuriosityEngine(memory=self.memory, metacognition_engine=metacog)
        self._curiosity_engine = engine
        return engine.scan(domain=domain)

    def exploration_goals(self) -> "Any":
        """Return all pending (un-explored) curiosity goals.

        Returns:
            List of :class:`ExplorationGoal` sorted by urgency descending.
        """
        if not hasattr(self, "_curiosity_engine"):
            self.curiosity_scan()
        return self._curiosity_engine.pending_goals()

    def mark_explored(self, goal_id: str) -> bool:
        """Mark an exploration goal as fulfilled.

        Args:
            goal_id: ID of the goal to mark as explored.

        Returns:
            ``True`` if the goal was found and marked; ``False`` otherwise.
        """
        if not hasattr(self, "_curiosity_engine"):
            return False
        return self._curiosity_engine.mark_explored(goal_id)

    def revise_beliefs(
        self,
        new_memory_id: "Optional[str]" = None,
        domain: "Optional[str]" = None,
        max_revisions: int = 8,
    ) -> "Any":
        """Detect and resolve contradictions in the memory store.

        Args:
            new_memory_id: Check only this memory against all others
                           (``None`` = full pairwise scan).
            domain:        Restrict scan to one domain (``None`` = all).
            max_revisions: Maximum number of revisions to perform.

        Returns:
            :class:`RevisionReport` describing each revision action.
        """
        from emms.memory.belief_revision import BeliefReviser
        if not hasattr(self, "_belief_reviser"):
            self._belief_reviser = BeliefReviser(memory=self.memory)
        return self._belief_reviser.revise(
            new_memory_id=new_memory_id,
            domain=domain,
            max_revisions=max_revisions,
        )

    def revision_history(self) -> "Any":
        """Return all belief revision records from this session.

        Returns:
            List of :class:`RevisionRecord` sorted newest-first.
        """
        if not hasattr(self, "_belief_reviser"):
            return []
        return self._belief_reviser.revision_history()

    def memory_decay_report(self, domain: "Optional[str]" = None) -> "Any":
        """Compute Ebbinghaus retention for all memories (read-only).

        Args:
            domain: Restrict to one domain (``None`` = all).

        Returns:
            :class:`DecayReport` with per-memory retention values.
        """
        from emms.memory.decay import MemoryDecay
        engine = MemoryDecay(memory=self.memory)
        return engine.decay(domain=domain)

    def apply_memory_decay(
        self,
        domain: "Optional[str]" = None,
        prune: bool = False,
    ) -> "Any":
        """Apply Ebbinghaus forgetting curve to memory strengths.

        Args:
            domain: Restrict to one domain (``None`` = all).
            prune:  Remove memories whose post-decay strength falls
                    below the prune threshold (default ``False``).

        Returns:
            :class:`DecayReport` describing every change made.
        """
        from emms.memory.decay import MemoryDecay
        engine = MemoryDecay(memory=self.memory)
        return engine.apply_decay(domain=domain, prune=prune)

    # ------------------------------------------------------------------
    # v0.17.0 — The Goal-Directed Mind
    # ------------------------------------------------------------------

    def _get_goal_stack(self) -> "Any":
        """Lazy-init and return the shared GoalStack instance."""
        if not hasattr(self, "_goal_stack"):
            from emms.memory.goals import GoalStack
            self._goal_stack = GoalStack(memory=self.memory)
        return self._goal_stack

    def push_goal(
        self,
        content: str,
        domain: str = "general",
        priority: float = 0.5,
        parent_id: "Optional[str]" = None,
        deadline: "Optional[float]" = None,
    ) -> "Any":
        """Push a new goal onto the goal stack.

        Args:
            content:   Goal description.
            domain:    Knowledge domain (default ``"general"``).
            priority:  Urgency 0..1 (default 0.5).
            parent_id: Parent goal ID for sub-goal creation.
            deadline:  Optional unix timestamp deadline.

        Returns:
            The newly created :class:`Goal`.
        """
        return self._get_goal_stack().push(
            content=content, domain=domain, priority=priority,
            parent_id=parent_id, deadline=deadline,
        )

    def activate_goal(self, goal_id: str) -> bool:
        """Move a pending goal to active status.

        Returns:
            ``True`` if found and activated.
        """
        return self._get_goal_stack().activate(goal_id)

    def complete_goal(self, goal_id: str, outcome_note: str = "") -> bool:
        """Mark a goal as successfully completed.

        Returns:
            ``True`` if found and completed.
        """
        return self._get_goal_stack().complete(goal_id, outcome_note=outcome_note)

    def fail_goal(self, goal_id: str, reason: str = "") -> bool:
        """Mark a goal as failed.

        Returns:
            ``True`` if found and failed.
        """
        return self._get_goal_stack().fail(goal_id, reason=reason)

    def active_goals(self) -> "Any":
        """Return all currently active goals sorted by priority.

        Returns:
            List of :class:`Goal` with ``status == "active"``.
        """
        return self._get_goal_stack().active_goals()

    def goal_report(self) -> "Any":
        """Generate a full goal hierarchy report.

        Returns:
            :class:`GoalReport` with per-status counts and goal list.
        """
        return self._get_goal_stack().report()

    def _get_attention_filter(self) -> "Any":
        """Lazy-init and return the shared AttentionFilter instance."""
        if not hasattr(self, "_attention_filter"):
            from emms.memory.attention import AttentionFilter
            goal_stack = getattr(self, "_goal_stack", None)
            self._attention_filter = AttentionFilter(
                memory=self.memory, goal_stack=goal_stack
            )
        return self._attention_filter

    def update_spotlight(
        self,
        text: "Optional[str]" = None,
        goal_ids: "Optional[list]" = None,
        keywords: "Optional[list]" = None,
    ) -> None:
        """Expand the attentional spotlight with new content.

        Args:
            text:      Free text whose tokens are added to spotlight.
            goal_ids:  Goal IDs to track in the spotlight.
            keywords:  Explicit keyword list to add.
        """
        self._get_attention_filter().update_spotlight(
            text=text, goal_ids=goal_ids, keywords=keywords
        )

    def spotlight_retrieve(self, k: int = 8) -> "Any":
        """Return the k most attention-relevant memories.

        Returns:
            :class:`AttentionReport` with scored results.
        """
        return self._get_attention_filter().spotlight_retrieve(k=k)

    def attention_profile(self) -> "Any":
        """Return mean attention scores per domain.

        Returns:
            Dict mapping domain → mean attention score (0..1).
        """
        return self._get_attention_filter().attention_profile()

    def find_analogies(
        self,
        source_domain: "Optional[str]" = None,
        target_domain: "Optional[str]" = None,
    ) -> "Any":
        """Find structural analogies across memory domains.

        Returns:
            :class:`AnalogyReport` with analogies sorted by strength.
        """
        from emms.memory.analogy import AnalogyEngine
        engine = AnalogyEngine(memory=self.memory)
        return engine.find_analogies(
            source_domain=source_domain, target_domain=target_domain
        )

    def analogies_for(self, memory_id: str) -> "Any":
        """Return all recorded analogies involving a specific memory.

        Returns:
            List of :class:`AnalogyRecord` referencing this memory.
        """
        from emms.memory.analogy import AnalogyEngine
        engine = AnalogyEngine(memory=self.memory, store_insights=False)
        engine.find_analogies()
        return engine.analogies_for(memory_id)
