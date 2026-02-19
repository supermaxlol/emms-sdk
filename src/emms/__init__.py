"""EMMS — Enhanced Memory Management System for AI Agents.

v0.9.0: Scalability & federation layer
- CompactionIndex: O(1) dual-hash lookup by id, experience_id, content_hash; auto-wired in EMMS.store()
- GraphCommunityDetection: Label Propagation Algorithm for topic cluster discovery; modularity Q
- ExperienceReplay: Prioritized experience replay (PER) with IS weights and alias sampling
- MemoryFederation: Multi-agent snapshot merge with 3 conflict policies and content-hash dedup
- MemoryQueryPlanner: Heuristic query decomposition + parallel HybridRetriever + cross-boost merge
- EMMS facade: get_memory_by_id, get_memory_by_experience_id, find_memories_by_content,
  rebuild_index, index_stats, graph_communities, graph_community_for_entity,
  enable_experience_replay, replay_sample, replay_context, replay_top,
  merge_from, federation_export, plan_retrieve, plan_retrieve_simple
- MCP tools (5 new): emms_index_lookup, emms_graph_communities, emms_replay_sample,
  emms_merge_from, emms_plan_retrieve (32 total)
- CLI commands (7 new): index-lookup, index-stats, graph-communities, replay, replay-top,
  merge-from, plan-retrieve (35 total)

v0.8.0: Retrieval intelligence layer
- HybridRetriever: BM25 + embedding cosine fused via Reciprocal Rank Fusion (RRF)
- MemoryTimeline: chronological reconstruction with gap detection and density histograms
- AdaptiveRetriever: Thompson Sampling Beta-Bernoulli bandit over retrieval strategies
- MemoryBudget: token-budget-aware tiered eviction with dry_run and composite scoring
- MultiHopGraphReasoner: multi-hop BFS graph reasoning with path-strength scoring and DOT export
- EMMS facade: hybrid_retrieve, build_timeline, enable_adaptive_retrieval, adaptive_retrieve,
  adaptive_feedback, get_retrieval_beliefs, memory_token_footprint, enforce_memory_budget, multihop_query
- MCP tools (5 new): emms_hybrid_retrieve, emms_build_timeline, emms_adaptive_retrieve,
  emms_enforce_budget, emms_multihop_query (27 total)
- CLI commands (7 new): hybrid-retrieve, timeline, adaptive-retrieve, retrieval-beliefs,
  budget-status, budget-enforce, multihop (28 total)

v0.7.0: Intelligence consolidation layer
- MemoryDiff: session-to-session memory snapshot comparison (added/removed/strengthened/weakened/superseded)
- MemoryCluster: pure-Python k-means++ with TF-IDF fallback; auto_k via elbow method
- ConversationBuffer: sliding-window conversation history with extractive/LLM summarisation
- stream_retrieve: async generator yielding RetrievalResult tier-by-tier (astream_retrieve on EMMS)
- LLMConsolidator: union-find cluster synthesis via LLM + extractive fallback; auto_consolidate()
- EMMS facade: astream_retrieve, diff_since, cluster_memories, llm_consolidate, build_conversation_context
- MCP tools (2 new): emms_cluster_memories, emms_llm_consolidate (24 total)
- CLI command (1 new): emms diff (21 total)

v0.6.0: Advanced intelligence layer
- ImportanceClassifier: auto-score importance from 6 content signals (entity density,
  novelty, emotional weight, length, keywords, structure) — no ML dependency
- RAGContextBuilder: token-budget-aware context packing (4 formats: markdown/xml/json/plain)
- SemanticDeduplicator: cosine+lexical near-duplicate detection + intelligent merge
- MemoryScheduler: composable async background maintenance with 5 built-in jobs
  (consolidation, ttl_purge, deduplication, pattern_detection, srs_review)
- SpacedRepetitionSystem: SM-2 algorithm; srs_enroll/record_review/get_due_items
- Graph Visualization: to_dot() (Graphviz) + to_d3() (D3.js JSON) on GraphMemory
- SRS fields on MemoryItem: srs_enrolled, srs_next_review, srs_interval_days
- MemoryConfig: dedup_cosine_threshold, dedup_lexical_threshold, enable_auto_dedup
- EMMS facade: build_rag_context, deduplicate, srs_enroll/enroll_all/record_review/due,
  start_scheduler/stop_scheduler, export_graph_dot/d3, score_importance

v0.5.1: GitHub Copilot + LangMem inspired extensions
- ProceduralMemory: 5th memory tier — evolving behavioral rules for system prompt injection
- citations field on Experience: GitHub Copilot citation-based validation
- validate_citations(): strengthen cited memories on retrieval
- search_by_file(): find memories referencing specific file paths
- GraphMemory save/load: graph state persists alongside hierarchical memory
- TTL-aware filtering: expired/superseded memories excluded from retrieval
- Patch update_mode: update existing memories in-place with conflict archival (superseded_by)
- Debounced consolidation in SessionManager: auto-consolidate after N stores
- ImportanceStrategy: LangMem-inspired importance+strength retrieval signal
- strategy_scores + explanation on RetrievalResult: per-strategy scoring breakdown
- EMCPServer: MCP adapter exposing EMMS as AI tool server
- CLI: `emms store/retrieve/compact/search-file/stats/save/load/procedures`
- Procedural memory save/load wired into EMMS.save()/EMMS.load()

v0.5.0: claude-mem inspired extensions
- SessionManager: persistent session log (JSONL), auto session_id injection
- ObsType + ConceptTag: semantic observation classification (6 types, 7 tags)
- Progressive disclosure retrieval: search_compact / get_full / get_timeline
- EnsembleRetriever.from_balanced() preset (60/20/10/10 weighting incl. ImportanceStrategy)
- EnsembleRetriever.from_identity() preset (6-strategy identity workload)
- ChromaSemanticStrategy: ChromaDB-backed high-fidelity semantic retrieval
- Endless Mode: biomimetic O(N²)→O(N) real-time compression for long sessions
- ToolObserver: PostToolUse hook → Experience converter
- JSONL export/import on HierarchicalMemory
- SessionSummary: structured per-session narrative (request/learned/completed/…)
- facts, files_read, files_modified, title, subtitle on Experience
- token_estimate on CompactResult
- observe_prompt() on ToolObserver
- generate_context_injection() on SessionManager

v0.4.0: The Ultimate System
- EventBus for inter-component communication
- GraphMemory for entity-relationship extraction
- Multi-strategy ensemble retrieval (5 strategies)
- Memory persistence (save/load full state)
- VectorIndex for fast batch cosine similarity
- Enhanced consciousness modules (traits, autobiographical, milestones)
- Advanced episode detection (spectral, conductance, multi-algorithm)
- Pattern detection in memory compression
- LLM integration layer (Claude, GPT, Ollama)
- Real-time data pipeline
- Background consolidation
"""

from emms.core.models import (
    Experience, MemoryItem, MemoryConfig,
    CompactResult, SessionSummary,
    ObsType, ConceptTag,
)
from emms.core.embeddings import HashEmbedder, cosine_similarity
from emms.core.events import EventBus
from emms.memory.hierarchical import HierarchicalMemory, VectorIndex
from emms.memory.compression import MemoryCompressor, CompressedMemory, PatternDetector, SemanticDeduplicator
from emms.memory.graph import GraphMemory, Entity, Relationship
from emms.memory.procedural import ProceduralMemory, ProcedureEntry
from emms.memory.spaced_repetition import SpacedRepetitionSystem, SRSCard
from emms.core.importance import ImportanceClassifier
from emms.context.rag_builder import RAGContextBuilder, ContextBlock
from emms.scheduler import MemoryScheduler, ScheduledJob
from emms.context.token_manager import TokenContextManager
from emms.identity.consciousness import (
    ContinuousNarrator,
    MeaningMaker,
    TemporalIntegrator,
    EgoBoundaryTracker,
    MetaCognitiveMonitor,
)
from emms.retrieval.strategies import (
    EnsembleRetriever,
    SemanticStrategy,
    TemporalStrategy,
    EmotionalStrategy,
    GraphStrategy,
    DomainStrategy,
    ChromaSemanticStrategy,
    ImportanceStrategy,
)
from emms.sessions.manager import SessionManager
from emms.sessions.conversation import ConversationBuffer, ConversationTurn, ConversationChunk
from emms.hooks.tool_observer import ToolObserver
from emms.analytics.memory_analytics import MemoryAnalytics
from emms.adapters.mcp_server import EMCPServer
from emms.memory.diff import MemoryDiff, DiffResult, ItemSnapshot
from emms.memory.clustering import MemoryClustering, MemoryCluster
from emms.llm.consolidator import LLMConsolidator, ConsolidationResult
from emms.retrieval.hybrid import HybridRetriever, HybridSearchResult
from emms.retrieval.adaptive import AdaptiveRetriever, StrategyBelief
from emms.analytics.timeline import MemoryTimeline, TimelineResult, TimelineEvent, TemporalGap, DensityBucket
from emms.context.budget import MemoryBudget, BudgetReport, EvictionCandidate, EvictionPolicy
from emms.memory.multihop import MultiHopGraphReasoner, MultiHopResult, HopPath, ReachableEntity
from emms.storage.index import CompactionIndex
from emms.memory.communities import GraphCommunityDetector, Community, CommunityResult
from emms.memory.replay import ExperienceReplay, ReplayEntry, ReplayBatch
from emms.storage.federation import MemoryFederation, FederationResult, ConflictEntry, ConflictPolicy
from emms.retrieval.planner import MemoryQueryPlanner, QueryPlan, SubQueryResult
from emms.emms import EMMS
from emms.prompts.identity import IdentityPromptBuilder, PROVIDER_RECOMMENDATIONS

__version__ = "0.9.0"
__all__ = [
    # Core
    "EMMS",
    "Experience",
    "MemoryItem",
    "MemoryConfig",
    "CompactResult",
    "SessionSummary",
    "ObsType",
    "ConceptTag",
    "EventBus",
    # Memory
    "HierarchicalMemory",
    "VectorIndex",
    "GraphMemory",
    "Entity",
    "Relationship",
    "ProceduralMemory",
    "ProcedureEntry",
    # Compression & patterns
    "MemoryCompressor",
    "CompressedMemory",
    "PatternDetector",
    # Retrieval
    "EnsembleRetriever",
    "SemanticStrategy",
    "TemporalStrategy",
    "EmotionalStrategy",
    "GraphStrategy",
    "DomainStrategy",
    "ChromaSemanticStrategy",
    "ImportanceStrategy",
    # Embeddings
    "HashEmbedder",
    "cosine_similarity",
    # Context
    "TokenContextManager",
    # Consciousness
    "ContinuousNarrator",
    "MeaningMaker",
    "TemporalIntegrator",
    "EgoBoundaryTracker",
    "MetaCognitiveMonitor",
    # Sessions
    "SessionManager",
    # Hooks
    "ToolObserver",
    # Analytics
    "MemoryAnalytics",
    # Adapters
    "EMCPServer",
    # Prompts (identity adoption)
    "IdentityPromptBuilder",
    "PROVIDER_RECOMMENDATIONS",
    # v0.6.0 additions
    "SemanticDeduplicator",
    "SpacedRepetitionSystem",
    "SRSCard",
    "ImportanceClassifier",
    "RAGContextBuilder",
    "ContextBlock",
    "MemoryScheduler",
    "ScheduledJob",
    # v0.7.0 additions
    "MemoryDiff",
    "DiffResult",
    "ItemSnapshot",
    "MemoryClustering",
    "MemoryCluster",
    "ConversationBuffer",
    "ConversationTurn",
    "ConversationChunk",
    "LLMConsolidator",
    "ConsolidationResult",
    # v0.8.0 additions
    "HybridRetriever",
    "HybridSearchResult",
    "AdaptiveRetriever",
    "StrategyBelief",
    "MemoryTimeline",
    "TimelineResult",
    "TimelineEvent",
    "TemporalGap",
    "DensityBucket",
    "MemoryBudget",
    "BudgetReport",
    "EvictionCandidate",
    "EvictionPolicy",
    "MultiHopGraphReasoner",
    "MultiHopResult",
    "HopPath",
    "ReachableEntity",
    # v0.9.0 additions
    "CompactionIndex",
    "GraphCommunityDetector",
    "Community",
    "CommunityResult",
    "ExperienceReplay",
    "ReplayEntry",
    "ReplayBatch",
    "MemoryFederation",
    "FederationResult",
    "ConflictEntry",
    "ConflictPolicy",
    "MemoryQueryPlanner",
    "QueryPlan",
    "SubQueryResult",
]
