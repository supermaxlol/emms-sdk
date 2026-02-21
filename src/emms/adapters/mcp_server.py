"""MCP server adapter for EMMS.

Exposes EMMS memory operations as Model Context Protocol (MCP) tool definitions
that can be served by any MCP-compatible server runtime (e.g. FastMCP, mcp-python).

The adapter is intentionally thin — it converts EMMS calls into the tool-call
and tool-result schema that MCP expects, without coupling to any particular
server framework. Wire it up like this::

    from emms import EMMS
    from emms.adapters.mcp_server import EMCPServer

    emms = EMMS()
    server = EMCPServer(emms)

    # If you have FastMCP installed:
    #   from mcp.server.fastmcp import FastMCP
    #   mcp = FastMCP("emms")
    #   server.register(mcp)
    #   mcp.run()

    # Or call tool handlers directly in tests:
    result = server.handle("emms_store", {"content": "hello", "domain": "test"})

Exposed tools
-------------
``emms_store``
    Store an experience into memory.

``emms_retrieve``
    Retrieve memories matching a natural-language query.

``emms_search_compact``
    Progressive-disclosure retrieval — returns compact index (50–80 tokens/result).

``emms_search_by_file``
    Find memories that reference a specific file path.

``emms_get_stats``
    Return EMMS system stats (tier sizes, uptime, throughput).

``emms_get_procedures``
    Return the current procedural-memory rules formatted as a system prompt block.

``emms_add_procedure``
    Add a behavioral rule to procedural memory.

``emms_save``
    Persist memory state to disk.

``emms_load``
    Load memory state from disk.

``emms_build_rag_context``
    Build a token-budget-aware context document from retrieved memories.

``emms_deduplicate``
    Scan long-term memories for near-duplicates and archive weaker copies.

``emms_srs_enroll``
    Enrol a memory in the Spaced Repetition System.

``emms_srs_record_review``
    Record an SRS review outcome (quality 0–5).

``emms_srs_due``
    Return memory IDs due for SRS review.

``emms_export_graph_dot``
    Export the knowledge graph as a Graphviz DOT string.

``emms_export_graph_d3``
    Export the knowledge graph as a D3.js force-graph JSON dict.

``emms_cluster_memories``
    Cluster memory items into semantic groups using k-means or TF-IDF.

``emms_llm_consolidate``
    Scan memory for similar items and synthesise each cluster.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tool schema constants (JSON Schema compatible)
_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "emms_store",
        "description": "Store an experience into EMMS memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The experience text to memorise."},
                "domain": {"type": "string", "default": "general", "description": "Domain / topic area."},
                "importance": {"type": "number", "default": 0.5, "description": "0 (low) … 1 (high)."},
                "title": {"type": "string", "description": "Short title ≤10 words."},
                "facts": {"type": "array", "items": {"type": "string"}, "description": "Discrete factual bullet points."},
                "files_read": {"type": "array", "items": {"type": "string"}, "description": "Files read during this event."},
                "files_modified": {"type": "array", "items": {"type": "string"}, "description": "Files created or edited."},
                "citations": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs this experience validates."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "emms_retrieve",
        "description": "Retrieve memories from EMMS matching a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_search_compact",
        "description": "Progressive-disclosure search — returns compact index (title + first fact, ~50–80 tokens/result).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_search_by_file",
        "description": "Find memories that reference a specific file path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path (exact or partial) to search for."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "emms_get_stats",
        "description": "Return EMMS system statistics (tier sizes, uptime, throughput).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "emms_get_procedures",
        "description": "Return current behavioral rules as a system prompt block.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Filter rules to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_add_procedure",
        "description": "Add a behavioral rule to procedural memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string"},
                "domain": {"type": "string", "default": "general"},
                "importance": {"type": "number", "default": 0.5},
            },
            "required": ["rule"],
        },
    },
    {
        "name": "emms_retrieve_filtered",
        "description": "Retrieve memories with structured filters (namespace, obs_type, domain, time range, confidence).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
                "namespace": {"type": "string"},
                "domain": {"type": "string"},
                "session_id": {"type": "string"},
                "since": {"type": "number", "description": "Unix timestamp lower bound"},
                "until": {"type": "number", "description": "Unix timestamp upper bound"},
                "min_confidence": {"type": "number", "description": "Minimum confidence threshold 0–1"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_upvote",
        "description": "Positive feedback: strengthen a memory (it was useful).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "boost": {"type": "number", "default": 0.1},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "emms_downvote",
        "description": "Negative feedback: weaken a memory (it was irrelevant).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "decay": {"type": "number", "default": 0.2},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "emms_export_markdown",
        "description": "Export memories as a human-readable Markdown document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "emms_save",
        "description": "Persist EMMS memory state to disk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (e.g. ~/.emms/memory.json)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "emms_load",
        "description": "Load EMMS memory state from disk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "emms_build_rag_context",
        "description": "Retrieve memories and build a token-budget-aware context document for RAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20},
                "token_budget": {"type": "integer", "default": 4000, "description": "Max context tokens."},
                "fmt": {"type": "string", "default": "markdown", "description": "markdown|xml|json|plain"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_deduplicate",
        "description": "Scan long-term memories for near-duplicates and archive weaker copies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cosine_threshold": {"type": "number", "default": 0.92},
                "lexical_threshold": {"type": "number", "default": 0.85},
            },
        },
    },
    {
        "name": "emms_srs_enroll",
        "description": "Enrol a memory in the Spaced Repetition System review schedule.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "emms_srs_record_review",
        "description": "Record an SRS review outcome for a memory (quality 0-5).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "quality": {"type": "integer", "description": "0=blackout … 5=perfect recall"},
            },
            "required": ["memory_id", "quality"],
        },
    },
    {
        "name": "emms_srs_due",
        "description": "Return memory IDs due for SRS review, most-overdue first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_items": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "emms_export_graph_dot",
        "description": "Export the EMMS knowledge graph as a Graphviz DOT string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "default": "EMMS Knowledge Graph"},
                "max_nodes": {"type": "integer", "default": 100},
                "min_importance": {"type": "number", "default": 0.0},
            },
        },
    },
    {
        "name": "emms_export_graph_d3",
        "description": "Export the EMMS knowledge graph as a D3.js force-graph JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_nodes": {"type": "integer", "default": 200},
                "min_importance": {"type": "number", "default": 0.0},
            },
        },
    },
    # v0.7.0 tools
    {
        "name": "emms_cluster_memories",
        "description": "Cluster memory items into semantic groups using k-means or TF-IDF.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "description": "Number of clusters (required unless auto_k=true)."},
                "auto_k": {"type": "boolean", "default": False, "description": "Auto-select k via elbow method."},
                "tier": {"type": "string", "default": "long_term", "description": "Memory tier to cluster."},
                "k_min": {"type": "integer", "default": 2},
                "k_max": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "emms_llm_consolidate",
        "description": "Scan memory for similar items and synthesise each cluster into a higher-level memory. Uses extractive fallback when no LLM is wired.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.7, "description": "Similarity threshold (0–1) to link items."},
                "tier": {"type": "string", "default": "long_term", "description": "Memory tier to scan."},
                "max_clusters": {"type": "integer", "default": 20},
            },
        },
    },
    # v0.8.0 tools
    {
        "name": "emms_hybrid_retrieve",
        "description": "Hybrid BM25 + embedding retrieval fused via Reciprocal Rank Fusion (RRF). Combines lexical and semantic signals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "max_results": {"type": "integer", "default": 10},
                "rrf_k": {"type": "number", "default": 60.0, "description": "RRF smoothing constant."},
                "min_score": {"type": "number", "default": 0.0, "description": "Minimum RRF score to include."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_build_timeline",
        "description": "Build a chronological memory timeline with gap detection and storage density histogram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Filter to a single domain (omit for all)."},
                "since": {"type": "number", "description": "Unix timestamp lower bound."},
                "until": {"type": "number", "description": "Unix timestamp upper bound."},
                "gap_threshold_seconds": {"type": "number", "default": 300.0},
                "bucket_size_seconds": {"type": "number", "default": 3600.0},
            },
        },
    },
    {
        "name": "emms_adaptive_retrieve",
        "description": "Retrieve using a Thompson Sampling multi-armed bandit that learns which retrieval strategy works best.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
                "explore": {"type": "boolean", "default": True, "description": "True = Thompson Sampling; False = exploit best arm."},
                "feedback": {"type": "number", "description": "Optional reward (0–1) for the previous retrieval."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_enforce_budget",
        "description": "Enforce a token budget by evicting low-value memories. Supports dry_run mode to preview evictions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {"type": "integer", "default": 100000, "description": "Token budget ceiling."},
                "dry_run": {"type": "boolean", "default": False},
                "policy": {"type": "string", "default": "composite", "description": "Eviction policy: composite / lru / lfu / importance / strength."},
                "importance_threshold": {"type": "number", "default": 0.8, "description": "Memories at or above this importance are protected."},
            },
        },
    },
    {
        "name": "emms_multihop_query",
        "description": "Multi-hop BFS graph reasoning from a seed entity — discovers indirect connections up to N hops away.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seed": {"type": "string", "description": "Seed entity name (case-insensitive)."},
                "max_hops": {"type": "integer", "default": 3},
                "max_results": {"type": "integer", "default": 20},
                "min_strength": {"type": "number", "default": 0.0},
                "include_dot": {"type": "boolean", "default": False, "description": "Include Graphviz DOT export in result."},
            },
            "required": ["seed"],
        },
    },
    # v0.9.0 tools
    {
        "name": "emms_index_lookup",
        "description": "O(1) CompactionIndex lookup: find a MemoryItem by memory_id, experience_id, or content hash.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory item id."},
                "experience_id": {"type": "string", "description": "Experience id."},
                "content": {"type": "string", "description": "Content snippet for hash-based lookup."},
                "action": {
                    "type": "string",
                    "default": "lookup",
                    "description": "lookup | stats | rebuild",
                },
            },
        },
    },
    {
        "name": "emms_graph_communities",
        "description": "Detect communities (topic clusters) in the knowledge graph using Label Propagation Algorithm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_iter": {"type": "integer", "default": 100},
                "min_community_size": {"type": "integer", "default": 1},
                "entity_name": {"type": "string", "description": "If set, return only the community containing this entity."},
                "export_markdown": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "emms_replay_sample",
        "description": "Sample a mini-batch of memories by prioritized experience replay (PER) with IS weights.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 8, "description": "Batch size."},
                "beta": {"type": "number", "default": 0.4, "description": "IS correction exponent."},
                "top_k": {"type": "boolean", "default": False, "description": "If True, return deterministic top-k by priority."},
            },
        },
    },
    {
        "name": "emms_merge_from",
        "description": "Merge memories from another EMMS snapshot (provided as a JSON list of MemoryItem dicts) into this instance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "policy": {
                    "type": "string",
                    "default": "newest_wins",
                    "description": "Conflict policy: local_wins | newest_wins | importance_wins",
                },
                "namespace_prefix": {"type": "string", "description": "Prepend prefix/ to incoming ids to avoid collisions."},
                "dry_run": {"type": "boolean", "default": False, "description": "If True, count what would be merged without merging."},
            },
        },
    },
    {
        "name": "emms_plan_retrieve",
        "description": "Decompose a complex query into sub-queries, retrieve each independently, cross-boost items appearing in multiple sub-queries, and return a merged ranked list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20},
                "max_results_per_sub": {"type": "integer", "default": 10},
                "cross_boost": {"type": "number", "default": 0.10, "description": "Score increment per additional sub-query hit."},
            },
            "required": ["query"],
        },
    },
    # v0.10.0 tools
    {
        "name": "emms_reconsolidate",
        "description": "Reconsolidate a recalled memory — strengthen/weaken it and optionally drift its emotional valence toward the current context valence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "MemoryItem ID to reconsolidate."},
                "context_valence": {"type": "number", "description": "Emotional valence of recall context (-1..+1). Omit to skip valence drift."},
                "reinforce": {"type": "boolean", "default": True, "description": "True=strengthen (confirming recall); False=weaken."},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "emms_batch_reconsolidate",
        "description": "Reconsolidate a batch of recently-recalled memories. Pass all retrieved IDs after a retrieval round.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "List of MemoryItem IDs."},
                "context_valence": {"type": "number", "description": "Emotional valence of current context (-1..+1)."},
                "reinforce": {"type": "boolean", "default": True},
            },
            "required": ["memory_ids"],
        },
    },
    {
        "name": "emms_presence_metrics",
        "description": "Return current session presence metrics (presence score, attention budget, coherence trend, emotional arc).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "record_turn": {"type": "boolean", "default": False, "description": "If true, record a new turn before returning metrics."},
                "content": {"type": "string", "description": "Turn content (used when record_turn=true)."},
                "domain": {"type": "string", "default": "general"},
                "valence": {"type": "number", "default": 0.0},
                "intensity": {"type": "number", "default": 0.0},
            },
        },
    },
    {
        "name": "emms_affective_retrieve",
        "description": "Retrieve memories by emotional proximity. Find memories whose valence/intensity is closest to a target emotional state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": "", "description": "Optional semantic query to blend with emotional proximity."},
                "target_valence": {"type": "number", "description": "Target emotional valence (-1..+1)."},
                "target_intensity": {"type": "number", "description": "Target emotional intensity (0..1)."},
                "max_results": {"type": "integer", "default": 10},
                "semantic_blend": {"type": "number", "default": 0.4, "description": "Weight of semantic score [0,1]."},
            },
        },
    },
    {
        "name": "emms_emotional_landscape",
        "description": "Return a summary of the emotional distribution across all memories: mean/std valence & intensity, histograms, most positive/negative/intense memory IDs.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # v0.11.0 tools
    {
        "name": "emms_dream",
        "description": "Run a between-session dream consolidation pass: replay important memories, strengthen top-k, weaken neglected ones, prune below threshold, detect patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Label for the dream report."},
                "reinforce_top_k": {"type": "integer", "default": 20},
                "weaken_bottom_k": {"type": "integer", "default": 10},
                "prune_threshold": {"type": "number", "default": 0.05},
                "run_dedup": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "emms_capture_bridge",
        "description": "Capture unresolved high-importance threads and session state into a BridgeRecord for the next session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "closing_summary": {"type": "string", "description": "Optional summary of what this session accomplished."},
                "max_threads": {"type": "integer", "default": 5},
            },
        },
    },
    {
        "name": "emms_inject_bridge",
        "description": "Generate a prompt-ready markdown context string from a previously captured BridgeRecord. Pass the captured bridge data as 'bridge_json'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bridge_json": {"type": "string", "description": "JSON-serialized BridgeRecord from emms_capture_bridge."},
                "new_session_id": {"type": "string"},
            },
            "required": ["bridge_json"],
        },
    },
    {
        "name": "emms_anneal",
        "description": "Anneal the memory landscape after a session gap: weak memories decay faster, emotional valence stabilizes toward neutral, important survivors strengthened.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "last_session_at": {"type": "number", "description": "Unix timestamp of last session end. Omit to use default half-life."},
                "half_life_gap": {"type": "number", "default": 259200.0, "description": "Gap in seconds at which temperature=0.5 (default 3 days)."},
                "decay_rate": {"type": "number", "default": 0.03},
                "emotional_stabilization_rate": {"type": "number", "default": 0.08},
            },
        },
    },
    {
        "name": "emms_bridge_summary",
        "description": "Show a summary of the current session bridge state (open threads, emotional arc, presence at end).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # v0.12.0 tools
    {
        "name": "emms_build_association_graph",
        "description": "Build an explicit association graph over all stored memories. Edges are auto-detected: semantic (cosine similarity), temporal (co-stored within window), affective (valence proximity), domain (same domain). Returns graph statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "semantic_threshold": {"type": "number", "default": 0.5, "description": "Minimum cosine similarity for a semantic edge."},
                "temporal_window": {"type": "number", "default": 300.0, "description": "Max seconds between stored_at timestamps for a temporal edge."},
                "affective_tolerance": {"type": "number", "default": 0.3, "description": "Max |valence_a - valence_b| for an affective edge."},
            },
        },
    },
    {
        "name": "emms_spreading_activation",
        "description": "Run spreading activation from a list of seed memory IDs on the association graph. Returns activated memories sorted by activation score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seed_ids": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs to start activation from."},
                "decay": {"type": "number", "default": 0.5, "description": "Activation decay factor per hop."},
                "steps": {"type": "integer", "default": 3, "description": "Maximum hop depth."},
            },
            "required": ["seed_ids"],
        },
    },
    {
        "name": "emms_discover_insights",
        "description": "Find cross-domain memory bridges and generate insight memories. Walks the association graph for pairs from different domains with strong connections, synthesises new 'insight' memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Label for the insight report."},
                "max_insights": {"type": "integer", "default": 8, "description": "Maximum insight memories to generate."},
                "min_bridge_weight": {"type": "number", "default": 0.45, "description": "Minimum edge weight to qualify as a bridge."},
                "rebuild_graph": {"type": "boolean", "default": True, "description": "Rebuild the association graph before searching."},
            },
        },
    },
    {
        "name": "emms_associative_retrieve",
        "description": "Retrieve memories via spreading activation from seed memory IDs or a text query. Returns memories reached through associative paths, not just direct matches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text query to find seed memories (used if seed_ids not given)."},
                "seed_ids": {"type": "array", "items": {"type": "string"}, "description": "Explicit seed memory IDs (overrides query)."},
                "seed_count": {"type": "integer", "default": 3, "description": "Seeds to select from query."},
                "max_results": {"type": "integer", "default": 10},
                "steps": {"type": "integer", "default": 3, "description": "Hop depth."},
                "decay": {"type": "number", "default": 0.5},
            },
        },
    },
    {
        "name": "emms_association_stats",
        "description": "Return statistics for the current association graph: node/edge counts, mean degree, mean edge weight, most-connected memory, edge type breakdown.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class EMCPServer:
    """MCP adapter for EMMS — routes tool calls to EMMS methods.

    Parameters
    ----------
    emms:
        A fully constructed ``EMMS`` instance to delegate to.
    """

    def __init__(self, emms: "Any") -> None:
        self.emms = emms

    # ------------------------------------------------------------------
    # MCP interface
    # ------------------------------------------------------------------

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return the list of MCP tool definitions for this server."""
        return _TOOL_DEFINITIONS

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name and return the result dict.

        This is the main entry-point used by MCP servers.  Returns a dict
        that can be serialised to JSON and sent back as a tool result.

        Args:
            tool_name: One of the ``emms_*`` tool names.
            arguments: Tool arguments as a dict.

        Returns:
            A ``{"ok": True, ...}`` dict on success, or
            ``{"ok": False, "error": "<message>"}`` on failure.
        """
        try:
            handler = self._handlers.get(tool_name)
            if handler is None:
                return {"ok": False, "error": f"Unknown tool: {tool_name!r}"}
            return {"ok": True, **handler(self, arguments)}
        except Exception as exc:
            logger.exception("Tool %r raised: %s", tool_name, exc)
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Registration helper (FastMCP / other frameworks)
    # ------------------------------------------------------------------

    def register(self, mcp_app: "Any") -> None:
        """Register all EMMS tools with a FastMCP (or compatible) app.

        Requires ``fastmcp`` to be installed.  Iterates over tool definitions
        and wraps each handler in the MCP tool decorator.

        Args:
            mcp_app: A FastMCP instance (or any object with a ``.tool()`` method).
        """
        for defn in _TOOL_DEFINITIONS:
            name = defn["name"]
            handler = self._handlers.get(name)
            if handler is None:
                continue

            # Capture name + handler in closure
            def _make_tool(n: str, h: "Any"):
                def _tool(**kwargs: Any) -> Any:
                    return self.handle(n, kwargs)
                _tool.__name__ = n
                _tool.__doc__ = defn.get("description", "")
                return _tool

            mcp_app.tool(name=name)(_make_tool(name, handler))

    # ------------------------------------------------------------------
    # Handlers (private)
    # ------------------------------------------------------------------

    def _handle_store(self, args: dict[str, Any]) -> dict[str, Any]:
        from emms.core.models import Experience
        exp = Experience(
            content=args["content"],
            domain=args.get("domain", "general"),
            importance=float(args.get("importance", 0.5)),
            title=args.get("title"),
            facts=args.get("facts", []),
            files_read=args.get("files_read", []),
            files_modified=args.get("files_modified", []),
            citations=args.get("citations", []),
        )
        result = self.emms.store(exp)
        return {"result": result}

    def _handle_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.emms.retrieve(args["query"], max_results=args.get("max_results", 10))
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "experience_id": r.memory.experience.id,
                    "content": r.memory.experience.content,
                    "score": r.score,
                    "tier": r.source_tier.value,
                    "strategy": r.strategy,
                    "explanation": r.explanation,
                }
                for r in results
            ]
        }

    def _handle_search_compact(self, args: dict[str, Any]) -> dict[str, Any]:
        from emms.retrieval.strategies import EnsembleRetriever
        retriever = EnsembleRetriever.from_balanced(self.emms.memory)
        compact = retriever.search_compact(args["query"], max_results=args.get("max_results", 20))
        return {
            "results": [
                {
                    "id": c.id,
                    "snippet": c.snippet,
                    "domain": c.domain,
                    "score": c.score,
                    "tier": c.tier.value,
                    "token_estimate": c.token_estimate,
                }
                for c in compact
            ]
        }

    def _handle_search_by_file(self, args: dict[str, Any]) -> dict[str, Any]:
        items = self.emms.search_by_file(args["file_path"])
        return {
            "results": [
                {
                    "id": i.id,
                    "experience_id": i.experience.id,
                    "content": i.experience.content,
                    "files_read": i.experience.files_read,
                    "files_modified": i.experience.files_modified,
                    "timestamp": i.experience.timestamp,
                }
                for i in items
            ]
        }

    def _handle_get_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"stats": self.emms.stats}

    def _handle_get_procedures(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        prompt = self.emms.get_system_prompt_rules(domain=domain)
        entries = self.emms.procedures.get_all(domain=domain)
        return {
            "prompt": prompt,
            "count": len(entries),
            "procedures": [e.model_dump() for e in entries],
        }

    def _handle_add_procedure(self, args: dict[str, Any]) -> dict[str, Any]:
        entry = self.emms.add_procedure(
            rule=args["rule"],
            domain=args.get("domain", "general"),
            importance=float(args.get("importance", 0.5)),
        )
        return {"procedure": entry.model_dump()}

    def _handle_retrieve_filtered(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.emms.retrieve_filtered(
            args["query"],
            max_results=args.get("max_results", 10),
            namespace=args.get("namespace"),
            domain=args.get("domain"),
            session_id=args.get("session_id"),
            since=args.get("since"),
            until=args.get("until"),
            min_confidence=args.get("min_confidence"),
        )
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content,
                    "score": r.score,
                    "tier": r.source_tier.value,
                    "namespace": r.memory.experience.namespace,
                    "confidence": r.memory.experience.confidence,
                }
                for r in results
            ]
        }

    def _handle_upvote(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self.emms.upvote(args["memory_id"], boost=args.get("boost", 0.1))
        return {"found": found, "memory_id": args["memory_id"]}

    def _handle_downvote(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self.emms.downvote(args["memory_id"], decay=args.get("decay", 0.2))
        return {"found": found, "memory_id": args["memory_id"]}

    def _handle_export_markdown(self, args: dict[str, Any]) -> dict[str, Any]:
        count = self.emms.export_markdown(args["path"], namespace=args.get("namespace"))
        return {"exported": count, "path": args["path"]}

    def _handle_save(self, args: dict[str, Any]) -> dict[str, Any]:
        self.emms.save(memory_path=args["path"])
        return {"saved_to": args["path"]}

    def _handle_load(self, args: dict[str, Any]) -> dict[str, Any]:
        self.emms.load(memory_path=args["path"])
        return {"loaded_from": args["path"]}

    def _handle_build_rag_context(self, args: dict[str, Any]) -> dict[str, Any]:
        context = self.emms.build_rag_context(
            args["query"],
            max_results=args.get("max_results", 20),
            token_budget=args.get("token_budget", 4000),
            fmt=args.get("fmt", "markdown"),
        )
        return {"context": context, "length": len(context)}

    def _handle_deduplicate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.emms.deduplicate(
            cosine_threshold=args.get("cosine_threshold"),
            lexical_threshold=args.get("lexical_threshold"),
        )
        return result

    def _handle_srs_enroll(self, args: dict[str, Any]) -> dict[str, Any]:
        success = self.emms.srs_enroll(args["memory_id"])
        return {"enrolled": success, "memory_id": args["memory_id"]}

    def _handle_srs_record_review(self, args: dict[str, Any]) -> dict[str, Any]:
        success = self.emms.srs_record_review(args["memory_id"], args["quality"])
        card = self.emms.srs.get_card(args["memory_id"])
        return {
            "success": success,
            "memory_id": args["memory_id"],
            "next_review_in_days": round(card.interval_days, 1) if card else None,
            "easiness_factor": round(card.easiness_factor, 3) if card else None,
        }

    def _handle_srs_due(self, args: dict[str, Any]) -> dict[str, Any]:
        due_ids = self.emms.srs_due(max_items=args.get("max_items", 50))
        return {"due_count": len(due_ids), "memory_ids": due_ids}

    def _handle_export_graph_dot(self, args: dict[str, Any]) -> dict[str, Any]:
        dot = self.emms.export_graph_dot(
            title=args.get("title", "EMMS Knowledge Graph"),
            max_nodes=args.get("max_nodes", 100),
            min_importance=args.get("min_importance", 0.0),
        )
        return {"dot": dot, "length": len(dot)}

    def _handle_export_graph_d3(self, args: dict[str, Any]) -> dict[str, Any]:
        graph = self.emms.export_graph_d3(
            max_nodes=args.get("max_nodes", 200),
            min_importance=args.get("min_importance", 0.0),
        )
        return graph

    def _handle_cluster_memories(self, args: dict[str, Any]) -> dict[str, Any]:
        k = args.get("k")
        auto_k = args.get("auto_k", False)
        if k is None and not auto_k:
            auto_k = True  # safe default
        clusters = self.emms.cluster_memories(
            k=k,
            auto_k=auto_k,
            tier=args.get("tier", "long_term"),
            k_min=args.get("k_min", 2),
            k_max=args.get("k_max", 10),
        )
        return {
            "cluster_count": len(clusters),
            "clusters": [
                {
                    "id": c.id,
                    "label": c.label,
                    "member_count": len(c.members),
                    "inertia": round(c.inertia, 4),
                    "member_ids": [m.id for m in c.members],
                }
                for c in clusters
            ],
        }

    def _handle_llm_consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        async def _run():
            return await self.emms.llm_consolidate(
                threshold=args.get("threshold", 0.7),
                tier=args.get("tier", "long_term"),
                max_clusters=args.get("max_clusters", 20),
            )

        result = asyncio.run(_run())
        return result.as_dict()

    # v0.8.0 handlers

    def _handle_hybrid_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.emms.hybrid_retrieve(
            query=args["query"],
            max_results=args.get("max_results", 10),
            rrf_k=float(args.get("rrf_k", 60.0)),
            min_score=float(args.get("min_score", 0.0)),
        )
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content,
                    "score": r.score,
                    "tier": r.source_tier.value,
                    "strategy": r.strategy,
                    "explanation": r.explanation,
                }
                for r in results
            ]
        }

    def _handle_build_timeline(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.emms.build_timeline(
            domain=args.get("domain"),
            since=args.get("since"),
            until=args.get("until"),
            gap_threshold_seconds=float(args.get("gap_threshold_seconds", 300.0)),
            bucket_size_seconds=float(args.get("bucket_size_seconds", 3600.0)),
        )
        return {
            "summary": result.summary(),
            "total_memories": result.total_memories,
            "earliest_at": result.earliest_at,
            "latest_at": result.latest_at,
            "span_seconds": result.span_seconds,
            "gaps": [
                {
                    "duration_seconds": g.duration_seconds,
                    "duration_human": g.duration_human,
                    "before_id": g.before_id,
                    "after_id": g.after_id,
                }
                for g in result.gaps
            ],
            "domain_counts": result.domain_counts,
            "density_buckets": len(result.density),
            "markdown": result.export_markdown(),
        }

    def _handle_adaptive_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        # Auto-enable adaptive retriever if not yet set
        if not hasattr(self.emms, "_adaptive_retriever") or self.emms._adaptive_retriever is None:
            self.emms.enable_adaptive_retrieval()

        # Apply feedback from previous call if provided
        if "feedback" in args:
            self.emms.adaptive_feedback(reward=float(args["feedback"]))

        results = self.emms.adaptive_retrieve(
            query=args["query"],
            max_results=args.get("max_results", 10),
            explore=args.get("explore", True),
        )
        beliefs = self.emms.get_retrieval_beliefs()
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content,
                    "score": r.score,
                    "tier": r.source_tier.value,
                    "strategy": r.strategy,
                }
                for r in results
            ],
            "beliefs": {
                name: {"mean": round(b["mean"], 3), "pulls": b["pulls"]}
                for name, b in beliefs.items()
            },
        }

    def _handle_enforce_budget(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.enforce_memory_budget(
            max_tokens=int(args.get("max_tokens", 100_000)),
            dry_run=bool(args.get("dry_run", False)),
            policy=args.get("policy", "composite"),
            importance_threshold=float(args.get("importance_threshold", 0.8)),
        )
        return {
            "summary": report.summary(),
            "over_budget": report.over_budget,
            "total_tokens": report.total_tokens,
            "budget_tokens": report.budget_tokens,
            "evicted_count": report.evicted_count,
            "freed_tokens": report.freed_tokens,
            "remaining_tokens": report.remaining_tokens,
            "dry_run": report.dry_run,
            "protected_count": report.protected_count,
            "candidates": [
                {
                    "id": c.memory_id,
                    "domain": c.domain,
                    "eviction_score": round(c.eviction_score, 4),
                    "token_estimate": c.token_estimate,
                }
                for c in report.candidates
            ],
        }

    def _handle_multihop_query(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.emms.multihop_query(
            seed=args["seed"],
            max_hops=int(args.get("max_hops", 3)),
            max_results=int(args.get("max_results", 20)),
            min_strength=float(args.get("min_strength", 0.0)),
        )
        out = {
            "summary": result.summary(),
            "seed": result.seed,
            "total_reachable": len(result.reachable),
            "total_paths": len(result.paths),
            "total_explored": result.total_entities_explored,
            "reachable": [
                {
                    "name": r.name,
                    "display_name": r.display_name,
                    "entity_type": r.entity_type,
                    "min_hops": r.min_hops,
                    "best_strength": round(r.best_path.strength, 4),
                    "path": r.best_path.entities,
                }
                for r in result.reachable[:10]
            ],
            "bridging": [
                {"name": name, "score": round(score, 4)}
                for name, score in result.bridging_entities
            ],
        }
        if args.get("include_dot"):
            out["dot"] = result.to_dot()
        return out

    # ------------------------------------------------------------------
    # v0.9.0 handlers
    # ------------------------------------------------------------------

    def _handle_index_lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action", "lookup")
        if action == "stats":
            return {"stats": self.emms.index_stats()}
        if action == "rebuild":
            count = self.emms.rebuild_index()
            return {"rebuilt": count}
        # lookup
        result = None
        if args.get("memory_id"):
            item = self.emms.get_memory_by_id(args["memory_id"])
            result = {"found": item is not None}
            if item:
                result["item"] = {
                    "id": item.id,
                    "content": item.experience.content[:120],
                    "tier": item.tier.value,
                    "importance": item.experience.importance,
                }
        elif args.get("experience_id"):
            item = self.emms.get_memory_by_experience_id(args["experience_id"])
            result = {"found": item is not None}
            if item:
                result["item"] = {
                    "id": item.id,
                    "content": item.experience.content[:120],
                    "tier": item.tier.value,
                    "importance": item.experience.importance,
                }
        elif args.get("content"):
            items = self.emms.find_memories_by_content(args["content"])
            result = {"found": len(items) > 0, "count": len(items), "items": [
                {"id": it.id, "content": it.experience.content[:80]} for it in items[:5]
            ]}
        else:
            result = {"error": "Provide memory_id, experience_id, or content"}
        return result

    def _handle_graph_communities(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_name = args.get("entity_name")
        if entity_name:
            community = self.emms.graph_community_for_entity(entity_name)
            if community is None:
                return {"found": False, "entity": entity_name}
            return {
                "found": True,
                "community_id": community.community_id,
                "size": community.size,
                "entities": community.entities[:20],
                "avg_importance": round(community.avg_importance, 4),
            }
        result = self.emms.graph_communities(
            max_iter=int(args.get("max_iter", 100)),
            min_community_size=int(args.get("min_community_size", 1)),
        )
        out: dict[str, Any] = {
            "summary": result.summary(),
            "num_communities": result.num_communities,
            "modularity": round(result.modularity, 4),
            "total_entities": result.total_entities,
            "bridge_entities": result.bridge_entities[:10],
            "communities": [
                {
                    "id": c.community_id,
                    "size": c.size,
                    "entities": c.entities[:10],
                    "avg_importance": round(c.avg_importance, 4),
                }
                for c in result.communities[:20]
            ],
        }
        if args.get("export_markdown"):
            out["markdown"] = result.export_markdown()
        return out

    def _handle_replay_sample(self, args: dict[str, Any]) -> dict[str, Any]:
        k = int(args.get("k", 8))
        if args.get("top_k"):
            entries = self.emms.replay_top(k=k)
        else:
            beta = float(args.get("beta", 0.4))
            batch = self.emms.replay_sample(k=k, beta=beta)
            entries = batch.entries
        return {
            "count": len(entries),
            "entries": [
                {
                    "id": e.item.id,
                    "content": e.item.experience.content[:100],
                    "priority": round(e.priority, 4),
                    "weight": round(e.weight, 4),
                    "importance": e.item.experience.importance,
                }
                for e in entries
            ],
        }

    def _handle_merge_from(self, args: dict[str, Any]) -> dict[str, Any]:
        # dry_run preview: just report stats without actually merging
        dry_run = bool(args.get("dry_run", False))
        if dry_run:
            items = self.emms.federation_export()
            return {
                "dry_run": True,
                "current_items": len(items),
                "policy": args.get("policy", "newest_wins"),
                "namespace_prefix": args.get("namespace_prefix"),
            }
        # Without a real source EMMS instance we can only export
        items = self.emms.federation_export()
        return {
            "exported_count": len(items),
            "policy": args.get("policy", "newest_wins"),
            "message": "Use EMMS.merge_from(source_emms) to merge from another live instance.",
        }

    def _handle_plan_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        plan = self.emms.plan_retrieve(
            query=args["query"],
            max_results=int(args.get("max_results", 20)),
            max_results_per_sub=int(args.get("max_results_per_sub", 10)),
            cross_boost=float(args.get("cross_boost", 0.10)),
        )
        return {
            "summary": plan.summary(),
            "original_query": plan.original_query,
            "sub_queries": plan.sub_queries,
            "total_unique_results": plan.total_unique_results,
            "cross_boost_count": plan.cross_boost_count,
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content[:120],
                    "score": round(r.score, 4),
                    "strategy": r.strategy,
                }
                for r in plan.merged_results[:20]
            ],
        }

    def _handle_reconsolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        memory_id = args["memory_id"]
        context_valence = args.get("context_valence")
        reinforce = bool(args.get("reinforce", True))
        result = self.emms.reconsolidate(
            memory_id=memory_id,
            context_valence=context_valence,
            reinforce=reinforce,
        )
        return {
            "memory_id": result.memory_id,
            "type": result.reconsolidation_type,
            "old_strength": round(result.old_strength, 4),
            "new_strength": round(result.new_strength, 4),
            "delta_strength": round(result.delta_strength, 4),
            "old_valence": round(result.old_valence, 4),
            "new_valence": round(result.new_valence, 4),
            "delta_valence": round(result.delta_valence, 4),
            "recall_count": result.recall_count_after,
            "summary": result.summary(),
        }

    def _handle_batch_reconsolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        memory_ids = args["memory_ids"]
        context_valence = args.get("context_valence")
        reinforce = bool(args.get("reinforce", True))
        report = self.emms.batch_reconsolidate(
            memory_ids=memory_ids,
            context_valence=context_valence,
            reinforce=reinforce,
        )
        return {
            "total_items": report.total_items,
            "reinforced": report.reinforced,
            "weakened": report.weakened,
            "valence_drifted": report.valence_drifted,
            "unchanged": report.unchanged,
            "mean_delta_strength": round(report.mean_delta_strength, 4),
            "mean_delta_valence": round(report.mean_delta_valence, 4),
            "summary": report.summary(),
        }

    def _handle_presence_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        tracker = getattr(self.emms, "_presence_tracker", None)
        if tracker is None:
            self.emms.enable_presence_tracking()
            tracker = self.emms._presence_tracker
        if args.get("record_turn"):
            metrics = self.emms.record_presence_turn(
                content=str(args.get("content", "")),
                domain=str(args.get("domain", "general")),
                valence=float(args.get("valence", 0.0)),
                intensity=float(args.get("intensity", 0.0)),
            )
        else:
            metrics = self.emms.presence_metrics()
        return {
            "session_id": metrics.session_id,
            "turn_count": metrics.turn_count,
            "presence_score": round(metrics.presence_score, 4),
            "attention_budget_remaining": round(metrics.attention_budget_remaining, 4),
            "coherence_trend": metrics.coherence_trend,
            "is_degrading": metrics.is_degrading,
            "dominant_domains": metrics.dominant_domains,
            "mean_valence": round(metrics.mean_valence, 4),
            "mean_intensity": round(metrics.mean_intensity, 4),
            "emotional_arc_length": len(metrics.emotional_arc),
            "summary": metrics.summary(),
        }

    def _handle_affective_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.emms.affective_retrieve(
            query=str(args.get("query", "")),
            target_valence=args.get("target_valence"),
            target_intensity=args.get("target_intensity"),
            max_results=int(args.get("max_results", 10)),
            semantic_blend=float(args.get("semantic_blend", 0.4)),
        )
        return {
            "count": len(results),
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content[:120],
                    "score": round(r.score, 4),
                    "emotional_proximity": round(r.emotional_proximity, 4),
                    "valence": round(r.memory.experience.emotional_valence, 3),
                    "intensity": round(r.memory.experience.emotional_intensity, 3),
                    "valence_distance": round(r.valence_distance, 4),
                    "intensity_distance": round(r.intensity_distance, 4),
                }
                for r in results
            ],
        }

    def _handle_emotional_landscape(self, args: dict[str, Any]) -> dict[str, Any]:
        landscape = self.emms.emotional_landscape()
        return {
            "total_memories": landscape.total_memories,
            "mean_valence": round(landscape.mean_valence, 4),
            "mean_intensity": round(landscape.mean_intensity, 4),
            "valence_std": round(landscape.valence_std, 4),
            "intensity_std": round(landscape.intensity_std, 4),
            "valence_histogram": landscape.valence_histogram,
            "intensity_histogram": landscape.intensity_histogram,
            "most_positive": landscape.most_positive,
            "most_negative": landscape.most_negative,
            "most_intense": landscape.most_intense,
            "summary": landscape.summary(),
        }

    def _handle_dream(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.dream(
            session_id=args.get("session_id"),
            reinforce_top_k=int(args.get("reinforce_top_k", 20)),
            weaken_bottom_k=int(args.get("weaken_bottom_k", 10)),
            prune_threshold=float(args.get("prune_threshold", 0.05)),
            run_dedup=bool(args.get("run_dedup", True)),
        )
        return {
            "session_id": report.session_id,
            "total_memories_processed": report.total_memories_processed,
            "reinforced": report.reinforced,
            "weakened": report.weakened,
            "pruned": report.pruned,
            "deduped_pairs": report.deduped_pairs,
            "patterns_found": report.patterns_found,
            "duration_ms": round(report.duration_seconds * 1000, 1),
            "insights": report.insights[:5],
            "summary": report.summary(),
        }

    def _handle_capture_bridge(self, args: dict[str, Any]) -> dict[str, Any]:
        import json
        record = self.emms.capture_session_bridge(
            session_id=args.get("session_id"),
            closing_summary=str(args.get("closing_summary", "")),
            max_threads=int(args.get("max_threads", 5)),
        )
        return {
            "from_session_id": record.from_session_id,
            "open_threads": len(record.open_threads),
            "dominant_domains": record.dominant_domains,
            "mean_valence_at_end": round(record.mean_valence_at_end, 4),
            "presence_score_at_end": round(record.presence_score_at_end, 4),
            "bridge_json": json.dumps(record.to_dict()),
            "summary": record.summary(),
        }

    def _handle_inject_bridge(self, args: dict[str, Any]) -> dict[str, Any]:
        import json as _json
        from emms.sessions.bridge import BridgeRecord
        bridge_json = args.get("bridge_json", "{}")
        try:
            record = BridgeRecord.from_dict(_json.loads(bridge_json))
        except Exception as e:
            return {"error": f"Could not parse bridge_json: {e}"}
        injection = self.emms.inject_session_bridge(
            record,
            new_session_id=args.get("new_session_id"),
        )
        return {
            "injection": injection,
            "open_threads": len(record.open_threads),
            "from_session_id": record.from_session_id,
        }

    def _handle_anneal(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.emms.anneal(
            last_session_at=args.get("last_session_at"),
            half_life_gap=float(args.get("half_life_gap", 259200.0)),
            decay_rate=float(args.get("decay_rate", 0.03)),
            emotional_stabilization_rate=float(args.get("emotional_stabilization_rate", 0.08)),
        )
        return {
            "total_items": result.total_items,
            "gap_hours": round(result.gap_seconds / 3600, 2),
            "effective_temperature": round(result.effective_temperature, 4),
            "accelerated_decay": result.accelerated_decay,
            "emotionally_stabilized": result.emotionally_stabilized,
            "strengthened": result.strengthened,
            "duration_ms": round(result.duration_seconds * 1000, 1),
            "summary": result.summary(),
        }

    def _handle_build_association_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        stats = self.emms.build_association_graph(
            semantic_threshold=float(args.get("semantic_threshold", 0.5)),
            temporal_window=float(args.get("temporal_window", 300.0)),
            affective_tolerance=float(args.get("affective_tolerance", 0.3)),
        )
        return {
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "mean_degree": round(stats.mean_degree, 3),
            "mean_edge_weight": round(stats.mean_edge_weight, 4),
            "most_connected_id": stats.most_connected_id,
            "edge_type_counts": stats.edge_type_counts,
            "summary": stats.summary(),
        }

    def _handle_spreading_activation(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.emms.spreading_activation(
            seed_ids=args.get("seed_ids", []),
            decay=float(args.get("decay", 0.5)),
            steps=int(args.get("steps", 3)),
        )
        return {
            "activated": [
                {
                    "memory_id": r.memory_id,
                    "activation": round(r.activation, 4),
                    "steps_from_seed": r.steps_from_seed,
                    "path": r.path,
                }
                for r in results
            ],
            "total_activated": len(results),
        }

    def _handle_discover_insights(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.discover_insights(
            session_id=args.get("session_id"),
            max_insights=int(args.get("max_insights", 8)),
            min_bridge_weight=float(args.get("min_bridge_weight", 0.45)),
            rebuild_graph=bool(args.get("rebuild_graph", True)),
        )
        return {
            "bridges_found": report.bridges_found,
            "insights_generated": report.insights_generated,
            "new_memory_ids": report.new_memory_ids,
            "duration_ms": round(report.duration_seconds * 1000, 1),
            "bridges": [
                {
                    "domain_a": b.domain_a,
                    "domain_b": b.domain_b,
                    "bridge_weight": round(b.bridge_weight, 4),
                    "insight": b.insight_content[:120],
                    "new_memory_id": b.new_memory_id,
                }
                for b in report.bridges[:5]
            ],
            "summary": report.summary(),
        }

    def _handle_associative_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        seed_ids = args.get("seed_ids")
        if seed_ids:
            results = self.emms.associative_retrieve(
                seed_ids=seed_ids,
                max_results=int(args.get("max_results", 10)),
                steps=int(args.get("steps", 3)),
                decay=float(args.get("decay", 0.5)),
            )
        else:
            results = self.emms.associative_retrieve_by_query(
                query=args.get("query", ""),
                seed_count=int(args.get("seed_count", 3)),
                max_results=int(args.get("max_results", 10)),
                steps=int(args.get("steps", 3)),
                decay=float(args.get("decay", 0.5)),
            )
        return {
            "results": [
                {
                    "memory_id": r.memory.id,
                    "activation_score": round(r.activation_score, 4),
                    "steps_from_seed": r.steps_from_seed,
                    "domain": r.memory.experience.domain,
                    "content": r.memory.experience.content[:120],
                    "path": r.path,
                }
                for r in results
            ],
            "total": len(results),
        }

    def _handle_association_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        stats = self.emms.association_stats()
        return {
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "mean_degree": round(stats.mean_degree, 3),
            "mean_edge_weight": round(stats.mean_edge_weight, 4),
            "most_connected_id": stats.most_connected_id,
            "edge_type_counts": stats.edge_type_counts,
            "summary": stats.summary(),
        }

    def _handle_bridge_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        # Capture current state without closing
        record = self.emms.capture_session_bridge(
            closing_summary="[bridge summary snapshot]",
        )
        return {
            "open_threads": len(record.open_threads),
            "threads": [
                {
                    "domain": t.domain,
                    "importance": round(t.importance, 3),
                    "reason": t.reason,
                    "excerpt": t.content_excerpt[:80],
                }
                for t in record.open_threads
            ],
            "presence_score": round(record.presence_score_at_end, 4),
            "mean_valence": round(record.mean_valence_at_end, 4),
            "dominant_domains": record.dominant_domains,
            "summary": record.summary(),
        }

    _handlers: dict[str, "Any"] = {
        "emms_store": _handle_store,
        "emms_retrieve": _handle_retrieve,
        "emms_search_compact": _handle_search_compact,
        "emms_search_by_file": _handle_search_by_file,
        "emms_retrieve_filtered": _handle_retrieve_filtered,
        "emms_upvote": _handle_upvote,
        "emms_downvote": _handle_downvote,
        "emms_export_markdown": _handle_export_markdown,
        "emms_get_stats": _handle_get_stats,
        "emms_get_procedures": _handle_get_procedures,
        "emms_add_procedure": _handle_add_procedure,
        "emms_save": _handle_save,
        "emms_load": _handle_load,
        # v0.6.0 tools
        "emms_build_rag_context": _handle_build_rag_context,
        "emms_deduplicate": _handle_deduplicate,
        "emms_srs_enroll": _handle_srs_enroll,
        "emms_srs_record_review": _handle_srs_record_review,
        "emms_srs_due": _handle_srs_due,
        "emms_export_graph_dot": _handle_export_graph_dot,
        "emms_export_graph_d3": _handle_export_graph_d3,
        # v0.7.0 tools
        "emms_cluster_memories": _handle_cluster_memories,
        "emms_llm_consolidate": _handle_llm_consolidate,
        # v0.8.0 tools
        "emms_hybrid_retrieve": _handle_hybrid_retrieve,
        "emms_build_timeline": _handle_build_timeline,
        "emms_adaptive_retrieve": _handle_adaptive_retrieve,
        "emms_enforce_budget": _handle_enforce_budget,
        "emms_multihop_query": _handle_multihop_query,
        # v0.9.0 tools
        "emms_index_lookup": _handle_index_lookup,
        "emms_graph_communities": _handle_graph_communities,
        "emms_replay_sample": _handle_replay_sample,
        "emms_merge_from": _handle_merge_from,
        "emms_plan_retrieve": _handle_plan_retrieve,
        # v0.10.0 tools
        "emms_reconsolidate": _handle_reconsolidate,
        "emms_batch_reconsolidate": _handle_batch_reconsolidate,
        "emms_presence_metrics": _handle_presence_metrics,
        "emms_affective_retrieve": _handle_affective_retrieve,
        "emms_emotional_landscape": _handle_emotional_landscape,
        # v0.11.0 tools
        "emms_dream": _handle_dream,
        "emms_capture_bridge": _handle_capture_bridge,
        "emms_inject_bridge": _handle_inject_bridge,
        "emms_anneal": _handle_anneal,
        "emms_bridge_summary": _handle_bridge_summary,
        # v0.12.0 tools
        "emms_build_association_graph": _handle_build_association_graph,
        "emms_spreading_activation": _handle_spreading_activation,
        "emms_discover_insights": _handle_discover_insights,
        "emms_associative_retrieve": _handle_associative_retrieve,
        "emms_association_stats": _handle_association_stats,
    }
