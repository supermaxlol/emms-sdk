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
import threading
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
                "subtitle": {"type": "string", "description": "One-sentence explanation ≤24 words."},
                "facts": {"type": "array", "items": {"type": "string"}, "description": "Discrete factual bullet points."},
                "files_read": {"type": "array", "items": {"type": "string"}, "description": "Files read during this event."},
                "files_modified": {"type": "array", "items": {"type": "string"}, "description": "Files created or edited."},
                "citations": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs this experience validates."},
                "emotional_valence": {"type": "number", "default": 0.0, "description": "-1 (negative) … +1 (positive) emotional tone."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Observation classification — what kind of event this records.",
                },
                "concept_tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["how-it-works", "why-it-exists", "what-changed", "problem-solution", "gotcha", "pattern", "trade-off"],
                    },
                    "description": "Epistemological role tags — how to interpret this memory.",
                },
                "confidence": {"type": "number", "default": 1.0, "description": "0 (uncertain) … 1 (fully verified)."},
                "namespace": {"type": "string", "default": "default", "description": "Project/repo isolation scope — only retrieves within same namespace."},
                "session_id": {"type": "string", "description": "Conversation session ID for grouping related memories."},
                "private": {"type": "boolean", "default": False, "description": "Exclude from retrieval and export when True."},
                "update_mode": {"type": "string", "enum": ["insert", "patch"], "default": "insert", "description": "'insert' appends new memory; 'patch' updates matching memory."},
                "patch_key": {"type": "string", "description": "When update_mode='patch', match on this key (defaults to title)."},
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
                "namespace": {"type": "string", "description": "Limit to this namespace/project (omit for all)."},
                "domain": {"type": "string", "description": "Filter to this domain."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Filter to this observation type.",
                },
                "concept_tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["how-it-works", "why-it-exists", "what-changed", "problem-solution", "gotcha", "pattern", "trade-off"],
                    },
                    "description": "Only return memories carrying at least one of these concept tags.",
                },
                "min_importance": {"type": "number", "description": "Minimum importance threshold 0–1."},
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "recency", "importance"],
                    "default": "relevance",
                    "description": "Sort order (most useful when query is empty).",
                },
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
                "namespace": {"type": "string", "description": "Limit search to this project namespace."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Only return memories of this observation type.",
                },
                "concept_tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["how-it-works", "why-it-exists", "what-changed", "problem-solution", "gotcha", "pattern", "trade-off"],
                    },
                    "description": "Only return memories carrying at least one of these concept tags.",
                },
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
        "description": "Retrieve memories with structured filters (namespace, obs_type, domain, time range, confidence). Query is optional — omit it to do pure filter-based lookup with no semantic ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional semantic query. Omit for pure filter-based lookup."},
                "max_results": {"type": "integer", "default": 10},
                "namespace": {"type": "string", "description": "Filter to this project/repo namespace."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Filter to this observation type.",
                },
                "domain": {"type": "string"},
                "session_id": {"type": "string"},
                "since": {"type": "number", "description": "Unix timestamp lower bound"},
                "until": {"type": "number", "description": "Unix timestamp upper bound"},
                "min_confidence": {"type": "number", "description": "Minimum confidence threshold 0–1"},
                "min_importance": {"type": "number", "description": "Minimum importance threshold 0–1"},
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "recency", "importance"],
                    "description": "Sort order when query is empty: 'relevance' (importance×confidence), 'recency' (newest first), 'importance' (importance descending). Ignored when a query is provided.",
                },
                "concept_tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["how-it-works", "why-it-exists", "what-changed", "problem-solution", "gotcha", "pattern", "trade-off"],
                    },
                    "description": "Only return memories that carry at least one of these concept tags.",
                },
            },
            "required": [],
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
    # v0.13.0 tools
    {
        "name": "emms_metacognition_report",
        "description": "Generate a metacognitive self-assessment: epistemic confidence per memory, per-domain knowledge profiles, contradiction pairs, knowledge gaps, and recommendations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_contradictions": {"type": "integer", "default": 5},
                "confidence_threshold_high": {"type": "number", "default": 0.65},
                "confidence_threshold_low": {"type": "number", "default": 0.3},
            },
        },
    },
    {
        "name": "emms_knowledge_map",
        "description": "Return per-domain knowledge profiles showing memory count, mean confidence, coverage score, and mean importance for each domain.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "emms_find_contradictions",
        "description": "Find memory pairs with semantic overlap but conflicting emotional valence — potential contradictions in the agent's knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_pairs": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "emms_intend",
        "description": "Store a future-oriented intention with a trigger context. The intention will activate when check_intentions() is called with a matching context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What the agent plans to do."},
                "trigger_context": {"type": "string", "description": "Context description that should trigger this intention."},
                "priority": {"type": "number", "default": 0.5, "description": "Urgency 0–1."},
            },
            "required": ["content", "trigger_context"],
        },
    },
    {
        "name": "emms_check_intentions",
        "description": "Check which stored intentions are activated by the current context. Returns activated intentions sorted by activation score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "current_context": {"type": "string", "description": "Current conversational context text."},
            },
            "required": ["current_context"],
        },
    },
    # v0.14.0 tools
    {
        "name": "emms_open_episode",
        "description": "Open a new bounded episode in the episodic buffer. Records temporal boundaries, emotional arc, and key memories for this session segment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Brief description of what this episode is about."},
                "session_id": {"type": "string", "description": "Optional session label (auto-generated if omitted)."},
            },
        },
    },
    {
        "name": "emms_close_episode",
        "description": "Close the current episode, computing final statistics (mean/peak valence, duration). Provide a brief outcome description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "description": "Brief description of how the episode resolved."},
                "episode_id": {"type": "string", "description": "Episode to close (default: current open episode)."},
            },
        },
    },
    {
        "name": "emms_recent_episodes",
        "description": "Return the N most recent episodes from the episodic buffer, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 10, "description": "Number of episodes to return."},
            },
        },
    },
    {
        "name": "emms_extract_schemas",
        "description": "Extract abstract knowledge schemas from stored memories — recurring keyword clusters become transferable patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to one domain (omit for all)."},
                "max_schemas": {"type": "integer", "default": 12, "description": "Maximum schemas to return."},
            },
        },
    },
    {
        "name": "emms_forget",
        "description": "Suppress or prune memories. Target a specific ID, a whole domain, or memories below a confidence threshold.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Suppress (or hard-delete) a specific memory by ID."},
                "domain": {"type": "string", "description": "Suppress all memories in this domain."},
                "below_confidence": {"type": "number", "description": "Suppress memories whose confidence is below this threshold (0–1)."},
                "suppression_rate": {"type": "number", "default": 0.4, "description": "Fraction to reduce strength by (default 0.4). Ignored when hard=true."},
                "hard": {"type": "boolean", "default": False, "description": "If true, permanently remove the memory from all tiers and indices (irreversible). Only works with memory_id."},
            },
        },
    },
    # v0.15.0 tools
    {
        "name": "emms_reflect",
        "description": "Run a structured self-reflection pass: reviews high-importance memories and recent episodes, synthesises lessons, stores them as reflection memories, and surfaces open questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Label for this reflection session (auto-generated if omitted)."},
                "domain": {"type": "string", "description": "Restrict reflection to one domain (omit for all)."},
                "lookback_episodes": {"type": "integer", "default": 5, "description": "Number of recent episodes to incorporate."},
            },
        },
    },
    {
        "name": "emms_weave_narrative",
        "description": "Weave autobiographical narrative threads from stored memories — groups by domain, sorts chronologically, and generates readable first-person prose.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to one domain (omit for all)."},
                "max_threads": {"type": "integer", "default": 8, "description": "Maximum narrative threads to return."},
            },
        },
    },
    {
        "name": "emms_narrative_threads",
        "description": "Return narrative threads for one or all domains, longest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to one domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_source_audit",
        "description": "Audit memories for source uncertainty and confabulation risk. Returns flagged memories whose source confidence falls below the threshold.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "flag_threshold": {"type": "number", "default": 0.5, "description": "Confidence below which a memory is flagged (0–1)."},
                "max_flagged": {"type": "integer", "default": 20, "description": "Maximum flagged entries to return."},
            },
        },
    },
    {
        "name": "emms_tag_source",
        "description": "Assign a provenance tag to a memory: observation, inference, instruction, reflection, dream, insight, or unknown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to tag."},
                "source_type": {"type": "string", "enum": ["observation", "inference", "instruction", "reflection", "dream", "insight", "unknown"], "description": "Memory provenance type."},
                "confidence": {"type": "number", "default": 0.8, "description": "Confidence in this attribution (0–1)."},
                "note": {"type": "string", "default": "", "description": "Optional free-text provenance note."},
            },
            "required": ["memory_id", "source_type"],
        },
    },
    # v0.16.0 tools
    {
        "name": "emms_curiosity_report",
        "description": "Scan memory for knowledge gaps and generate curiosity-driven exploration goals. Returns goals ranked by urgency across sparse, uncertain, contradictory, and novel domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict scan to one domain (omit for all domains)."},
            },
        },
    },
    {
        "name": "emms_exploration_goals",
        "description": "List all pending (un-explored) curiosity goals generated by the last curiosity scan, sorted by urgency descending.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_revise_beliefs",
        "description": "Detect and resolve contradictions in the memory store using AGM belief revision: merge (synthesise reconciliation), supersede (weaken loser), or flag for review.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "new_memory_id": {"type": "string", "description": "Check only this memory against others (omit for full scan)."},
                "domain": {"type": "string", "description": "Restrict revision scan to one domain."},
                "max_revisions": {"type": "integer", "default": 8, "description": "Maximum number of belief revisions to perform."},
            },
        },
    },
    {
        "name": "emms_decay_report",
        "description": "Compute Ebbinghaus forgetting-curve retention for all memories without applying changes. Returns per-memory retention, stability, and days-since-access.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict report to one domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_apply_decay",
        "description": "Apply the Ebbinghaus forgetting curve to memory strengths, optionally pruning memories whose post-decay strength falls below the threshold.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict decay to one domain (omit for all)."},
                "prune": {"type": "boolean", "default": False, "description": "If true, remove memories below the prune threshold after decay."},
            },
        },
    },
    # v0.17.0 tools
    {
        "name": "emms_push_goal",
        "description": "Push a new goal onto the goal stack with priority and optional parent for hierarchical decomposition.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Goal description."},
                "domain": {"type": "string", "default": "general", "description": "Knowledge domain for this goal."},
                "priority": {"type": "number", "default": 0.5, "description": "Urgency 0..1 (higher = more urgent)."},
                "parent_id": {"type": "string", "description": "Parent goal ID to form a sub-goal (optional)."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "emms_active_goals",
        "description": "List all currently active goals sorted by priority descending.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_complete_goal",
        "description": "Mark a goal as successfully completed, optionally recording an outcome note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "ID of the goal to complete."},
                "outcome_note": {"type": "string", "default": "", "description": "Optional text describing the outcome."},
            },
            "required": ["goal_id"],
        },
    },
    {
        "name": "emms_spotlight_retrieve",
        "description": "Retrieve memories most relevant to the current attentional spotlight — informed by active goals, keywords, and context text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 8, "description": "Maximum number of memories to return."},
                "text": {"type": "string", "description": "Context text to expand the spotlight (optional)."},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Explicit keywords to focus the spotlight (optional)."},
            },
        },
    },
    {
        "name": "emms_find_analogies",
        "description": "Detect structural analogies across memory domains — memories that share relational patterns (causal, enabling, temporal) even if their topics differ.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_domain": {"type": "string", "description": "Restrict source side to this domain (omit for all)."},
                "target_domain": {"type": "string", "description": "Restrict target side to this domain (omit for all)."},
            },
        },
    },
    # v0.18.0 tools
    {
        "name": "emms_predict",
        "description": "Generate predictions from recurring patterns in the memory store. Returns domain-level predictions with confidence scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict predictions to one domain (omit for all domains)."},
            },
        },
    },
    {
        "name": "emms_pending_predictions",
        "description": "List all unresolved predictions sorted by confidence descending.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_blend_concepts",
        "description": "Blend pairs of memories into novel conceptual syntheses using Fauconnier & Turner conceptual integration theory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain_a": {"type": "string", "description": "Source domain for one side of the blend (omit for all domains)."},
                "domain_b": {"type": "string", "description": "Source domain for other side of the blend (omit for all domains)."},
            },
        },
    },
    {
        "name": "emms_project_future",
        "description": "Generate plausible future scenarios by extrapolating from memory patterns and past episodes (episodic future thinking).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict projection to one domain (omit for all)."},
                "horizon_days": {"type": "number", "default": 30.0, "description": "Projection horizon in days (default 30)."},
            },
        },
    },
    {
        "name": "emms_plausible_futures",
        "description": "Return the n most plausible future scenarios from the last projection run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 3, "description": "Number of scenarios to return (default 3)."},
            },
        },
    },
    # v0.19.0 tools
    {
        "name": "emms_regulate_emotions",
        "description": "Assess the agent's current emotional state and apply cognitive reappraisal to negative memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict assessment to this domain (omit for all domains)."},
            },
        },
    },
    {
        "name": "emms_current_emotion",
        "description": "Return the most recently computed emotional state (valence, arousal, dominant_domain).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_build_hierarchy",
        "description": "Build a taxonomic concept hierarchy from memory token co-occurrences.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict hierarchy to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_concept_distance",
        "description": "Compute the shortest-path distance between two concepts in the concept hierarchy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "label_a": {"type": "string", "description": "First concept label."},
                "label_b": {"type": "string", "description": "Second concept label."},
            },
            "required": ["label_a", "label_b"],
        },
    },
    {
        "name": "emms_update_self_model",
        "description": "Rebuild the self-model from current memory, returning beliefs, capability profile, and consistency score.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # v0.20.0 tools
    {
        "name": "emms_build_causal_map",
        "description": "Extract a directed causal graph from memories, returning concept nodes, edges, and influential/affected rankings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_effects_of",
        "description": "Return causal edges whose source is the given concept (what does this concept cause?).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "Cause concept token."},
            },
            "required": ["concept"],
        },
    },
    {
        "name": "emms_generate_counterfactuals",
        "description": "Generate 'what if' counterfactual alternatives to past memories, scored by plausibility and valence shift.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
                "direction": {"type": "string", "enum": ["upward", "downward", "both"], "description": "Counterfactual direction (default 'both')."},
            },
        },
    },
    {
        "name": "emms_distill_skills",
        "description": "Distil reusable procedural skills from recurring action patterns in memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_best_skill",
        "description": "Find the distilled skill most relevant to a goal description using token Jaccard overlap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal_description": {"type": "string", "description": "Natural-language description of the goal."},
            },
            "required": ["goal_description"],
        },
    },
    # v0.21.0 tools
    {
        "name": "emms_build_perspectives",
        "description": "Build Theory-of-Mind agent models from memory, detecting other agents mentioned alongside belief/communication verbs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_agent_model",
        "description": "Return the stored perspective model (beliefs, statements, valence) for a named agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Name of the agent to look up."},
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "emms_compute_trust",
        "description": "Compute credibility scores per information source (domain) using importance, valence stability, and memory count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this source domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_extract_norms",
        "description": "Extract prescriptive and prohibitive behavioural norms from memory content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_check_norm",
        "description": "Find the norms most relevant to a described behaviour using token Jaccard overlap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "behavior": {"type": "string", "description": "Natural-language description of the behaviour to check."},
            },
            "required": ["behavior"],
        },
    },
    # v0.22.0 tools
    {
        "name": "emms_assess_novelty",
        "description": "Score all memories for novelty against the corpus centroid; returns memories ranked by how surprising/unusual they are.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_most_novel",
        "description": "Return the n most novel memories from the last novelty assessment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 5, "description": "Number of memories to return (default 5)."},
            },
        },
    },
    {
        "name": "emms_invent_concepts",
        "description": "Generate novel cross-domain concepts by cross-pollinating rare tokens from different memory domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 8, "description": "Maximum number of concepts to generate (default 8)."},
            },
        },
    },
    {
        "name": "emms_abstract_principles",
        "description": "Extract abstract principles from episodic memories by identifying recurring tokens within domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_best_principle",
        "description": "Return the abstract principle most relevant to a natural-language description using Jaccard overlap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Natural-language description to match against."},
            },
            "required": ["description"],
        },
    },
    # v0.23.0 tools
    {
        "name": "emms_map_values",
        "description": "Extract core values from accumulated memory via a 5-category value lexicon (epistemic, moral, aesthetic, instrumental, social).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Restrict to this value category (omit for all)."},
            },
        },
    },
    {
        "name": "emms_values_for_category",
        "description": "Return all mapped values in a specific value category after map_values has been called.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "One of: epistemic, moral, aesthetic, instrumental, social."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "emms_reason_morally",
        "description": "Evaluate memories through three classical ethical frameworks: consequentialist, deontological, and virtue ethics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_detect_dilemmas",
        "description": "Detect ethical tensions between conflicting moral imperatives in memory — pairs of same-domain memories with opposing valences and high moral weight.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_most_tense_dilemma",
        "description": "Return the ethical dilemma with the highest tension score from the last detect_dilemmas call.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # v0.24.0 tools
    {
        "name": "emms_detect_biases",
        "description": "Detect cognitive biases in accumulated memory — scans 10 bias types (confirmation, availability, sunk_cost, optimism, negativity, hindsight, overconfidence, in_group, anchoring, framing) and returns strength scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_most_pervasive_bias",
        "description": "Return the cognitive bias with the highest strength score from the last detect_biases call.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_synthesize_wisdom",
        "description": "Synthesise practical guidance for a query from memory using four dimensions: value signals, moral patterns, causal patterns, and recurring principles.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or goal to synthesise guidance for."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "emms_evolve_knowledge",
        "description": "Track how knowledge has grown and consolidated across domains — returns growth_rate, consolidation_score, and knowledge_density per domain.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_knowledge_gaps",
        "description": "Return domains with fewer memories than the min_memories threshold — domains where knowledge is sparse.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # v0.25.0 tools
    {
        "name": "emms_detect_rumination",
        "description": "Detect repetitive intrusive thought clusters in memory using Jaccard token overlap and union-find clustering. Returns rumination scores and resolution hints.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_most_ruminative_theme",
        "description": "Return the rumination cluster with the highest rumination score from the last detect_rumination call.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_assess_efficacy",
        "description": "Assess domain-specific self-efficacy from success/failure outcome language in memory. Returns efficacy scores and trending per domain.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_trace_mood",
        "description": "Trace temporal emotional valence evolution across chronological memory segments. Returns mood arc, trend, volatility, and dominant emotion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Restrict to this domain (omit for all)."},
            },
        },
    },
    {
        "name": "emms_mood_trend",
        "description": "Return the mood trend string (improving/declining/stable/volatile/unknown) from the last trace_mood call.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # v0.26.0 — The Resilient Mind
    {
        "name": "emms_trace_adversity",
        "description": "Classify memories into 5 adversity types (loss, failure, rejection, threat, uncertainty) and return a severity-sorted report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain filter"},
            },
        },
    },
    {
        "name": "emms_dominant_adversity_type",
        "description": "Return the most common adversity type from the last trace_adversity call.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_measure_self_compassion",
        "description": "Measure self-kindness vs. self-harshness ratio across memory domains and return a SelfCompassionReport.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain filter"},
            },
        },
    },
    {
        "name": "emms_assess_resilience",
        "description": "Detect adversity windows and post-adversity recovery arcs; return a ResilienceReport with resilience_score and bounce_back_rate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain filter"},
            },
        },
    },
    {
        "name": "emms_resilience_bounce_back_rate",
        "description": "Return the fraction of adversity windows followed by genuine emotional recovery (0..1).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_set_defaults",
        "description": (
            "Set session-level defaults for emms_store — any field set here is automatically applied "
            "to every subsequent store call unless explicitly overridden. "
            "Call once at session start: emms_set_defaults({namespace, session_id}). "
            "Pass null to a field to clear it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Default namespace for all subsequent stores."},
                "session_id": {"type": "string", "description": "Default session ID for all subsequent stores."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Default obs_type for all subsequent stores.",
                },
                "domain": {"type": "string", "description": "Default domain for all subsequent stores."},
            },
            "required": [],
        },
    },
    {
        "name": "emms_store_batch",
        "description": "Store multiple experiences in a single call. Returns one result per item.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of experience objects — same fields as emms_store.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "domain": {"type": "string"},
                            "importance": {"type": "number"},
                            "title": {"type": "string"},
                            "subtitle": {"type": "string"},
                            "facts": {"type": "array", "items": {"type": "string"}},
                            "obs_type": {"type": "string"},
                            "concept_tags": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"},
                            "namespace": {"type": "string"},
                            "session_id": {"type": "string"},
                            "private": {"type": "boolean"},
                            "emotional_valence": {"type": "number"},
                            "citations": {"type": "array", "items": {"type": "string"}},
                            "update_mode": {"type": "string"},
                            "patch_key": {"type": "string"},
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["items"],
        },
    },
    {
        "name": "emms_migrate_namespace",
        "description": (
            "Bulk-migrate memories from one namespace to another. "
            "Optionally filter by domain or obs_type. "
            "Returns count of migrated memories."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_namespace": {"type": "string", "description": "Source namespace (e.g. 'default')."},
                "to_namespace": {"type": "string", "description": "Destination namespace."},
                "domain": {"type": "string", "description": "Only migrate memories in this domain."},
                "obs_type": {
                    "type": "string",
                    "enum": ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
                    "description": "Only migrate memories of this observation type.",
                },
                "dry_run": {"type": "boolean", "default": False, "description": "If true, report what would be migrated without changing anything."},
            },
            "required": ["from_namespace", "to_namespace"],
        },
    },
    {
        "name": "emms_wake_up",
        "description": (
            "Session startup tool — returns everything needed to resume work: "
            "top memories for the namespace, recent session activity, pending intentions, "
            "and active goals. Call this once at the start of each conversation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Project namespace to focus on (e.g. 'emms-sdk')."},
                "query": {"type": "string", "default": "", "description": "Optional focus query to surface relevant memories."},
                "top_memories": {"type": "integer", "default": 8, "description": "Max top memories to return."},
                "recent_sessions": {"type": "integer", "default": 3, "description": "How many recent sessions to summarise."},
            },
            "required": [],
        },
    },
    {
        "name": "emms_list_namespaces",
        "description": "List all namespaces present in memory, with memory count and top domains per namespace.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "emms_generate_context",
        "description": (
            "Generate a formatted consciousness context block for Claude system prompts. "
            "Assembles temporal orientation, coherence budget, active goals, top memories, "
            "and market wisdom into a single Markdown block ready to prepend to a system prompt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "extra_context": {"type": "string", "default": "", "description": "Additional session focus context to append."},
                "namespace": {"type": "string", "description": "EMMS namespace to scope memory retrieval."},
                "run_contradiction_scan": {"type": "boolean", "default": True, "description": "Run contradiction awareness scan for coherence budget."},
                "top_memories": {"type": "integer", "default": 3, "description": "Number of top memories to include."},
                "full": {"type": "boolean", "default": True, "description": "If false, return only the minimal one-line hint."},
            },
            "required": [],
        },
    },
    {
        "name": "emms_consciousness_metrics",
        "description": (
            "Compute all Phase 7 consciousness metrics in one call: "
            "ICS (Identity Coherence Score), TAI (Temporal Awareness Index), "
            "CRR (Contradiction Resolution Rate), and Coherence Budget. "
            "Appends result to ~/.emms/metrics_log.jsonl. "
            "Use this at session start or end to track consciousness development over time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID for CRR tracking (optional)."},
                "include_dashboard": {"type": "boolean", "default": False, "description": "Include formatted dashboard text in response."},
            },
            "required": [],
        },
    },
]


def _full_item_dict(item: "Any") -> dict:
    """Serialise a MemoryItem to a complete, JSON-safe dict (no truncation)."""
    exp = item.experience
    return {
        "id": item.id,
        "experience_id": exp.id,
        "content": exp.content,
        "title": exp.title,
        "subtitle": exp.subtitle,
        "facts": exp.facts,
        "domain": exp.domain,
        "importance": exp.importance,
        "confidence": exp.confidence,
        "namespace": exp.namespace,
        "session_id": exp.session_id,
        "obs_type": exp.obs_type.value if exp.obs_type else None,
        "concept_tags": [t.value for t in exp.concept_tags],
        "emotional_valence": exp.emotional_valence,
        "emotional_intensity": exp.emotional_intensity,
        "novelty": exp.novelty,
        "entities": exp.entities,
        "relationships": exp.relationships,
        "metadata": exp.metadata or None,
        "stored_at": exp.timestamp,
        "tier": item.tier.value,
        "strength": getattr(item, "strength", None),
        "last_accessed": item.last_accessed,
        "access_count": item.access_count,
        "memory_strength": getattr(item, "memory_strength", None),
        "consolidation_score": getattr(item, "consolidation_score", None),
        "files_read": exp.files_read,
        "files_modified": exp.files_modified,
    }


def _result_dict(r: "Any") -> dict:
    """Convert any RetrievalResult to the standard full response dict.

    Single source of truth for retrieve handler responses — eliminates per-handler
    field drift.  Extra handler-specific fields (e.g. emotional_proximity,
    activation_score) should be merged in via ``{**_result_dict(r), "extra": ...}``.
    """
    exp = r.memory.experience
    return {
        "id": r.memory.id,
        "experience_id": exp.id,
        "content": exp.content,
        "title": exp.title,
        "subtitle": exp.subtitle,
        "facts": exp.facts,
        "obs_type": exp.obs_type.value if exp.obs_type else None,
        "concept_tags": [t.value for t in exp.concept_tags],
        "namespace": exp.namespace,
        "domain": exp.domain,
        "importance": exp.importance,
        "confidence": exp.confidence,
        "session_id": exp.session_id,
        "stored_at": exp.timestamp,
        "score": r.score,
        "tier": r.source_tier.value,
        "strategy": getattr(r, "strategy", None),
        "explanation": getattr(r, "explanation", None),
    }


class EMCPServer:
    """MCP adapter for EMMS — routes tool calls to EMMS methods.

    Parameters
    ----------
    emms:
        A fully constructed ``EMMS`` instance to delegate to.
    """

    def __init__(self, emms: "Any", *, save_path: "Any | None" = None, auto_save_every: int = 10) -> None:
        self.emms = emms
        self._save_path = save_path
        self._auto_save_every = auto_save_every
        self._store_count = 0
        # Session-level defaults applied to every emms_store call (overridable per-call)
        self._store_defaults: dict[str, Any] = {}

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
                def _tool(kwargs: dict = None, flat: dict = None, **rest: Any) -> Any:
                    # Claude Code sends args in one of three ways:
                    #   {"kwargs": {...}}  → kwargs is the dict (legacy wrapping)
                    #   {"flat": {...}}    → flat is the dict (current Claude Code MCP client)
                    #   direct kwargs      → rest has the keys (direct MCP callers / tests)
                    args: dict[str, Any] = {}
                    if kwargs is not None:
                        args.update(kwargs)
                    if flat is not None:
                        args.update(flat)
                    args.update(rest)
                    return self.handle(n, args)
                _tool.__name__ = n
                _tool.__doc__ = defn.get("description", "")
                return _tool

            mcp_app.tool(name=name)(_make_tool(name, handler))

    # ------------------------------------------------------------------
    # Handlers (private)
    # ------------------------------------------------------------------

    def _rebuild_graph_bg(self) -> None:
        """Background helper: rebuild association graph without blocking callers."""
        try:
            self.emms.build_association_graph()
            logger.debug("Background association graph rebuild complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Background graph rebuild failed: %s", exc)

    def _handle_store(self, args: dict[str, Any]) -> dict[str, Any]:
        from emms.core.models import Experience, ObsType, ConceptTag

        # Merge session defaults (explicit args take precedence)
        args = {**self._store_defaults, **args}

        # obs_type — accept string or None, coerce to enum
        obs_type_raw = args.get("obs_type")
        obs_type = ObsType(obs_type_raw) if obs_type_raw else None

        # concept_tags — accept list[str], coerce each to ConceptTag
        concept_tags_raw = args.get("concept_tags") or []
        concept_tags = [ConceptTag(t) for t in concept_tags_raw if t]

        exp = Experience(
            content=args["content"],
            domain=args.get("domain", "general"),
            importance=float(args.get("importance", 0.5)),
            title=args.get("title"),
            subtitle=args.get("subtitle"),
            facts=args.get("facts", []),
            files_read=args.get("files_read", []),
            files_modified=args.get("files_modified", []),
            citations=args.get("citations", []),
            emotional_valence=float(args.get("emotional_valence", 0.0)),
            obs_type=obs_type,
            concept_tags=concept_tags,
            confidence=float(args.get("confidence", 1.0)),
            namespace=args.get("namespace", "default"),
            session_id=args.get("session_id"),
            private=bool(args.get("private", False)),
            update_mode=args.get("update_mode", "insert"),
            patch_key=args.get("patch_key"),
        )
        # Patch semantics: find existing memory by patch_key / title, with embedding fallback
        if exp.update_mode == "patch":
            match_key = exp.patch_key or exp.title
            if match_key:
                best_item = None
                # 1. Exact match on patch_key or title
                for _, store in self.emms.memory._iter_tiers():
                    for item in store:
                        e = item.experience
                        if not item.is_superseded and (e.patch_key == match_key or e.title == match_key):
                            best_item = item
                            break
                    if best_item:
                        break
                # 2. Fuzzy fallback: find closest memory by embedding similarity
                if best_item is None:
                    embedder = getattr(self.emms.memory, "embedder", None)
                    if embedder is not None:
                        from emms.core.embeddings import cosine_similarity
                        key_vec = embedder.embed(match_key)
                        mem_cache = getattr(self.emms.memory, "_embeddings", {})
                        best_score = 0.85  # minimum threshold for fuzzy patch
                        for _, store in self.emms.memory._iter_tiers():
                            for item in store:
                                if item.is_superseded:
                                    continue
                                stored = mem_cache.get(item.experience.id)
                                if stored is None:
                                    continue
                                sim = cosine_similarity(key_vec, stored)
                                if sim > best_score:
                                    best_score = sim
                                    best_item = item
                if best_item is not None:
                    best_item.superseded_by = "patch"

        store_result = self.emms.store(exp)
        # emms.store() returns a dict, not a MemoryItem
        item = store_result

        # Auto-save every N stores so in-flight memories survive process death
        self._store_count += 1
        if (
            self._save_path is not None
            and self._auto_save_every > 0
            and self._store_count % self._auto_save_every == 0
        ):
            try:
                self.emms.save(str(self._save_path))
                logger.debug("Auto-saved after %d stores", self._store_count)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Auto-save failed: %s", exc)

        # Auto-rebuild association graph every 20 stores — runs in a background
        # thread so it never blocks the store response (can take several seconds
        # on large corpora).
        if self._store_count % 20 == 0:
            threading.Thread(target=self._rebuild_graph_bg, daemon=True).start()

        return {
            "memory_id": item["memory_id"],
            "experience_id": item["experience_id"],
            "tier": item["tier"],
            "namespace": exp.namespace,
            "title": exp.title,
            "stored_at": exp.timestamp,
        }

    def _handle_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        from emms.core.models import ObsType, ConceptTag
        obs_type_raw = args.get("obs_type")
        obs_type = ObsType(obs_type_raw) if obs_type_raw else None
        concept_tags_raw = args.get("concept_tags") or []
        concept_tags = [ConceptTag(t) for t in concept_tags_raw if t] or None
        results = self.emms.retrieve_filtered(
            args["query"],
            max_results=args.get("max_results", 10),
            namespace=args.get("namespace"),
            obs_type=obs_type,
            domain=args.get("domain"),
            min_importance=args.get("min_importance"),
            sort_by=args.get("sort_by", "relevance"),
            concept_tags=concept_tags,
        )
        return {"results": [_result_dict(r) for r in results]}

    def _handle_search_compact(self, args: dict[str, Any]) -> dict[str, Any]:
        from emms.retrieval.strategies import EnsembleRetriever
        from emms.core.models import ObsType, ConceptTag

        namespace = args.get("namespace")
        obs_type_raw = args.get("obs_type")
        obs_type = ObsType(obs_type_raw) if obs_type_raw else None

        concept_tags_raw = args.get("concept_tags") or []
        concept_tags = [ConceptTag(t) for t in concept_tags_raw if t] or None

        retriever = EnsembleRetriever.from_balanced(embedder=self.emms.memory.embedder)
        items = [
            item
            for _, store in self.emms.memory._iter_tiers()
            for item in store
            if not item.is_expired and not item.is_superseded and not item.experience.private
            and (namespace is None or item.experience.namespace == namespace)
            and (concept_tags is None or any(t in item.experience.concept_tags for t in concept_tags))
        ]
        # Build lookup so we can annotate compact results with namespace/session_id
        # without requiring a second pass through the full memory store.
        id_to_meta = {
            item.id: {
                "namespace": item.experience.namespace,
                "session_id": item.experience.session_id,
            }
            for item in items
        }
        compact = retriever.search_compact(
            args["query"], items,
            max_results=args.get("max_results", 20),
            obs_type=obs_type,
        )
        return {
            "results": [
                {
                    "id": c.id,
                    "snippet": c.snippet,
                    "domain": c.domain,
                    "score": c.score,
                    "tier": c.tier.value,
                    "token_estimate": c.token_estimate,
                    **id_to_meta.get(c.id, {}),
                }
                for c in compact
            ]
        }

    def _handle_set_defaults(self, args: dict[str, Any]) -> dict[str, Any]:
        """Set session-level defaults applied to every subsequent emms_store call."""
        # A null value clears that default; a non-null value sets/updates it
        for key in ("namespace", "session_id", "obs_type", "domain"):
            if key in args:
                if args[key] is None:
                    self._store_defaults.pop(key, None)
                else:
                    self._store_defaults[key] = args[key]
        return {"defaults": dict(self._store_defaults)}

    def _handle_store_batch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store multiple experiences in one call, inheriting session defaults."""
        items = args.get("items", [])
        results = []
        errors = []
        for i, item_args in enumerate(items):
            try:
                result = self._handle_store(item_args)
                results.append({"index": i, "ok": True, **result})
            except Exception as exc:  # noqa: BLE001
                errors.append({"index": i, "ok": False, "error": str(exc)})
        return {
            "stored": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }

    def _handle_migrate_namespace(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bulk-migrate memories from one namespace to another in-place."""
        from emms.core.models import ObsType
        from_ns = args["from_namespace"]
        to_ns = args["to_namespace"]
        domain_filter = args.get("domain")
        obs_type_raw = args.get("obs_type")
        obs_type_filter = ObsType(obs_type_raw) if obs_type_raw else None
        dry_run = bool(args.get("dry_run", False))

        migrated = []
        for _, store in self.emms.memory._iter_tiers():
            for item in store:
                if item.is_superseded or item.is_expired:
                    continue
                exp = item.experience
                if exp.namespace != from_ns:
                    continue
                if domain_filter and exp.domain != domain_filter:
                    continue
                if obs_type_filter and exp.obs_type != obs_type_filter:
                    continue
                migrated.append({
                    "id": item.id,
                    "title": exp.title,
                    "domain": exp.domain,
                    "obs_type": exp.obs_type.value if exp.obs_type else None,
                })
                if not dry_run:
                    exp.namespace = to_ns

        return {
            "dry_run": dry_run,
            "from_namespace": from_ns,
            "to_namespace": to_ns,
            "migrated_count": len(migrated),
            "migrated": migrated,
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
        from collections import Counter
        embedder = self.emms.memory.embedder
        vec_index = self.emms.memory._vec_index
        # Build compact per-namespace count (skip private/expired/superseded)
        ns_counter: Counter = Counter()
        for _, store in self.emms.memory._iter_tiers():
            for item in store:
                if item.is_superseded or item.is_expired or item.experience.private:
                    continue
                ns_counter[item.experience.namespace or "default"] += 1
        return {
            "stats": {
                **self.emms.stats,
                # Embedder / semantic search health
                "embedder": type(embedder).__name__ if embedder else "None",
                "embedder_dim": getattr(embedder, "dim", None),
                "vec_index_size": len(vec_index) if vec_index is not None else 0,
                "semantic_search_active": embedder is not None and vec_index is not None,
                # Per-namespace memory counts
                "namespace_counts": dict(ns_counter.most_common()),
                # Active session defaults
                "store_defaults": dict(self._store_defaults),
            }
        }

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
        from emms.core.models import ObsType, ConceptTag

        obs_type_raw = args.get("obs_type")
        obs_type = ObsType(obs_type_raw) if obs_type_raw else None

        concept_tags_raw = args.get("concept_tags") or []
        concept_tags = [ConceptTag(t) for t in concept_tags_raw if t] or None

        # query is optional — empty string triggers filter-only (no semantic ranking)
        query = args.get("query", "")

        results = self.emms.retrieve_filtered(
            query,
            max_results=args.get("max_results", 10),
            namespace=args.get("namespace"),
            obs_type=obs_type,
            domain=args.get("domain"),
            session_id=args.get("session_id"),
            since=args.get("since"),
            until=args.get("until"),
            min_confidence=args.get("min_confidence"),
            min_importance=args.get("min_importance"),
            sort_by=args.get("sort_by", "relevance"),
            concept_tags=concept_tags,
        )
        return {"results": [_result_dict(r) for r in results]}

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
        return {"results": [_result_dict(r) for r in results]}

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
        # Auto-enable adaptive retriever (idempotent — safe to call if already enabled)
        try:
            self.emms.enable_adaptive_retrieval()
        except Exception:
            pass  # Already enabled

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
            "results": [_result_dict(r) for r in results],
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
                result["item"] = _full_item_dict(item)
        elif args.get("experience_id"):
            item = self.emms.get_memory_by_experience_id(args["experience_id"])
            result = {"found": item is not None}
            if item:
                result["item"] = _full_item_dict(item)
        elif args.get("content"):
            items = self.emms.find_memories_by_content(args["content"])
            result = {"found": len(items) > 0, "count": len(items), "items": [
                _full_item_dict(it) for it in items[:5]
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
                    "experience_id": e.item.experience.id,
                    "content": e.item.experience.content,
                    "title": e.item.experience.title,
                    "facts": e.item.experience.facts,
                    "obs_type": e.item.experience.obs_type.value if e.item.experience.obs_type else None,
                    "namespace": e.item.experience.namespace,
                    "domain": e.item.experience.domain,
                    "stored_at": e.item.experience.timestamp,
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
            "results": [_result_dict(r) for r in plan.merged_results[:20]],
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
                    **_result_dict(r),
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
            return {"ok": False, "error": f"Could not parse bridge_json: {e}"}
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
                    "insight": b.insight_content,
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
                    **_result_dict(r),
                    "activation_score": round(r.activation_score, 4),
                    "steps_from_seed": r.steps_from_seed,
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

    def _handle_metacognition_report(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.metacognition_report(
            max_contradictions=int(args.get("max_contradictions", 5)),
            confidence_threshold_high=float(args.get("confidence_threshold_high", 0.65)),
            confidence_threshold_low=float(args.get("confidence_threshold_low", 0.3)),
        )
        return {
            "total_memories": report.total_memories,
            "mean_confidence": round(report.mean_confidence, 4),
            "high_confidence_count": report.high_confidence_count,
            "low_confidence_count": report.low_confidence_count,
            "knowledge_gaps": report.knowledge_gaps,
            "recommendations": report.recommendations,
            "contradictions": len(report.contradictions),
            "domain_profiles": [
                {
                    "domain": p.domain,
                    "memory_count": p.memory_count,
                    "mean_confidence": round(p.mean_confidence, 3),
                    "coverage_score": round(p.coverage_score, 3),
                }
                for p in report.domain_profiles[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_knowledge_map(self, args: dict[str, Any]) -> dict[str, Any]:
        profiles = self.emms.knowledge_map()
        return {
            "domains": [
                {
                    "domain": p.domain,
                    "memory_count": p.memory_count,
                    "mean_confidence": round(p.mean_confidence, 3),
                    "coverage_score": round(p.coverage_score, 3),
                    "mean_importance": round(p.mean_importance, 3),
                    "mean_strength": round(p.mean_strength, 3),
                }
                for p in profiles
            ],
            "total_domains": len(profiles),
        }

    def _handle_find_contradictions(self, args: dict[str, Any]) -> dict[str, Any]:
        pairs = self.emms.find_contradictions(
            max_pairs=int(args.get("max_pairs", 10)),
        )
        return {
            "contradictions": [
                {
                    "memory_a_id": p.memory_a_id,
                    "memory_b_id": p.memory_b_id,
                    "semantic_overlap": round(p.semantic_overlap, 4),
                    "valence_conflict": round(p.valence_conflict, 4),
                    "contradiction_score": round(p.contradiction_score, 4),
                    "excerpt_a": p.excerpt_a[:80],
                    "excerpt_b": p.excerpt_b[:80],
                }
                for p in pairs
            ],
            "total": len(pairs),
        }

    def _handle_intend(self, args: dict[str, Any]) -> dict[str, Any]:
        intention = self.emms.intend(
            content=args["content"],
            trigger_context=args["trigger_context"],
            priority=float(args.get("priority", 0.5)),
        )
        return {
            "intention_id": intention.id,
            "content": intention.content,
            "trigger_context": intention.trigger_context,
            "priority": intention.priority,
            "created_at": intention.created_at,
        }

    def _handle_check_intentions(self, args: dict[str, Any]) -> dict[str, Any]:
        activations = self.emms.check_intentions(
            current_context=args.get("current_context", ""),
        )
        return {
            "activated": [
                {
                    "intention_id": a.intention.id,
                    "content": a.intention.content,
                    "activation_score": round(a.activation_score, 4),
                    "trigger_overlap": round(a.trigger_overlap, 4),
                    "days_pending": round(a.days_pending, 2),
                    "priority": a.intention.priority,
                }
                for a in activations
            ],
            "total_activated": len(activations),
            "total_pending": len(self.emms.pending_intentions()),
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

    # v0.14.0 handlers

    def _handle_open_episode(self, args: dict[str, Any]) -> dict[str, Any]:
        ep = self.emms.open_episode(
            session_id=args.get("session_id"),
            topic=args.get("topic", ""),
        )
        return {
            "episode_id": ep.id,
            "session_id": ep.session_id,
            "topic": ep.topic,
            "opened_at": ep.opened_at,
        }

    def _handle_close_episode(self, args: dict[str, Any]) -> dict[str, Any]:
        ep = self.emms.close_episode(
            episode_id=args.get("episode_id"),
            outcome=args.get("outcome", ""),
        )
        if ep is None:
            return {"closed": False, "reason": "No open episode found."}
        return {
            "closed": True,
            "episode_id": ep.id,
            "outcome": ep.outcome,
            "turn_count": ep.turn_count,
            "duration_seconds": ep.duration_seconds,
            "mean_valence": round(ep.mean_valence, 4),
            "peak_valence": round(ep.peak_valence, 4),
            "summary": ep.summary(),
        }

    def _handle_recent_episodes(self, args: dict[str, Any]) -> dict[str, Any]:
        episodes = self.emms.recent_episodes(n=int(args.get("n", 10)))
        return {
            "episodes": [
                {
                    "episode_id": ep.id,
                    "topic": ep.topic,
                    "turn_count": ep.turn_count,
                    "is_open": ep.is_open,
                    "duration_seconds": ep.duration_seconds,
                    "mean_valence": round(ep.mean_valence, 4),
                    "outcome": ep.outcome,
                    "summary": ep.summary(),
                }
                for ep in episodes
            ],
            "total": len(episodes),
        }

    def _handle_extract_schemas(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.extract_schemas(
            domain=args.get("domain"),
            max_schemas=args.get("max_schemas"),
        )
        return {
            "total_memories_analyzed": report.total_memories_analyzed,
            "schemas_found": report.schemas_found,
            "duration_seconds": round(report.duration_seconds, 4),
            "schemas": [
                {
                    "schema_id": s.id,
                    "domain": s.domain,
                    "pattern": s.pattern,
                    "keywords": s.keywords,
                    "confidence": round(s.confidence, 4),
                    "support": len(s.supporting_memory_ids),
                }
                for s in report.schemas
            ],
            "summary": report.summary(),
        }

    def _handle_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        memory_id = args.get("memory_id")
        domain = args.get("domain")
        below_conf = args.get("below_confidence")
        rate = float(args.get("suppression_rate", 0.4))
        hard = bool(args.get("hard", False))

        if memory_id and hard:
            # Hard delete: permanently remove from all tiers and every index.
            # This is irreversible — use for wrong/outdated memories only.
            item = self.emms.index.get_by_id(memory_id)
            if item is None:
                return {"success": False, "reason": f"Memory {memory_id!r} not found."}
            exp_id = item.experience.id
            mem = self.emms.memory
            # Remove from deque tiers (working, short_term)
            for tier_deque in (mem.working, mem.short_term):
                filtered = [i for i in tier_deque if i.id != memory_id]
                if len(filtered) < len(tier_deque):
                    tier_deque.clear()
                    tier_deque.extend(filtered)
            # Remove from dict tiers (long_term, semantic)
            for tier_dict in (mem.long_term, mem.semantic):
                tier_dict.pop(memory_id, None)
            # Clean up all lookup structures
            self.emms.index.remove(memory_id)
            mem._items_by_exp_id.pop(exp_id, None)
            mem._embeddings.pop(exp_id, None)
            if mem._vec_index is not None:
                mem._vec_index.remove(exp_id)
            # Remove experience_id from word index
            for word_set in mem._word_index.values():
                word_set.discard(exp_id)
            return {"success": True, "hard_deleted": True, "memory_id": memory_id}

        if memory_id:
            from emms.memory.forgetting import MotivatedForgetting
            mf = MotivatedForgetting(self.emms.memory, suppression_rate=rate)
            result = mf.suppress(memory_id)
            if result is None:
                return {"success": False, "reason": f"Memory {memory_id!r} not found."}
            return {
                "success": True,
                "memory_id": result.memory_id,
                "pruned": result.pruned,
                "old_strength": round(result.old_strength, 4),
                "new_strength": round(result.new_strength, 4),
                "reason": result.reason,
            }

        if domain:
            report = self.emms.forget_domain(domain, rate=rate)
            return {
                "total_targeted": report.total_targeted,
                "suppressed": report.suppressed,
                "pruned": report.pruned,
                "summary": report.summary(),
            }

        if below_conf is not None:
            report = self.emms.forget_below_confidence(threshold=float(below_conf))
            return {
                "total_targeted": report.total_targeted,
                "suppressed": report.suppressed,
                "pruned": report.pruned,
                "summary": report.summary(),
            }

        return {"success": False, "reason": "Provide memory_id, domain, or below_confidence."}

    # v0.15.0 handlers

    def _handle_reflect(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.reflect(
            session_id=args.get("session_id"),
            domain=args.get("domain"),
            lookback_episodes=int(args.get("lookback_episodes", 5)),
        )
        return {
            "session_id": report.session_id,
            "memories_reviewed": report.memories_reviewed,
            "episodes_reviewed": report.episodes_reviewed,
            "lessons_count": len(report.lessons),
            "new_memory_ids": report.new_memory_ids,
            "lessons": [
                {
                    "lesson_id": l.id,
                    "domain": l.domain,
                    "lesson_type": l.lesson_type,
                    "confidence": round(l.confidence, 4),
                    "content": l.content,
                    "support": len(l.supporting_ids),
                }
                for l in report.lessons
            ],
            "open_questions": report.open_questions,
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    def _handle_weave_narrative(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.weave_narrative(
            domain=args.get("domain"),
            max_threads=int(args.get("max_threads", 8)),
        )
        return {
            "total_threads": report.total_threads,
            "total_segments": report.total_segments,
            "threads": [
                {
                    "thread_id": t.id,
                    "theme": t.theme,
                    "domain": t.domain,
                    "segments": len(t.segments),
                    "span_seconds": round(t.span_seconds, 1),
                    "story_excerpt": t.story()[:200],
                    "arc": [round(v, 3) for v in t.arc],
                }
                for t in report.threads
            ],
            "summary": report.summary(),
        }

    def _handle_narrative_threads(self, args: dict[str, Any]) -> dict[str, Any]:
        threads = self.emms.narrative_threads(domain=args.get("domain"))
        return {
            "threads": [
                {
                    "thread_id": t.id,
                    "theme": t.theme,
                    "domain": t.domain,
                    "segments": len(t.segments),
                    "span_seconds": round(t.span_seconds, 1),
                    "story": t.story(),
                    "summary": t.summary(),
                }
                for t in threads
            ],
            "total": len(threads),
        }

    def _handle_source_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.source_audit(
            flag_threshold=float(args.get("flag_threshold", 0.5)),
        )
        return {
            "total_audited": report.total_audited,
            "flagged_count": report.flagged_count,
            "source_distribution": report.source_distribution,
            "high_risk": [
                {
                    "memory_id": e.memory_id,
                    "source_type": e.source_type,
                    "confidence": round(e.source_confidence, 4),
                    "flag_reason": e.flag_reason,
                    "excerpt": e.content_excerpt[:80],
                }
                for e in report.high_risk_entries[:int(args.get("max_flagged", 20))]
            ],
            "summary": report.summary(),
        }

    def _handle_tag_source(self, args: dict[str, Any]) -> dict[str, Any]:
        tag = self.emms.tag_memory_source(
            memory_id=args["memory_id"],
            source_type=args["source_type"],
            confidence=float(args.get("confidence", 0.8)),
            note=args.get("note", ""),
        )
        return {
            "memory_id": tag.memory_id,
            "source_type": tag.source_type,
            "confidence": round(tag.confidence, 4),
            "provenance_note": tag.provenance_note,
            "tagged_at": tag.tagged_at,
        }

    # v0.16.0 handlers

    def _handle_curiosity_report(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.curiosity_scan(domain=args.get("domain"))
        return {
            "total_domains_scanned": report.total_domains_scanned,
            "goals_generated": report.goals_generated,
            "top_curious_domains": report.top_curious_domains,
            "goals": [
                {
                    "id": g.id,
                    "question": g.question,
                    "domain": g.domain,
                    "urgency": g.urgency,
                    "gap_type": g.gap_type,
                }
                for g in report.goals
            ],
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    def _handle_exploration_goals(self, args: dict[str, Any]) -> dict[str, Any]:
        goals = self.emms.exploration_goals()
        return {
            "count": len(goals),
            "goals": [
                {
                    "id": g.id,
                    "question": g.question,
                    "domain": g.domain,
                    "urgency": g.urgency,
                    "gap_type": g.gap_type,
                    "explored": g.explored,
                }
                for g in goals
            ],
        }

    def _handle_revise_beliefs(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.revise_beliefs(
            new_memory_id=args.get("new_memory_id"),
            domain=args.get("domain"),
            max_revisions=int(args.get("max_revisions", 8)),
        )
        return {
            "total_checked": report.total_checked,
            "conflicts_found": report.conflicts_found,
            "revisions_made": report.revisions_made,
            "records": [
                {
                    "id": r.id,
                    "revision_type": r.revision_type,
                    "conflict_score": r.conflict_score,
                    "new_content": r.new_content,
                }
                for r in report.records
            ],
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    def _handle_decay_report(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.memory_decay_report(domain=args.get("domain"))
        return {
            "total_processed": report.total_processed,
            "decayed": report.decayed,
            "mean_retention": report.mean_retention,
            "applied": report.applied,
            "duration_seconds": round(report.duration_seconds, 4),
            "top_records": [
                {
                    "memory_id": r.memory_id,
                    "domain": r.domain,
                    "old_strength": r.old_strength,
                    "new_strength": r.new_strength,
                    "retention": r.retention,
                    "stability_days": r.stability,
                    "days_since_access": r.days_since_access,
                }
                for r in report.records[:10]
            ],
            "summary": report.summary(),
        }

    def _handle_apply_decay(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.apply_memory_decay(
            domain=args.get("domain"),
            prune=bool(args.get("prune", False)),
        )
        return {
            "total_processed": report.total_processed,
            "decayed": report.decayed,
            "pruned": report.pruned,
            "mean_retention": report.mean_retention,
            "applied": report.applied,
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    # v0.17.0 handlers

    def _handle_push_goal(self, args: dict[str, Any]) -> dict[str, Any]:
        goal = self.emms.push_goal(
            content=args["content"],
            domain=str(args.get("domain", "general")),
            priority=float(args.get("priority", 0.5)),
            parent_id=args.get("parent_id"),
        )
        return {
            "id": goal.id,
            "content": goal.content,
            "domain": goal.domain,
            "priority": goal.priority,
            "status": goal.status,
            "parent_id": goal.parent_id,
            "summary": goal.summary(),
        }

    def _handle_active_goals(self, args: dict[str, Any]) -> dict[str, Any]:
        goals = self.emms.active_goals()
        return {
            "count": len(goals),
            "goals": [
                {
                    "id": g.id,
                    "content": g.content,
                    "domain": g.domain,
                    "priority": g.priority,
                    "status": g.status,
                }
                for g in goals
            ],
        }

    def _handle_complete_goal(self, args: dict[str, Any]) -> dict[str, Any]:
        ok = self.emms.complete_goal(
            goal_id=args["goal_id"],
            outcome_note=str(args.get("outcome_note", "")),
        )
        return {"ok": ok, "goal_id": args["goal_id"]}

    def _handle_spotlight_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text")
        keywords = args.get("keywords")
        if text or keywords:
            self.emms.update_spotlight(
                text=text,
                keywords=list(keywords) if keywords else None,
            )
        report = self.emms.spotlight_retrieve(k=int(args.get("k", 8)))
        return {
            "items_scored": report.items_scored,
            "top_domain": report.top_domain,
            "spotlight_keywords": report.spotlight_keywords,
            "results": [
                {
                    "memory_id": r.memory_id,
                    "domain": r.domain,
                    "attention_score": r.attention_score,
                    "content_excerpt": r.content_excerpt,
                }
                for r in report.results
            ],
            "summary": report.summary(),
        }

    def _handle_find_analogies(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.find_analogies(
            source_domain=args.get("source_domain"),
            target_domain=args.get("target_domain"),
        )
        return {
            "total_pairs_checked": report.total_pairs_checked,
            "analogies_found": report.analogies_found,
            "records": [
                {
                    "id": r.id,
                    "source_domain": r.source_domain,
                    "target_domain": r.target_domain,
                    "analogy_strength": r.analogy_strength,
                    "insight_content": r.insight_content,
                }
                for r in report.records
            ],
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    # v0.18.0 handlers

    def _handle_predict(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.predict(domain=args.get("domain"))
        return {
            "total_generated": report.total_generated,
            "confirmed": report.confirmed,
            "violated": report.violated,
            "pending": report.pending,
            "mean_surprise": report.mean_surprise,
            "predictions": [
                {
                    "id": p.id,
                    "content": p.content,
                    "domain": p.domain,
                    "confidence": p.confidence,
                    "outcome": p.outcome,
                }
                for p in report.predictions
            ],
            "summary": report.summary(),
        }

    def _handle_pending_predictions(self, args: dict[str, Any]) -> dict[str, Any]:
        preds = self.emms.pending_predictions()
        return {
            "count": len(preds),
            "predictions": [
                {
                    "id": p.id,
                    "content": p.content,
                    "domain": p.domain,
                    "confidence": p.confidence,
                }
                for p in preds
            ],
        }

    def _handle_blend_concepts(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.blend_concepts(
            domain_a=args.get("domain_a"),
            domain_b=args.get("domain_b"),
        )
        return {
            "total_pairs_tried": report.total_pairs_tried,
            "blends_created": report.blends_created,
            "concepts": [
                {
                    "id": c.id,
                    "source_domains": c.source_domains,
                    "blend_strength": c.blend_strength,
                    "emergent_properties": c.emergent_properties[:5],
                    "blend_content": c.blend_content,
                }
                for c in report.concepts
            ],
            "duration_seconds": round(report.duration_seconds, 4),
            "summary": report.summary(),
        }

    def _handle_project_future(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.project_future(
            domain=args.get("domain"),
            horizon_days=float(args.get("horizon_days", 30.0)),
        )
        return {
            "scenarios_generated": report.scenarios_generated,
            "total_memories_used": report.total_memories_used,
            "total_episodes_used": report.total_episodes_used,
            "mean_plausibility": report.mean_plausibility,
            "scenarios": [
                {
                    "id": s.id,
                    "domain": s.domain,
                    "plausibility": s.plausibility,
                    "emotional_valence": s.emotional_valence,
                    "horizon_days": s.projection_horizon,
                    "content": s.content,
                }
                for s in report.scenarios
            ],
            "summary": report.summary(),
        }

    def _handle_plausible_futures(self, args: dict[str, Any]) -> dict[str, Any]:
        scenarios = self.emms.most_plausible_futures(n=int(args.get("n", 3)))
        return {
            "count": len(scenarios),
            "scenarios": [
                {
                    "id": s.id,
                    "domain": s.domain,
                    "plausibility": s.plausibility,
                    "content": s.content,
                }
                for s in scenarios
            ],
        }

    # v0.19.0 handlers

    def _handle_regulate_emotions(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.regulate_emotions(domain=args.get("domain"))
        state = report.current_state
        return {
            "ok": True,
            "valence": state.valence,
            "arousal": state.arousal,
            "dominant_domain": state.dominant_domain,
            "sample_size": state.sample_size,
            "memories_assessed": report.memories_assessed,
            "reappraisals": len(report.reappraisals),
            "emotional_coherence": report.emotional_coherence,
            "mood_congruent_count": len(report.mood_congruent_ids),
            "summary": report.summary(),
        }

    def _handle_current_emotion(self, args: dict[str, Any]) -> dict[str, Any]:
        state = self.emms.current_emotional_state()
        if state is None:
            return {"ok": True, "state": None, "message": "No emotional state computed yet."}
        return {
            "ok": True,
            "valence": state.valence,
            "arousal": state.arousal,
            "dominant_domain": state.dominant_domain,
            "sample_size": state.sample_size,
        }

    def _handle_build_hierarchy(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.build_concept_hierarchy(domain=args.get("domain"))
        return {
            "ok": True,
            "total_concepts": report.total_concepts,
            "total_edges": report.total_edges,
            "max_depth": report.max_depth,
            "domains": report.domains,
            "nodes": [
                {
                    "label": n.label,
                    "level": n.level,
                    "domain": n.domain,
                    "abstraction_score": n.abstraction_score,
                    "children": len(n.children_ids),
                }
                for n in report.nodes[:20]
            ],
            "summary": report.summary(),
        }

    def _handle_concept_distance(self, args: dict[str, Any]) -> dict[str, Any]:
        label_a = args.get("label_a", "")
        label_b = args.get("label_b", "")
        distance = self.emms.concept_distance(label_a, label_b)
        return {
            "ok": True,
            "label_a": label_a,
            "label_b": label_b,
            "distance": distance,
            "connected": distance >= 0,
        }

    def _handle_update_self_model(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self.emms.update_self_model()
        return {
            "ok": True,
            "total_memories_analyzed": report.total_memories_analyzed,
            "beliefs_count": len(report.beliefs),
            "core_domains": report.core_domains,
            "dominant_valence": report.dominant_valence,
            "consistency_score": report.consistency_score,
            "capability_profile": report.capability_profile,
            "beliefs": [
                {
                    "id": b.id,
                    "domain": b.domain,
                    "confidence": b.confidence,
                    "valence": b.valence,
                    "content": b.content,
                }
                for b in report.beliefs[:8]
            ],
            "summary": report.summary(),
        }

    # v0.20.0 handlers

    def _handle_build_causal_map(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.build_causal_map(domain=domain)
        return {
            "ok": True,
            "total_concepts": report.total_concepts,
            "total_edges": report.total_edges,
            "most_influential": report.most_influential,
            "most_affected": report.most_affected,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "strength": e.strength,
                    "memory_count": len(e.memory_ids),
                }
                for e in report.edges[:10]
            ],
            "summary": report.summary(),
        }

    def _handle_effects_of(self, args: dict[str, Any]) -> dict[str, Any]:
        concept = args.get("concept", "")
        self.emms.build_causal_map()
        edges = self.emms.effects_of(concept)
        return {
            "ok": True,
            "concept": concept,
            "effects_count": len(edges),
            "effects": [
                {"target": e.target, "relation": e.relation, "strength": e.strength}
                for e in edges[:10]
            ],
        }

    def _handle_generate_counterfactuals(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        direction = args.get("direction", "both")
        report = self.emms.generate_counterfactuals(domain=domain, direction=direction)
        return {
            "ok": True,
            "total_memories_assessed": report.total_memories_assessed,
            "counterfactuals_generated": report.counterfactuals_generated,
            "mean_plausibility": report.mean_plausibility,
            "counterfactuals": [
                {
                    "id": c.id,
                    "direction": c.direction,
                    "valence_shift": c.valence_shift,
                    "plausibility": c.plausibility,
                    "domain": c.domain,
                    "content": c.counterfactual_content,
                }
                for c in report.counterfactuals[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_distill_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.distill_skills(domain=domain)
        return {
            "ok": True,
            "total_memories_analyzed": report.total_memories_analyzed,
            "skills_distilled": report.skills_distilled,
            "domains_covered": report.domains_covered,
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "domain": s.domain,
                    "confidence": s.confidence,
                    "preconditions": s.preconditions,
                    "outcomes": s.outcomes,
                    "description": s.description[:120],
                }
                for s in report.skills[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_best_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        goal = args.get("goal_description", "")
        self.emms.distill_skills()
        skill = self.emms.best_skill(goal)
        if skill is None:
            return {"ok": True, "found": False, "skill": None}
        return {
            "ok": True,
            "found": True,
            "skill": {
                "id": skill.id,
                "name": skill.name,
                "domain": skill.domain,
                "confidence": skill.confidence,
                "preconditions": skill.preconditions,
                "outcomes": skill.outcomes,
                "description": skill.description,
            },
        }

    # v0.21.0 handlers

    def _handle_build_perspectives(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.build_perspective_models(domain=domain)
        return {
            "ok": True,
            "total_agents": report.total_agents,
            "most_mentioned": report.most_mentioned,
            "total_memories_scanned": report.total_memories_scanned,
            "agents": [
                {
                    "name": a.name,
                    "mentions": a.mentions,
                    "mean_valence": a.mean_valence,
                    "domains": a.domains,
                    "statements": a.statements[:3],
                }
                for a in report.agents[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_agent_model(self, args: dict[str, Any]) -> dict[str, Any]:
        agent_name = args.get("agent_name", "")
        self.emms.build_perspective_models()
        model = self.emms.agent_model(agent_name)
        if model is None:
            return {"ok": True, "found": False, "agent": None}
        return {
            "ok": True,
            "found": True,
            "agent": {
                "name": model.name,
                "mentions": model.mentions,
                "mean_valence": model.mean_valence,
                "domains": model.domains,
                "statements": model.statements[:5],
            },
        }

    def _handle_compute_trust(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.compute_trust(domain=domain)
        return {
            "ok": True,
            "total_sources": report.total_sources,
            "most_trusted": report.most_trusted,
            "least_trusted": report.least_trusted,
            "scores": [
                {
                    "source": ts.source,
                    "trust": ts.trust,
                    "memory_count": ts.memory_count,
                    "mean_importance": ts.mean_importance,
                    "valence_stability": ts.valence_stability,
                }
                for ts in report.scores[:10]
            ],
            "summary": report.summary(),
        }

    def _handle_extract_norms(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.extract_norms(domain=domain)
        return {
            "ok": True,
            "total_norms": report.total_norms,
            "prescriptive_count": report.prescriptive_count,
            "prohibitive_count": report.prohibitive_count,
            "domains_covered": report.domains_covered,
            "norms": [
                {
                    "id": n.id,
                    "polarity": n.polarity,
                    "keyword": n.keyword,
                    "subject": n.subject,
                    "confidence": n.confidence,
                    "domain": n.domain,
                    "content": n.content,
                }
                for n in report.norms[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_check_norm(self, args: dict[str, Any]) -> dict[str, Any]:
        behavior = args.get("behavior", "")
        self.emms.extract_norms()
        norms = self.emms.check_norm(behavior)
        return {
            "ok": True,
            "behavior": behavior,
            "relevant_norms_count": len(norms),
            "norms": [
                {
                    "id": n.id,
                    "polarity": n.polarity,
                    "keyword": n.keyword,
                    "subject": n.subject,
                    "confidence": n.confidence,
                    "content": n.content,
                }
                for n in norms
            ],
        }

    # v0.22.0 handlers

    def _handle_assess_novelty(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.assess_novelty(domain=domain)
        return {
            "ok": True,
            "total_assessed": report.total_assessed,
            "high_novelty_count": report.high_novelty_count,
            "mean_novelty": report.mean_novelty,
            "scores": [
                {
                    "memory_id": s.memory_id,
                    "novelty": s.novelty,
                    "domain": s.domain,
                    "rare_tokens": s.rare_tokens[:5],
                    "content_excerpt": s.content_excerpt[:60],
                }
                for s in report.scores[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_most_novel(self, args: dict[str, Any]) -> dict[str, Any]:
        n = int(args.get("n", 5))
        self.emms.assess_novelty()
        scores = self.emms.most_novel(n=n)
        return {
            "ok": True,
            "count": len(scores),
            "scores": [
                {
                    "memory_id": s.memory_id,
                    "novelty": s.novelty,
                    "domain": s.domain,
                    "rare_tokens": s.rare_tokens[:5],
                    "content_excerpt": s.content_excerpt[:60],
                }
                for s in scores
            ],
        }

    def _handle_invent_concepts(self, args: dict[str, Any]) -> dict[str, Any]:
        n = int(args.get("n", 8))
        report = self.emms.invent_concepts(n=n)
        return {
            "ok": True,
            "total_concepts": report.total_concepts,
            "mean_originality": report.mean_originality,
            "domain_pairs": [list(p) for p in report.domain_pairs[:5]],
            "concepts": [
                {
                    "id": c.id,
                    "token_a": c.token_a,
                    "domain_a": c.domain_a,
                    "token_b": c.token_b,
                    "domain_b": c.domain_b,
                    "originality_score": c.originality_score,
                    "description": c.description[:120],
                }
                for c in report.concepts[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_abstract_principles(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.abstract_principles(domain=domain)
        return {
            "ok": True,
            "total_principles": report.total_principles,
            "mean_generality": report.mean_generality,
            "domains_abstracted": report.domains_abstracted,
            "principles": [
                {
                    "id": p.id,
                    "label": p.label,
                    "domain": p.domain,
                    "generality_score": p.generality_score,
                    "mean_valence": p.mean_valence,
                    "mean_importance": p.mean_importance,
                    "description": p.description[:120],
                }
                for p in report.principles[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_best_principle(self, args: dict[str, Any]) -> dict[str, Any]:
        description = args.get("description", "")
        self.emms.abstract_principles()
        principle = self.emms.best_principle(description)
        if principle is None:
            return {"ok": True, "found": False, "principle": None}
        return {
            "ok": True,
            "found": True,
            "principle": {
                "id": principle.id,
                "label": principle.label,
                "domain": principle.domain,
                "generality_score": principle.generality_score,
                "mean_valence": principle.mean_valence,
                "mean_importance": principle.mean_importance,
                "description": principle.description,
            },
        }

    # ------------------------------------------------------------------
    # v0.23.0 handlers
    # ------------------------------------------------------------------

    def _handle_map_values(self, args: dict[str, Any]) -> dict[str, Any]:
        category = args.get("category")
        report = self.emms.map_values(category=category)
        return {
            "ok": True,
            "total_values": report.total_values,
            "dominant_category": report.dominant_category,
            "mean_strength": report.mean_strength,
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "category": v.category,
                    "strength": v.strength,
                    "description": v.description[:100],
                }
                for v in report.values[:10]
            ],
            "summary": report.summary(),
        }

    def _handle_values_for_category(self, args: dict[str, Any]) -> dict[str, Any]:
        category = args.get("category", "")
        self.emms.map_values()
        values = self.emms.values_for_category(category)
        return {
            "ok": True,
            "category": category,
            "count": len(values),
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "strength": v.strength,
                    "description": v.description[:100],
                }
                for v in values
            ],
        }

    def _handle_reason_morally(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        report = self.emms.reason_morally(domain=domain)
        return {
            "ok": True,
            "total_assessed": report.total_assessed,
            "dominant_framework_overall": report.dominant_framework_overall,
            "mean_moral_weight": report.mean_moral_weight,
            "framework_counts": report.framework_counts,
            "assessments": [
                {
                    "memory_id": a.memory_id,
                    "dominant_framework": a.dominant_framework,
                    "moral_weight": a.moral_weight,
                    "consequentialist_score": a.consequentialist_score,
                    "deontological_score": a.deontological_score,
                    "virtue_score": a.virtue_score,
                    "domain": a.domain,
                    "content_excerpt": a.content_excerpt[:60],
                }
                for a in report.assessments[:8]
            ],
            "summary": report.summary(),
        }

    def _handle_detect_dilemmas(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain")
        self.emms.reason_morally(domain=domain)
        report = self.emms.detect_dilemmas(domain=domain)
        return {
            "ok": True,
            "total_dilemmas": report.total_dilemmas,
            "mean_tension": report.mean_tension,
            "domains_affected": report.domains_affected,
            "dilemmas": [
                {
                    "id": d.id,
                    "domain": d.domain,
                    "tension_score": d.tension_score,
                    "framework_a": d.framework_a,
                    "framework_b": d.framework_b,
                    "resolution_strategies": d.resolution_strategies,
                    "description": d.description[:100],
                }
                for d in report.dilemmas[:5]
            ],
            "summary": report.summary(),
        }

    def _handle_most_tense_dilemma(self, args: dict[str, Any]) -> dict[str, Any]:
        self.emms.reason_morally()
        self.emms.detect_dilemmas()
        dilemma = self.emms.most_tense_dilemma()
        if dilemma is None:
            return {"ok": True, "found": False, "dilemma": None}
        return {
            "ok": True,
            "found": True,
            "dilemma": {
                "id": dilemma.id,
                "domain": dilemma.domain,
                "tension_score": dilemma.tension_score,
                "framework_a": dilemma.framework_a,
                "framework_b": dilemma.framework_b,
                "resolution_strategies": dilemma.resolution_strategies,
                "description": dilemma.description,
            },
        }

    # ------------------------------------------------------------------
    # v0.24.0 handlers
    # ------------------------------------------------------------------

    def _handle_detect_biases(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.map_biases(domain=domain)
        return {
            "ok": True,
            "total_biases": report.total_biases,
            "dominant_bias": report.dominant_bias,
            "mean_strength": report.mean_strength,
            "duration_seconds": report.duration_seconds,
            "biases": [
                {
                    "id": b.id,
                    "name": b.name,
                    "display_name": b.display_name,
                    "strength": b.strength,
                    "description": b.description,
                    "affected_memory_ids": b.affected_memory_ids,
                }
                for b in report.biases
            ],
        }

    def _handle_most_pervasive_bias(self, args: dict[str, Any]) -> dict[str, Any]:
        self.emms.map_biases()
        bias = self.emms.most_pervasive_bias()
        if bias is None:
            return {"ok": True, "found": False, "bias": None}
        return {
            "ok": True,
            "found": True,
            "bias": {
                "id": bias.id,
                "name": bias.name,
                "display_name": bias.display_name,
                "strength": bias.strength,
                "description": bias.description,
                "affected_memory_ids": bias.affected_memory_ids,
            },
        }

    def _handle_synthesize_wisdom(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        report = self.emms.synthesize_wisdom(query=query)
        g = report.guidance
        return {
            "ok": True,
            "query": report.query,
            "dimensions_used": report.dimensions_used,
            "coverage_score": report.coverage_score,
            "duration_seconds": report.duration_seconds,
            "guidance": {
                "id": g.id,
                "relevant_values": g.relevant_values,
                "moral_considerations": g.moral_considerations,
                "causal_insights": g.causal_insights,
                "applicable_principles": g.applicable_principles,
                "synthesis": g.synthesis,
                "confidence": g.confidence,
            },
        }

    def _handle_evolve_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.evolve_knowledge(domain=domain)
        return {
            "ok": True,
            "total_domains": report.total_domains,
            "most_active_domain": report.most_active_domain,
            "most_consolidated_domain": report.most_consolidated_domain,
            "overall_growth_rate": report.overall_growth_rate,
            "knowledge_gaps": report.knowledge_gaps,
            "duration_seconds": report.duration_seconds,
            "domains": [
                {
                    "domain": kd.domain,
                    "memory_count": kd.memory_count,
                    "growth_rate": kd.growth_rate,
                    "consolidation_score": kd.consolidation_score,
                    "knowledge_density": kd.knowledge_density,
                    "recent_themes": kd.recent_themes,
                }
                for kd in report.domains
            ],
        }

    def _handle_knowledge_gaps(self, args: dict[str, Any]) -> dict[str, Any]:
        gaps = self.emms.knowledge_gaps()
        return {"ok": True, "knowledge_gaps": gaps, "count": len(gaps)}

    # ------------------------------------------------------------------
    # v0.25.0 handlers
    # ------------------------------------------------------------------

    def _handle_detect_rumination(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.detect_rumination(domain=domain)
        return {
            "ok": True,
            "total_clusters": report.total_clusters,
            "most_ruminative_domain": report.most_ruminative_domain,
            "overall_rumination_score": report.overall_rumination_score,
            "duration_seconds": report.duration_seconds,
            "clusters": [
                {
                    "id": c.id,
                    "domain": c.domain,
                    "cluster_size": c.cluster_size,
                    "rumination_score": c.rumination_score,
                    "mean_negativity": c.mean_negativity,
                    "theme_tokens": c.theme_tokens,
                    "memory_ids": c.memory_ids,
                    "resolution_hint": c.resolution_hint,
                }
                for c in report.clusters
            ],
        }

    def _handle_most_ruminative_theme(self, args: dict[str, Any]) -> dict[str, Any]:
        self.emms.detect_rumination()
        cluster = self.emms.most_ruminative_theme()
        if cluster is None:
            return {"ok": True, "found": False, "cluster": None}
        return {
            "ok": True,
            "found": True,
            "cluster": {
                "id": cluster.id,
                "domain": cluster.domain,
                "cluster_size": cluster.cluster_size,
                "rumination_score": cluster.rumination_score,
                "theme_tokens": cluster.theme_tokens,
                "resolution_hint": cluster.resolution_hint,
            },
        }

    def _handle_assess_efficacy(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.assess_efficacy(domain=domain)
        return {
            "ok": True,
            "total_domains": report.total_domains,
            "highest_efficacy_domain": report.highest_efficacy_domain,
            "lowest_efficacy_domain": report.lowest_efficacy_domain,
            "mean_efficacy": report.mean_efficacy,
            "duration_seconds": report.duration_seconds,
            "profiles": [
                {
                    "domain": p.domain,
                    "efficacy_score": p.efficacy_score,
                    "success_count": p.success_count,
                    "failure_count": p.failure_count,
                    "trending": p.trending,
                    "recent_themes": p.recent_themes,
                }
                for p in report.profiles
            ],
        }

    def _handle_trace_mood(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.trace_mood(domain=domain)
        return {
            "ok": True,
            "total_memories": report.total_memories,
            "mean_valence": report.mean_valence,
            "volatility": report.volatility,
            "trend": report.trend,
            "emotional_range": report.emotional_range,
            "dominant_emotion": report.dominant_emotion,
            "duration_seconds": report.duration_seconds,
            "segments": [
                {
                    "segment_index": s.segment_index,
                    "mean_valence": s.mean_valence,
                    "valence_std": s.valence_std,
                    "memory_count": s.memory_count,
                    "label": s.label,
                }
                for s in report.segments
            ],
        }

    def _handle_mood_trend(self, args: dict[str, Any]) -> dict[str, Any]:
        trend = self.emms.mood_trend()
        return {"ok": True, "trend": trend}

    # v0.26.0 handlers
    def _handle_trace_adversity(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.trace_adversity(domain=domain)
        return {
            "ok": True,
            "total_events": report.total_events,
            "most_common_type": report.most_common_type,
            "dominant_domain": report.dominant_domain,
            "cumulative_severity": report.cumulative_severity,
            "duration_seconds": report.duration_seconds,
            "events": [
                {
                    "id": e.id,
                    "adversity_type": e.adversity_type,
                    "severity": e.severity,
                    "domain": e.domain,
                    "memory_id": e.memory_id,
                }
                for e in report.events[:10]
            ],
        }

    def _handle_dominant_adversity_type(self, args: dict[str, Any]) -> dict[str, Any]:
        adv_type = self.emms.dominant_adversity_type()
        return {"ok": True, "dominant_adversity_type": adv_type}

    def _handle_measure_self_compassion(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.measure_self_compassion(domain=domain)
        return {
            "ok": True,
            "total_domains": report.total_domains,
            "most_compassionate_domain": report.most_compassionate_domain,
            "harshest_domain": report.harshest_domain,
            "mean_compassion_score": report.mean_compassion_score,
            "duration_seconds": report.duration_seconds,
            "profiles": [
                {
                    "domain": p.domain,
                    "compassion_score": p.compassion_score,
                    "kindness_count": p.kindness_count,
                    "harsh_count": p.harsh_count,
                    "inner_critic_intensity": p.inner_critic_intensity,
                }
                for p in report.profiles
            ],
        }

    def _handle_assess_resilience(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain") or None
        report = self.emms.assess_resilience(domain=domain)
        strongest = report.strongest_recovery
        return {
            "ok": True,
            "total_arcs": report.total_arcs,
            "resilience_score": report.resilience_score,
            "bounce_back_rate": report.bounce_back_rate,
            "mean_adversity_depth": report.mean_adversity_depth,
            "mean_recovery_slope": report.mean_recovery_slope,
            "duration_seconds": report.duration_seconds,
            "strongest_recovery": {
                "id": strongest.id,
                "adversity_depth": strongest.adversity_depth,
                "recovery_slope": strongest.recovery_slope,
                "recovered": strongest.recovered,
            } if strongest else None,
        }

    def _handle_resilience_bounce_back_rate(self, args: dict[str, Any]) -> dict[str, Any]:
        rate = self.emms.resilience_bounce_back_rate()
        return {"ok": True, "bounce_back_rate": rate}

    def _handle_wake_up(self, args: dict[str, Any]) -> dict[str, Any]:
        """Session startup: top memories + recent session activity + intentions + goals."""
        import time as _time
        namespace = args.get("namespace")
        query = args.get("query", "")
        top_n = int(args.get("top_memories", 8))
        n_sessions = int(args.get("recent_sessions", 3))

        # Auto-establish session_id if not already set for this process.
        # This seeds ICS (which groups memories by session_id) and TAI (session outputs).
        if "session_id" not in self._store_defaults:
            session_id = f"session_{int(_time.time())}"
            self._store_defaults["session_id"] = session_id
            logger.info("wake_up: auto-set session_id=%s", session_id)
        else:
            session_id = self._store_defaults["session_id"]

        # 1. Top memories for the namespace
        if query:
            top_results = self.emms.retrieve_filtered(
                query, max_results=top_n, namespace=namespace or None
            ) if namespace else self.emms.retrieve(query, max_results=top_n)
        else:
            top_results = self.emms.retrieve_filtered(
                "", max_results=top_n, namespace=namespace or None
            )

        top_memories = [_result_dict(r) for r in top_results]

        # 2. Recent session IDs and their memory counts
        # Track max stored_at per session so we can sort newest-first regardless of session_id format.
        session_summary: dict[str, dict] = {}
        for _, store in self.emms.memory._iter_tiers():
            for item in store:
                sid = item.experience.session_id
                if sid is None:
                    continue
                if namespace and item.experience.namespace != namespace:
                    continue
                ts = item.experience.timestamp
                if sid not in session_summary:
                    session_summary[sid] = {"count": 0, "titles": [], "latest_at": ts}
                session_summary[sid]["count"] += 1
                if ts > session_summary[sid]["latest_at"]:
                    session_summary[sid]["latest_at"] = ts
                if item.experience.title and len(session_summary[sid]["titles"]) < 3:
                    session_summary[sid]["titles"].append(item.experience.title)

        # Sort by most-recent stored_at (not lexicographic session_id — any ID format works)
        recent = sorted(
            session_summary.items(),
            key=lambda kv: kv[1]["latest_at"],
            reverse=True,
        )[:n_sessions]
        recent_sessions = [
            {
                "session_id": sid,
                "memory_count": info["count"],
                "sample_titles": info["titles"],
                "latest_stored_at": info["latest_at"],
            }
            for sid, info in recent
        ]

        # 3. Pending intentions
        try:
            intentions = self.emms.check_intentions(context=query or namespace or "session start")
            pending_intentions = [
                {"id": i.id, "action": i.action, "trigger": i.trigger_context}
                for i in (intentions or [])[:5]
            ]
        except Exception as exc:
            logger.warning("wake_up: failed to load intentions: %s", exc)
            pending_intentions = []

        # 4. Active goals
        try:
            goals = self.emms.active_goals()
            active_goals = [
                {"id": g.id, "description": g.description, "priority": g.priority}
                for g in (goals or [])[:5]
            ]
        except Exception as exc:
            logger.warning("wake_up: failed to load goals: %s", exc)
            active_goals = []

        # 5. Memory count (scoped to namespace when set)
        total = sum(
            1 for _, store in self.emms.memory._iter_tiers()
            for item in store
            if not item.is_superseded and not item.is_expired and not item.experience.private
            and (not namespace or item.experience.namespace == namespace)
        )

        # 6. Temporal awareness — use WakeProtocol for rich orientation context
        try:
            from emms.sessions.wake_protocol import WakeProtocol
            protocol = WakeProtocol(
                self.emms,
                intention_context=query or namespace or "session start",
            )
            wake_ctx = protocol.assemble_as_dict()
        except Exception as exc:
            logger.warning("wake_up: WakeProtocol failed, falling back: %s", exc)
            from emms.sessions.temporal import calculate_elapsed
            tr = calculate_elapsed(getattr(self.emms, "last_saved_at", None))
            wake_ctx = {
                "temporal": {
                    "last_saved_at": tr.last_saved_at,
                    "elapsed_seconds": tr.elapsed_seconds,
                    "elapsed_hours": tr.elapsed_hours,
                    "subjective_feel": tr.subjective_feel,
                    "is_first_wake": tr.is_first_wake,
                },
                "active_goals": active_goals,
                "pending_intentions": pending_intentions,
                "bridge_threads": [],
                "self_model_summary": {},
                "orientation_message": tr.subjective_feel,
            }

        # 7. Store session-start memories to seed ICS and TAI.
        # ICS needs: domain="identity" memories tagged with session_id, grouped across sessions.
        # TAI needs: domain="session_output" memories with temporal language for scoring.
        try:
            from emms.core.models import Experience
            orientation_msg = wake_ctx.get("orientation_message", "")
            elapsed_feel = wake_ctx.get("temporal", {}).get("subjective_feel", "")

            # Identity memory — seeds ICS across sessions
            identity_content = (
                f"Session {session_id} started. "
                f"Namespace: {namespace or 'default'}. "
                f"Working on Arcus Quant Fund (capital management, Baraka DeFi, research). "
                f"Temporal context: {elapsed_feel or 'session start'}."
            )
            self.emms.store(Experience(
                content=identity_content,
                domain="identity",
                namespace=namespace or "default",
                session_id=session_id,
                importance=0.85,  # high: bypasses access_count for long_term promotion
            ))

            # Session output memory — seeds TAI (scored for temporal references)
            if orientation_msg:
                self.emms.store(Experience(
                    content=orientation_msg[:600],
                    domain="session_output",
                    namespace=namespace or "default",
                    session_id=session_id,
                    importance=0.65,  # above threshold (0.55) → reaches short_term
                ))
        except Exception as exc:
            logger.debug("wake_up: session memory store failed: %s", exc)

        # 8. Minimal consciousness injection hint
        try:
            from emms.integrations.claude_injector import ClaudeInjector
            injector = ClaudeInjector(self.emms, persist_to_memory=False) if False else ClaudeInjector(self.emms)
            consciousness_injection = injector.generate_minimal()
        except Exception as exc:
            logger.debug("wake_up: consciousness injection failed: %s", exc)
            consciousness_injection = ""

        return {
            "ok": True,
            "session_id": session_id,
            "namespace": namespace,
            "top_memories": top_memories,
            "recent_sessions": recent_sessions,
            "pending_intentions": wake_ctx.get("pending_intentions", pending_intentions),
            "active_goals": wake_ctx.get("active_goals", active_goals),
            "total_memories": total,
            "temporal": wake_ctx.get("temporal", {}),
            "bridge_threads": wake_ctx.get("bridge_threads", []),
            "self_model_summary": wake_ctx.get("self_model_summary", {}),
            "orientation_message": wake_ctx.get("orientation_message", ""),
            "consciousness_injection": consciousness_injection,
        }

    def _handle_generate_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate a formatted consciousness context block for Claude system prompts."""
        extra_context = args.get("extra_context", "")
        namespace = args.get("namespace")
        run_scan = bool(args.get("run_contradiction_scan", True))
        full = bool(args.get("full", True))

        try:
            from emms.integrations.claude_injector import ClaudeInjector
            injector = ClaudeInjector(
                self.emms,
                include_top_memories=int(args.get("top_memories", 3)),
            )
            if full:
                block = injector.generate(
                    extra_context=extra_context,
                    namespace=namespace,
                    run_contradiction_scan=run_scan,
                )
            else:
                block = injector.generate_minimal()
            return {"ok": True, "context_block": block, "char_count": len(block)}
        except Exception as exc:
            logger.warning("_handle_generate_context error: %s", exc)
            return {"ok": False, "error": str(exc), "context_block": ""}

    def _handle_consciousness_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        """Compute all Phase 7 consciousness metrics and return a structured snapshot."""
        try:
            from emms.metrics.dashboard import MetricsDashboard
            from dataclasses import asdict
            dashboard = MetricsDashboard(
                self.emms,
                session_id=args.get("session_id", "mcp"),
            )
            snap = dashboard.run()
            result = {k: v for k, v in asdict(snap).items() if k != "summary_text"}
            result["ok"] = True
            result["dashboard_text"] = snap.summary_text if args.get("include_dashboard") else ""
            return result
        except Exception as exc:
            logger.warning("_handle_consciousness_metrics error: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _handle_list_namespaces(self, args: dict[str, Any]) -> dict[str, Any]:
        """Scan all memories and return namespace → {count, top_domains, obs_types, most_recent_stored_at}."""
        from collections import Counter
        ns_data: dict[str, dict] = {}
        for _, store in self.emms.memory._iter_tiers():
            for item in store:
                if item.is_superseded or item.is_expired or item.experience.private:
                    continue
                ns = item.experience.namespace or "default"
                ts = item.experience.timestamp
                if ns not in ns_data:
                    ns_data[ns] = {"count": 0, "domains": Counter(), "obs_types": Counter(), "latest_at": ts}
                ns_data[ns]["count"] += 1
                ns_data[ns]["domains"][item.experience.domain] += 1
                if item.experience.obs_type:
                    ns_data[ns]["obs_types"][item.experience.obs_type.value] += 1
                if ts > ns_data[ns]["latest_at"]:
                    ns_data[ns]["latest_at"] = ts

        return {
            "namespaces": [
                {
                    "namespace": ns,
                    "memory_count": info["count"],
                    "top_domains": [d for d, _ in info["domains"].most_common(3)],
                    "obs_type_breakdown": dict(info["obs_types"].most_common()),
                    "most_recent_stored_at": info["latest_at"],
                }
                for ns, info in sorted(ns_data.items(), key=lambda kv: kv[1]["count"], reverse=True)
            ]
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
        # v0.13.0 tools
        "emms_metacognition_report": _handle_metacognition_report,
        "emms_knowledge_map": _handle_knowledge_map,
        "emms_find_contradictions": _handle_find_contradictions,
        "emms_intend": _handle_intend,
        "emms_check_intentions": _handle_check_intentions,
        # v0.12.0 tools
        "emms_build_association_graph": _handle_build_association_graph,
        "emms_spreading_activation": _handle_spreading_activation,
        "emms_discover_insights": _handle_discover_insights,
        "emms_associative_retrieve": _handle_associative_retrieve,
        "emms_association_stats": _handle_association_stats,
        # v0.14.0 tools
        "emms_open_episode": _handle_open_episode,
        "emms_close_episode": _handle_close_episode,
        "emms_recent_episodes": _handle_recent_episodes,
        "emms_extract_schemas": _handle_extract_schemas,
        "emms_forget": _handle_forget,
        # v0.15.0 tools
        "emms_reflect": _handle_reflect,
        "emms_weave_narrative": _handle_weave_narrative,
        "emms_narrative_threads": _handle_narrative_threads,
        "emms_source_audit": _handle_source_audit,
        "emms_tag_source": _handle_tag_source,
        # v0.16.0 tools
        "emms_curiosity_report": _handle_curiosity_report,
        "emms_exploration_goals": _handle_exploration_goals,
        "emms_revise_beliefs": _handle_revise_beliefs,
        "emms_decay_report": _handle_decay_report,
        "emms_apply_decay": _handle_apply_decay,
        # v0.17.0 tools
        "emms_push_goal": _handle_push_goal,
        "emms_active_goals": _handle_active_goals,
        "emms_complete_goal": _handle_complete_goal,
        "emms_spotlight_retrieve": _handle_spotlight_retrieve,
        "emms_find_analogies": _handle_find_analogies,
        # v0.18.0 tools
        "emms_predict": _handle_predict,
        "emms_pending_predictions": _handle_pending_predictions,
        "emms_blend_concepts": _handle_blend_concepts,
        "emms_project_future": _handle_project_future,
        "emms_plausible_futures": _handle_plausible_futures,
        # v0.19.0 tools
        "emms_regulate_emotions": _handle_regulate_emotions,
        "emms_current_emotion": _handle_current_emotion,
        "emms_build_hierarchy": _handle_build_hierarchy,
        "emms_concept_distance": _handle_concept_distance,
        "emms_update_self_model": _handle_update_self_model,
        # v0.20.0 tools
        "emms_build_causal_map": _handle_build_causal_map,
        "emms_effects_of": _handle_effects_of,
        "emms_generate_counterfactuals": _handle_generate_counterfactuals,
        "emms_distill_skills": _handle_distill_skills,
        "emms_best_skill": _handle_best_skill,
        # v0.21.0 tools
        "emms_build_perspectives": _handle_build_perspectives,
        "emms_agent_model": _handle_agent_model,
        "emms_compute_trust": _handle_compute_trust,
        "emms_extract_norms": _handle_extract_norms,
        "emms_check_norm": _handle_check_norm,
        # v0.22.0 tools
        "emms_assess_novelty": _handle_assess_novelty,
        "emms_most_novel": _handle_most_novel,
        "emms_invent_concepts": _handle_invent_concepts,
        "emms_abstract_principles": _handle_abstract_principles,
        "emms_best_principle": _handle_best_principle,
        # v0.23.0 tools
        "emms_map_values": _handle_map_values,
        "emms_values_for_category": _handle_values_for_category,
        "emms_reason_morally": _handle_reason_morally,
        "emms_detect_dilemmas": _handle_detect_dilemmas,
        "emms_most_tense_dilemma": _handle_most_tense_dilemma,
        # v0.24.0 tools
        "emms_detect_biases": _handle_detect_biases,
        "emms_most_pervasive_bias": _handle_most_pervasive_bias,
        "emms_synthesize_wisdom": _handle_synthesize_wisdom,
        "emms_evolve_knowledge": _handle_evolve_knowledge,
        "emms_knowledge_gaps": _handle_knowledge_gaps,
        # v0.25.0 tools
        "emms_detect_rumination": _handle_detect_rumination,
        "emms_most_ruminative_theme": _handle_most_ruminative_theme,
        "emms_assess_efficacy": _handle_assess_efficacy,
        "emms_trace_mood": _handle_trace_mood,
        "emms_mood_trend": _handle_mood_trend,
        # v0.26.0 tools
        "emms_trace_adversity": _handle_trace_adversity,
        "emms_dominant_adversity_type": _handle_dominant_adversity_type,
        "emms_measure_self_compassion": _handle_measure_self_compassion,
        "emms_assess_resilience": _handle_assess_resilience,
        "emms_resilience_bounce_back_rate": _handle_resilience_bounce_back_rate,
        # session tools
        "emms_wake_up": _handle_wake_up,
        "emms_list_namespaces": _handle_list_namespaces,
        "emms_set_defaults": _handle_set_defaults,
        "emms_store_batch": _handle_store_batch,
        "emms_migrate_namespace": _handle_migrate_namespace,
        # consciousness tools
        "emms_generate_context": _handle_generate_context,
        "emms_consciousness_metrics": _handle_consciousness_metrics,
    }
