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
    }
