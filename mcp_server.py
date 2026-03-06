#!/usr/bin/env python3
"""EMMS FastMCP stdio server for Claude Code integration.

Usage:
    python emms-sdk/mcp_server.py

State is automatically loaded from .emms_state.json at startup and saved
via atexit when the process exits.

Note on argument handling:
    Claude Code's MCP client sends tool arguments nested under a single
    "kwargs" key (e.g. {"kwargs": {"content": "..."}}).  Each registered
    tool unwraps that layer before dispatching to EMCPServer.handle().
"""

import sys
import atexit
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

STATE_FILE = Path.home() / ".emms" / "emms_state.json"


def main():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("mcp package not found. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    from emms import EMMS
    from emms.adapters.mcp_server import EMCPServer

    # Start with NO embedder so the MCP handshake completes immediately and
    # state can be loaded without corrupting stored embeddings.  Starting with
    # HashEmbedder would cause load_state() to see a dim-mismatch (128 ≠ 384)
    # and silently re-embed every stored item with HashEmbedder, overwriting
    # the 384-dim SentenceTransformer vectors saved from the previous session.
    # With embedder=None the load just restores embeddings as-is (correct dims
    # preserved) and retrieval falls back to lexical until ST is ready.
    emms_instance = EMMS(embedder=None)

    if STATE_FILE.exists():
        try:
            emms_instance.load(str(STATE_FILE))
            print(f"EMMS: state loaded from {STATE_FILE}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: could not load state from {STATE_FILE}: {e}", file=sys.stderr)

    def _load_st_embedder():
        try:
            from emms.core.embeddings import SentenceTransformerEmbedder
            from emms.memory.hierarchical import VectorIndex
            st = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
            mem = emms_instance.memory

            # Wire embedder on both the EMMS façade and the memory layer so
            # hybrid/adaptive retrievers pick it up too.
            emms_instance.embedder = st
            mem.embedder = st

            # Build a fresh 384-dim VectorIndex from the embedding cache that
            # load_state() restored.  Items with the correct dim are added
            # directly; anything else (e.g. stale hash-dim entries) is
            # re-embedded now.
            vec = VectorIndex(dim=st.dim)
            cache = mem._embeddings
            reembedded = 0
            added = 0
            for exp_id, emb in list(cache.items()):
                if len(emb) == st.dim:
                    vec.add(exp_id, emb)
                    added += 1
                else:
                    item = mem._items_by_exp_id.get(exp_id)
                    if item:
                        new_emb = st.embed(item.experience.content)
                        cache[exp_id] = new_emb
                        vec.add(exp_id, new_emb)
                        reembedded += 1

            mem._vec_index = vec

            # Mop-up: any item that was embedded (by the now-set mem.embedder)
            # between the list(cache.items()) snapshot and the assignment above
            # was stored in _embeddings but missed by the loop — add them now.
            for exp_id, emb in cache.items():
                if exp_id not in vec._id_to_idx and len(emb) == st.dim:
                    vec.add(exp_id, emb)

            print(
                f"EMMS: SentenceTransformer ready — VectorIndex rebuilt "
                f"({added} cached, {reembedded} re-embedded)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"EMMS: SentenceTransformer unavailable ({e}), using lexical retrieval", file=sys.stderr)

    import threading
    threading.Thread(target=_load_st_embedder, daemon=True).start()

    atexit.register(lambda: emms_instance.save(str(STATE_FILE)))

    server = EMCPServer(emms_instance, save_path=STATE_FILE, auto_save_every=10)
    mcp = FastMCP("emms")

    # Register each tool with an explicit `kwargs: dict` parameter so that
    # Claude Code's wrapper (which sends {"kwargs": {...}}) is handled
    # correctly.  Falls back to flat keyword args for direct callers.
    for defn in server.tool_definitions:
        name = defn["name"]
        description = defn.get("description", "")

        def _make_tool(n: str, desc: str):
            def _tool(kwargs: dict = None, flat: dict = None, rest: dict = None) -> Any:
                # FastMCP turns **variadics into required schema fields — so use
                # only explicit optional params.  Claude Code calls one of:
                #   {"kwargs": {...}}  legacy wrapping
                #   {"flat": {...}}    older MCP client
                #   {"rest": {...}}    current MCP client (FastMCP names **rest → "rest")
                args: dict[str, Any] = {}
                if kwargs is not None:
                    args.update(kwargs)
                if flat is not None:
                    args.update(flat)
                if rest is not None:
                    args.update(rest)
                return server.handle(n, args)

            _tool.__name__ = n
            _tool.__doc__ = desc
            return _tool

        mcp.tool(name=name, description=description)(_make_tool(name, description))

    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
