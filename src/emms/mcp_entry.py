#!/usr/bin/env python3
"""EMMS MCP Server — persistent cognitive memory for AI agents.

Install:
    pip install emms-mcp
    # or
    npx -y emms-mcp

Usage:
    emms-mcp                          # uses default state file
    emms-mcp --state-file /path/to   # custom state location
    emms-mcp --version

The server exposes 129 cognitive memory tools over MCP stdio transport,
including storage, retrieval, reflection, emotional memory, knowledge
graphs, goal tracking, metacognition, and more.
"""

import sys
import os
import argparse
import atexit
import threading
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser(
        prog="emms-mcp",
        description="EMMS — Enhanced Memory Management System MCP Server",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get(
            "EMMS_STATE_FILE",
            str(Path.home() / ".emms" / "emms_state.json"),
        ),
        help="Path to EMMS state file (default: ~/.emms/emms_state.json)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    args = parser.parse_args()

    if args.version:
        from emms import __version__
        print(f"emms-mcp {__version__}")
        sys.exit(0)

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "mcp package not found. Install with: pip install emms-mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    from emms import EMMS
    from emms.adapters.mcp_server import EMCPServer

    state_file = Path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Start with no embedder so MCP handshake completes immediately.
    # SentenceTransformer loads in background thread.
    emms_instance = EMMS(embedder=None)

    if state_file.exists():
        try:
            emms_instance.load(str(state_file))
            print(f"EMMS: state loaded from {state_file}", file=sys.stderr)
        except Exception as e:
            print(
                f"Warning: could not load state from {state_file}: {e}",
                file=sys.stderr,
            )

    def _load_st_embedder():
        try:
            from emms.core.embeddings import SentenceTransformerEmbedder
            from emms.memory.hierarchical import VectorIndex

            st = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
            mem = emms_instance.memory
            emms_instance.embedder = st
            mem.embedder = st

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

            for exp_id, emb in cache.items():
                if exp_id not in vec._id_to_idx and len(emb) == st.dim:
                    vec.add(exp_id, emb)

            print(
                f"EMMS: SentenceTransformer ready — VectorIndex rebuilt "
                f"({added} cached, {reembedded} re-embedded)",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"EMMS: SentenceTransformer unavailable ({e}), "
                f"using lexical retrieval",
                file=sys.stderr,
            )

    threading.Thread(target=_load_st_embedder, daemon=True).start()

    atexit.register(lambda: emms_instance.save(str(state_file)))

    server = EMCPServer(emms_instance, save_path=state_file, auto_save_every=10)
    mcp = FastMCP("emms")

    for defn in server.tool_definitions:
        name = defn["name"]
        description = defn.get("description", "")

        def _make_tool(n: str, desc: str):
            def _tool(
                kwargs: dict = None, flat: dict = None, rest: dict = None
            ) -> Any:
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

        mcp.tool(name=name, description=description)(
            _make_tool(name, description)
        )

    mcp.run()


if __name__ == "__main__":
    main()
