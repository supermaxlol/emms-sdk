"""EMMS HTTP REST API — thin FastAPI wrapper for non-MCP callers.

Exposes core EMMS operations over HTTP so Slack/Discord bots,
the trader daemon, and any other process can query/store memories
without needing MCP tool access.

Endpoints:
  POST /store      — store a new memory
  POST /retrieve   — semantic search with filters
  GET  /stats      — system statistics
  GET  /health     — liveness + uptime

Usage (non-blocking daemon thread):
  from emms.adapters.http_server import start_server
  start_server(emms_instance, port=8765)
"""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

# Module-level state — set via start_server()
_emms: "EMMS | None" = None
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="EMMS REST API", version="1.0", docs_url="/docs")

    class StoreRequest(BaseModel):
        content: str
        domain: str = "default"
        namespace: str = "default"
        title: str | None = None
        importance: float = 0.5
        emotional_valence: float = 0.0
        session_id: str | None = None

    class RetrieveRequest(BaseModel):
        query: str
        max_results: int = 10
        namespace: str | None = None
        domain: str | None = None
        min_importance: float | None = None
        sort_by: str = "relevance"

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "uptime_s": round(time.time() - _start_time),
            "pid": os.getpid(),
            "emms_loaded": _emms is not None,
        }

    @app.get("/stats")
    def stats():
        if _emms is None:
            raise HTTPException(status_code=503, detail="EMMS not loaded")
        return {"ok": True, **_emms.get_stats()}

    @app.post("/store")
    def store(req: StoreRequest):
        if _emms is None:
            raise HTTPException(status_code=503, detail="EMMS not loaded")
        from emms.core.models import Experience
        exp = Experience(
            content=req.content,
            domain=req.domain,
            namespace=req.namespace,
            title=req.title,
            importance=req.importance,
            emotional_valence=req.emotional_valence,
            session_id=req.session_id,
        )
        result = _emms.store(exp)
        return {
            "ok": True,
            "memory_id": result["memory_id"],
            "tier": result["tier"],
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        if _emms is None:
            raise HTTPException(status_code=503, detail="EMMS not loaded")
        results = _emms.retrieve_filtered(
            req.query,
            max_results=req.max_results,
            namespace=req.namespace,
            domain=req.domain,
            min_importance=req.min_importance,
            sort_by=req.sort_by,
        )
        return {
            "ok": True,
            "count": len(results),
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.experience.content,
                    "title": r.memory.experience.title,
                    "domain": r.memory.experience.domain,
                    "namespace": r.memory.experience.namespace,
                    "importance": r.memory.experience.importance,
                    "score": r.score,
                    "tier": r.source_tier.value,
                }
                for r in results
            ],
        }

    _FASTAPI_AVAILABLE = True

except ImportError:
    _FASTAPI_AVAILABLE = False
    app = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_server(
    emms_instance: "EMMS",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> threading.Thread | None:
    """Start the EMMS HTTP server as a non-blocking daemon thread.

    Args:
        emms_instance: Loaded EMMS instance to serve.
        host:          Bind address (default localhost-only).
        port:          TCP port (default 8765).

    Returns:
        The daemon Thread, or None if FastAPI/uvicorn not installed.
    """
    global _emms, _start_time
    _emms = emms_instance
    _start_time = time.time()

    if not _FASTAPI_AVAILABLE:
        import logging
        logging.getLogger(__name__).warning(
            "EMMS HTTP server not started — fastapi/uvicorn not installed. "
            "Run: pip install 'emms[api]'"
        )
        return None

    t = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
        daemon=True,
        name="emms-http-server",
    )
    t.start()
    return t
