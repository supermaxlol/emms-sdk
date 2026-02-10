"""ChromaDB vector store backend for EMMS.

Stores experience embeddings in ChromaDB for fast semantic retrieval.
Requires: pip install chromadb

Usage:
    from emms.storage.chroma import ChromaStore
    from emms.core.embeddings import HashEmbedder

    store = ChromaStore(embedder=HashEmbedder())
    store.add("exp_001", "The market rose 2%", {"domain": "finance"})
    results = store.query("stock market trends", n_results=5)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

from emms.core.embeddings import EmbeddingProvider, HashEmbedder

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings

    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False


class ChromaStore:
    """Vector store backed by ChromaDB.

    Supports both persistent (on-disk) and ephemeral (in-memory) modes.
    Handles embedding via the supplied EmbeddingProvider — ChromaDB's
    built-in embedding is bypassed so we control the vectors.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        collection_name: str = "emms_memories",
        persist_directory: str | Path | None = None,
    ):
        if not _HAS_CHROMA:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        self.embedder = embedder or HashEmbedder()
        self.collection_name = collection_name

        # Initialise client
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
            )
        else:
            self._client = chromadb.EphemeralClient()

        # Get or create collection (no auto-embedding — we provide our own)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Add a single document with its embedding."""
        vec = embedding or self.embedder.embed(content)
        meta = {
            "content": content[:512],   # ChromaDB metadata value limit
            "stored_at": time.time(),
            **(metadata or {}),
        }
        # Ensure metadata values are str/int/float/bool (ChromaDB requirement)
        clean_meta = {
            k: v for k, v in meta.items()
            if isinstance(v, (str, int, float, bool))
        }
        self._collection.upsert(
            ids=[id],
            embeddings=[vec],
            documents=[content],
            metadatas=[clean_meta],
        )

    def add_batch(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        embeddings: Sequence[list[float]] | None = None,
    ) -> int:
        """Add multiple documents at once. Returns count added."""
        if embeddings:
            vecs = list(embeddings)
        else:
            vecs = self.embedder.embed_batch(contents)

        metas = []
        for i, content in enumerate(contents):
            m = {"content": content[:512], "stored_at": time.time()}
            if metadatas and i < len(metadatas):
                m.update(metadatas[i])
            metas.append({
                k: v for k, v in m.items()
                if isinstance(v, (str, int, float, bool))
            })

        self._collection.upsert(
            ids=list(ids),
            embeddings=vecs,
            documents=list(contents),
            metadatas=metas,
        )
        return len(ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {id, content, score, metadata}."""
        vec = query_embedding or self.embedder.embed(query_text)

        kwargs: dict[str, Any] = {
            "query_embeddings": [vec],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        if raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw["distances"] else 0
                # ChromaDB cosine distance = 1 - similarity
                similarity = 1.0 - distance
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "score": similarity,
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                })

        return results

    # ------------------------------------------------------------------
    # Delete / stats
    # ------------------------------------------------------------------

    def delete(self, ids: Sequence[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=list(ids))

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
