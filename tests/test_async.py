"""Tests for async API support.

Uses asyncio.run() directly to avoid pytest-asyncio configuration issues.
"""

import asyncio
import pytest
from emms.core.models import Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.emms import EMMS


def _make_emms():
    return EMMS(embedder=HashEmbedder(dim=64))


def test_astore():
    async def _run():
        emms = _make_emms()
        exp = Experience(content="Async store test", domain="test", importance=0.7)
        result = await emms.astore(exp)
        assert "experience_id" in result
        assert result["tier"] == "working"
    asyncio.run(_run())


def test_astore_batch():
    async def _run():
        emms = _make_emms()
        exps = [
            Experience(content=f"Batch item {i}", domain="test")
            for i in range(5)
        ]
        results = await emms.astore_batch(exps)
        assert len(results) == 5
    asyncio.run(_run())


def test_aretrieve():
    async def _run():
        emms = _make_emms()
        exp = Experience(content="Stock market analysis report", domain="finance")
        await emms.astore(exp)
        results = await emms.aretrieve("stock market")
        assert len(results) > 0
    asyncio.run(_run())


def test_aretrieve_semantic():
    async def _run():
        emms = _make_emms()
        exp = Experience(content="Bitcoin price hit 100k", domain="finance")
        await emms.astore(exp)
        results = await emms.aretrieve_semantic("crypto prices")
        assert len(results) > 0
    asyncio.run(_run())


def test_aconsolidate():
    async def _run():
        emms = _make_emms()
        for i in range(10):
            exp = Experience(
                content=f"Important experience {i}",
                domain="test",
                importance=0.9,
                novelty=0.8,
            )
            await emms.astore(exp)
        result = await emms.aconsolidate()
        assert "items_consolidated" in result
    asyncio.run(_run())


def test_concurrent_stores():
    """Test that concurrent async stores don't corrupt state."""
    async def _run():
        emms = _make_emms()
        async def store_one(i):
            exp = Experience(content=f"Concurrent item {i}", domain="test")
            return await emms.astore(exp)
        results = await asyncio.gather(*[store_one(i) for i in range(20)])
        assert len(results) == 20
        assert emms.stats["total_stored"] == 20
    asyncio.run(_run())


def test_concurrent_store_and_retrieve():
    """Test store and retrieve can run concurrently."""
    async def _run():
        emms = _make_emms()
        for i in range(5):
            exp = Experience(content=f"Finance topic {i}", domain="finance")
            await emms.astore(exp)

        async def store_more():
            for i in range(5):
                exp = Experience(content=f"Additional item {i}", domain="test")
                await emms.astore(exp)

        async def retrieve_some():
            return await emms.aretrieve("finance")

        await asyncio.gather(store_more(), retrieve_some())
        assert emms.stats["total_stored"] == 10
    asyncio.run(_run())
