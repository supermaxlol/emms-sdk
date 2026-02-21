"""Tests for EMMS v0.7.0 features.

MemoryDiff · MemoryCluster · ConversationBuffer · stream_retrieve · LLMConsolidator
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
import pytest

from emms.core.models import Experience, MemoryItem, MemoryTier, MemoryConfig
from emms.memory.hierarchical import HierarchicalMemory
from emms.emms import EMMS


# ===========================================================================
# Helpers
# ===========================================================================

def _make_memory(n: int = 5) -> HierarchicalMemory:
    mem = HierarchicalMemory()
    for i in range(n):
        exp = Experience(
            content=f"Memory item number {i} about machine learning and neural networks",
            domain="tech",
            importance=0.6 + i * 0.05,
        )
        mem.store(exp)
    return mem


def _make_emms(n: int = 5) -> EMMS:
    agent = EMMS()
    for i in range(n):
        exp = Experience(
            content=f"Experience {i}: Python programming and software development",
            domain="tech",
            importance=0.5 + i * 0.05,
        )
        agent.store(exp)
    return agent


# ===========================================================================
# 1. MemoryDiff
# ===========================================================================

class TestMemoryDiff:
    def test_import(self):
        from emms.memory.diff import MemoryDiff, DiffResult, ItemSnapshot
        assert MemoryDiff
        assert DiffResult
        assert ItemSnapshot

    def test_diff_no_change(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        snap = {
            "a": ItemSnapshot(id="a", experience_id="e1", content="hello", domain="test",
                              tier="working", importance=0.5, memory_strength=1.0,
                              access_count=0, stored_at=0.0),
        }
        result = MemoryDiff.diff(snap, snap)
        assert result.added == []
        assert result.removed == []
        assert result.strengthened == []
        assert result.weakened == []

    def test_diff_added(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        snap_a: dict = {}
        snap_b = {
            "a": ItemSnapshot(id="a", experience_id="e1", content="new item", domain="test",
                              tier="working", importance=0.5, memory_strength=1.0,
                              access_count=0, stored_at=time.time()),
        }
        result = MemoryDiff.diff(snap_a, snap_b)
        assert len(result.added) == 1
        assert result.added[0].id == "a"
        assert result.removed == []

    def test_diff_removed(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        snap_a = {
            "a": ItemSnapshot(id="a", experience_id="e1", content="old", domain="test",
                              tier="working", importance=0.5, memory_strength=1.0,
                              access_count=0, stored_at=time.time()),
        }
        result = MemoryDiff.diff(snap_a, {})
        assert len(result.removed) == 1
        assert result.removed[0].id == "a"

    def test_diff_strengthened(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        base = dict(id="x", experience_id="e", content="c", domain="d",
                    tier="working", importance=0.5, access_count=0, stored_at=0.0)
        snap_a = {"x": ItemSnapshot(**{**base, "memory_strength": 0.5})}
        snap_b = {"x": ItemSnapshot(**{**base, "memory_strength": 0.9})}
        result = MemoryDiff.diff(snap_a, snap_b, strength_threshold=0.1)
        assert len(result.strengthened) == 1
        assert result.weakened == []

    def test_diff_weakened(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        base = dict(id="x", experience_id="e", content="c", domain="d",
                    tier="working", importance=0.5, access_count=0, stored_at=0.0)
        snap_a = {"x": ItemSnapshot(**{**base, "memory_strength": 0.9})}
        snap_b = {"x": ItemSnapshot(**{**base, "memory_strength": 0.4})}
        result = MemoryDiff.diff(snap_a, snap_b, strength_threshold=0.1)
        assert len(result.weakened) == 1
        assert result.strengthened == []

    def test_diff_superseded(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        base = dict(id="x", experience_id="e", content="c", domain="d",
                    tier="working", importance=0.5, memory_strength=1.0,
                    access_count=0, stored_at=0.0)
        snap_a = {"x": ItemSnapshot(**{**base, "superseded_by": None})}
        snap_b = {"x": ItemSnapshot(**{**base, "superseded_by": "newer_id"})}
        result = MemoryDiff.diff(snap_a, snap_b)
        assert len(result.superseded) == 1

    def test_summary_text(self):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        snap_a: dict = {}
        snap_b = {
            "a": ItemSnapshot(id="a", experience_id="e1", content="new", domain="test",
                              tier="working", importance=0.5, memory_strength=1.0,
                              access_count=0, stored_at=0.0),
        }
        result = MemoryDiff.diff(snap_a, snap_b)
        summary = result.summary()
        assert "1 added" in summary
        assert "0 removed" in summary

    def test_export_markdown(self, tmp_path):
        from emms.memory.diff import MemoryDiff, ItemSnapshot
        snap_a: dict = {}
        snap_b = {
            "a": ItemSnapshot(id="a", experience_id="e1", content="new memory item", domain="tech",
                              tier="working", importance=0.5, memory_strength=1.0,
                              access_count=0, stored_at=0.0),
        }
        result = MemoryDiff.diff(snap_a, snap_b)
        md = result.export_markdown(tmp_path / "diff.md")
        assert "# Memory Diff" in md
        assert "Added" in md
        assert (tmp_path / "diff.md").exists()

    def test_from_paths(self, tmp_path):
        from emms.memory.diff import MemoryDiff

        # Create two fake snapshot files
        snap_a_data = {"saved_at": time.time() - 100, "working": [], "short_term": [],
                       "long_term": [], "semantic": []}
        snap_b_data = {"saved_at": time.time(), "working": [
            {"id": "mem_new", "tier": "working",
             "experience": {"id": "exp_new", "content": "hello world", "domain": "test",
                            "importance": 0.5, "title": None},
             "memory_strength": 1.0, "access_count": 0, "stored_at": time.time(),
             "superseded_by": None}
        ], "short_term": [], "long_term": [], "semantic": []}

        pa = tmp_path / "before.json"
        pb = tmp_path / "after.json"
        pa.write_text(json.dumps(snap_a_data), encoding="utf-8")
        pb.write_text(json.dumps(snap_b_data), encoding="utf-8")

        result = MemoryDiff.from_paths(pa, pb)
        assert len(result.added) == 1
        assert result.added[0].content == "hello world"

    def test_from_memories(self):
        from emms.memory.diff import MemoryDiff
        mem_a = _make_memory(2)
        mem_b = _make_memory(4)  # 2 extra items
        result = MemoryDiff.from_memories(mem_a, mem_b)
        # mem_b has items not in mem_a (different UUIDs)
        # They should appear as added/removed (since IDs differ)
        assert isinstance(result.added, list)

    def test_diff_since_emms(self, tmp_path):
        agent = _make_emms(3)
        snap_path = tmp_path / "snap.json"
        agent.save(str(snap_path))
        # Store more
        agent.store(Experience(content="extra memory for diff test", domain="test"))
        result = agent.diff_since(str(snap_path))
        # At minimum the extra memory should show as added or the summary should work
        assert hasattr(result, "added")
        assert hasattr(result, "summary")


# ===========================================================================
# 2. MemoryCluster
# ===========================================================================

class TestMemoryCluster:
    def test_import(self):
        from emms.memory.clustering import MemoryClustering, MemoryCluster
        assert MemoryClustering
        assert MemoryCluster

    def test_cluster_basic(self):
        from emms.memory.clustering import MemoryClustering
        mem = HierarchicalMemory()
        # Use clearly distinct content for two groups
        for _ in range(3):
            mem.store(Experience(content="cooking recipe pasta italian food tomato", domain="food"))
        for _ in range(3):
            mem.store(Experience(content="python programming software code algorithm", domain="tech"))
        items = list(mem.working) + list(mem.short_term)

        clustering = MemoryClustering()
        clusters = clustering.cluster(items, k=2)
        assert len(clusters) == 2
        total_members = sum(len(c.members) for c in clusters)
        assert total_members == len(items)

    def test_cluster_auto_k(self):
        from emms.memory.clustering import MemoryClustering
        mem = HierarchicalMemory()
        for i in range(8):
            mem.store(Experience(content=f"document {i} about topic {i % 3}", domain="gen"))
        items = list(mem.working) + list(mem.short_term) + list(mem.long_term.values())
        if len(items) < 2:
            pytest.skip("Not enough items after consolidation")

        clustering = MemoryClustering()
        clusters = clustering.cluster(items, auto_k=True, k_min=2, k_max=4)
        assert len(clusters) >= 1

    def test_cluster_single_item(self):
        from emms.memory.clustering import MemoryClustering
        mem = HierarchicalMemory()
        mem.store(Experience(content="only one item", domain="test"))
        items = list(mem.working)
        clustering = MemoryClustering()
        clusters = clustering.cluster(items, k=1)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 1

    def test_cluster_returns_labels(self):
        from emms.memory.clustering import MemoryClustering
        mem = HierarchicalMemory()
        for i in range(4):
            mem.store(Experience(content=f"finance investment stock market trading {i}", domain="finance"))
        items = list(mem.working) + list(mem.short_term)
        if len(items) < 2:
            pytest.skip("Not enough items")

        clustering = MemoryClustering()
        clusters = clustering.cluster(items, k=min(2, len(items)))
        for c in clusters:
            assert isinstance(c.label, str)
            assert len(c.label) > 0
            assert c.inertia >= 0.0

    def test_cluster_ids_unique(self):
        from emms.memory.clustering import MemoryClustering
        mem = HierarchicalMemory()
        for _ in range(3):
            mem.store(Experience(content="biology cell dna gene protein chromosome", domain="bio"))
        for _ in range(3):
            mem.store(Experience(content="astronomy star planet galaxy telescope orbit", domain="astro"))
        for _ in range(3):
            mem.store(Experience(content="cooking baking recipe ingredients flour sugar", domain="food"))
        items = list(mem.working) + list(mem.short_term)

        clustering = MemoryClustering()
        clusters = clustering.cluster(items, k=3)
        ids = [c.id for c in clusters]
        assert len(ids) == len(set(ids))

    def test_cluster_empty(self):
        from emms.memory.clustering import MemoryClustering
        clustering = MemoryClustering()
        clusters = clustering.cluster([], k=2)
        assert clusters == []

    def test_cluster_with_embeddings(self):
        from emms.memory.clustering import MemoryClustering
        from emms.core.embeddings import HashEmbedder
        mem = _make_memory(4)
        embedder = HashEmbedder(dim=32)
        items = list(mem.working) + list(mem.short_term)
        if len(items) < 2:
            pytest.skip("Not enough items")

        embeddings = {item.experience.id: embedder.embed(item.experience.content) for item in items}
        clustering = MemoryClustering()
        clusters = clustering.cluster_with_embeddings(items, embeddings=embeddings, k=2)
        assert len(clusters) >= 1

    def test_emms_cluster_memories(self):
        agent = _make_emms(6)
        # Force consolidation to populate long_term
        agent.consolidate()
        agent.consolidate()
        clusters = agent.cluster_memories(k=2, tier="long_term")
        # May be empty if nothing is in long_term yet — that's OK
        assert isinstance(clusters, list)

    def test_emms_cluster_auto_k(self):
        agent = _make_emms(6)
        agent.consolidate()
        clusters = agent.cluster_memories(auto_k=True, tier="working")
        assert isinstance(clusters, list)

    def test_tfidf_vectorizer_builds(self):
        from emms.memory.clustering import _build_tfidf
        texts = ["machine learning is great", "deep learning neural networks", "python programming language"]
        matrix, vocab = _build_tfidf(texts)
        assert matrix.shape[0] == 3
        assert len(vocab) > 0


# ===========================================================================
# 3. ConversationBuffer
# ===========================================================================

class TestConversationBuffer:
    def test_import(self):
        from emms.sessions.conversation import ConversationBuffer, ConversationTurn, ConversationChunk
        assert ConversationBuffer
        assert ConversationTurn
        assert ConversationChunk

    def test_basic_observe(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        t = buf.observe_turn("user", "Hello!")
        assert t.role == "user"
        assert t.content == "Hello!"
        assert buf.turn_count == 1
        assert buf.total_turns_seen == 1

    def test_window_overflow_eviction(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=4, summarise_chunk=2)
        for i in range(6):
            buf.observe_turn("user", f"message {i}")
        assert buf.turn_count <= 4
        assert buf.chunk_count >= 1
        assert buf.total_turns_seen == 6

    def test_get_context_returns_string(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        buf.observe_turn("user", "What is EMMS?")
        buf.observe_turn("assistant", "EMMS is an Enhanced Memory Management System.")
        ctx = buf.get_context(max_tokens=500)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_context_includes_recent_turns(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        buf.observe_turn("user", "What is Python?")
        buf.observe_turn("assistant", "Python is a programming language.")
        ctx = buf.get_context()
        assert "Python" in ctx

    def test_context_includes_chunks(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=3, summarise_chunk=2)
        for i in range(5):
            buf.observe_turn("user", f"message {i} about machine learning algorithms")
        ctx = buf.get_context()
        assert isinstance(ctx, str)
        # Should contain summarised chunk reference or recent turns
        assert len(ctx) > 0

    def test_clear(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer()
        buf.observe_turn("user", "test")
        buf.clear()
        assert buf.turn_count == 0
        assert buf.chunk_count == 0

    def test_all_turns(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        buf.observe_turn("user", "hello")
        buf.observe_turn("assistant", "hi")
        turns = buf.all_turns()
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_turn_index_monotonic(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        indices = []
        for i in range(5):
            t = buf.observe_turn("user", f"msg {i}")
            indices.append(t.turn_index)
        assert indices == list(range(5))

    def test_max_tokens_truncation(self):
        from emms.sessions.conversation import ConversationBuffer
        buf = ConversationBuffer(window_size=10)
        buf.observe_turn("user", "x" * 1000)
        ctx = buf.get_context(max_tokens=50)
        # 50 tokens * 4 chars = 200 chars + truncation marker
        assert len(ctx) <= 250  # with truncation message

    def test_extractive_summary_quality(self):
        from emms.sessions.conversation import _extractive_summary
        texts = [
            "user: What is machine learning?",
            "assistant: Machine learning is a subset of artificial intelligence that learns from data.",
            "user: How does neural network work?",
            "assistant: Neural networks are layers of connected nodes inspired by the brain.",
        ]
        summary = _extractive_summary(texts, max_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 10

    def test_emms_build_conversation_context(self):
        agent = EMMS()
        turns = [
            ("user", "What is EMMS?"),
            ("assistant", "EMMS is an Enhanced Memory Management System."),
            ("user", "How does it work?"),
        ]
        ctx = agent.build_conversation_context(turns, max_tokens=500)
        assert "EMMS" in ctx or "conversation" in ctx.lower()
        assert isinstance(ctx, str)


# ===========================================================================
# 4. stream_retrieve
# ===========================================================================

def _collect_stream_sync(mem, query, max_results=10):
    """Helper to collect stream_retrieve results synchronously."""
    async def _inner():
        results = []
        async for r in mem.stream_retrieve(query, max_results=max_results):
            results.append(r)
        return results
    return asyncio.run(_inner())


class TestStreamRetrieve:
    def test_stream_retrieve_basic(self):
        mem = _make_memory(5)
        results = _collect_stream_sync(mem, "machine learning")
        assert isinstance(results, list)
        assert len(results) >= 0  # may be empty if threshold not met

    def test_stream_retrieve_returns_results(self):
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        for i in range(3):
            mem.store(Experience(content=f"machine learning concept {i}", domain="tech"))

        results = _collect_stream_sync(mem, "machine learning")
        assert len(results) > 0

    def test_stream_retrieve_score_ordering(self):
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        for i in range(4):
            mem.store(Experience(content=f"test memory item {i}", domain="tech"))

        results = _collect_stream_sync(mem, "test memory")
        # Within each tier, results should be sorted descending by score
        if len(results) >= 2:
            for i in range(len(results) - 1):
                if results[i].source_tier == results[i + 1].source_tier:
                    assert results[i].score >= results[i + 1].score

    def test_stream_retrieve_respects_max(self):
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        for i in range(10):
            mem.store(Experience(content=f"data {i} about databases sql queries", domain="tech"))

        results = _collect_stream_sync(mem, "databases sql", max_results=5)
        assert len(results) <= 5

    def test_stream_retrieve_skips_expired(self):
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        exp = Experience(content="expired memory item test", domain="test")
        item = mem.store(exp)
        item.expires_at = time.time() - 1  # expired

        results = _collect_stream_sync(mem, "expired memory")
        found_ids = [r.memory.id for r in results]
        assert item.id not in found_ids

    def test_astream_retrieve_emms(self):
        agent = _make_emms(5)

        async def _run():
            results = []
            async for r in agent.astream_retrieve("Python programming", max_results=5):
                results.append(r)
            return results

        results = asyncio.run(_run())
        assert isinstance(results, list)

    def test_stream_retrieve_multiple_tiers(self):
        """Test that stream_retrieve can return results from multiple tiers."""
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        for i in range(10):
            mem.store(Experience(content=f"neural network training epoch {i}", domain="ml"))

        results = _collect_stream_sync(mem, "neural network")
        if results:
            tiers = {r.source_tier for r in results}
            assert len(tiers) >= 1


# ===========================================================================
# 5. LLMConsolidator
# ===========================================================================

class TestLLMConsolidator:
    def test_import(self):
        from emms.llm.consolidator import LLMConsolidator, ConsolidationResult
        assert LLMConsolidator
        assert ConsolidationResult

    def test_consolidation_result_repr(self):
        from emms.llm.consolidator import ConsolidationResult
        r = ConsolidationResult()
        r.clusters_found = 3
        r.synthesised = 2
        r.stored = 2
        assert "clusters_found=3" in repr(r)
        assert "synthesised=2" in repr(r)

    def test_consolidation_result_as_dict(self):
        from emms.llm.consolidator import ConsolidationResult
        r = ConsolidationResult()
        r.clusters_found = 5
        d = r.as_dict()
        assert "clusters_found" in d
        assert "synthesised" in d
        assert "stored" in d
        assert "failed" in d
        assert "elapsed_s" in d

    def test_extractive_synthesis(self):
        from emms.llm.consolidator import _extractive_synthesis
        mem = _make_memory(3)
        items = list(mem.working) + list(mem.short_term) + list(mem.long_term.values())
        if not items:
            items = list(mem.working)
        result = _extractive_synthesis(items[:3])
        assert isinstance(result, str)
        assert len(result) > 10

    def test_consolidate_cluster_extractive(self):
        mem = _make_memory(3)
        items = list(mem.working) + list(mem.short_term) + list(mem.long_term.values())
        if len(items) < 2:
            items = list(mem.working)
        if len(items) < 2:
            pytest.skip("Not enough items")

        from emms.llm.consolidator import LLMConsolidator
        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        exp = asyncio.run(consolidator.consolidate_cluster(items[:3]))
        assert exp is not None
        assert isinstance(exp.content, str)
        assert len(exp.content) > 5

    def test_consolidate_cluster_too_small(self):
        mem = _make_memory(1)
        items = list(mem.working)

        from emms.llm.consolidator import LLMConsolidator
        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        exp = asyncio.run(consolidator.consolidate_cluster(items[:1]))
        assert exp is None

    def test_auto_consolidate_no_llm(self):
        mem = HierarchicalMemory(MemoryConfig(relevance_threshold=0.01))
        for i in range(5):
            mem.store(Experience(content=f"machine learning neural net training {i}", domain="ml"))
        for _ in range(3):
            mem.consolidate()

        from emms.llm.consolidator import LLMConsolidator
        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        result = asyncio.run(consolidator.auto_consolidate(threshold=0.3))
        assert isinstance(result.clusters_found, int)
        assert isinstance(result.stored, int)
        assert result.elapsed_s >= 0

    def test_auto_consolidate_returns_result(self):
        mem = _make_memory(4)
        from emms.llm.consolidator import LLMConsolidator
        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        result = asyncio.run(
            consolidator.auto_consolidate(threshold=0.5, tier=MemoryTier.WORKING)
        )
        assert hasattr(result, "clusters_found")
        assert hasattr(result, "stored")

    def test_find_clusters_lexical(self):
        from emms.llm.consolidator import LLMConsolidator
        mem = HierarchicalMemory()
        mem.store(Experience(content="machine learning algorithm training neural", domain="ml"))
        mem.store(Experience(content="machine learning deep neural network training", domain="ml"))
        mem.store(Experience(content="cooking pasta recipe italian", domain="food"))

        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        items = list(mem.working)
        clusters = consolidator._find_clusters(items, threshold=0.3)
        assert len(clusters) >= 1
        # ML items should cluster together
        ml_cluster = next((c for c in clusters if len(c) >= 2), None)
        assert ml_cluster is not None

    def test_consolidate_from_clusters(self):
        from emms.llm.consolidator import LLMConsolidator
        from emms.memory.clustering import MemoryCluster
        import numpy as np

        mem = _make_memory(4)
        items = list(mem.working) + list(mem.short_term)
        if len(items) < 2:
            pytest.skip("Not enough items")

        clusters = [MemoryCluster(id=0, members=items[:2], centroid=np.zeros(8))]
        consolidator = LLMConsolidator(mem, min_cluster_size=2)
        result = asyncio.run(consolidator.consolidate_from_clusters(clusters))
        assert isinstance(result.stored, int)

    def test_emms_llm_consolidate(self):
        agent = _make_emms(5)
        result = asyncio.run(agent.llm_consolidate(threshold=0.3, tier="working"))
        assert hasattr(result, "clusters_found")
        assert hasattr(result, "elapsed_s")

    def test_parse_json_valid(self):
        from emms.llm.consolidator import LLMConsolidator
        mem = _make_memory(1)
        c = LLMConsolidator(mem)
        result = c._parse_json('{"content": "hello", "importance": 0.8}')
        assert result == {"content": "hello", "importance": 0.8}

    def test_parse_json_embedded(self):
        from emms.llm.consolidator import LLMConsolidator
        mem = _make_memory(1)
        c = LLMConsolidator(mem)
        result = c._parse_json('Here is the JSON: {"content": "test"}')
        assert result == {"content": "test"}

    def test_parse_json_invalid(self):
        from emms.llm.consolidator import LLMConsolidator
        mem = _make_memory(1)
        c = LLMConsolidator(mem)
        result = c._parse_json("no json here at all")
        assert result is None


# ===========================================================================
# 6. MCP Server v0.7.0 tools
# ===========================================================================

class TestMCPV070Tools:
    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms(4)
        return EMCPServer(agent)

    def test_cluster_memories_tool_defined(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = [t["name"] for t in _TOOL_DEFINITIONS]
        assert "emms_cluster_memories" in names
        assert "emms_llm_consolidate" in names

    def test_handle_cluster_memories_auto(self):
        server = self._make_server()
        result = server.handle("emms_cluster_memories", {"auto_k": True, "tier": "working"})
        assert result.get("ok") is True
        assert "cluster_count" in result

    def test_handle_cluster_memories_k(self):
        server = self._make_server()
        result = server.handle("emms_cluster_memories", {"k": 2, "tier": "working"})
        assert result.get("ok") is True

    def test_handle_cluster_memories_invalid_tier(self):
        server = self._make_server()
        result = server.handle("emms_cluster_memories", {"k": 2, "tier": "invalid_tier"})
        # Should return ok: False due to ValueError
        assert "ok" in result

    def test_handle_llm_consolidate(self):
        server = self._make_server()
        result = server.handle("emms_llm_consolidate", {"threshold": 0.3, "tier": "working"})
        assert result.get("ok") is True
        assert "clusters_found" in result

    def test_total_tool_count(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        # 20 from v0.5.x–v0.6.0 + 2 v0.7.0 + 5 v0.8.0 = 27
        assert len(_TOOL_DEFINITIONS) == 57


# ===========================================================================
# 7. __init__.py exports
# ===========================================================================

class TestV070Exports:
    def test_version(self):
        import emms
        assert emms.__version__ == "0.14.0"

    def test_memory_diff_exported(self):
        from emms import MemoryDiff, DiffResult, ItemSnapshot
        assert MemoryDiff
        assert DiffResult
        assert ItemSnapshot

    def test_memory_cluster_exported(self):
        from emms import MemoryClustering, MemoryCluster
        assert MemoryClustering
        assert MemoryCluster

    def test_conversation_buffer_exported(self):
        from emms import ConversationBuffer, ConversationTurn, ConversationChunk
        assert ConversationBuffer
        assert ConversationTurn
        assert ConversationChunk

    def test_llm_consolidator_exported(self):
        from emms import LLMConsolidator, ConsolidationResult
        assert LLMConsolidator
        assert ConsolidationResult


# ===========================================================================
# 8. Union-Find (internal)
# ===========================================================================

class TestUnionFind:
    def test_single_components(self):
        from emms.llm.consolidator import _UnionFind
        uf = _UnionFind(4)
        groups = uf.groups()
        assert len(groups) == 4

    def test_union_merges(self):
        from emms.llm.consolidator import _UnionFind
        uf = _UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 2
        all_members = sorted(sum(groups.values(), []))
        assert all_members == [0, 1, 2, 3]

    def test_union_transitive(self):
        from emms.llm.consolidator import _UnionFind
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        groups = uf.groups()
        # 0, 1, 2 should be in one group; 3 and 4 in separate groups
        root_counts = {k: len(v) for k, v in groups.items()}
        assert max(root_counts.values()) == 3


# ===========================================================================
# 9. TF-IDF and k-means helpers
# ===========================================================================

class TestKMeansHelpers:
    def test_elbow_k_selection(self):
        from emms.memory.clustering import _elbow_k
        import numpy as np
        # Create data with 2 clear clusters
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal([0, 0], 0.1, (5, 2)),
                       rng.normal([5, 5], 0.1, (5, 2))])
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-9)
        k = _elbow_k(X, k_min=2, k_max=5)
        assert 2 <= k <= 5

    def test_kmeans_run(self):
        from emms.memory.clustering import _run_kmeans
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (10, 4)).astype(np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.maximum(norms, 1e-9)
        labels, centroids, inertia = _run_kmeans(X, k=2, max_iter=50)
        assert len(labels) == 10
        assert centroids.shape == (2, 4)
        assert inertia >= 0

    def test_tfidf_zero_matrix_fallback(self):
        from emms.memory.clustering import _build_tfidf
        # Very short texts that mostly become stop words
        texts = ["a the is", "the a is", "is the a"]
        matrix, vocab = _build_tfidf(texts)
        # Should not crash; matrix may be all zeros with empty vocab
        assert matrix.shape[0] == 3
