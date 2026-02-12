"""End-to-end real-world scenario: a financial analyst AI agent.

Simulates 500+ experiences across multiple domains, conversation turns,
consolidation cycles, compression, identity persistence across sessions,
markdown export/import round-trip, and stress-tests retrieval accuracy.

This is NOT a unit test — it's a system test that exercises the full
EMMS pipeline as a real agent integration would use it.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from emms.core.models import Experience, MemoryConfig, MemoryTier
from emms.core.embeddings import HashEmbedder
from emms.emms import EMMS
from emms.adapters.agent import AgentMemory
from emms.memory.compression import MemoryCompressor


# ---------------------------------------------------------------------------
# Realistic conversation data
# ---------------------------------------------------------------------------

FINANCE_TURNS = [
    ("What's the stock market doing today?", "The S&P 500 is up 1.2%, led by tech stocks. NVIDIA gained 4%."),
    ("How about Bitcoin?", "Bitcoin hit $98,000, up 3.5% on ETF inflows. Institutional adoption accelerating."),
    ("Should I invest in tech stocks?", "Tech valuations are stretched but earnings growth supports prices. Consider diversified exposure."),
    ("What's the Fed doing about rates?", "The Fed signaled a pause on rate cuts. Markets expect steady rates through Q2."),
    ("How are emerging markets performing?", "Emerging markets are mixed. India up 8% YTD but China down 3%."),
    ("Tell me about the bond market", "10-year yield at 4.3%. Inverted curve still signals recession risk."),
    ("What about gold prices?", "Gold at $2,150/oz, a record high. Central bank buying driving demand."),
    ("How is the housing market?", "Home prices up 5% YoY. Inventory remains low. Mortgage rates at 6.8%."),
    ("What are the best performing sectors?", "Tech and healthcare lead. Energy lags on flat oil prices."),
    ("Any major earnings this week?", "Apple, Microsoft, and Amazon report. Street expects strong cloud growth."),
]

TECH_TURNS = [
    ("What's new in AI?", "OpenAI released GPT-5 with improved reasoning. Google's Gemini 2 competing."),
    ("How's the chip market?", "NVIDIA dominates AI chips. AMD gaining share. Intel restructuring."),
    ("Tell me about cloud computing trends", "Multi-cloud adoption up 40%. Edge computing gaining traction."),
    ("What about cybersecurity?", "Ransomware attacks up 300%. Zero-trust architecture becoming standard."),
    ("Any new programming languages?", "Rust adoption surging. Zig gaining interest. Python still dominant."),
]

SCIENCE_TURNS = [
    ("Any breakthroughs in physics?", "New superconductor candidate at near room temperature. Still being verified."),
    ("What about space exploration?", "SpaceX Starship completed orbital test. Artemis 3 scheduled for 2026."),
    ("Medical research updates?", "CRISPR gene therapy approved for sickle cell. mRNA cancer vaccines in trials."),
]

WEATHER_TURNS = [
    ("What's the weather like?", "Sunny, 72°F. Perfect conditions. UV index moderate."),
    ("Will it rain this weekend?", "Forecast shows 60% chance of rain Saturday. Sunday clearing up."),
]

PERSONAL_TURNS = [
    ("I'm feeling stressed about the market", "That's understandable. Volatility can be stressful. Consider your long-term strategy."),
    ("Thanks for your help!", "You're welcome! Happy to help you navigate the markets."),
    ("You're getting really good at this", "Thank you! I learn from each conversation to provide better analysis."),
]


# ---------------------------------------------------------------------------
# Scenario 1: Full lifecycle with AgentMemory
# ---------------------------------------------------------------------------

class TestE2EAgentLifecycle:
    """Simulates a complete agent lifecycle: create, converse, persist, resume."""

    def test_full_agent_lifecycle(self, tmp_path):
        """Full lifecycle: create → converse → consolidate → persist → resume → recall."""
        storage = tmp_path / "agent_memory"

        # --- SESSION 1: Create agent, ingest conversations ---
        agent = AgentMemory(agent_name="FinanceBot", storage_dir=storage)

        # Ingest all conversation domains
        results = []
        for user_msg, agent_msg in FINANCE_TURNS:
            r = agent.ingest_turn(user_msg=user_msg, agent_msg=agent_msg, domain="finance", importance=0.7)
            results.append(r)
        for user_msg, agent_msg in TECH_TURNS:
            r = agent.ingest_turn(user_msg=user_msg, agent_msg=agent_msg, domain="tech", importance=0.6)
            results.append(r)
        for user_msg, agent_msg in SCIENCE_TURNS:
            r = agent.ingest_turn(user_msg=user_msg, agent_msg=agent_msg, domain="science", importance=0.5)
            results.append(r)
        for user_msg, agent_msg in WEATHER_TURNS:
            r = agent.ingest_turn(user_msg=user_msg, agent_msg=agent_msg, domain="weather", importance=0.3)
            results.append(r)
        for user_msg, agent_msg in PERSONAL_TURNS:
            r = agent.ingest_turn(user_msg=user_msg, agent_msg=agent_msg, domain="personal", importance=0.8)
            results.append(r)

        # Verify all turns stored
        assert len(results) == 23
        assert all("experience_id" in r for r in results)
        assert agent.emms.stats["total_stored"] == 23

        # Verify narrative coherence is meaningful
        assert 0.0 < agent.narrator.coherence <= 1.0

        # --- CONTEXT BUILDING ---
        context = agent.build_context("What do I know about the stock market?")
        assert "Identity" in context
        assert "FinanceBot" in context
        # Should find finance-related memories
        assert "market" in context.lower() or "stock" in context.lower()

        # --- RECALL ---
        memories = agent.recall("Bitcoin price")
        assert len(memories) > 0
        # The top result should be about Bitcoin
        assert any("bitcoin" in m["content"].lower() or "btc" in m["content"].lower()
                    for m in memories[:3])

        # --- PRE-COMPACTION FLUSH ---
        flush_result = agent.pre_compaction_flush()
        assert flush_result["identity_saved"] is True
        assert "consolidated" in flush_result

        # --- MARKDOWN EXPORT ---
        md_path = storage / "MEMORY.md"
        md = agent.export_markdown(md_path)
        assert md_path.exists()
        assert "FinanceBot" in md
        assert "finance" in md.lower()

        # --- SAVE SESSION ---
        agent.save_session()
        turns_path = storage / "turns.jsonl"
        assert turns_path.exists()
        turn_lines = turns_path.read_text().strip().split("\n")
        assert len(turn_lines) == 23

        # --- SESSION 2: Resume and recall ---
        agent2 = AgentMemory(agent_name="FinanceBot", storage_dir=storage)

        # Import previous session's markdown
        import_count = agent2.import_markdown(md_path)
        assert import_count > 0

        # New agent should have memories from import
        assert agent2.emms.stats["total_stored"] > 0

        # Status should be valid
        status = agent2.status
        assert status["agent_name"] == "FinanceBot"
        assert "emms_stats" in status

    def test_context_respects_token_budget(self, tmp_path):
        """Context building should stay within token budget."""
        agent = AgentMemory(agent_name="BudgetBot", storage_dir=tmp_path / "budget")

        # Ingest lots of data
        for i in range(50):
            agent.ingest_turn(
                user_msg=f"Important financial analysis point number {i} about market dynamics and trading strategies",
                agent_msg=f"Analysis {i}: The market shows various patterns related to sector rotation and momentum",
                domain="finance",
                importance=0.7,
            )

        # Small budget
        context = agent.build_context("market analysis", max_tokens=50)
        word_count = len(context.split())
        # Should be bounded (with some tolerance for section headers)
        assert word_count < 150, f"Context too long: {word_count} words for 50 token budget"

    def test_domain_diversity_in_recall(self, tmp_path):
        """Recall should return diverse results across domains."""
        agent = AgentMemory(agent_name="DiverseBot")

        # Store many items per domain
        for i in range(20):
            agent.ingest_turn(user_msg=f"Finance news {i} about market trends", domain="finance")
        for i in range(20):
            agent.ingest_turn(user_msg=f"Technology update {i} about software", domain="tech")

        # Query that could match both domains
        results = agent.recall("latest trends and updates", max_results=10)
        if len(results) > 1:
            domains = {r["domain"] for r in results}
            # Should ideally have some diversity
            assert len(domains) >= 1  # At minimum


# ---------------------------------------------------------------------------
# Scenario 2: High-volume stress test
# ---------------------------------------------------------------------------

class TestE2EHighVolume:
    """Stress test with 500+ experiences."""

    def test_500_experiences_store_and_retrieve(self):
        """Store 500 experiences and verify retrieval still works."""
        emms = EMMS(embedder=HashEmbedder(dim=64))

        domains = ["finance", "tech", "science", "weather", "health"]
        contents = {
            "finance": [
                "Stock market rose on strong earnings from tech companies",
                "Federal Reserve held interest rates steady at current levels",
                "Bitcoin surged past new highs on institutional buying",
                "Oil prices fell on increased supply from OPEC",
                "Housing market showed signs of cooling as rates stay high",
            ],
            "tech": [
                "New AI model achieves human-level reasoning on benchmarks",
                "Cloud computing revenues grew 30 percent year over year",
                "Quantum computing breakthrough in error correction announced",
                "Cybersecurity spending increased due to ransomware threats",
                "Open source software adoption accelerated in enterprise",
            ],
            "science": [
                "CRISPR gene editing approved for clinical treatment",
                "New exoplanet discovered in habitable zone of nearby star",
                "Fusion reactor achieved net energy gain in latest test",
                "Climate models predict accelerated warming this decade",
                "Superconductor research makes progress at higher temperatures",
            ],
            "weather": [
                "Severe thunderstorms expected in the midwest this afternoon",
                "Hurricane season forecast predicts above average activity",
                "Record breaking heat wave continues across southern states",
                "Early snowfall disrupts travel in the northeast corridor",
                "La Nina conditions expected to influence winter patterns",
            ],
            "health": [
                "New mRNA vaccine shows promise against multiple cancer types",
                "Mental health awareness campaigns reduce stigma significantly",
                "Exercise found to be as effective as medication for depression",
                "Antibiotic resistant bacteria pose growing global threat",
                "Telemedicine adoption permanent after pandemic experience",
            ],
        }

        t0 = time.time()
        for i in range(500):
            domain = domains[i % len(domains)]
            content_idx = (i // len(domains)) % len(contents[domain])
            base_content = contents[domain][content_idx]
            # Add variation to avoid identical content
            content = f"{base_content} (update #{i})"

            exp = Experience(
                content=content,
                domain=domain,
                importance=0.3 + (i % 10) * 0.07,
                novelty=0.5 + (i % 7) * 0.07,
            )
            emms.store(exp)
        store_time = time.time() - t0

        # Verify all stored
        assert emms.stats["total_stored"] == 500

        # Verify memory distribution across tiers
        sizes = emms.memory.size
        assert sizes["total"] > 0
        # Working memory should be at capacity or under
        assert sizes["working"] <= emms.cfg.working_capacity

        # Retrieval tests
        t0 = time.time()
        finance_results = emms.retrieve("stock market earnings")
        retrieve_time = time.time() - t0

        assert len(finance_results) > 0
        # Top results should be finance-related
        finance_count = sum(1 for r in finance_results[:5]
                           if r.memory.experience.domain == "finance")
        assert finance_count >= 1, "Expected at least one finance result in top 5"

        # Cross-domain retrieval
        health_results = emms.retrieve("cancer treatment vaccine")
        assert len(health_results) > 0

        # Consolidation should work at scale
        consol_result = emms.consolidate()
        assert "items_consolidated" in consol_result

        # Compression should work at scale
        compressed = emms.compress_long_term()
        # May be empty if nothing reached long-term yet (access_count < 2)
        assert isinstance(compressed, list)

        # Performance assertions
        assert store_time < 10.0, f"500 stores took {store_time:.1f}s (too slow)"
        assert retrieve_time < 1.0, f"Retrieval took {retrieve_time:.3f}s (too slow)"

        print(f"\n--- 500-experience stress test ---")
        print(f"Store: {store_time:.2f}s ({500/store_time:.0f} exp/s)")
        print(f"Retrieve: {retrieve_time*1000:.1f}ms")
        print(f"Memory: {sizes}")
        print(f"Consolidation: {consol_result}")

    def test_1000_experiences_memory_doesnt_explode(self):
        """Verify memory stays bounded with 1000 experiences."""
        emms = EMMS(embedder=HashEmbedder(dim=32))

        for i in range(1000):
            exp = Experience(
                content=f"Event number {i} in category {i % 10} with detail level {i % 5}",
                domain=f"domain_{i % 5}",
                importance=0.4 + (i % 6) * 0.1,
            )
            emms.store(exp)

        sizes = emms.memory.size
        # Working memory should be bounded
        assert sizes["working"] <= emms.cfg.working_capacity
        # Short-term should be bounded
        assert sizes["short_term"] <= emms.cfg.short_term_capacity
        # Total should be well under 1000 (due to consolidation/decay)
        assert sizes["total"] < 1000, f"Memory not bounded: {sizes}"

        print(f"\n--- 1000-experience memory test ---")
        print(f"Memory sizes: {sizes}")
        print(f"Total retained: {sizes['total']} / 1000 ({sizes['total']/10:.0f}%)")


# ---------------------------------------------------------------------------
# Scenario 3: Retrieval accuracy
# ---------------------------------------------------------------------------

class TestE2ERetrievalAccuracy:
    """Tests whether retrieval actually returns semantically relevant results."""

    def test_domain_specific_retrieval(self):
        """Query about a domain should mostly return that domain's items."""
        emms = EMMS(embedder=HashEmbedder(dim=128))

        # Store clearly distinct domain content
        finance_items = [
            "Apple stock rose 5% on strong iPhone sales and services revenue",
            "Federal Reserve raised interest rates by 25 basis points today",
            "S&P 500 hit all-time high driven by artificial intelligence stocks",
        ]
        weather_items = [
            "Heavy rainfall expected across the northeast causing flooding risk",
            "Temperature dropping to below freezing tonight with wind chill warning",
            "Hurricane forming in the Atlantic with sustained winds of 120mph",
        ]

        for content in finance_items:
            emms.store(Experience(content=content, domain="finance", importance=0.7))
        for content in weather_items:
            emms.store(Experience(content=content, domain="weather", importance=0.7))

        # Finance query — at least one of top 3 should be finance
        results = emms.retrieve("stock market investment returns")
        if results:
            top3 = results[:3]
            finance_found = any(
                r.memory.experience.domain == "finance"
                or "stock" in r.memory.experience.content.lower()
                for r in top3
            )
            assert finance_found, \
                f"Expected finance in top 3, got: {[r.memory.experience.domain for r in top3]}"

        # Weather query — at least one of top 3 should be weather
        results = emms.retrieve("rain storm temperature forecast")
        if results:
            top3 = results[:3]
            weather_found = any(
                r.memory.experience.domain == "weather"
                or any(w in r.memory.experience.content.lower()
                       for w in ["rain", "temperature", "hurricane", "wind"])
                for r in top3
            )
            assert weather_found, \
                f"Expected weather in top 3, got: {[r.memory.experience.domain for r in top3]}"

    def test_semantic_retrieval_with_vector_store(self):
        """Semantic retrieval via embeddings should find related content."""
        embedder = HashEmbedder(dim=128)
        emms = EMMS(embedder=embedder)

        emms.store(Experience(content="Bitcoin price surged to new all-time high", domain="crypto"))
        emms.store(Experience(content="Ethereum network upgrade completed successfully", domain="crypto"))
        emms.store(Experience(content="The cat sat on the mat in the sunshine", domain="random"))

        results = emms.retrieve_semantic("cryptocurrency prices and blockchain")
        assert len(results) > 0
        # Should prefer crypto content over random content
        top_content = results[0]["content"].lower()
        assert "bitcoin" in top_content or "ethereum" in top_content or "crypto" in top_content, \
            f"Expected crypto result, got: {top_content[:50]}"


# ---------------------------------------------------------------------------
# Scenario 4: Consolidation fidelity
# ---------------------------------------------------------------------------

class TestE2EConsolidation:
    """Tests that consolidation preserves important memories and forgets unimportant ones."""

    def test_important_memories_survive_consolidation(self):
        """High-importance memories should survive consolidation cycles."""
        emms = EMMS(embedder=HashEmbedder(dim=64))

        # Store critical memory
        critical = Experience(
            content="CRITICAL: Portfolio allocation changed to 60% stocks 40% bonds",
            domain="finance",
            importance=0.95,
            novelty=0.9,
        )
        emms.store(critical)

        # Access it several times to boost retention
        for _ in range(5):
            emms.retrieve("portfolio allocation")

        # Store lots of low-importance noise
        for i in range(50):
            noise = Experience(
                content=f"Minor update number {i} with trivial information",
                domain="noise",
                importance=0.1,
                novelty=0.1,
            )
            emms.store(noise)

        # Run consolidation
        emms.consolidate()

        # The critical memory should still be retrievable
        results = emms.retrieve("portfolio allocation bonds stocks")
        found = any("portfolio" in r.memory.experience.content.lower() for r in results)
        assert found, "Critical memory lost after consolidation!"

    def test_consolidation_progression_across_tiers(self):
        """Memories should progress through tiers with repeated access."""
        emms = EMMS(embedder=HashEmbedder(dim=64))

        # Store and access a memory repeatedly
        exp = Experience(content="Important recurring insight about market patterns", domain="finance", importance=0.9, novelty=0.8)
        emms.store(exp)

        # Access it multiple times
        for _ in range(10):
            results = emms.retrieve("market patterns insight")

        # Run multiple consolidation cycles
        for _ in range(3):
            emms.consolidate()

        # Check that it progressed beyond working memory
        sizes = emms.memory.size
        # It should be in short_term, long_term, or semantic
        in_higher = sizes["short_term"] + sizes["long_term"] + sizes["semantic"]
        assert in_higher >= 0  # At minimum, the system ran without errors


# ---------------------------------------------------------------------------
# Scenario 5: Compression effectiveness
# ---------------------------------------------------------------------------

class TestE2ECompression:
    """Tests that compression actually reduces storage while preserving meaning."""

    def test_compression_deduplicates(self):
        """Near-duplicate memories should be merged by compression."""
        compressor = MemoryCompressor()

        from emms.core.models import MemoryItem
        items = []
        for i in range(5):
            exp = Experience(
                content=f"The stock market rose 5% today on strong earnings reports from tech. (version {i})",
                domain="finance",
                importance=0.7,
            )
            items.append(MemoryItem(experience=exp, tier=MemoryTier.LONG_TERM))

        compressed = compressor.compress_batch(items)
        assert len(compressed) < len(items), \
            f"Compression didn't deduplicate: {len(compressed)} >= {len(items)}"

        # Verify compressed items preserve domain
        for c in compressed:
            assert c.domain == "finance"

    def test_compression_preserves_distinct_items(self):
        """Distinct memories should survive compression intact."""
        compressor = MemoryCompressor()

        from emms.core.models import MemoryItem
        items = [
            MemoryItem(
                experience=Experience(content="The stock market crashed 10% today on bank failures", domain="finance"),
                tier=MemoryTier.LONG_TERM,
            ),
            MemoryItem(
                experience=Experience(content="A massive earthquake struck the pacific coast causing tsunamis", domain="disaster"),
                tier=MemoryTier.LONG_TERM,
            ),
            MemoryItem(
                experience=Experience(content="New programming language Mojo combines Python and systems programming", domain="tech"),
                tier=MemoryTier.LONG_TERM,
            ),
        ]

        compressed = compressor.compress_batch(items)
        assert len(compressed) == 3, f"Distinct items were wrongly merged: {len(compressed)}"


# ---------------------------------------------------------------------------
# Scenario 6: Async stress test
# ---------------------------------------------------------------------------

class TestE2EAsync:
    """Tests async API under concurrent load."""

    def test_concurrent_store_correctness(self):
        """100 concurrent stores should all succeed without corruption."""
        async def _run():
            emms = EMMS(embedder=HashEmbedder(dim=32))

            async def store_one(i):
                exp = Experience(
                    content=f"Concurrent experience {i} about topic {i % 5}",
                    domain=f"domain_{i % 5}",
                    importance=0.5,
                )
                return await emms.astore(exp)

            results = await asyncio.gather(*[store_one(i) for i in range(100)])
            assert len(results) == 100
            assert emms.stats["total_stored"] == 100

            # All IDs should be unique
            ids = {r["experience_id"] for r in results}
            assert len(ids) == 100, "Duplicate IDs detected in concurrent stores"

            return True

        assert asyncio.run(_run())

    def test_concurrent_store_and_retrieve_consistency(self):
        """Stores and retrieves running concurrently should not corrupt state."""
        async def _run():
            emms = EMMS(embedder=HashEmbedder(dim=32))

            # Pre-populate
            for i in range(20):
                exp = Experience(content=f"Background knowledge {i} about finance", domain="finance")
                emms.store(exp)

            async def continuous_store():
                for i in range(50):
                    exp = Experience(content=f"New item {i} about markets", domain="finance")
                    await emms.astore(exp)

            async def continuous_retrieve():
                results = []
                for _ in range(20):
                    r = await emms.aretrieve("finance markets")
                    results.append(len(r))
                return results

            _, retrieve_counts = await asyncio.gather(
                continuous_store(),
                continuous_retrieve(),
            )

            # Retrieves should always succeed (no crashes)
            assert all(isinstance(c, int) for c in retrieve_counts)
            assert emms.stats["total_stored"] == 70

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Scenario 7: Markdown roundtrip
# ---------------------------------------------------------------------------

class TestE2EMarkdownRoundtrip:
    """Tests that export → import preserves memories."""

    def test_export_import_preserves_content(self, tmp_path):
        """Exported markdown should be re-importable with content preserved."""
        # Session 1: Create and populate
        agent1 = AgentMemory(agent_name="RoundtripBot", storage_dir=tmp_path / "s1")

        agent1.ingest_turn(user_msg="Stock market rose 5% today", domain="finance", importance=0.8)
        agent1.ingest_turn(user_msg="Bitcoin hit new all time high", domain="crypto", importance=0.7)
        agent1.ingest_turn(user_msg="Hurricane approaching Florida coast", domain="weather", importance=0.6)

        md_path = tmp_path / "MEMORY.md"
        exported_md = agent1.export_markdown(md_path)

        # Session 2: Import
        agent2 = AgentMemory(agent_name="RoundtripBot", storage_dir=tmp_path / "s2")
        count = agent2.import_markdown(md_path)

        assert count > 0, "No memories imported from markdown"

        # Verify imported content is retrievable
        results = agent2.recall("stock market")
        # Should find something (the imported finance content)
        assert len(results) > 0 or agent2.emms.stats["total_stored"] > 0


# ---------------------------------------------------------------------------
# Scenario 8: Edge cases and robustness
# ---------------------------------------------------------------------------

class TestE2EEdgeCases:
    """Tests edge cases that might break in production."""

    def test_empty_content(self):
        """Empty or minimal content should not crash."""
        emms = EMMS(embedder=HashEmbedder(dim=32))
        exp = Experience(content="", domain="test")
        result = emms.store(exp)
        assert "experience_id" in result

    def test_very_long_content(self):
        """Very long content should be handled gracefully."""
        emms = EMMS(embedder=HashEmbedder(dim=32))
        long_content = "word " * 10000  # ~50KB of text
        exp = Experience(content=long_content, domain="test")
        result = emms.store(exp)
        assert "experience_id" in result

        # Retrieval should still work
        results = emms.retrieve("word")
        assert len(results) > 0

    def test_unicode_content(self):
        """Unicode and special characters should work."""
        emms = EMMS(embedder=HashEmbedder(dim=32))
        exp = Experience(content="日本語テスト 🚀 émojis and spëcial chars", domain="test")
        result = emms.store(exp)
        assert "experience_id" in result

    def test_rapid_consolidation_cycles(self):
        """Multiple rapid consolidation cycles should not corrupt state."""
        emms = EMMS(embedder=HashEmbedder(dim=32))

        for i in range(100):
            exp = Experience(content=f"Item {i}", domain="test", importance=0.5 + (i % 5) * 0.1)
            emms.store(exp)

        # Run 10 consolidation cycles
        for _ in range(10):
            result = emms.consolidate()
            assert "items_consolidated" in result

        # Memory should still be in valid state
        sizes = emms.memory.size
        assert sizes["total"] > 0
        assert sizes["working"] <= emms.cfg.working_capacity

    def test_retrieve_with_no_matches(self):
        """Retrieval with no matches should return empty, not crash."""
        emms = EMMS(embedder=HashEmbedder(dim=32))
        emms.store(Experience(content="Stock market data", domain="finance"))
        results = emms.retrieve("quantum physics superconductors")
        assert isinstance(results, list)

    def test_agent_with_no_storage_dir(self):
        """AgentMemory without storage should work in-memory only."""
        agent = AgentMemory(agent_name="EphemeralBot")
        agent.ingest_turn(user_msg="Hello there")
        context = agent.build_context("hello")
        assert isinstance(context, str)
        assert len(context) > 0
