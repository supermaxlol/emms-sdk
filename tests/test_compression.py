"""Tests for memory compression module."""

import pytest
from emms.core.models import Experience, MemoryItem, MemoryTier
from emms.memory.compression import MemoryCompressor, CompressedMemory


@pytest.fixture
def compressor():
    return MemoryCompressor()


@pytest.fixture
def sample_item():
    exp = Experience(
        content="The stock market rose 5% today driven by strong earnings reports from major tech companies",
        domain="finance",
        importance=0.8,
    )
    return MemoryItem(experience=exp, tier=MemoryTier.LONG_TERM)


class TestCompressSingle:
    def test_compress_returns_compressed_memory(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert isinstance(result, CompressedMemory)
        assert result.id == sample_item.id

    def test_compress_extracts_keywords(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert len(result.keywords) > 0
        assert any("stock" in k or "market" in k for k in result.keywords)

    def test_compress_has_summary(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert len(result.summary) > 0
        assert len(result.summary) <= len(sample_item.experience.content) + 10

    def test_compress_tracks_source(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert sample_item.experience.id in result.source_ids

    def test_compress_preserves_domain(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert result.domain == "finance"

    def test_compress_fidelity_positive(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert 0.0 < result.fidelity <= 1.0

    def test_compress_ratio_positive(self, compressor, sample_item):
        result = compressor.compress(sample_item)
        assert result.compression_ratio >= 1.0 or len(sample_item.experience.content) < 20


class TestCompressBatch:
    def test_batch_compress_groups_duplicates(self, compressor):
        items = []
        for i in range(3):
            exp = Experience(
                content="The stock market rose 5% today on earnings",
                domain="finance",
                importance=0.7,
            )
            items.append(MemoryItem(experience=exp, tier=MemoryTier.LONG_TERM))

        results = compressor.compress_batch(items)
        # Near-duplicates should be merged
        assert len(results) < len(items)

    def test_batch_compress_different_items_stay_separate(self, compressor):
        items = [
            MemoryItem(
                experience=Experience(content="The stock market rose 5% today", domain="finance"),
                tier=MemoryTier.LONG_TERM,
            ),
            MemoryItem(
                experience=Experience(content="A hurricane hit the coast causing floods", domain="weather"),
                tier=MemoryTier.LONG_TERM,
            ),
        ]
        results = compressor.compress_batch(items)
        assert len(results) == 2

    def test_empty_batch(self, compressor):
        assert compressor.compress_batch([]) == []


class TestExpand:
    def test_expand_returns_original(self, compressor, sample_item):
        compressed = compressor.compress(sample_item)
        expanded = compressor.expand(compressed)
        assert expanded == sample_item.experience.content

    def test_expand_unknown_returns_summary(self, compressor):
        compressed = CompressedMemory(
            id="unknown_id",
            summary="Market rose today",
            source_ids=["exp_123"],
            keywords=["market", "rose"],
        )
        expanded = compressor.expand(compressed)
        assert "Market rose today" in expanded


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self, compressor):
        keywords = compressor._extract_keywords(
            "The quantum computing breakthrough enables faster drug discovery"
        )
        assert "quantum" in keywords or "computing" in keywords
        assert "the" not in keywords

    def test_empty_text(self, compressor):
        assert compressor._extract_keywords("") == []


class TestSummarisation:
    def test_short_text_unchanged(self, compressor):
        text = "Markets rose today."
        summary = compressor._summarise(text)
        assert summary == text.strip()

    def test_long_text_shortened(self, compressor):
        text = (
            "The stock market rose sharply today. "
            "This was driven by strong earnings reports. "
            "Tech companies led the gains with double digit growth. "
            "Analysts expect continued momentum next quarter. "
            "However some sectors remain under pressure. "
            "Energy stocks declined on falling oil prices. "
            "Overall market sentiment remains positive."
        )
        summary = compressor._summarise(text)
        assert len(summary.split()) <= compressor.max_summary_words + 10
