"""Tests for the PatternDetector in compression module."""

import pytest
from emms.core.models import Experience, MemoryItem, MemoryTier
from emms.memory.compression import PatternDetector


@pytest.fixture
def detector():
    return PatternDetector()


@pytest.fixture
def finance_items():
    """Create a sequence of finance-related memory items."""
    items = []
    for i in range(10):
        exp = Experience(
            content=f"Stock market trading volume item {i} with significant changes",
            domain="finance",
            importance=0.7,
        )
        items.append(MemoryItem(experience=exp, tier=MemoryTier.LONG_TERM))
    return items


@pytest.fixture
def mixed_items():
    """Create items across multiple domains."""
    domains = ["finance", "tech", "finance", "science", "tech",
               "finance", "tech", "finance", "science", "health"]
    items = []
    for i, domain in enumerate(domains):
        exp = Experience(
            content=f"Content about {domain} topics including important research findings {i}",
            domain=domain,
            importance=0.5 + i * 0.05,
        )
        items.append(MemoryItem(experience=exp, tier=MemoryTier.LONG_TERM))
    return items


class TestSequencePatterns:
    def test_finds_repeated_sequences(self, detector, mixed_items):
        result = detector.find_sequence_patterns(mixed_items)
        assert "patterns" in result
        assert "count" in result

    def test_empty_items(self, detector):
        result = detector.find_sequence_patterns([])
        assert result["count"] == 0

    def test_single_item(self, detector, finance_items):
        result = detector.find_sequence_patterns(finance_items[:1])
        assert result["count"] == 0  # need at least 2 for a pattern


class TestContentPatterns:
    def test_finds_recurring_concepts(self, detector, finance_items):
        result = detector.find_content_patterns(finance_items)
        assert "concepts" in result
        # "stock", "market", "trading" should appear frequently
        concept_terms = {c["term"] for c in result["concepts"]}
        assert len(concept_terms) > 0

    def test_finds_bigrams(self, detector, finance_items):
        result = detector.find_content_patterns(finance_items)
        assert "bigrams" in result

    def test_empty_items(self, detector):
        result = detector.find_content_patterns([])
        assert result["count"] == 0


class TestDomainPatterns:
    def test_domain_distribution(self, detector, mixed_items):
        result = detector.find_domain_patterns(mixed_items)
        assert "distribution" in result
        assert "dominant" in result
        assert result["dominant"] == "finance"  # most common domain

    def test_trends(self, detector, mixed_items):
        result = detector.find_domain_patterns(mixed_items)
        assert "trends" in result
        assert isinstance(result["trends"], list)

    def test_empty_items(self, detector):
        result = detector.find_domain_patterns([])
        assert result["dominant"] is None

    def test_single_domain(self, detector, finance_items):
        result = detector.find_domain_patterns(finance_items)
        assert result["dominant"] == "finance"
        assert result["distribution"]["finance"] == 1.0
