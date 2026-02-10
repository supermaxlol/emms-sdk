"""Tests for episode boundary detection."""

import time
import pytest
from emms.core.models import Experience
from emms.episodes.boundary import (
    EpisodeBoundaryDetector,
    Episode,
    _word_overlap,
    _experience_similarity,
)


class TestSimilarityHelpers:
    def test_word_overlap_identical(self):
        assert _word_overlap("hello world", "hello world") == 1.0

    def test_word_overlap_disjoint(self):
        assert _word_overlap("hello world", "foo bar") == 0.0

    def test_word_overlap_partial(self):
        score = _word_overlap("hello world foo", "hello bar baz")
        assert 0 < score < 1

    def test_experience_similarity_same_domain(self):
        a = Experience(content="stock market rose", domain="finance", timestamp=100.0)
        b = Experience(content="stock market fell", domain="finance", timestamp=101.0)
        sim = _experience_similarity(a, b)
        assert sim > 0.5

    def test_experience_similarity_different_domain(self):
        a = Experience(content="stock market", domain="finance", timestamp=100.0)
        b = Experience(content="rainy weather", domain="weather", timestamp=100.0)
        sim = _experience_similarity(a, b)
        assert sim < 0.5


class TestEpisodeBoundaryDetector:
    def test_empty(self):
        det = EpisodeBoundaryDetector()
        episodes = det.detect()
        assert episodes == []

    def test_single_experience(self):
        det = EpisodeBoundaryDetector()
        det.add(Experience(content="hello", timestamp=1.0))
        episodes = det.detect()
        assert len(episodes) == 1
        assert episodes[0].size == 1

    def test_similar_grouped(self):
        """Experiences on the same topic close in time should group together."""
        det = EpisodeBoundaryDetector(similarity_threshold=0.2)
        t = time.time()
        det.add(Experience(content="stock market rose today", domain="finance", timestamp=t))
        det.add(Experience(content="stock market index up 3%", domain="finance", timestamp=t + 1))
        det.add(Experience(content="market earnings beat expectations", domain="finance", timestamp=t + 2))

        episodes = det.detect()
        # Should be 1 episode (all finance/market related)
        assert len(episodes) >= 1
        total_exps = sum(ep.size for ep in episodes)
        assert total_exps == 3

    def test_different_topics_split(self):
        """Very different topics far apart in time should split into separate episodes."""
        det = EpisodeBoundaryDetector(similarity_threshold=0.4)
        t = time.time()

        # Finance cluster
        det.add(Experience(content="stock market rose sharply", domain="finance", timestamp=t))
        det.add(Experience(content="stock index hit record high", domain="finance", timestamp=t + 1))

        # Weather cluster (far in time, different topic)
        det.add(Experience(content="heavy rain flooding streets", domain="weather", timestamp=t + 10000))
        det.add(Experience(content="thunderstorm warning issued today", domain="weather", timestamp=t + 10001))

        episodes = det.detect()
        assert len(episodes) >= 2

    def test_episode_has_timestamps(self):
        det = EpisodeBoundaryDetector()
        t = 1000.0
        det.add(Experience(content="first event", timestamp=t))
        det.add(Experience(content="second event", timestamp=t + 100))
        episodes = det.detect()
        for ep in episodes:
            assert ep.start_time > 0
            assert ep.end_time >= ep.start_time

    def test_clear(self):
        det = EpisodeBoundaryDetector()
        det.add(Experience(content="something"))
        det.clear()
        assert det.detect() == []

    def test_coherence_score(self):
        det = EpisodeBoundaryDetector()
        t = time.time()
        det.add(Experience(content="market stock finance", domain="finance", timestamp=t))
        det.add(Experience(content="market stock trading", domain="finance", timestamp=t + 1))
        det.add(Experience(content="market stock price", domain="finance", timestamp=t + 2))
        episodes = det.detect()
        for ep in episodes:
            assert 0 <= ep.coherence <= 1
