"""Tests for core data models."""

import time
import pytest
from emms.core.models import (
    Experience,
    MemoryConfig,
    MemoryItem,
    MemoryTier,
    Modality,
    RetrievalResult,
)


class TestExperience:
    def test_defaults(self):
        exp = Experience(content="hello world")
        assert exp.content == "hello world"
        assert exp.domain == "general"
        assert exp.id.startswith("exp_")
        assert exp.novelty == 0.5
        assert exp.importance == 0.5
        assert exp.emotional_valence == 0.0

    def test_custom_fields(self):
        exp = Experience(
            content="market crash",
            domain="finance",
            novelty=0.9,
            importance=0.95,
            emotional_valence=-0.8,
            emotional_intensity=0.9,
        )
        assert exp.domain == "finance"
        assert exp.novelty == 0.9
        assert exp.emotional_valence == -0.8

    def test_modality_features(self):
        exp = Experience(
            content="test",
            modality_features={Modality.TEXT: [0.1, 0.2, 0.3]},
        )
        assert Modality.TEXT in exp.modality_features
        assert len(exp.modality_features[Modality.TEXT]) == 3


class TestMemoryItem:
    def test_creation(self):
        exp = Experience(content="test content")
        item = MemoryItem(experience=exp)
        assert item.tier == MemoryTier.WORKING
        assert item.access_count == 0
        assert item.memory_strength == 1.0

    def test_touch(self):
        exp = Experience(content="test")
        item = MemoryItem(experience=exp)
        old_access = item.last_accessed
        time.sleep(0.01)
        item.touch()
        assert item.access_count == 1
        assert item.last_accessed >= old_access

    def test_decay(self):
        exp = Experience(content="test")
        item = MemoryItem(experience=exp)
        original = item.memory_strength
        # Force some age
        item.last_accessed = time.time() - 86400  # 1 day ago
        new_strength = item.decay(half_life=86400.0)
        assert new_strength < original
        assert new_strength == pytest.approx(0.5, abs=0.05)

    def test_age(self):
        exp = Experience(content="test")
        item = MemoryItem(experience=exp)
        assert item.age >= 0
        assert item.age < 1.0  # should be near-instant


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.working_capacity == 7
        assert cfg.short_term_capacity == 50
        assert cfg.context_window == 32_000

    def test_custom(self):
        cfg = MemoryConfig(working_capacity=5, context_window=8000)
        assert cfg.working_capacity == 5
        assert cfg.context_window == 8000
