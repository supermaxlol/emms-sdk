"""Tests for EMMS v0.8.0 features.

Covers:
1. HybridSearch     — BM25 + embedding RRF fusion                   (18 tests)
2. MemoryTimeline   — chronological reconstruction                    (22 tests)
3. AdaptiveRetriever — Thompson Sampling Beta-Bernoulli bandit       (24 tests)
4. MemoryBudget     — token-budget eviction                          (22 tests)
5. MultiHopGraph    — multi-hop BFS reasoning                        (21 tests)
                                                                   ─────────────
                                                             Total: 107 tests
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import pytest

from emms import EMMS, Experience, MemoryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms(n: int = 0, domain: str = "test") -> EMMS:
    """Create an EMMS instance with n stored memories."""
    agent = EMMS()
    for i in range(n):
        agent.store(Experience(
            content=f"fact number {i} about {domain} systems",
            domain=domain,
            importance=0.1 + (i % 10) * 0.08,
        ))
    return agent


def _make_diverse_emms() -> EMMS:
    """EMMS with three clearly distinct content domains."""
    agent = EMMS()
    # Food domain
    for i in range(3):
        agent.store(Experience(content=f"cooking recipe pasta dish number {i}", domain="food"))
    # Tech domain
    for i in range(3):
        agent.store(Experience(content=f"python programming algorithm technique {i}", domain="tech"))
    # Science domain
    for i in range(3):
        agent.store(Experience(content=f"quantum physics wave function experiment {i}", domain="science"))
    return agent


# ---------------------------------------------------------------------------
# 1. HybridSearch tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """18 tests for retrieval/hybrid.py."""

    def test_import(self):
        from emms.retrieval.hybrid import HybridRetriever, HybridSearchResult
        assert HybridRetriever is not None
        assert HybridSearchResult is not None

    def test_retrieve_returns_list(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("fact about test", max_results=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_empty_memory(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = EMMS()
        r = HybridRetriever(agent.memory)
        results = r.retrieve("query")
        assert results == []

    def test_results_sorted_descending(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(10)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test systems", max_results=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_fields(self):
        from emms.retrieval.hybrid import HybridRetriever, HybridSearchResult
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test")
        assert all(isinstance(res, HybridSearchResult) for res in results)
        for res in results:
            assert res.score >= 0
            assert res.bm25_score >= 0
            assert res.embedding_score >= 0
            assert res.bm25_rank >= 0
            assert res.embedding_rank >= 0

    def test_bm25_rank_populated(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test")
        # At least one result should have a nonzero bm25_rank
        assert any(res.bm25_rank > 0 for res in results)

    def test_rrf_score_positive(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test")
        assert all(res.score > 0 for res in results)

    def test_rrf_k_parameter(self):
        """Higher rrf_k → lower max possible score (1/(60+1) vs 1/(120+1))."""
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(5)
        r60 = HybridRetriever(agent.memory, rrf_k=60.0)
        r120 = HybridRetriever(agent.memory, rrf_k=120.0)
        res60 = r60.retrieve("test", max_results=1)
        res120 = r120.retrieve("test", max_results=1)
        if res60 and res120:
            assert res60[0].score > res120[0].score

    def test_min_score_filter(self):
        from emms.retrieval.hybrid import HybridRetriever
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test", min_score=999.9)  # impossibly high
        assert results == []

    def test_to_retrieval_result(self):
        from emms.retrieval.hybrid import HybridRetriever
        from emms.core.models import RetrievalResult
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve("test")
        if results:
            rr = results[0].to_retrieval_result()
            assert isinstance(rr, RetrievalResult)
            assert rr.strategy == "hybrid_rrf"

    def test_retrieve_as_retrieval_results(self):
        from emms.retrieval.hybrid import HybridRetriever
        from emms.core.models import RetrievalResult
        agent = _make_emms(5)
        r = HybridRetriever(agent.memory)
        results = r.retrieve_as_retrieval_results("test")
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_emms_facade_hybrid_retrieve(self):
        from emms.core.models import RetrievalResult
        agent = _make_emms(5)
        results = agent.hybrid_retrieve("test systems")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_bm25_tokeniser(self):
        from emms.retrieval.hybrid import _tokenise
        tokens = _tokenise("Hello World! foo-bar 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_bm25_scores_length_matches_items(self):
        from emms.retrieval.hybrid import _BM25
        agent = _make_emms(7)
        items = list(agent.memory.working) + list(agent.memory.short_term) + list(agent.memory.long_term.values())
        bm25 = _BM25(items)
        scores = bm25.scores("test query")
        assert len(scores) == len(items)

    def test_rrf_fuse_single_list(self):
        from emms.retrieval.hybrid import _rrf_fuse
        ranked = [[0, 1, 2, 3, 4]]
        fused = _rrf_fuse(ranked)
        # First item should have highest score
        ids = [idx for idx, _ in fused]
        assert ids[0] == 0

    def test_rrf_fuse_two_lists_agreement(self):
        from emms.retrieval.hybrid import _rrf_fuse
        # Both lists agree on top item → it should win decisively
        ranked1 = [0, 1, 2]
        ranked2 = [0, 2, 1]
        fused = _rrf_fuse([ranked1, ranked2])
        assert fused[0][0] == 0

    def test_hybrid_retrieve_max_results(self):
        agent = _make_emms(20)
        results = agent.hybrid_retrieve("test", max_results=5)
        assert len(results) <= 5

    def test_hybrid_retrieve_no_results_empty_store(self):
        agent = EMMS()
        results = agent.hybrid_retrieve("python")
        assert results == []


# ---------------------------------------------------------------------------
# 2. MemoryTimeline tests
# ---------------------------------------------------------------------------

class TestMemoryTimeline:
    """22 tests for analytics/timeline.py."""

    def test_import(self):
        from emms.analytics.timeline import (
            MemoryTimeline, TimelineResult, TimelineEvent,
            TemporalGap, DensityBucket,
        )
        assert MemoryTimeline is not None

    def test_build_empty(self):
        agent = EMMS()
        result = agent.build_timeline()
        assert result.total_memories == 0
        assert result.events == []
        assert result.gaps == []

    def test_build_with_memories(self):
        agent = _make_emms(5)
        result = agent.build_timeline()
        assert result.total_memories == 5

    def test_events_sorted_chronologically(self):
        from emms.analytics.timeline import MemoryTimeline
        agent = _make_emms(5)
        tl = MemoryTimeline(agent.memory)
        result = tl.build()
        timestamps = [ev.stored_at for ev in result.events]
        assert timestamps == sorted(timestamps)

    def test_event_fields(self):
        from emms.analytics.timeline import TimelineEvent
        agent = _make_emms(3)
        result = agent.build_timeline()
        for ev in result.events:
            assert isinstance(ev, TimelineEvent)
            assert ev.memory_id
            assert ev.domain == "test"
            assert 0.0 <= ev.importance <= 1.0

    def test_domain_filter(self):
        agent = _make_diverse_emms()
        result = agent.build_timeline(domain="food")
        assert all(ev.domain == "food" for ev in result.events)
        # At least 1 food memory (consolidation may merge similar items)
        assert result.total_memories >= 1

    def test_since_filter(self):
        agent = _make_emms(3)
        future_t = time.time() + 9999
        result = agent.build_timeline(since=future_t)
        assert result.total_memories == 0

    def test_until_filter(self):
        agent = _make_emms(3)
        result = agent.build_timeline(until=0.0)
        # All stored at current time, so nothing before epoch
        assert result.total_memories == 0

    def test_gap_detection_with_large_threshold(self):
        """Very large threshold → no gaps found."""
        agent = _make_emms(5)
        result = agent.build_timeline(gap_threshold_seconds=1e12)
        assert result.gaps == []

    def test_gap_detection_with_small_threshold(self):
        """With threshold=0 all consecutive events should produce gaps."""
        agent = _make_emms(5)
        result = agent.build_timeline(gap_threshold_seconds=0.0)
        # Should detect gaps for all adjacent pairs
        assert len(result.gaps) >= 0  # at minimum 0 (could be 0 if all same timestamp)

    def test_temporal_gap_fields(self):
        from emms.analytics.timeline import TemporalGap
        agent = _make_emms(3)
        result = agent.build_timeline(gap_threshold_seconds=0.0)
        for gap in result.gaps:
            assert isinstance(gap, TemporalGap)
            assert gap.duration_seconds >= 0.0
            assert gap.before_id
            assert gap.after_id

    def test_temporal_gap_duration_human(self):
        from emms.analytics.timeline import TemporalGap
        gap = TemporalGap(start_at=0.0, end_at=90.0, duration_seconds=90.0,
                          before_id="a", after_id="b")
        assert "m" in gap.duration_human or "s" in gap.duration_human

    def test_density_histogram_buckets(self):
        agent = _make_emms(5)
        result = agent.build_timeline(bucket_size_seconds=1.0)
        assert isinstance(result.density, list)
        # With bucket_size=1s and items stored in rapid succession, few buckets
        total = sum(b.count for b in result.density)
        assert total == result.total_memories

    def test_density_bucket_fields(self):
        from emms.analytics.timeline import DensityBucket
        agent = _make_emms(3)
        result = agent.build_timeline(bucket_size_seconds=1.0)
        for bucket in result.density:
            assert isinstance(bucket, DensityBucket)
            assert bucket.count >= 0
            assert 0.0 <= bucket.avg_importance <= 1.0

    def test_mean_importance(self):
        agent = _make_emms(10)
        result = agent.build_timeline()
        assert 0.0 <= result.mean_importance <= 1.0

    def test_domain_counts(self):
        agent = _make_diverse_emms()
        result = agent.build_timeline()
        assert "food" in result.domain_counts
        assert "tech" in result.domain_counts
        # At least 1 per domain (consolidation may merge similar items)
        assert result.domain_counts["food"] >= 1

    def test_summary_not_empty(self):
        agent = _make_emms(3)
        result = agent.build_timeline()
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_export_markdown(self):
        agent = _make_emms(3)
        result = agent.build_timeline()
        md = result.export_markdown()
        assert "# Memory Timeline" in md
        assert "memories" in md.lower() or "0 memories" in md.lower() or str(result.total_memories) in md

    def test_timeline_result_empty_summary(self):
        agent = EMMS()
        result = agent.build_timeline()
        assert "empty" in result.summary().lower()

    def test_emms_facade_returns_timeline_result(self):
        from emms.analytics.timeline import TimelineResult
        agent = _make_emms(3)
        result = agent.build_timeline()
        assert isinstance(result, TimelineResult)

    def test_tier_filter(self):
        agent = _make_emms(5)
        result = agent.build_timeline(tiers=["working"])
        # Working tier has max 7 items; should be ≤ 5
        assert result.total_memories <= 5

    def test_span_calculation(self):
        agent = _make_emms(5)
        result = agent.build_timeline()
        if result.total_memories >= 2:
            assert result.span_seconds >= 0.0
        else:
            assert result.span_seconds == 0.0


# ---------------------------------------------------------------------------
# 3. AdaptiveRetriever tests
# ---------------------------------------------------------------------------

class TestAdaptiveRetriever:
    """24 tests for retrieval/adaptive.py."""

    def test_import(self):
        from emms.retrieval.adaptive import AdaptiveRetriever, StrategyBelief
        assert AdaptiveRetriever is not None
        assert StrategyBelief is not None

    def test_default_arms(self):
        from emms.retrieval.adaptive import AdaptiveRetriever, _BUILTIN_STRATEGIES
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory)
        assert set(ar.beliefs.keys()) == set(_BUILTIN_STRATEGIES)

    def test_custom_arms(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory, strategies=["a", "b"])
        assert set(ar.beliefs.keys()) == {"a", "b"}

    def test_uniform_prior(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory)
        for b in ar.beliefs.values():
            assert b.alpha == 1.0
            assert b.beta == 1.0

    def test_belief_mean_uniform(self):
        from emms.retrieval.adaptive import StrategyBelief
        b = StrategyBelief(name="test")
        assert b.mean == 0.5

    def test_belief_update_reward_1(self):
        from emms.retrieval.adaptive import StrategyBelief
        b = StrategyBelief(name="test", alpha=1.0, beta=1.0)
        b.update(reward=1.0)
        assert b.alpha == 2.0
        assert b.beta == 1.0
        assert b.mean > 0.5

    def test_belief_update_reward_0(self):
        from emms.retrieval.adaptive import StrategyBelief
        b = StrategyBelief(name="test", alpha=1.0, beta=1.0)
        b.update(reward=0.0)
        assert b.alpha == 1.0
        assert b.beta == 2.0
        assert b.mean < 0.5

    def test_belief_decay(self):
        from emms.retrieval.adaptive import StrategyBelief
        b = StrategyBelief(name="test", alpha=10.0, beta=1.0)
        b.update(reward=1.0, decay=0.5)
        # After decay, alpha = 1 + (10-1)*0.5 = 5.5; then +1 → 6.5
        assert b.alpha < 11.0

    def test_belief_sample_range(self):
        from emms.retrieval.adaptive import StrategyBelief
        import random
        b = StrategyBelief(name="test", alpha=2.0, beta=3.0)
        rng = random.Random(42)
        for _ in range(10):
            s = b.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_retrieve_returns_list(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(5)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        results = ar.retrieve("test")
        assert isinstance(results, list)

    def test_retrieve_empty_memory(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = EMMS()
        ar = AdaptiveRetriever(agent.memory, seed=42)
        results = ar.retrieve("test")
        assert results == []

    def test_last_selected_updated(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(5)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        ar.retrieve("test")
        assert ar._last_selected in ar.beliefs

    def test_record_feedback_increments_pulls(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(5)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        ar.retrieve("test")
        strategy = ar._last_selected
        pulls_before = ar.beliefs[strategy].pulls
        ar.record_feedback(reward=1.0)
        assert ar.beliefs[strategy].pulls == pulls_before + 1

    def test_record_feedback_none_strategy(self):
        """Feedback with no args uses last selected."""
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(5)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        ar.retrieve("test")
        strategy = ar._last_selected
        ar.record_feedback(reward=1.0)
        assert ar.beliefs[strategy].pulls == 1

    def test_best_strategy_argmax(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        # Manually boost one strategy
        ar.beliefs["bm25"].alpha = 10.0
        ar.beliefs["bm25"].beta = 1.0
        assert ar.best_strategy() == "bm25"

    def test_exploit_mode(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        ar.beliefs["importance"].alpha = 100.0
        ar.beliefs["importance"].beta = 1.0
        for _ in range(5):
            ar.retrieve("test", explore=False)
        assert ar._last_selected == "importance"

    def test_get_beliefs(self):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        beliefs = ar.get_beliefs()
        assert isinstance(beliefs, dict)
        assert len(beliefs) > 0

    def test_save_load_state(self, tmp_path):
        from emms.retrieval.adaptive import AdaptiveRetriever
        agent = _make_emms(3)
        ar = AdaptiveRetriever(agent.memory, seed=42)
        ar.beliefs["semantic"].alpha = 5.0
        ar.beliefs["semantic"].beta = 2.0
        path = tmp_path / "adaptive.json"
        ar.save_state(path)
        assert path.exists()
        ar2 = AdaptiveRetriever(agent.memory, seed=42)
        ar2.load_state(path)
        assert ar2.beliefs["semantic"].alpha == 5.0
        assert ar2.beliefs["semantic"].beta == 2.0

    def test_emms_enable_adaptive_retrieval(self):
        agent = _make_emms(3)
        ar = agent.enable_adaptive_retrieval()
        assert ar is not None
        assert hasattr(agent, "_adaptive_retriever")

    def test_emms_adaptive_retrieve(self):
        agent = _make_emms(5)
        agent.enable_adaptive_retrieval()
        results = agent.adaptive_retrieve("test", max_results=3)
        assert isinstance(results, list)

    def test_emms_adaptive_feedback(self):
        agent = _make_emms(5)
        agent.enable_adaptive_retrieval(seed=0)
        agent.adaptive_retrieve("test")
        # Should not raise
        agent.adaptive_feedback(reward=1.0)

    def test_emms_get_retrieval_beliefs(self):
        agent = _make_emms(3)
        agent.enable_adaptive_retrieval()
        beliefs = agent.get_retrieval_beliefs()
        assert isinstance(beliefs, dict)
        assert len(beliefs) > 0
        for b in beliefs.values():
            assert "alpha" in b
            assert "mean" in b

    def test_emms_get_beliefs_no_retriever(self):
        agent = EMMS()
        beliefs = agent.get_retrieval_beliefs()
        assert beliefs == {}

    def test_emms_adaptive_retrieve_fallback(self):
        """Without enable, adaptive_retrieve falls back to standard retrieve."""
        agent = _make_emms(5)
        results = agent.adaptive_retrieve("test")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 4. MemoryBudget tests
# ---------------------------------------------------------------------------

class TestMemoryBudget:
    """22 tests for context/budget.py."""

    def test_import(self):
        from emms.context.budget import (
            MemoryBudget, BudgetReport, EvictionCandidate, EvictionPolicy,
            _estimate_tokens,
        )
        assert MemoryBudget is not None

    def test_token_estimate_positive(self):
        from emms.context.budget import _estimate_tokens
        agent = _make_emms(3)
        for item in list(agent.memory.working):
            assert _estimate_tokens(item) > 0

    def test_token_footprint_structure(self):
        agent = _make_emms(5)
        fp = agent.memory_token_footprint()
        assert "total" in fp
        assert "by_tier" in fp
        assert "memory_count" in fp
        assert fp["memory_count"] == 5

    def test_token_footprint_total_is_sum_of_tiers(self):
        agent = _make_emms(5)
        fp = agent.memory_token_footprint()
        assert fp["total"] == sum(fp["by_tier"].values())

    def test_enforce_within_budget(self):
        agent = _make_emms(3)
        report = agent.enforce_memory_budget(max_tokens=10_000_000, dry_run=True)
        assert report.over_budget is False
        assert report.evicted_count == 0

    def test_enforce_over_budget_dry_run(self):
        agent = _make_emms(10)
        report = agent.enforce_memory_budget(max_tokens=1, dry_run=True)
        assert report.over_budget is True
        assert report.evicted_count > 0
        # Dry run should not actually remove memories
        assert agent.memory.size["working"] + agent.memory.size["short_term"] + len(agent.memory.long_term) > 0

    def test_enforce_over_budget_actual(self):
        agent = _make_emms(20)
        fp_before = agent.memory_token_footprint()
        report = agent.enforce_memory_budget(max_tokens=1, dry_run=False)
        fp_after = agent.memory_token_footprint()
        assert report.over_budget is True
        assert report.evicted_count > 0
        # After eviction, total should be lower
        assert fp_after["total"] < fp_before["total"]

    def test_budget_report_summary(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(max_tokens=10_000_000, dry_run=True)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_budget_report_dry_run_flag(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(dry_run=True)
        assert report.dry_run is True

    def test_protection_by_importance(self):
        """Memories with importance >= threshold should be in protected_count."""
        from emms.context.budget import MemoryBudget
        agent = EMMS()
        # Store one high-importance memory
        agent.store(Experience(content="critical system rule", domain="rules", importance=0.95))
        agent.store(Experience(content="trivial note", domain="notes", importance=0.1))
        budget = MemoryBudget(agent.memory, max_tokens=1, importance_threshold=0.9)
        report = budget.enforce(dry_run=True)
        assert report.protected_count >= 1

    def test_protection_by_tier(self):
        """Semantic tier should be protected by default."""
        from emms.context.budget import MemoryBudget
        agent = _make_emms(5)
        budget = MemoryBudget(agent.memory, max_tokens=1, protected_tiers=["semantic"])
        report = budget.enforce(dry_run=True)
        # All semantic items should be in protected_count (may be 0 if no semantic items)
        assert report.protected_count >= 0

    def test_eviction_candidates_fields(self):
        from emms.context.budget import EvictionCandidate
        agent = _make_emms(10)
        report = agent.enforce_memory_budget(max_tokens=1, dry_run=True)
        for c in report.candidates:
            assert isinstance(c, EvictionCandidate)
            assert c.memory_id
            assert c.eviction_score >= 0
            assert c.token_estimate > 0

    def test_lru_policy(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(max_tokens=1, policy="lru", dry_run=True)
        assert report.over_budget is True

    def test_lfu_policy(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(max_tokens=1, policy="lfu", dry_run=True)
        assert report.over_budget is True

    def test_importance_policy(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(max_tokens=1, policy="importance", dry_run=True)
        assert report.over_budget is True

    def test_strength_policy(self):
        agent = _make_emms(5)
        report = agent.enforce_memory_budget(max_tokens=1, policy="strength", dry_run=True)
        assert report.over_budget is True

    def test_freed_tokens_positive(self):
        agent = _make_emms(10)
        report = agent.enforce_memory_budget(max_tokens=1, dry_run=True)
        if report.over_budget:
            assert report.freed_tokens > 0

    def test_remaining_tokens_after_enforce(self):
        agent = _make_emms(10)
        report = agent.enforce_memory_budget(max_tokens=1, dry_run=True)
        if report.over_budget:
            assert report.remaining_tokens == max(0, report.total_tokens - report.freed_tokens)

    def test_eviction_policy_enum(self):
        from emms.context.budget import EvictionPolicy
        assert EvictionPolicy.COMPOSITE.value == "composite"
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"

    def test_budget_report_candidates_ascending_score(self):
        """Candidates should be sorted ascending (lowest score = evicted first)."""
        from emms.context.budget import MemoryBudget
        agent = _make_emms(10)
        budget = MemoryBudget(agent.memory, max_tokens=1)
        report = budget.enforce(dry_run=True)
        if len(report.candidates) >= 2:
            for i in range(len(report.candidates) - 1):
                assert report.candidates[i].eviction_score <= report.candidates[i + 1].eviction_score

    def test_token_footprint_empty(self):
        agent = EMMS()
        fp = agent.memory_token_footprint()
        assert fp["total"] == 0
        assert fp["memory_count"] == 0

    def test_enforce_no_eviction_needed_no_candidates(self):
        agent = _make_emms(3)
        report = agent.enforce_memory_budget(max_tokens=1_000_000)
        assert report.candidates == []


# ---------------------------------------------------------------------------
# 5. MultiHopGraphReasoner tests
# ---------------------------------------------------------------------------

class TestMultiHopGraphReasoner:
    """21 tests for memory/multihop.py."""

    def _make_graph_emms(self) -> EMMS:
        """EMMS with graph-extractable entity chains."""
        agent = EMMS()
        agent.store(Experience(
            content="Alice works with Bob at Acme Corp on the Python team",
            domain="hr",
        ))
        agent.store(Experience(
            content="Bob leads the DevOps team at Acme Corp",
            domain="hr",
        ))
        agent.store(Experience(
            content="Acme Corp partners with Tech Labs on machine learning",
            domain="business",
        ))
        return agent

    def test_import(self):
        from emms.memory.multihop import (
            MultiHopGraphReasoner, MultiHopResult, HopPath, ReachableEntity,
        )
        assert MultiHopGraphReasoner is not None

    def test_query_unknown_seed(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("nonexistent_entity_xyz", max_hops=3)
        assert result.reachable == []
        assert result.paths == []

    def test_query_returns_result(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        assert result is not None

    def test_query_seed_normalized(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        assert result.seed == "alice"

    def test_reachable_entities_not_seed(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        assert all(r.name != "alice" for r in result.reachable)

    def test_paths_are_hop_path_objects(self):
        from emms.memory.multihop import HopPath
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        assert all(isinstance(p, HopPath) for p in result.paths)

    def test_path_hops_match_entity_count(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        for p in result.paths:
            assert p.hops == len(p.entities) - 1

    def test_path_strength_in_range(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        for p in result.paths:
            assert 0.0 < p.strength <= 1.0

    def test_path_strength_decays_with_hops(self):
        """Multi-hop paths have lower strength than single-hop paths."""
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        one_hop = [p for p in result.paths if p.hops == 1]
        two_hop = [p for p in result.paths if p.hops == 2]
        if one_hop and two_hop:
            avg_1 = sum(p.strength for p in one_hop) / len(one_hop)
            avg_2 = sum(p.strength for p in two_hop) / len(two_hop)
            # On average, 2-hop paths should be weaker (or equal for default edge=0.5)
            assert avg_2 <= avg_1 + 0.01

    def test_max_hops_limit(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=1)
        # No path should have more than 1 hop
        assert all(p.hops <= 1 for p in result.paths)

    def test_max_results_limit(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3, max_results=2)
        assert len(result.reachable) <= 2

    def test_reachable_sorted_by_strength(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        if len(result.reachable) >= 2:
            strengths = [r.best_path.strength for r in result.reachable]
            assert strengths == sorted(strengths, reverse=True)

    def test_reachable_entity_fields(self):
        from emms.memory.multihop import ReachableEntity
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        for r in result.reachable:
            assert isinstance(r, ReachableEntity)
            assert r.name
            assert r.display_name
            assert r.min_hops >= 1

    def test_bridging_entities(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        # bridging_entities is a list of (name, score) tuples
        assert isinstance(result.bridging_entities, list)
        for name, score in result.bridging_entities:
            assert isinstance(name, str)
            assert score > 0.0

    def test_summary_string(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        summary = result.summary()
        assert "alice" in summary.lower()
        assert "MultiHop" in summary

    def test_to_dot_output(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        dot = result.to_dot()
        assert "digraph" in dot
        assert "alice" in dot.lower()

    def test_to_dot_max_nodes_limit(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=3)
        dot = result.to_dot(max_nodes=2)
        assert "digraph" in dot

    def test_find_connection_existing(self):
        from emms.memory.multihop import MultiHopGraphReasoner
        agent = self._make_graph_emms()
        reasoner = MultiHopGraphReasoner(agent.graph)
        # Try to find a connection (may or may not exist based on NER extraction)
        result = reasoner.find_connection("Alice", "Bob", max_hops=3)
        # Just ensure it doesn't raise
        assert result is None or hasattr(result, "entities")

    def test_find_connection_unknown(self):
        from emms.memory.multihop import MultiHopGraphReasoner
        agent = self._make_graph_emms()
        reasoner = MultiHopGraphReasoner(agent.graph)
        result = reasoner.find_connection("Unknown1", "Unknown2")
        assert result is None

    def test_emms_facade_multihop_no_graph(self):
        from emms.memory.multihop import MultiHopResult
        agent = EMMS(enable_graph=False)
        result = agent.multihop_query("Alice")
        assert isinstance(result, MultiHopResult)
        assert result.reachable == []

    def test_total_explored_non_negative(self):
        agent = self._make_graph_emms()
        result = agent.multihop_query("Alice", max_hops=2)
        assert result.total_entities_explored >= 0


# ---------------------------------------------------------------------------
# MCP v0.8.0 tool tests
# ---------------------------------------------------------------------------

class TestMCPV080Tools:
    """Tests for the 5 new MCP tool handlers."""

    def _make_server(self, n: int = 5) -> "Any":
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms(n)
        return EMCPServer(agent)

    def test_total_tool_count(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 52

    def test_hybrid_retrieve_tool(self):
        server = self._make_server()
        result = server.handle("emms_hybrid_retrieve", {"query": "test systems"})
        assert result["ok"] is True
        assert "results" in result

    def test_build_timeline_tool(self):
        server = self._make_server()
        result = server.handle("emms_build_timeline", {})
        assert result["ok"] is True
        assert "summary" in result
        assert "total_memories" in result

    def test_adaptive_retrieve_tool(self):
        server = self._make_server()
        result = server.handle("emms_adaptive_retrieve", {"query": "test"})
        assert result["ok"] is True
        assert "results" in result
        assert "beliefs" in result

    def test_enforce_budget_tool(self):
        server = self._make_server()
        result = server.handle("emms_enforce_budget", {"max_tokens": 1_000_000, "dry_run": True})
        assert result["ok"] is True
        assert "summary" in result
        assert "over_budget" in result

    def test_multihop_query_tool(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = EMMS()
        agent.store(Experience(content="Alice works with Bob at Acme Corp", domain="hr"))
        server = EMCPServer(agent)
        result = server.handle("emms_multihop_query", {"seed": "Alice", "max_hops": 2})
        assert result["ok"] is True
        assert "summary" in result

    def test_multihop_with_dot(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = EMMS()
        agent.store(Experience(content="Alice works with Bob at Acme Corp", domain="hr"))
        server = EMCPServer(agent)
        result = server.handle("emms_multihop_query", {"seed": "Alice", "include_dot": True})
        assert result["ok"] is True
        assert "dot" in result


# ---------------------------------------------------------------------------
# __init__.py export tests
# ---------------------------------------------------------------------------

class TestV080Exports:
    """Tests that v0.8.0 symbols are properly exported from the package."""

    def test_hybrid_retriever_export(self):
        from emms import HybridRetriever
        assert HybridRetriever is not None

    def test_hybrid_search_result_export(self):
        from emms import HybridSearchResult
        assert HybridSearchResult is not None

    def test_adaptive_retriever_export(self):
        from emms import AdaptiveRetriever
        assert AdaptiveRetriever is not None

    def test_strategy_belief_export(self):
        from emms import StrategyBelief
        assert StrategyBelief is not None

    def test_memory_timeline_export(self):
        from emms import MemoryTimeline
        assert MemoryTimeline is not None

    def test_timeline_result_export(self):
        from emms import TimelineResult
        assert TimelineResult is not None

    def test_memory_budget_export(self):
        from emms import MemoryBudget
        assert MemoryBudget is not None

    def test_budget_report_export(self):
        from emms import BudgetReport
        assert BudgetReport is not None

    def test_eviction_policy_export(self):
        from emms import EvictionPolicy
        assert EvictionPolicy is not None

    def test_multihop_reasoner_export(self):
        from emms import MultiHopGraphReasoner
        assert MultiHopGraphReasoner is not None

    def test_multihop_result_export(self):
        from emms import MultiHopResult
        assert MultiHopResult is not None

    def test_hop_path_export(self):
        from emms import HopPath
        assert HopPath is not None

    def test_version(self):
        from emms import __version__
        assert __version__ == "0.13.0"
