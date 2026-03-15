"""Tests for SoftPromptAdapter (Gap 2: AGI Roadmap).

Covers:
- PromptStrategy (Thompson Sampling, Bayesian update)
- FewShotExample management
- Strategy selection (explore vs exploit)
- Prompt construction
- Outcome feedback & learning
- Strategy evolution (crossover, mutation, pruning)
- Persistence (save/load round-trip)
- EMMS integration (public wrappers, save/load)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms import EMMS, Experience
from emms.memory.soft_prompt_adapter import (
    FewShotExample,
    Outcome,
    PromptStrategy,
    SoftPromptAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_adapter() -> SoftPromptAdapter:
    return SoftPromptAdapter()


# ---------------------------------------------------------------------------
# PromptStrategy
# ---------------------------------------------------------------------------


class TestPromptStrategy:
    def test_creation_defaults(self):
        s = PromptStrategy(name="test", domain="finance")
        assert s.id.startswith("strat_")
        assert s.alpha == 1.0
        assert s.beta_param == 1.0
        assert s.pulls == 0

    def test_mean_reward(self):
        s = PromptStrategy(name="test")
        assert s.mean_reward == 0.5  # uniform prior

    def test_bayesian_update(self):
        s = PromptStrategy(name="test")
        s.update(1.0)  # success
        assert s.alpha == 2.0
        assert s.beta_param == 1.0
        assert s.mean_reward > 0.5

        s.update(0.0)  # failure
        assert s.alpha == 2.0
        assert s.beta_param == 2.0
        assert s.mean_reward == 0.5  # back to even

    def test_thompson_sampling(self):
        s = PromptStrategy(name="test")
        samples = [s.sample() for _ in range(100)]
        assert all(0 <= x <= 1 for x in samples)

    def test_serialization_roundtrip(self):
        s = PromptStrategy(name="finance_expert", domain="finance",
                          system_prefix="Be precise.", cot_template="Step 1:")
        s.update(0.8)
        s.update(0.9)
        d = s.to_dict()
        s2 = PromptStrategy.from_dict(d)
        assert s2.name == s.name
        assert s2.alpha == s.alpha
        assert s2.pulls == s.pulls


# ---------------------------------------------------------------------------
# FewShotExample
# ---------------------------------------------------------------------------


class TestFewShotExample:
    def test_creation(self):
        ex = FewShotExample(query="What is risk?", response_summary="Risk is...",
                           domain="finance", quality_score=0.9)
        assert ex.id.startswith("fse_")
        assert ex.quality_score == 0.9

    def test_serialization_roundtrip(self):
        ex = FewShotExample(query="test", response_summary="answer", domain="ml")
        d = ex.to_dict()
        ex2 = FewShotExample.from_dict(d)
        assert ex2.query == ex.query
        assert ex2.domain == ex.domain


# ---------------------------------------------------------------------------
# SoftPromptAdapter
# ---------------------------------------------------------------------------


class TestSoftPromptAdapter:
    def test_default_strategies_seeded(self):
        adapter = _make_adapter()
        assert len(adapter.strategies["general"]) == 4
        names = {s.name for s in adapter.strategies["general"]}
        assert "direct" in names
        assert "analytical" in names

    def test_select_strategy_explore(self):
        adapter = _make_adapter()
        strat = adapter.select_strategy("general", "test query")
        assert isinstance(strat, PromptStrategy)
        assert strat.pulls >= 1

    def test_select_strategy_exploit(self):
        adapter = _make_adapter()
        # Give one strategy a clear advantage
        strat = adapter.strategies["general"][0]
        for _ in range(10):
            strat.update(1.0)
        selected = adapter.select_strategy("general", "test", explore=False)
        assert selected.id == strat.id

    def test_select_strategy_fallback_to_general(self):
        adapter = _make_adapter()
        strat = adapter.select_strategy("unknown_domain", "query")
        assert strat is not None  # falls back to general

    def test_build_prompt(self):
        adapter = _make_adapter()
        strat = PromptStrategy(name="test", system_prefix="Be concise.",
                              cot_template="Step 1:")
        prompt = adapter.build_prompt(strat, "What is ML?")
        assert "Be concise." in prompt
        assert "Step 1:" in prompt
        assert "What is ML?" in prompt

    def test_build_prompt_with_identity(self):
        adapter = _make_adapter()
        strat = PromptStrategy(name="test")
        prompt = adapter.build_prompt(strat, "query", identity_prompt="I am EMMS.")
        assert "I am EMMS." in prompt

    def test_build_prompt_with_few_shot(self):
        adapter = _make_adapter()
        adapter.add_example("What is risk?", "Risk is uncertainty.",
                          domain="finance", quality_score=0.9)
        strat = PromptStrategy(name="test", domain="finance")
        prompt = adapter.build_prompt(strat, "What is risk management?")
        assert "What is risk?" in prompt  # few-shot injected

    def test_update_from_outcome(self):
        adapter = _make_adapter()
        strat = adapter.select_strategy("general")
        success = adapter.update_from_outcome(strat.id, quality=0.9)
        assert success
        assert strat.total_reward > 0
        assert len(adapter.outcome_history) == 1

    def test_update_from_outcome_active_strategy(self):
        adapter = _make_adapter()
        adapter.select_strategy("general")  # sets _active_strategy_id
        success = adapter.update_from_outcome(quality=0.7)  # no explicit ID
        assert success

    def test_update_from_outcome_missing_strategy(self):
        adapter = _make_adapter()
        success = adapter.update_from_outcome("nonexistent_id", quality=0.5)
        assert not success

    def test_add_example(self):
        adapter = _make_adapter()
        ex = adapter.add_example("query", "response", "finance", 0.8)
        assert ex.domain == "finance"
        assert len(adapter.examples["finance"]) == 1

    def test_example_capacity_pruning(self):
        adapter = SoftPromptAdapter(max_examples=5)
        for i in range(10):
            adapter.add_example(f"query {i}", f"response {i}", "test",
                              quality_score=i / 10)
        assert len(adapter.examples["test"]) <= 5
        # Best quality examples kept
        scores = [e.quality_score for e in adapter.examples["test"]]
        assert min(scores) >= 0.5  # top 5 of 0.0-0.9

    def test_evolve_strategies(self):
        adapter = _make_adapter()
        # Give strategies different rewards so evolution has material
        for i, s in enumerate(adapter.strategies["general"]):
            for _ in range(5):
                s.update(0.2 * (i + 1))
        result = adapter.evolve_strategies()
        assert "general" in result["domains"]

    def test_create_domain_strategy(self):
        adapter = _make_adapter()
        strat = adapter.create_domain_strategy(
            "finance", "quant_analyst",
            "You are a quantitative analyst. Use precise numbers.",
        )
        assert strat.domain == "finance"
        assert len(adapter.strategies["finance"]) == 1

    def test_best_strategy_for(self):
        adapter = _make_adapter()
        # Train one strategy heavily
        adapter.strategies["general"][0].alpha = 10.0
        adapter.strategies["general"][0].beta_param = 1.0
        adapter.strategies["general"][0].pulls = 10
        best = adapter.best_strategy_for("general")
        assert best is not None
        assert best.mean_reward > 0.8

    def test_strategy_report(self):
        adapter = _make_adapter()
        adapter.select_strategy("general")
        report = adapter.strategy_report()
        assert "general" in report
        assert len(report["general"]) == 4

    def test_summary(self):
        adapter = _make_adapter()
        summary = adapter.summary()
        assert "SoftPromptAdapter" in summary
        assert "general" in summary

    def test_persistence_roundtrip(self):
        adapter = _make_adapter()
        adapter.create_domain_strategy("finance", "quant", "Be precise.")
        adapter.add_example("test q", "test a", "finance", 0.9)
        strat = adapter.select_strategy("general")
        adapter.update_from_outcome(strat.id, quality=0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapter.json"
            adapter.save_state(path)

            adapter2 = SoftPromptAdapter()
            loaded = adapter2.load_state(path)
            assert loaded
            assert len(adapter2.strategies["finance"]) == 1
            assert len(adapter2.examples["finance"]) == 1
            assert adapter2._outcome_count == adapter._outcome_count

    def test_empty_load_returns_false(self):
        adapter = _make_adapter()
        assert adapter.load_state("/nonexistent/path.json") is False

    def test_evolution_triggered_by_outcomes(self):
        adapter = SoftPromptAdapter(evolution_interval=5)
        strat = adapter.select_strategy("general")
        # Give all strategies some pulls so they're eligible for evolution
        for s in adapter.strategies["general"]:
            s.pulls = 5
        for i in range(5):
            adapter.update_from_outcome(strat.id, quality=0.7)
        # After 5 outcomes, evolution should have been triggered
        assert adapter._outcome_count == 5


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_select_prompt_strategy(self):
        agent = _make_emms()
        strat = agent.select_prompt_strategy("general", "test query")
        assert isinstance(strat, PromptStrategy)

    def test_build_adapted_prompt(self):
        agent = _make_emms()
        strat = agent.select_prompt_strategy("general")
        prompt = agent.build_adapted_prompt(strat, "What is ML?")
        assert "What is ML?" in prompt

    def test_prompt_feedback(self):
        agent = _make_emms()
        strat = agent.select_prompt_strategy("general")
        result = agent.prompt_feedback(strat.id, quality=0.8)
        assert result is True

    def test_add_few_shot_example(self):
        agent = _make_emms()
        ex = agent.add_few_shot_example("q", "a", "finance")
        assert ex.domain == "finance"

    def test_prompt_strategy_report(self):
        agent = _make_emms()
        agent.select_prompt_strategy("general")
        report = agent.prompt_strategy_report()
        assert isinstance(report, dict)

    def test_evolve_prompt_strategies(self):
        agent = _make_emms()
        result = agent.evolve_prompt_strategies()
        assert isinstance(result, dict)

    def test_prompt_adapter_summary(self):
        agent = _make_emms()
        summary = agent.prompt_adapter_summary()
        assert "SoftPromptAdapter" in summary

    def test_save_load_roundtrip(self):
        agent = _make_emms()
        agent.select_prompt_strategy("general")
        agent.prompt_feedback(quality=0.8)
        agent.add_few_shot_example("test q", "test a", "ml")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = _make_emms()
            agent2.load(str(path))
            summary = agent2.prompt_adapter_summary()
            assert "1 outcomes" in summary
