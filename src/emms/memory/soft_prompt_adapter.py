"""SoftPromptAdapter — Prompt-level learning without weight updates.

Gap 2 in the EMMS → AGI roadmap: since we can't fine-tune the LLM,
we learn to prompt it better. Each domain maintains a library of
prompt strategies (system prefixes, few-shot examples, CoT templates)
that evolve from outcome feedback via Thompson Sampling.

This is "learning" at the prompt level — the model stays frozen, but
the prompt engineering around it evolves per-domain.

Depends on:
- AdaptiveRetriever's Thompson Sampling (Beta-Bernoulli bandits)
- ProceduralMemory's rule injection
- IdentityPromptBuilder's system prompt templates
- LiveSelfModel's CalibrationTracker (Gap 6)
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Sequence


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class FewShotExample:
    """A successful past interaction for few-shot injection."""

    id: str = ""
    query: str = ""
    response_summary: str = ""
    domain: str = ""
    quality_score: float = 0.0  # how good was this interaction
    created_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = f"fse_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FewShotExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptStrategy:
    """A learned prompt strategy for a specific domain.

    Contains a system prefix, few-shot examples, and a CoT template
    that have been empirically found to produce good results.
    """

    id: str = ""
    name: str = ""
    domain: str = "general"
    system_prefix: str = ""  # injected before user query
    cot_template: str = ""  # chain-of-thought reasoning template
    few_shot_ids: list[str] = field(default_factory=list)  # refs to FewShotExample
    tags: list[str] = field(default_factory=list)

    # Thompson Sampling state (Beta-Bernoulli)
    alpha: float = 1.0  # successes + prior
    beta_param: float = 1.0  # failures + prior (renamed to avoid clash with beta)
    pulls: int = 0  # how many times selected
    total_reward: float = 0.0  # cumulative reward

    created_at: float = 0.0
    last_used: float = 0.0
    last_rewarded: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = f"strat_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    @property
    def mean_reward(self) -> float:
        """Posterior mean: α / (α + β)."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        """Posterior variance."""
        ab = self.alpha + self.beta_param
        return (self.alpha * self.beta_param) / (ab * ab * (ab + 1))

    def sample(self) -> float:
        """Thompson sample from Beta(α, β)."""
        return _beta_sample(self.alpha, self.beta_param)

    def update(self, reward: float):
        """Bayesian update from outcome. Reward ∈ [0, 1]."""
        self.alpha += reward
        self.beta_param += (1.0 - reward)
        self.pulls += 1
        self.total_reward += reward
        self.last_rewarded = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptStrategy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Outcome:
    """Feedback from an interaction where a strategy was used."""

    strategy_id: str = ""
    domain: str = ""
    quality: float = 0.0  # 0 = bad, 1 = excellent
    feedback_type: str = ""  # "explicit", "implicit", "self-assessed"
    notes: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Thompson Sampling Helpers ────────────────────────────────────────────────


def _gamma_sample(alpha: float) -> float:
    """Sample from Gamma(alpha, 1) using Marsaglia-Tsang for alpha >= 1."""
    if alpha < 1.0:
        # Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        return _gamma_sample(alpha + 1.0) * (random.random() ** (1.0 / alpha))
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = random.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v > 0 and math.log(random.random()) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


def _beta_sample(alpha: float, beta: float) -> float:
    """Sample from Beta(alpha, beta) via two Gamma samples."""
    if alpha <= 0:
        alpha = 0.01
    if beta <= 0:
        beta = 0.01
    x = _gamma_sample(alpha)
    y = _gamma_sample(beta)
    return x / (x + y) if (x + y) > 0 else 0.5


# ── Default Strategies ───────────────────────────────────────────────────────


_DEFAULT_STRATEGIES = [
    PromptStrategy(
        name="direct",
        domain="general",
        system_prefix="Answer directly and concisely.",
        cot_template="",
    ),
    PromptStrategy(
        name="analytical",
        domain="general",
        system_prefix="Think step by step. Break the problem into components before answering.",
        cot_template="Let me analyze this step by step:\n1. ",
    ),
    PromptStrategy(
        name="expert",
        domain="general",
        system_prefix="You are a domain expert. Provide deep, technical analysis with evidence.",
        cot_template="As an expert, I'll consider the key factors:\n",
    ),
    PromptStrategy(
        name="cautious",
        domain="general",
        system_prefix="Consider edge cases and potential issues before answering. Flag uncertainty explicitly.",
        cot_template="Before answering, let me consider what could go wrong:\n",
    ),
]


# ── SoftPromptAdapter ────────────────────────────────────────────────────────


class SoftPromptAdapter:
    """Learns optimal prompt strategies per domain from outcome feedback.

    For each domain, maintains a library of prompt strategies (system prefixes,
    few-shot examples, CoT templates). Uses Thompson Sampling to select
    strategies and Bayesian updates to learn from outcomes.

    This is the core of Gap 2: learning without weight updates.

    Usage:
        adapter = SoftPromptAdapter(memory)

        # Before an interaction:
        strategy = adapter.select_strategy("finance", "What's the risk?")
        prompt = adapter.build_prompt(strategy, "What's the risk?")

        # After interaction (with quality feedback):
        adapter.update_from_outcome(strategy.id, quality=0.8)

        # Periodically:
        adapter.evolve_strategies()  # crossover/mutation of top strategies
    """

    def __init__(
        self,
        memory: Any = None,  # HierarchicalMemory
        max_strategies_per_domain: int = 10,
        max_few_shot: int = 5,
        max_examples: int = 50,
        evolution_interval: int = 20,  # evolve every N outcomes
    ):
        self.memory = memory
        self.max_strategies_per_domain = max_strategies_per_domain
        self.max_few_shot = max_few_shot
        self.max_examples = max_examples
        self.evolution_interval = evolution_interval

        # Strategy library: domain → list of strategies
        self.strategies: dict[str, list[PromptStrategy]] = defaultdict(list)

        # Few-shot example bank: domain → list of examples
        self.examples: dict[str, list[FewShotExample]] = defaultdict(list)

        # Outcome history for evolution
        self.outcome_history: list[Outcome] = []
        self._outcome_count: int = 0

        # Currently active strategy (for deferred feedback)
        self._active_strategy_id: str | None = None

        # Seed default strategies
        self._seed_defaults()

    def _seed_defaults(self):
        """Initialize with default strategies if empty."""
        if not self.strategies.get("general"):
            for strat in _DEFAULT_STRATEGIES:
                # Create fresh copies with unique IDs
                s = PromptStrategy(
                    name=strat.name,
                    domain="general",
                    system_prefix=strat.system_prefix,
                    cot_template=strat.cot_template,
                )
                self.strategies["general"].append(s)

    # ── Strategy Selection ───────────────────────────────────────────────

    def select_strategy(
        self, domain: str, query: str = "", explore: bool = True
    ) -> PromptStrategy:
        """Select the best strategy for a domain via Thompson Sampling.

        Args:
            domain: The domain context (e.g., "finance", "ml", "general").
            query: Optional query for context-dependent selection.
            explore: If True, uses Thompson Sampling. If False, exploits best.

        Returns:
            The selected PromptStrategy.
        """
        candidates = self._get_candidates(domain)
        if not candidates:
            # Fall back to general strategies
            candidates = self._get_candidates("general")
        if not candidates:
            # Last resort: create a direct strategy
            s = PromptStrategy(name="fallback", domain=domain)
            self.strategies[domain].append(s)
            return s

        if explore:
            # Thompson Sampling: sample from each arm's Beta posterior
            scored = [(s, s.sample()) for s in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[0][0]
        else:
            # Exploitation: pick highest posterior mean
            selected = max(candidates, key=lambda s: s.mean_reward)

        selected.last_used = time.time()
        selected.pulls += 1
        self._active_strategy_id = selected.id
        return selected

    def _get_candidates(self, domain: str) -> list[PromptStrategy]:
        """Get all strategies for a domain, including inherited general ones."""
        domain_strats = list(self.strategies.get(domain, []))
        if domain != "general":
            # Also consider general strategies
            general = list(self.strategies.get("general", []))
            domain_strats.extend(general)
        return domain_strats

    # ── Prompt Construction ──────────────────────────────────────────────

    def build_prompt(
        self,
        strategy: PromptStrategy,
        query: str,
        include_few_shot: bool = True,
        identity_prompt: str = "",
    ) -> str:
        """Build a complete prompt using the selected strategy.

        Args:
            strategy: The selected PromptStrategy.
            query: The user's query.
            include_few_shot: Whether to include few-shot examples.
            identity_prompt: Optional identity prompt from IdentityPromptBuilder.

        Returns:
            Complete prompt string ready for LLM.
        """
        parts = []

        # 1. Identity prompt (from IdentityPromptBuilder if available)
        if identity_prompt:
            parts.append(identity_prompt)

        # 2. Strategy system prefix
        if strategy.system_prefix:
            parts.append(f"\n## Approach\n{strategy.system_prefix}")

        # 3. Few-shot examples
        if include_few_shot:
            examples = self._select_few_shot(strategy.domain, query)
            if examples:
                parts.append("\n## Relevant Examples")
                for ex in examples:
                    parts.append(f"\nQ: {ex.query}\nA: {ex.response_summary}")

        # 4. CoT template
        if strategy.cot_template:
            parts.append(f"\n## Reasoning\n{strategy.cot_template}")

        # 5. User query
        parts.append(f"\n## Query\n{query}")

        return "\n".join(parts)

    def _select_few_shot(self, domain: str, query: str) -> list[FewShotExample]:
        """Select few-shot examples relevant to the query."""
        candidates = list(self.examples.get(domain, []))
        if not candidates:
            candidates = list(self.examples.get("general", []))
        if not candidates:
            return []

        # Score by token overlap with query + quality
        query_tokens = set(query.lower().split())
        scored = []
        for ex in candidates:
            ex_tokens = set(ex.query.lower().split())
            overlap = len(query_tokens & ex_tokens)
            score = overlap * 0.3 + ex.quality_score * 0.7
            scored.append((ex, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[: self.max_few_shot]]

    # ── Feedback & Learning ──────────────────────────────────────────────

    def update_from_outcome(
        self,
        strategy_id: str | None = None,
        quality: float = 0.5,
        feedback_type: str = "implicit",
        notes: str = "",
    ) -> bool:
        """Update strategy from interaction outcome.

        Args:
            strategy_id: Which strategy to update. None = last active.
            quality: 0.0 = terrible, 1.0 = excellent.
            feedback_type: "explicit" (user said), "implicit" (inferred), "self-assessed".
            notes: Optional notes about why.

        Returns:
            True if strategy was found and updated.
        """
        sid = strategy_id or self._active_strategy_id
        if not sid:
            return False

        strategy = self._find_strategy(sid)
        if not strategy:
            return False

        # Bayesian update
        strategy.update(quality)

        # Record outcome
        outcome = Outcome(
            strategy_id=sid,
            domain=strategy.domain,
            quality=quality,
            feedback_type=feedback_type,
            notes=notes,
        )
        self.outcome_history.append(outcome)
        if len(self.outcome_history) > 500:
            self.outcome_history = self.outcome_history[-500:]

        self._outcome_count += 1

        # Trigger evolution periodically
        if self._outcome_count % self.evolution_interval == 0:
            self.evolve_strategies()

        return True

    def add_example(
        self,
        query: str,
        response_summary: str,
        domain: str = "general",
        quality_score: float = 0.7,
    ) -> FewShotExample:
        """Add a successful interaction as a few-shot example.

        Call this when an interaction went well and could serve as
        a template for future similar queries.
        """
        ex = FewShotExample(
            query=query,
            response_summary=response_summary,
            domain=domain,
            quality_score=quality_score,
        )
        self.examples[domain].append(ex)
        # Cap per domain
        if len(self.examples[domain]) > self.max_examples:
            # Keep best by quality
            self.examples[domain].sort(key=lambda e: e.quality_score, reverse=True)
            self.examples[domain] = self.examples[domain][: self.max_examples]
        return ex

    # ── Strategy Evolution ───────────────────────────────────────────────

    def evolve_strategies(self) -> dict[str, Any]:
        """Evolve strategies: crossover top performers, mutate, prune weak.

        Called periodically after N outcomes. Produces new strategies by:
        1. Crossover: combine elements of two high-performing strategies
        2. Mutation: randomly modify a strategy's components
        3. Pruning: remove consistently underperforming strategies

        Returns:
            Summary of evolution actions taken.
        """
        actions = {"crossed": 0, "mutated": 0, "pruned": 0, "domains": []}

        for domain in list(self.strategies.keys()):
            strats = self.strategies[domain]
            if len(strats) < 2:
                continue

            actions["domains"].append(domain)

            # Sort by posterior mean
            strats.sort(key=lambda s: s.mean_reward, reverse=True)

            # 1. Crossover: top 2 → new child
            if len(strats) >= 2 and len(strats) < self.max_strategies_per_domain:
                parent_a, parent_b = strats[0], strats[1]
                child = self._crossover(parent_a, parent_b, domain)
                if child:
                    strats.append(child)
                    actions["crossed"] += 1

            # 2. Mutate: randomly modify a mid-tier strategy
            if len(strats) >= 3:
                mid = strats[len(strats) // 2]
                if mid.pulls >= 3:  # only mutate after some usage
                    mutated = self._mutate(mid, domain)
                    if mutated:
                        strats.append(mutated)
                        actions["mutated"] += 1

            # 3. Prune: remove worst if over capacity
            while len(strats) > self.max_strategies_per_domain:
                # Don't prune strategies that haven't been tried enough
                prunable = [s for s in strats if s.pulls >= 5]
                if not prunable:
                    break
                worst = min(prunable, key=lambda s: s.mean_reward)
                strats.remove(worst)
                actions["pruned"] += 1

            self.strategies[domain] = strats

        return actions

    def _crossover(
        self, parent_a: PromptStrategy, parent_b: PromptStrategy, domain: str
    ) -> PromptStrategy | None:
        """Create child strategy by combining elements of two parents."""
        # Take system prefix from better parent, CoT from the other
        child = PromptStrategy(
            name=f"cross_{parent_a.name}_{parent_b.name}"[:30],
            domain=domain,
            system_prefix=parent_a.system_prefix,
            cot_template=parent_b.cot_template,
            few_shot_ids=list(set(parent_a.few_shot_ids + parent_b.few_shot_ids))[
                : self.max_few_shot
            ],
            tags=["evolved", "crossover"],
        )
        return child

    def _mutate(
        self, strategy: PromptStrategy, domain: str
    ) -> PromptStrategy | None:
        """Create a mutated copy of a strategy."""
        # Choose what to mutate
        mutation_type = random.choice(["prefix", "cot", "swap_cot"])

        child = PromptStrategy(
            name=f"mut_{strategy.name}"[:30],
            domain=domain,
            system_prefix=strategy.system_prefix,
            cot_template=strategy.cot_template,
            few_shot_ids=list(strategy.few_shot_ids),
            tags=["evolved", "mutation"],
        )

        if mutation_type == "prefix":
            # Augment the prefix
            augments = [
                " Consider multiple perspectives.",
                " Be precise and quantitative.",
                " Cite evidence for claims.",
                " Focus on practical implications.",
                " Identify assumptions and limitations.",
            ]
            child.system_prefix += random.choice(augments)
        elif mutation_type == "cot":
            # Add structure to CoT
            templates = [
                "First, let me identify the key variables:\n",
                "I'll approach this by:\n1. Understanding the context\n2. Analyzing the data\n3. Drawing conclusions\n",
                "Key question: What's the core issue here?\n",
                "Let me consider this from first principles:\n",
            ]
            child.cot_template = random.choice(templates)
        elif mutation_type == "swap_cot":
            # Remove CoT entirely (test if direct is better)
            child.cot_template = ""

        return child

    def create_domain_strategy(
        self,
        domain: str,
        name: str,
        system_prefix: str,
        cot_template: str = "",
    ) -> PromptStrategy:
        """Manually create a domain-specific strategy.

        Use this when you know what works for a domain based on
        experience or procedural memory rules.
        """
        strategy = PromptStrategy(
            name=name,
            domain=domain,
            system_prefix=system_prefix,
            cot_template=cot_template,
            tags=["manual"],
        )
        self.strategies[domain].append(strategy)
        return strategy

    # ── Query Methods ────────────────────────────────────────────────────

    def _find_strategy(self, strategy_id: str) -> PromptStrategy | None:
        """Find a strategy by ID across all domains."""
        for strats in self.strategies.values():
            for s in strats:
                if s.id == strategy_id:
                    return s
        return None

    def strategy_report(self) -> dict[str, Any]:
        """Per-domain strategy performance summary."""
        report: dict[str, Any] = {}
        for domain, strats in self.strategies.items():
            if not strats:
                continue
            domain_report = []
            for s in sorted(strats, key=lambda s: s.mean_reward, reverse=True):
                domain_report.append({
                    "id": s.id,
                    "name": s.name,
                    "mean_reward": round(s.mean_reward, 3),
                    "pulls": s.pulls,
                    "total_reward": round(s.total_reward, 2),
                    "tags": s.tags,
                })
            report[domain] = domain_report
        return report

    def best_strategy_for(self, domain: str) -> PromptStrategy | None:
        """Return the best-performing strategy for a domain (exploitation)."""
        candidates = self._get_candidates(domain)
        if not candidates:
            return None
        # Only consider strategies with enough data
        experienced = [s for s in candidates if s.pulls >= 3]
        if not experienced:
            return candidates[0]
        return max(experienced, key=lambda s: s.mean_reward)

    def summary(self) -> str:
        """Human-readable summary of the adapter's state."""
        lines = [
            f"SoftPromptAdapter — {self._outcome_count} outcomes processed",
            f"Domains: {list(self.strategies.keys())}",
            f"Evolution interval: every {self.evolution_interval} outcomes",
            "",
        ]
        for domain, strats in self.strategies.items():
            lines.append(f"  {domain}: {len(strats)} strategies")
            best = self.best_strategy_for(domain)
            if best and best.pulls > 0:
                lines.append(
                    f"    Best: {best.name} (reward={best.mean_reward:.3f}, n={best.pulls})"
                )
        examples_total = sum(len(v) for v in self.examples.values())
        lines.append(f"\nFew-shot examples: {examples_total}")
        lines.append(f"Outcome history: {len(self.outcome_history)}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self, path: str | Path) -> None:
        """Save adapter state to JSON."""
        data = {
            "strategies": {
                domain: [s.to_dict() for s in strats]
                for domain, strats in self.strategies.items()
            },
            "examples": {
                domain: [e.to_dict() for e in exs]
                for domain, exs in self.examples.items()
            },
            "outcome_history": [o.to_dict() for o in self.outcome_history[-200:]],
            "outcome_count": self._outcome_count,
            "active_strategy_id": self._active_strategy_id,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        """Load adapter state from JSON."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.strategies = defaultdict(list)
            for domain, strats in data.get("strategies", {}).items():
                self.strategies[domain] = [PromptStrategy.from_dict(s) for s in strats]
            self.examples = defaultdict(list)
            for domain, exs in data.get("examples", {}).items():
                self.examples[domain] = [FewShotExample.from_dict(e) for e in exs]
            self.outcome_history = [
                Outcome.from_dict(o) for o in data.get("outcome_history", [])
            ]
            self._outcome_count = data.get("outcome_count", 0)
            self._active_strategy_id = data.get("active_strategy_id")
            return True
        except Exception:
            return False
