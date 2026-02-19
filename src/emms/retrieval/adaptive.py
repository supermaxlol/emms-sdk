"""Adaptive retrieval via Thompson Sampling Beta-Bernoulli bandit.

Each retrieval strategy is modelled as an arm in a multi-armed bandit.
When the agent calls ``adaptive_retrieve()``, the retriever samples one θ
from each arm's Beta distribution and selects the arm with the highest sample.
After the user provides feedback (0 = miss, 1 = hit), the corresponding arm's
parameters are updated.

This implements a contextual variant: the query is also used to route between
strategies (via domain keyword matching), but Thompson Sampling governs the
final selection when no strong prior match exists.

Parameters and defaults follow standard Beta-Bernoulli bandit literature:
    - α₀ = β₀ = 1  (uniform / ignorance prior)
    - decay ∈ (0, 1]  (geometric discount applied after each step to forget stale wins)

Usage::

    from emms import EMMS
    from emms.retrieval.adaptive import AdaptiveRetriever

    agent = EMMS()
    retriever = AdaptiveRetriever(agent.memory)
    results = retriever.retrieve("neural networks", max_results=5)

    # Provide positive feedback for the first result
    retriever.record_feedback(results[0].strategy, reward=1)

    # Inspect current beliefs
    for name, belief in retriever.get_beliefs().items():
        print(f"{name}: α={belief.alpha:.2f}, β={belief.beta:.2f}, "
              f"μ={belief.mean:.3f}")

    # Save / restore across sessions
    retriever.save_state("/tmp/adaptive.json")
    retriever.load_state("/tmp/adaptive.json")
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# StrategyBelief — Beta-Bernoulli arm state
# ---------------------------------------------------------------------------

@dataclass
class StrategyBelief:
    """Beta distribution belief state for a single retrieval strategy.

    Parameters
    ----------
    name : str
        Strategy identifier (e.g. ``"semantic"``, ``"bm25"``, ``"temporal"``).
    alpha : float
        Beta distribution α (successes + prior).  Default 1.0 (uniform prior).
    beta : float
        Beta distribution β (failures + prior).  Default 1.0 (uniform prior).
    pulls : int
        How many times this arm has been pulled.
    rewards : int
        How many times this arm received a positive reward signal.
    last_updated : float
        Unix timestamp of the last update.
    """
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    rewards: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def mean(self) -> float:
        """Posterior mean E[θ] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance Var[θ]."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one Thompson sample from Beta(α, β).

        Uses the Johnk method (sum of two log-uniform variates) for
        pure-Python Beta sampling without NumPy.
        """
        r = rng or random
        return _beta_sample(self.alpha, self.beta, r)

    def update(self, reward: float, decay: float = 1.0) -> None:
        """Update beliefs given a reward in [0, 1].

        Args:
            reward: 1.0 = success, 0.0 = failure, intermediate values allowed.
            decay: Geometric discount applied to existing α and β before update.
                   1.0 = no decay; 0.9 = 10% annual-style forgetting.
        """
        # Apply decay (shrink existing counts towards prior = 1)
        if decay < 1.0:
            self.alpha = 1.0 + (self.alpha - 1.0) * decay
            self.beta = 1.0 + (self.beta - 1.0) * decay

        # Bayesian update
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.pulls += 1
        if reward > 0.5:
            self.rewards += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyBelief":
        return cls(**d)


# ---------------------------------------------------------------------------
# Beta sampling (pure Python — Johnk / Cheng methods)
# ---------------------------------------------------------------------------

def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Pure-Python Beta(α, β) sampler using Cheng's BA algorithm for α,β > 1,
    and the standard X/(X+Y) method (via Gamma) otherwise.

    For simplicity we use the X/(X+Y) approach with Gamma approximation via
    the Marsaglia-Tsang method.  This avoids NumPy while being numerically
    stable for the parameter ranges we encounter (1 ≤ α,β ≤ ~100).
    """
    # Special case: uniform prior
    if alpha == 1.0 and beta == 1.0:
        return rng.random()

    x = _gamma_sample(alpha, rng)
    y = _gamma_sample(beta, rng)
    total = x + y
    if total == 0.0:
        return 0.5
    return x / total


def _gamma_sample(shape: float, rng: random.Random) -> float:
    """Marsaglia-Tsang Gamma(shape, 1) sampler."""
    if shape < 1.0:
        # Use the boosting trick: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        return _gamma_sample(shape + 1.0, rng) * (rng.random() ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v ** 3
        u = rng.random()
        if u < 1.0 - 0.0331 * (x * x) ** 2:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


# ---------------------------------------------------------------------------
# AdaptiveRetriever
# ---------------------------------------------------------------------------

# Built-in strategy names — these map to methods on the retriever
_BUILTIN_STRATEGIES = [
    "semantic",
    "bm25",
    "temporal",
    "domain",
    "importance",
]


class AdaptiveRetriever:
    """Multi-armed bandit retriever with Thompson Sampling.

    Each retrieval strategy arm starts with a Beta(1, 1) uniform prior.
    On each retrieve() call, one arm is selected by drawing a Thompson sample
    from each arm's posterior and picking the argmax.  The caller should then
    provide feedback via ``record_feedback()`` to close the learning loop.

    Parameters
    ----------
    memory : HierarchicalMemory
        The hierarchical memory store to search.
    strategies : list[str] | None
        Names of the arms.  Defaults to all 5 built-in strategies.
    decay : float
        Geometric discount applied to α, β before each update.
        1.0 = no forgetting; 0.95 = gentle forgetting.
    seed : int | None
        RNG seed for reproducible Thompson sampling.
    embedder : EmbeddingProvider | None
        Optional embedder for semantic and BM25 channels.
    """

    def __init__(
        self,
        memory: Any,
        *,
        strategies: list[str] | None = None,
        decay: float = 1.0,
        seed: int | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.memory = memory
        self.decay = decay
        self.embedder = embedder
        self._rng = random.Random(seed)

        arm_names = strategies if strategies is not None else list(_BUILTIN_STRATEGIES)
        self.beliefs: dict[str, StrategyBelief] = {
            name: StrategyBelief(name=name) for name in arm_names
        }
        self._last_selected: str | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        explore: bool = True,
    ) -> list[Any]:
        """Retrieve memories using the Thompson-sampled strategy.

        Args:
            query: Natural-language search query.
            max_results: Maximum results to return.
            explore: If True (default), use Thompson Sampling to select the
                     strategy.  If False, use the argmax (pure exploitation).

        Returns:
            List of RetrievalResult objects from the selected strategy.
        """
        strategy_name = self._select_strategy(explore=explore)
        self._last_selected = strategy_name
        return self._run_strategy(strategy_name, query, max_results)

    def record_feedback(
        self,
        strategy_name: str | None = None,
        reward: float = 1.0,
    ) -> None:
        """Record feedback for a strategy arm.

        Args:
            strategy_name: Which arm to update.  If None, updates the last
                           strategy selected by retrieve().
            reward: 1.0 = the retrieval was helpful; 0.0 = it was not.
                    Intermediate values are also accepted.
        """
        name = strategy_name or self._last_selected
        if name is None or name not in self.beliefs:
            return
        self.beliefs[name].update(reward, decay=self.decay)

    def get_beliefs(self) -> dict[str, StrategyBelief]:
        """Return the current Beta belief state for all arms."""
        return dict(self.beliefs)

    def best_strategy(self) -> str:
        """Return the arm with the highest posterior mean (exploitation)."""
        return max(self.beliefs, key=lambda n: self.beliefs[n].mean)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Persist belief state to a JSON file."""
        data = {
            "decay": self.decay,
            "beliefs": {name: b.to_dict() for name, b in self.beliefs.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_state(self, path: str | Path) -> None:
        """Load belief state from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.decay = data.get("decay", self.decay)
        for name, bdict in data.get("beliefs", {}).items():
            self.beliefs[name] = StrategyBelief.from_dict(bdict)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, explore: bool) -> str:
        """Select an arm via Thompson Sampling (explore=True) or argmax."""
        if explore:
            samples = {name: b.sample(self._rng) for name, b in self.beliefs.items()}
            return max(samples, key=lambda n: samples[n])
        return self.best_strategy()

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------

    def _collect_all_items(self) -> list[Any]:
        """Gather all non-expired MemoryItems from all tiers."""
        items = []
        for _, store in self.memory._iter_tiers():
            for item in store:
                if not (item.is_expired or item.is_superseded):
                    items.append(item)
        return items

    def _run_strategy(self, strategy_name: str, query: str, max_results: int) -> list[Any]:
        """Dispatch to the correct retrieval implementation."""
        dispatch = {
            "semantic": self._retrieve_semantic,
            "bm25": self._retrieve_bm25,
            "temporal": self._retrieve_temporal,
            "domain": self._retrieve_domain,
            "importance": self._retrieve_importance,
        }
        fn = dispatch.get(strategy_name, self._retrieve_semantic)
        return fn(query, max_results)

    def _retrieve_semantic(self, query: str, max_results: int) -> list[Any]:
        """Embedding cosine similarity retrieval."""
        from emms.core.models import RetrievalResult
        embedder = self.embedder or getattr(self.memory, "embedder", None)
        items = self._collect_all_items()
        if not items:
            return []

        if embedder is None:
            # Fall back to BM25 without embedder
            return self._retrieve_bm25(query, max_results)

        from emms.core.embeddings import cosine_similarity
        q_emb = embedder.embed(query)
        scored = []
        for item in items:
            if item.embedding is not None:
                sim = max(0.0, float(cosine_similarity(q_emb, item.embedding)))
            else:
                i_emb = embedder.embed(item.experience.content)
                sim = max(0.0, float(cosine_similarity(q_emb, i_emb)))
            scored.append((item, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="adaptive_semantic",
                strategy_scores={"semantic": score},
                explanation=f"adaptive:semantic score={score:.3f}",
            )
            for item, score in scored[:max_results]
        ]

    def _retrieve_bm25(self, query: str, max_results: int) -> list[Any]:
        """BM25 lexical retrieval."""
        from emms.core.models import RetrievalResult
        from emms.retrieval.hybrid import _BM25

        items = self._collect_all_items()
        if not items:
            return []

        bm25 = _BM25(items)
        raw_scores = bm25.scores(query)
        # Normalise to [0, 1]
        max_s = max(raw_scores) if raw_scores else 1.0
        if max_s == 0.0:
            max_s = 1.0

        scored = sorted(
            zip(items, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            RetrievalResult(
                memory=item,
                score=score / max_s,
                source_tier=item.tier,
                strategy="adaptive_bm25",
                strategy_scores={"bm25": score / max_s},
                explanation=f"adaptive:bm25 score={score:.3f}",
            )
            for item, score in scored[:max_results]
        ]

    def _retrieve_temporal(self, query: str, max_results: int) -> list[Any]:
        """Recency-weighted temporal retrieval."""
        from emms.core.models import RetrievalResult
        items = self._collect_all_items()
        if not items:
            return []

        now = time.time()
        half_life = 86400.0  # 1 day

        scored = []
        for item in items:
            age = now - item.stored_at
            score = math.exp(-0.693 * age / half_life)
            # Boost by memory_strength
            score *= item.memory_strength
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="adaptive_temporal",
                strategy_scores={"temporal": score},
                explanation=f"adaptive:temporal score={score:.3f}",
            )
            for item, score in scored[:max_results]
        ]

    def _retrieve_domain(self, query: str, max_results: int) -> list[Any]:
        """Domain keyword matching retrieval."""
        from emms.core.models import RetrievalResult
        import re as _re

        items = self._collect_all_items()
        if not items:
            return []

        q_tokens = set(_re.findall(r"[a-z0-9]+", query.lower()))

        scored = []
        for item in items:
            domain_tokens = set(_re.findall(r"[a-z0-9]+", item.experience.domain.lower()))
            content_tokens = set(_re.findall(r"[a-z0-9]+", item.experience.content.lower()))
            domain_overlap = len(q_tokens & domain_tokens) / max(1, len(q_tokens))
            content_overlap = len(q_tokens & content_tokens) / max(1, len(q_tokens | content_tokens))
            score = 0.6 * domain_overlap + 0.4 * content_overlap
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="adaptive_domain",
                strategy_scores={"domain": score},
                explanation=f"adaptive:domain score={score:.3f}",
            )
            for item, score in scored[:max_results]
        ]

    def _retrieve_importance(self, query: str, max_results: int) -> list[Any]:
        """Importance × strength retrieval."""
        from emms.core.models import RetrievalResult

        items = self._collect_all_items()
        if not items:
            return []

        scored = [
            (item, 0.6 * item.experience.importance + 0.4 * item.memory_strength)
            for item in items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                memory=item,
                score=score,
                source_tier=item.tier,
                strategy="adaptive_importance",
                strategy_scores={"importance": score},
                explanation=f"adaptive:importance score={score:.3f}",
            )
            for item, score in scored[:max_results]
        ]
