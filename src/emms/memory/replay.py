"""ExperienceReplay — prioritized experience replay for EMMS.

Inspired by Prioritized Experience Replay (PER, Schaul et al. 2016):
  priority = w_imp*importance + w_str*strength + w_rec*recency + w_nov*novelty

Importance Sampling (IS) weights correct for the sampling bias introduced
by non-uniform priorities.

Usage:
    replay = ExperienceReplay(memory)
    batch = replay.sample(k=8)
    for entry in batch.entries:
        print(entry.item.content, entry.weight)
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from emms.core.models import MemoryItem, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReplayEntry:
    """A single item drawn from the replay buffer."""
    item: MemoryItem
    priority: float
    weight: float          # IS correction weight  w_i = (1/N * 1/P(i))^β
    sampled_at: float = field(default_factory=time.time)


@dataclass
class ReplayBatch:
    """A sampled mini-batch with metadata."""
    entries: list[ReplayEntry]
    batch_size: int
    total_items_considered: int
    max_priority: float
    min_priority: float
    mean_priority: float
    beta_used: float

    def to_retrieval_results(self) -> list[RetrievalResult]:
        """Convert to RetrievalResult list sorted by priority (desc)."""
        out = []
        for entry in sorted(self.entries, key=lambda e: -e.priority):
            out.append(RetrievalResult(
                memory=entry.item,
                score=entry.priority,
                source_tier=entry.item.tier,
                strategy="replay",
            ))
        return out


# ---------------------------------------------------------------------------
# ExperienceReplay
# ---------------------------------------------------------------------------

class ExperienceReplay:
    """Prioritized experience replay buffer.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing memory object.
    alpha : float
        Priority exponentiation.  0 = uniform, 1 = fully prioritized.
    beta : float
        IS correction exponent.  0 = no correction, 1 = full correction.
    w_imp : float
        Weight for importance signal.
    w_str : float
        Weight for memory strength.
    w_rec : float
        Weight for recency (exponential decay).
    w_nov : float
        Weight for novelty (inverse access count).
    recency_half_life : float
        Half-life in seconds for recency scoring.
    exclusion_window : int
        Number of recently-sampled IDs to exclude from next draw (prevent
        over-sampling hot items).
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        memory: Any,
        alpha: float = 0.6,
        beta: float = 0.4,
        w_imp: float = 0.35,
        w_str: float = 0.30,
        w_rec: float = 0.20,
        w_nov: float = 0.15,
        recency_half_life: float = 3600.0,
        exclusion_window: int = 20,
        seed: int | None = None,
    ) -> None:
        self.memory = memory
        self.alpha = alpha
        self.beta = beta
        self.w_imp = w_imp
        self.w_str = w_str
        self.w_rec = w_rec
        self.w_nov = w_nov
        self.recency_half_life = recency_half_life
        self._rng = random.Random(seed)
        self._exclusion: deque[str] = deque(maxlen=exclusion_window)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def sample(self, k: int = 8, beta: float | None = None) -> ReplayBatch:
        """Draw a mini-batch of k items by priority.

        Parameters
        ----------
        k : int
            Batch size (capped at number of eligible items).
        beta : float | None
            Override IS exponent for this call only.
        """
        beta = beta if beta is not None else self.beta
        items = self._collect_all()
        eligible = [it for it in items if it.id not in self._exclusion]
        if not eligible:
            eligible = items  # relax exclusion if nothing eligible

        k = min(k, len(eligible))
        if k == 0:
            return ReplayBatch(
                entries=[],
                batch_size=0,
                total_items_considered=0,
                max_priority=0.0,
                min_priority=0.0,
                mean_priority=0.0,
                beta_used=beta,
            )

        now = time.time()
        priorities = [self._priority(it, now) for it in eligible]

        # Stochastic prioritised sampling via alias method
        indices = self._alias_sample(priorities, k)
        sampled_items = [eligible[i] for i in indices]
        sampled_prios = [priorities[i] for i in indices]

        # IS weights
        N = len(eligible)
        prio_sum = sum(p ** self.alpha for p in priorities)
        weights = []
        for p in sampled_prios:
            prob_i = (p ** self.alpha) / prio_sum
            raw_weight = (1.0 / (N * prob_i)) ** beta
            weights.append(raw_weight)
        # normalize
        max_w = max(weights) if weights else 1.0
        weights = [w / max_w for w in weights]

        entries = [
            ReplayEntry(item=it, priority=p, weight=w)
            for it, p, w in zip(sampled_items, sampled_prios, weights)
        ]

        # Update exclusion window
        for it in sampled_items:
            self._exclusion.append(it.id)

        return ReplayBatch(
            entries=entries,
            batch_size=len(entries),
            total_items_considered=len(eligible),
            max_priority=max(priorities),
            min_priority=min(priorities),
            mean_priority=sum(priorities) / len(priorities),
            beta_used=beta,
        )

    def sample_top(self, k: int = 8) -> list[ReplayEntry]:
        """Return top-k highest-priority items (deterministic)."""
        items = self._collect_all()
        if not items:
            return []
        now = time.time()
        scored = sorted(items, key=lambda it: -self._priority(it, now))
        return [
            ReplayEntry(item=it, priority=self._priority(it, now), weight=1.0)
            for it in scored[:k]
        ]

    def replay_context(self, k: int = 5) -> list[RetrievalResult]:
        """Convenience: sample k items and return as RetrievalResult list."""
        batch = self.sample(k=k)
        return batch.to_retrieval_results()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[MemoryItem]:
        """Flatten all 4 tiers of hierarchical memory."""
        items: list[MemoryItem] = []
        for tier_store in (self.memory.working, self.memory.short_term):
            items.extend(tier_store)
        for tier_store in (self.memory.long_term, self.memory.semantic):
            items.extend(tier_store.values())
        return items

    def _priority(self, item: MemoryItem, now: float) -> float:
        """Compute raw priority ∈ (0, 1] for an item."""
        exp = getattr(item, "experience", None)
        imp = float(exp.importance if exp else 0.5)
        strength = float(getattr(item, "memory_strength", 1.0))
        access = max(1, int(getattr(item, "access_count", 0)))
        stored_at = float(getattr(item, "stored_at", now))

        # Recency: exponential decay
        age = now - stored_at
        rec = math.exp(-math.log(2) * age / max(self.recency_half_life, 1.0))

        # Novelty: inverse of access count (log-scaled)
        nov = 1.0 / (1.0 + math.log1p(access))

        priority = (
            self.w_imp * imp
            + self.w_str * strength
            + self.w_rec * rec
            + self.w_nov * nov
        )
        return max(priority, 1e-6)

    # ------------------------------------------------------------------
    # Alias method (Vose, 1991) for O(n) setup, O(1) sample
    # ------------------------------------------------------------------

    def _alias_sample(self, weights: list[float], k: int) -> list[int]:
        """Sample k indices without replacement using weighted sampling.

        Falls back to simple weighted random.choices when k > n/2 since
        the alias method is not designed for without-replacement sampling.
        """
        n = len(weights)
        if n == 0:
            return []
        w_sum = sum(weights)
        if w_sum <= 0:
            return self._rng.sample(range(n), min(k, n))

        normalised = [w / w_sum for w in weights]

        # For small batches relative to pool, use weighted sampling without
        # replacement via a greedy approach
        if k >= n:
            return list(range(n))

        # Build alias table
        prob = [p * n for p in normalised]
        alias = list(range(n))
        small: list[int] = []
        large: list[int] = []
        for i, p in enumerate(prob):
            (small if p < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            l = large[-1]
            alias[s] = l
            prob[l] = prob[l] + prob[s] - 1.0
            if prob[l] < 1.0:
                large.pop()
                small.append(l)
            elif not large:
                break

        # Sample k distinct indices
        seen: set[int] = set()
        result: list[int] = []
        attempts = 0
        while len(result) < k and attempts < k * 10:
            attempts += 1
            col = self._rng.randint(0, n - 1)
            idx = col if self._rng.random() < prob[col] else alias[col]
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        # fallback
        if len(result) < k:
            remaining = [i for i in range(n) if i not in seen]
            self._rng.shuffle(remaining)
            result.extend(remaining[: k - len(result)])
        return result
