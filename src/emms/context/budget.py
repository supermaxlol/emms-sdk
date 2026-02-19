"""Token-budget-aware memory eviction: MemoryBudget.

When the total token footprint of stored memories exceeds a configured budget,
MemoryBudget selects low-value memories for eviction using a composite score.

Eviction score (lower = evicted first)::

    score = w_imp  * importance
          + w_str  * memory_strength
          + w_acc  * log1p(access_count)   (normalised)
          + w_rec  * recency_score          (exponential decay)

Protection rules (applied before scoring):
- Memories in protected tiers (e.g. SEMANTIC) are never evicted.
- Memories with importance >= importance_threshold are never evicted.

Usage::

    from emms import EMMS, Experience
    from emms.context.budget import MemoryBudget, EvictionPolicy

    agent = EMMS()
    # … store experiences …

    budget = MemoryBudget(
        agent.memory,
        max_tokens=50_000,
        policy=EvictionPolicy.COMPOSITE,
        protected_tiers=["semantic"],
        importance_threshold=0.8,
    )
    report = budget.enforce(dry_run=True)
    print(report.summary())

    # Actually evict
    report = budget.enforce(dry_run=False)
    print(f"Evicted {report.evicted_count} memories, freed {report.freed_tokens} tokens")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# EvictionPolicy
# ---------------------------------------------------------------------------

class EvictionPolicy(str, Enum):
    """Eviction scoring policy.

    COMPOSITE
        Weighted combination of importance, strength, recency, access count.
        (Default and recommended.)
    LRU
        Least-Recently-Used: evict by oldest ``stored_at`` first.
    LFU
        Least-Frequently-Used: evict by lowest ``access_count`` first.
    IMPORTANCE
        Evict lowest importance first.
    STRENGTH
        Evict lowest memory_strength first.
    """
    COMPOSITE = "composite"
    LRU = "lru"
    LFU = "lfu"
    IMPORTANCE = "importance"
    STRENGTH = "strength"


# ---------------------------------------------------------------------------
# EvictionCandidate
# ---------------------------------------------------------------------------

@dataclass
class EvictionCandidate:
    """A memory item selected for potential eviction.

    Attributes
    ----------
    memory_id : str
        ID of the MemoryItem.
    experience_id : str
        ID of the underlying Experience.
    content_excerpt : str
        First 80 characters of the experience content.
    domain : str
        Domain label.
    tier : str
        Memory tier name.
    importance : float
        Importance score.
    memory_strength : float
        Decay-adjusted strength.
    access_count : int
        Number of accesses.
    stored_at : float
        When the memory was stored.
    eviction_score : float
        Composite score (lower = evicted sooner).
    token_estimate : int
        Approximate token cost of this memory.
    """
    memory_id: str
    experience_id: str
    content_excerpt: str
    domain: str
    tier: str
    importance: float
    memory_strength: float
    access_count: int
    stored_at: float
    eviction_score: float
    token_estimate: int


# ---------------------------------------------------------------------------
# BudgetReport
# ---------------------------------------------------------------------------

@dataclass
class BudgetReport:
    """Result of a MemoryBudget enforcement pass.

    Attributes
    ----------
    total_memories : int
        Memories examined.
    total_tokens : int
        Token footprint before enforcement.
    budget_tokens : int
        Configured budget ceiling.
    over_budget : bool
        Whether the total exceeded the budget.
    evicted_count : int
        How many memories were (or would be) evicted.
    freed_tokens : int
        Tokens freed by eviction.
    remaining_tokens : int
        Estimated token footprint after enforcement.
    candidates : list[EvictionCandidate]
        The candidates chosen for eviction (in eviction order).
    dry_run : bool
        If True, nothing was actually deleted.
    protected_count : int
        How many memories were skipped due to protection rules.
    """
    total_memories: int
    total_tokens: int
    budget_tokens: int
    over_budget: bool
    evicted_count: int
    freed_tokens: int
    remaining_tokens: int
    candidates: list[EvictionCandidate]
    dry_run: bool
    protected_count: int

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "DRY RUN — " if self.dry_run else ""
        if not self.over_budget:
            return (
                f"{status}Budget OK: {self.total_tokens:,} / {self.budget_tokens:,} tokens "
                f"({self.total_memories} memories, {self.protected_count} protected)"
            )
        return (
            f"{status}Over budget: {self.total_tokens:,} tokens "
            f"(limit {self.budget_tokens:,}), "
            f"evicted {self.evicted_count} memories, "
            f"freed {self.freed_tokens:,} tokens → "
            f"{self.remaining_tokens:,} remaining"
        )


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

def _estimate_tokens(item: Any) -> int:
    """Rough token count for a MemoryItem (4 chars ≈ 1 token)."""
    content = item.experience.content or ""
    facts = " ".join(item.experience.facts or [])
    title = item.experience.title or ""
    total_chars = len(content) + len(facts) + len(title) + 50  # metadata overhead
    return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# MemoryBudget
# ---------------------------------------------------------------------------

class MemoryBudget:
    """Token-budget-aware memory eviction.

    Parameters
    ----------
    memory : HierarchicalMemory
        The hierarchical memory store to manage.
    max_tokens : int
        Maximum allowed total token footprint (default 100,000).
    policy : EvictionPolicy
        Scoring policy for ranking eviction candidates.
    protected_tiers : list[str]
        Tier names that are immune to eviction (default: ``["semantic"]``).
    importance_threshold : float
        Memories with importance >= this value are never evicted (default 0.8).
    weights : dict[str, float] | None
        Weights for the COMPOSITE policy.  Keys: ``importance``, ``strength``,
        ``access``, ``recency``.  Values must sum to 1.0.
    recency_half_life : float
        Half-life (seconds) for the recency decay term (default 86400 = 1 day).
    """

    _DEFAULT_WEIGHTS = {
        "importance": 0.35,
        "strength": 0.30,
        "access": 0.20,
        "recency": 0.15,
    }

    def __init__(
        self,
        memory: Any,
        *,
        max_tokens: int = 100_000,
        policy: EvictionPolicy = EvictionPolicy.COMPOSITE,
        protected_tiers: list[str] | None = None,
        importance_threshold: float = 0.8,
        weights: dict[str, float] | None = None,
        recency_half_life: float = 86400.0,
    ) -> None:
        self.memory = memory
        self.max_tokens = max_tokens
        self.policy = policy
        self.protected_tiers: set[str] = set(
            protected_tiers if protected_tiers is not None else ["semantic"]
        )
        self.importance_threshold = importance_threshold
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)
        self.recency_half_life = recency_half_life

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def token_footprint(self) -> dict[str, Any]:
        """Return per-tier and total token estimates.

        Returns
        -------
        dict with keys: ``total``, ``by_tier``, ``memory_count``.
        """
        by_tier: dict[str, int] = {}
        total = 0
        count = 0
        for tier_name, store in self.memory._iter_tiers():
            tier_str = tier_name.value if hasattr(tier_name, "value") else str(tier_name)
            tier_total = 0
            for item in store:
                tier_total += _estimate_tokens(item)
                count += 1
            by_tier[tier_str] = tier_total
            total += tier_total
        return {"total": total, "by_tier": by_tier, "memory_count": count}

    def enforce(self, dry_run: bool = False) -> BudgetReport:
        """Enforce the token budget by evicting low-value memories.

        Args:
            dry_run: If True, compute the eviction plan but do not delete anything.

        Returns:
            BudgetReport describing the result.
        """
        all_items = self._collect_all()
        total_tokens = sum(_estimate_tokens(item) for item in all_items)
        total_memories = len(all_items)

        if total_tokens <= self.max_tokens:
            return BudgetReport(
                total_memories=total_memories,
                total_tokens=total_tokens,
                budget_tokens=self.max_tokens,
                over_budget=False,
                evicted_count=0,
                freed_tokens=0,
                remaining_tokens=total_tokens,
                candidates=[],
                dry_run=dry_run,
                protected_count=0,
            )

        # Split into evictable and protected
        evictable, protected = self._split_protected(all_items)
        protected_count = len(protected)

        # Score and sort evictable items (ascending eviction_score = first to go)
        candidates = self._score_and_sort(evictable)

        # Greedy eviction until within budget
        tokens_to_free = total_tokens - self.max_tokens
        freed = 0
        selected: list[EvictionCandidate] = []
        evictable_ids: list[str] = []

        for cand in candidates:
            if freed >= tokens_to_free:
                break
            selected.append(cand)
            freed += cand.token_estimate
            evictable_ids.append(cand.memory_id)

        if not dry_run:
            self._evict_ids(evictable_ids)

        return BudgetReport(
            total_memories=total_memories,
            total_tokens=total_tokens,
            budget_tokens=self.max_tokens,
            over_budget=True,
            evicted_count=len(selected),
            freed_tokens=freed,
            remaining_tokens=max(0, total_tokens - freed),
            candidates=selected,
            dry_run=dry_run,
            protected_count=protected_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        """Gather all MemoryItems from all tiers."""
        items = []
        for _, store in self.memory._iter_tiers():
            for item in store:
                items.append(item)
        return items

    def _split_protected(self, items: list[Any]) -> tuple[list[Any], list[Any]]:
        """Split items into (evictable, protected)."""
        evictable, protected = [], []
        for item in items:
            tier_str = item.tier.value if hasattr(item.tier, "value") else str(item.tier)
            if tier_str in self.protected_tiers:
                protected.append(item)
            elif item.experience.importance >= self.importance_threshold:
                protected.append(item)
            else:
                evictable.append(item)
        return evictable, protected

    def _score_and_sort(self, items: list[Any]) -> list[EvictionCandidate]:
        """Score items and return ascending (lowest score first = evicted first)."""
        if not items:
            return []

        now = time.time()

        # Compute raw stats for normalisation
        max_access = max((i.access_count for i in items), default=1) or 1

        candidates: list[EvictionCandidate] = []
        for item in items:
            score = self._composite_score(item, now, max_access)
            tier_str = item.tier.value if hasattr(item.tier, "value") else str(item.tier)
            candidates.append(EvictionCandidate(
                memory_id=item.id,
                experience_id=item.experience.id,
                content_excerpt=item.experience.content[:80],
                domain=item.experience.domain,
                tier=tier_str,
                importance=item.experience.importance,
                memory_strength=item.memory_strength,
                access_count=item.access_count,
                stored_at=item.stored_at,
                eviction_score=score,
                token_estimate=_estimate_tokens(item),
            ))

        # Sort ascending: lowest score = evicted first
        candidates.sort(key=lambda c: c.eviction_score)
        return candidates

    def _composite_score(self, item: Any, now: float, max_access: int) -> float:
        """Compute composite eviction score (higher = keep, lower = evict)."""
        if self.policy == EvictionPolicy.LRU:
            return item.stored_at  # older = lower score = evicted first

        if self.policy == EvictionPolicy.LFU:
            return float(item.access_count)

        if self.policy == EvictionPolicy.IMPORTANCE:
            return item.experience.importance

        if self.policy == EvictionPolicy.STRENGTH:
            return item.memory_strength

        # COMPOSITE (default)
        w = self.weights
        importance_term = w.get("importance", 0.35) * item.experience.importance
        strength_term = w.get("strength", 0.30) * item.memory_strength
        access_term = w.get("access", 0.20) * (
            math.log1p(item.access_count) / math.log1p(max_access)
        )
        age = now - item.stored_at
        recency_term = w.get("recency", 0.15) * math.exp(
            -0.693 * age / max(1.0, self.recency_half_life)
        )
        return importance_term + strength_term + access_term + recency_term

    def _evict_ids(self, ids: list[str]) -> int:
        """Remove memories by ID from all tiers. Returns count removed."""
        id_set = set(ids)
        removed = 0

        from emms.core.models import MemoryTier
        from collections import deque

        # Working / short_term are deques — rebuild without evicted items
        for tier_attr in ("working", "short_term"):
            store = getattr(self.memory, tier_attr, None)
            if store is None:
                continue
            keep = deque(item for item in store if item.id not in id_set)
            # Check which were removed
            removed_here = len(store) - len(keep)
            removed += removed_here
            # Replace contents in-place
            store.clear()
            store.extend(keep)

        # Long-term / semantic are dicts
        for tier_attr in ("long_term", "semantic"):
            store = getattr(self.memory, tier_attr, None)
            if store is None:
                continue
            to_del = [k for k in store if k in id_set]
            for k in to_del:
                del store[k]
                removed += 1

        return removed
