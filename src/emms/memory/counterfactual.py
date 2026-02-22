"""CounterfactualEngine — "what if" alternatives to past experiences.

v0.20.0: The Reasoning Mind

Counterfactual thinking — imagining how things could have been different — is
one of the most powerful and uniquely human cognitive capacities. Upward
counterfactuals ("if only I had studied harder, I would have passed") motivate
future improvement by revealing the gap between what happened and what could
have been better. Downward counterfactuals ("at least I didn't fail completely")
provide consolation and resilience by contrasting reality with worse alternatives.

Both serve adaptive functions: upward counterfactuals drive learning and goal
pursuit; downward counterfactuals regulate negative affect and build resilience.
Together they enable an agent to learn not just from what happened but from the
rich space of what could have happened.

The CounterfactualEngine operationalises this for the memory store. For each
memory, it identifies whether it calls for an upward or downward counterfactual
based on emotional valence, generates the alternative framing, and tracks the
valence shift and plausibility. The resulting Counterfactual objects can be
filtered by direction, retrieved per memory, or reviewed for overall affective
quality of the agent's experience base.

Biological analogue: counterfactual thinking as a fundamental cognitive capacity
(Roese 1997); upward counterfactuals as motivators of future behaviour (Markman
et al. 1993); orbitofrontal cortex in evaluating counterfactual outcomes
(Camille et al. 2004); medial prefrontal cortex in mental simulation and
perspective-shifting (Buckner & Carroll 2007); the brain as a generative model
capable of simulating non-actual states of the world (Friston 2010).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Counterfactual:
    """A counterfactual alternative to a past memory."""

    id: str
    basis_memory_id: str
    original_content: str           # excerpt of the original memory
    counterfactual_content: str     # the "what if" alternative
    direction: str                  # "upward" or "downward"
    valence_shift: float            # how much better (+) or worse (-) the CF is
    plausibility: float             # 0..1 — how realistic the alternative is
    domain: str
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        arrow = "↑" if self.direction == "upward" else "↓"
        return (
            f"Counterfactual [{self.domain}] {arrow}  "
            f"shift={self.valence_shift:+.2f}  "
            f"plausibility={self.plausibility:.2f}\n"
            f"  {self.id[:12]}: {self.counterfactual_content[:80]}"
        )


@dataclass
class CounterfactualReport:
    """Result of a CounterfactualEngine.generate() call."""

    total_memories_assessed: int
    counterfactuals_generated: int
    counterfactuals: list[Counterfactual]  # sorted by |valence_shift| desc
    mean_plausibility: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"CounterfactualReport: {self.counterfactuals_generated} counterfactuals "
            f"from {self.total_memories_assessed} memories  "
            f"mean_plausibility={self.mean_plausibility:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for cf in self.counterfactuals[:5]:
            lines.append(f"  {cf.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------


class CounterfactualEngine:
    """Generates counterfactual alternatives to past memory experiences.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    max_counterfactuals:
        Maximum :class:`Counterfactual` objects to generate per call (default 10).
    plausibility_threshold:
        Minimum plausibility for a counterfactual to be retained (default 0.2).
    store_results:
        If ``True``, persist each counterfactual as a ``"counterfactual"``
        domain memory (default False).
    """

    def __init__(
        self,
        memory: Any,
        max_counterfactuals: int = 10,
        plausibility_threshold: float = 0.2,
        store_results: bool = False,
    ) -> None:
        self.memory = memory
        self.max_counterfactuals = max_counterfactuals
        self.plausibility_threshold = plausibility_threshold
        self.store_results = store_results
        self._counterfactuals: list[Counterfactual] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        domain: Optional[str] = None,
        direction: str = "both",
    ) -> CounterfactualReport:
        """Generate counterfactual alternatives from accumulated memories.

        Args:
            domain:    Restrict to this domain (``None`` = all domains).
            direction: ``"upward"`` | ``"downward"`` | ``"both"`` (default).

        Returns:
            :class:`CounterfactualReport` with counterfactuals sorted by
            absolute valence shift descending.
        """
        t0 = time.time()
        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        new_cfs: list[Counterfactual] = []
        for item in items:
            if len(new_cfs) >= self.max_counterfactuals:
                break
            valence = getattr(item.experience, "emotional_valence", 0.0) or 0.0

            if direction in ("both", "upward") and valence < 0.1:
                cf = self._make_counterfactual(item, "upward")
                if cf and cf.plausibility >= self.plausibility_threshold:
                    if self.store_results:
                        self._store_cf(cf)
                    new_cfs.append(cf)
                    self._counterfactuals.append(cf)

            if direction in ("both", "downward") and valence > -0.1:
                cf = self._make_counterfactual(item, "downward")
                if cf and cf.plausibility >= self.plausibility_threshold:
                    if self.store_results:
                        self._store_cf(cf)
                    new_cfs.append(cf)
                    self._counterfactuals.append(cf)

        new_cfs.sort(key=lambda c: abs(c.valence_shift), reverse=True)

        mean_p = (
            sum(c.plausibility for c in new_cfs) / len(new_cfs)
            if new_cfs else 0.0
        )

        return CounterfactualReport(
            total_memories_assessed=len(items),
            counterfactuals_generated=len(new_cfs),
            counterfactuals=new_cfs,
            mean_plausibility=round(mean_p, 4),
            duration_seconds=time.time() - t0,
        )

    def upward(self, n: int = 5) -> list[Counterfactual]:
        """Return the n best upward counterfactuals ("could have been better").

        Args:
            n: Number of counterfactuals to return (default 5).
        """
        ups = [c for c in self._counterfactuals if c.direction == "upward"]
        return sorted(ups, key=lambda c: c.valence_shift, reverse=True)[:n]

    def downward(self, n: int = 5) -> list[Counterfactual]:
        """Return the n best downward counterfactuals ("could have been worse").

        Args:
            n: Number of counterfactuals to return (default 5).
        """
        downs = [c for c in self._counterfactuals if c.direction == "downward"]
        return sorted(downs, key=lambda c: c.valence_shift)[:n]

    def for_memory(self, memory_id: str) -> list[Counterfactual]:
        """Return all counterfactuals generated from a specific memory.

        Args:
            memory_id: ID of the basis memory.

        Returns:
            List of :class:`Counterfactual` for that memory.
        """
        return [c for c in self._counterfactuals if c.basis_memory_id == memory_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_counterfactual(
        self, item: Any, direction: str
    ) -> Optional[Counterfactual]:
        """Generate a single counterfactual for a memory item."""
        content = getattr(item.experience, "content", "") or ""
        domain = getattr(item.experience, "domain", None) or "general"
        strength = min(1.0, max(0.0, getattr(item, "memory_strength", 0.5)))

        # Extract top meaningful token
        tokens = [
            w.strip(".,!?;:\"'()") for w in content.lower().split()
            if len(w.strip(".,!?;:\"'()")) >= 4
            and w.strip(".,!?;:\"'()") not in _STOP_WORDS
        ]
        top_token = tokens[0] if tokens else "this situation"

        excerpt = content[:60].rstrip()

        if direction == "upward":
            cf_content = (
                f"What if [{excerpt}] had led to a more positive outcome? "
                f"If '{top_token}' had been approached differently, "
                f"{domain} outcomes might have been more constructive and successful."
            )
            valence_shift = +0.4
        else:
            cf_content = (
                f"What if [{excerpt}] had encountered more obstacles? "
                f"If '{top_token}' had faced unexpected challenges, "
                f"{domain} outcomes might have been disrupted and less favourable."
            )
            valence_shift = -0.4

        plausibility = round(strength * 0.8, 4)

        return Counterfactual(
            id=f"cf_{uuid.uuid4().hex[:8]}",
            basis_memory_id=item.id,
            original_content=content[:120],
            counterfactual_content=cf_content,
            direction=direction,
            valence_shift=valence_shift,
            plausibility=plausibility,
            domain=domain,
        )

    def _store_cf(self, cf: Counterfactual) -> None:
        """Persist counterfactual as a new memory."""
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=cf.counterfactual_content,
                domain="counterfactual",
                importance=0.5,
                emotional_valence=cf.valence_shift,
            )
            self.memory.store(exp)
        except Exception:
            pass

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
