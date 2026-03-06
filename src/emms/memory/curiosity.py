"""CuriosityEngine — epistemic curiosity and knowledge-gap detection.

v0.16.0: The Curious Mind

A well-functioning mind is not just passive — it actively notices what it
doesn't know and generates questions worth exploring. The CuriosityEngine
models epistemic curiosity as a first-class cognitive process: it scans
the memory store for knowledge gaps (sparse domains, low-confidence areas,
contradiction hotspots), ranks them by urgency, and generates structured
ExplorationGoal objects that the agent can use to guide future learning.

Gap types
---------
``sparse``        — domain has few memories (below sparse_threshold)
``uncertain``     — domain has many low-confidence memories
``contradictory`` — domain has detected belief conflicts
``novel``         — domain seen only once; breadth needed

Biological analogue: information-gap theory of curiosity (Loewenstein 1994)
— curiosity arises from a perceived gap between what is known and what one
feels should be known. The stronger the felt gap, the stronger the drive
to close it. Curiosity also predicts memory formation: curious states
enhance hippocampal encoding of subsequently learned information
(Gruber, Gelman & Ranganath 2014).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# Question templates per gap type
# ---------------------------------------------------------------------------

_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "sparse": [
        "What are the foundational concepts in {domain} that have not yet been captured?",
        "What key facts about {domain} are still missing from the knowledge base?",
        "Which aspects of {domain} deserve deeper investigation?",
    ],
    "uncertain": [
        "Which beliefs in {domain} have the weakest evidentiary support?",
        "What would increase confidence in the {domain} knowledge base?",
        "What additional evidence is needed to solidify {domain} understanding?",
    ],
    "contradictory": [
        "What is the correct resolution to the detected contradictions in {domain}?",
        "Which conflicting {domain} beliefs can be reconciled, and how?",
        "What evidence would settle the open disputes in {domain}?",
    ],
    "novel": [
        "What breadth of {domain} knowledge is still unexplored?",
        "What adjacent concepts should be learned to build richer {domain} understanding?",
        "How does {domain} connect to other domains in the knowledge base?",
    ],
}

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExplorationGoal:
    """A curiosity-driven question worth exploring."""

    id: str
    question: str                # Human-readable exploration question
    domain: str
    urgency: float               # 0..1 — how important to explore
    gap_type: str                # "sparse" | "uncertain" | "contradictory" | "novel"
    supporting_memory_ids: list[str]  # Memories that motivated this question
    created_at: float = field(default_factory=time.time)
    explored: bool = False

    def summary(self) -> str:
        return (
            f"ExplorationGoal [{self.domain}] urgency={self.urgency:.2f} "
            f"type={self.gap_type}\n"
            f"  Q: {self.question}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "question": self.question,
            "domain": self.domain,
            "urgency": self.urgency,
            "gap_type": self.gap_type,
            "supporting_memory_ids": list(self.supporting_memory_ids),
            "created_at": self.created_at,
            "explored": self.explored,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExplorationGoal":
        """Deserialize from a plain dict."""
        return cls(
            id=d["id"],
            question=d["question"],
            domain=d["domain"],
            urgency=d["urgency"],
            gap_type=d["gap_type"],
            supporting_memory_ids=list(d.get("supporting_memory_ids", [])),
            created_at=d.get("created_at", time.time()),
            explored=d.get("explored", False),
        )


@dataclass
class CuriosityReport:
    """Result of a CuriosityEngine.scan() call."""

    total_domains_scanned: int
    goals_generated: int
    goals: list[ExplorationGoal]       # Sorted by urgency descending
    top_curious_domains: list[str]     # Domains with highest curiosity scores
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"CuriosityReport: {self.goals_generated} goals from "
            f"{self.total_domains_scanned} domains in {self.duration_seconds:.2f}s",
            f"  Top curious domains: {', '.join(self.top_curious_domains[:5])}",
        ]
        for g in self.goals[:5]:
            lines.append(f"  [{g.domain}] urgency={g.urgency:.2f} ({g.gap_type}): "
                         f"{g.question[:70]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CuriosityEngine
# ---------------------------------------------------------------------------

class CuriosityEngine:
    """Detects knowledge gaps and generates exploration questions.

    For each domain the engine computes a curiosity score based on:

    - **Sparsity** — fewer than ``sparse_threshold`` memories → high curiosity
    - **Uncertainty** — mean confidence below ``uncertain_threshold``
    - **Contradictions** — overlapping memories with opposing valence
    - **Novelty** — domain appears only once

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    metacognition_engine:
        Optional :class:`MetacognitionEngine` for accurate confidence scores.
    max_goals:
        Maximum :class:`ExplorationGoal` objects to generate (default 10).
    sparse_threshold:
        Domain with fewer than this many memories is considered sparse
        (default 3).
    uncertain_threshold:
        Mean confidence below this triggers an uncertainty goal (default 0.4).
    """

    def __init__(
        self,
        memory: Any,
        metacognition_engine: Optional[Any] = None,
        max_goals: int = 10,
        sparse_threshold: int = 3,
        uncertain_threshold: float = 0.4,
    ) -> None:
        self.memory = memory
        self.metacognition_engine = metacognition_engine
        self.max_goals = max_goals
        self.sparse_threshold = sparse_threshold
        self.uncertain_threshold = uncertain_threshold
        self._goals: dict[str, ExplorationGoal] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, domain: Optional[str] = None) -> CuriosityReport:
        """Scan memory for knowledge gaps and generate exploration goals.

        Args:
            domain: Restrict scan to one domain (``None`` = all domains).

        Returns:
            :class:`CuriosityReport` with generated goals sorted by urgency.
        """
        t0 = time.time()
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        # Get confidence map if metacognition available
        conf_map: dict[str, float] = {}
        if self.metacognition_engine is not None:
            for mc in self.metacognition_engine.assess_all():
                conf_map[mc.memory_id] = mc.confidence

        # Generate goals per domain
        all_goals: list[ExplorationGoal] = []
        domain_scores: dict[str, float] = {}

        for dom, dom_items in by_domain.items():
            goals, score = self._scan_domain(dom, dom_items, conf_map)
            all_goals.extend(goals)
            domain_scores[dom] = score

        # Sort by urgency descending, take top-k
        all_goals.sort(key=lambda g: g.urgency, reverse=True)
        all_goals = all_goals[: self.max_goals]

        # Store goals for later retrieval
        for goal in all_goals:
            self._goals[goal.id] = goal

        # Top curious domains
        top_domains = sorted(domain_scores, key=lambda d: domain_scores[d], reverse=True)[:5]

        return CuriosityReport(
            total_domains_scanned=len(by_domain),
            goals_generated=len(all_goals),
            goals=all_goals,
            top_curious_domains=top_domains,
            duration_seconds=time.time() - t0,
        )

    def pending_goals(self) -> list[ExplorationGoal]:
        """Return all un-explored goals sorted by urgency descending.

        Returns:
            List of :class:`ExplorationGoal` not yet marked as explored.
        """
        return sorted(
            (g for g in self._goals.values() if not g.explored),
            key=lambda g: g.urgency,
            reverse=True,
        )

    def mark_explored(self, goal_id: str) -> bool:
        """Mark an exploration goal as fulfilled.

        Args:
            goal_id: ID of the goal to mark.

        Returns:
            ``True`` if found and marked; ``False`` if not found.
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.explored = True
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Union[str, Path]) -> None:
        """Persist ``_goals`` to *path* using atomic write."""
        path = Path(path)
        state = {
            gid: goal.to_dict()
            for gid, goal in self._goals.items()
        }
        data = json.dumps(state, indent=2, default=str).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        closed = False
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            closed = True
            os.replace(tmp, path)
        except BaseException:
            if not closed:
                os.close(fd)
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def load_state(self, path: Union[str, Path]) -> None:
        """Restore ``_goals`` from *path* (no-op if file missing)."""
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
        self._goals = {
            gid: ExplorationGoal.from_dict(d)
            for gid, d in raw.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_domain(
        self,
        domain: str,
        items: list[Any],
        conf_map: dict[str, float],
    ) -> tuple[list[ExplorationGoal], float]:
        """Scan one domain and return (goals, curiosity_score)."""
        n = len(items)
        goals: list[ExplorationGoal] = []
        score = 0.0

        # --- Sparsity gap ---
        if n == 0:
            return [], 0.0
        if n == 1:
            gap_type = "novel"
            urgency = 0.75
            score += urgency
            goals.append(self._make_goal(domain, gap_type, urgency, items[:1]))
        elif n < self.sparse_threshold:
            gap_type = "sparse"
            urgency = 0.5 + 0.1 * (self.sparse_threshold - n)
            score += urgency
            goals.append(self._make_goal(domain, gap_type, urgency, items))

        # --- Uncertainty gap ---
        confs = [conf_map.get(it.id, it.memory_strength) for it in items]
        mean_conf = sum(confs) / len(confs)
        if mean_conf < self.uncertain_threshold:
            gap_type = "uncertain"
            urgency = max(0.3, 1.0 - mean_conf)
            score += urgency * 0.8
            goals.append(self._make_goal(domain, gap_type, urgency, items))

        # --- Contradiction gap ---
        n_contradictions = self._count_contradictions(items)
        if n_contradictions >= 1:
            gap_type = "contradictory"
            urgency = min(1.0, 0.5 + 0.15 * n_contradictions)
            score += urgency
            goals.append(self._make_goal(domain, gap_type, urgency, items))

        return goals, score

    def _make_goal(
        self,
        domain: str,
        gap_type: str,
        urgency: float,
        items: list[Any],
    ) -> ExplorationGoal:
        """Create an ExplorationGoal from templates."""
        templates = _QUESTION_TEMPLATES.get(gap_type, _QUESTION_TEMPLATES["sparse"])
        # Pick template based on number of items
        idx = min(len(items) // 2, len(templates) - 1)
        question = templates[idx].format(domain=domain)
        return ExplorationGoal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            question=question,
            domain=domain,
            urgency=round(min(1.0, urgency), 4),
            gap_type=gap_type,
            supporting_memory_ids=[it.id for it in items[:5]],
        )

    def _count_contradictions(self, items: list[Any]) -> int:
        """Count memory pairs with semantic overlap but opposing valence."""
        count = 0
        for i, a in enumerate(items):
            va = getattr(a.experience, "emotional_valence", 0.0) or 0.0
            for b in items[i + 1:]:
                vb = getattr(b.experience, "emotional_valence", 0.0) or 0.0
                if abs(va - vb) >= 0.5 and self._token_overlap(
                    a.experience.content, b.experience.content
                ) >= 0.25:
                    count += 1
        return count

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        def tokens(t: str) -> set[str]:
            return {
                w.strip(".,!?;:\"'()")
                for w in t.lower().split()
                if len(w) >= 4 and w not in _STOP_WORDS
            }
        a, b = tokens(text_a), tokens(text_b)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
