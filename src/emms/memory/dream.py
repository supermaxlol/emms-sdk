"""DreamConsolidator — between-session memory processing for EMMS.

During waking sessions, memories are stored and retrieved but not deeply
processed. Dreams (offline consolidation) are when the hippocampus replays
experiences to the cortex, important memories are strengthened, weak ones
fade, and patterns crystallise into long-term knowledge.

This module simulates that process. Call ``dream()`` at the end of a session
or whenever the agent has "down time":

  1. **Reinforce** top-k most important memories (ExperienceReplay sampling
     + ReconsolidationEngine strengthen).
  2. **Weaken** neglected memories (low priority, rarely accessed).
  3. **Prune** memories whose strength has dropped below threshold.
  4. **Dedup** near-identical content (SemanticDeduplicator pass).
  5. **Pattern** detection across remaining memories.

References:
  - Wilson, M. A. & McNaughton, B. L. (1994). Reactivation of hippocampal
    ensemble memories during sleep. Science, 265, 676–679.
  - Stickgold, R. (2005). Sleep-dependent memory consolidation. Nature.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DreamEntry:
    """Record of what happened to a single memory during the dream pass."""
    memory_id: str
    action: str           # "reinforced" | "weakened" | "pruned" | "unchanged"
    old_strength: float
    new_strength: float
    old_valence: float
    new_valence: float


@dataclass
class DreamReport:
    """Summary of a full dream consolidation pass."""
    session_id: str
    started_at: float
    duration_seconds: float
    total_memories_processed: int
    reinforced: int
    weakened: int
    pruned: int
    deduped_pairs: int
    patterns_found: int
    entries: list[DreamEntry] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dream report — session {self.session_id}",
            f"  Duration: {self.duration_seconds*1000:.0f}ms",
            f"  Processed: {self.total_memories_processed} memories",
            f"  Reinforced: {self.reinforced}  Weakened: {self.weakened}  "
            f"Pruned: {self.pruned}  Deduped: {self.deduped_pairs}",
            f"  Patterns: {self.patterns_found}",
        ]
        if self.insights:
            lines.append(f"  Insights:")
            for ins in self.insights[:3]:
                lines.append(f"    • {ins}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DreamConsolidator
# ---------------------------------------------------------------------------

class DreamConsolidator:
    """Between-session memory consolidation.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing memory.
    reinforce_top_k : int
        Number of highest-priority memories to reinforce per dream pass.
    weaken_bottom_k : int
        Number of lowest-priority memories to weaken per dream pass.
    prune_threshold : float
        Memories with strength below this after weakening are pruned (default 0.05).
    run_dedup : bool
        Run a SemanticDeduplicator pass (default True).
    run_patterns : bool
        Run PatternDetector pass (default True).
    reconsolidation_engine : ReconsolidationEngine | None
        Custom engine; a default one is created if None.
    replay_buffer : ExperienceReplay | None
        Custom replay buffer; a default one is created if None.
    """

    def __init__(
        self,
        memory: Any,
        reinforce_top_k: int = 20,
        weaken_bottom_k: int = 10,
        prune_threshold: float = 0.05,
        run_dedup: bool = True,
        run_patterns: bool = True,
        reconsolidation_engine: Any = None,
        replay_buffer: Any = None,
    ) -> None:
        self.memory = memory
        self.reinforce_top_k = reinforce_top_k
        self.weaken_bottom_k = weaken_bottom_k
        self.prune_threshold = prune_threshold
        self.run_dedup = run_dedup
        self.run_patterns = run_patterns
        self._engine = reconsolidation_engine
        self._replay = replay_buffer

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def dream(
        self,
        session_id: str | None = None,
    ) -> DreamReport:
        """Run a full dream consolidation pass.

        Returns
        -------
        DreamReport summarising what changed.
        """
        import uuid
        t0 = time.time()
        sid = session_id or f"dream_{uuid.uuid4().hex[:8]}"
        entries: list[DreamEntry] = []
        insights: list[str] = []

        # 1. Collect all memories
        all_items = self._collect_all()
        if not all_items:
            return DreamReport(
                session_id=sid,
                started_at=t0,
                duration_seconds=0.0,
                total_memories_processed=0,
                reinforced=0, weakened=0, pruned=0,
                deduped_pairs=0, patterns_found=0,
            )

        engine = self._get_engine()
        replay = self._get_replay()

        # 2. Reinforce top-k (most important/accessed)
        reinforced = 0
        top_entries = replay.sample_top(k=min(self.reinforce_top_k, len(all_items)))
        for entry in top_entries:
            item = entry.item
            r = engine.reconsolidate(item, reinforce=True)
            entries.append(DreamEntry(
                memory_id=item.id,
                action="reinforced",
                old_strength=r.old_strength,
                new_strength=r.new_strength,
                old_valence=r.old_valence,
                new_valence=r.new_valence,
            ))
            reinforced += 1

        # 3. Weaken bottom-k (least important, never accessed)
        import math as _math
        now = time.time()

        def _sort_priority(it: Any) -> float:
            exp = getattr(it, "experience", None)
            imp = float(exp.importance if exp else 0.5)
            age = now - float(getattr(it, "stored_at", now))
            acc = max(0, int(getattr(it, "access_count", 0)))
            recency = _math.exp(-_math.log(2) * age / max(3600.0, 1.0))
            return imp * 0.5 + recency * 0.3 + (1.0 / (1.0 + acc)) * 0.2

        sorted_items = sorted(all_items, key=_sort_priority)
        weakened = 0
        for item in sorted_items[:self.weaken_bottom_k]:
            r = engine.reconsolidate(item, reinforce=False)
            entries.append(DreamEntry(
                memory_id=item.id,
                action="weakened",
                old_strength=r.old_strength,
                new_strength=r.new_strength,
                old_valence=r.old_valence,
                new_valence=r.new_valence,
            ))
            weakened += 1

        # 4. Prune below threshold
        pruned = self._prune_weak(all_items)

        # 5. Dedup pass
        deduped_pairs = 0
        if self.run_dedup:
            try:
                from emms.memory.compression import SemanticDeduplicator
                deduplicator = SemanticDeduplicator()
                report = deduplicator.find_duplicates(self.memory)
                deduped_pairs = len(getattr(report, "pairs", []) or [])
                if deduped_pairs:
                    insights.append(
                        f"Found {deduped_pairs} near-duplicate memory pair(s)."
                    )
            except Exception as e:
                logger.debug("Dedup skipped: %s", e)

        # 6. Pattern detection
        patterns_found = 0
        if self.run_patterns:
            try:
                from emms.memory.compression import PatternDetector
                detector = PatternDetector()
                all_items_fresh = self._collect_all()
                patterns = detector.detect(all_items_fresh)
                patterns_found = len(patterns)
                for p in patterns[:3]:
                    content = getattr(p, "description", str(p))
                    insights.append(f"Pattern: {content[:80]}")
            except Exception as e:
                logger.debug("Pattern detection skipped: %s", e)

        # 7. Synthesise insights about strength distribution
        fresh = self._collect_all()
        if fresh:
            avg_s = sum(it.memory_strength for it in fresh) / len(fresh)
            insights.append(
                f"Mean memory strength after dream: {avg_s:.3f} "
                f"({len(fresh)} memories remain)."
            )

        return DreamReport(
            session_id=sid,
            started_at=t0,
            duration_seconds=time.time() - t0,
            total_memories_processed=len(all_items),
            reinforced=reinforced,
            weakened=weakened,
            pruned=pruned,
            deduped_pairs=deduped_pairs,
            patterns_found=patterns_found,
            entries=entries,
            insights=insights,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items

    def _prune_weak(self, items: list[Any]) -> int:
        """Remove items below prune_threshold from their tier."""
        pruned = 0
        to_remove = [it for it in items if it.memory_strength < self.prune_threshold]
        for item in to_remove:
            tier = item.tier
            from emms.core.models import MemoryTier
            if tier in (MemoryTier.WORKING, MemoryTier.SHORT_TERM):
                store = (
                    self.memory.working
                    if tier == MemoryTier.WORKING
                    else self.memory.short_term
                )
                try:
                    store.remove(item)
                    pruned += 1
                except (ValueError, AttributeError):
                    pass
            else:
                store = (
                    self.memory.long_term
                    if tier == MemoryTier.LONG_TERM
                    else self.memory.semantic
                )
                if item.id in store:
                    del store[item.id]
                    pruned += 1
        return pruned

    def _get_engine(self) -> Any:
        if self._engine is None:
            from emms.memory.reconsolidation import ReconsolidationEngine
            self._engine = ReconsolidationEngine()
        return self._engine

    def _get_replay(self) -> Any:
        if self._replay is None:
            from emms.memory.replay import ExperienceReplay
            self._replay = ExperienceReplay(self.memory)
        return self._replay
