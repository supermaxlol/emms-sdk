"""BeliefReviser — systematic belief revision when contradictions arise.

v0.16.0: The Curious Mind

Intelligent agents don't simply accumulate beliefs — they update them when
new contradictory evidence arrives. BeliefReviser implements the AGM belief
revision framework in a practical, heuristic form: when a newly stored
memory conflicts with existing beliefs (high semantic overlap, opposing
emotional valence), the reviser creates a reconciled synthesis memory,
optionally supersedes the weaker conflicting belief, and logs the change
as a RevisionRecord.

Revision strategies
-------------------
``merge``      — both beliefs are plausible; a synthesised reconciliation
                 memory is created that acknowledges both perspectives
``supersede``  — new belief is clearly stronger; old belief is weakened
                 and marked as superseded_by the new memory
``flag``       — conflict detected but evidence insufficient to resolve;
                 both beliefs are flagged for human review

Biological analogue: belief revision and cognitive dissonance reduction
(Festinger 1957) — when new information conflicts with existing beliefs,
the cognitive system resolves the tension by updating, rationalising, or
compartmentalising. Cognitive flexibility requires willingness to revise;
rigidity leads to entrenchment of outdated beliefs.

AGM postulates (Alchourrón, Gärdenfors & Makinson 1985): the logic of
belief revision; revised belief set should be minimal and consistent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RevisionRecord:
    """Log entry for a single belief revision event."""

    id: str
    trigger_memory_id: str       # The new memory that triggered revision
    conflicting_memory_id: str   # The existing memory that conflicted
    revision_type: str           # "merge" | "supersede" | "flag"
    conflict_score: float        # How strong the conflict was (0..1)
    new_content: str             # Reconciled belief content (if merged)
    new_memory_id: Optional[str] # ID of newly created reconciliation memory
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"RevisionRecord [{self.revision_type}] conflict={self.conflict_score:.3f}\n"
            f"  trigger:     {self.trigger_memory_id[:16]}\n"
            f"  conflicting: {self.conflicting_memory_id[:16]}\n"
            f"  resolution:  {self.new_content[:80]}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger_memory_id": self.trigger_memory_id,
            "conflicting_memory_id": self.conflicting_memory_id,
            "revision_type": self.revision_type,
            "conflict_score": self.conflict_score,
            "new_content": self.new_content,
            "new_memory_id": self.new_memory_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RevisionRecord":
        return cls(
            id=d["id"],
            trigger_memory_id=d["trigger_memory_id"],
            conflicting_memory_id=d["conflicting_memory_id"],
            revision_type=d["revision_type"],
            conflict_score=d["conflict_score"],
            new_content=d["new_content"],
            new_memory_id=d.get("new_memory_id"),
            created_at=d.get("created_at", 0.0),
        )


@dataclass
class RevisionReport:
    """Result of a BeliefReviser.revise() run."""

    total_checked: int
    conflicts_found: int
    revisions_made: int
    records: list[RevisionRecord]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"RevisionReport: {self.total_checked} checked, "
            f"{self.conflicts_found} conflicts, {self.revisions_made} revised "
            f"in {self.duration_seconds:.2f}s",
        ]
        for r in self.records[:5]:
            lines.append(
                f"  [{r.revision_type}] conf={r.conflict_score:.2f}: "
                f"{r.new_content[:60]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BeliefReviser
# ---------------------------------------------------------------------------

class BeliefReviser:
    """Detects and resolves contradictions in the memory store.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    overlap_threshold:
        Minimum token Jaccard overlap for two memories to be considered
        about the same topic (default 0.3).
    valence_conflict_threshold:
        Minimum valence difference to qualify as a contradiction
        (default 0.4).
    merge_importance:
        Importance assigned to newly created merge memories (default 0.65).
    max_revisions:
        Maximum revisions per call (default 8).
    """

    def __init__(
        self,
        memory: Any,
        overlap_threshold: float = 0.30,
        valence_conflict_threshold: float = 0.40,
        merge_importance: float = 0.65,
        max_revisions: int = 8,
    ) -> None:
        self.memory = memory
        self.overlap_threshold = overlap_threshold
        self.valence_conflict_threshold = valence_conflict_threshold
        self.merge_importance = merge_importance
        self.max_revisions = max_revisions
        self._history: list[RevisionRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def revise(
        self,
        new_memory_id: Optional[str] = None,
        domain: Optional[str] = None,
        max_revisions: Optional[int] = None,
    ) -> RevisionReport:
        """Run a belief revision pass.

        If ``new_memory_id`` is given, only checks conflicts between that
        memory and all others in the same domain.  Otherwise scans all
        memories for pairwise conflicts.

        Args:
            new_memory_id: Trigger memory to check against the rest
                           (optional — full scan if omitted).
            domain:        Restrict scan to one domain (``None`` = all).
            max_revisions: Override instance default.

        Returns:
            :class:`RevisionReport` describing each revision action.
        """
        t0 = time.time()
        limit = max_revisions if max_revisions is not None else self.max_revisions
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Find the trigger item if specified
        trigger: Optional[Any] = None
        if new_memory_id:
            for it in items:
                if it.id == new_memory_id or it.experience.id == new_memory_id:
                    trigger = it
                    break

        # Find conflicts
        conflicts: list[tuple[Any, Any, float]] = self._find_conflicts(items, trigger)
        conflicts_found = len(conflicts)

        records: list[RevisionRecord] = []
        revised_ids: set[str] = set()

        for item_a, item_b, score in conflicts:
            if len(records) >= limit:
                break
            if item_a.id in revised_ids or item_b.id in revised_ids:
                continue  # Don't revise already-revised pair

            record = self._resolve(item_a, item_b, score)
            records.append(record)
            self._history.append(record)
            revised_ids.add(item_a.id)
            revised_ids.add(item_b.id)

        return RevisionReport(
            total_checked=len(items),
            conflicts_found=conflicts_found,
            revisions_made=len(records),
            records=records,
            duration_seconds=time.time() - t0,
        )

    def revision_history(self) -> list[RevisionRecord]:
        """Return all revision records from this session, newest first.

        Returns:
            List of :class:`RevisionRecord` sorted by ``created_at`` descending.
        """
        return sorted(self._history, key=lambda r: r.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path) -> None:
        import json, os, tempfile
        from pathlib import Path as _P
        path = _P(path)
        data = {"version": "0.16.0", "history": [r.to_dict() for r in self._history]}
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise

    def load_state(self, path) -> bool:
        import json
        from pathlib import Path as _P
        p = _P(path)
        if not p.exists(): return False
        try:
            data = json.loads(p.read_text())
            self._history = [RevisionRecord.from_dict(d) for d in data.get("history", [])]
            return True
        except Exception: return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_conflicts(
        self,
        items: list[Any],
        trigger: Optional[Any] = None,
    ) -> list[tuple[Any, Any, float]]:
        """Find conflicting memory pairs, sorted by conflict score descending."""
        conflicts: list[tuple[Any, Any, float]] = []

        if trigger is not None:
            # Only check trigger vs. all others
            va = getattr(trigger.experience, "emotional_valence", 0.0) or 0.0
            for other in items:
                if other.id == trigger.id:
                    continue
                vb = getattr(other.experience, "emotional_valence", 0.0) or 0.0
                valence_diff = abs(va - vb)
                if valence_diff < self.valence_conflict_threshold:
                    continue
                overlap = self._token_overlap(
                    trigger.experience.content, other.experience.content
                )
                if overlap >= self.overlap_threshold:
                    score = overlap * valence_diff
                    conflicts.append((trigger, other, score))
        else:
            # Full pairwise scan (limited to reasonable size)
            candidates = items[:60]  # Cap to avoid O(N²) explosion
            for i, a in enumerate(candidates):
                va = getattr(a.experience, "emotional_valence", 0.0) or 0.0
                for b in candidates[i + 1:]:
                    vb = getattr(b.experience, "emotional_valence", 0.0) or 0.0
                    valence_diff = abs(va - vb)
                    if valence_diff < self.valence_conflict_threshold:
                        continue
                    overlap = self._token_overlap(
                        a.experience.content, b.experience.content
                    )
                    if overlap >= self.overlap_threshold:
                        score = overlap * valence_diff
                        conflicts.append((a, b, score))

        conflicts.sort(key=lambda x: x[2], reverse=True)
        return conflicts

    def _resolve(
        self,
        item_a: Any,
        item_b: Any,
        conflict_score: float,
    ) -> RevisionRecord:
        """Choose and apply a revision strategy."""
        # Choose strategy based on relative strength
        strength_a = item_a.memory_strength
        strength_b = item_b.memory_strength
        strength_ratio = abs(strength_a - strength_b) / max(strength_a + strength_b, 1e-9)

        if conflict_score >= 0.5 and strength_ratio >= 0.3:
            # One belief clearly stronger — supersede
            if strength_a >= strength_b:
                stronger, weaker = item_a, item_b
            else:
                stronger, weaker = item_b, item_a
            return self._supersede(stronger, weaker, conflict_score)
        elif conflict_score >= 0.3:
            # Moderate conflict — merge into synthesis
            return self._merge(item_a, item_b, conflict_score)
        else:
            # Weak conflict — flag for review
            return self._flag(item_a, item_b, conflict_score)

    def _merge(
        self,
        item_a: Any,
        item_b: Any,
        conflict_score: float,
    ) -> RevisionRecord:
        """Create a synthesis memory reconciling both beliefs."""
        content_a = item_a.experience.content[:120]
        content_b = item_b.experience.content[:120]
        domain_a = getattr(item_a.experience, "domain", None) or "general"

        merged_content = (
            f"Reconciled belief: Both perspectives hold partial truth. "
            f"Perspective A: {content_a}. "
            f"Perspective B: {content_b}. "
            f"These views may reflect different contexts or framings."
        )

        new_id: Optional[str] = None
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=merged_content,
                domain=domain_a,
                importance=self.merge_importance,
                title="Reconciled belief (belief revision)",
                facts=["Merged from conflicting memories",
                       f"Conflict score: {conflict_score:.3f}"],
            )
            self.memory.store(exp)
            # Find the stored item's id
            for item in list(self.memory.working):
                if item.experience.id == exp.id:
                    new_id = item.id
                    break
        except Exception:
            pass

        return RevisionRecord(
            id=f"rev_{uuid.uuid4().hex[:8]}",
            trigger_memory_id=item_a.id,
            conflicting_memory_id=item_b.id,
            revision_type="merge",
            conflict_score=round(conflict_score, 4),
            new_content=merged_content,
            new_memory_id=new_id,
        )

    def _supersede(
        self,
        stronger: Any,
        weaker: Any,
        conflict_score: float,
    ) -> RevisionRecord:
        """Weaken the less-supported belief and mark it superseded."""
        weaker.memory_strength = max(0.0, weaker.memory_strength * 0.5)
        if hasattr(weaker.experience, "superseded_by"):
            weaker.experience.superseded_by = stronger.id

        return RevisionRecord(
            id=f"rev_{uuid.uuid4().hex[:8]}",
            trigger_memory_id=stronger.id,
            conflicting_memory_id=weaker.id,
            revision_type="supersede",
            conflict_score=round(conflict_score, 4),
            new_content=f"Weaker belief superseded by: {stronger.experience.content[:120]}",
            new_memory_id=None,
        )

    def _flag(
        self,
        item_a: Any,
        item_b: Any,
        conflict_score: float,
    ) -> RevisionRecord:
        """Flag both beliefs as conflicting without changing them."""
        return RevisionRecord(
            id=f"rev_{uuid.uuid4().hex[:8]}",
            trigger_memory_id=item_a.id,
            conflicting_memory_id=item_b.id,
            revision_type="flag",
            conflict_score=round(conflict_score, 4),
            new_content=(
                f"Flagged conflict between: '{item_a.experience.content[:60]}' "
                f"and '{item_b.experience.content[:60]}'"
            ),
            new_memory_id=None,
        )

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        stop = frozenset({
            "a", "an", "the", "and", "or", "in", "on", "to", "for",
            "of", "with", "by", "from", "is", "was", "are", "be",
        })

        def tokens(t: str) -> set[str]:
            return {
                w.strip(".,!?;:\"'()")
                for w in t.lower().split()
                if len(w) >= 4 and w not in stop
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
