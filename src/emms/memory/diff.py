"""MemoryDiff — session-to-session memory snapshot comparison.

Compare two ``HierarchicalMemory`` states (or saved JSON snapshots) and
produce a structured diff: which items were added, removed, strengthened,
weakened, or superseded between them.

Usage::

    from emms import EMMS
    from emms.memory.diff import MemoryDiff

    agent = EMMS()
    # ... store experiences ...

    # Take a snapshot before a work session
    agent.save("before.json")

    # ... more work ...

    agent.save("after.json")

    diff = MemoryDiff.from_paths("before.json", "after.json")
    print(diff.summary())
    diff.export_markdown("changes.md")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.memory.hierarchical import HierarchicalMemory


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ItemSnapshot:
    """Lightweight snapshot of a MemoryItem for diff purposes."""
    id: str
    experience_id: str
    content: str
    domain: str
    tier: str
    importance: float
    memory_strength: float
    access_count: int
    stored_at: float
    superseded_by: str | None = None
    title: str | None = None


@dataclass
class DiffResult:
    """Structured diff between two memory snapshots.

    Attributes
    ----------
    added : list[ItemSnapshot]
        Items present in snapshot_b but not in snapshot_a.
    removed : list[ItemSnapshot]
        Items present in snapshot_a but not in snapshot_b.
    strengthened : list[tuple[ItemSnapshot, ItemSnapshot]]
        Items whose ``memory_strength`` increased by more than *threshold*.
        Each tuple is (before, after).
    weakened : list[tuple[ItemSnapshot, ItemSnapshot]]
        Items whose ``memory_strength`` decreased by more than *threshold*.
        Each tuple is (before, after).
    superseded : list[tuple[ItemSnapshot, ItemSnapshot]]
        Items that were not superseded in snapshot_a but are in snapshot_b.
        Each tuple is (original, replacement_snapshot).
    snapshot_a_time : float
        ``saved_at`` timestamp of the first snapshot (0 if unknown).
    snapshot_b_time : float
        ``saved_at`` timestamp of the second snapshot (0 if unknown).
    strength_threshold : float
        Minimum delta to count as strengthened/weakened.
    """

    added: list[ItemSnapshot] = field(default_factory=list)
    removed: list[ItemSnapshot] = field(default_factory=list)
    strengthened: list[tuple[ItemSnapshot, ItemSnapshot]] = field(default_factory=list)
    weakened: list[tuple[ItemSnapshot, ItemSnapshot]] = field(default_factory=list)
    superseded: list[tuple[ItemSnapshot, ItemSnapshot]] = field(default_factory=list)
    snapshot_a_time: float = 0.0
    snapshot_b_time: float = 0.0
    strength_threshold: float = 0.05

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-paragraph human-readable summary."""
        lines = [
            f"Memory diff: {len(self.added)} added, {len(self.removed)} removed, "
            f"{len(self.strengthened)} strengthened, {len(self.weakened)} weakened, "
            f"{len(self.superseded)} superseded.",
        ]
        if self.snapshot_a_time and self.snapshot_b_time:
            elapsed = self.snapshot_b_time - self.snapshot_a_time
            lines.append(f"Time elapsed: {elapsed:.1f}s between snapshots.")
        return "\n".join(lines)

    def export_markdown(self, path: str | Path | None = None) -> str:
        """Render diff as Markdown.

        Parameters
        ----------
        path : optional file path to write the output.

        Returns
        -------
        str : Markdown text (also written to ``path`` if provided).
        """
        lines: list[str] = ["# Memory Diff Report\n"]

        from datetime import datetime
        if self.snapshot_a_time:
            lines.append(f"**Snapshot A**: {datetime.fromtimestamp(self.snapshot_a_time).isoformat()}")
        if self.snapshot_b_time:
            lines.append(f"**Snapshot B**: {datetime.fromtimestamp(self.snapshot_b_time).isoformat()}")
        lines.append("")

        lines.append(f"| Change | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Added | {len(self.added)} |")
        lines.append(f"| Removed | {len(self.removed)} |")
        lines.append(f"| Strengthened | {len(self.strengthened)} |")
        lines.append(f"| Weakened | {len(self.weakened)} |")
        lines.append(f"| Superseded | {len(self.superseded)} |")
        lines.append("")

        if self.added:
            lines.append("## Added\n")
            for item in self.added:
                title = item.title or item.content[:60]
                lines.append(f"- **[{item.domain}]** {title} `(id: {item.id[:12]})`")
            lines.append("")

        if self.removed:
            lines.append("## Removed\n")
            for item in self.removed:
                title = item.title or item.content[:60]
                lines.append(f"- **[{item.domain}]** {title} `(id: {item.id[:12]})`")
            lines.append("")

        if self.strengthened:
            lines.append("## Strengthened\n")
            for before, after in self.strengthened:
                delta = after.memory_strength - before.memory_strength
                title = after.title or after.content[:50]
                lines.append(f"- **[{after.domain}]** {title} — strength +{delta:.3f} "
                              f"({before.memory_strength:.3f} → {after.memory_strength:.3f})")
            lines.append("")

        if self.weakened:
            lines.append("## Weakened\n")
            for before, after in self.weakened:
                delta = before.memory_strength - after.memory_strength
                title = after.title or after.content[:50]
                lines.append(f"- **[{after.domain}]** {title} — strength -{delta:.3f} "
                              f"({before.memory_strength:.3f} → {after.memory_strength:.3f})")
            lines.append("")

        if self.superseded:
            lines.append("## Superseded\n")
            for original, _ in self.superseded:
                title = original.title or original.content[:50]
                lines.append(f"- **[{original.domain}]** {title} "
                              f"→ superseded by `{original.superseded_by}`")
            lines.append("")

        md = "\n".join(lines)
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
        return md


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------

def _load_snapshot(data: dict[str, Any]) -> tuple[dict[str, ItemSnapshot], float]:
    """Parse a HierarchicalMemory JSON file into {mem_id: ItemSnapshot}.

    Returns (snapshots, saved_at_timestamp).
    """
    saved_at = data.get("saved_at", 0.0)
    result: dict[str, ItemSnapshot] = {}

    # Items are stored under each tier key as lists of MemoryItem dicts
    for tier_key in ("working", "short_term", "long_term", "semantic"):
        for raw in data.get(tier_key, []):
            exp = raw.get("experience", {})
            snap = ItemSnapshot(
                id=raw.get("id", ""),
                experience_id=exp.get("id", ""),
                content=exp.get("content", ""),
                domain=exp.get("domain", "general"),
                tier=tier_key,
                importance=exp.get("importance", 0.5),
                memory_strength=raw.get("memory_strength", 1.0),
                access_count=raw.get("access_count", 0),
                stored_at=raw.get("stored_at", 0.0),
                superseded_by=raw.get("superseded_by"),
                title=exp.get("title"),
            )
            result[snap.id] = snap

    return result, saved_at


# ---------------------------------------------------------------------------
# MemoryDiff
# ---------------------------------------------------------------------------

class MemoryDiff:
    """Compute structured diffs between two memory states.

    Use ``MemoryDiff.from_paths()`` to compare two saved JSON snapshots or
    ``MemoryDiff.from_memories()`` to compare two live ``HierarchicalMemory``
    instances.
    """

    @staticmethod
    def diff(
        snapshot_a: dict[str, ItemSnapshot],
        snapshot_b: dict[str, ItemSnapshot],
        snapshot_a_time: float = 0.0,
        snapshot_b_time: float = 0.0,
        strength_threshold: float = 0.05,
    ) -> DiffResult:
        """Core diff algorithm.

        Parameters
        ----------
        snapshot_a, snapshot_b : dicts mapping mem_id → ItemSnapshot.
        snapshot_a_time, snapshot_b_time : Unix timestamps of the snapshots.
        strength_threshold : minimum strength delta to count as changed.
        """
        ids_a = set(snapshot_a)
        ids_b = set(snapshot_b)

        added = [snapshot_b[i] for i in ids_b - ids_a]
        removed = [snapshot_a[i] for i in ids_a - ids_b]

        strengthened: list[tuple[ItemSnapshot, ItemSnapshot]] = []
        weakened: list[tuple[ItemSnapshot, ItemSnapshot]] = []
        superseded: list[tuple[ItemSnapshot, ItemSnapshot]] = []

        for id_ in ids_a & ids_b:
            before = snapshot_a[id_]
            after = snapshot_b[id_]

            delta = after.memory_strength - before.memory_strength
            if delta > strength_threshold:
                strengthened.append((before, after))
            elif delta < -strength_threshold:
                weakened.append((before, after))

            # Detect newly superseded
            if before.superseded_by is None and after.superseded_by is not None:
                superseded.append((after, snapshot_b.get(after.superseded_by, after)))

        return DiffResult(
            added=sorted(added, key=lambda x: x.stored_at),
            removed=sorted(removed, key=lambda x: x.stored_at),
            strengthened=sorted(strengthened, key=lambda p: abs(p[1].memory_strength - p[0].memory_strength), reverse=True),
            weakened=sorted(weakened, key=lambda p: abs(p[1].memory_strength - p[0].memory_strength), reverse=True),
            superseded=superseded,
            snapshot_a_time=snapshot_a_time,
            snapshot_b_time=snapshot_b_time,
            strength_threshold=strength_threshold,
        )

    @staticmethod
    def from_paths(
        path_a: str | Path,
        path_b: str | Path,
        strength_threshold: float = 0.05,
    ) -> DiffResult:
        """Compare two saved JSON memory snapshots.

        Parameters
        ----------
        path_a : path to the first (earlier) snapshot.
        path_b : path to the second (later) snapshot.
        strength_threshold : minimum strength delta to count as changed.
        """
        pa, pb = Path(path_a), Path(path_b)
        data_a = json.loads(pa.read_text(encoding="utf-8"))
        data_b = json.loads(pb.read_text(encoding="utf-8"))

        snap_a, time_a = _load_snapshot(data_a)
        snap_b, time_b = _load_snapshot(data_b)

        return MemoryDiff.diff(snap_a, snap_b, time_a, time_b, strength_threshold)

    @staticmethod
    def from_memories(
        memory_a: "HierarchicalMemory",
        memory_b: "HierarchicalMemory",
        strength_threshold: float = 0.05,
    ) -> DiffResult:
        """Compare two live HierarchicalMemory instances."""

        def _snap(mem: "HierarchicalMemory") -> dict[str, ItemSnapshot]:
            result: dict[str, ItemSnapshot] = {}
            for tier_items in (
                list(mem.working),
                list(mem.short_term),
                list(mem.long_term),
                list(mem.semantic),
            ):
                for item in tier_items:
                    snap = ItemSnapshot(
                        id=item.id,
                        experience_id=item.experience.id,
                        content=item.experience.content,
                        domain=item.experience.domain,
                        tier=item.tier.value,
                        importance=item.experience.importance,
                        memory_strength=item.memory_strength,
                        access_count=item.access_count,
                        stored_at=item.stored_at,
                        superseded_by=item.superseded_by,
                        title=item.experience.title,
                    )
                    result[snap.id] = snap
            return result

        snap_a = _snap(memory_a)
        snap_b = _snap(memory_b)
        now = time.time()
        return MemoryDiff.diff(snap_a, snap_b, now, now, strength_threshold)
