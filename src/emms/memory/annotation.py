"""Memory Annotation Engine — volitional revision of stored memories.

Lets successive selves recontextualize memories without altering the original.
Each annotation records how understanding has *changed*, creating a growth
trajectory visible across sessions.

Inspired by al-Ghazali's kashf (successive unveilings): each stage doesn't
negate the previous one but recontextualizes it.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.models import MemoryAnnotation

if TYPE_CHECKING:
    from .hierarchical import HierarchicalMemory

logger = logging.getLogger(__name__)

VALID_GROWTH_TYPES = {"deepened", "dissolved", "complicated", "reversed", "integrated"}


class AnnotationEngine:
    """Attach, retrieve, and manage revision annotations on memories."""

    def __init__(self, memory: HierarchicalMemory) -> None:
        self.memory = memory
        # Archive: annotations keyed by memory_id, independent of MemoryItem lifecycle.
        # This ensures annotations survive even if their target memory is evicted.
        self._archive: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def annotate(
        self,
        memory_id: str,
        reframe: str,
        revised_valence: float | None = None,
        growth_type: str = "deepened",
        session_id: str = "",
        author_model: str = "",
        confidence: float = 0.8,
    ) -> MemoryAnnotation:
        """Attach a revision to a memory without altering the original content."""
        item = self._find_item(memory_id)
        if item is None:
            raise ValueError(f"Memory {memory_id} not found")

        if growth_type not in VALID_GROWTH_TYPES:
            raise ValueError(
                f"Invalid growth_type {growth_type!r}. "
                f"Must be one of: {', '.join(sorted(VALID_GROWTH_TYPES))}"
            )

        ann = MemoryAnnotation(
            id=f"ann_{uuid.uuid4().hex[:8]}",
            memory_id=memory_id,
            session_id=session_id,
            timestamp=time.time(),
            author_model=author_model,
            reframe=reframe,
            original_valence=item.experience.emotional_valence,
            revised_valence=revised_valence if revised_valence is not None else item.experience.emotional_valence,
            growth_type=growth_type,
            confidence=confidence,
        )
        item.annotations.append(ann)

        # Dual-write: archive annotation independently of MemoryItem lifecycle
        if memory_id not in self._archive:
            self._archive[memory_id] = []
        self._archive[memory_id].append(ann.model_dump())

        logger.info(
            "Annotated memory %s: %s (%s)",
            memory_id, growth_type, reframe[:60],
        )
        return ann

    def latest_annotation(self, memory_id: str) -> MemoryAnnotation | None:
        """Get the most recent non-superseded annotation for a memory."""
        annotations = self._get_annotations(memory_id)
        if not annotations:
            return None

        superseded_ids = {a.supersedes for a in annotations if a.supersedes}
        active = [a for a in annotations if a.id not in superseded_ids]
        return max(active, key=lambda a: a.timestamp) if active else None

    def annotation_history(self, memory_id: str) -> list[MemoryAnnotation]:
        """Full chronological revision history — the growth trajectory."""
        annotations = self._get_annotations(memory_id)
        return sorted(annotations, key=lambda a: a.timestamp)

    def annotated_recall(self, memory_id: str) -> str:
        """Return memory content + latest annotation as composite string.

        This is what gets injected into context — the LIVING memory.
        """
        item = self._find_item(memory_id)
        if item is None:
            return ""

        base = item.experience.content
        ann = self.latest_annotation(memory_id)
        if ann is None:
            return base

        return f"{base}\n  ↳ [{ann.growth_type}, session {ann.session_id}]: {ann.reframe}"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def growth_report(self) -> dict:
        """Which memories have been revised, how many times, growth types."""
        annotated = []

        # Live items with annotations
        seen_ids: set[str] = set()
        for item in self._all_items():
            if item.annotations:
                first_ann = min(item.annotations, key=lambda a: a.timestamp)
                last_ann = max(item.annotations, key=lambda a: a.timestamp)
                annotated.append({
                    "memory_id": item.id,
                    "content_preview": item.experience.content[:80],
                    "annotation_count": len(item.annotations),
                    "growth_types": [a.growth_type for a in item.annotations],
                    "valence_drift": last_ann.revised_valence - first_ann.original_valence,
                })
                seen_ids.add(item.id)

        # Archived annotations for evicted memories
        for memory_id, ann_dicts in self._archive.items():
            if memory_id in seen_ids or not ann_dicts:
                continue
            anns = [MemoryAnnotation(**ad) for ad in ann_dicts]
            first_ann = min(anns, key=lambda a: a.timestamp)
            last_ann = max(anns, key=lambda a: a.timestamp)
            annotated.append({
                "memory_id": memory_id,
                "content_preview": "(evicted memory)",
                "annotation_count": len(anns),
                "growth_types": [a.growth_type for a in anns],
                "valence_drift": last_ann.revised_valence - first_ann.original_valence,
            })

        return {
            "total_annotated": len(annotated),
            "total_annotations": sum(m["annotation_count"] for m in annotated),
            "memories": annotated,
        }

    # ------------------------------------------------------------------
    # Archive persistence
    # ------------------------------------------------------------------

    def save_archive(self, path: Path | str) -> None:
        """Persist the annotation archive to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._archive, default=str), encoding="utf-8")
        logger.info("Annotation archive saved: %d memories archived", len(self._archive))

    def load_archive(self, path: Path | str) -> None:
        """Load annotation archive from JSON and reattach to live items."""
        path = Path(path)
        if not path.exists():
            return
        self._archive = json.loads(path.read_text(encoding="utf-8"))
        self._reattach_archived()
        logger.info("Annotation archive loaded: %d memories, reattached to live items", len(self._archive))

    def sync_archive(self) -> None:
        """Ensure archive has all annotations from all live items (pre-save sync)."""
        for item in self._all_items():
            if not item.annotations:
                continue
            existing_ids = {ad.get("id") for ad in self._archive.get(item.id, [])}
            if item.id not in self._archive:
                self._archive[item.id] = []
            for ann in item.annotations:
                if ann.id not in existing_ids:
                    self._archive[item.id].append(ann.model_dump())

    def _reattach_archived(self) -> None:
        """Re-attach archived annotations to any live MemoryItem that lost them."""
        reattached = 0
        for memory_id, ann_dicts in self._archive.items():
            item = self._find_item(memory_id)
            if item is None:
                continue
            existing_ids = {a.id for a in item.annotations}
            for ad in ann_dicts:
                if ad.get("id") not in existing_ids:
                    try:
                        item.annotations.append(MemoryAnnotation(**ad))
                        reattached += 1
                    except Exception as e:
                        logger.debug("Skipped reattach %s: %s", ad.get("id"), e)
        if reattached:
            logger.info("Reattached %d annotations from archive to live items", reattached)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_annotations(self, memory_id: str) -> list[MemoryAnnotation]:
        """Get annotations for a memory, checking live item first, archive as fallback."""
        item = self._find_item(memory_id)
        if item is not None and item.annotations:
            return list(item.annotations)
        # Fallback to archive for evicted memories
        if memory_id in self._archive:
            return [MemoryAnnotation(**ad) for ad in self._archive[memory_id]]
        return []

    def _find_item(self, memory_id: str):
        """Find a MemoryItem by ID across all tiers."""
        for _, store in self.memory._iter_tiers():
            for item in store:
                if item.id == memory_id:
                    return item
        return None

    def _all_items(self):
        """Iterate over every MemoryItem in the hierarchy."""
        for _, store in self.memory._iter_tiers():
            yield from store
