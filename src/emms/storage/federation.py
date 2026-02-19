"""MemoryFederation — multi-agent snapshot merging.

Allows two or more EMMS instances to share memories by merging their
flat memory snapshots (list of MemoryItem) into a single target instance.

Conflict policies (when same memory_id or content-hash exists in both):
  - LOCAL_WINS    : keep the target's version
  - NEWEST_WINS   : keep whichever was stored most recently
  - IMPORTANCE_WINS : keep whichever has higher importance

Additionally, near-duplicate content (same content hash) is deduplicated.
A namespace_prefix can be prepended to incoming ids to avoid collisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from emms.core.models import MemoryItem, MemoryTier
from emms.storage.index import CompactionIndex, _content_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / models
# ---------------------------------------------------------------------------

class ConflictPolicy(str, Enum):
    LOCAL_WINS = "local_wins"
    NEWEST_WINS = "newest_wins"
    IMPORTANCE_WINS = "importance_wins"


@dataclass
class ConflictEntry:
    """Records a single conflict that was resolved during merge."""
    memory_id: str
    content_excerpt: str
    local_importance: float
    remote_importance: float
    local_stored_at: float
    remote_stored_at: float
    resolution: str          # "kept_local" | "kept_remote"
    policy_used: str


@dataclass
class FederationResult:
    """Summary of a completed merge operation."""
    items_in_source: int
    items_merged: int           # net new items added to target
    items_skipped_duplicate: int
    items_skipped_conflict_lost: int
    conflicts: list[ConflictEntry]
    namespaced: int             # items that got a prefix applied
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"Source items: {self.items_in_source}",
            f"Merged: {self.items_merged}",
            f"Skipped (duplicate): {self.items_skipped_duplicate}",
            f"Skipped (conflict, local won): {self.items_skipped_conflict_lost}",
            f"Conflicts resolved: {len(self.conflicts)}",
            f"Namespaced: {self.namespaced}",
            f"Duration: {self.duration_seconds:.3f}s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryFederation
# ---------------------------------------------------------------------------

class MemoryFederation:
    """Merge a foreign memory snapshot into a local EMMS instance.

    Parameters
    ----------
    target : EMMS
        The local EMMS instance to receive merged memories.
    policy : ConflictPolicy
        Strategy for resolving id or content-hash collisions.
    namespace_prefix : str | None
        If set, prepend "prefix/" to all incoming memory ids to avoid
        id-space collisions.  Content hashes are still compared globally.
    merge_graph : bool
        If True, also merge entities and relationships from source GraphMemory
        into target's graph (when both graphs are enabled).
    """

    def __init__(
        self,
        target: Any,
        policy: ConflictPolicy = ConflictPolicy.NEWEST_WINS,
        namespace_prefix: str | None = None,
        merge_graph: bool = True,
    ) -> None:
        self.target = target
        self.policy = policy
        self.namespace_prefix = namespace_prefix
        self.merge_graph = merge_graph

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def merge_from(self, source: Any) -> FederationResult:
        """Merge all memories from *source* EMMS into *target*.

        Parameters
        ----------
        source : EMMS
            Another EMMS instance.  Only its memory items are read.
        """
        t0 = time.time()
        source_items = self._collect_from(source)

        # Build index of target's current items
        local_index = CompactionIndex()
        local_index.rebuild_from(self.target.memory)

        merged = 0
        skipped_dup = 0
        skipped_conflict_lost = 0
        conflicts: list[ConflictEntry] = []
        namespaced = 0

        for item in source_items:
            # Apply namespace prefix to id
            original_id = item.id
            if self.namespace_prefix:
                if not item.id.startswith(self.namespace_prefix + "/"):
                    item = item.model_copy(
                        update={"id": f"{self.namespace_prefix}/{item.id}"}
                    )
                    namespaced += 1

            item_content = item.experience.content if item.experience else ""
            item_importance = item.experience.importance if item.experience else 0.5

            # Check content-hash deduplication first (strongest signal)
            local_matches = local_index.find_by_content(item_content)
            if local_matches:
                # near-duplicate exists — skip
                skipped_dup += 1
                continue

            # Check id collision
            local_existing = local_index.get_by_id(item.id)
            if local_existing is not None:
                winner, resolution = self._resolve_conflict(local_existing, item)
                local_content = local_existing.experience.content if local_existing.experience else ""
                local_importance = local_existing.experience.importance if local_existing.experience else 0.5
                entry = ConflictEntry(
                    memory_id=item.id,
                    content_excerpt=item_content[:80],
                    local_importance=local_importance,
                    remote_importance=item_importance,
                    local_stored_at=local_existing.stored_at,
                    remote_stored_at=item.stored_at,
                    resolution=resolution,
                    policy_used=self.policy.value,
                )
                conflicts.append(entry)
                if winner == "local":
                    skipped_conflict_lost += 1
                    continue
                # remote wins → remove local, insert remote below
                local_index.remove(local_existing.id)

            # Insert into target memory
            self._insert_item(item)
            local_index.register(item)
            merged += 1

        # Optionally merge graph entities/relationships
        if self.merge_graph:
            self._merge_graph(source)

        return FederationResult(
            items_in_source=len(source_items),
            items_merged=merged,
            items_skipped_duplicate=skipped_dup,
            items_skipped_conflict_lost=skipped_conflict_lost,
            conflicts=conflicts,
            namespaced=namespaced,
            duration_seconds=time.time() - t0,
        )

    def export_snapshot(self) -> list[MemoryItem]:
        """Return all MemoryItems from target as a flat list (for sharing)."""
        return self._collect_from(self.target)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_from(self, emms_instance: Any) -> list[MemoryItem]:
        """Flatten all 4 tiers of hierarchical memory."""
        items: list[MemoryItem] = []
        mem = emms_instance.memory
        for tier_store in (mem.working, mem.short_term):
            items.extend(tier_store)
        for tier_store in (mem.long_term, mem.semantic):
            items.extend(tier_store.values())
        return items

    def _resolve_conflict(
        self, local: MemoryItem, remote: MemoryItem
    ) -> tuple[str, str]:
        """Return ("local"|"remote", description)."""
        if self.policy == ConflictPolicy.LOCAL_WINS:
            return "local", "kept_local"
        if self.policy == ConflictPolicy.NEWEST_WINS:
            if remote.stored_at > local.stored_at:
                return "remote", "kept_remote"
            return "local", "kept_local"
        if self.policy == ConflictPolicy.IMPORTANCE_WINS:
            remote_imp = remote.experience.importance if remote.experience else 0.5
            local_imp = local.experience.importance if local.experience else 0.5
            if remote_imp > local_imp:
                return "remote", "kept_remote"
            return "local", "kept_local"
        return "local", "kept_local"

    def _insert_item(self, item: MemoryItem) -> None:
        """Insert a MemoryItem into the appropriate tier of the target memory.

        deque tiers use maxlen so capacity is enforced automatically.
        """
        mem = self.target.memory
        tier = item.tier
        if tier == MemoryTier.WORKING:
            mem.working.append(item)  # deque maxlen handles eviction
        elif tier == MemoryTier.SHORT_TERM:
            mem.short_term.append(item)  # deque maxlen handles eviction
        elif tier == MemoryTier.LONG_TERM:
            mem.long_term[item.id] = item
        else:  # SEMANTIC
            mem.semantic[item.id] = item

    def _merge_graph(self, source: Any) -> None:
        """Copy entities and relationships from source graph to target graph."""
        src_graph = getattr(source, "graph", None)
        tgt_graph = getattr(self.target, "graph", None)
        if src_graph is None or tgt_graph is None:
            return

        # Merge entities
        src_entities: dict[str, Any] = getattr(src_graph, "_entities", {})
        tgt_entities: dict[str, Any] = getattr(tgt_graph, "_entities", {})
        for name, entity in src_entities.items():
            if name not in tgt_entities:
                tgt_entities[name] = entity
                # update adjacency
                if not hasattr(tgt_graph, "_adj"):
                    tgt_graph._adj = {}
                tgt_graph._adj.setdefault(name, set())
            else:
                # merge mentions
                tgt_entities[name].mentions += entity.mentions

        # Merge relationships
        src_rels: dict[tuple, Any] = getattr(src_graph, "_rel_index", {})
        tgt_rels: dict[tuple, Any] = getattr(tgt_graph, "_rel_index", {})
        for key, rel in src_rels.items():
            if key not in tgt_rels:
                tgt_rels[key] = rel
                # update adjacency
                tgt_graph._adj.setdefault(rel.source, set()).add(rel.target)
                tgt_graph._adj.setdefault(rel.target, set()).add(rel.source)
            else:
                # strengthen existing
                tgt_rels[key].strength = min(
                    1.0, (tgt_rels[key].strength + rel.strength) / 2.0 + 0.05
                )
