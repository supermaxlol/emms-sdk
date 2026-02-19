"""CompactionIndex — O(1) lookup for MemoryItems by id, experience_id, or content hash.

Maintains three parallel dicts:
  by_id          : memory_id  → MemoryItem
  by_experience_id : experience_id → MemoryItem
  by_content_hash  : sha256(content[:200]) → list[MemoryItem]   (collision bucket)

The index is a pure helper; it does NOT own the memories.  Callers must
call register() / remove() in sync with the backing store.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterator

from emms.core.models import MemoryItem

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    """SHA-256 of the first 200 chars (lower-cased, stripped)."""
    normalised = content.strip().lower()[:200]
    return hashlib.sha256(normalised.encode()).hexdigest()


class CompactionIndex:
    """Fast O(1) lookup index over a HierarchicalMemory.

    Parameters
    ----------
    max_per_content_hash : int
        Collision-bucket cap per content hash.  Set to 0 for unlimited.
    """

    def __init__(self, max_per_content_hash: int = 100) -> None:
        self._by_id: dict[str, MemoryItem] = {}
        self._by_experience_id: dict[str, MemoryItem] = {}
        self._by_content_hash: dict[str, list[MemoryItem]] = {}
        self._max_per_hash = max_per_content_hash

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, item: MemoryItem) -> None:
        """Add or update an item in the index."""
        self._by_id[item.id] = item
        exp_id = item.experience.id if item.experience else None
        if exp_id:
            self._by_experience_id[exp_id] = item
        ch = _content_hash(item.experience.content if item.experience else "")
        bucket = self._by_content_hash.setdefault(ch, [])
        # update-in-place if already there
        for i, existing in enumerate(bucket):
            if existing.id == item.id:
                bucket[i] = item
                return
        if self._max_per_hash > 0 and len(bucket) >= self._max_per_hash:
            bucket.pop(0)  # evict oldest
        bucket.append(item)

    def remove(self, memory_id: str) -> bool:
        """Remove an item by its memory id.  Returns True if found."""
        item = self._by_id.pop(memory_id, None)
        if item is None:
            return False
        exp_id = item.experience.id if item.experience else None
        self._by_experience_id.pop(exp_id or "", None)
        ch = _content_hash(item.experience.content if item.experience else "")
        bucket = self._by_content_hash.get(ch, [])
        self._by_content_hash[ch] = [x for x in bucket if x.id != memory_id]
        if not self._by_content_hash[ch]:
            del self._by_content_hash[ch]
        return True

    def clear(self) -> None:
        """Reset the index."""
        self._by_id.clear()
        self._by_experience_id.clear()
        self._by_content_hash.clear()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, memory_id: str) -> MemoryItem | None:
        """O(1) lookup by memory id."""
        return self._by_id.get(memory_id)

    def get_by_experience_id(self, experience_id: str) -> MemoryItem | None:
        """O(1) lookup by experience id."""
        return self._by_experience_id.get(experience_id)

    def find_by_content(self, content: str) -> list[MemoryItem]:
        """Return all items whose content hash matches the given text snippet.

        The hash is computed on the first 200 chars (lower-cased, stripped),
        so this does exact-prefix-match deduplication rather than substring search.
        """
        ch = _content_hash(content)
        return list(self._by_content_hash.get(ch, []))

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self._by_id.values())

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def bulk_register(self, items: list[MemoryItem]) -> int:
        """Register many items at once.  Returns count registered."""
        for item in items:
            self.register(item)
        return len(items)

    def rebuild_from(self, memory: Any) -> int:
        """Rebuild from a HierarchicalMemory instance.

        Iterates all 4 tiers and re-registers every MemoryItem.  Returns
        the total count of items registered.
        """
        self.clear()
        count = 0
        # deque tiers
        for tier_store in (memory.working, memory.short_term):
            for item in tier_store:
                self.register(item)
                count += 1
        # dict tiers
        for tier_store in (memory.long_term, memory.semantic):
            for item in tier_store.values():
                self.register(item)
                count += 1
        logger.debug("CompactionIndex rebuilt: %d items", count)
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return a summary dict."""
        total_hash_buckets = sum(
            len(v) for v in self._by_content_hash.values()
        )
        return {
            "total_items": len(self._by_id),
            "by_experience_id": len(self._by_experience_id),
            "content_hash_buckets": len(self._by_content_hash),
            "total_in_hash_buckets": total_hash_buckets,
        }
