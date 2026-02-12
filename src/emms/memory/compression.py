"""Memory compression for long-term storage efficiency.

Compresses memory items using pattern extraction, abstraction, and deduplication.
Ported from the original EMMS.py research prototype and modernised for the SDK.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from emms.core.models import Experience, MemoryItem

logger = logging.getLogger(__name__)


class CompressedMemory(BaseModel):
    """A compressed representation of one or more memories."""

    id: str
    summary: str
    source_ids: list[str]
    domain: str = "general"
    importance: float = 0.5
    pattern_type: str = "single"  # single | merged | abstracted
    keywords: list[str] = Field(default_factory=list)
    compression_ratio: float = 1.0
    fidelity: float = 1.0  # reconstruction accuracy 0..1
    original_size: int = 0
    compressed_size: int = 0


class MemoryCompressor:
    """Compresses memories for efficient long-term storage.

    Three compression strategies:
    1. **Deduplication** — merge near-duplicate experiences
    2. **Summarisation** — extract key phrases and patterns
    3. **Abstraction** — generalise concrete details into abstract patterns

    Usage::

        compressor = MemoryCompressor()
        compressed = compressor.compress_batch(memory_items)
        expanded = compressor.expand(compressed[0])
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_summary_words: int = 8,
        max_summary_words: int = 50,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_summary_words = min_summary_words
        self.max_summary_words = max_summary_words
        self._originals: dict[str, str] = {}  # id → original content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, item: MemoryItem) -> CompressedMemory:
        """Compress a single memory item."""
        content = item.experience.content
        self._originals[item.id] = content

        summary = self._summarise(content)
        keywords = self._extract_keywords(content)

        return CompressedMemory(
            id=item.id,
            summary=summary,
            source_ids=[item.experience.id],
            domain=item.experience.domain,
            importance=item.experience.importance,
            pattern_type="single",
            keywords=keywords,
            compression_ratio=len(content) / max(len(summary), 1),
            fidelity=self._estimate_fidelity(content, summary),
            original_size=len(content),
            compressed_size=len(summary),
        )

    def compress_batch(self, items: list[MemoryItem]) -> list[CompressedMemory]:
        """Compress a batch, merging near-duplicates."""
        if not items:
            return []

        # Step 1: Group near-duplicates
        groups = self._find_duplicate_groups(items)

        results: list[CompressedMemory] = []
        for group in groups:
            if len(group) == 1:
                results.append(self.compress(group[0]))
            else:
                results.append(self._merge_group(group))

        return results

    def expand(self, compressed: CompressedMemory) -> str:
        """Reconstruct from compressed form (best effort)."""
        # If we still have the original cached, return it
        if compressed.id in self._originals:
            return self._originals[compressed.id]
        # Otherwise return the summary + keywords as reconstruction
        kw_str = ", ".join(compressed.keywords[:10])
        return f"{compressed.summary} [keywords: {kw_str}]"

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _find_duplicate_groups(
        self, items: list[MemoryItem]
    ) -> list[list[MemoryItem]]:
        """Group items by content similarity using fingerprinting."""
        assigned: set[int] = set()
        groups: list[list[MemoryItem]] = []

        for i, a in enumerate(items):
            if i in assigned:
                continue
            group = [a]
            assigned.add(i)

            for j in range(i + 1, len(items)):
                if j in assigned:
                    continue
                sim = self._content_similarity(
                    a.experience.content, items[j].experience.content
                )
                if sim >= self.similarity_threshold:
                    group.append(items[j])
                    assigned.add(j)

            groups.append(group)

        return groups

    def _content_similarity(self, a: str, b: str) -> float:
        """Fast Jaccard + character trigram similarity."""
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa or not wb:
            return 0.0

        jaccard = len(wa & wb) / len(wa | wb)

        # Trigram overlap for finer grain
        ta = {a.lower()[i:i+3] for i in range(len(a) - 2)}
        tb = {b.lower()[i:i+3] for i in range(len(b) - 2)}
        if ta and tb:
            trigram = len(ta & tb) / len(ta | tb)
        else:
            trigram = 0.0

        return jaccard * 0.6 + trigram * 0.4

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarise(self, text: str) -> str:
        """Extract key sentences via importance scoring."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text.strip()

        # Score each sentence
        scored = []
        all_words = set(text.lower().split())
        for i, sent in enumerate(sentences):
            words = set(sent.lower().split())
            # Coverage: what fraction of total vocabulary does this sentence cover?
            coverage = len(words & all_words) / max(len(all_words), 1)
            # Position: first and last sentences are more important
            position = 1.0 if i == 0 else (0.8 if i == len(sentences) - 1 else 0.5)
            # Length: prefer substantive sentences
            length_score = min(1.0, len(words) / 15)
            score = coverage * 0.4 + position * 0.3 + length_score * 0.3
            scored.append((score, i, sent))

        scored.sort(reverse=True)

        # Take top sentences up to word limit, preserving original order
        selected = []
        word_count = 0
        for _, idx, sent in scored:
            words_in_sent = len(sent.split())
            if word_count + words_in_sent > self.max_summary_words and selected:
                break
            selected.append((idx, sent))
            word_count += words_in_sent

        selected.sort(key=lambda x: x[0])
        summary = " ".join(s for _, s in selected)
        return summary

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str, max_keywords: int = 15) -> list[str]:
        """Extract keywords using TF-based scoring."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not words:
            return []

        # Simple stopword removal
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "have", "been", "were", "will", "with", "this", "that",
            "from", "they", "been", "said", "each", "which", "their",
            "about", "would", "there", "could", "other", "into",
            "more", "some", "than", "them", "very", "when", "what",
        }

        freq: dict[str, int] = {}
        for w in words:
            if w not in stopwords and len(w) > 2:
                freq[w] = freq.get(w, 0) + 1

        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:max_keywords]]

    # ------------------------------------------------------------------
    # Merge group
    # ------------------------------------------------------------------

    def _merge_group(self, group: list[MemoryItem]) -> CompressedMemory:
        """Merge near-duplicate items into one compressed memory."""
        # Pick the most important item as the representative
        group.sort(key=lambda m: m.experience.importance, reverse=True)
        representative = group[0]

        for item in group:
            self._originals[item.id] = item.experience.content

        # Combine all content for keyword extraction
        all_content = " ".join(m.experience.content for m in group)
        keywords = self._extract_keywords(all_content)
        summary = self._summarise(representative.experience.content)

        total_original = sum(len(m.experience.content) for m in group)

        return CompressedMemory(
            id=representative.id,
            summary=summary,
            source_ids=[m.experience.id for m in group],
            domain=representative.experience.domain,
            importance=max(m.experience.importance for m in group),
            pattern_type="merged",
            keywords=keywords,
            compression_ratio=total_original / max(len(summary), 1),
            fidelity=self._estimate_fidelity(
                representative.experience.content, summary
            ),
            original_size=total_original,
            compressed_size=len(summary),
        )

    # ------------------------------------------------------------------
    # Fidelity estimation
    # ------------------------------------------------------------------

    def _estimate_fidelity(self, original: str, summary: str) -> float:
        """Estimate how much information the summary preserves."""
        orig_words = set(original.lower().split())
        summ_words = set(summary.lower().split())
        if not orig_words:
            return 1.0
        # Keyword recall: what fraction of original words appear in summary
        recall = len(orig_words & summ_words) / len(orig_words)
        # Boost if summary is reasonably long
        length_ratio = min(1.0, len(summary) / max(len(original) * 0.3, 1))
        return min(1.0, recall * 0.7 + length_ratio * 0.3)


# ---------------------------------------------------------------------------
# Pattern detection — finds recurring patterns across memory items
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detects recurring patterns in memory items.

    Three pattern types:
    1. Sequence patterns — repeated temporal sequences
    2. Content patterns — recurring concepts/phrases
    3. Domain patterns — domain-level trends and shifts
    """

    def find_sequence_patterns(
        self, items: list[MemoryItem], min_pattern_length: int = 2
    ) -> dict[str, Any]:
        """Find repeated sequences of domains/themes."""
        if len(items) < min_pattern_length:
            return {"patterns": [], "count": 0}

        # Extract domain sequence
        domains = [item.experience.domain for item in items]
        patterns: dict[str, int] = {}

        for length in range(min_pattern_length, min(6, len(domains) // 2 + 1)):
            for i in range(len(domains) - length + 1):
                pattern = " → ".join(domains[i : i + length])
                patterns[pattern] = patterns.get(pattern, 0) + 1

        # Filter to repeated patterns only
        repeated = {k: v for k, v in patterns.items() if v >= 2}
        ranked = sorted(repeated.items(), key=lambda x: x[1], reverse=True)

        return {
            "patterns": [{"sequence": k, "count": v} for k, v in ranked[:10]],
            "count": len(repeated),
        }

    def find_content_patterns(
        self, items: list[MemoryItem], min_frequency: int = 3
    ) -> dict[str, Any]:
        """Find recurring concepts and phrases across items."""
        if not items:
            return {"concepts": [], "bigrams": [], "count": 0}

        # Word frequency across all items
        word_freq: dict[str, int] = {}
        bigram_freq: dict[str, int] = {}

        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "was", "one", "our", "has", "have", "been",
            "were", "will", "with", "this", "that", "from", "they",
            "said", "each", "which", "their", "about", "would",
            "there", "could", "other", "into", "more", "some",
        }

        for item in items:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', item.experience.content.lower())
            words = [w for w in words if w not in stopwords]

            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1

        # Filter by minimum frequency
        concepts = sorted(
            [(w, c) for w, c in word_freq.items() if c >= min_frequency],
            key=lambda x: x[1], reverse=True,
        )[:20]

        bigrams = sorted(
            [(b, c) for b, c in bigram_freq.items() if c >= min_frequency],
            key=lambda x: x[1], reverse=True,
        )[:10]

        return {
            "concepts": [{"term": w, "frequency": c} for w, c in concepts],
            "bigrams": [{"phrase": b, "frequency": c} for b, c in bigrams],
            "count": len(concepts) + len(bigrams),
        }

    def find_domain_patterns(
        self, items: list[MemoryItem]
    ) -> dict[str, Any]:
        """Find domain-level trends and distribution."""
        if not items:
            return {"distribution": {}, "trends": [], "dominant": None}

        domain_counts: dict[str, int] = {}
        domain_importance: dict[str, list[float]] = {}
        domain_timeline: dict[str, list[float]] = {}

        for item in items:
            d = item.experience.domain
            domain_counts[d] = domain_counts.get(d, 0) + 1
            domain_importance.setdefault(d, []).append(item.experience.importance)
            domain_timeline.setdefault(d, []).append(item.experience.timestamp)

        # Distribution
        total = len(items)
        distribution = {d: c / total for d, c in domain_counts.items()}

        # Dominant domain
        dominant = max(domain_counts, key=domain_counts.get) if domain_counts else None

        # Trend: is each domain growing or declining?
        trends = []
        for d, timestamps in domain_timeline.items():
            if len(timestamps) >= 3:
                mid = len(timestamps) // 2
                first_half = len(timestamps[:mid])
                second_half = len(timestamps[mid:])
                trend = "growing" if second_half > first_half else (
                    "declining" if second_half < first_half else "stable"
                )
                trends.append({
                    "domain": d,
                    "trend": trend,
                    "count": domain_counts[d],
                    "avg_importance": float(np.mean(domain_importance[d])),
                })

        return {
            "distribution": distribution,
            "trends": trends,
            "dominant": dominant,
        }
