"""NormExtractor — extracting social and behavioural norms from memory.

v0.21.0: The Social Mind

Human social life is governed by norms — implicit and explicit rules about what
one should and should not do. Norms regulate cooperation, signal group membership,
and provide the shared framework that makes coordinated social action possible.
Learning norms from observation — watching what is praised, punished, expected, or
forbidden — is a fundamental capacity that develops in children by age two and
continues throughout life.

NormExtractor operationalises norm learning for the memory store. It scans
accumulated memories for norm-indicating language: prescriptive keywords that
signal what *should* or *must* happen (should, must, ought, always, expected,
appropriate, acceptable, required, standard, recommended) and prohibitive keywords
that signal what *must not* happen (never, forbidden, inappropriate, unacceptable,
prohibited, avoid). For each norm keyword, it extracts the first meaningful token
after the keyword as the norm's subject and builds a compact norm record. Norms
are aggregated across memories, scored by frequency and source importance, and
made queryable — both by domain and by relevance to a described behaviour.

The `check_norm` method allows an agent to ask "what norms apply to this behaviour?"
using token Jaccard overlap between the behaviour description and stored norm
content — enabling normative self-regulation before acting.

Biological analogue: social norm learning from observation (Fehr & Gächter 2002 —
the enforcement of social norms as a public good); anterior cingulate cortex as
the neural substrate of norm violation detection (Berns et al. 2012); the role of
the insula in moral and norm processing (Damasio 1994); cultural learning and
normative convention acquisition (Tomasello 1999); the internalization of social
norms as a transition from external enforcement to intrinsic motivation (Elster
1989).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_PRESCRIPTIVE_KEYWORDS: frozenset[str] = frozenset({
    "should", "must", "ought", "always", "expected", "appropriate",
    "acceptable", "required", "standard", "recommended",
})

_PROHIBITIVE_KEYWORDS: frozenset[str] = frozenset({
    "never", "forbidden", "inappropriate", "unacceptable", "prohibited", "avoid",
})

_ALL_NORM_KEYWORDS: frozenset[str] = _PRESCRIPTIVE_KEYWORDS | _PROHIBITIVE_KEYWORDS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SocialNorm:
    """A behavioural norm extracted from memory content."""

    id: str                     # prefixed "norm_"
    content: str                # excerpt describing the norm
    domain: str
    polarity: str               # "prescriptive" | "prohibitive"
    keyword: str                # the triggering norm keyword
    subject: str                # first meaningful token after keyword
    confidence: float           # 0..1 — frequency × importance blend
    memory_ids: list[str]       # supporting memories

    def summary(self) -> str:
        arrow = "✓" if self.polarity == "prescriptive" else "✗"
        return (
            f"SocialNorm [{self.domain}] {arrow}  "
            f"confidence={self.confidence:.3f}\n"
            f"  {self.id[:12]}: [{self.keyword}] {self.subject} — "
            f"{self.content[:60]}"
        )


@dataclass
class NormReport:
    """Result of a NormExtractor.extract_norms() call."""

    total_norms: int
    norms: list[SocialNorm]         # sorted by confidence desc
    prescriptive_count: int
    prohibitive_count: int
    domains_covered: list[str]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"NormReport: {self.total_norms} norms  "
            f"({self.prescriptive_count} prescriptive, "
            f"{self.prohibitive_count} prohibitive)  "
            f"domains={self.domains_covered}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for n in self.norms[:5]:
            lines.append(f"  {n.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NormExtractor
# ---------------------------------------------------------------------------


class NormExtractor:
    """Extracts behavioural norms from accumulated memory content.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_norm_frequency:
        Minimum number of supporting memories for a norm to be included
        (default 1).
    max_norms:
        Maximum number of :class:`SocialNorm` objects to generate per call
        (default 20).
    """

    def __init__(
        self,
        memory: Any,
        min_norm_frequency: int = 1,
        max_norms: int = 20,
    ) -> None:
        self.memory = memory
        self.min_norm_frequency = min_norm_frequency
        self.max_norms = max_norms
        self._norms: list[SocialNorm] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_norms(self, domain: Optional[str] = None) -> NormReport:
        """Extract social norms from accumulated memories.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`NormReport` with norms sorted by confidence descending.
        """
        t0 = time.time()
        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        n_items = max(len(items), 1)
        raw = self._extract_raw_norms(items)

        # Group by (domain, subject, polarity, keyword)
        grouped: dict[tuple[str, str, str, str], list[tuple[str, str, str, float]]] = defaultdict(list)
        for dom, keyword, subject, content, mid, importance, polarity in raw:
            grouped[(dom, subject, polarity, keyword)].append((content, mid, keyword, importance))

        new_norms: list[SocialNorm] = []
        for (dom, subject, polarity, keyword), entries in grouped.items():
            if len(new_norms) >= self.max_norms:
                break
            freq = len(entries)
            if freq < self.min_norm_frequency:
                continue

            # Best content = most representative excerpt
            best_content = entries[0][0]
            mem_ids = list(dict.fromkeys(e[1] for e in entries))[:5]
            importances = [e[3] for e in entries]
            mean_imp = sum(importances) / len(importances)

            freq_ratio = min(1.0, freq / n_items)
            confidence = round(freq_ratio * 0.6 + mean_imp * 0.4, 4)

            norm = SocialNorm(
                id=f"norm_{uuid.uuid4().hex[:8]}",
                content=best_content,
                domain=dom,
                polarity=polarity,
                keyword=keyword,
                subject=subject,
                confidence=confidence,
                memory_ids=mem_ids,
            )
            new_norms.append(norm)

        new_norms.sort(key=lambda n: n.confidence, reverse=True)
        self._norms = new_norms

        prescriptive = sum(1 for n in new_norms if n.polarity == "prescriptive")
        prohibitive = sum(1 for n in new_norms if n.polarity == "prohibitive")
        domains_covered = sorted({n.domain for n in new_norms})

        return NormReport(
            total_norms=len(new_norms),
            norms=new_norms,
            prescriptive_count=prescriptive,
            prohibitive_count=prohibitive,
            domains_covered=domains_covered,
            duration_seconds=time.time() - t0,
        )

    def norms_for_domain(self, domain: str) -> list[SocialNorm]:
        """Return all extracted norms for a given domain.

        Args:
            domain: The domain to filter by.

        Returns:
            List of :class:`SocialNorm` sorted by confidence descending.
        """
        return sorted(
            [n for n in self._norms if n.domain == domain],
            key=lambda n: n.confidence,
            reverse=True,
        )

    def check_norm(self, behavior_description: str) -> list[SocialNorm]:
        """Find norms most relevant to a described behaviour.

        Uses token Jaccard overlap between the behaviour description and each
        norm's content + subject + keyword.

        Args:
            behavior_description: Natural-language description of the behaviour.

        Returns:
            List of :class:`SocialNorm` sorted by relevance descending (up to 5).
        """
        if not self._norms:
            return []

        goal_tokens = frozenset(
            w.strip(".,!?;:\"'()").lower()
            for w in behavior_description.split()
            if len(w.strip(".,!?;:\"'()")) >= 3
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        )
        if not goal_tokens:
            return self._norms[:5]

        scored: list[tuple[float, SocialNorm]] = []
        for norm in self._norms:
            norm_tokens = frozenset(
                w.lower()
                for w in (norm.content + " " + norm.subject + " " + norm.keyword).split()
                if len(w) >= 3
            )
            union = len(goal_tokens | norm_tokens)
            if union == 0:
                continue
            score = len(goal_tokens & norm_tokens) / union
            scored.append((score, norm))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:5]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw_norms(
        self, items: list[Any]
    ) -> list[tuple[str, str, str, str, str, float, str]]:
        """Extract (domain, keyword, subject, content, memory_id, importance, polarity)."""
        raw = []
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            dom = getattr(item.experience, "domain", None) or "general"
            importance = min(1.0, max(0.0, getattr(item.experience, "importance", 0.5) or 0.5))
            words = text.lower().split()

            for i, word in enumerate(words):
                tok = word.strip(".,!?;:\"'()")
                if tok not in _ALL_NORM_KEYWORDS:
                    continue
                polarity = (
                    "prescriptive" if tok in _PRESCRIPTIVE_KEYWORDS else "prohibitive"
                )
                # Subject: first meaningful token after keyword
                subject = None
                for j in range(i + 1, min(i + 5, len(words))):
                    candidate = words[j].strip(".,!?;:\"'()")
                    if (
                        len(candidate) >= 3
                        and candidate not in _STOP_WORDS
                        and candidate not in _ALL_NORM_KEYWORDS
                    ):
                        subject = candidate
                        break
                if subject is None:
                    continue
                # Content: excerpt around the keyword (±4 words)
                start = max(0, i - 3)
                end = min(len(words), i + 6)
                excerpt = " ".join(words[start:end])
                raw.append((dom, tok, subject, excerpt, item.id, importance, polarity))
        return raw

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
