"""EpistemicEvolution — tracking how knowledge grows and consolidates over time.

v0.24.0: The Wise Mind

An agent that merely accumulates memories is not learning in the deepest sense —
it is archiving. True learning involves the progressive restructuring of knowledge:
new experiences connect to existing ones, vocabulary consolidates around recurring
themes, and sparse early acquaintance deepens into rich, inter-connected expertise.
Tracking this evolution is essential for metacognitive awareness: the agent can
identify which knowledge domains are actively growing, which have consolidated
into stable patterns, and which have stagnated or never developed beyond minimal
coverage.

EpistemicEvolution operationalises knowledge growth tracking by analysing the
temporal sequence of memories within each domain. It splits each domain's
memories into an early half and a recent half, computes the token sets of each,
and derives three measures: growth_rate (the net gain of new vocabulary in recent
memories relative to early ones), consolidation_score (the Jaccard overlap of
vocabularies across time — high consolidation means recent memories reinforce
existing concepts rather than introducing new ones), and knowledge_density
(memories per day, normalised across all domains). It also identifies knowledge
gaps — domains with fewer memories than a configurable threshold.

Biological analogue: hippocampal-neocortical transfer theory (McClelland et al.
1995) — knowledge consolidates from episodic to semantic over time; power law
of practice (Newell & Rosenbloom 1981); schema theory as progressive abstraction
(Bartlett 1932); expert-novice knowledge restructuring (Chi et al. 1982);
neural efficiency in skill acquisition (Haier et al. 1992); sleep-dependent
memory consolidation and vocabulary integration.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
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
class KnowledgeDomain:
    """Epistemic profile of a single memory domain."""

    domain: str
    memory_count: int
    growth_rate: float            # -1..1; positive = new tokens accumulating
    consolidation_score: float    # 0..1; high = recent reinforces early
    knowledge_density: float      # 0..1 normalised across all domains
    recent_themes: list[str]      # top 5 tokens from recent half
    oldest_memory_ts: float
    newest_memory_ts: float

    def summary(self) -> str:
        age_days = max((self.newest_memory_ts - self.oldest_memory_ts) / 86400, 0)
        return (
            f"KnowledgeDomain [{self.domain}]  "
            f"n={self.memory_count}  "
            f"growth={self.growth_rate:+.3f}  "
            f"consolidation={self.consolidation_score:.3f}  "
            f"density={self.knowledge_density:.3f}  "
            f"span={age_days:.1f}d\n"
            f"  recent_themes: {self.recent_themes[:5]}"
        )


@dataclass
class EvolutionReport:
    """Result of an EpistemicEvolution.evolve() call."""

    total_domains: int
    domains: list[KnowledgeDomain]   # sorted by knowledge_density desc
    most_active_domain: str          # highest growth_rate
    most_consolidated_domain: str    # highest consolidation_score
    overall_growth_rate: float       # mean growth_rate across domains
    knowledge_gaps: list[str]        # domain names with < min_memories
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"EvolutionReport: {self.total_domains} domains  "
            f"most_active={self.most_active_domain}  "
            f"most_consolidated={self.most_consolidated_domain}  "
            f"overall_growth={self.overall_growth_rate:+.3f}  "
            f"gaps={self.knowledge_gaps}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for d in self.domains[:5]:
            lines.append(f"  {d.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EpistemicEvolution
# ---------------------------------------------------------------------------


class EpistemicEvolution:
    """Tracks how knowledge has grown and consolidated across domains over time.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_memories:
        Minimum memories for a domain to not be flagged as a gap (default 3).
    """

    def __init__(
        self,
        memory: Any,
        min_memories: int = 3,
    ) -> None:
        self.memory = memory
        self.min_memories = min_memories
        self._domains: list[KnowledgeDomain] = []
        self._gaps: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(self, domain: Optional[str] = None) -> EvolutionReport:
        """Compute epistemic evolution profiles for all memory domains.

        Args:
            domain: Restrict to this domain only (``None`` = all domains).

        Returns:
            :class:`EvolutionReport` with domains sorted by knowledge_density desc.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain, sort each by timestamp
        domain_items: dict[str, list[Any]] = defaultdict(list)
        for item in all_items:
            dom = getattr(item.experience, "domain", None) or "general"
            domain_items[dom].append(item)

        for dom in domain_items:
            domain_items[dom].sort(
                key=lambda it: getattr(it.experience, "timestamp", 0.0) or 0.0
            )

        # Compute raw density (memories per day) for normalisation
        raw_densities: dict[str, float] = {}
        for dom, items in domain_items.items():
            n = len(items)
            ts_list = [
                getattr(it.experience, "timestamp", 0.0) or 0.0
                for it in items
            ]
            if len(ts_list) >= 2:
                span_days = max((ts_list[-1] - ts_list[0]) / 86400, 1e-6)
            else:
                span_days = 1.0
            raw_densities[dom] = n / span_days

        # Normalise densities 0..1
        min_d = min(raw_densities.values(), default=0.0)
        max_d = max(raw_densities.values(), default=1.0)
        density_range = max(max_d - min_d, 1e-9)

        kd_list: list[KnowledgeDomain] = []
        gaps: list[str] = []

        for dom, items in domain_items.items():
            n = len(items)
            if n < self.min_memories:
                gaps.append(dom)

            # Split into early / recent halves
            mid = max(n // 2, 1)
            early_items = items[:mid]
            recent_items = items[mid:] if n > 1 else items

            early_tokens = set(self._tokens_from(early_items))
            recent_tokens = set(self._tokens_from(recent_items))

            union_size = max(len(early_tokens | recent_tokens), 1)
            new_in_recent = len(recent_tokens - early_tokens)
            lost_in_recent = len(early_tokens - recent_tokens)
            growth_rate = round(
                max(-1.0, min(1.0, (new_in_recent - lost_in_recent) / union_size)),
                4,
            )

            inter_size = len(early_tokens & recent_tokens)
            consolidation_score = round(inter_size / union_size, 4)

            norm_density = round(
                (raw_densities[dom] - min_d) / density_range, 4
            )

            # Recent themes: top 5 tokens in recent half
            recent_freq: Counter = Counter()
            for item in recent_items:
                content = getattr(item.experience, "content", "") or ""
                for tok in self._tokenise(content):
                    recent_freq[tok] += 1
            recent_themes = [tok for tok, _ in recent_freq.most_common(5)]

            ts_list = [
                getattr(it.experience, "timestamp", 0.0) or 0.0
                for it in items
            ]

            kd_list.append(KnowledgeDomain(
                domain=dom,
                memory_count=n,
                growth_rate=growth_rate,
                consolidation_score=consolidation_score,
                knowledge_density=norm_density,
                recent_themes=recent_themes,
                oldest_memory_ts=ts_list[0] if ts_list else 0.0,
                newest_memory_ts=ts_list[-1] if ts_list else 0.0,
            ))

        kd_list.sort(key=lambda kd: kd.knowledge_density, reverse=True)
        self._domains = kd_list
        self._gaps = gaps

        most_active = (
            max(kd_list, key=lambda kd: kd.growth_rate).domain
            if kd_list else "none"
        )
        most_consolidated = (
            max(kd_list, key=lambda kd: kd.consolidation_score).domain
            if kd_list else "none"
        )
        overall_growth = round(
            sum(kd.growth_rate for kd in kd_list) / len(kd_list)
            if kd_list else 0.0,
            4,
        )

        return EvolutionReport(
            total_domains=len(kd_list),
            domains=kd_list,
            most_active_domain=most_active,
            most_consolidated_domain=most_consolidated,
            overall_growth_rate=overall_growth,
            knowledge_gaps=gaps,
            duration_seconds=time.time() - t0,
        )

    def domain_profile(self, domain: str) -> Optional[KnowledgeDomain]:
        """Return the epistemic profile for a specific domain.

        Args:
            domain: Domain name to query.

        Returns:
            :class:`KnowledgeDomain`, or ``None`` if not found.
        """
        for kd in self._domains:
            if kd.domain == domain:
                return kd
        return None

    def knowledge_gaps(self) -> list[str]:
        """Return domains with fewer than min_memories memories.

        Returns:
            List of domain names flagged as knowledge gaps.
        """
        return list(self._gaps)

    def most_active(self) -> Optional[KnowledgeDomain]:
        """Return the domain with the highest growth_rate, or ``None``."""
        if not self._domains:
            return None
        return max(self._domains, key=lambda kd: kd.growth_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_from(self, items: list[Any]) -> list[str]:
        """Collect all meaningful tokens from a list of memory items."""
        tokens: list[str] = []
        for item in items:
            content = getattr(item.experience, "content", "") or ""
            tokens.extend(self._tokenise(content))
        return tokens

    def _tokenise(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 4
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
