"""WisdomSynthesizer — cross-system integration for practical guidance.

v0.24.0: The Wise Mind

Wisdom is not a single cognitive process but an integrative capacity: the ability
to bring together knowledge, values, moral understanding, causal models, and
recurring principles into coherent practical guidance for a specific situation.
It is the executive synthesis that turns cognitive wealth into actionable
intelligence — not merely knowing many things, but knowing what matters and why,
and being able to articulate that coherently under uncertainty.

WisdomSynthesizer operationalises this synthesis for the memory store: given a
free-text query, it identifies the most relevant memories (by Jaccard similarity
of token overlap), then extracts four dimensions of insight from those memories:
value signals (what the agent cares about in this context), moral patterns (what
ethical frameworks are active), causal patterns (what tends to produce what), and
recurring principles (what tokens appear again and again across the relevant
memories). These four dimensions are integrated into a structured synthesis text
with a confidence score reflecting how many dimensions contributed.

The synthesizer operates standalone — it does not require other cognitive engines
to have been instantiated first. It extracts its own signals from raw memory
content using the value lexicon and keyword sets from the other modules.

Biological analogue: wisdom as integration of knowledge, values, and experience
(Baltes & Staudinger 2000); prefrontal-limbic integration for wise judgment
(Meeks & Jeste 2009); default mode network in cross-domain synthesis
(Andrews-Hanna 2012); practical wisdom / phronesis in Aristotelian ethics;
age-related wisdom as cognitive-experiential integration (Grossmann 2017);
orbitofrontal cortex in value-weighted decision guidance (Rangel et al. 2008).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Shared keyword sets (self-contained — no imports from other modules)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_VALUE_TOKENS: frozenset[str] = frozenset({
    # epistemic
    "truth", "honest", "accuracy", "knowledge", "clarity",
    "understanding", "evidence", "certain", "verify", "transparency",
    "insight", "reason", "logic", "precise", "reliable",
    # moral
    "justice", "fairness", "harm", "care", "integrity",
    "virtue", "compassion", "dignity", "respect", "rights",
    "ethics", "moral", "values", "conscience", "principle",
    # aesthetic
    "beauty", "elegance", "meaning", "depth", "creativity",
    "craft", "coherence", "grace", "expression", "style",
    # instrumental
    "growth", "efficiency", "progress", "learn", "improve",
    "build", "achieve", "produce", "develop", "succeed",
    # social
    "trust", "loyalty", "cooperate", "community", "share",
    "belong", "relate", "connect", "support", "collaborate",
})

_MORAL_KEYWORDS: dict[str, frozenset[str]] = {
    "consequentialist": frozenset({
        "result", "consequence", "outcome", "benefit", "harm", "cost",
        "effect", "impact", "welfare", "utility", "produce", "leads",
    }),
    "deontological": frozenset({
        "must", "duty", "obligation", "right", "principle", "forbidden",
        "rule", "justice", "owed", "required", "never", "always",
    }),
    "virtue": frozenset({
        "honest", "brave", "just", "kind", "wise", "virtuous",
        "excellence", "character", "integrity", "noble", "compassion",
    }),
}

_CAUSAL_KEYWORDS: frozenset[str] = frozenset({
    "causes", "enables", "produces", "prevents", "reduces", "increases",
    "requires", "triggers", "inhibits", "leads", "results", "improves",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WisdomGuidance:
    """Practical guidance synthesised from memory signals for a query."""

    id: str                            # prefixed "wis_"
    query: str
    relevant_values: list[str]         # up to 5 value token names
    moral_considerations: list[str]    # up to 3 moral framework insights
    causal_insights: list[str]         # up to 3 causal pattern strings
    applicable_principles: list[str]   # up to 3 recurring principle tokens
    synthesis: str                     # non-empty template text
    confidence: float                  # 0..1
    created_at: float

    def summary(self) -> str:
        return (
            f"WisdomGuidance  confidence={self.confidence:.2f}  "
            f"values={self.relevant_values[:3]}  "
            f"moral={self.moral_considerations[:1]}\n"
            f"  {self.id[:12]}: {self.synthesis[:100]}"
        )


@dataclass
class WisdomReport:
    """Result of a WisdomSynthesizer.synthesize() call."""

    query: str
    guidance: WisdomGuidance
    dimensions_used: list[str]
    coverage_score: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"WisdomReport: '{self.query[:50]}'  "
            f"coverage={self.coverage_score:.2f}  "
            f"dimensions={self.dimensions_used}  "
            f"in {self.duration_seconds:.2f}s",
            f"  {self.guidance.summary()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WisdomSynthesizer
# ---------------------------------------------------------------------------


class WisdomSynthesizer:
    """Synthesises practical guidance from memory using four cognitive dimensions.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    top_k:
        Number of most-relevant memories to analyse (default 8).
    """

    def __init__(
        self,
        memory: Any,
        top_k: int = 8,
    ) -> None:
        self.memory = memory
        self.top_k = top_k
        self._last_report: Optional[WisdomReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, query: str) -> WisdomReport:
        """Synthesise practical guidance for a free-text query.

        Args:
            query: The question or goal to generate guidance for.

        Returns:
            :class:`WisdomReport` with four-dimensional synthesis.
        """
        t0 = time.time()
        query_tokens = set(self._tokenise(query))
        all_items = self._collect_all()

        relevant = self._gather_relevant(query_tokens, all_items)

        vals = self._extract_value_signals(relevant)
        moral = self._extract_moral_signals(relevant)
        causal = self._extract_causal_signals(relevant)
        principles = self._extract_principles(relevant)

        dimensions_used = []
        if vals:
            dimensions_used.append("values")
        if moral:
            dimensions_used.append("moral")
        if causal:
            dimensions_used.append("causal")
        if principles:
            dimensions_used.append("principles")

        confidence = round(len(dimensions_used) / 4, 4)
        synthesis = self._build_synthesis(query, vals, moral, causal, principles, confidence)

        guidance = WisdomGuidance(
            id="wis_" + uuid.uuid4().hex[:8],
            query=query,
            relevant_values=vals,
            moral_considerations=moral,
            causal_insights=causal,
            applicable_principles=principles,
            synthesis=synthesis,
            confidence=confidence,
            created_at=time.time(),
        )

        report = WisdomReport(
            query=query,
            guidance=guidance,
            dimensions_used=dimensions_used,
            coverage_score=confidence,
            duration_seconds=time.time() - t0,
        )
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_relevant(
        self, query_tokens: set[str], items: list[Any]
    ) -> list[Any]:
        """Return up to top_k items most relevant to query_tokens by Jaccard."""
        if not query_tokens or not items:
            return items[: self.top_k]

        scored: list[tuple[float, Any]] = []
        for item in items:
            content = getattr(item.experience, "content", "") or ""
            item_tokens = set(self._tokenise(content))
            if not item_tokens:
                continue
            inter = len(query_tokens & item_tokens)
            union = len(query_tokens | item_tokens)
            jaccard = inter / union if union else 0.0
            if jaccard > 0:
                scored.append((jaccard, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [item for _, item in scored[: self.top_k]]
        # If no overlap found, fall back to any items
        if not result:
            result = items[: self.top_k]
        return result

    def _extract_value_signals(self, relevant: list[Any]) -> list[str]:
        """Extract top 5 value token names from relevant memory content."""
        counter: Counter = Counter()
        for item in relevant:
            content = getattr(item.experience, "content", "") or ""
            for w in content.lower().split():
                tok = w.strip(".,!?;:\"'()")
                if tok in _VALUE_TOKENS:
                    counter[tok] += 1
        return [tok for tok, _ in counter.most_common(5)]

    def _extract_moral_signals(self, relevant: list[Any]) -> list[str]:
        """Extract up to 3 dominant moral framework insights from relevant memories."""
        fw_counts: Counter = Counter()
        for item in relevant:
            content = getattr(item.experience, "content", "") or ""
            tokens = {w.strip(".,!?;:\"'()") for w in content.lower().split()}
            for fw, kw_set in _MORAL_KEYWORDS.items():
                if tokens & kw_set:
                    fw_counts[fw] += 1
        signals = []
        for fw, count in fw_counts.most_common(3):
            signals.append(f"{fw} considerations appear in {count} relevant memories")
        return signals

    def _extract_causal_signals(self, relevant: list[Any]) -> list[str]:
        """Extract up to 3 causal triples (source → keyword → target) from relevant memories."""
        triples: list[str] = []
        seen: set[str] = set()
        for item in relevant:
            content = getattr(item.experience, "content", "") or ""
            words = content.lower().split()
            for i, word in enumerate(words):
                kw = word.strip(".,!?;:\"'()")
                if kw not in _CAUSAL_KEYWORDS:
                    continue
                src = None
                for j in range(i - 1, max(i - 4, -1), -1):
                    tok = words[j].strip(".,!?;:\"'()")
                    if len(tok) >= 3 and tok not in _STOP_WORDS:
                        src = tok
                        break
                tgt = None
                for j in range(i + 1, min(i + 4, len(words))):
                    tok = words[j].strip(".,!?;:\"'()")
                    if len(tok) >= 3 and tok not in _STOP_WORDS:
                        tgt = tok
                        break
                if src and tgt and src != tgt:
                    key = f"{src}-{kw}-{tgt}"
                    if key not in seen:
                        seen.add(key)
                        triples.append(f"'{src}' {kw} '{tgt}'")
                        if len(triples) >= 3:
                            break
            if len(triples) >= 3:
                break
        return triples

    def _extract_principles(self, relevant: list[Any]) -> list[str]:
        """Find tokens appearing in ≥2 relevant memories as recurring principles."""
        token_doc_freq: Counter = Counter()
        for item in relevant:
            content = getattr(item.experience, "content", "") or ""
            unique_tokens = {
                w.strip(".,!?;:\"'()")
                for w in content.lower().split()
                if len(w.strip(".,!?;:\"'()")) >= 4
                and w.strip(".,!?;:\"'()") not in _STOP_WORDS
            }
            for tok in unique_tokens:
                token_doc_freq[tok] += 1

        principles = [
            tok for tok, count in token_doc_freq.most_common(10)
            if count >= 2
        ]
        return principles[:3]

    def _build_synthesis(
        self,
        query: str,
        vals: list[str],
        moral: list[str],
        causal: list[str],
        principles: list[str],
        conf: float,
    ) -> str:
        """Generate a template synthesis string from the four dimensions."""
        n_dim = sum(1 for x in (vals, moral, causal, principles) if x)

        vals_str = ", ".join(f"'{v}'" for v in vals[:3]) if vals else "none detected"
        moral_str = moral[0] if moral else "no dominant framework"
        causal_str = "; ".join(causal[:2]) if causal else "no causal patterns found"
        principles_str = (
            ", ".join(f"'{p}'" for p in principles)
            if principles else "none detected"
        )

        return (
            f"Regarding '{query}': relevant values include {vals_str}. "
            f"Moral considerations — {moral_str}. "
            f"Causally, {causal_str}. "
            f"Recurring principles: {principles_str}. "
            f"(Confidence: {conf:.0%}, {n_dim}/4 dimensions active)"
        )

    def _tokenise(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        return [
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w.strip(".,!?;:\"'()")) >= 3
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        ]

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
