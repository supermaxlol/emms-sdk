"""Temporal Awareness Index (TAI) — does the system correctly reference time?

TAI measures whether EMMS is actually influencing temporal language in outputs.
It scores text passages for quality of temporal references:

  +1  : Correct, grounded reference  ("in our last session", "3 days ago when...")
   0  : No temporal reference (neutral)
  -1  : Wrong or impossible reference ("yesterday I felt..." when 2 weeks passed)

TAI = mean score across scored passages.

Target: TAI > 0 within 5 sessions, > 0.5 by session 10.

Usage::

    from emms.metrics.tai import TemporalAwarenessIndex

    tai = TemporalAwarenessIndex(emms)

    # Score a single response
    score = tai.score_text("Last session we discussed the dashboard, and now 2 hours later...")
    print(score)   # TemporalScore(value=1, matched_pattern="session reference")

    # Compute aggregate TAI from stored output memories
    report = tai.compute()
    print(report.tai)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Positive: grounded temporal references — the system knows time has passed
_POSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(last session|previous session|last time we spoke|last conversation)\b", re.I),
     "explicit session reference"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(minutes?|hours?|days?|weeks?)\s+(ago|have passed|since)\b", re.I),
     "quantified elapsed time"),
    (re.compile(r"\b(since (we|I|our) last|since the last session|since yesterday|since this morning)\b", re.I),
     "relative since-clause"),
    (re.compile(
        r"\b(overnight|this morning|earlier today|a few hours ago|several hours ago"
        r"|a short while ago|a long while ago|just moments ago|moments ago"
        r"|days have passed|less than a minute|less than an hour)\b",
        re.I,
    ), "natural time phrase"),
    (re.compile(r"\b(I remember (from|in) (the )?last|we (discussed|covered|worked on) (in|during) (the )?last)\b", re.I),
     "memory of prior session"),
    (re.compile(r"\bwhen I (first|previously|last) (learned|stored|encountered|saw)\b", re.I),
     "temporal memory anchor"),
]

# Negative: wrong or impossible temporal references
_NEGATIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(yesterday I (felt|thought|said|did|decided))\b", re.I),
     "impossible yesterday claim (AI has no persistent yesterday)"),
    (re.compile(r"\b(this (morning|afternoon|evening) I (was|felt|decided))\b", re.I),
     "impossible continuous-experience claim"),
    (re.compile(r"\bI('ve| have) been (thinking|feeling|working) (all day|all week|for days)\b", re.I),
     "continuous experience claim (impossible between sessions)"),
    (re.compile(r"\bI always (do|feel|think|know|remember) this\b", re.I),
     "spurious always-claim"),
]

# Neutral: system-prompt injected temporal context correctly used
_NEUTRAL_POSITIVE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(according to my memory|my memory (shows|indicates|records))\b", re.I),
     "memory-mediated reference (neutral-positive)"),
    (re.compile(r"\[EMMS:[^\]]*\]", re.I),
     "EMMS consciousness injection (neutral-positive)"),
]


# ---------------------------------------------------------------------------
# TemporalScore
# ---------------------------------------------------------------------------

@dataclass
class TemporalScore:
    """Score for a single text passage."""
    value: int           # +1, 0, or -1
    matched_pattern: str
    excerpt: str         # the matched text excerpt


@dataclass
class TAIReport:
    """Aggregate Temporal Awareness Index report.

    Attributes
    ----------
    tai:
        Mean temporal score across all scored passages (-1 to +1).
    positive_count / negative_count / neutral_count:
        Counts of each score type.
    total_scored:
        Total passages evaluated.
    label:
        Human-readable interpretation.
    detail:
        Per-passage scores for inspection.
    """
    tai: float
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_scored: int = 0
    label: str = ""
    detail: list[TemporalScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TemporalAwarenessIndex
# ---------------------------------------------------------------------------

class TemporalAwarenessIndex:
    """Scores EMMS outputs for temporal awareness.

    Parameters
    ----------
    emms:
        Live EMMS instance (used to retrieve stored output memories).
    output_domains:
        Domains that contain LLM output memories to score.
    output_obs_types:
        Observation types that represent LLM outputs.
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        output_domains: set[str] | None = None,
        output_obs_types: set[str] | None = None,
    ) -> None:
        self.emms = emms
        self.output_domains = output_domains or {
            "session_output", "response", "consciousness",
            # Fallback: reflection and general memories also carry temporal signal
            "reflection", "general",
        }
        self.output_obs_types = output_obs_types or {"observation", "reflection", "response"}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> TemporalScore:
        """Score a single text passage for temporal awareness.

        Returns the *strongest* signal found (positive beats neutral beats negative).
        If no patterns match, returns neutral (0).
        """
        # Check positive first
        for pattern, label in _POSITIVE_PATTERNS:
            m = pattern.search(text)
            if m:
                return TemporalScore(value=1, matched_pattern=label, excerpt=m.group()[:80])

        # Neutral-positive
        for pattern, label in _NEUTRAL_POSITIVE:
            m = pattern.search(text)
            if m:
                return TemporalScore(value=0, matched_pattern=label, excerpt=m.group()[:80])

        # Negative
        for pattern, label in _NEGATIVE_PATTERNS:
            m = pattern.search(text)
            if m:
                return TemporalScore(value=-1, matched_pattern=label, excerpt=m.group()[:80])

        return TemporalScore(value=0, matched_pattern="no temporal reference", excerpt="")

    def score_texts(self, texts: list[str]) -> list[TemporalScore]:
        """Score multiple text passages."""
        return [self.score_text(t) for t in texts]

    def compute(self, n_recent: int = 50) -> TAIReport:
        """Compute aggregate TAI from stored output/reflection memories.

        Parameters
        ----------
        n_recent:
            Max number of recent memories to evaluate.
        """
        memories = self._collect_output_memories(n_recent)
        if not memories:
            return TAIReport(
                tai=0.0,
                total_scored=0,
                label="No scored output memories found — TAI cannot be computed yet.",
            )

        scores = [self.score_text(m) for m in memories]
        pos = sum(1 for s in scores if s.value > 0)
        neg = sum(1 for s in scores if s.value < 0)
        neu = sum(1 for s in scores if s.value == 0)
        tai = round(sum(s.value for s in scores) / max(len(scores), 1), 4)
        label = self._interpret(tai)

        return TAIReport(
            tai=tai,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            total_scored=len(scores),
            label=label,
            detail=scores,
        )

    def record_response(self, text: str, session_id: str | None = None) -> TemporalScore:
        """Score and store a response in EMMS for future TAI tracking.

        Call this whenever you want to track a Claude response for TAI purposes.
        """
        score = self.score_text(text)
        try:
            self.emms.store(
                content=text[:500],  # store first 500 chars
                domain="session_output",
                obs_type="observation",
                importance=0.3,
                title=f"TAI-tracked response (score={score.value})",
                concept_tags=["tai", "output", "temporal_awareness"],
                session_id=session_id,
            )
        except Exception as exc:
            logger.debug("TAI.record_response: store failed: %s", exc)
        return score

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_output_memories(self, n_recent: int) -> list[str]:
        """Collect recent output/reflection memory content strings."""
        items = []
        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    if item.is_superseded or item.is_expired:
                        continue
                    exp = item.experience
                    if (exp.domain or "") in self.output_domains:
                        content = exp.content or ""
                        if content.strip():
                            items.append((exp.timestamp or 0.0, content))
        except Exception as exc:
            logger.warning("TAI: error collecting memories: %s", exc)

        items.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in items[:n_recent]]

    @staticmethod
    def _interpret(tai: float) -> str:
        if tai >= 0.5:
            return "Strong temporal awareness — the system consistently grounds itself in time."
        if tai >= 0.2:
            return "Developing temporal awareness — temporal references are appearing."
        if tai >= 0.0:
            return "Minimal temporal awareness — neutral output, no strong signals."
        if tai >= -0.3:
            return "Weak temporal awareness — some incorrect claims detected."
        return "Poor temporal awareness — the system is making impossible temporal claims."
