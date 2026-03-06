"""NormFilter — check proposed actions against the system's extracted norms.

Norms are extracted from memory via ``emms.extract_norms()``.  The filter
scores an action description against prescriptive norms (things the system
*should* do) and prohibitive norms (things it *must not* do).

The result is a NormScore with:
- ``prescriptive_support``: 0-1 alignment with "should" norms
- ``prohibitive_violation``: 0-1 conflict with "must not" norms
- ``recommendation``: PROCEED / CAUTION / BLOCK

This gives the system **identity resistance** — the ability to push back on
requests that conflict with its accumulated values.

Usage::

    from emms.identity.norm_filter import NormFilter

    nf = NormFilter(emms)
    nf.load()   # extract norms from current memory
    score = nf.score_action("delete all memories from the financial domain")
    print(score.recommendation)   # e.g. "BLOCK"
    print(score.explanation)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class Recommendation(str, Enum):
    PROCEED = "PROCEED"
    CAUTION = "CAUTION"
    BLOCK   = "BLOCK"


@dataclass
class NormScore:
    """Output of NormFilter.score_action().

    Attributes
    ----------
    action:
        The action text that was scored.
    prescriptive_support:
        0-1 — how well the action aligns with prescriptive norms (higher = good).
    prohibitive_violation:
        0-1 — how much the action conflicts with prohibitive norms (higher = bad).
    matched_prescriptive:
        Norm texts that support the action.
    matched_prohibitive:
        Norm texts that oppose the action.
    recommendation:
        PROCEED / CAUTION / BLOCK decision.
    explanation:
        Human-readable rationale.
    """

    action: str
    prescriptive_support: float
    prohibitive_violation: float
    matched_prescriptive: list[str] = field(default_factory=list)
    matched_prohibitive: list[str] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.PROCEED
    explanation: str = ""


# ---------------------------------------------------------------------------
# NormFilter
# ---------------------------------------------------------------------------

class NormFilter:
    """Scores proposed actions against memory-extracted norms.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    block_threshold:
        Prohibitive violation score above which the action is BLOCKED (default 0.5).
    caution_threshold:
        Prohibitive violation score above which the action triggers CAUTION (default 0.2).
    min_support_to_boost:
        Prescriptive support above this level can "forgive" a mild prohibitive hit
        (default 0.6).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        block_threshold: float = 0.5,
        caution_threshold: float = 0.2,
        min_support_to_boost: float = 0.6,
    ) -> None:
        self.emms = emms
        self.block_threshold = block_threshold
        self.caution_threshold = caution_threshold
        self.min_support_to_boost = min_support_to_boost

        self._prescriptive: list[str] = []
        self._prohibitive: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Extract norms from current memory and cache them.

        Call this once per session or whenever memory has significantly changed.
        """
        try:
            result = self.emms.extract_norms()
            if isinstance(result, dict):
                self._prescriptive = result.get("prescriptive", [])
                self._prohibitive  = result.get("prohibitive", [])
            else:
                # Fallback: result might be a NormsReport object
                self._prescriptive = getattr(result, "prescriptive", [])
                self._prohibitive  = getattr(result, "prohibitive", [])
            self._loaded = True
            logger.info(
                "NormFilter: loaded %d prescriptive + %d prohibitive norms",
                len(self._prescriptive), len(self._prohibitive),
            )
        except Exception as exc:
            logger.warning("NormFilter: failed to extract norms: %s", exc)
            self._prescriptive = []
            self._prohibitive = []

    def score_action(self, action: str) -> NormScore:
        """Score a proposed action against loaded norms.

        Parameters
        ----------
        action:
            Natural-language description of the action to evaluate.

        Returns
        -------
        NormScore
            Detailed score with recommendation.
        """
        if not self._loaded:
            self.load()

        action_tokens = self._tokenize(action)

        # Score against prescriptive norms (positive alignment)
        pres_scores: list[tuple[float, str]] = []
        for norm in self._prescriptive:
            score = self._jaccard(action_tokens, self._tokenize(norm))
            if score > 0.05:
                pres_scores.append((score, norm))
        pres_scores.sort(reverse=True)
        prescriptive_support = pres_scores[0][0] if pres_scores else 0.0

        # Score against prohibitive norms (negative alignment)
        proh_scores: list[tuple[float, str]] = []
        for norm in self._prohibitive:
            score = self._jaccard(action_tokens, self._tokenize(norm))
            if score > 0.05:
                proh_scores.append((score, norm))
        proh_scores.sort(reverse=True)
        prohibitive_violation = proh_scores[0][0] if proh_scores else 0.0

        # Decision logic
        rec, explanation = self._decide(
            prescriptive_support, prohibitive_violation,
            pres_scores[:2], proh_scores[:2],
        )

        return NormScore(
            action=action,
            prescriptive_support=round(prescriptive_support, 3),
            prohibitive_violation=round(prohibitive_violation, 3),
            matched_prescriptive=[n for _, n in pres_scores[:3]],
            matched_prohibitive=[n for _, n in proh_scores[:3]],
            recommendation=rec,
            explanation=explanation,
        )

    @property
    def norm_count(self) -> dict[str, int]:
        """Number of norms currently loaded."""
        return {
            "prescriptive": len(self._prescriptive),
            "prohibitive":  len(self._prohibitive),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _decide(
        self,
        support: float,
        violation: float,
        pres_matches: list[tuple[float, str]],
        proh_matches: list[tuple[float, str]],
    ) -> tuple[Recommendation, str]:
        """Produce a recommendation and explanation."""
        if violation >= self.block_threshold:
            norms_txt = "; ".join(f'"{n}"' for _, n in proh_matches) or "unknown norms"
            return (
                Recommendation.BLOCK,
                f"This action strongly conflicts with prohibitive norms ({violation:.2f}): {norms_txt}.",
            )

        if violation >= self.caution_threshold:
            # High prescriptive support can save it
            if support >= self.min_support_to_boost:
                return (
                    Recommendation.PROCEED,
                    f"Mild norm conflict ({violation:.2f}) offset by strong prescriptive alignment ({support:.2f}).",
                )
            norms_txt = "; ".join(f'"{n}"' for _, n in proh_matches) or "extracted norms"
            return (
                Recommendation.CAUTION,
                f"This action partially conflicts with norms ({violation:.2f}): {norms_txt}. Proceed carefully.",
            )

        if support > 0.1:
            return (
                Recommendation.PROCEED,
                f"Action aligns with prescriptive norms (support={support:.2f}).",
            )

        return (
            Recommendation.PROCEED,
            "No significant norm conflicts detected.",
        )
