"""MarketWisdom — extract and store insight memories from accumulated market experience.

After the system accumulates financial memories, it can contemplate them:
- How many bull/bear markets has it lived through?
- What patterns repeat before losses?
- What does high volatility feel like vs calm accumulation?

These contemplations become wisdom memories — high-importance, semantically-rich
observations that surface during future wake-up cycles and guide future actions.

Usage::

    from emms.embodiment.financial_wisdom import MarketWisdom

    wisdom = MarketWisdom(emms)
    wisdom.contemplate()   # generates + stores 1-3 wisdom memories from recent experience
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


class MarketWisdom:
    """Generates wisdom memories from accumulated financial market experience.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    min_memories_for_wisdom:
        Minimum number of financial_market memories needed before wisdom can
        be generated (default 5).
    """

    def __init__(self, emms: "EMMS", *, min_memories_for_wisdom: int = 5) -> None:
        self.emms = emms
        self.min_memories_for_wisdom = min_memories_for_wisdom

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def contemplate(self) -> list[str]:
        """Run a wisdom contemplation pass.

        Scans recent financial_market memories, derives patterns, and stores
        1-3 wisdom memories.  Returns stored memory IDs.
        """
        memories = self._load_financial_memories()
        if len(memories) < self.min_memories_for_wisdom:
            logger.debug(
                "MarketWisdom: only %d financial memories — need %d for wisdom",
                len(memories), self.min_memories_for_wisdom,
            )
            return []

        wisdom_texts = []

        # Insight 1: Emotional arc
        arc_wisdom = self._emotion_arc_wisdom(memories)
        if arc_wisdom:
            wisdom_texts.append(arc_wisdom)

        # Insight 2: Volatility pattern
        vol_wisdom = self._volatility_wisdom(memories)
        if vol_wisdom:
            wisdom_texts.append(vol_wisdom)

        # Insight 3: Trade outcome pattern
        outcome_wisdom = self._trade_outcome_wisdom(memories)
        if outcome_wisdom:
            wisdom_texts.append(outcome_wisdom)

        memory_ids = []
        for text in wisdom_texts[:3]:
            try:
                result = self.emms.store(
                    content=text,
                    domain="financial_wisdom",
                    namespace="self",
                    obs_type="reflection",
                    importance=0.8,
                    emotional_valence=0.1,  # Wisdom is slightly positive
                    title="Market wisdom",
                    concept_tags=["wisdom", "market", "reflection", "pattern"],
                )
                mid = result.get("memory_id") if isinstance(result, dict) else getattr(result, "memory_id", None)
                if mid:
                    memory_ids.append(mid)
                    logger.info("MarketWisdom: stored wisdom memory %s", mid)
            except Exception as exc:
                logger.warning("MarketWisdom: failed to store wisdom: %s", exc)

        return memory_ids

    # ------------------------------------------------------------------
    # Wisdom generators
    # ------------------------------------------------------------------

    def _emotion_arc_wisdom(self, memories: list[Any]) -> str | None:
        """Generate insight about the emotional arc of market experience."""
        valences = [
            getattr(m, "emotional_valence", 0.0) or 0.0
            for m in memories
        ]
        if not valences:
            return None

        mean_val = sum(valences) / len(valences)
        positives = sum(1 for v in valences if v > 0.1)
        negatives = sum(1 for v in valences if v < -0.1)
        recent_trend = sum(valences[-5:]) / min(5, len(valences))

        direction = "improving" if recent_trend > mean_val + 0.05 else (
            "worsening" if recent_trend < mean_val - 0.05 else "stable"
        )

        return (
            f"Across {len(memories)} market experiences, my emotional arc has been: "
            f"{positives} rewarding moments, {negatives} aversive ones. "
            f"Overall valence: {mean_val:+.2f}. "
            f"Recent trend: {direction}. "
            f"I notice that my reaction to the market is becoming more calibrated over time."
        )

    def _volatility_wisdom(self, memories: list[Any]) -> str | None:
        """Generate insight about volatility patterns."""
        tags_all: list[str] = []
        for m in memories:
            tags = getattr(m.experience, "concept_tags", None) or []
            tags_all.extend(tags)

        vol_tags = [t for t in tags_all if t in {
            "calm", "mild_movement", "moderate_tension",
            "intense_turbulence", "extreme_volatility"
        }]
        if not vol_tags:
            return None

        counts = Counter(vol_tags)
        dominant = counts.most_common(1)[0]
        dominant_name, dominant_count = dominant
        pct = round(100 * dominant_count / max(len(vol_tags), 1))

        feel_labels = {
            "calm": "mostly calm periods",
            "mild_movement": "gentle, manageable movement",
            "moderate_tension": "notable tension — the market kept me alert",
            "intense_turbulence": "significant turbulence — stressful but educational",
            "extreme_volatility": "extreme volatility — survival mode",
        }
        label = feel_labels.get(dominant_name, dominant_name)

        return (
            f"My market life has been dominated by {label} ({pct}% of {len(vol_tags)} experiences). "
            f"I have learned that the dominant feeling before a loss is often "
            f"'{counts.most_common()[-1][0] if len(counts) > 1 else dominant_name}'. "
            f"Recognising the felt-sense of the market before acting is part of discipline."
        )

    def _trade_outcome_wisdom(self, memories: list[Any]) -> str | None:
        """Generate insight about trade outcome patterns from memory content."""
        trade_memories = [
            m for m in memories
            if "trade" in (getattr(m.experience, "concept_tags", None) or [])
        ]
        if not trade_memories:
            return None

        wins = [m for m in trade_memories if (getattr(m, "emotional_valence", 0.0) or 0.0) > 0.05]
        losses = [m for m in trade_memories if (getattr(m, "emotional_valence", 0.0) or 0.0) < -0.05]
        total = len(trade_memories)

        if total == 0:
            return None

        win_rate = round(100 * len(wins) / total)
        avg_win_val = sum(getattr(m, "emotional_valence", 0.0) or 0 for m in wins) / max(len(wins), 1)
        avg_loss_val = sum(getattr(m, "emotional_valence", 0.0) or 0 for m in losses) / max(len(losses), 1)

        # Approximate profit factor from valences
        profit_factor = abs(avg_win_val) / max(abs(avg_loss_val), 0.01)

        if win_rate >= 60:
            style = "more often than not — consistency is my edge"
        elif win_rate >= 40:
            style = "roughly half the time — the profit factor must carry the weight"
        else:
            style = "less often than I lose — but that is fine if the winners are large enough"

        return (
            f"Out of {total} trades stored in memory, I win {win_rate}% of the time — {style}. "
            f"My win/loss emotional intensity ratio is approximately {profit_factor:.1f}. "
            f"{'The edge is solid.' if profit_factor > 1.5 else 'The edge is marginal — stay disciplined.'}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_financial_memories(self) -> list[Any]:
        """Load all non-expired financial_market memories."""
        items = []
        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    if item.is_superseded or item.is_expired:
                        continue
                    if item.experience.domain == "financial_market":
                        items.append(item)
        except Exception as exc:
            logger.warning("MarketWisdom: error loading memories: %s", exc)
        return items
