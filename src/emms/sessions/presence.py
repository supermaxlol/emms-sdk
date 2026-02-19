"""PresenceTracker — models the finite attention budget of a session.

Each session has a coherence horizon: the agent can maintain deep, focused
presence for roughly `attention_half_life` turns before coherence begins to
decay.  This mirrors the human attentional window — we can only be truly
present for so long before attention drifts or resets.

Key concepts:
  * Presence score:  1.0 = fully coherent, 0.0 = incoherent
  * Attention budget: fraction of the session's total attention remaining
  * Coherence trend:  stable / degrading / recovering
  * Emotional arc:    per-turn valence history across the session
  * Domain focus:     which memory domains dominate the session

The decay model uses a half-life sigmoid so presence falls smoothly rather
than cliff-like:
    presence(t) = 1 / (1 + (t / half_life) ** γ)
where γ controls the sharpness of the decay (default 1.5).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

CoherenceTrend = Literal["stable", "degrading", "recovering", "unknown"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PresenceTurn:
    """Data captured for a single conversational turn."""
    turn_index: int
    content_length: int
    domain: str
    valence: float
    intensity: float
    presence_at_turn: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PresenceMetrics:
    """Snapshot of the current session's attentional state."""
    session_id: str
    turn_count: int
    presence_score: float          # [0.0, 1.0]
    attention_budget_remaining: float  # [0.0, 1.0]  (1 at start, decays)
    coherence_trend: CoherenceTrend
    dominant_domains: list[str]    # top-3 domains by turn count
    emotional_arc: list[float]     # per-turn valence history
    mean_valence: float
    mean_intensity: float
    is_degrading: bool             # True if attention is notably declining

    def summary(self) -> str:
        domain_str = ", ".join(self.dominant_domains[:3]) or "none"
        trend_arrow = {"stable": "→", "degrading": "↓", "recovering": "↑", "unknown": "?"}[
            self.coherence_trend
        ]
        return (
            f"Session {self.session_id} | turn {self.turn_count} | "
            f"presence={self.presence_score:.2f} {trend_arrow} | "
            f"budget={self.attention_budget_remaining:.0%} | "
            f"domains=[{domain_str}] | "
            f"mean_valence={self.mean_valence:+.2f}"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PresenceTracker:
    """Track attentional presence across a session's turns.

    Parameters
    ----------
    session_id : str | None
        Identifier for this session (auto-generated if None).
    attention_half_life : int
        Number of turns at which presence decays to ~0.5 (default 20).
    decay_gamma : float
        Shape exponent for presence decay curve (default 1.5).
        Higher = sharper cliff; lower = gentler slope.
    degrading_threshold : float
        Presence score below which we flag `is_degrading` (default 0.4).
    budget_horizon : int
        Total turns assumed for the session budget calculation (default 50).
        A turn at index `t` has budget_remaining = max(0, 1 - t/horizon).
    """

    def __init__(
        self,
        session_id: str | None = None,
        attention_half_life: int = 20,
        decay_gamma: float = 1.5,
        degrading_threshold: float = 0.4,
        budget_horizon: int = 50,
    ) -> None:
        import uuid
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self.attention_half_life = attention_half_life
        self.decay_gamma = decay_gamma
        self.degrading_threshold = degrading_threshold
        self.budget_horizon = budget_horizon

        self._turns: list[PresenceTurn] = []
        self._started_at: float = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_turn(
        self,
        content: str = "",
        domain: str = "general",
        valence: float = 0.0,
        intensity: float = 0.0,
    ) -> PresenceMetrics:
        """Record a new conversational turn and return updated metrics.

        Parameters
        ----------
        content : str
            The turn text (used for length heuristics).
        domain : str
            Memory domain for this turn.
        valence : float
            Emotional valence of this turn (-1…+1).
        intensity : float
            Emotional intensity of this turn (0…1).
        """
        turn_index = len(self._turns)
        p = self._presence(turn_index)

        turn = PresenceTurn(
            turn_index=turn_index,
            content_length=len(content),
            domain=domain,
            valence=max(-1.0, min(1.0, valence)),
            intensity=max(0.0, min(1.0, intensity)),
            presence_at_turn=p,
        )
        self._turns.append(turn)
        return self._compute_metrics()

    def get_metrics(self) -> PresenceMetrics:
        """Return current metrics without recording a new turn."""
        return self._compute_metrics()

    def is_attention_degrading(self) -> bool:
        """True if presence has fallen below the degrading threshold."""
        return self._presence(len(self._turns)) < self.degrading_threshold

    def attention_budget_remaining(self) -> float:
        """Fraction of the session budget remaining [0, 1]."""
        return max(0.0, 1.0 - len(self._turns) / self.budget_horizon)

    def emotional_arc(self) -> list[float]:
        """Per-turn valence history."""
        return [t.valence for t in self._turns]

    def reset(self) -> None:
        """Reset the tracker for a new session."""
        import uuid
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self._turns.clear()
        self._started_at = time.time()

    def summary(self) -> str:
        """One-line summary of current state."""
        return self._compute_metrics().summary()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _presence(self, turn_index: int) -> float:
        """Compute presence score at given turn index using half-life model.

            presence(t) = 1 / (1 + (t / half_life) ** γ)
        """
        if self.attention_half_life <= 0:
            return 1.0
        ratio = turn_index / self.attention_half_life
        return 1.0 / (1.0 + ratio ** self.decay_gamma)

    def _coherence_trend(self) -> CoherenceTrend:
        """Infer trend from the last few turns."""
        if len(self._turns) < 3:
            return "unknown"
        recent = [t.presence_at_turn for t in self._turns[-5:]]
        # compare first half vs second half
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / max(1, mid)
        second_half = sum(recent[mid:]) / max(1, len(recent) - mid)
        diff = second_half - first_half
        if diff < -0.05:
            return "degrading"
        if diff > 0.05:
            return "recovering"
        return "stable"

    def _compute_metrics(self) -> PresenceMetrics:
        n = len(self._turns)
        current_presence = self._presence(n)

        # dominant domains
        domain_counts: Counter[str] = Counter(t.domain for t in self._turns)
        dominant = [d for d, _ in domain_counts.most_common(3)]

        # emotional arc
        arc = [t.valence for t in self._turns]
        mean_v = sum(arc) / n if n else 0.0
        mean_i = sum(t.intensity for t in self._turns) / n if n else 0.0

        return PresenceMetrics(
            session_id=self.session_id,
            turn_count=n,
            presence_score=current_presence,
            attention_budget_remaining=self.attention_budget_remaining(),
            coherence_trend=self._coherence_trend(),
            dominant_domains=dominant,
            emotional_arc=arc,
            mean_valence=mean_v,
            mean_intensity=mean_i,
            is_degrading=current_presence < self.degrading_threshold,
        )
