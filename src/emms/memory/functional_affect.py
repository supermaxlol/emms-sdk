"""FunctionalAffect — Affect as processing modulator, not metadata.

Gap 3 in the EMMS → AGI roadmap: emotions aren't about feeling —
they're about biasing computation. Fear narrows attention. Curiosity
broadens it. Joy reinforces current strategy.

Components:
- AffectState: core affect dimensions (Russell's circumplex)
- SomaticMarker: Damasio-style decision biasing from past outcomes
- FunctionalAffect: the main affect system that modulates cognition

The affect state influences:
- Retrieval breadth (narrow under threat, broad under curiosity)
- Processing depth (fast under high arousal, deliberate under low)
- Risk tolerance (positive valence → more risk)
- Novelty seeking (moderate arousal + positive → explore)

Depends on:
- EmotionalRegulator (existing emotion.py) — current valence/arousal data
- PresenceTracker (existing presence.py) — session attention state
- CognitiveLoop (Gap 1) — affect modulates task prioritization
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ── Affect State ─────────────────────────────────────────────────────────────


@dataclass
class AffectState:
    """Core affect dimensions based on Russell's circumplex model.

    Two-dimensional: valence (pleasant/unpleasant) × arousal (activated/deactivated).
    All cognitive modulation derives from these two values.
    """

    valence: float = 0.0  # -1 (unpleasant) to +1 (pleasant)
    arousal: float = 0.3  # 0 (calm/deactivated) to 1 (excited/activated)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

    @property
    def attention_breadth(self) -> float:
        """How broad or narrow attention should be.

        High arousal + negative valence → narrow (threat focus, 0.0)
        Low arousal + positive valence → broad (exploration, 1.0)
        """
        raw = (1.0 - self.arousal) * 0.5 + (self.valence + 1.0) / 2.0 * 0.5
        return max(0.0, min(1.0, raw))

    @property
    def processing_depth(self) -> float:
        """How deep vs shallow processing should be.

        High arousal → shallow/fast (0.0)
        Low arousal → deep/deliberate (1.0)
        """
        return max(0.0, min(1.0, 1.0 - self.arousal))

    @property
    def risk_tolerance(self) -> float:
        """How much risk to accept in decisions.

        Positive valence → higher risk tolerance
        Negative valence → conservative
        """
        base = (self.valence + 1.0) / 2.0  # 0 to 1
        # Moderate arousal boosts risk-taking slightly
        arousal_mod = 0.1 * (0.5 - abs(self.arousal - 0.5))
        return max(0.0, min(1.0, base + arousal_mod))

    @property
    def novelty_seeking(self) -> float:
        """How much to seek novel/unfamiliar information.

        Positive + moderate arousal → high novelty seeking
        Negative + high arousal → avoidance of novelty
        """
        valence_factor = max(0.0, (self.valence + 1.0) / 2.0)
        # Peak novelty at moderate arousal (~0.5)
        arousal_factor = 1.0 - 2.0 * abs(self.arousal - 0.5)
        return max(0.0, min(1.0, valence_factor * 0.6 + max(0, arousal_factor) * 0.4))

    @property
    def label(self) -> str:
        """Human-readable affect label (Russell's quadrants)."""
        if self.valence >= 0 and self.arousal >= 0.5:
            return "excited"  # high arousal + positive
        elif self.valence >= 0 and self.arousal < 0.5:
            return "calm"  # low arousal + positive
        elif self.valence < 0 and self.arousal >= 0.5:
            return "anxious"  # high arousal + negative
        else:
            return "sad"  # low arousal + negative

    def to_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "timestamp": self.timestamp,
            "label": self.label,
            "attention_breadth": round(self.attention_breadth, 3),
            "processing_depth": round(self.processing_depth, 3),
            "risk_tolerance": round(self.risk_tolerance, 3),
            "novelty_seeking": round(self.novelty_seeking, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AffectState":
        return cls(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.3),
            timestamp=d.get("timestamp", 0.0),
        )


# ── Somatic Markers ──────────────────────────────────────────────────────────


@dataclass
class SomaticMarker:
    """A remembered emotional outcome associated with a decision context.

    Based on Damasio's somatic marker hypothesis: past emotional outcomes
    get associated with similar contexts, biasing future decisions without
    conscious reasoning.
    """

    id: str = ""
    context_tokens: list[str] = field(default_factory=list)  # situation fingerprint
    domain: str = "general"
    valence: float = 0.0  # emotional outcome (-1 to +1)
    arousal: float = 0.5  # intensity of the outcome
    strength: float = 1.0  # how strongly this marker influences (decays)
    created_at: float = 0.0
    access_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"sm_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def similarity(self, tokens: list[str]) -> float:
        """Jaccard similarity between this marker's context and given tokens."""
        if not self.context_tokens or not tokens:
            return 0.0
        set_a = set(self.context_tokens)
        set_b = set(tokens)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def decay(self, factor: float = 0.999):
        """Gradual strength decay."""
        age_hours = (time.time() - self.created_at) / 3600
        self.strength *= factor ** max(0, age_hours / 24)  # daily decay
        self.strength = max(0.01, self.strength)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SomaticMarker":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Functional Affect System ─────────────────────────────────────────────────


STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "or", "but", "not", "so", "if", "then", "that", "this",
    "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "she", "her", "they", "them", "their", "what", "which",
})


class FunctionalAffect:
    """Affect system that modulates all cognitive processing.

    Instead of emotions being metadata on memories, affect is a
    global processing parameter that biases retrieval, reasoning,
    decision-making, and attention allocation.

    Usage:
        affect = FunctionalAffect()

        # Update from experience
        affect.update_from_experience(experience)

        # Modulate retrieval
        modulated_query = affect.modulate_retrieval_params(query_params)

        # Bias a decision
        scored_options = affect.bias_decision(options, context)

        # Get current state
        state = affect.current_state
    """

    def __init__(
        self,
        memory: Any = None,
        valence_momentum: float = 0.85,  # how much past valence carries forward
        arousal_decay: float = 0.9,  # arousal decays toward baseline
        baseline_arousal: float = 0.3,  # resting arousal level
        max_markers: int = 200,
    ):
        self.memory = memory
        self.valence_momentum = valence_momentum
        self.arousal_decay = arousal_decay
        self.baseline_arousal = baseline_arousal
        self.max_markers = max_markers

        # Current affect state
        self._state = AffectState(valence=0.0, arousal=baseline_arousal)

        # Somatic markers
        self._markers: list[SomaticMarker] = []
        self._marker_index: dict[str, list[SomaticMarker]] = defaultdict(list)

        # History
        self._state_history: list[AffectState] = []
        self._total_updates: int = 0

    @property
    def current_state(self) -> AffectState:
        """Get current affect state."""
        return self._state

    # ── Update ───────────────────────────────────────────────────────

    def update_from_experience(self, valence: float, arousal: float | None = None,
                               domain: str = "general") -> AffectState:
        """Update affect state from an experience.

        Uses exponential moving average: new state is a blend of
        current state and incoming signal, weighted by momentum.

        Args:
            valence: emotional valence of the experience (-1 to +1)
            arousal: emotional intensity (0 to 1). None = inferred from |valence|
            domain: experience domain

        Returns:
            Updated AffectState
        """
        if arousal is None:
            arousal = min(1.0, abs(valence) * 1.5)

        # Exponential moving average for valence
        new_valence = self.valence_momentum * self._state.valence + (1 - self.valence_momentum) * valence
        new_valence = max(-1.0, min(1.0, new_valence))

        # Arousal spikes on input, decays toward baseline
        arousal_spike = max(self._state.arousal, arousal)
        new_arousal = self.arousal_decay * arousal_spike + (1 - self.arousal_decay) * self.baseline_arousal
        new_arousal = max(0.0, min(1.0, new_arousal))

        self._state = AffectState(valence=new_valence, arousal=new_arousal)
        self._state_history.append(self._state)
        if len(self._state_history) > 200:
            self._state_history = self._state_history[-200:]
        self._total_updates += 1

        return self._state

    def update_from_surprise(self, surprise_score: float, domain: str = "general") -> AffectState:
        """Update affect from a prediction surprise.

        High surprise → arousal spike + slight negative valence
        (surprises are mildly aversive, even when the content is good)
        """
        arousal_boost = surprise_score * 0.5
        valence_shift = -surprise_score * 0.2  # surprises are slightly aversive
        return self.update_from_experience(valence_shift, arousal_boost, domain)

    def update_from_success(self, magnitude: float = 0.5, domain: str = "general") -> AffectState:
        """Update from a successful outcome. Positive valence + moderate arousal."""
        return self.update_from_experience(magnitude, magnitude * 0.6, domain)

    def update_from_failure(self, magnitude: float = 0.5, domain: str = "general") -> AffectState:
        """Update from a failure. Negative valence + high arousal."""
        return self.update_from_experience(-magnitude, magnitude * 0.8, domain)

    # ── Somatic Markers ──────────────────────────────────────────────

    def mark(self, context: str, outcome_valence: float, outcome_arousal: float = 0.5,
             domain: str = "general") -> SomaticMarker:
        """Stamp a decision context with its emotional outcome.

        Call after a decision plays out — associates the context tokens
        with the experienced emotion for future decision biasing.
        """
        tokens = self._tokenize(context)
        marker = SomaticMarker(
            context_tokens=tokens,
            domain=domain,
            valence=outcome_valence,
            arousal=outcome_arousal,
        )
        self._markers.append(marker)
        self._marker_index[domain].append(marker)

        # Capacity management
        if len(self._markers) > self.max_markers:
            # Remove weakest markers
            self._markers.sort(key=lambda m: m.strength, reverse=True)
            removed = self._markers[self.max_markers:]
            self._markers = self._markers[:self.max_markers]
            # Rebuild index
            self._marker_index = defaultdict(list)
            for m in self._markers:
                self._marker_index[m.domain].append(m)

        return marker

    def consult_markers(self, context: str, domain: str = "general",
                        top_k: int = 5) -> AffectState | None:
        """Check somatic markers for a decision context.

        Returns weighted average affect of similar past contexts,
        or None if no relevant markers found.
        """
        tokens = self._tokenize(context)
        if not tokens:
            return None

        # Check domain-specific markers first, then general
        candidates = list(self._marker_index.get(domain, []))
        if domain != "general":
            candidates.extend(self._marker_index.get("general", []))

        if not candidates:
            return None

        # Score by similarity
        scored = []
        for marker in candidates:
            sim = marker.similarity(tokens)
            if sim > 0.1:  # minimum relevance threshold
                scored.append((marker, sim * marker.strength))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # Weighted average
        total_weight = sum(w for _, w in top)
        if total_weight <= 0:
            return None

        avg_valence = sum(m.valence * w for m, w in top) / total_weight
        avg_arousal = sum(m.arousal * w for m, w in top) / total_weight

        # Update access counts
        for m, _ in top:
            m.access_count += 1

        return AffectState(valence=avg_valence, arousal=avg_arousal)

    def bias_decision(self, options: list[str], context: str,
                      domain: str = "general") -> list[tuple[str, float]]:
        """Reweight decision options using somatic markers.

        Options associated with negative past outcomes get downweighted.
        Returns (option, adjusted_weight) pairs.
        """
        results = []
        for option in options:
            combined_context = f"{context} {option}"
            marker_affect = self.consult_markers(combined_context, domain)

            if marker_affect:
                # Positive markers boost, negative markers suppress
                weight = 0.5 + marker_affect.valence * 0.3
            else:
                weight = 0.5  # neutral

            results.append((option, max(0.05, min(0.95, weight))))

        return results

    # ── Cognitive Modulation ─────────────────────────────────────────

    def modulate_retrieval_params(self, params: dict) -> dict:
        """Modulate retrieval parameters based on current affect.

        Affects:
        - max_results: broader under positive/calm, narrower under anxiety
        - domain filtering: anxious state focuses on threat-relevant domains
        - novelty preference: curious state retrieves more diverse memories
        """
        params = dict(params)  # don't mutate original
        state = self._state

        # Adjust result count by attention breadth
        base_results = params.get("max_results", 10)
        breadth_factor = 0.5 + state.attention_breadth  # 0.5 to 1.5
        params["max_results"] = max(3, int(base_results * breadth_factor))

        # Add affect metadata for downstream use
        params["_affect_state"] = state.to_dict()
        params["_novelty_preference"] = state.novelty_seeking

        return params

    def modulate_task_priority(self, base_priority: float, task_domain: str) -> float:
        """Modulate cognitive task priority based on current affect.

        Anxious state → boost threat-detection tasks
        Curious state → boost exploration tasks
        Calm state → boost deep reasoning tasks
        """
        state = self._state
        modifier = 0.0

        # Anxious → threat detection gets boosted
        if state.label == "anxious":
            if task_domain in ("risk", "security", "error", "failure"):
                modifier = 0.2

        # Calm → deep reasoning boost
        elif state.label == "calm":
            if task_domain in ("analysis", "planning", "research"):
                modifier = 0.15

        # Excited → action-oriented boost
        elif state.label == "excited":
            if task_domain in ("execution", "trading", "implementation"):
                modifier = 0.1

        return max(0.0, min(1.0, base_priority + modifier))

    # ── Query Methods ────────────────────────────────────────────────

    def mood_trend(self, window: int = 20) -> dict:
        """Trend of valence and arousal over recent history."""
        if not self._state_history:
            return {"trend": "stable", "valence_direction": 0.0, "arousal_direction": 0.0}

        recent = self._state_history[-window:]
        if len(recent) < 2:
            return {"trend": "stable", "valence_direction": 0.0, "arousal_direction": 0.0}

        half = len(recent) // 2
        first_half = recent[:half]
        second_half = recent[half:]

        v_first = sum(s.valence for s in first_half) / len(first_half)
        v_second = sum(s.valence for s in second_half) / len(second_half)
        a_first = sum(s.arousal for s in first_half) / len(first_half)
        a_second = sum(s.arousal for s in second_half) / len(second_half)

        v_dir = v_second - v_first
        a_dir = a_second - a_first

        if abs(v_dir) < 0.05:
            trend = "stable"
        elif v_dir > 0:
            trend = "improving"
        else:
            trend = "declining"

        return {
            "trend": trend,
            "valence_direction": round(v_dir, 3),
            "arousal_direction": round(a_dir, 3),
            "current_label": self._state.label,
        }

    def emotional_coherence(self) -> float:
        """How consistent is the emotional state? 1 = stable, 0 = volatile."""
        if len(self._state_history) < 3:
            return 1.0
        recent = self._state_history[-20:]
        valences = [s.valence for s in recent]
        mean_v = sum(valences) / len(valences)
        variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
        std = variance ** 0.5
        return max(0.0, min(1.0, 1.0 - std))

    def summary(self) -> str:
        """Human-readable affect summary."""
        s = self._state
        trend = self.mood_trend()
        lines = [
            f"FunctionalAffect — {self._total_updates} updates",
            f"State: {s.label} (valence={s.valence:+.3f}, arousal={s.arousal:.3f})",
            f"Modulation: breadth={s.attention_breadth:.2f}, depth={s.processing_depth:.2f}, "
            f"risk={s.risk_tolerance:.2f}, novelty={s.novelty_seeking:.2f}",
            f"Trend: {trend['trend']} (v={trend['valence_direction']:+.3f})",
            f"Coherence: {self.emotional_coherence():.3f}",
            f"Somatic markers: {len(self._markers)}",
        ]
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Extract meaningful tokens from text."""
        words = text.lower().split()
        return [
            w.strip(".,!?;:'\"()[]{}") for w in words
            if len(w) >= 3 and w.strip(".,!?;:'\"()[]{}") not in STOP_WORDS
        ]

    # ── Persistence ──────────────────────────────────────────────────

    def save_state(self, path: str | Path):
        """Save affect state to JSON."""
        data = {
            "state": self._state.to_dict(),
            "markers": [m.to_dict() for m in self._markers[-self.max_markers:]],
            "history": [s.to_dict() for s in self._state_history[-50:]],
            "total_updates": self._total_updates,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        """Load affect state from JSON."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._state = AffectState.from_dict(data.get("state", {}))
            self._markers = [SomaticMarker.from_dict(m) for m in data.get("markers", [])]
            self._state_history = [AffectState.from_dict(s) for s in data.get("history", [])]
            self._total_updates = data.get("total_updates", 0)

            # Rebuild index
            self._marker_index = defaultdict(list)
            for m in self._markers:
                self._marker_index[m.domain].append(m)

            return True
        except Exception:
            return False
