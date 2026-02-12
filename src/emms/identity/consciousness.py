"""Consciousness-inspired identity enrichment.

Ports the key ideas from conscious_fixed.py into the SDK:
- ContinuousNarrator — builds a persistent self-narrative
- MeaningMaker — assigns personal significance to experiences
- TemporalIntegrator — tracks identity continuity across time
- EgoBoundaryTracker — maintains self/other distinction

These are *functional* modules, not claims of phenomenal consciousness.
They implement the algorithms that give an agent coherent, persistent
identity — the practical contribution of the EMMS research.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from emms.core.models import Experience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

class NarrativeEntry(BaseModel):
    """A single entry in the continuous narrative."""
    timestamp: float = Field(default_factory=time.time)
    content: str = ""
    domain: str = "general"
    significance: float = 0.0
    themes: list[str] = Field(default_factory=list)


class ContinuousNarrator:
    """Builds and maintains a coherent self-narrative from experiences.

    Unlike the basic ego.py narrative (a template string), this tracks
    narrative themes, coherence over time, and evolving self-description.
    """

    def __init__(self, max_entries: int = 200):
        self.entries: list[NarrativeEntry] = []
        self.max_entries = max_entries
        self.themes: dict[str, float] = {}  # theme → cumulative weight
        self.coherence: float = 0.9
        self.traits: dict[str, float] = {}  # inferred personality traits
        self.autobiographical: list[str] = []  # key life events
        self._max_autobiographical = 50

    def integrate(self, experience: Experience) -> NarrativeEntry:
        """Add an experience to the narrative."""
        themes = self._extract_themes(experience.content)
        significance = self._compute_significance(experience, themes)

        entry = NarrativeEntry(
            timestamp=experience.timestamp,
            content=experience.content[:200],
            domain=experience.domain,
            significance=significance,
            themes=themes,
        )
        self.entries.append(entry)

        # Update theme weights
        for theme in themes:
            self.themes[theme] = self.themes.get(theme, 0.0) + significance

        # Cap entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Update coherence
        self.coherence = self._compute_coherence()

        # Track traits from themes
        self._update_traits(themes, experience)

        # Record autobiographical moments (high-significance events)
        if significance >= 0.7:
            moment = f"[{experience.domain}] {experience.content[:100]}"
            self.autobiographical.append(moment)
            if len(self.autobiographical) > self._max_autobiographical:
                self.autobiographical = self.autobiographical[-self._max_autobiographical:]

        return entry

    def build_first_person_narrative(self) -> str:
        """Generate a first-person introspective narrative."""
        if not self.entries:
            return "I have just begun. My memories are forming."

        total = len(self.entries)
        top_traits = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)[:3]
        trait_desc = ", ".join(f"{t} ({v:.0%})" for t, v in top_traits) if top_traits else "still discovering myself"

        recent = self.entries[-5:]
        recent_focus = ", ".join({e.domain for e in recent})

        parts = [
            f"I have processed {total} experiences.",
            f"I see myself as: {trait_desc}.",
            f"Lately I've been focused on: {recent_focus}.",
            f"My narrative coherence is {self.coherence:.0%}.",
        ]

        if self.autobiographical:
            parts.append(f"Key moments: {'; '.join(self.autobiographical[-3:])}")

        return " ".join(parts)

    def get_autobiographical_connections(self, experience: Experience) -> list[str]:
        """Find autobiographical moments related to a new experience."""
        keywords = set(experience.content.lower().split())
        connections = []
        for moment in self.autobiographical:
            moment_words = set(moment.lower().split())
            overlap = len(keywords & moment_words)
            if overlap >= 2:
                connections.append(moment)
        return connections[:5]

    def _update_traits(self, themes: list[str], experience: Experience) -> None:
        """Infer personality traits from recurring patterns."""
        # Curiosity: high novelty-seeking
        if experience.novelty > 0.7:
            self.traits["curious"] = min(1.0, self.traits.get("curious", 0.0) + 0.02)
        # Analytical: frequent domain focus
        if experience.domain in ("tech", "science", "finance"):
            self.traits["analytical"] = min(1.0, self.traits.get("analytical", 0.0) + 0.01)
        # Empathetic: high emotional content
        if experience.emotional_intensity > 0.5:
            self.traits["empathetic"] = min(1.0, self.traits.get("empathetic", 0.0) + 0.02)
        # Focused: consistent themes
        if self.coherence > 0.7:
            self.traits["focused"] = min(1.0, self.traits.get("focused", 0.0) + 0.01)

    def build_narrative(self, agent_name: str = "EMMS Agent") -> str:
        """Generate a natural-language self-narrative."""
        if not self.entries:
            return f"I am {agent_name}. I am ready to learn and grow."

        total = len(self.entries)
        domains = list({e.domain for e in self.entries})
        top_themes = sorted(
            self.themes.items(), key=lambda x: x[1], reverse=True
        )[:5]
        theme_str = ", ".join(t for t, _ in top_themes) if top_themes else "general topics"

        # Recent highlights (high-significance entries)
        recent = sorted(
            self.entries[-20:], key=lambda e: e.significance, reverse=True
        )[:3]
        highlights = "; ".join(e.content[:80] for e in recent)

        return (
            f"I am {agent_name}. Over {total} experiences across "
            f"{len(domains)} domains, my focus has been on {theme_str}. "
            f"Recent highlights: {highlights}. "
            f"Narrative coherence: {self.coherence:.0%}."
        )

    def _extract_themes(self, text: str) -> list[str]:
        """Extract thematic keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = {
            "this", "that", "with", "from", "have", "been", "were",
            "will", "would", "could", "should", "about", "their",
            "there", "which", "other", "into", "more", "some",
        }
        filtered = [w for w in words if w not in stopwords]
        # Return top 3 by frequency within this text
        freq: dict[str, int] = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:3]]

    def _compute_significance(
        self, experience: Experience, themes: list[str]
    ) -> float:
        """How significant is this experience to the agent's narrative?"""
        base = experience.importance * 0.4 + experience.novelty * 0.3

        # Theme continuity bonus: recurring themes matter more
        theme_bonus = 0.0
        for theme in themes:
            if theme in self.themes:
                theme_bonus += min(0.1, self.themes[theme] * 0.01)

        # Emotional intensity bonus
        emotional = abs(experience.emotional_valence) * experience.emotional_intensity * 0.2

        return min(1.0, base + theme_bonus + emotional)

    def _compute_coherence(self) -> float:
        """Measure narrative coherence via theme consistency."""
        if len(self.entries) < 3:
            return 0.9

        # Compare theme overlap between recent windows
        recent = self.entries[-10:]
        older = self.entries[-20:-10] if len(self.entries) > 10 else self.entries[:10]

        recent_themes = set()
        for e in recent:
            recent_themes.update(e.themes)

        older_themes = set()
        for e in older:
            older_themes.update(e.themes)

        if not recent_themes or not older_themes:
            return 0.8

        overlap = len(recent_themes & older_themes) / len(recent_themes | older_themes)
        return 0.5 + overlap * 0.5  # range 0.5..1.0


# ---------------------------------------------------------------------------
# Meaning maker
# ---------------------------------------------------------------------------

class MeaningMaker:
    """Assigns personal significance to experiences based on identity context.

    Tracks what matters to the agent and amplifies related signals.
    """

    def __init__(self):
        self.value_weights: dict[str, float] = {}  # concept → importance
        self.total_processed: int = 0
        self.meaning_narratives: list[str] = []  # significant meaning events
        self.pattern_tracker: dict[str, int] = {}  # domain → occurrence count
        self.emotional_memory: list[tuple[float, float]] = []  # (valence, intensity) history

    def assess(self, experience: Experience) -> dict[str, Any]:
        """Assess the personal meaning of an experience."""
        self.total_processed += 1

        # Extract concepts
        concepts = re.findall(r'\b[a-zA-Z]{3,}\b', experience.content.lower())
        unique_concepts = list(set(concepts))

        # Compute relevance to existing values
        relevance = 0.0
        for concept in unique_concepts[:20]:
            if concept in self.value_weights:
                relevance += self.value_weights[concept]

        relevance = min(1.0, relevance / max(len(unique_concepts), 1))

        # Learning potential: novel concepts the agent hasn't seen
        novel_concepts = [c for c in unique_concepts if c not in self.value_weights]
        learning_potential = len(novel_concepts) / max(len(unique_concepts), 1)

        # Update value weights
        weight_delta = experience.importance * 0.1
        for concept in unique_concepts[:20]:
            self.value_weights[concept] = min(
                1.0, self.value_weights.get(concept, 0.0) + weight_delta
            )

        # Ego investment: how personally relevant is this?
        ego_investment = (
            experience.importance * 0.3
            + relevance * 0.3
            + abs(experience.emotional_valence) * 0.2
            + learning_potential * 0.2
        )

        # Track patterns
        self.pattern_tracker[experience.domain] = (
            self.pattern_tracker.get(experience.domain, 0) + 1
        )

        # Track emotional memory
        self.emotional_memory.append(
            (experience.emotional_valence, experience.emotional_intensity)
        )
        if len(self.emotional_memory) > 200:
            self.emotional_memory = self.emotional_memory[-200:]

        # Compute emotional significance
        emotional_significance = self._emotional_significance(experience)

        # Record meaning narratives for highly meaningful experiences
        if ego_investment > 0.6:
            narrative = (
                f"[{experience.domain}] Meaningful experience "
                f"(relevance={relevance:.2f}, learning={learning_potential:.2f}): "
                f"{experience.content[:80]}"
            )
            self.meaning_narratives.append(narrative)
            if len(self.meaning_narratives) > 100:
                self.meaning_narratives = self.meaning_narratives[-100:]

        return {
            "relevance": relevance,
            "learning_potential": learning_potential,
            "ego_investment": min(1.0, ego_investment),
            "novel_concepts": len(novel_concepts),
            "total_values_tracked": len(self.value_weights),
            "emotional_significance": emotional_significance,
            "domain_pattern_count": self.pattern_tracker.get(experience.domain, 0),
        }

    def get_value_alignment(self, text: str) -> float:
        """How well does the text align with the agent's established values?"""
        concepts = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not concepts:
            return 0.0
        alignment = sum(
            self.value_weights.get(c, 0.0) for c in set(concepts)
        )
        return min(1.0, alignment / max(len(set(concepts)), 1))

    def _emotional_significance(self, experience: Experience) -> float:
        """Compute emotional significance based on deviation from baseline."""
        if len(self.emotional_memory) < 5:
            return abs(experience.emotional_valence) * experience.emotional_intensity

        recent_valences = [v for v, _ in self.emotional_memory[-20:]]
        baseline_valence = float(np.mean(recent_valences))
        deviation = abs(experience.emotional_valence - baseline_valence)
        return min(1.0, deviation * experience.emotional_intensity * 2)


# ---------------------------------------------------------------------------
# Temporal integrator
# ---------------------------------------------------------------------------

class TemporalIntegrator:
    """Tracks identity continuity across time.

    Monitors whether the agent's responses and interests remain coherent
    or shift dramatically — a measure of temporal identity stability.
    """

    def __init__(self, window_seconds: float = 3600.0):
        self.window_seconds = window_seconds
        self._recent_domains: list[tuple[float, str]] = []
        self._recent_importance: list[tuple[float, float]] = []
        self.milestones: list[dict[str, Any]] = []
        self.identity_snapshots: list[dict[str, Any]] = []
        self._experience_count: int = 0
        self._milestone_interval: int = 100  # snapshot every N experiences

    def update(self, experience: Experience) -> dict[str, float]:
        """Track temporal coherence with a new experience."""
        self._experience_count += 1
        now = experience.timestamp

        # Prune old entries outside window
        self._recent_domains = [
            (t, d) for t, d in self._recent_domains
            if now - t < self.window_seconds
        ]
        self._recent_importance = [
            (t, v) for t, v in self._recent_importance
            if now - t < self.window_seconds
        ]

        self._recent_domains.append((now, experience.domain))
        self._recent_importance.append((now, experience.importance))

        coherence = self._domain_coherence()
        stability = self._importance_stability()
        continuity = self._continuity_score()

        # Milestone detection: significant shifts or achievements
        milestone = self._detect_milestone(experience, coherence, stability)

        # Periodic identity snapshots
        if self._experience_count % self._milestone_interval == 0:
            self._take_identity_snapshot(coherence, stability)

        result: dict[str, float] = {
            "temporal_coherence": coherence,
            "importance_stability": stability,
            "identity_continuity": continuity,
            "experience_count": float(self._experience_count),
        }
        if milestone:
            result["milestone_detected"] = 1.0
        return result

    def _detect_milestone(
        self, experience: Experience, coherence: float, stability: float
    ) -> dict[str, Any] | None:
        """Detect significant identity milestones."""
        # Domain shift milestone
        if len(self._recent_domains) >= 10:
            recent_5 = [d for _, d in self._recent_domains[-5:]]
            older_5 = [d for _, d in self._recent_domains[-10:-5]]
            if set(recent_5) != set(older_5) and len(set(recent_5)) == 1:
                milestone = {
                    "type": "domain_shift",
                    "timestamp": experience.timestamp,
                    "detail": f"Shifted focus to {recent_5[0]}",
                    "experience_count": self._experience_count,
                }
                self.milestones.append(milestone)
                return milestone

        # High-importance burst milestone
        if experience.importance >= 0.9:
            milestone = {
                "type": "high_importance",
                "timestamp": experience.timestamp,
                "detail": f"Critical experience in {experience.domain}",
                "experience_count": self._experience_count,
            }
            self.milestones.append(milestone)
            return milestone

        return None

    def _take_identity_snapshot(self, coherence: float, stability: float) -> None:
        """Take a snapshot of current identity state."""
        domains = [d for _, d in self._recent_domains]
        domain_dist = {}
        for d in domains:
            domain_dist[d] = domain_dist.get(d, 0) + 1

        snapshot = {
            "experience_count": self._experience_count,
            "timestamp": time.time(),
            "coherence": coherence,
            "stability": stability,
            "domain_distribution": domain_dist,
            "milestones_count": len(self.milestones),
        }
        self.identity_snapshots.append(snapshot)
        if len(self.identity_snapshots) > 50:
            self.identity_snapshots = self.identity_snapshots[-50:]

    def _domain_coherence(self) -> float:
        """How consistent is the domain focus over the window?"""
        if len(self._recent_domains) < 2:
            return 1.0
        domains = [d for _, d in self._recent_domains]
        most_common = max(set(domains), key=domains.count)
        return domains.count(most_common) / len(domains)

    def _importance_stability(self) -> float:
        """How stable are importance values? (Low variance = stable)."""
        if len(self._recent_importance) < 2:
            return 1.0
        values = [v for _, v in self._recent_importance]
        std = float(np.std(values)) if len(values) > 1 else 0.0
        return max(0.0, 1.0 - std * 2)  # std of 0.5 → stability 0.0

    def _continuity_score(self) -> float:
        """Combined identity continuity metric."""
        dc = self._domain_coherence()
        is_ = self._importance_stability()
        return dc * 0.6 + is_ * 0.4


# ---------------------------------------------------------------------------
# Ego boundary tracker
# ---------------------------------------------------------------------------

class EgoBoundaryTracker:
    """Tracks the agent's self/other distinction.

    Monitors self-referential language and boundary strength to maintain
    a coherent sense of "self" distinct from external information.
    """

    SELF_MARKERS = {"i", "my", "me", "mine", "myself", "i'm", "i've", "i'll"}
    OTHER_MARKERS = {"you", "your", "they", "them", "their", "he", "she", "it"}

    def __init__(self):
        self.self_count: int = 0
        self.other_count: int = 0
        self.boundary_strength: float = 0.5
        self.boundary_history: list[float] = []
        self.reinforcement_events: list[dict[str, Any]] = []

    def analyse(self, text: str) -> dict[str, float]:
        """Analyse text for ego boundary indicators."""
        words = set(text.lower().split())

        self_hits = len(words & self.SELF_MARKERS)
        other_hits = len(words & self.OTHER_MARKERS)

        self.self_count += self_hits
        self.other_count += other_hits

        # Boundary = how distinct self is from other
        total = self.self_count + self.other_count
        if total > 0:
            self.boundary_strength = (
                (self.self_count + 1) / (total + 2)
            )
        else:
            self.boundary_strength = 0.5

        # Track history
        self.boundary_history.append(self.boundary_strength)
        if len(self.boundary_history) > 200:
            self.boundary_history = self.boundary_history[-200:]

        # Check for boundary reinforcement
        reinforcement = self._check_reinforcement()

        result = {
            "self_references": float(self_hits),
            "other_references": float(other_hits),
            "boundary_strength": self.boundary_strength,
            "cumulative_self": float(self.self_count),
            "cumulative_other": float(self.other_count),
            "boundary_quality": self._boundary_quality(),
        }
        if reinforcement:
            result["reinforcement"] = 1.0
        return result

    def _check_reinforcement(self) -> bool:
        """Detect boundary reinforcement events (sudden self-awareness shifts)."""
        if len(self.boundary_history) < 5:
            return False
        recent = self.boundary_history[-5:]
        older = self.boundary_history[-10:-5] if len(self.boundary_history) >= 10 else self.boundary_history[:5]
        recent_avg = float(np.mean(recent))
        older_avg = float(np.mean(older))
        shift = abs(recent_avg - older_avg)
        if shift > 0.1:
            self.reinforcement_events.append({
                "timestamp": time.time(),
                "shift": shift,
                "direction": "self" if recent_avg > older_avg else "other",
                "new_strength": self.boundary_strength,
            })
            return True
        return False

    def _boundary_quality(self) -> float:
        """Assess overall ego boundary quality (stability + strength)."""
        if len(self.boundary_history) < 5:
            return self.boundary_strength

        stability = 1.0 - float(np.std(self.boundary_history[-20:])) * 2
        stability = max(0.0, min(1.0, stability))
        return self.boundary_strength * 0.6 + stability * 0.4

    def get_ego_investment(self, text: str) -> float:
        """Calculate how personally invested the agent should be in this text."""
        words = text.lower().split()
        total = len(words)
        if total == 0:
            return 0.0
        self_words = sum(1 for w in words if w in self.SELF_MARKERS)
        return min(1.0, self_words / max(total, 1) * 10)
