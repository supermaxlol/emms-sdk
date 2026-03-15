"""LiveSelfModel — Persistent, incrementally-updated self-model.

Unlike SelfModel which rebuilds from scratch on every call, LiveSelfModel
maintains state vectors that update incrementally on each store() call.
This makes the self-model a live accumulator, not a periodic report.

Gap 6 in the EMMS → AGI roadmap: the foundation for all other gaps.
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

from emms.core.models import MemoryItem, MemoryTier


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class LiveBelief:
    """A belief maintained by incremental evidence accumulation."""

    id: str = ""
    content: str = ""
    domain: str = ""
    confidence: float = 0.0
    evidence_count: int = 0  # how many memories support this
    mean_strength: float = 0.0  # running mean of supporting memory strengths
    mean_valence: float = 0.0  # running mean of supporting memory valence
    supporting_memory_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    last_updated: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = f"lbel_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_updated:
            self.last_updated = time.time()

    def update(self, memory_strength: float, valence: float, memory_id: str):
        """Incremental Bayesian-style update from a new supporting memory."""
        n = self.evidence_count
        self.mean_strength = (self.mean_strength * n + memory_strength) / (n + 1)
        self.mean_valence = (self.mean_valence * n + valence) / (n + 1)
        self.evidence_count = n + 1
        self.confidence = min(1.0, self.mean_strength * 0.4 + (self.evidence_count / (self.evidence_count + 5)) * 0.6)
        if memory_id not in self.supporting_memory_ids:
            self.supporting_memory_ids.append(memory_id)
            if len(self.supporting_memory_ids) > 10:
                self.supporting_memory_ids = self.supporting_memory_ids[-10:]
        self.last_updated = time.time()

    def decay(self, factor: float = 0.995):
        """Gradual confidence decay for beliefs not recently reinforced."""
        age_hours = (time.time() - self.last_updated) / 3600
        if age_hours > 1:
            self.confidence *= factor ** age_hours
            self.confidence = max(0.01, self.confidence)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LiveBelief":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DriftEvent:
    """Records when a capability or belief shifts significantly."""

    timestamp: float = 0.0
    domain: str = ""
    metric: str = ""  # "capability", "belief_confidence", "calibration"
    old_value: float = 0.0
    new_value: float = 0.0
    delta: float = 0.0
    description: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        self.delta = self.new_value - self.old_value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DriftEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CalibrationEntry:
    """A single prediction with its outcome for calibration tracking."""

    prediction_id: str = ""
    domain: str = ""
    predicted_confidence: float = 0.0
    outcome: float = 0.0  # 1.0 = correct, 0.0 = wrong
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Calibration Tracker ─────────────────────────────────────────────────────


class CalibrationTracker:
    """Per-domain Brier scores with bias detection.

    Tracks how well predictions match outcomes, per domain.
    Enables the self-model to say "I'm systematically overconfident in finance"
    and adjust future confidence accordingly.
    """

    def __init__(self, max_entries_per_domain: int = 200):
        self.entries: dict[str, list[CalibrationEntry]] = defaultdict(list)
        self.max_entries = max_entries_per_domain

    def record(self, prediction_id: str, domain: str, confidence: float, correct: bool):
        """Record a prediction outcome."""
        entry = CalibrationEntry(
            prediction_id=prediction_id,
            domain=domain,
            predicted_confidence=confidence,
            outcome=1.0 if correct else 0.0,
            timestamp=time.time(),
        )
        self.entries[domain].append(entry)
        if len(self.entries[domain]) > self.max_entries:
            self.entries[domain] = self.entries[domain][-self.max_entries:]

    def brier_score(self, domain: str | None = None) -> float:
        """Brier score: mean squared error between confidence and outcome.
        Lower is better. 0 = perfect calibration, 0.25 = random.
        """
        entries = self._get_entries(domain)
        if not entries:
            return 0.25  # uninformative prior
        total = sum((e.predicted_confidence - e.outcome) ** 2 for e in entries)
        return total / len(entries)

    def calibration_bias(self, domain: str | None = None) -> float:
        """Positive = overconfident, negative = underconfident."""
        entries = self._get_entries(domain)
        if not entries:
            return 0.0
        mean_conf = sum(e.predicted_confidence for e in entries) / len(entries)
        mean_outcome = sum(e.outcome for e in entries) / len(entries)
        return mean_conf - mean_outcome

    def adjusted_confidence(self, raw_confidence: float, domain: str) -> float:
        """Post-hoc calibration: adjust confidence based on historical bias."""
        bias = self.calibration_bias(domain)
        adjusted = raw_confidence - bias * 0.5  # partial correction
        return max(0.05, min(0.95, adjusted))

    def domain_report(self) -> dict[str, dict[str, float]]:
        """Per-domain calibration summary."""
        report = {}
        for domain in self.entries:
            entries = self.entries[domain]
            if not entries:
                continue
            report[domain] = {
                "brier_score": self.brier_score(domain),
                "bias": self.calibration_bias(domain),
                "n_predictions": len(entries),
                "accuracy": sum(1 for e in entries if (e.predicted_confidence > 0.5) == (e.outcome > 0.5)) / len(entries),
            }
        return report

    def _get_entries(self, domain: str | None) -> list[CalibrationEntry]:
        if domain:
            return self.entries.get(domain, [])
        return [e for entries in self.entries.values() for e in entries]

    def save_state(self, path: str | Path) -> None:
        data = {
            domain: [e.to_dict() for e in entries]
            for domain, entries in self.entries.items()
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        self.entries = defaultdict(list)
        for domain, entries in data.items():
            self.entries[domain] = [CalibrationEntry.from_dict(e) for e in entries]
        return True


# ── Live Self-Model ──────────────────────────────────────────────────────────


class LiveSelfModel:
    """Persistent, incrementally-updated self-model.

    Instead of rebuilding beliefs from scratch (like SelfModel.update()),
    this maintains running accumulators that update on every store() call.

    State vectors:
    - capability_vector: domain → competence score (running average)
    - live_beliefs: incrementally accumulated beliefs
    - calibration: per-domain prediction accuracy tracking
    - drift_log: records of significant changes in self-model
    """

    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "that", "this", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "what", "which", "who",
        "whom", "source", "turn", "conversation",
    })

    def __init__(
        self,
        memory: Any,  # HierarchicalMemory
        goal_stack: Any = None,
        max_beliefs: int = 20,
        drift_threshold: float = 0.15,
    ):
        self.memory = memory
        self.goal_stack = goal_stack
        self.max_beliefs = max_beliefs
        self.drift_threshold = drift_threshold

        # Live state vectors
        self.capability_vector: dict[str, float] = {}
        self._domain_counts: dict[str, int] = defaultdict(int)
        self._domain_strength_sum: dict[str, float] = defaultdict(float)

        self.live_beliefs: list[LiveBelief] = []
        self._belief_index: dict[str, LiveBelief] = {}  # token_domain → belief

        self.calibration = CalibrationTracker()
        self.drift_log: list[DriftEvent] = []

        self._total_experiences: int = 0
        self._dominant_valence: float = 0.0
        self._valence_sum: float = 0.0

    # ── Incremental Update (called on every store) ───────────────────────

    def update_from_experience(self, item: MemoryItem) -> dict[str, Any]:
        """Incrementally update the self-model from a single new memory.

        This is the core method — called on every store(), not periodically.
        Returns a dict of what changed for logging/events.
        """
        changes: dict[str, Any] = {"beliefs_updated": [], "drift_events": [], "domain": ""}

        exp = item.experience
        domain = exp.domain or "general"
        strength = item.memory_strength
        valence = getattr(exp, "emotional_valence", 0.0) or getattr(exp, "valence", 0.0) or 0.0
        changes["domain"] = domain

        # 1. Update capability vector
        old_cap = self.capability_vector.get(domain, 0.0)
        self._domain_counts[domain] += 1
        self._domain_strength_sum[domain] += strength
        count = self._domain_counts[domain]
        mean_str = self._domain_strength_sum[domain] / count
        new_cap = min(1.0, mean_str * math.log(1 + count) / math.log(1 + 5))
        self.capability_vector[domain] = new_cap

        if abs(new_cap - old_cap) > self.drift_threshold:
            drift = DriftEvent(
                domain=domain,
                metric="capability",
                old_value=old_cap,
                new_value=new_cap,
                description=f"Capability in {domain} shifted from {old_cap:.3f} to {new_cap:.3f}",
            )
            self.drift_log.append(drift)
            changes["drift_events"].append(drift.to_dict())

        # 2. Update running valence
        self._total_experiences += 1
        self._valence_sum += valence
        self._dominant_valence = self._valence_sum / self._total_experiences

        # 3. Extract tokens and update beliefs
        content = exp.content or ""
        tokens = self._extract_tokens(content)
        for token in tokens[:5]:  # top 5 tokens per experience
            key = f"{token}_{domain}"
            if key in self._belief_index:
                belief = self._belief_index[key]
                old_conf = belief.confidence
                belief.update(strength, valence, item.id)
                if abs(belief.confidence - old_conf) > 0.05:
                    changes["beliefs_updated"].append(belief.content)
            else:
                belief = LiveBelief(
                    content=f"In {domain}, '{token}' is a recurring pattern",
                    domain=domain,
                    evidence_count=1,
                    mean_strength=strength,
                    mean_valence=valence,
                    supporting_memory_ids=[item.id],
                    confidence=min(1.0, strength * 0.4 + 0.1),
                )
                self._belief_index[key] = belief
                self.live_beliefs.append(belief)
                changes["beliefs_updated"].append(belief.content)

        # 4. Prune weak beliefs if over capacity
        if len(self.live_beliefs) > self.max_beliefs * 2:
            self._prune_beliefs()

        return changes

    def _extract_tokens(self, text: str) -> list[str]:
        """Extract meaningful tokens from text, sorted by length (longer = more specific)."""
        words = text.lower().split()
        tokens = [
            w.strip(".,!?;:'\"()[]{}") for w in words
            if len(w) >= 4 and w.lower().strip(".,!?;:'\"()[]{}") not in self.STOP_WORDS
        ]
        # Deduplicate preserving order, prefer longer tokens
        seen = set()
        unique = []
        for t in sorted(tokens, key=len, reverse=True):
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _prune_beliefs(self):
        """Remove low-confidence beliefs to stay within capacity."""
        # Apply decay first
        for b in self.live_beliefs:
            b.decay()
        # Sort by confidence and keep top max_beliefs
        self.live_beliefs.sort(key=lambda b: b.confidence, reverse=True)
        pruned = self.live_beliefs[:self.max_beliefs]
        # Rebuild index
        self._belief_index = {}
        for b in pruned:
            parts = b.content.split("'")
            token = parts[1] if len(parts) > 1 else b.domain
            key = f"{token}_{b.domain}"
            self._belief_index[key] = b
        self.live_beliefs = pruned

    # ── Drift Detection ──────────────────────────────────────────────────

    def detect_drift(self, window: int = 50) -> list[DriftEvent]:
        """Return recent drift events."""
        return self.drift_log[-window:]

    # ── Query Methods ────────────────────────────────────────────────────

    def beliefs(self) -> list[LiveBelief]:
        """Return current beliefs sorted by confidence."""
        return sorted(self.live_beliefs, key=lambda b: b.confidence, reverse=True)

    def top_beliefs(self, n: int = 5) -> list[LiveBelief]:
        """Return top-N beliefs by confidence."""
        return self.beliefs()[:n]

    def beliefs_for_domain(self, domain: str) -> list[LiveBelief]:
        """Return beliefs for a specific domain."""
        return [b for b in self.beliefs() if b.domain == domain]

    def consistency_score(self) -> float:
        """Emotional consistency across beliefs. 1 = coherent, 0 = conflicted."""
        if len(self.live_beliefs) < 2:
            return 1.0
        valences = [b.mean_valence for b in self.live_beliefs if b.evidence_count >= 2]
        if len(valences) < 2:
            return 1.0
        mean_v = sum(valences) / len(valences)
        variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
        std = variance ** 0.5
        return max(0.0, min(1.0, 1.0 - std))

    def capability_profile(self) -> dict[str, float]:
        """Return domain → competence mapping."""
        return dict(sorted(self.capability_vector.items(), key=lambda x: x[1], reverse=True))

    def summary(self) -> str:
        """Generate a human-readable self-model summary."""
        lines = [
            f"LiveSelfModel — {self._total_experiences} experiences processed",
            f"Consistency: {self.consistency_score():.3f}",
            f"Dominant valence: {self._dominant_valence:+.3f}",
            "",
            "Top Capabilities:",
        ]
        for domain, cap in list(self.capability_profile().items())[:5]:
            lines.append(f"  {domain}: {cap:.3f} ({self._domain_counts.get(domain, 0)} memories)")

        lines.append("")
        lines.append("Top Beliefs:")
        for b in self.top_beliefs(5):
            lines.append(f"  [{b.confidence:.2f}] {b.content} (n={b.evidence_count})")

        cal = self.calibration.domain_report()
        if cal:
            lines.append("")
            lines.append("Calibration:")
            for domain, stats in cal.items():
                lines.append(
                    f"  {domain}: Brier={stats['brier_score']:.3f}, "
                    f"bias={stats['bias']:+.3f}, "
                    f"accuracy={stats['accuracy']:.1%} "
                    f"(n={stats['n_predictions']})"
                )

        drift = self.detect_drift(5)
        if drift:
            lines.append("")
            lines.append("Recent Drift:")
            for d in drift:
                lines.append(f"  {d.description}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self, path: str | Path) -> None:
        """Save the live self-model state to JSON."""
        data = {
            "capability_vector": self.capability_vector,
            "domain_counts": dict(self._domain_counts),
            "domain_strength_sum": dict(self._domain_strength_sum),
            "live_beliefs": [b.to_dict() for b in self.live_beliefs],
            "drift_log": [d.to_dict() for d in self.drift_log[-100:]],
            "total_experiences": self._total_experiences,
            "valence_sum": self._valence_sum,
            "dominant_valence": self._dominant_valence,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str), encoding="utf-8")

        # Save calibration separately
        cal_path = p.parent / (p.stem + "_calibration.json")
        self.calibration.save_state(cal_path)

    def load_state(self, path: str | Path) -> bool:
        """Load the live self-model state from JSON."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.capability_vector = data.get("capability_vector", {})
            self._domain_counts = defaultdict(int, data.get("domain_counts", {}))
            self._domain_strength_sum = defaultdict(float, data.get("domain_strength_sum", {}))
            self.live_beliefs = [LiveBelief.from_dict(b) for b in data.get("live_beliefs", [])]
            self.drift_log = [DriftEvent.from_dict(d) for d in data.get("drift_log", [])]
            self._total_experiences = data.get("total_experiences", 0)
            self._valence_sum = data.get("valence_sum", 0.0)
            self._dominant_valence = data.get("dominant_valence", 0.0)

            # Rebuild index
            self._belief_index = {}
            for b in self.live_beliefs:
                token = b.content.split("'")[1] if "'" in b.content else b.domain
                key = f"{token}_{b.domain}"
                self._belief_index[key] = b

            # Load calibration
            cal_path = p.parent / (p.stem + "_calibration.json")
            self.calibration.load_state(cal_path)

            return True
        except Exception:
            return False
