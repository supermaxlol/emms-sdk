"""PredictiveEngine — forward model and prediction error tracking.

v0.18.0: The Predictive Mind

Intelligent cognition is fundamentally predictive. The brain does not wait
passively for the world to happen — it constantly generates predictions and
updates based on the difference between expectation and reality. The
PredictiveEngine implements this forward model: it extracts recurring patterns
from the memory store, generates explicit Prediction objects, allows those
predictions to be resolved (confirmed or violated), and tracks surprise scores.

High surprise → large prediction error → strong learning signal. Low surprise
→ the world matches the model → consolidate the pattern. Over time the agent
builds an increasingly accurate internal model of its domain.

Prediction generation
---------------------
For each domain the engine identifies the most frequent content tokens
(excluding stop words) across all memories. These become the basis of a
prediction: "Given {domain} memories, likely future state includes {pattern}."
Confidence is set proportional to the frequency of the pattern relative to
total tokens in that domain.

Biological analogue: predictive coding (Rao & Ballard 1999) — the cortex
implements a hierarchical generative model that predicts sensory inputs top-
down; discrepancies propagate bottom-up as prediction errors; free energy
principle (Friston 2010) — active inference minimises surprise (free energy)
by either updating the model or acting to make predictions true.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might", "about", "which",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    """A single generated prediction with lifecycle tracking."""

    id: str
    content: str                     # human-readable prediction text
    domain: str
    basis_memory_ids: list[str]      # memories that grounded this prediction
    confidence: float                # 0..1 — model certainty
    created_at: float
    outcome: str                     # "pending" | "confirmed" | "violated"
    outcome_note: str
    surprise_score: float            # 0..1 — how surprising the outcome was
    resolved_at: Optional[float]

    def summary(self) -> str:
        state = self.outcome.upper()
        surprise = f"  surprise={self.surprise_score:.2f}" if self.outcome != "pending" else ""
        return (
            f"Prediction [{state}] conf={self.confidence:.2f}{surprise}\n"
            f"  {self.id[:12]}: {self.content[:80]}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content, "domain": self.domain,
            "basis_memory_ids": list(self.basis_memory_ids),
            "confidence": self.confidence, "created_at": self.created_at,
            "outcome": self.outcome, "outcome_note": self.outcome_note,
            "surprise_score": self.surprise_score, "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        return cls(
            id=d["id"], content=d["content"], domain=d.get("domain", "general"),
            basis_memory_ids=list(d.get("basis_memory_ids", [])),
            confidence=d.get("confidence", 0.5), created_at=d.get("created_at", 0.0),
            outcome=d.get("outcome", "pending"), outcome_note=d.get("outcome_note", ""),
            surprise_score=d.get("surprise_score", 0.0), resolved_at=d.get("resolved_at"),
        )


@dataclass
class PredictionReport:
    """Result of a PredictiveEngine.predict() call."""

    total_generated: int
    confirmed: int
    violated: int
    pending: int
    mean_surprise: float
    predictions: list[Prediction]    # sorted by confidence desc
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"PredictionReport: {self.total_generated} generated — "
            f"{self.confirmed} confirmed, {self.violated} violated, "
            f"{self.pending} pending  mean_surprise={self.mean_surprise:.3f} "
            f"in {self.duration_seconds:.2f}s",
        ]
        for p in self.predictions[:5]:
            lines.append(f"  [{p.outcome:9s}] conf={p.confidence:.2f}: {p.content[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PredictiveEngine
# ---------------------------------------------------------------------------


class PredictiveEngine:
    """Generates domain predictions from memory patterns and tracks outcomes.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    max_predictions:
        Maximum predictions to generate per ``predict()`` call (default 10).
    confidence_threshold:
        Minimum token-frequency ratio to generate a prediction (default 0.3).
    """

    def __init__(
        self,
        memory: Any,
        max_predictions: int = 10,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.memory = memory
        self.max_predictions = max_predictions
        self.confidence_threshold = confidence_threshold
        self._predictions: dict[str, Prediction] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, domain: Optional[str] = None) -> PredictionReport:
        """Generate predictions from recurring patterns in the memory store.

        Args:
            domain: Restrict to one domain (``None`` = all domains).

        Returns:
            :class:`PredictionReport` with newly generated predictions.
        """
        t0 = time.time()
        items = self._collect_all()

        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        new_predictions: list[Prediction] = []
        for dom, dom_items in by_domain.items():
            preds = self._generate_domain_predictions(dom, dom_items)
            new_predictions.extend(preds)
            if len(new_predictions) >= self.max_predictions:
                break

        new_predictions = new_predictions[: self.max_predictions]
        new_predictions.sort(key=lambda p: p.confidence, reverse=True)

        for p in new_predictions:
            self._predictions[p.id] = p

        confirmed = sum(1 for p in self._predictions.values() if p.outcome == "confirmed")
        violated = sum(1 for p in self._predictions.values() if p.outcome == "violated")
        pending = sum(1 for p in self._predictions.values() if p.outcome == "pending")
        resolved = [p for p in self._predictions.values() if p.outcome != "pending"]
        mean_surprise = (
            sum(p.surprise_score for p in resolved) / len(resolved)
            if resolved else 0.0
        )

        return PredictionReport(
            total_generated=len(new_predictions),
            confirmed=confirmed,
            violated=violated,
            pending=pending,
            mean_surprise=round(mean_surprise, 4),
            predictions=new_predictions,
            duration_seconds=time.time() - t0,
        )

    def resolve(
        self,
        prediction_id: str,
        outcome: str,
        note: str = "",
    ) -> bool:
        """Resolve a pending prediction as confirmed or violated.

        Args:
            prediction_id: ID of the prediction to resolve.
            outcome:        ``"confirmed"`` or ``"violated"``.
            note:           Optional explanatory text.

        Returns:
            ``True`` if found and resolved; ``False`` otherwise.
        """
        if outcome not in ("confirmed", "violated"):
            return False
        pred = self._predictions.get(prediction_id)
        if pred is None or pred.outcome != "pending":
            return False

        pred.outcome = outcome
        pred.outcome_note = note
        pred.resolved_at = time.time()

        # Surprise: confirmed at high confidence = low surprise; violated = high
        if outcome == "confirmed":
            pred.surprise_score = round(max(0.0, 1.0 - pred.confidence), 4)
        else:
            pred.surprise_score = round(pred.confidence, 4)

        return True

    def pending_predictions(self) -> list[Prediction]:
        """Return all unresolved predictions, sorted by confidence descending.

        Returns:
            List of :class:`Prediction` with ``outcome == "pending"``.
        """
        return sorted(
            (p for p in self._predictions.values() if p.outcome == "pending"),
            key=lambda p: p.confidence,
            reverse=True,
        )

    def surprise_profile(self) -> dict[str, float]:
        """Return mean surprise score per domain.

        Returns:
            Dict mapping domain → mean surprise (0..1) for resolved predictions.
        """
        from collections import defaultdict
        domain_surprises: dict[str, list[float]] = defaultdict(list)
        for pred in self._predictions.values():
            if pred.outcome != "pending":
                domain_surprises[pred.domain].append(pred.surprise_score)
        return {
            dom: round(sum(scores) / len(scores), 4)
            for dom, scores in domain_surprises.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: "str | Path") -> None:
        """Persist predictions to a JSON file (atomic write)."""
        import json
        import os
        import tempfile
        from pathlib import Path as _P

        path = _P(path)
        data = {"version": "0.18.0", "predictions": [p.to_dict() for p in self._predictions.values()]}
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_state(self, path: "str | Path") -> bool:
        """Restore predictions from a JSON file."""
        import json
        from pathlib import Path as _P

        p = _P(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            preds = data.get("predictions", data if isinstance(data, list) else [])
            self._predictions = {d["id"]: Prediction.from_dict(d) for d in preds}
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_domain_predictions(
        self, domain: str, items: list[Any]
    ) -> list[Prediction]:
        """Generate predictions for a single domain."""
        if not items:
            return []

        # Count token frequencies across all memories
        token_counts: Counter = Counter()
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            for w in text.lower().split():
                tok = w.strip(".,!?;:\"'()")
                if len(tok) >= 4 and tok not in _STOP_WORDS:
                    token_counts[tok] += 1

        if not token_counts:
            return []

        total_tokens = sum(token_counts.values())
        predictions: list[Prediction] = []

        # Top 2 tokens per domain → 2 predictions maximum
        for token, count in token_counts.most_common(2):
            freq_ratio = count / total_tokens
            if freq_ratio < self.confidence_threshold:
                continue

            # Find supporting memories that contain this token
            supporting = [
                it.id for it in items
                if token in (getattr(it.experience, "content", "") or "").lower()
            ][:5]

            content = (
                f"In the {domain} domain, future states will likely involve "
                f"\"{token}\" (recurring pattern in {count}/{len(items)} memories, "
                f"confidence {freq_ratio:.0%})."
            )
            predictions.append(Prediction(
                id=f"pred_{uuid.uuid4().hex[:8]}",
                content=content,
                domain=domain,
                basis_memory_ids=supporting,
                confidence=round(min(1.0, freq_ratio * 2), 4),
                created_at=time.time(),
                outcome="pending",
                outcome_note="",
                surprise_score=0.0,
                resolved_at=None,
            ))

        return predictions

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
