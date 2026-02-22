"""BiasDetector — recognising systematic cognitive biases in accumulated memory.

v0.24.0: The Wise Mind

Cognitive biases are systematic patterns of deviation from rationality in
judgment — regularities in the errors that reasoning agents make. They arise
from heuristics: mental shortcuts that are efficient in most circumstances but
produce predictable distortions when applied outside their optimal range.
Recognising one's own biases is a foundational metacognitive skill: without it,
an agent cannot distinguish reliable from unreliable inference, or distinguish
genuine learning from mere reinforcement of existing prejudice.

BiasDetector operationalises bias detection for the memory store: it maintains
a lexicon of 10 bias indicator vocabulary sets and scans all memory content for
their occurrence. For each bias type, it computes a strength score that integrates
frequency (how often bias-indicator tokens appear across memories) with the
importance of the memories in which they appear. Biases that colour many
high-importance memories are strong; biases that appear rarely or only in
low-importance memories are weak.

The 10 bias types detected:
  confirmation_bias     — seeking evidence that validates existing beliefs
  availability_heuristic — treating readily recalled events as representative
  sunk_cost             — justifying continued investment by past expenditure
  optimism_bias         — overestimating the probability of positive outcomes
  negativity_bias       — weighting negative experiences more heavily than positive
  hindsight_bias        — viewing past events as more predictable than they were
  overconfidence        — holding higher certainty than evidence warrants
  in_group_bias         — favouring in-group members over out-group members
  anchoring             — over-weighting the first piece of information encountered
  framing_effect        — allowing presentation format to distort evaluation

Biological analogue: dual-process theory (Kahneman 2011) — biases arise from
System 1 fast heuristic processing; cognitive bias research (Tversky & Kahneman
1974); motivated reasoning (Kunda 1990); metacognitive bias monitoring in
prefrontal cortex (Fleming & Dolan 2012); debiasing through explicit awareness;
anterior cingulate cortex in error monitoring and self-correction.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Bias lexicon — 10 bias types × ~13 tokens each
# ---------------------------------------------------------------------------

_BIAS_LEXICON: dict[str, frozenset[str]] = {
    "confirmation_bias": frozenset({
        "confirms", "proves", "validates", "supports", "consistent",
        "expected", "affirms", "corroborates", "justifies", "vindicates",
        "agree", "proven", "verify",
    }),
    "availability_heuristic": frozenset({
        "always", "never", "every", "constantly", "typically",
        "usually", "common", "frequent", "often", "mostly",
        "generally", "everywhere", "invariably",
    }),
    "sunk_cost": frozenset({
        "already", "invested", "spent", "committed", "wasted",
        "poured", "continue", "persist", "stop", "quit",
        "lost", "sacrifice", "abandon",
    }),
    "optimism_bias": frozenset({
        "definitely", "certainly", "surely", "guaranteed", "succeed",
        "bound", "win", "confident", "positive", "hopeful",
        "optimistic", "promising", "achieve",
    }),
    "negativity_bias": frozenset({
        "terrible", "awful", "horrible", "disaster", "catastrophe",
        "failure", "worst", "hopeless", "ruined", "destroyed",
        "devastating", "dreadful", "appalling",
    }),
    "hindsight_bias": frozenset({
        "obvious", "clearly", "predictable", "inevitable", "course",
        "knew", "expected", "foresaw", "apparent", "evident",
        "natural", "should", "hindsight",
    }),
    "overconfidence": frozenset({
        "certain", "definite", "impossible", "absolutely", "doubt",
        "perfect", "flawless", "infallible", "undeniable", "unquestionable",
        "undoubtedly", "without", "necessarily",
    }),
    "in_group_bias": frozenset({
        "we", "our", "ours", "them", "they", "those",
        "outsiders", "others", "foreign", "stranger", "enemy",
        "ally", "tribe",
    }),
    "anchoring": frozenset({
        "first", "initial", "originally", "baseline", "starting",
        "reference", "begin", "start", "opening", "default",
        "standard", "foundation", "anchor",
    }),
    "framing_effect": frozenset({
        "gain", "loss", "win", "lose", "save", "risk",
        "safe", "dangerous", "profit", "cost", "benefit",
        "downside", "upside",
    }),
}

# Human-readable display names
_DISPLAY_NAMES: dict[str, str] = {
    "confirmation_bias":      "Confirmation Bias",
    "availability_heuristic": "Availability Heuristic",
    "sunk_cost":              "Sunk Cost Fallacy",
    "optimism_bias":          "Optimism Bias",
    "negativity_bias":        "Negativity Bias",
    "hindsight_bias":         "Hindsight Bias",
    "overconfidence":         "Overconfidence",
    "in_group_bias":          "In-Group Bias",
    "anchoring":              "Anchoring Bias",
    "framing_effect":         "Framing Effect",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BiasInstance:
    """A detected cognitive bias in the memory store."""

    id: str                        # prefixed "bia_"
    name: str                      # e.g. "confirmation_bias"
    display_name: str              # e.g. "Confirmation Bias"
    strength: float                # 0..1
    description: str
    affected_memory_ids: list[str] # up to 10
    created_at: float

    def summary(self) -> str:
        n = len(self.affected_memory_ids)
        return (
            f"BiasInstance [{self.display_name}]  strength={self.strength:.3f}  "
            f"({n} memories)\n"
            f"  {self.id[:12]}: {self.description[:70]}"
        )


@dataclass
class BiasReport:
    """Result of a BiasDetector.detect() call."""

    total_biases: int
    biases: list[BiasInstance]     # sorted by strength desc
    dominant_bias: str             # name of strongest, or "none"
    mean_strength: float
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"BiasReport: {self.total_biases} biases  "
            f"dominant={self.dominant_bias}  "
            f"mean_strength={self.mean_strength:.3f}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for b in self.biases[:5]:
            lines.append(f"  {b.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BiasDetector
# ---------------------------------------------------------------------------


class BiasDetector:
    """Detects cognitive biases in memory by scanning bias-indicator vocabulary.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_strength:
        Minimum strength to include a bias (default 0.05).
    max_biases:
        Maximum number of :class:`BiasInstance` objects to retain (default 10).
    """

    def __init__(
        self,
        memory: Any,
        min_strength: float = 0.05,
        max_biases: int = 10,
    ) -> None:
        self.memory = memory
        self.min_strength = min_strength
        self.max_biases = max_biases
        self._biases: list[BiasInstance] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, domain: Optional[str] = None) -> BiasReport:
        """Detect cognitive biases in accumulated memory.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`BiasReport` with biases sorted by strength descending.
        """
        t0 = time.time()
        all_items = self._collect_all()

        if domain:
            all_items = [
                it for it in all_items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        n_total = max(len(all_items), 1)

        # For each bias type, find affected memories
        detected: list[BiasInstance] = []
        for bias_name, kw_set in _BIAS_LEXICON.items():
            affected: list[tuple[str, float]] = []  # (memory_id, importance)
            for item in all_items:
                content = getattr(item.experience, "content", "") or ""
                importance = getattr(item.experience, "importance", 0.5) or 0.5
                tokens = {
                    w.strip(".,!?;:\"'()")
                    for w in content.lower().split()
                }
                if tokens & kw_set:
                    affected.append((item.id, importance))

            if not affected:
                continue

            unique_ids = list(dict.fromkeys(mid for mid, _ in affected))
            mean_imp = sum(imp for _, imp in affected) / len(affected)
            strength = round(min(1.0, (len(unique_ids) / n_total) * mean_imp), 4)

            if strength < self.min_strength:
                continue

            display = _DISPLAY_NAMES.get(bias_name, bias_name.replace("_", " ").title())
            description = (
                f"'{display}' detected in {len(unique_ids)} memories "
                f"(mean_importance={mean_imp:.2f})"
            )

            detected.append(BiasInstance(
                id="bia_" + uuid.uuid4().hex[:8],
                name=bias_name,
                display_name=display,
                strength=strength,
                description=description,
                affected_memory_ids=unique_ids[:10],
                created_at=time.time(),
            ))

        detected.sort(key=lambda b: b.strength, reverse=True)
        self._biases = detected[: self.max_biases]

        dominant = self._biases[0].name if self._biases else "none"
        mean_s = (
            sum(b.strength for b in self._biases) / len(self._biases)
            if self._biases else 0.0
        )

        return BiasReport(
            total_biases=len(self._biases),
            biases=self._biases,
            dominant_bias=dominant,
            mean_strength=round(mean_s, 4),
            duration_seconds=time.time() - t0,
        )

    def biases_of_type(self, bias_type: str) -> list[BiasInstance]:
        """Return all detected biases of a specific type.

        Args:
            bias_type: One of the 10 bias type names (e.g. "confirmation_bias").

        Returns:
            List of :class:`BiasInstance` matching that type.
        """
        return [b for b in self._biases if b.name == bias_type]

    def most_pervasive(self) -> Optional[BiasInstance]:
        """Return the bias with the highest strength, or ``None`` if none detected."""
        return self._biases[0] if self._biases else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
