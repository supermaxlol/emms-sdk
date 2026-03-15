"""Gap 5 — RealityChecker: verify beliefs against live world data.

Takes a belief (text + domain + confidence) and attempts to find
contradicting or confirming evidence from WorldSensor readings
and memory contents.  Produces a VerificationResult with an
updated confidence and explanation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emms.memory.world_sensor import WorldSensor


class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    STALE = "stale"  # belief is about outdated state


@dataclass
class VerificationResult:
    """Outcome of checking a belief against reality."""
    belief_text: str
    status: VerificationStatus
    confidence_delta: float  # positive = more confident, negative = less
    evidence: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    domain: str = "general"

    def to_dict(self) -> dict:
        return {
            "belief_text": self.belief_text,
            "status": self.status.value,
            "confidence_delta": self.confidence_delta,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerificationResult:
        return cls(
            belief_text=d["belief_text"],
            status=VerificationStatus(d["status"]),
            confidence_delta=d["confidence_delta"],
            evidence=d.get("evidence", []),
            timestamp=d.get("timestamp", time.time()),
            domain=d.get("domain", "general"),
        )


class RealityChecker:
    """Cross-references beliefs and predictions with observable reality.

    Strategies:
    1. **File-based verification**: If a belief mentions a file path,
       check whether it exists and its properties match.
    2. **Temporal verification**: If a belief has a timestamp, check
       whether it's stale.
    3. **Token overlap**: Compare belief tokens against recent world
       readings for supporting/contradicting evidence.
    4. **Memory cross-reference**: Check if memory items contradict the belief.
    """

    def __init__(self, *, sensor: WorldSensor | None = None,
                 memory: Any = None,
                 max_history: int = 200) -> None:
        self._sensor = sensor
        self._memory = memory
        self._results: list[VerificationResult] = []
        self._max_history = max_history
        self._total_checks = 0

    @property
    def sensor(self) -> WorldSensor | None:
        return self._sensor

    @sensor.setter
    def sensor(self, s: WorldSensor) -> None:
        self._sensor = s

    # -- main check ---------------------------------------------------------

    def check(self, belief_text: str, domain: str = "general",
              current_confidence: float = 0.5) -> VerificationResult:
        """Verify a belief and return the result with confidence adjustment."""
        self._total_checks += 1
        evidence: list[str] = []
        delta = 0.0

        # Strategy 1: file-path checking
        file_delta, file_evidence = self._check_file_references(belief_text)
        delta += file_delta
        evidence.extend(file_evidence)

        # Strategy 2: temporal staleness
        stale_delta, stale_evidence = self._check_staleness(belief_text)
        delta += stale_delta
        evidence.extend(stale_evidence)

        # Strategy 3: token overlap with sensor readings
        sensor_delta, sensor_evidence = self._check_sensor_overlap(belief_text)
        delta += sensor_delta
        evidence.extend(sensor_evidence)

        # Strategy 4: memory cross-reference
        memory_delta, memory_evidence = self._check_memory_cross_ref(belief_text, domain)
        delta += memory_delta
        evidence.extend(memory_evidence)

        # Determine status
        if delta > 0.1:
            status = VerificationStatus.CONFIRMED
        elif delta < -0.1:
            status = VerificationStatus.CONTRADICTED
        elif any("stale" in e.lower() for e in evidence):
            status = VerificationStatus.STALE
        else:
            status = VerificationStatus.UNCERTAIN

        # Clamp delta
        delta = max(-0.5, min(0.5, delta))

        result = VerificationResult(
            belief_text=belief_text,
            status=status,
            confidence_delta=round(delta, 3),
            evidence=evidence,
            domain=domain,
        )
        self._record(result)
        return result

    def batch_check(self, beliefs: list[dict]) -> list[VerificationResult]:
        """Check multiple beliefs. Each dict: {text, domain?, confidence?}."""
        results = []
        for b in beliefs:
            r = self.check(
                b["text"],
                domain=b.get("domain", "general"),
                current_confidence=b.get("confidence", 0.5),
            )
            results.append(r)
        return results

    # -- verification strategies --------------------------------------------

    def _check_file_references(self, text: str) -> tuple[float, list[str]]:
        """Look for file paths in the belief and verify they exist."""
        if self._sensor is None:
            return 0.0, []

        # Simple path detection: /foo/bar or ~/foo
        paths = re.findall(r'(?:/[\w./-]+|~/[\w./-]+)', text)
        if not paths:
            return 0.0, []

        delta = 0.0
        evidence = []
        for p in paths[:3]:  # limit to 3 paths
            p_expanded = str(Path(p).expanduser())
            reading = self._sensor.sense_filesystem(p_expanded)
            exists = reading.value.get("exists", False)
            if exists:
                delta += 0.15
                size = reading.value.get("size_bytes", 0)
                evidence.append(f"file '{p}' exists ({size} bytes)")
            else:
                delta -= 0.2
                evidence.append(f"file '{p}' does not exist")
        return delta, evidence

    def _check_staleness(self, text: str) -> tuple[float, list[str]]:
        """Check if the belief references old timestamps."""
        # Look for Unix timestamps or ISO dates in the text
        unix_matches = re.findall(r'\b(1[6-9]\d{8})\b', text)
        now = time.time()
        for ts_str in unix_matches[:2]:
            ts = float(ts_str)
            age_days = (now - ts) / 86400
            if age_days > 30:
                return -0.15, [f"references timestamp {age_days:.0f} days old — stale"]
            elif age_days > 7:
                return -0.05, [f"references timestamp {age_days:.0f} days old — aging"]
        return 0.0, []

    def _check_sensor_overlap(self, text: str) -> tuple[float, list[str]]:
        """Compare belief tokens against recent sensor readings."""
        if self._sensor is None:
            return 0.0, []

        text_tokens = set(text.lower().split())
        if not text_tokens:
            return 0.0, []

        readings = self._sensor.history(limit=20)
        if not readings:
            return 0.0, []

        delta = 0.0
        evidence = []
        for reading in readings:
            value_str = str(reading.value).lower()
            value_tokens = set(value_str.split())
            overlap = text_tokens & value_tokens
            if len(overlap) >= 2:
                delta += 0.05
                evidence.append(
                    f"sensor '{reading.key}' shares tokens: {', '.join(list(overlap)[:5])}"
                )
        return min(delta, 0.2), evidence[:3]

    def _check_memory_cross_ref(self, text: str, domain: str) -> tuple[float, list[str]]:
        """Check memory for contradicting or confirming content."""
        if self._memory is None:
            return 0.0, []

        text_tokens = set(text.lower().split())
        if not text_tokens:
            return 0.0, []

        delta = 0.0
        evidence = []

        # Search recent memory items
        try:
            items = list(self._memory.long_term.values())[-50:]
            for item in items:
                content = item.experience.content.lower()
                content_tokens = set(content.split())
                overlap = text_tokens & content_tokens
                if len(overlap) >= 3:
                    # Check for negation patterns
                    negation_words = {"not", "no", "never", "false", "wrong", "incorrect"}
                    if negation_words & content_tokens and overlap - negation_words:
                        delta -= 0.1
                        evidence.append(f"memory contradicts: '{item.experience.content[:80]}...'")
                    else:
                        delta += 0.05
                        evidence.append(f"memory supports: '{item.experience.content[:60]}...'")
        except Exception:
            pass

        return max(-0.3, min(0.2, delta)), evidence[:3]

    # -- history & reporting ------------------------------------------------

    def results(self, limit: int = 20) -> list[VerificationResult]:
        return list(self._results[-limit:])

    def accuracy_report(self) -> dict:
        """Breakdown of verification outcomes."""
        counts = {s.value: 0 for s in VerificationStatus}
        for r in self._results:
            counts[r.status.value] += 1
        return {
            "total_checks": self._total_checks,
            "results_in_history": len(self._results),
            "status_counts": counts,
            "avg_confidence_delta": (
                round(sum(r.confidence_delta for r in self._results) / len(self._results), 3)
                if self._results else 0.0
            ),
        }

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_checks": self._total_checks,
            "results": [r.to_dict() for r in self._results[-self._max_history:]],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_checks = data.get("total_checks", 0)
            self._results = [VerificationResult.from_dict(r) for r in data.get("results", [])]
            return True
        except Exception:
            return False

    def summary(self) -> str:
        report = self.accuracy_report()
        parts = [f"RealityChecker: {report['total_checks']} checks"]
        for status, count in report["status_counts"].items():
            if count > 0:
                parts.append(f"  {status}: {count}")
        if report["avg_confidence_delta"]:
            parts.append(f"  avg delta: {report['avg_confidence_delta']:+.3f}")
        return "\n".join(parts)

    # -- internal -----------------------------------------------------------

    def _record(self, result: VerificationResult) -> None:
        self._results.append(result)
        if len(self._results) > self._max_history:
            self._results = self._results[-self._max_history:]
