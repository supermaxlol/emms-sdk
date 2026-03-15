"""Gap 7 — AbductiveReasoner: hypothesis generation from surprise.

Triggered by: EgoModule surprise events, CalibrationTracker misses,
RealityChecker contradictions.

4 hypothesis generation methods:
1. INVERSION: What if the opposite of my belief is true?
2. ANALOGY: What worked in a similar domain?
3. DECOMPOSITION: Which sub-component of my model failed?
4. CROSS-DOMAIN: Does another domain have this pattern?

NOT using LLM for hypothesis generation. Using structured
transformations on existing beliefs and memories.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Hypothesis:
    """A candidate explanation for a surprising observation."""
    id: str
    observation: str
    explanation: str
    generation_method: str  # "inversion", "analogy", "decomposition", "cross_domain"
    testable: bool = True
    test_plan: str = ""
    confidence: float = 0.3
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    domain: str = "general"
    status: str = "untested"  # untested, confirmed, refuted, inconclusive
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "observation": self.observation,
            "explanation": self.explanation,
            "generation_method": self.generation_method,
            "testable": self.testable,
            "test_plan": self.test_plan,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "domain": self.domain,
            "status": self.status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        return cls(
            id=d["id"],
            observation=d["observation"],
            explanation=d["explanation"],
            generation_method=d["generation_method"],
            testable=d.get("testable", True),
            test_plan=d.get("test_plan", ""),
            confidence=d.get("confidence", 0.3),
            evidence_for=d.get("evidence_for", []),
            evidence_against=d.get("evidence_against", []),
            domain=d.get("domain", "general"),
            status=d.get("status", "untested"),
            timestamp=d.get("timestamp", time.time()),
        )


class AbductiveReasoner:
    """Generates hypotheses when predictions fail or surprising events occur."""

    # Polarity swaps for inversion method
    NEGATIONS: dict[str, str] = {
        "increases": "decreases", "decreases": "increases",
        "better": "worse", "worse": "better",
        "high": "low", "low": "high",
        "strong": "weak", "weak": "strong",
        "positive": "negative", "negative": "positive",
        "always": "never", "never": "always",
        "more": "less", "less": "more",
        "beats": "loses to", "wins": "loses",
        "rising": "falling", "falling": "rising",
        "bullish": "bearish", "bearish": "bullish",
    }

    def __init__(self, *, max_hypotheses: int = 100,
                 max_per_observation: int = 5) -> None:
        self._hypotheses: list[Hypothesis] = []
        self._max_hypotheses = max_hypotheses
        self._max_per_observation = max_per_observation
        self._generation_log: list[dict] = []
        self._total_generated = 0
        self._counter = 0

    # -- main entry point ---------------------------------------------------

    def generate_from_surprise(
        self,
        observation: str,
        failed_prediction: dict | None = None,
        relevant_beliefs: list[str] | None = None,
        relevant_memories: list[dict] | None = None,
        domain: str = "general",
    ) -> list[Hypothesis]:
        """Generate hypotheses for a surprising observation."""
        beliefs = relevant_beliefs or []
        memories = relevant_memories or []
        candidates: list[Hypothesis] = []

        candidates.extend(self._inversion(observation, beliefs, domain))
        candidates.extend(self._analogy(observation, memories, domain))
        if failed_prediction:
            candidates.extend(self._decomposition(observation, failed_prediction, beliefs, domain))
        candidates.extend(self._cross_domain(observation, memories, domain))

        # Rank: testable first, then by parsimony (shorter = better prior)
        ranked = sorted(
            candidates,
            key=lambda h: (h.testable, -len(h.explanation.split())),
            reverse=True,
        )
        kept = ranked[:self._max_per_observation]

        self._hypotheses.extend(kept)
        self._total_generated += len(kept)
        self._enforce_capacity()

        self._generation_log.append({
            "observation": observation[:100],
            "candidates": len(candidates),
            "kept": len(kept),
            "domain": domain,
            "timestamp": time.time(),
        })

        return kept

    # -- generation methods -------------------------------------------------

    def _inversion(self, observation: str, beliefs: list[str],
                   domain: str) -> list[Hypothesis]:
        """For each relevant belief, generate its negation as a hypothesis."""
        hypotheses = []
        for belief in beliefs[:5]:
            negated = self._negate(belief)
            if negated != belief:
                self._counter += 1
                hypotheses.append(Hypothesis(
                    id=f"hyp_inv_{self._counter}",
                    observation=observation,
                    explanation=f"Inversion: What if {negated}?",
                    generation_method="inversion",
                    testable=True,
                    test_plan=f"Check: is '{negated}' consistent with available data?",
                    confidence=0.3,
                    evidence_against=[belief],
                    domain=domain,
                ))
        return hypotheses

    def _analogy(self, observation: str, memories: list[dict],
                 domain: str) -> list[Hypothesis]:
        """Find memories from OTHER domains with similar content."""
        obs_tokens = set(observation.lower().split())
        if not obs_tokens:
            return []

        hypotheses = []
        for mem in memories[:20]:
            mem_domain = mem.get("domain", "unknown")
            mem_content = mem.get("content", "")
            if mem_domain == domain:
                continue  # same domain, not cross-domain analogy

            mem_tokens = set(mem_content.lower().split())
            if not mem_tokens:
                continue
            overlap = obs_tokens & mem_tokens
            jaccard = len(overlap) / max(len(obs_tokens | mem_tokens), 1)
            if jaccard > 0.1 and len(overlap) >= 2:
                self._counter += 1
                hypotheses.append(Hypothesis(
                    id=f"hyp_ana_{self._counter}",
                    observation=observation,
                    explanation=f"Analogy from {mem_domain}: {mem_content[:100]}",
                    generation_method="analogy",
                    testable=True,
                    test_plan=f"Check if pattern from {mem_domain} applies to {domain}",
                    confidence=min(0.5, 0.4 * jaccard * 10),
                    evidence_for=[mem_content[:80]],
                    domain=domain,
                ))
        return sorted(hypotheses, key=lambda h: -h.confidence)[:3]

    def _decomposition(self, observation: str, failed_prediction: dict,
                       beliefs: list[str], domain: str) -> list[Hypothesis]:
        """If prediction failed, which supporting belief was wrong?"""
        pred_content = failed_prediction.get("content", "")
        pred_tokens = set(pred_content.lower().split())
        if not pred_tokens:
            return []

        hypotheses = []
        for belief in beliefs[:5]:
            belief_tokens = set(belief.lower().split())
            overlap = pred_tokens & belief_tokens
            score = len(overlap) / max(len(pred_tokens), 1)
            if score > 0.15:
                self._counter += 1
                hypotheses.append(Hypothesis(
                    id=f"hyp_dec_{self._counter}",
                    observation=observation,
                    explanation=f"Possibly wrong premise: {belief[:80]}",
                    generation_method="decomposition",
                    testable=True,
                    test_plan=f"Verify independently: {belief[:60]}",
                    confidence=min(0.5, 0.5 * score),
                    evidence_against=[pred_content[:60]],
                    domain=domain,
                ))
        return hypotheses

    def _cross_domain(self, observation: str, memories: list[dict],
                      domain: str) -> list[Hypothesis]:
        """Check if the observation's structural pattern exists elsewhere."""
        structural_words = {
            "causes", "leads", "prevents", "enables", "blocks",
            "increases", "decreases", "correlates", "predicts",
            "triggers", "follows", "precedes",
        }
        obs_structure = set(observation.lower().split()) & structural_words
        if not obs_structure:
            return []

        hypotheses = []
        for mem in memories[:20]:
            content = mem.get("content", "")
            mem_domain = mem.get("domain", "unknown")
            if mem_domain == domain:
                continue
            mem_structure = set(content.lower().split()) & structural_words
            shared = obs_structure & mem_structure
            if shared:
                self._counter += 1
                hypotheses.append(Hypothesis(
                    id=f"hyp_xdom_{self._counter}",
                    observation=observation,
                    explanation=f"Cross-domain from {mem_domain}: {content[:80]}",
                    generation_method="cross_domain",
                    testable=False,
                    confidence=0.2,
                    evidence_for=[content[:60]],
                    domain=domain,
                ))
        return hypotheses[:2]

    def _negate(self, statement: str) -> str:
        """Simple keyword-based polarity swap."""
        result = statement
        for word, neg in self.NEGATIONS.items():
            lower = result.lower()
            if word in lower:
                idx = lower.index(word)
                result = result[:idx] + neg + result[idx + len(word):]
                break
        return result

    # -- hypothesis management ----------------------------------------------

    def update_hypothesis(self, hypothesis_id: str, evidence: str,
                          supports: bool) -> bool:
        """Add evidence for/against a hypothesis. Returns True if found."""
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                if supports:
                    h.evidence_for.append(evidence)
                    h.confidence = min(h.confidence + 0.1, 0.95)
                else:
                    h.evidence_against.append(evidence)
                    h.confidence = max(h.confidence - 0.15, 0.05)
                # Auto-resolve
                if len(h.evidence_for) >= 3 and h.confidence > 0.7:
                    h.status = "confirmed"
                elif len(h.evidence_against) >= 3 and h.confidence < 0.2:
                    h.status = "refuted"
                return True
        return False

    def resolve_hypothesis(self, hypothesis_id: str,
                           status: str = "confirmed") -> bool:
        """Manually resolve a hypothesis."""
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                h.status = status
                return True
        return False

    def active_hypotheses(self) -> list[Hypothesis]:
        """Return untested hypotheses sorted by confidence desc."""
        return sorted(
            [h for h in self._hypotheses if h.status == "untested"],
            key=lambda h: -h.confidence,
        )

    def hypotheses_by_status(self, status: str) -> list[Hypothesis]:
        return [h for h in self._hypotheses if h.status == status]

    @property
    def all_hypotheses(self) -> list[Hypothesis]:
        return list(self._hypotheses)

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        by_status = {}
        for h in self._hypotheses:
            by_status[h.status] = by_status.get(h.status, 0) + 1
        by_method = {}
        for h in self._hypotheses:
            by_method[h.generation_method] = by_method.get(h.generation_method, 0) + 1
        parts = [
            f"AbductiveReasoner: {self._total_generated} generated, "
            f"{len(self._hypotheses)} in history"
        ]
        if by_status:
            parts.append("  by status: " + ", ".join(
                f"{k}={v}" for k, v in sorted(by_status.items())))
        if by_method:
            parts.append("  by method: " + ", ".join(
                f"{k}={v}" for k, v in sorted(by_method.items())))
        return "\n".join(parts)

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_generated": self._total_generated,
            "counter": self._counter,
            "hypotheses": [h.to_dict() for h in self._hypotheses[-self._max_hypotheses:]],
            "generation_log": self._generation_log[-50:],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_generated = data.get("total_generated", 0)
            self._counter = data.get("counter", 0)
            self._hypotheses = [Hypothesis.from_dict(h) for h in data.get("hypotheses", [])]
            self._generation_log = data.get("generation_log", [])
            return True
        except Exception:
            return False

    # -- internal -----------------------------------------------------------

    def _enforce_capacity(self) -> None:
        if len(self._hypotheses) > self._max_hypotheses:
            # Keep confirmed/refuted + most recent untested
            resolved = [h for h in self._hypotheses if h.status in ("confirmed", "refuted")]
            untested = sorted(
                [h for h in self._hypotheses if h.status == "untested"],
                key=lambda h: -h.timestamp,
            )
            self._hypotheses = resolved + untested[:self._max_hypotheses - len(resolved)]
