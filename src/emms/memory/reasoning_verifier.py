"""Gap 7 — ReasoningVerifier: check reasoning chains for logical issues.

Heuristic pattern matching (not formal logic):
1. Premise support: claims backed by memory evidence?
2. Circularity: conclusion appears in premises?
3. Relevance: consecutive steps topically connected?
4. Overgeneralization: absolute claims without qualification?
5. Contradiction: chain contradicts stored beliefs?
6. Counter-evidence: is counter-evidence addressed?

This is a reasoning linter, not a theorem prover.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LogicalIssue:
    """A detected problem in a reasoning chain."""
    issue_type: str  # "unsupported_premise", "circular", "non_sequitur",
                     # "overgeneralization", "missing_step", "contradiction",
                     # "ignored_counterevidence"
    location: str    # which step has the problem
    description: str
    severity: str = "minor"  # minor, major, fatal
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "location": self.location,
            "description": self.description,
            "severity": self.severity,
            "suggested_fix": self.suggested_fix,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LogicalIssue:
        return cls(
            issue_type=d["issue_type"],
            location=d["location"],
            description=d["description"],
            severity=d.get("severity", "minor"),
            suggested_fix=d.get("suggested_fix", ""),
        )


class ReasoningVerifier:
    """Checks reasoning chains for logical issues."""

    ABSOLUTE_WORDS = frozenset({
        "always", "never", "all", "none", "every", "impossible",
        "certainly", "definitely", "guaranteed", "proven",
    })

    NEGATION_WORDS = frozenset({
        "not", "no", "never", "without", "fail", "wrong",
        "incorrect", "false", "opposite",
    })

    def __init__(self, *, max_log: int = 200) -> None:
        self._verification_log: list[dict] = []
        self._max_log = max_log
        self._total_verifications = 0

    # -- main entry ---------------------------------------------------------

    def verify_chain(
        self,
        steps: list[str],
        hypothesis: str = "",
        evidence_for: list[str] | None = None,
        evidence_against: list[str] | None = None,
        memory_contents: list[str] | None = None,
        beliefs: list[str] | None = None,
    ) -> list[LogicalIssue]:
        """Run all checks on a reasoning chain. Returns list of issues."""
        self._total_verifications += 1
        issues: list[LogicalIssue] = []

        issues.extend(self._check_support(steps, memory_contents or []))
        issues.extend(self._check_circularity(steps, hypothesis))
        issues.extend(self._check_relevance(steps))
        issues.extend(self._check_overgeneralization(steps + ([hypothesis] if hypothesis else [])))
        if beliefs:
            issues.extend(self._check_contradictions(hypothesis, beliefs))
        if evidence_against and not self._counter_evidence_addressed(steps):
            issues.append(LogicalIssue(
                issue_type="ignored_counterevidence",
                location="chain",
                description=f"{len(evidence_against)} piece(s) of counter-evidence not addressed",
                severity="major",
            ))

        self._verification_log.append({
            "steps_count": len(steps),
            "hypothesis": hypothesis[:60],
            "issues_found": len(issues),
            "fatal": sum(1 for i in issues if i.severity == "fatal"),
            "timestamp": time.time(),
        })
        if len(self._verification_log) > self._max_log:
            self._verification_log = self._verification_log[-self._max_log:]

        return issues

    # -- individual checks --------------------------------------------------

    def _check_support(self, steps: list[str],
                       memory_contents: list[str]) -> list[LogicalIssue]:
        """Are claims backed by memory evidence?"""
        if not memory_contents:
            return []

        memory_tokens = set()
        for content in memory_contents[:50]:
            memory_tokens.update(content.lower().split())

        issues = []
        for i, step in enumerate(steps):
            step_tokens = set(step.lower().split())
            if len(step_tokens) <= 5:
                continue  # too short to evaluate
            overlap = len(step_tokens & memory_tokens) / max(len(step_tokens), 1)
            if overlap < 0.2:
                issues.append(LogicalIssue(
                    issue_type="unsupported_premise",
                    location=f"step_{i}",
                    description=f"Low memory support ({overlap:.0%}): {step[:60]}",
                    severity="major",
                    suggested_fix="Find supporting evidence or flag as assumption",
                ))
        return issues

    def _check_circularity(self, steps: list[str],
                           hypothesis: str) -> list[LogicalIssue]:
        """Does the conclusion appear in the premises?"""
        if not hypothesis:
            return []
        hyp_tokens = set(hypothesis.lower().split())
        if not hyp_tokens:
            return []

        issues = []
        for i, step in enumerate(steps):
            step_tokens = set(step.lower().split())
            if not step_tokens:
                continue
            overlap = len(hyp_tokens & step_tokens) / max(len(hyp_tokens), 1)
            if overlap > 0.8:
                issues.append(LogicalIssue(
                    issue_type="circular",
                    location=f"step_{i}",
                    description=f"Step {i} restates the conclusion ({overlap:.0%} overlap)",
                    severity="fatal",
                    suggested_fix="Find independent evidence for this step",
                ))
        return issues

    def _check_relevance(self, steps: list[str]) -> list[LogicalIssue]:
        """Are consecutive steps topically connected?"""
        issues = []
        for i in range(1, len(steps)):
            prev_tokens = set(steps[i - 1].lower().split())
            curr_tokens = set(steps[i].lower().split())
            if len(prev_tokens) < 4 or len(curr_tokens) < 4:
                continue
            union = prev_tokens | curr_tokens
            if not union:
                continue
            overlap = len(prev_tokens & curr_tokens) / len(union)
            if overlap < 0.05:
                issues.append(LogicalIssue(
                    issue_type="non_sequitur",
                    location=f"step_{i - 1}→step_{i}",
                    description=f"Topic jump between steps ({overlap:.0%} overlap)",
                    severity="minor",
                    suggested_fix="Add bridging step connecting these ideas",
                ))
        return issues

    def _check_overgeneralization(self, texts: list[str]) -> list[LogicalIssue]:
        """Flag absolute claims without qualification."""
        issues = []
        for i, text in enumerate(texts):
            words = set(text.lower().split())
            found = words & self.ABSOLUTE_WORDS
            if found:
                issues.append(LogicalIssue(
                    issue_type="overgeneralization",
                    location=f"item_{i}",
                    description=f"Absolute claim ({', '.join(found)}): {text[:60]}",
                    severity="minor",
                    suggested_fix="Qualify with 'typically', 'in most cases', or cite evidence",
                ))
        return issues

    def _check_contradictions(self, hypothesis: str,
                              beliefs: list[str]) -> list[LogicalIssue]:
        """Does the hypothesis contradict stored beliefs?"""
        if not hypothesis:
            return []
        hyp_tokens = set(hypothesis.lower().split())
        hyp_negations = hyp_tokens & self.NEGATION_WORDS

        issues = []
        for belief in beliefs[:20]:
            belief_tokens = set(belief.lower().split())
            overlap = hyp_tokens & belief_tokens
            if len(overlap) <= 3:
                continue
            belief_negations = belief_tokens & self.NEGATION_WORDS
            # One has negation, other doesn't → potential contradiction
            if bool(hyp_negations) != bool(belief_negations):
                issues.append(LogicalIssue(
                    issue_type="contradiction",
                    location="hypothesis",
                    description=f"May contradict belief: {belief[:60]}",
                    severity="major",
                    suggested_fix="Reconcile with existing belief or update belief",
                ))
        return issues

    def _counter_evidence_addressed(self, steps: list[str]) -> bool:
        """Check if any step acknowledges counter-evidence."""
        hedge_words = {"however", "despite", "although", "nevertheless",
                       "but", "conversely", "counter", "exception"}
        for step in steps:
            if set(step.lower().split()) & hedge_words:
                return True
        return False

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        return (
            f"ReasoningVerifier: {self._total_verifications} verifications, "
            f"{len(self._verification_log)} in log"
        )

    def verification_log(self, limit: int = 20) -> list[dict]:
        return list(self._verification_log[-limit:])

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_verifications": self._total_verifications,
            "verification_log": self._verification_log[-self._max_log:],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_verifications = data.get("total_verifications", 0)
            self._verification_log = data.get("verification_log", [])
            return True
        except Exception:
            return False
