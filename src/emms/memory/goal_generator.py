"""Gap 4 — GoalGenerator: autonomous goal formation from internal state.

Five goal sources:
1. CURIOSITY: low capability + high belief count = confusion → investigate
2. CALIBRATION: poor Brier scores → practice predictions
3. AFFECT: unresolved negative somatic markers → revisit
4. MAINTENANCE: memory health (large domains) → consolidate
5. OBLIGATION: stalling human goals → follow up

Human goals ALWAYS take priority. Generated goals fill gaps in attention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeneratedGoal:
    """A goal created by the system, not by the human."""
    description: str
    source: str  # "curiosity", "calibration", "affect", "maintenance", "obligation"
    domain: str = "general"
    priority: float = 0.5  # 0-1
    reasoning: str = ""
    parent_goal: str = ""
    deadline: float | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "source": self.source,
            "domain": self.domain,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "parent_goal": self.parent_goal,
            "deadline": self.deadline,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneratedGoal:
        return cls(
            description=d["description"],
            source=d["source"],
            domain=d.get("domain", "general"),
            priority=d.get("priority", 0.5),
            reasoning=d.get("reasoning", ""),
            parent_goal=d.get("parent_goal", ""),
            deadline=d.get("deadline"),
            timestamp=d.get("timestamp", time.time()),
        )


class GoalGenerator:
    """Generates goals from internal state, not just external commands.

    Each generation cycle runs all 5 sources, deduplicates, filters
    suppressed goals, and keeps the top N by priority.
    """

    def __init__(self, *, max_goals: int = 10) -> None:
        self._generated: list[GeneratedGoal] = []
        self._max_goals = max_goals
        self._suppressed: set[str] = set()
        self._generation_log: list[dict] = []
        self._total_generations = 0

    # -- main generation ----------------------------------------------------

    def generate(self, *,
                 capability_profile: dict[str, float] | None = None,
                 belief_counts: dict[str, int] | None = None,
                 calibration_report: dict | None = None,
                 somatic_markers: list[dict] | None = None,
                 memory_domain_counts: dict[str, int] | None = None,
                 human_goals: list[dict] | None = None) -> list[GeneratedGoal]:
        """Run all 5 goal generators, deduplicate, rank by priority."""
        self._total_generations += 1
        candidates: list[GeneratedGoal] = []

        candidates.extend(self._curiosity_goals(
            capability_profile or {}, belief_counts or {}))
        candidates.extend(self._calibration_goals(calibration_report or {}))
        candidates.extend(self._affect_goals(somatic_markers or []))
        candidates.extend(self._maintenance_goals(memory_domain_counts or {}))
        candidates.extend(self._obligation_goals(human_goals or []))

        # Deduplicate
        unique = self._deduplicate(candidates)

        # Filter suppressed
        filtered = [g for g in unique if g.description not in self._suppressed]

        # Rank by priority
        ranked = sorted(filtered, key=lambda g: -g.priority)
        self._generated = ranked[:self._max_goals]

        self._generation_log.append({
            "timestamp": time.time(),
            "candidates": len(candidates),
            "unique": len(unique),
            "filtered": len(filtered),
            "final": len(self._generated),
        })

        return list(self._generated)

    # -- goal sources -------------------------------------------------------

    def _curiosity_goals(self, capability: dict[str, float],
                         beliefs: dict[str, int]) -> list[GeneratedGoal]:
        """Domains with low capability but many beliefs = confusion."""
        goals = []
        for domain, cap in capability.items():
            count = beliefs.get(domain, 0)
            if cap < 0.4 and count > 5:
                goals.append(GeneratedGoal(
                    description=f"Investigate {domain} — low capability ({cap:.0%}) despite {count} beliefs",
                    source="curiosity",
                    domain=domain,
                    priority=0.6 * (1 - cap),
                    reasoning=f"Many beliefs but low competence suggests confusion or contradictions",
                ))
        return goals

    def _calibration_goals(self, report: dict) -> list[GeneratedGoal]:
        """Domains with poor prediction calibration."""
        goals = []
        domain_scores = report.get("domain_scores", {})
        for domain, score_data in domain_scores.items():
            brier = score_data if isinstance(score_data, (int, float)) else score_data.get("brier", 0)
            if brier > 0.3:
                goals.append(GeneratedGoal(
                    description=f"Improve {domain} prediction calibration (Brier={brier:.2f})",
                    source="calibration",
                    domain=domain,
                    priority=0.5 * min(brier, 1.0),
                    reasoning=f"Brier score {brier:.2f} > 0.3 threshold — poorly calibrated",
                ))
        return goals

    def _affect_goals(self, markers: list[dict]) -> list[GeneratedGoal]:
        """Somatic markers with unresolved negative valence."""
        goals = []
        for marker in markers:
            valence = marker.get("valence", 0)
            strength = marker.get("strength", 0)
            if valence < -0.3 and strength > 0.5:
                ctx = marker.get("context", marker.get("context_tokens", "unknown"))
                goals.append(GeneratedGoal(
                    description=f"Resolve negatively-marked context: {str(ctx)[:60]}",
                    source="affect",
                    domain="reflection",
                    priority=0.4 * abs(valence),
                    reasoning=f"Somatic marker valence={valence:.2f}, strength={strength:.2f}",
                ))
        return goals

    def _maintenance_goals(self, domain_counts: dict[str, int]) -> list[GeneratedGoal]:
        """Large domains that may benefit from consolidation."""
        goals = []
        for domain, count in domain_counts.items():
            if count > 100:
                goals.append(GeneratedGoal(
                    description=f"Consolidate {domain} memories ({count} items)",
                    source="maintenance",
                    domain=domain,
                    priority=0.3 * min(count / 500, 1.0),
                    reasoning=f"{count} memories in {domain} — may benefit from consolidation",
                ))
        return goals

    def _obligation_goals(self, human_goals: list[dict]) -> list[GeneratedGoal]:
        """Human goals that are stalling."""
        goals = []
        for goal in human_goals:
            created = goal.get("created_at", time.time())
            age_days = (time.time() - created) / 86400
            status = goal.get("status", "active")
            if age_days > 3 and status not in ("completed", "done"):
                desc = goal.get("description", "unknown")[:60]
                goals.append(GeneratedGoal(
                    description=f"Follow up on stalling goal: {desc}",
                    source="obligation",
                    domain=goal.get("domain", "general"),
                    priority=0.7 * min(age_days / 14, 1.0),
                    reasoning=f"Created {age_days:.0f} days ago, status='{status}'",
                ))
        return goals

    # -- deduplication & suppression ----------------------------------------

    def _deduplicate(self, goals: list[GeneratedGoal]) -> list[GeneratedGoal]:
        """Remove near-duplicate goals by Jaccard on description tokens."""
        if not goals:
            return []
        unique = [goals[0]]
        for g in goals[1:]:
            g_tokens = set(g.description.lower().split())
            is_dup = False
            for u in unique:
                u_tokens = set(u.description.lower().split())
                union = g_tokens | u_tokens
                if not union:
                    continue
                jaccard = len(g_tokens & u_tokens) / len(union)
                if jaccard > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(g)
        return unique

    def suppress(self, description: str) -> None:
        """Human says 'not now' to a generated goal."""
        self._suppressed.add(description)

    def unsuppress(self, description: str) -> bool:
        """Remove a suppression. Returns True if was suppressed."""
        try:
            self._suppressed.remove(description)
            return True
        except KeyError:
            return False

    # -- accessors ----------------------------------------------------------

    @property
    def goals(self) -> list[GeneratedGoal]:
        return list(self._generated)

    @property
    def suppressed(self) -> set[str]:
        return set(self._suppressed)

    def generation_log(self, limit: int = 20) -> list[dict]:
        return list(self._generation_log[-limit:])

    def summary(self) -> str:
        sources = {}
        for g in self._generated:
            sources[g.source] = sources.get(g.source, 0) + 1
        parts = [f"GoalGenerator: {len(self._generated)} goals, "
                 f"{self._total_generations} generations, "
                 f"{len(self._suppressed)} suppressed"]
        if sources:
            parts.append("  by source: " + ", ".join(
                f"{k}={v}" for k, v in sorted(sources.items())))
        return "\n".join(parts)

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        data = {
            "version": "0.28.0",
            "total_generations": self._total_generations,
            "generated": [g.to_dict() for g in self._generated],
            "suppressed": list(self._suppressed),
            "generation_log": self._generation_log[-50:],
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._total_generations = data.get("total_generations", 0)
            self._generated = [GeneratedGoal.from_dict(g) for g in data.get("generated", [])]
            self._suppressed = set(data.get("suppressed", []))
            self._generation_log = data.get("generation_log", [])
            return True
        except Exception:
            return False
