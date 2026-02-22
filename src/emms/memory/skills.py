"""SkillDistiller — extract reusable procedural skills from episodic memory.

v0.20.0: The Reasoning Mind

Human expertise is not merely the accumulation of facts — it is the distillation
of experience into transferable *skills*: abstract, reusable procedures that can
be applied across novel situations. When a chess player learns not just a specific
game sequence but the general principle of controlling the centre, they have
abstracted a skill from experience. When a surgeon refines a technique across
hundreds of operations, they are distilling procedural knowledge.

Skill abstraction transforms raw episodic memories into compact, transferable
competencies. It identifies recurring action patterns — sequences of context →
action → outcome — and compresses them into reusable templates. The resulting
skills can be matched against new goals, enabling the agent to draw on its
procedural history rather than treating every new problem as novel.

SkillDistiller operationalises this for the memory store: it scans accumulated
memories for action tokens (improve, build, learn, implement, design, etc.),
extracts the contextual tokens that precede each action (preconditions) and the
outcome tokens that follow (effects), groups patterns by domain and action type,
and synthesises skills from recurring patterns. Skills are scored by confidence
— a blend of how frequently the pattern appeared and how strongly it was
remembered. The best_skill() method enables goal-directed skill retrieval via
token Jaccard overlap.

Biological analogue: procedural learning and skill abstraction (Fitts 1964 —
the automatisation of cognitive sequences); basal ganglia in habit and skill
formation (Graybiel 2008); episodic-to-procedural transfer (Cohen & Squire
1980); chunking of procedural sequences in motor cortex (Sakai et al. 2004);
the cognitive compilation theory of skill acquisition (Anderson 1982) — skills
emerge from repeated interpretation of declarative knowledge.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_ACTION_TOKENS: frozenset[str] = frozenset({
    "improve", "reduce", "increase", "enable", "build", "create",
    "learn", "apply", "practice", "develop", "analyze", "optimize",
    "implement", "design", "test", "solve", "train", "strengthen",
    "generate", "construct", "establish",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DistilledSkill:
    """A reusable procedural skill distilled from recurring memory patterns."""

    id: str                         # prefixed "skill_"
    name: str                       # action token label
    domain: str
    description: str                # synthesised skill description
    preconditions: list[str]        # contextual tokens before action
    outcomes: list[str]             # contextual tokens after action
    confidence: float               # 0..1 — frequency × strength blend
    source_memory_ids: list[str]    # supporting memories
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        pre = ", ".join(self.preconditions) or "—"
        out = ", ".join(self.outcomes) or "—"
        return (
            f"DistilledSkill [{self.domain}] '{self.name}'  "
            f"confidence={self.confidence:.3f}\n"
            f"  {self.id[:12]}: pre=[{pre}] → out=[{out}]"
        )


@dataclass
class SkillReport:
    """Result of a SkillDistiller.distill() call."""

    total_memories_analyzed: int
    skills_distilled: int
    skills: list[DistilledSkill]    # sorted by confidence desc
    domains_covered: list[str]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"SkillReport: {self.skills_distilled} skills from "
            f"{self.total_memories_analyzed} memories  "
            f"domains={self.domains_covered}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for sk in self.skills[:5]:
            lines.append(f"  {sk.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SkillDistiller
# ---------------------------------------------------------------------------


class SkillDistiller:
    """Distils reusable procedural skills from recurring memory action patterns.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_skill_frequency:
        Minimum number of memories exhibiting a pattern for it to become a
        skill (default 2).
    max_skills:
        Maximum number of :class:`DistilledSkill` objects to generate per
        call (default 10).
    store_skills:
        If ``True``, persist each skill as a ``"skill"`` domain memory
        (default True).
    skill_importance:
        Importance assigned to skill memories when stored (default 0.7).
    """

    def __init__(
        self,
        memory: Any,
        min_skill_frequency: int = 2,
        max_skills: int = 10,
        store_skills: bool = True,
        skill_importance: float = 0.7,
    ) -> None:
        self.memory = memory
        self.min_skill_frequency = min_skill_frequency
        self.max_skills = max_skills
        self.store_skills = store_skills
        self.skill_importance = skill_importance
        self._skills: list[DistilledSkill] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill(self, domain: Optional[str] = None) -> SkillReport:
        """Distil procedural skills from accumulated memories.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`SkillReport` with skills sorted by confidence descending.
        """
        t0 = time.time()
        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        if not items:
            return SkillReport(
                total_memories_analyzed=0,
                skills_distilled=0,
                skills=[],
                domains_covered=[],
                duration_seconds=time.time() - t0,
            )

        n_items = len(items)
        patterns = self._extract_action_patterns(items)

        # Group patterns by (domain, action_token)
        # Each entry: list of (preconditions, outcomes, memory_id, strength)
        grouped: dict[tuple[str, str], list[tuple[list[str], list[str], str, float]]] = defaultdict(list)
        for action, pre, out, mem_id, strength, dom in patterns:
            grouped[(dom, action)].append((pre, out, mem_id, strength))

        new_skills: list[DistilledSkill] = []
        for (dom, action), entries in grouped.items():
            if len(new_skills) >= self.max_skills:
                break
            freq = len(entries)
            if freq < self.min_skill_frequency:
                continue

            # Aggregate preconditions and outcomes across occurrences
            all_pre: list[str] = []
            all_out: list[str] = []
            mem_ids: list[str] = []
            strengths: list[float] = []
            for pre, out, mid, st in entries:
                all_pre.extend(pre)
                all_out.extend(out)
                if mid not in mem_ids:
                    mem_ids.append(mid)
                strengths.append(st)

            # Top-3 most common tokens
            top_pre = [tok for tok, _ in Counter(all_pre).most_common(3)]
            top_out = [tok for tok, _ in Counter(all_out).most_common(3)]

            freq_ratio = min(1.0, freq / n_items)
            mean_strength = sum(strengths) / len(strengths) if strengths else 0.0
            confidence = round(freq_ratio * 0.6 + mean_strength * 0.4, 4)

            pre_str = ", ".join(top_pre) if top_pre else "context"
            out_str = ", ".join(top_out) if top_out else "result"
            description = (
                f"Skill '{action}' in {dom}: apply {action} using "
                f"[{pre_str}] to achieve [{out_str}]. "
                f"Observed in {freq} memories."
            )

            skill = DistilledSkill(
                id=f"skill_{uuid.uuid4().hex[:8]}",
                name=action,
                domain=dom,
                description=description,
                preconditions=top_pre,
                outcomes=top_out,
                confidence=confidence,
                source_memory_ids=mem_ids[:5],
            )
            if self.store_skills:
                self._store_skill(skill)
            new_skills.append(skill)
            self._skills.append(skill)

        new_skills.sort(key=lambda s: s.confidence, reverse=True)

        domains_covered = sorted({s.domain for s in new_skills})

        return SkillReport(
            total_memories_analyzed=n_items,
            skills_distilled=len(new_skills),
            skills=new_skills,
            domains_covered=domains_covered,
            duration_seconds=time.time() - t0,
        )

    def skills_for_domain(self, domain: str) -> list[DistilledSkill]:
        """Return all distilled skills for a given domain.

        Args:
            domain: The domain to filter by.

        Returns:
            List of :class:`DistilledSkill` sorted by confidence descending.
        """
        return sorted(
            [s for s in self._skills if s.domain == domain],
            key=lambda s: s.confidence,
            reverse=True,
        )

    def best_skill(self, goal_description: str) -> Optional[DistilledSkill]:
        """Find the skill most relevant to a goal using token Jaccard overlap.

        Args:
            goal_description: Natural-language description of the goal.

        Returns:
            :class:`DistilledSkill` with highest overlap, or ``None`` if no
            skills have been distilled.
        """
        if not self._skills:
            return None

        goal_tokens = frozenset(
            w.strip(".,!?;:\"'()").lower()
            for w in goal_description.split()
            if len(w.strip(".,!?;:\"'()")) >= 3
            and w.strip(".,!?;:\"'()").lower() not in _STOP_WORDS
        )
        if not goal_tokens:
            return self._skills[0]

        best: Optional[DistilledSkill] = None
        best_score = -1.0
        for skill in self._skills:
            skill_tokens = frozenset(
                skill.preconditions + skill.outcomes + [skill.name]
            )
            if not skill_tokens:
                continue
            intersection = len(goal_tokens & skill_tokens)
            union = len(goal_tokens | skill_tokens)
            score = intersection / union if union else 0.0
            if score > best_score:
                best_score = score
                best = skill
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_action_patterns(
        self, items: list[Any]
    ) -> list[tuple[str, list[str], list[str], str, float, str]]:
        """Extract (action, preconditions, outcomes, memory_id, strength, domain)."""
        patterns = []
        for item in items:
            text = getattr(item.experience, "content", "") or ""
            dom = getattr(item.experience, "domain", None) or "general"
            strength = min(1.0, max(0.0, getattr(item, "memory_strength", 0.5)))
            words = text.lower().split()
            for i, word in enumerate(words):
                tok = word.strip(".,!?;:\"'()")
                if tok not in _ACTION_TOKENS:
                    continue
                # Preconditions: up to 3 meaningful tokens before action
                pre: list[str] = []
                for j in range(i - 1, max(i - 5, -1), -1):
                    candidate = words[j].strip(".,!?;:\"'()")
                    if (
                        len(candidate) >= 3
                        and candidate not in _STOP_WORDS
                        and candidate not in _ACTION_TOKENS
                    ):
                        pre.append(candidate)
                    if len(pre) >= 3:
                        break
                pre.reverse()
                # Outcomes: up to 3 meaningful tokens after action
                out: list[str] = []
                for j in range(i + 1, min(i + 6, len(words))):
                    candidate = words[j].strip(".,!?;:\"'()")
                    if (
                        len(candidate) >= 3
                        and candidate not in _STOP_WORDS
                        and candidate not in _ACTION_TOKENS
                    ):
                        out.append(candidate)
                    if len(out) >= 3:
                        break
                patterns.append((tok, pre, out, item.id, strength, dom))
        return patterns

    def _store_skill(self, skill: DistilledSkill) -> None:
        """Persist a skill as a new memory."""
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=skill.description,
                domain="skill",
                importance=self.skill_importance,
                emotional_valence=0.0,
            )
            self.memory.store(exp)
        except Exception:
            pass

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
