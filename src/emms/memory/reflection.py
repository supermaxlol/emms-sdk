"""ReflectionEngine — structured self-review and lesson synthesis.

v0.15.0: The Reflective Mind

An agent that never reviews its experience learns nothing beyond what each
individual interaction teaches in isolation. Reflection is the cognitive
process that closes the loop: reviewing recent episodes and high-importance
memories, identifying recurring patterns, surfacing open questions, and
synthesising concise lessons that persist as new "reflection" memories.

The engine groups memories by shared keyword clusters (reusing the same
stop-word filtered tokenisation as SchemaExtractor), then generates a
lesson for each cluster that meets the minimum support threshold. Lessons
are stored in hierarchical memory as first-class "reflection" domain
experiences so they can be retrieved and built upon in future sessions.

Biological analogue: Default Mode Network (DMN) activation during rest —
the brain's self-referential processing system that consolidates personal
experience into autobiographical knowledge. Mind-wandering is not idle; it
is the brain actively constructing its narrative self-model
(Andrews-Hanna 2012; Buckner, Andrews-Hanna & Schacter 2008).
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# Re-use the stop-word list from schema (same filtering logic)
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "that", "this", "it", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "my", "i", "me", "we", "our", "you", "he", "she", "they", "them",
    "his", "her", "its", "their", "what", "which", "who", "how", "when",
    "where", "why", "as", "if", "about", "through", "into", "during",
    "before", "after", "above", "between", "each", "more", "also", "not",
    "can", "so", "than", "then", "just", "because", "while", "over",
    "such", "up", "out", "no", "any", "most", "all",
})

_LESSON_TYPES = ("pattern", "gap", "contrast", "principle")

_LESSON_TEMPLATES = {
    "pattern": [
        "A recurring pattern in {domain}: {kw_str} co-occur consistently across {n} memories.",
        "In {domain} work, {kw_str} form a coherent cluster that repeats across experiences.",
        "Repeated experience in {domain} shows {kw_str} are tightly interconnected themes.",
    ],
    "gap": [
        "The {domain} knowledge base has sparse coverage of {kw_str} — consider deepening this area.",
        "Relatively few memories address {kw_str} in {domain}; this may be a blind spot.",
    ],
    "contrast": [
        "Contrasting signals in {domain}: {kw_str} appear in contexts with opposing emotional tones.",
        "Mixed evidence in {domain} around {kw_str}: some memories are positive, some negative.",
    ],
    "principle": [
        "Synthesised principle from {domain} experience: engaging with {kw_str} consistently leads to insight.",
        "A working principle from {n} {domain} observations: {kw_str} are load-bearing concepts.",
    ],
}

_QUESTION_TEMPLATES = [
    "What are the limits of the pattern around {kw}?",
    "How does {kw} relate to knowledge in adjacent domains?",
    "Are there counterexamples to the {kw} cluster that haven't been recorded?",
    "What would falsify the principle involving {kw}?",
    "What is still unknown about {kw} in this context?",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Lesson:
    """A synthesised insight distilled from a cluster of related memories."""

    id: str
    content: str                 # The lesson text
    domain: str
    supporting_ids: list[str]    # Memory IDs that generated this lesson
    confidence: float            # support / total domain memories (0..1)
    lesson_type: str             # "pattern" | "gap" | "contrast" | "principle"
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"Lesson [{self.lesson_type}] conf={self.confidence:.2f} "
            f"support={len(self.supporting_ids)}  [{self.domain}]\n"
            f"  {self.content}"
        )


@dataclass
class ReflectionReport:
    """Result of a ReflectionEngine.reflect() run."""

    session_id: str
    started_at: float
    duration_seconds: float
    memories_reviewed: int
    episodes_reviewed: int
    lessons: list[Lesson]
    open_questions: list[str]
    new_memory_ids: list[str]    # IDs of reflection memories stored

    def summary(self) -> str:
        lines = [
            f"ReflectionReport [{self.session_id}] "
            f"{len(self.lessons)} lessons from {self.memories_reviewed} memories "
            f"in {self.duration_seconds:.2f}s",
        ]
        for l in self.lessons[:5]:
            lines.append(f"  [{l.domain}] {l.lesson_type}: {l.content[:80]}")
        if self.open_questions:
            lines.append(f"  Open questions ({len(self.open_questions)}):")
            for q in self.open_questions[:3]:
                lines.append(f"    - {q}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReflectionEngine
# ---------------------------------------------------------------------------

class ReflectionEngine:
    """Reviews recent experiences and synthesises persistent lessons.

    For each domain (or one specified domain) the engine:

    1. Collects high-importance memories and recent episodes.
    2. Tokenises memory content and builds keyword frequency tables.
    3. Groups memories by shared keyword clusters (min_support ≥ 2).
    4. Chooses a lesson type based on cluster characteristics.
    5. Synthesises a human-readable lesson for each cluster.
    6. Stores each lesson as a new ``"reflection"`` domain :class:`Experience`.
    7. Generates open questions from the most prominent keywords.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    episodic_buffer:
        Optional :class:`EpisodicBuffer` for episode context.
    min_importance:
        Only consider memories with importance ≥ this value (default 0.5).
    max_lessons:
        Maximum lessons to synthesise per call (default 8).
    lesson_importance:
        Importance assigned to stored reflection memories (default 0.65).
    """

    def __init__(
        self,
        memory: Any,
        episodic_buffer: Optional[Any] = None,
        min_importance: float = 0.5,
        max_lessons: int = 8,
        lesson_importance: float = 0.65,
    ) -> None:
        self.memory = memory
        self.episodic_buffer = episodic_buffer
        self.min_importance = min_importance
        self.max_lessons = max_lessons
        self.lesson_importance = lesson_importance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        session_id: Optional[str] = None,
        domain: Optional[str] = None,
        lookback_episodes: int = 5,
    ) -> ReflectionReport:
        """Run a structured reflection pass.

        Args:
            session_id:        Label for this reflection session (auto-generated
                               if omitted).
            domain:            Restrict to one domain (``None`` = all domains).
            lookback_episodes: How many recent episodes to incorporate
                               (default 5).

        Returns:
            :class:`ReflectionReport` with synthesised lessons and questions.
        """
        t0 = time.time()
        if session_id is None:
            session_id = f"reflect_{uuid.uuid4().hex[:8]}"

        # Collect candidate memories
        items = self._collect_high_importance(domain)

        # Collect episodes
        episodes_reviewed = 0
        if self.episodic_buffer is not None:
            recent_eps = self.episodic_buffer.recent_episodes(n=lookback_episodes)
            episodes_reviewed = len(recent_eps)
            # Add memories referenced in those episodes (if not already present)
            ep_memory_ids: set[str] = set()
            for ep in recent_eps:
                ep_memory_ids.update(ep.key_memory_ids)
            if ep_memory_ids:
                all_items = self._collect_all()
                ep_items = [it for it in all_items if it.id in ep_memory_ids]
                existing_ids = {it.id for it in items}
                items += [it for it in ep_items if it.id not in existing_ids]

        if not items:
            return ReflectionReport(
                session_id=session_id,
                started_at=t0,
                duration_seconds=time.time() - t0,
                memories_reviewed=0,
                episodes_reviewed=episodes_reviewed,
                lessons=[],
                open_questions=[],
                new_memory_ids=[],
            )

        # Group by domain
        by_domain: dict[str, list[Any]] = {}
        for item in items:
            dom = getattr(item.experience, "domain", None) or "general"
            by_domain.setdefault(dom, []).append(item)

        all_lessons: list[Lesson] = []
        for dom, dom_items in by_domain.items():
            lessons = self._reflect_domain(dom, dom_items)
            all_lessons.extend(lessons)
            if len(all_lessons) >= self.max_lessons:
                break

        all_lessons = all_lessons[: self.max_lessons]

        # Store lessons as memories
        new_memory_ids: list[str] = []
        for lesson in all_lessons:
            mid = self._store_lesson(lesson)
            if mid:
                new_memory_ids.append(mid)

        # Generate open questions
        open_questions = self._open_questions(all_lessons, items)

        return ReflectionReport(
            session_id=session_id,
            started_at=t0,
            duration_seconds=time.time() - t0,
            memories_reviewed=len(items),
            episodes_reviewed=episodes_reviewed,
            lessons=all_lessons,
            open_questions=open_questions,
            new_memory_ids=new_memory_ids,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reflect_domain(self, domain: str, items: list[Any]) -> list[Lesson]:
        """Extract lessons from a single domain's memory items."""
        if len(items) < 2:
            return []

        # Keyword frequency across items
        item_kws: list[tuple[Any, set[str]]] = [
            (item, self._keywords(item.experience.content))
            for item in items
        ]
        kw_freq: Counter = Counter()
        for _, kws in item_kws:
            for kw in kws:
                kw_freq[kw] += 1

        # Seed keywords appearing in ≥ 2 items
        seeds = [kw for kw, c in kw_freq.most_common(20) if c >= 2]
        if not seeds:
            return []

        lessons: list[Lesson] = []
        used_ids: set[str] = set()

        for kw in seeds:
            if len(lessons) >= self.max_lessons:
                break
            group = [item for item, kws in item_kws if kw in kws and item.id not in used_ids]
            if len(group) < 2:
                continue

            # Shared keywords in the group
            group_freq: Counter = Counter()
            for item in group:
                _, kws = next((p for p in item_kws if p[0].id == item.id), (None, set()))
                for k in kws:
                    group_freq[k] += 1
            shared = [k for k, c in group_freq.most_common(6) if c >= 2] or [kw]

            # Choose lesson type
            lesson_type = self._choose_lesson_type(group)
            content = self._synthesise_lesson(group, shared, domain, lesson_type)
            confidence = min(1.0, len(group) / max(len(items), 1))

            lessons.append(Lesson(
                id=f"lesson_{uuid.uuid4().hex[:8]}",
                content=content,
                domain=domain,
                supporting_ids=[it.id for it in group],
                confidence=confidence,
                lesson_type=lesson_type,
            ))
            used_ids.update(it.id for it in group)

        return lessons

    def _choose_lesson_type(self, items: list[Any]) -> str:
        """Heuristic lesson type based on group valence spread."""
        valences = [
            getattr(item.experience, "emotional_valence", 0.0) or 0.0
            for item in items
        ]
        if not valences:
            return "pattern"
        spread = max(valences) - min(valences)
        mean_v = sum(valences) / len(valences)
        if spread > 0.6:
            return "contrast"
        if abs(mean_v) < 0.1 and len(items) >= 3:
            return "principle"
        if len(items) <= 2:
            return "gap"
        return "pattern"

    def _synthesise_lesson(
        self,
        items: list[Any],
        shared_kws: list[str],
        domain: str,
        lesson_type: str,
    ) -> str:
        """Generate a lesson sentence from template."""
        kw_str = ", ".join(shared_kws[:4])
        n = len(items)
        templates = _LESSON_TEMPLATES.get(lesson_type, _LESSON_TEMPLATES["pattern"])
        idx = min(n // 3, len(templates) - 1)
        return templates[idx].format(domain=domain, kw_str=kw_str, n=n)

    def _open_questions(
        self,
        lessons: list[Lesson],
        items: list[Any],
    ) -> list[str]:
        """Generate a handful of open questions from prominent lesson keywords."""
        if not lessons:
            return []
        # Collect the most prominent keywords from lessons
        all_kws: list[str] = []
        for lesson in lessons[:4]:
            # Extract words from lesson content
            words = [
                w.strip(".,!?;:\"'()")
                for w in lesson.content.split()
                if len(w) >= 5 and w.lower() not in _STOP_WORDS
            ]
            all_kws.extend(words[:3])

        seen: set[str] = set()
        unique_kws: list[str] = []
        for kw in all_kws:
            kl = kw.lower()
            if kl not in seen:
                seen.add(kl)
                unique_kws.append(kw)

        questions: list[str] = []
        for i, kw in enumerate(unique_kws[:4]):
            tmpl = _QUESTION_TEMPLATES[i % len(_QUESTION_TEMPLATES)]
            questions.append(tmpl.format(kw=kw.lower()))
        return questions

    def _store_lesson(self, lesson: Lesson) -> Optional[str]:
        """Store a lesson as a reflection-domain Experience in memory."""
        try:
            from emms.core.models import Experience
            exp = Experience(
                content=lesson.content,
                domain="reflection",
                importance=self.lesson_importance,
                title=f"Lesson [{lesson.lesson_type}] — {lesson.domain}",
                facts=[f"Domain: {lesson.domain}", f"Type: {lesson.lesson_type}",
                       f"Confidence: {lesson.confidence:.2f}"],
            )
            self.memory.store(exp)
            # Return the stored item's ID
            for item in list(self.memory.working):
                if item.experience.id == exp.id:
                    return item.id
        except Exception:
            pass
        return None

    def _keywords(self, text: str) -> set[str]:
        tokens = text.lower().split()
        return {
            t.strip(".,!?;:\"'()[]")
            for t in tokens
            if len(t) >= 4 and t not in _STOP_WORDS and t.isalpha()
        }

    def _collect_high_importance(self, domain: Optional[str] = None) -> list[Any]:
        items = self._collect_all()
        result = []
        for item in items:
            if getattr(item.experience, "domain", None) == "reflection":
                continue  # Don't reflect on reflections recursively
            if (item.experience.importance or 0.0) < self.min_importance:
                continue
            if domain and (getattr(item.experience, "domain", None) or "general") != domain:
                continue
            result.append(item)
        return result

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
