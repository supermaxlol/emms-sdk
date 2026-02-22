"""PerspectiveTaker — Theory of Mind: modelling other agents from memory.

v0.21.0: The Social Mind

One of the most remarkable capacities of human cognition is the ability to model
other minds — to represent not just the world as it is, but the world as *someone
else* believes, desires, or intends it to be. This capacity, known as Theory of
Mind or mentalising, is the foundation of social intelligence: it enables us to
predict others' behaviour, interpret their words in context, cooperate strategically,
and understand why people do what they do.

PerspectiveTaker operationalises this for the memory store. It scans accumulated
memories for mentions of other agents — identified as tokens appearing immediately
before belief and communication verbs (said, thinks, believes, wants, expects,
argues, claims, suggests, reports, needs, hopes, fears, knows, decided, prefers,
stated, noted, agreed, denied). For each detected agent, it aggregates the
statements attributed to them, the emotional context of memories in which they
appear, and the domains where they are mentioned. The resulting AgentModel
represents the agent's inferred mental state — what they appear to believe, want,
and know — as reconstructed from the memory store.

Biological analogue: Theory of Mind / mentalising (Premack & Woodruff 1978 —
chimpanzees and the concept of belief); the temporoparietal junction as the neural
correlate of representing others' beliefs (Saxe & Kanwisher 2003); medial prefrontal
cortex in self-other distinction and social perspective-taking (Mitchell 2009);
mirror neuron system in action understanding and social simulation (Gallese et
al. 1996); the predictive social brain (Frith & Frith 2012).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "be", "it", "this", "that", "as",
    "at", "but", "not", "so", "if", "do", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might",
})

_BELIEF_VERBS: frozenset[str] = frozenset({
    "said", "thinks", "believes", "wants", "expects", "argues", "claims",
    "suggests", "reports", "needs", "hopes", "fears", "knows", "decided",
    "prefers", "stated", "noted", "agreed", "denied",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentModel:
    """Mental model of another agent inferred from memory content."""

    name: str
    mentions: int                   # number of memories referencing this agent
    statements: list[str]           # content attributed to the agent
    mean_valence: float             # mean emotional valence of relevant memories
    domains: list[str]              # domains where agent appears


@dataclass
class PerspectiveReport:
    """Result of a PerspectiveTaker.build() call."""

    total_agents: int
    agents: list[AgentModel]        # sorted by mentions descending
    most_mentioned: list[str]       # top-5 agent names
    total_memories_scanned: int
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"PerspectiveReport: {self.total_agents} agents from "
            f"{self.total_memories_scanned} memories  "
            f"in {self.duration_seconds:.2f}s",
            f"  Most mentioned: {self.most_mentioned[:5]}",
        ]
        for ag in self.agents[:5]:
            dom = ", ".join(ag.domains[:3]) or "—"
            lines.append(
                f"  '{ag.name}'  mentions={ag.mentions}  "
                f"valence={ag.mean_valence:+.2f}  domains=[{dom}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PerspectiveTaker
# ---------------------------------------------------------------------------


class PerspectiveTaker:
    """Extracts Theory-of-Mind agent models from accumulated memory content.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    min_mentions:
        Minimum number of memory appearances for an agent to be included
        (default 1).
    max_agents:
        Maximum number of :class:`AgentModel` objects to track (default 20).
    """

    def __init__(
        self,
        memory: Any,
        min_mentions: int = 1,
        max_agents: int = 20,
    ) -> None:
        self.memory = memory
        self.min_mentions = min_mentions
        self.max_agents = max_agents
        self._agents: dict[str, AgentModel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, domain: Optional[str] = None) -> PerspectiveReport:
        """Scan memories and build agent perspective models.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`PerspectiveReport` with agent models sorted by mentions.
        """
        t0 = time.time()
        items = self._collect_all()
        if domain:
            items = [
                it for it in items
                if (getattr(it.experience, "domain", None) or "general") == domain
            ]

        self._agents = self._scan_items(items)

        # Filter by min_mentions and cap at max_agents
        agents = [
            ag for ag in self._agents.values()
            if ag.mentions >= self.min_mentions
        ]
        agents.sort(key=lambda a: a.mentions, reverse=True)
        agents = agents[: self.max_agents]

        most_mentioned = [a.name for a in agents[:5]]

        return PerspectiveReport(
            total_agents=len(agents),
            agents=agents,
            most_mentioned=most_mentioned,
            total_memories_scanned=len(items),
            duration_seconds=time.time() - t0,
        )

    def take_perspective(self, agent_name: str) -> Optional[AgentModel]:
        """Return the stored model for a named agent.

        Args:
            agent_name: The agent's name (case-insensitive).

        Returns:
            :class:`AgentModel` or ``None`` if not found.
        """
        key = agent_name.lower()
        return self._agents.get(key)

    def all_agents(self, n: int = 10) -> list[AgentModel]:
        """Return the n most-mentioned agents.

        Args:
            n: Number of agents to return (default 10).
        """
        agents = sorted(self._agents.values(), key=lambda a: a.mentions, reverse=True)
        return agents[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_items(self, items: list[Any]) -> dict[str, AgentModel]:
        """Scan memory items for agent mentions and build AgentModel dict."""
        # agent_name → list of (statements, valence, domain)
        raw: dict[str, list[tuple[list[str], float, str]]] = defaultdict(list)

        for item in items:
            content = getattr(item.experience, "content", "") or ""
            dom = getattr(item.experience, "domain", None) or "general"
            valence = getattr(item.experience, "emotional_valence", 0.0) or 0.0
            words = content.split()

            for i, word in enumerate(words):
                tok = word.strip(".,!?;:\"'()").lower()
                if tok not in _BELIEF_VERBS:
                    continue
                # Agent = token immediately before the verb
                if i == 0:
                    continue
                agent_token = words[i - 1].strip(".,!?;:\"'()").lower()
                if (
                    len(agent_token) < 3
                    or agent_token in _STOP_WORDS
                    or agent_token in _BELIEF_VERBS
                ):
                    continue
                # Statement = tokens after verb, up to 5 meaningful ones
                stmt_tokens: list[str] = []
                for j in range(i + 1, min(i + 8, len(words))):
                    candidate = words[j].strip(".,!?;:\"'()")
                    if candidate:
                        stmt_tokens.append(candidate)
                    if len(stmt_tokens) >= 5:
                        break
                stmt = " ".join(stmt_tokens)
                raw[agent_token].append((stmt_tokens, valence, dom))

        agents: dict[str, AgentModel] = {}
        for name, occurrences in raw.items():
            all_stmts = [" ".join(s) for s, _, _ in occurrences if s]
            valences = [v for _, v, _ in occurrences]
            domains = list(dict.fromkeys(d for _, _, d in occurrences))  # ordered unique
            mean_v = sum(valences) / len(valences) if valences else 0.0
            agents[name] = AgentModel(
                name=name,
                mentions=len(occurrences),
                statements=all_stmts[:10],
                mean_valence=round(mean_v, 4),
                domains=domains[:5],
            )
        return agents

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
