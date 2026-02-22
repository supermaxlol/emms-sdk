"""DilemmaEngine — detecting ethical tensions between competing moral imperatives.

v0.23.0: The Moral Mind

Ethical dilemmas arise when two morally significant commitments conflict — when
doing right by one value or framework requires violating another. The canonical
cases (trolley problems, promises that harm, loyalty vs honesty) all share this
structure: two memories, experiences, or obligations that each carry genuine
moral weight but pull in opposite directions. Detecting dilemmas is therefore
the capacity to identify pairs of high-moral-weight experiences that exhibit
opposing valences within the same domain — to surface the agent's genuine
moral conflicts rather than merely cataloguing its values.

DilemmaEngine operationalises dilemma detection by pairing memories that have
been assessed as morally significant (by MoralReasoner), that share a domain,
and that have conflicting valences. The tension score captures both the moral
stakes (product of moral weights) and the degree of opposition (absolute
valence difference). Resolution strategies are generated from a template table
keyed by the dominant frameworks of the two conflicting memories, providing
practical guidance for navigating each specific dilemma type.

Biological analogue: trolley-problem studies revealing competing moral
intuitions (Greene et al. 2001); ACC as the neural site of conflict monitoring
where incompatible representations compete (Botvinick et al. 2001); TPJ
integrating perspectives to resolve competing social commitments; dual-process
tension between intuitive deontological responses and deliberative consequentialist
reasoning (Cushman & Young 2011); vmPFC as the integrator of competing value
signals under moral conflict (Bechara et al. 2000).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Resolution strategy templates
# ---------------------------------------------------------------------------

_RESOLUTION_TEMPLATES: dict[frozenset[str], str] = {
    frozenset({"consequentialist", "deontological"}): (
        "Weigh outcomes against duties — seek the action that respects core "
        "obligations while producing the greatest net benefit."
    ),
    frozenset({"consequentialist", "virtue"}): (
        "Ask what a person of excellent character would choose when outcomes "
        "conflict with virtue."
    ),
    frozenset({"deontological", "virtue"}): (
        "Find the duty that best expresses the virtues — the action that is "
        "both principled and character-affirming."
    ),
}

_SAME_FRAMEWORK_TEMPLATE = (
    "Both invoke {framework} reasoning — resolve by identifying which "
    "carries the greater moral weight."
)

_DEFAULT_TEMPLATE = (
    "Examine the stakes carefully: which action preserves the most important "
    "moral commitments while causing the least harm?"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EthicalDilemma:
    """An ethical tension between two conflicting moral imperatives."""

    id: str                        # prefixed "dil_"
    description: str
    memory_id_a: str
    memory_id_b: str
    domain: str
    tension_score: float           # 0..1
    framework_a: str
    framework_b: str
    resolution_strategies: list[str]
    created_at: float

    def summary(self) -> str:
        n = len(self.resolution_strategies)
        return (
            f"EthicalDilemma [{self.domain}]  tension={self.tension_score:.3f}  "
            f"{self.framework_a} vs {self.framework_b}  "
            f"({n} strategies)\n"
            f"  {self.id[:12]}: {self.description[:70]}"
        )


@dataclass
class DilemmaReport:
    """Result of a DilemmaEngine.detect_dilemmas() call."""

    total_dilemmas: int
    dilemmas: list[EthicalDilemma]  # sorted by tension_score desc
    mean_tension: float
    domains_affected: list[str]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            f"DilemmaReport: {self.total_dilemmas} dilemmas  "
            f"mean_tension={self.mean_tension:.3f}  "
            f"domains={self.domains_affected}  "
            f"in {self.duration_seconds:.2f}s",
        ]
        for d in self.dilemmas[:5]:
            lines.append(f"  {d.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DilemmaEngine
# ---------------------------------------------------------------------------


class DilemmaEngine:
    """Detects ethical tensions between competing moral imperatives in memory.

    Parameters
    ----------
    memory:
        The :class:`HierarchicalMemory` instance.
    moral_reasoner:
        A :class:`MoralReasoner` instance that has already called ``.reason()``.
    min_tension:
        Minimum tension score to include a dilemma (default 0.05).
    max_dilemmas:
        Maximum number of :class:`EthicalDilemma` objects to retain (default 10).
    """

    def __init__(
        self,
        memory: Any,
        moral_reasoner: Any,
        min_tension: float = 0.05,
        max_dilemmas: int = 10,
    ) -> None:
        self.memory = memory
        self.moral_reasoner = moral_reasoner
        self.min_tension = min_tension
        self.max_dilemmas = max_dilemmas
        self._dilemmas: list[EthicalDilemma] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_dilemmas(self, domain: Optional[str] = None) -> DilemmaReport:
        """Detect ethical tensions between conflicting moral memories.

        Args:
            domain: Restrict to memories in this domain (``None`` = all).

        Returns:
            :class:`DilemmaReport` with dilemmas sorted by tension_score desc.
        """
        t0 = time.time()
        self._dilemmas = []

        # Ensure moral assessments are available
        assessments = dict(self.moral_reasoner._assessments)
        if not assessments:
            self.moral_reasoner.reason(domain=domain)
            assessments = dict(self.moral_reasoner._assessments)

        # Filter by domain if requested
        if domain:
            assessments = {
                mid: a for mid, a in assessments.items()
                if a.domain == domain
            }

        # Collect items with their valences
        all_items = self._collect_all()
        valence_map: dict[str, float] = {}
        for item in all_items:
            valence_map[item.id] = getattr(item.experience, "emotional_valence", 0.0) or 0.0

        # Group assessments by domain
        domain_assessments: dict[str, list[Any]] = {}
        for mid, a in assessments.items():
            domain_assessments.setdefault(a.domain, []).append(a)

        dilemmas: list[EthicalDilemma] = []
        seen_pairs: set[frozenset[str]] = set()

        for dom, dom_list in domain_assessments.items():
            # Only pair items with high moral weight
            significant = [a for a in dom_list if a.moral_weight >= self.min_tension]

            for i, a_i in enumerate(significant):
                for a_j in significant[i + 1:]:
                    pair_key = frozenset([a_i.memory_id, a_j.memory_id])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    val_a = valence_map.get(a_i.memory_id, 0.0)
                    val_b = valence_map.get(a_j.memory_id, 0.0)
                    valence_diff = abs(val_a - val_b)

                    if valence_diff <= 0.5:
                        continue

                    tension = min(
                        1.0,
                        a_i.moral_weight * a_j.moral_weight * valence_diff
                    )
                    tension = round(tension, 4)
                    if tension < self.min_tension:
                        continue

                    # Resolution strategies
                    fw_a = a_i.dominant_framework
                    fw_b = a_j.dominant_framework
                    strategies = self._resolve(fw_a, fw_b)

                    description = (
                        f"Memory '{a_i.memory_id[:8]}' ({fw_a}) conflicts with "
                        f"'{a_j.memory_id[:8]}' ({fw_b}) in domain '{dom}': "
                        f"opposing valences ({val_a:+.2f} vs {val_b:+.2f})"
                    )

                    dilemmas.append(EthicalDilemma(
                        id="dil_" + uuid.uuid4().hex[:8],
                        description=description,
                        memory_id_a=a_i.memory_id,
                        memory_id_b=a_j.memory_id,
                        domain=dom,
                        tension_score=tension,
                        framework_a=fw_a,
                        framework_b=fw_b,
                        resolution_strategies=strategies,
                        created_at=time.time(),
                    ))

        dilemmas.sort(key=lambda d: d.tension_score, reverse=True)
        self._dilemmas = dilemmas[: self.max_dilemmas]

        domains_affected = list(dict.fromkeys(d.domain for d in self._dilemmas))
        mean_t = (
            sum(d.tension_score for d in self._dilemmas) / len(self._dilemmas)
            if self._dilemmas else 0.0
        )

        return DilemmaReport(
            total_dilemmas=len(self._dilemmas),
            dilemmas=self._dilemmas,
            mean_tension=round(mean_t, 4),
            domains_affected=domains_affected,
            duration_seconds=time.time() - t0,
        )

    def dilemmas_for_domain(self, domain: str) -> list[EthicalDilemma]:
        """Return all detected dilemmas in a specific domain.

        Args:
            domain: Domain name to filter by.

        Returns:
            List of :class:`EthicalDilemma` sorted by tension_score descending.
        """
        return [d for d in self._dilemmas if d.domain == domain]

    def most_tense_dilemma(self) -> Optional[EthicalDilemma]:
        """Return the dilemma with the highest tension score, or ``None``."""
        return self._dilemmas[0] if self._dilemmas else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, fw_a: str, fw_b: str) -> list[str]:
        """Generate resolution strategies for a pair of frameworks."""
        strategies: list[str] = []

        key = frozenset({fw_a, fw_b})
        template = _RESOLUTION_TEMPLATES.get(key)
        if template:
            strategies.append(template)
        elif fw_a == fw_b and fw_a != "none":
            strategies.append(
                _SAME_FRAMEWORK_TEMPLATE.format(framework=fw_a)
            )
        else:
            strategies.append(_DEFAULT_TEMPLATE)

        # Always add a reflective prompt
        strategies.append(
            "Reflect on which choice you could justify to those affected "
            "by both imperatives."
        )
        return strategies

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items
