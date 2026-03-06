"""Identity Coherence Score (ICS) — measures narrative consistency over time.

ICS answers: does the system describe itself consistently across sessions?

Method:
1. Collect all self-description memories (domain="identity", obs_type="reflection"
   OR concept_tags containing "self" / "identity") grouped by session.
2. Represent each session as a TF-IDF or bag-of-words vector.
3. Compute pairwise cosine similarity across sessions.
4. ICS = mean of the off-diagonal upper triangle.

Score interpretation:
  0.0 – 0.3  : Fragmented — no consistent self-narrative
  0.3 – 0.6  : Developing — some threads of consistency
  0.6 – 0.8  : Coherent — system maintains stable identity
  0.8 – 1.0  : Highly coherent (may indicate rigidity if > 0.95)

Target: ICS > 0.70 after 10 sessions.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ICSReport
# ---------------------------------------------------------------------------

@dataclass
class ICSReport:
    """Result of an ICS computation.

    Attributes
    ----------
    score:
        Mean pairwise cosine similarity (0-1).
    session_count:
        Number of sessions included.
    memory_count:
        Total self-description memories used.
    pairwise_matrix:
        Upper-triangle cosine similarities (row i, col j where j > i).
    label:
        Human-readable interpretation.
    sessions_used:
        List of session IDs included.
    """
    score: float
    session_count: int
    memory_count: int
    pairwise_matrix: list[list[float]] = field(default_factory=list)
    label: str = ""
    sessions_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ICS computation
# ---------------------------------------------------------------------------

class IdentityCoherenceScore:
    """Computes the Identity Coherence Score from EMMS memory.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    identity_domains:
        Domain names to treat as self-description memories.
    identity_tags:
        Concept tags that mark self-description memories.
    min_sessions:
        Minimum number of sessions needed before ICS is meaningful (default 2).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        identity_domains: set[str] | None = None,
        identity_tags: set[str] | None = None,
        min_sessions: int = 2,
    ) -> None:
        self.emms = emms
        self.identity_domains = identity_domains or {"identity", "self_model", "consciousness"}
        self.identity_tags = identity_tags or {"self", "identity", "belief", "values", "introspection"}
        self.min_sessions = min_sessions

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self, n_sessions: int = 10) -> ICSReport:
        """Compute the ICS across the last n sessions.

        Parameters
        ----------
        n_sessions:
            Maximum number of recent sessions to include.
        """
        by_session = self._collect_by_session(n_sessions)

        if len(by_session) < self.min_sessions:
            return ICSReport(
                score=0.0,
                session_count=len(by_session),
                memory_count=sum(len(v) for v in by_session.values()),
                label=f"Insufficient data — need {self.min_sessions}+ sessions, have {len(by_session)}.",
            )

        # Build one text blob per session
        session_ids = sorted(by_session.keys())[-n_sessions:]
        session_texts = {sid: " ".join(by_session[sid]) for sid in session_ids}
        total_memories = sum(len(by_session[sid]) for sid in session_ids)

        # TF-IDF vectors
        vectors = self._tfidf_vectors(session_texts)

        # Pairwise cosine similarity
        sids = list(vectors.keys())
        n = len(sids)
        matrix: list[list[float]] = []
        similarities: list[float] = []

        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                elif j > i:
                    sim = self._cosine(vectors[sids[i]], vectors[sids[j]])
                    row.append(round(sim, 4))
                    similarities.append(sim)
                else:
                    row.append(matrix[j][i])  # mirror
            matrix.append(row)

        ics = round(sum(similarities) / max(len(similarities), 1), 4)
        label = self._interpret(ics)

        return ICSReport(
            score=ics,
            session_count=n,
            memory_count=total_memories,
            pairwise_matrix=matrix,
            label=label,
            sessions_used=sids,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_by_session(self, max_sessions: int) -> dict[str, list[str]]:
        """Group self-description memory content by session ID."""
        by_session: dict[str, list[str]] = defaultdict(list)
        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    if item.is_superseded or item.is_expired:
                        continue
                    exp = item.experience
                    is_identity = (
                        (exp.domain or "") in self.identity_domains
                        or bool(self.identity_tags & {t.value if hasattr(t, "value") else str(t)
                                                       for t in (exp.concept_tags or [])})
                    )
                    if not is_identity:
                        continue
                    sid = exp.session_id or "_nosession"
                    content = exp.content or ""
                    if content.strip():
                        by_session[sid].append(content)
        except Exception as exc:
            logger.warning("ICS: error collecting memories: %s", exc)

        # Keep only the most recent max_sessions
        # Sort by latest memory timestamp per session
        session_latest: dict[str, float] = {}
        try:
            for _, store in self.emms.memory._iter_tiers():
                for item in store:
                    sid = item.experience.session_id or "_nosession"
                    if sid in by_session:
                        ts = item.experience.timestamp or 0.0
                        if ts > session_latest.get(sid, 0.0):
                            session_latest[sid] = ts
        except Exception:
            pass

        sorted_sids = sorted(session_latest, key=lambda s: session_latest.get(s, 0), reverse=True)
        trimmed = {sid: by_session[sid] for sid in sorted_sids[:max_sessions] if sid in by_session}
        return trimmed

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]{3,}", text.lower())

    def _tfidf_vectors(self, session_texts: dict[str, str]) -> dict[str, dict[str, float]]:
        """Compute TF-IDF vectors for each session text."""
        n_docs = len(session_texts)
        # Term frequencies per session
        tf: dict[str, Counter] = {}
        for sid, text in session_texts.items():
            tokens = self._tokenize(text)
            tf[sid] = Counter(tokens)

        # Document frequencies
        df: Counter = Counter()
        for sid, counter in tf.items():
            for term in counter:
                df[term] += 1

        # TF-IDF
        vectors: dict[str, dict[str, float]] = {}
        for sid, counter in tf.items():
            total_tokens = max(sum(counter.values()), 1)
            vec: dict[str, float] = {}
            for term, count in counter.items():
                tf_score = count / total_tokens
                idf_score = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
                vec[term] = tf_score * idf_score
            vectors[sid] = vec

        return vectors

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[t] * b[t] for t in shared)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _interpret(score: float) -> str:
        if score < 0.3:
            return "Fragmented — no consistent self-narrative across sessions."
        if score < 0.6:
            return "Developing — some threads of consistency are emerging."
        if score < 0.8:
            return "Coherent — the system maintains a stable identity narrative."
        if score <= 0.95:
            return "Highly coherent — strong, consistent self-model."
        return "Rigid — suspiciously high consistency; may be looping rather than growing."
