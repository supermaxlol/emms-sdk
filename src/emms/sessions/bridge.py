"""SessionBridge — session-to-session context handoff.

Between sessions, the agent's "birth" into the next conversation would
otherwise be empty — new presence, blank arc, no memory of what was left
unresolved. SessionBridge captures the important threads at the end of a
session and carries them into the next one.

A BridgeThread is an unresolved, high-importance memory: something that
mattered enough to store but was never fully consolidated — meaning the
conversation around it may need to continue.

At session end:
    record = bridge.capture(session_id="sess_abc")

At session start (next session):
    injection = bridge.inject(record)  # returns a prompt-ready string
    # inject into the agent's opening context

The bridge also records the session's emotional state, dominant domains,
and presence score at end — giving the next session a richer starting
context.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BridgeThread:
    """A single unresolved memory thread to carry forward."""
    memory_id: str
    content_excerpt: str       # first 120 chars of content
    importance: float
    domain: str
    consolidation_score: float
    emotional_valence: float
    reason: str                # why this is being carried forward


@dataclass
class BridgeRecord:
    """Everything needed to prime the next session."""
    from_session_id: str
    captured_at: float
    to_session_id: str | None      # filled when the next session starts

    # Open threads
    open_threads: list[BridgeThread] = field(default_factory=list)

    # Session-end emotional state
    mean_valence_at_end: float = 0.0
    mean_intensity_at_end: float = 0.0
    presence_score_at_end: float = 1.0

    # Dominant activity
    dominant_domains: list[str] = field(default_factory=list)
    turn_count: int = 0

    # Narrative carry-forward
    closing_summary: str = ""    # free-text summary of what this session did

    def to_dict(self) -> dict:
        return {
            "from_session_id": self.from_session_id,
            "captured_at": self.captured_at,
            "to_session_id": self.to_session_id,
            "open_threads": [vars(t) for t in self.open_threads],
            "mean_valence_at_end": self.mean_valence_at_end,
            "mean_intensity_at_end": self.mean_intensity_at_end,
            "presence_score_at_end": self.presence_score_at_end,
            "dominant_domains": self.dominant_domains,
            "turn_count": self.turn_count,
            "closing_summary": self.closing_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BridgeRecord":
        threads = [BridgeThread(**t) for t in d.get("open_threads", [])]
        return cls(
            from_session_id=d["from_session_id"],
            captured_at=d.get("captured_at", time.time()),
            to_session_id=d.get("to_session_id"),
            open_threads=threads,
            mean_valence_at_end=d.get("mean_valence_at_end", 0.0),
            mean_intensity_at_end=d.get("mean_intensity_at_end", 0.0),
            presence_score_at_end=d.get("presence_score_at_end", 1.0),
            dominant_domains=d.get("dominant_domains", []),
            turn_count=d.get("turn_count", 0),
            closing_summary=d.get("closing_summary", ""),
        )

    def summary(self) -> str:
        lines = [
            f"Session bridge: {self.from_session_id} → {self.to_session_id or '(next)'}",
            f"  Captured: {len(self.open_threads)} open thread(s)",
            f"  Emotional state at end: valence={self.mean_valence_at_end:+.2f} "
            f"intensity={self.mean_intensity_at_end:.2f}",
            f"  Presence at end: {self.presence_score_at_end:.2f}",
        ]
        if self.dominant_domains:
            lines.append(f"  Dominant domains: {', '.join(self.dominant_domains[:3])}")
        if self.closing_summary:
            lines.append(f"  Summary: {self.closing_summary[:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SessionBridge
# ---------------------------------------------------------------------------

class SessionBridge:
    """Captures and injects session context across the session boundary.

    Parameters
    ----------
    memory : HierarchicalMemory
        The backing memory.
    presence_tracker : PresenceTracker | None
        If provided, reads final metrics for the bridge record.
    max_threads : int
        Maximum number of open threads to carry forward (default 5).
    importance_threshold : float
        Only threads with importance >= this are considered (default 0.5).
    consolidation_threshold : float
        Threads with consolidation_score < this are considered "unresolved"
        (default 0.3).
    """

    def __init__(
        self,
        memory: Any,
        presence_tracker: Any = None,
        max_threads: int = 5,
        importance_threshold: float = 0.5,
        consolidation_threshold: float = 0.3,
    ) -> None:
        self.memory = memory
        self.presence_tracker = presence_tracker
        self.max_threads = max_threads
        self.importance_threshold = importance_threshold
        self.consolidation_threshold = consolidation_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def capture(
        self,
        session_id: str | None = None,
        closing_summary: str = "",
    ) -> BridgeRecord:
        """Capture the session's state into a BridgeRecord.

        Parameters
        ----------
        session_id : str | None
            The ending session's ID (auto-generated if None).
        closing_summary : str
            Optional free-text summary of what the session accomplished.

        Returns
        -------
        BridgeRecord ready to be passed to inject() in the next session.
        """
        sid = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        # Collect all items
        all_items = self._collect_all()

        # Find unresolved threads
        candidates = [
            it for it in all_items
            if it.experience.importance >= self.importance_threshold
            and it.consolidation_score < self.consolidation_threshold
        ]
        # Sort by importance descending
        candidates.sort(key=lambda it: -it.experience.importance)

        threads: list[BridgeThread] = []
        for item in candidates[:self.max_threads]:
            exp = item.experience
            reason = self._reason_for_thread(item)
            threads.append(BridgeThread(
                memory_id=item.id,
                content_excerpt=exp.content[:120],
                importance=exp.importance,
                domain=exp.domain,
                consolidation_score=item.consolidation_score,
                emotional_valence=exp.emotional_valence,
                reason=reason,
            ))

        # Presence state
        pres_score = 1.0
        turn_count = 0
        mean_v = 0.0
        mean_i = 0.0
        dominant_domains: list[str] = []

        if self.presence_tracker is not None:
            try:
                metrics = self.presence_tracker.get_metrics()
                pres_score = metrics.presence_score
                turn_count = metrics.turn_count
                mean_v = metrics.mean_valence
                mean_i = metrics.mean_intensity
                dominant_domains = list(metrics.dominant_domains)
            except Exception:
                pass
        else:
            # Infer from all items
            if all_items:
                valences = [it.experience.emotional_valence for it in all_items]
                mean_v = sum(valences) / len(valences)

        return BridgeRecord(
            from_session_id=sid,
            captured_at=time.time(),
            to_session_id=None,
            open_threads=threads,
            mean_valence_at_end=mean_v,
            mean_intensity_at_end=mean_i,
            presence_score_at_end=pres_score,
            dominant_domains=dominant_domains,
            turn_count=turn_count,
            closing_summary=closing_summary,
        )

    def inject(
        self,
        record: BridgeRecord,
        new_session_id: str | None = None,
    ) -> str:
        """Generate a context string to inject at the start of the next session.

        Parameters
        ----------
        record : BridgeRecord
            The bridge record from the previous session.
        new_session_id : str | None
            ID for the new session (recorded on the bridge record).

        Returns
        -------
        str — prompt-ready context injection.
        """
        if new_session_id:
            record.to_session_id = new_session_id

        lines = [
            "## Session Continuity Context",
            f"_Carried forward from session {record.from_session_id}_",
            "",
        ]

        if record.closing_summary:
            lines += [
                "**Previous session:**",
                record.closing_summary,
                "",
            ]

        if record.dominant_domains:
            lines.append(
                f"**Focus areas:** {', '.join(record.dominant_domains[:3])}"
            )

        emotional_state = record.mean_valence_at_end
        if emotional_state > 0.3:
            mood = "positive"
        elif emotional_state < -0.3:
            mood = "challenging"
        else:
            mood = "neutral"
        lines.append(f"**Emotional state at close:** {mood} ({emotional_state:+.2f})")

        if record.presence_score_at_end < 0.4:
            lines.append(
                "_Note: Previous session ended with low presence "
                f"({record.presence_score_at_end:.2f}). "
                "This session begins fresh._"
            )

        if record.open_threads:
            lines += ["", "**Unresolved threads to continue:**"]
            for i, thread in enumerate(record.open_threads, 1):
                lines.append(
                    f"{i}. [{thread.domain}] {thread.content_excerpt}"
                    f"  _(importance: {thread.importance:.2f})_"
                )

        return "\n".join(lines)

    def save(self, path: str | Path, record: BridgeRecord) -> None:
        """Persist a BridgeRecord to a JSON file."""
        Path(path).write_text(json.dumps(record.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> BridgeRecord | None:
        """Load a BridgeRecord from a JSON file. Returns None if not found."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            return BridgeRecord.from_dict(json.loads(p.read_text()))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_all(self) -> list[Any]:
        items: list[Any] = []
        for tier in (self.memory.working, self.memory.short_term):
            items.extend(tier)
        for tier in (self.memory.long_term, self.memory.semantic):
            items.extend(tier.values())
        return items

    def _reason_for_thread(self, item: Any) -> str:
        """Generate a human-readable reason why this thread is unresolved."""
        reasons = []
        if item.experience.importance >= 0.8:
            reasons.append("high importance")
        if item.consolidation_score < 0.1:
            reasons.append("never consolidated")
        if item.access_count == 0:
            reasons.append("never recalled")
        if item.experience.emotional_valence < -0.5:
            reasons.append("negative emotional charge")
        return ", ".join(reasons) if reasons else "low consolidation score"
