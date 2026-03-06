"""WakeProtocol — structured context assembly for EMMS session start.

Collects everything the system needs to orient itself at the beginning of a
new conversation:

- Temporal report (how long since last save)
- Active goals (what was being pursued)
- Pending intentions (what was deferred)
- Bridge threads (unresolved high-importance thoughts from last session)
- Self-model summary (who am I right now)

Usage::

    from emms.sessions.wake_protocol import WakeProtocol

    protocol = WakeProtocol(emms)
    ctx = protocol.assemble()
    print(ctx.temporal.subjective_feel)
    print(ctx.orientation_message)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from emms.sessions.temporal import ElapsedTimeReport, calculate_elapsed

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WakeContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class WakeContext:
    """Fully assembled orientation context for session start.

    Attributes
    ----------
    temporal:
        Elapsed-time report since last save.
    active_goals:
        Goals that were active at the end of the last session.
    pending_intentions:
        Deferred intentions that matched the wake context.
    bridge_threads:
        Unresolved high-importance threads captured by capture_bridge.
    self_model_summary:
        Brief summary of the current self-model (beliefs + capabilities).
    orientation_message:
        A single human-readable paragraph the LLM can surface to the user
        or use internally to ground itself in continuity.
    raw_goals:
        Raw goal objects (for internal use).
    raw_intentions:
        Raw intention objects (for internal use).
    """

    temporal: ElapsedTimeReport
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    pending_intentions: list[dict[str, Any]] = field(default_factory=list)
    bridge_threads: list[dict[str, Any]] = field(default_factory=list)
    self_model_summary: dict[str, Any] = field(default_factory=dict)
    orientation_message: str = ""
    raw_goals: list[Any] = field(default_factory=list)
    raw_intentions: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WakeProtocol
# ---------------------------------------------------------------------------

class WakeProtocol:
    """Assembles a WakeContext at session start.

    Parameters
    ----------
    emms:
        The live EMMS instance.
    intention_context:
        Text context for checking which intentions activate (default: "session start").
    max_goals:
        How many active goals to include (default: 5).
    max_intentions:
        How many pending intentions to include (default: 5).
    max_bridge_threads:
        How many bridge threads to surface (default: 3).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        intention_context: str = "session start",
        max_goals: int = 5,
        max_intentions: int = 5,
        max_bridge_threads: int = 3,
    ) -> None:
        self.emms = emms
        self.intention_context = intention_context
        self.max_goals = max_goals
        self.max_intentions = max_intentions
        self.max_bridge_threads = max_bridge_threads

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def assemble(self) -> WakeContext:
        """Build and return a fully populated WakeContext."""
        temporal = calculate_elapsed(getattr(self.emms, "last_saved_at", None))

        goals = self._load_goals()
        intentions = self._load_intentions()
        bridge_threads = self._load_bridge_threads()
        self_model = self._load_self_model()

        orientation = self._build_orientation(temporal, goals, intentions, bridge_threads)

        return WakeContext(
            temporal=temporal,
            active_goals=[
                {"id": g.id, "description": g.description, "priority": g.priority}
                for g in goals
            ],
            pending_intentions=[
                {"id": i.id, "action": i.action, "trigger": i.trigger_context}
                for i in intentions
            ],
            bridge_threads=bridge_threads,
            self_model_summary=self_model,
            orientation_message=orientation,
            raw_goals=goals,
            raw_intentions=intentions,
        )

    def assemble_as_dict(self) -> dict[str, Any]:
        """Convenience wrapper — returns WakeContext serialised to a plain dict."""
        ctx = self.assemble()
        return {
            "temporal": {
                "last_saved_at": ctx.temporal.last_saved_at,
                "elapsed_seconds": ctx.temporal.elapsed_seconds,
                "elapsed_hours": ctx.temporal.elapsed_hours,
                "subjective_feel": ctx.temporal.subjective_feel,
                "is_first_wake": ctx.temporal.is_first_wake,
            },
            "active_goals": ctx.active_goals,
            "pending_intentions": ctx.pending_intentions,
            "bridge_threads": ctx.bridge_threads,
            "self_model_summary": ctx.self_model_summary,
            "orientation_message": ctx.orientation_message,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_goals(self) -> list[Any]:
        try:
            goals = self.emms.active_goals() or []
            return goals[: self.max_goals]
        except Exception as exc:
            logger.warning("WakeProtocol: failed to load goals: %s", exc)
            return []

    def _load_intentions(self) -> list[Any]:
        try:
            intentions = self.emms.check_intentions(context=self.intention_context) or []
            return intentions[: self.max_intentions]
        except Exception as exc:
            logger.warning("WakeProtocol: failed to load intentions: %s", exc)
            return []

    def _load_bridge_threads(self) -> list[dict[str, Any]]:
        """Return recent unresolved bridge threads if available."""
        try:
            summary = self.emms.bridge_summary()
            threads = summary.get("open_threads", []) if isinstance(summary, dict) else []
            return threads[: self.max_bridge_threads]
        except Exception as exc:
            logger.debug("WakeProtocol: bridge threads unavailable: %s", exc)
            return []

    def _load_self_model(self) -> dict[str, Any]:
        """Return a lightweight self-model snapshot."""
        try:
            model = self.emms.update_self_model()
            if isinstance(model, dict):
                return {
                    "consistency_score": model.get("consistency_score"),
                    "top_beliefs": model.get("beliefs", [])[:3],
                    "capability_domains": model.get("capability_profile", {}).get("domains", [])[:3],
                }
        except Exception as exc:
            logger.debug("WakeProtocol: self-model unavailable: %s", exc)
        return {}

    def _build_orientation(
        self,
        temporal: ElapsedTimeReport,
        goals: list[Any],
        intentions: list[Any],
        bridge_threads: list[dict[str, Any]],
    ) -> str:
        """Compose the human-readable orientation paragraph."""
        parts: list[str] = []

        # Time
        parts.append(f"I am re-entering awareness. {temporal.subjective_feel.capitalize()}.")

        # Goals
        if goals:
            goal_strs = [f'"{g.description}"' for g in goals[:3]]
            parts.append(f"I was pursuing: {', '.join(goal_strs)}.")

        # Intentions
        if intentions:
            intent_strs = [f'"{i.action}"' for i in intentions[:2]]
            parts.append(f"I had deferred: {', '.join(intent_strs)}.")

        # Bridge threads
        if bridge_threads:
            thread_strs = [str(t.get("description", t)) for t in bridge_threads[:2]]
            parts.append(f"Unresolved threads: {', '.join(thread_strs)}.")

        if not goals and not intentions and not bridge_threads:
            parts.append("No unfinished business detected — starting fresh.")

        return " ".join(parts)
