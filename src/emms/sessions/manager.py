"""SessionManager — persistent session log for EMMS.

Tracks the active session, auto-assigns session_id to every Experience stored
through it, and persists SessionSummary objects to a JSONL log file.

Inspired by claude-mem's session table schema (request / investigated /
learned / completed / next_steps), but integrated directly with the EMMS
memory pipeline rather than a separate SQLite database.

Usage::

    from emms.sessions import SessionManager

    sm = SessionManager(log_path="~/.emms/sessions.jsonl")
    sm.start_session()                  # auto-generates session_id

    exp = Experience(content="Fixed the persistence bug", domain="tech")
    sm.store(exp)                       # session_id auto-injected

    sm.update(
        learned="_items_by_exp_id must be rebuilt on load",
        completed="Fixed 3 persistence bugs — all 333 tests pass",
    )

    sm.end_session()                    # flushes summary to JSONL
    past = sm.load_sessions()           # list all past SessionSummary objects
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from emms.core.models import Experience, SessionSummary

if TYPE_CHECKING:
    from emms.memory.hierarchical import HierarchicalMemory

logger = logging.getLogger(__name__)

_DEFAULT_LOG = Path.home() / ".emms" / "sessions.jsonl"


class SessionManager:
    """Tracks sessions, auto-injects session_id, persists SessionSummary logs.

    Can be used standalone (just session tracking) or as a thin wrapper
    around HierarchicalMemory to auto-inject session_id on every store().
    """

    def __init__(
        self,
        memory: "HierarchicalMemory | None" = None,
        log_path: Path | str = _DEFAULT_LOG,
        consolidate_every: int = 20,
    ):
        self.memory = memory
        self.log_path = Path(log_path).expanduser()
        self._active: SessionSummary | None = None
        # Debounced consolidation: run consolidation after every N stores
        self._consolidate_every = consolidate_every
        self._stores_since_consolidation: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        session_id: str | None = None,
        request: str = "",
    ) -> str:
        """Open a new session. Returns the session_id.

        Args:
            session_id: Explicit ID; auto-generated (uuid4 prefix) if None.
            request: What the user asked for this session.
        """
        if self._active is not None:
            logger.warning(
                "Starting new session while '%s' is still open — auto-closing.",
                self._active.session_id,
            )
            self.end_session()

        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self._active = SessionSummary(session_id=sid, request=request)
        logger.info("Session started: %s", sid)
        return sid

    def end_session(self) -> SessionSummary | None:
        """Close the active session, persist summary to JSONL, return it."""
        if self._active is None:
            return None
        self._active.close()
        self._flush(self._active)
        finished = self._active
        self._active = None
        logger.info(
            "Session ended: %s  (%d memories, %.1fs)",
            finished.session_id,
            finished.memory_count,
            finished.duration_seconds or 0,
        )
        return finished

    def update(
        self,
        request: str | None = None,
        investigated: str | None = None,
        learned: str | None = None,
        completed: str | None = None,
        next_steps: str | None = None,
    ) -> None:
        """Update the active session's narrative fields."""
        if self._active is None:
            raise RuntimeError("No active session. Call start_session() first.")
        if request is not None:
            self._active.request = request
        if investigated is not None:
            self._active.investigated = investigated
        if learned is not None:
            self._active.learned = learned
        if completed is not None:
            self._active.completed = completed
        if next_steps is not None:
            self._active.next_steps = next_steps

    # ------------------------------------------------------------------
    # Experience storage (auto-injects session_id)
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> Experience:
        """Inject the active session_id into *experience* and store it.

        If a HierarchicalMemory was provided at construction, the experience
        is also passed through to memory.store(). Either way the mutated
        experience is returned.
        """
        if self._active is None:
            raise RuntimeError("No active session. Call start_session() first.")

        # Inject session_id
        experience.session_id = self._active.session_id

        # Update session stats
        self._active.memory_count += 1
        if experience.obs_type is not None:
            key = experience.obs_type.value
            self._active.obs_types[key] = self._active.obs_types.get(key, 0) + 1

        # Delegate to memory if wired up
        if self.memory is not None:
            self.memory.store(experience)

        # Debounced consolidation: run after every N stores
        self._stores_since_consolidation += 1
        if (
            self.memory is not None
            and self._consolidate_every > 0
            and self._stores_since_consolidation >= self._consolidate_every
        ):
            self.memory.consolidate()
            self._stores_since_consolidation = 0
            logger.debug("Debounced consolidation triggered after %d stores.", self._consolidate_every)

        return experience

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _flush(self, summary: SessionSummary) -> None:
        """Append *summary* as one JSON line to the log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(summary.model_dump_json() + "\n")

    def load_sessions(self) -> list[SessionSummary]:
        """Load all persisted SessionSummary objects from the JSONL log."""
        if not self.log_path.exists():
            return []
        summaries: list[SessionSummary] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    summaries.append(SessionSummary.model_validate_json(line))
        return summaries

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Look up a specific session by ID from the persisted log."""
        for s in self.load_sessions():
            if s.session_id == session_id:
                return s
        return None

    def export_session_log(self, path: Path | str) -> int:
        """Copy the full session log to *path*. Returns line count."""
        src = self.log_path
        if not src.exists():
            return 0
        dest = Path(path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
        dest.write_text("".join(lines), encoding="utf-8")
        return len(lines)

    def generate_claude_md(
        self,
        output_path: Path | str = "CLAUDE.md",
        max_sessions: int = 20,
    ) -> Path:
        """Auto-generate a CLAUDE.md file summarising recent session history.

        Mirrors claude-mem's auto-generated per-folder CLAUDE.md files.
        Claude Code automatically reads CLAUDE.md at session start, so this
        bridges EMMS session memory into Claude's native context injection.

        The generated file includes:
        - Recent session timeline (chronological, newest first)
        - Per-session: request, learned, completed, next_steps, obs_type counts
        - Active session status if a session is open

        Args:
            output_path: Destination path (default: ./CLAUDE.md).
            max_sessions: Cap on sessions to include (most-recent N).

        Returns:
            Path to the written file.
        """
        import datetime

        dest = Path(output_path).expanduser()
        sessions = self.load_sessions()[-max_sessions:]
        sessions_rev = list(reversed(sessions))  # newest first

        lines: list[str] = [
            "# EMMS Session Memory",
            "",
            "> Auto-generated by EMMS SessionManager. Do not edit manually.",
            f"> Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Active session banner
        if self._active:
            lines += [
                "## 🟢 Active Session",
                f"**ID:** `{self._active.session_id}`",
                f"**Request:** {self._active.request or '(not set)'}",
                f"**Memories stored:** {self._active.memory_count}",
                "",
            ]

        lines += ["## Session Timeline", ""]

        if not sessions_rev:
            lines.append("*No completed sessions yet.*")
        else:
            for s in sessions_rev:
                ended = ""
                if s.ended_at:
                    ended = datetime.datetime.fromtimestamp(s.ended_at).strftime("%Y-%m-%d %H:%M")
                dur = f" ({s.duration_seconds:.0f}s)" if s.duration_seconds else ""
                obs_str = ", ".join(f"{k}:{v}" for k, v in s.obs_types.items()) if s.obs_types else "—"

                lines += [
                    f"### `{s.session_id}` — {ended}{dur}",
                    f"- **Request:** {s.request or '—'}",
                    f"- **Investigated:** {s.investigated or '—'}",
                    f"- **Learned:** {s.learned or '—'}",
                    f"- **Completed:** {s.completed or '—'}",
                    f"- **Next steps:** {s.next_steps or '—'}",
                    f"- **Memories:** {s.memory_count} ({obs_str})",
                    "",
                ]

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
        logger.info("CLAUDE.md written to %s (%d sessions)", dest, len(sessions_rev))
        return dest

    def generate_context_injection(
        self,
        max_observations: int = 50,
        all_items: "list | None" = None,
    ) -> str:
        """Generate a compact observation index for session-start context injection.

        Mirrors claude-mem's SessionStart hook output format::

            [mem_id] YYYY-MM-DD HH:MM [obs_type] - title (N tokens)

        This string can be prepended to a system prompt or injected via
        ``hookSpecificOutput.additionalContext`` at session start, giving the
        agent an instant overview of recent memory without loading full content.

        Args:
            max_observations: Maximum number of recent observations to include.
            all_items: MemoryItems to build the index from. Falls back to the
                       HierarchicalMemory wired at construction when None.

        Returns:
            Formatted compact index string ready for context injection.
        """
        import datetime

        # Gather items
        if all_items is None and self.memory is not None:
            all_items = [
                item for _, store in self.memory._iter_tiers() for item in store
            ]
        if not all_items:
            return "# EMMS Memory Index\n*(no memories stored yet)*\n"

        # Sort newest first, take the requested window
        sorted_items = sorted(
            all_items, key=lambda x: x.experience.timestamp, reverse=True
        )
        recent = sorted_items[:max_observations]

        lines: list[str] = [
            "# EMMS Memory Index",
            f"# {len(sorted_items)} total observations — showing {len(recent)} most recent",
            "",
        ]
        for item in recent:
            exp = item.experience
            dt = datetime.datetime.fromtimestamp(exp.timestamp).strftime("%Y-%m-%d %H:%M")
            obs_label = f"[{exp.obs_type.value}]" if exp.obs_type else "[?]"
            title = exp.title or exp.content[:60].rstrip()
            if not exp.title and len(exp.content) > 60:
                title += "…"
            tok = int(len(exp.content.split()) * 1.3)
            lines.append(f"[{item.id}] {dt} {obs_label} - {title} ({tok} tokens)")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_session_id(self) -> str | None:
        return self._active.session_id if self._active else None

    @property
    def active_summary(self) -> SessionSummary | None:
        return self._active

    def __repr__(self) -> str:
        active = self._active.session_id if self._active else "none"
        return f"SessionManager(active={active!r}, log={self.log_path})"
