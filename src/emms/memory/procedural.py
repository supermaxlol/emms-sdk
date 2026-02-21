"""ProceduralMemory — evolving system-prompt instructions (LangMem-inspired).

The fifth memory tier: stores reusable behavioral rules and procedures an
agent accumulates over time. Unlike episodic memory (what happened), procedural
memory captures *how to do things* — skills, policies, and heuristics that
remain applicable across sessions.

Inspired by LangMem's procedural memory which evolves the agent's system prompt
instructions based on observed patterns. Procedures can be added, patched (updated
in-place), or removed. The resulting text is formatted for direct injection into
an LLM system prompt.

Usage::

    from emms.memory.procedural import ProceduralMemory

    pm = ProceduralMemory()
    pm.add("Always cite sources when answering factual questions.", domain="research")
    pm.add("Prefer concise answers under 3 sentences.", domain="communication")

    # Get formatted rules for system prompt injection
    print(pm.get_prompt())
    # → ## Behavioral Rules
    # → - Always cite sources when answering factual questions.
    # → - Prefer concise answers under 3 sentences.

    # Patch an existing rule
    pm.patch("proc_abc123", "Always cite peer-reviewed sources.", importance=0.9)

    # Save/load for persistence
    pm.save_state("~/.emms/procedures.json")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcedureEntry(BaseModel):
    """A single behavioral rule or procedure."""

    id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    rule: str                          # The procedure text
    domain: str = "general"           # Which domain this rule applies to
    importance: float = 0.5           # 0 (low) … 1 (high)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    version: int = 1                   # Incremented on each patch
    active: bool = True                # Inactive rules are excluded from get_prompt()
    tags: list[str] = Field(default_factory=list)


class ProceduralMemory:
    """Fifth memory tier: accumulates and evolves behavioral rules.

    Operations:
    - ``add(rule)``        — append a new procedure
    - ``patch(id, rule)``  — update an existing procedure in-place (bumps version)
    - ``remove(id)``       — deactivate a procedure (soft delete)
    - ``get_prompt()``     — return rules formatted as a system prompt block
    - ``save_state()``     — persist to JSON
    - ``load_state()``     — restore from JSON
    """

    def __init__(self) -> None:
        self._procedures: dict[str, ProcedureEntry] = {}  # id → entry

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        rule: str,
        domain: str = "general",
        importance: float = 0.5,
        tags: list[str] | None = None,
        id: str | None = None,
    ) -> ProcedureEntry:
        """Append a new behavioral rule.

        Args:
            rule: The procedure text.
            domain: Domain this rule applies to (e.g. ``"research"``, ``"coding"``).
            importance: Priority weight; higher-importance rules sort to the top.
            tags: Optional classification tags.
            id: Explicit ID (auto-generated if None).

        Returns:
            The created ProcedureEntry.
        """
        entry = ProcedureEntry(
            rule=rule,
            domain=domain,
            importance=importance,
            tags=tags or [],
        )
        if id is not None:
            entry.id = id
        self._procedures[entry.id] = entry
        logger.debug("Procedure added: %s [domain=%s]", entry.id, domain)
        return entry

    def patch(
        self,
        id: str,
        rule: str,
        importance: float | None = None,
    ) -> ProcedureEntry | None:
        """Update an existing procedure in-place (LangMem patch semantics).

        Bumps the version counter and updates ``updated_at``.

        Args:
            id: ID of the procedure to update.
            rule: New rule text.
            importance: New importance weight (unchanged if None).

        Returns:
            The updated ProcedureEntry, or None if not found.
        """
        entry = self._procedures.get(id)
        if entry is None:
            logger.warning("patch() called on unknown procedure ID: %s", id)
            return None
        entry.rule = rule
        entry.updated_at = time.time()
        entry.version += 1
        if importance is not None:
            entry.importance = importance
        logger.debug("Procedure patched: %s (v%d)", id, entry.version)
        return entry

    def remove(self, id: str) -> bool:
        """Soft-delete a procedure (marks inactive, keeps history).

        Returns True if found and deactivated, False if not found.
        """
        entry = self._procedures.get(id)
        if entry is None:
            return False
        entry.active = False
        entry.updated_at = time.time()
        logger.debug("Procedure deactivated: %s", id)
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_all(
        self,
        domain: str | None = None,
        include_inactive: bool = False,
    ) -> list[ProcedureEntry]:
        """Return procedures sorted by importance descending.

        Args:
            domain: Filter to a specific domain (None = all domains).
            include_inactive: If True, include deactivated procedures.
        """
        entries = [
            e for e in self._procedures.values()
            if (include_inactive or e.active)
            and (domain is None or e.domain == domain or e.domain == "general")
        ]
        return sorted(entries, key=lambda e: e.importance, reverse=True)

    def get_prompt(
        self,
        domain: str | None = None,
        header: str = "## Behavioral Rules",
    ) -> str:
        """Format active procedures as a system prompt block.

        Args:
            domain: If provided, include only general + domain-specific rules.
            header: Section header inserted at the top.

        Returns:
            Formatted multiline string ready for system prompt injection.
            Returns an empty string if no active procedures exist.
        """
        entries = self.get_all(domain=domain)
        if not entries:
            return ""

        lines = [header, ""]
        for e in entries:
            prefix = f"[{e.domain}] " if domain is None and e.domain != "general" else ""
            lines.append(f"- {prefix}{e.rule}")

        return "\n".join(lines) + "\n"

    def get(self, id: str) -> ProcedureEntry | None:
        """Look up a procedure by ID."""
        return self._procedures.get(id)

    @property
    def size(self) -> dict[str, int]:
        active = sum(1 for e in self._procedures.values() if e.active)
        return {"total": len(self._procedures), "active": active}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        """Persist all procedures (including inactive) to JSON."""
        path = Path(path).expanduser()
        state = {
            "version": "0.5.1",
            "saved_at": time.time(),
            "procedures": [e.model_dump() for e in self._procedures.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, default=str), encoding="utf-8")
        logger.info("ProceduralMemory saved to %s (%d rules)", path, len(self._procedures))

    def load_state(self, path: Path | str) -> None:
        """Restore procedures from JSON."""
        path = Path(path).expanduser()
        if not path.exists():
            logger.warning("No procedural memory file at %s", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._procedures = {
            e["id"]: ProcedureEntry(**e)
            for e in data.get("procedures", [])
        }
        logger.info(
            "ProceduralMemory loaded from %s (%d rules, %d active)",
            path,
            len(self._procedures),
            self.size["active"],
        )

    def __repr__(self) -> str:
        return f"ProceduralMemory(total={self.size['total']}, active={self.size['active']})"
