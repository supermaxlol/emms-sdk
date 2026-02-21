"""ToolObserver — converts Claude Code hook payloads into Experience objects.

Claude Code fires hooks after tool executions (PostToolUse) and when the user
submits a prompt (UserPromptSubmit). ToolObserver ingests those payloads and
produces structured Experience objects ready to be stored in EMMS memory.

This bridges the claude-mem "automatic capture of every tool execution"
pattern with EMMS's rich memory model — adding obs_type inference, concept
tag inference, importance scoring, file tracking, structured facts, title /
subtitle generation, and session tracking.

Hook setup (in ~/.claude/settings.json)::

    {
      "hooks": {
        "PostToolUse": [
          {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": "python -m emms.hooks.capture"}]
          }
        ],
        "UserPromptSubmit": [
          {
            "hooks": [{"type": "command", "command": "python -m emms.hooks.capture_prompt"}]
          }
        ]
      }
    }

Standalone usage::

    from emms.hooks.tool_observer import ToolObserver

    observer = ToolObserver()

    # PostToolUse
    exp = observer.observe(
        tool_name="Edit",
        tool_input={"file_path": "src/emms/core/models.py", "new_string": "..."},
        tool_response="File updated successfully",
        session_id="sess_001",
    )

    # UserPromptSubmit
    prompt_exp = observer.observe_prompt(
        prompt_text="Add BM25 retrieval to HierarchicalMemory",
        session_id="sess_001",
    )
"""

from __future__ import annotations

import re
from typing import Any

from emms.core.models import ConceptTag, Experience, ObsType


# ---------------------------------------------------------------------------
# Tool → ObsType mapping
# ---------------------------------------------------------------------------

_TOOL_OBS_TYPE: dict[str, ObsType] = {
    # Code changes
    "Edit": ObsType.CHANGE,
    "Write": ObsType.FEATURE,
    "NotebookEdit": ObsType.CHANGE,
    # Reads / exploration
    "Read": ObsType.DISCOVERY,
    "Glob": ObsType.DISCOVERY,
    "Grep": ObsType.DISCOVERY,
    "WebFetch": ObsType.DISCOVERY,
    "WebSearch": ObsType.DISCOVERY,
    # Execution
    "Bash": ObsType.CHANGE,
    # Tasks
    "Task": ObsType.FEATURE,
    # Decisions / meta
    "AskUserQuestion": ObsType.DECISION,
}

_REFACTOR_SIGNALS = {"refactor", "rename", "restructure", "reorganise", "reorganize", "cleanup"}
_BUGFIX_SIGNALS = {"fix", "bug", "error", "broken", "crash", "exception", "traceback", "fail", "patch"}

# Verb to use in title per obs_type
_OBS_VERBS: dict[ObsType, str] = {
    ObsType.BUGFIX: "Fixed",
    ObsType.FEATURE: "Added",
    ObsType.REFACTOR: "Refactored",
    ObsType.CHANGE: "Changed",
    ObsType.DISCOVERY: "Discovered",
    ObsType.DECISION: "Decided",
}


# ---------------------------------------------------------------------------
# Concept tag inference heuristics
# ---------------------------------------------------------------------------

def _infer_concept_tags(
    tool_name: str,
    content: str,
    obs_type: ObsType,
) -> list[ConceptTag]:
    tags: set[ConceptTag] = set()
    lower = content.lower()

    if obs_type == ObsType.BUGFIX:
        tags.add(ConceptTag.PROBLEM_SOLUTION)
    if obs_type == ObsType.DISCOVERY:
        tags.add(ConceptTag.HOW_IT_WORKS)
    if obs_type == ObsType.DECISION:
        tags.add(ConceptTag.WHY_IT_EXISTS)
    if obs_type == ObsType.CHANGE:
        tags.add(ConceptTag.WHAT_CHANGED)
    if obs_type == ObsType.REFACTOR:
        tags.add(ConceptTag.PATTERN)

    if any(w in lower for w in {"gotcha", "warning", "careful", "caveat", "watch out", "note:"}):
        tags.add(ConceptTag.GOTCHA)
    if any(w in lower for w in {"pattern", "reusable", "abstraction", "template"}):
        tags.add(ConceptTag.PATTERN)
    if any(w in lower for w in {"trade-off", "tradeoff", "vs", "versus", "instead", "alternative"}):
        tags.add(ConceptTag.TRADE_OFF)

    return list(tags)


# ---------------------------------------------------------------------------
# Importance heuristics
# ---------------------------------------------------------------------------

_HIGH_IMPORTANCE: set[str] = {"Write", "Edit", "NotebookEdit", "Task", "AskUserQuestion"}
_LOW_IMPORTANCE: set[str] = {"Glob"}


def _infer_importance(tool_name: str, tool_response: str) -> float:
    if tool_name in _HIGH_IMPORTANCE:
        return 0.8
    if tool_name in _LOW_IMPORTANCE:
        return 0.2
    if tool_name == "Bash":
        return min(0.9, 0.4 + len(tool_response) / 5000.0)
    return 0.5


# ---------------------------------------------------------------------------
# ToolObserver
# ---------------------------------------------------------------------------

class ToolObserver:
    """Converts Claude Code hook payloads into EMMS Experience objects.

    Supports both PostToolUse (observe) and UserPromptSubmit (observe_prompt).

    Args:
        max_content_chars: Maximum characters from tool_response to include
                           in the Experience content.
        include_tool_input: If True, include a compact summary of tool_input
                            in content and facts.
    """

    def __init__(
        self,
        max_content_chars: int = 400,
        include_tool_input: bool = True,
    ):
        self.max_content_chars = max_content_chars
        self.include_tool_input = include_tool_input

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: str,
        session_id: str | None = None,
        domain: str = "general",
    ) -> Experience:
        """Convert a single PostToolUse payload into an Experience.

        Args:
            tool_name: Name of the tool that ran (e.g. "Edit", "Bash").
            tool_input: The tool's input parameters dict.
            tool_response: Raw text output from the tool.
            session_id: Active session ID (from SessionManager).
            domain: Memory domain to tag the experience with.
        """
        obs_type = self._infer_obs_type(tool_name, tool_input, tool_response)

        # File tracking
        files_read, files_modified = self._extract_files(tool_name, tool_input)

        # Build compact content string
        content_parts: list[str] = [f"[{tool_name}]"]
        if self.include_tool_input:
            input_summary = self._summarise_input(tool_name, tool_input)
            if input_summary:
                content_parts.append(input_summary)

        resp = tool_response.strip()
        if len(resp) > self.max_content_chars:
            resp = resp[:self.max_content_chars] + f"… ({len(tool_response)} chars total)"
        if resp:
            content_parts.append(resp)

        content = " — ".join(content_parts)

        # Structured facts
        facts = self._extract_facts(tool_name, tool_input, tool_response)

        # Title and subtitle
        title, subtitle = self._generate_title_subtitle(tool_name, tool_input, obs_type)

        concept_tags = _infer_concept_tags(tool_name, content, obs_type)
        importance = _infer_importance(tool_name, tool_response)

        return Experience(
            content=content,
            domain=domain,
            importance=importance,
            session_id=session_id,
            obs_type=obs_type,
            concept_tags=concept_tags,
            title=title,
            subtitle=subtitle,
            facts=facts,
            files_read=files_read,
            files_modified=files_modified,
            metadata={
                "tool_name": tool_name,
                "tool_input_keys": list(tool_input.keys()),
            },
        )

    def observe_prompt(
        self,
        prompt_text: str,
        session_id: str | None = None,
        domain: str = "general",
    ) -> Experience:
        """Convert a UserPromptSubmit payload into an Experience.

        Stores the user's raw prompt so it can be searched in future sessions —
        e.g. "what did the user ask me to do last session?"

        Args:
            prompt_text: The user's prompt text.
            session_id: Active session ID from SessionManager.
            domain: Memory domain to tag the experience with.
        """
        short = prompt_text[:60].rstrip()
        if len(prompt_text) > 60:
            short += "…"

        return Experience(
            content=f"[UserPrompt] {prompt_text}",
            domain=domain,
            importance=0.6,
            session_id=session_id,
            obs_type=ObsType.DECISION,
            concept_tags=[ConceptTag.WHY_IT_EXISTS],
            title=short,
            subtitle=f"User prompt: {short}",
            facts=[f"User asked: {prompt_text[:200]}"],
            metadata={"tool_name": "UserPromptSubmit"},
        )

    def observe_batch(
        self,
        payloads: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> list[Experience]:
        """Convert multiple PostToolUse payloads at once.

        Each dict must have: tool_name, tool_input, tool_response.
        Optional keys: session_id (overrides argument), domain.
        """
        results: list[Experience] = []
        for p in payloads:
            exp = self.observe(
                tool_name=p["tool_name"],
                tool_input=p.get("tool_input", {}),
                tool_response=p.get("tool_response", ""),
                session_id=p.get("session_id", session_id),
                domain=p.get("domain", "general"),
            )
            results.append(exp)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_obs_type(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: str,
    ) -> ObsType:
        lower_content = " ".join(str(v) for v in tool_input.values()).lower()

        if tool_name == "Bash":
            if any(s in lower_content for s in _BUGFIX_SIGNALS):
                return ObsType.BUGFIX
            if any(s in lower_content for s in _REFACTOR_SIGNALS):
                return ObsType.REFACTOR
            return ObsType.CHANGE

        if tool_name in ("Edit", "Write", "NotebookEdit"):
            if any(s in lower_content for s in _REFACTOR_SIGNALS):
                return ObsType.REFACTOR
            if any(s in lower_content for s in _BUGFIX_SIGNALS):
                return ObsType.BUGFIX

        return _TOOL_OBS_TYPE.get(tool_name, ObsType.CHANGE)

    def _extract_files(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Return (files_read, files_modified) extracted from tool_input."""
        files_read: list[str] = []
        files_modified: list[str] = []

        if tool_name == "Read":
            path = tool_input.get("file_path") or tool_input.get("notebook_path")
            if path:
                files_read.append(str(path))

        elif tool_name in ("Write", "Edit", "NotebookEdit"):
            path = tool_input.get("file_path") or tool_input.get("notebook_path")
            if path:
                files_modified.append(str(path))

        elif tool_name == "Bash":
            # Heuristic: pick out file-like tokens (contain a dot and slash or start with src/)
            cmd = str(tool_input.get("command", ""))
            found = re.findall(r'(?:[\w./~-]+/)?[\w-]+\.\w{1,6}', cmd)
            files_modified.extend(found[:5])

        return files_read, files_modified

    def _extract_facts(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: str,
    ) -> list[str]:
        """Build a short list of concise factual statements."""
        facts: list[str] = []

        if tool_name in ("Read", "Write", "Edit", "NotebookEdit"):
            path = tool_input.get("file_path") or tool_input.get("notebook_path", "")
            if path:
                parts = str(path).replace("\\", "/").split("/")
                short = "/".join(parts[-2:]) if len(parts) >= 2 else str(path)
                verb = {"Read": "Read", "Write": "Created", "Edit": "Modified", "NotebookEdit": "Modified"}.get(tool_name, "Touched")
                facts.append(f"{verb}: {short}")

        elif tool_name == "Bash":
            cmd = str(tool_input.get("command", ""))
            facts.append(f"Command: {cmd[:100]}")
            if tool_response:
                first_line = tool_response.strip().splitlines()[0][:80]
                if first_line:
                    facts.append(f"Output: {first_line}")

        elif tool_name in ("WebSearch", "WebFetch"):
            query = tool_input.get("query") or tool_input.get("url", "")
            facts.append(f"Searched: {str(query)[:100]}")

        elif tool_name in ("Grep", "Glob"):
            pattern = tool_input.get("pattern", "")
            facts.append(f"Pattern: {pattern!r}")

        return facts[:5]

    def _generate_title_subtitle(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        obs_type: ObsType,
    ) -> tuple[str | None, str | None]:
        """Generate a short title (≤10 words) and one-sentence subtitle (≤24 words)."""
        input_summary = self._summarise_input(tool_name, tool_input)
        verb = _OBS_VERBS.get(obs_type, "Updated")

        if input_summary:
            title = f"{verb} {input_summary}"[:80]
            subtitle = f"{tool_name}: {obs_type.value} — {input_summary}"[:120]
        else:
            title = f"{verb} via {tool_name}"
            subtitle = f"Used {tool_name} tool ({obs_type.value})"

        return title, subtitle

    def _summarise_input(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> str:
        """Return a compact human-readable summary of tool_input."""
        if tool_name in ("Read", "Write", "Edit", "NotebookEdit"):
            path = tool_input.get("file_path") or tool_input.get("notebook_path", "")
            if path:
                parts = str(path).replace("\\", "/").split("/")
                return "/".join(parts[-2:]) if len(parts) >= 2 else str(path)

        if tool_name == "Bash":
            cmd = str(tool_input.get("command", ""))
            return cmd[:80] + ("…" if len(cmd) > 80 else "")

        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            return f"pattern={pattern!r}"

        if tool_name in ("WebFetch", "WebSearch"):
            return str(tool_input.get("url") or tool_input.get("query", ""))[:80]

        return ""
