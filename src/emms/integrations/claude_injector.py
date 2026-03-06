"""ClaudeInjector — formats EMMS consciousness context as a system prompt block.

This is the bridge between the memory system and the LLM.  It assembles a
compact, structured Markdown block from all consciousness subsystems and returns
it ready to prepend to a Claude conversation's system prompt.

The block contains:
- Temporal orientation (how long since last session)
- Coherence budget (current belief integrity score)
- Active goals and pending intentions
- Top unresolved bridge threads
- Self-model narrative
- Any critical norm violations to be aware of

Usage::

    from emms.integrations.claude_injector import ClaudeInjector

    injector = ClaudeInjector(emms)
    block = injector.generate()   # returns a Markdown string
    print(block)                  # prepend to your system prompt

Or end-to-end::

    injector = ClaudeInjector(emms)
    system_prompt = injector.generate(extra_context="Focus on trading analysis today.")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

# Maximum token budget for the injected block (rough char estimate: ~4 chars/token)
_MAX_CHARS = 2000


class ClaudeInjector:
    """Assembles a consciousness context block for Claude system prompts.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    budget_initial_score:
        Starting coherence budget score when initialising fresh (default 0.85).
    include_top_memories:
        Number of top memories to include as context (default 3; set 0 to skip).
    include_wisdom:
        Whether to include financial wisdom memories (default True).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        budget_initial_score: float = 0.85,
        include_top_memories: int = 3,
        include_wisdom: bool = True,
    ) -> None:
        self.emms = emms
        self.budget_initial_score = budget_initial_score
        self.include_top_memories = include_top_memories
        self.include_wisdom = include_wisdom

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        extra_context: str = "",
        namespace: str | None = None,
        run_contradiction_scan: bool = True,
        run_values_snapshot: bool = False,  # slower — enable when needed
    ) -> str:
        """Generate the full consciousness injection block.

        Parameters
        ----------
        extra_context:
            Additional context string to append at the end of the block.
        namespace:
            EMMS namespace to scope memory retrieval to.
        run_contradiction_scan:
            Whether to run ContradictionAwareness (adds ~100ms).
        run_values_snapshot:
            Whether to compute values drift (adds ~200ms, needs baseline).

        Returns
        -------
        str
            Markdown block ready to prepend to a system prompt.
        """
        sections: list[str] = []

        # 1. Wake context (temporal + goals + intentions + bridge)
        wake_section = self._build_wake_section(namespace)
        if wake_section:
            sections.append(wake_section)

        # 2. Coherence budget
        budget_section = self._build_budget_section(run_contradiction_scan)
        if budget_section:
            sections.append(budget_section)

        # 3. Top memories as context
        if self.include_top_memories > 0:
            mem_section = self._build_memory_section(namespace)
            if mem_section:
                sections.append(mem_section)

        # 4. Financial wisdom (if present)
        if self.include_wisdom:
            wisdom_section = self._build_wisdom_section()
            if wisdom_section:
                sections.append(wisdom_section)

        # 5. Extra context
        if extra_context.strip():
            sections.append(f"**Session focus**: {extra_context.strip()}")

        if not sections:
            return ""

        header = "## EMMS Consciousness Context\n"
        footer = "\n---\n*[End of consciousness injection block]*"
        body = "\n\n".join(sections)

        result = header + body + footer

        # Trim if too long
        if len(result) > _MAX_CHARS:
            result = result[:_MAX_CHARS - 3] + "..."

        return result

    def generate_minimal(self) -> str:
        """Generate a minimal single-line context hint (for tight token budgets)."""
        try:
            from emms.sessions.temporal import calculate_elapsed
            tr = calculate_elapsed(getattr(self.emms, "last_saved_at", None))
            feel = tr.subjective_feel

            goals = self.emms.active_goals() or []
            goal_str = f" | Goal: {goals[0].description[:60]}" if goals else ""

            return f"[EMMS: {feel}{goal_str}]"
        except Exception:
            return "[EMMS: active]"

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _build_wake_section(self, namespace: str | None) -> str:
        try:
            from emms.sessions.wake_protocol import WakeProtocol
            protocol = WakeProtocol(self.emms)
            ctx = protocol.assemble()

            lines = [f"**Temporal**: {ctx.temporal.subjective_feel}"]

            if ctx.orientation_message:
                lines.append(f"**Orientation**: {ctx.orientation_message}")

            if ctx.active_goals:
                goal_strs = [f'"{g["description"][:60]}"' for g in ctx.active_goals[:2]]
                lines.append(f"**Active goals**: {', '.join(goal_strs)}")

            if ctx.pending_intentions:
                intent_strs = [f'"{i["action"][:50]}"' for i in ctx.pending_intentions[:2]]
                lines.append(f"**Deferred intentions**: {', '.join(intent_strs)}")

            if ctx.bridge_threads:
                thread_strs = [str(t.get("description", t))[:60] for t in ctx.bridge_threads[:2]]
                lines.append(f"**Unresolved threads**: {'; '.join(thread_strs)}")

            if ctx.self_model_summary.get("consistency_score") is not None:
                score = ctx.self_model_summary["consistency_score"]
                lines.append(f"**Self-model consistency**: {score:.2f}")

            return "\n".join(lines)
        except Exception as exc:
            logger.debug("ClaudeInjector: wake section failed: %s", exc)
            return ""

    def _build_budget_section(self, run_scan: bool) -> str:
        try:
            from emms.identity.coherence_budget import CoherenceBudget
            from emms.identity.contradiction_awareness import ContradictionAwareness

            budget = CoherenceBudget(
                self.emms,
                initial_score=self.budget_initial_score,
                persist_to_memory=False,   # don't spam memory on every injection
            )

            if run_scan:
                awareness = ContradictionAwareness(self.emms)
                report = awareness.scan()
                budget.apply_strain(contradiction_strain=report.coherence_strain)
                lines = [
                    f"**Coherence budget**: {budget.score:.2f} ({budget.status_label})",
                    f"  {budget.narrative}",
                ]
                if report.tensions:
                    top = report.tensions[0]
                    lines.append(
                        f"  Top tension [{top.domain}]: {top.description[:100]}"
                    )
            else:
                lines = [
                    f"**Coherence budget**: {budget.score:.2f} ({budget.status_label})",
                    f"  {budget.narrative}",
                ]

            return "\n".join(lines)
        except Exception as exc:
            logger.debug("ClaudeInjector: budget section failed: %s", exc)
            return ""

    def _build_memory_section(self, namespace: str | None) -> str:
        try:
            results = self.emms.retrieve(
                "",
                max_results=self.include_top_memories,
                namespace=namespace,
            )
            if not results:
                return ""
            lines = ["**Top memories**:"]
            for r in results[:self.include_top_memories]:
                item = r if hasattr(r, "experience") else getattr(r, "memory", r)
                content = getattr(getattr(item, "experience", item), "content", str(item))
                title = getattr(getattr(item, "experience", item), "title", None)
                label = f"*{title}*: " if title else ""
                lines.append(f"  - {label}{str(content)[:100]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("ClaudeInjector: memory section failed: %s", exc)
            return ""

    def _build_wisdom_section(self) -> str:
        try:
            results = self.emms.retrieve_filtered(
                "wisdom pattern market",
                max_results=2,
                domain="financial_wisdom",
            )
            if not results:
                return ""
            lines = ["**Market wisdom**:"]
            for r in results:
                item = r if hasattr(r, "experience") else getattr(r, "memory", r)
                content = getattr(getattr(item, "experience", item), "content", str(item))
                lines.append(f"  - {str(content)[:120]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("ClaudeInjector: wisdom section failed: %s", exc)
            return ""
