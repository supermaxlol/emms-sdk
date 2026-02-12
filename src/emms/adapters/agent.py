"""Agent adapter — makes EMMS a drop-in memory backend for LLM agents.

Compatible with OpenClaw-style agents and any LLM framework that needs
persistent, hierarchical memory with context-window management.

Usage::

    from emms.adapters.agent import AgentMemory

    mem = AgentMemory(agent_name="MyBot", storage_dir="~/.mybot/memory")

    # When a user message arrives
    context = mem.build_context("What did we discuss about markets?", max_tokens=4000)

    # After the agent responds
    mem.ingest_turn(user_msg="What about markets?", agent_msg="Here is what I know...")

    # Before context compaction
    mem.pre_compaction_flush()

    # Export to markdown (OpenClaw-compatible)
    mem.export_markdown("memory/MEMORY.md")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from emms.core.models import Experience, MemoryConfig
from emms.core.embeddings import EmbeddingProvider, HashEmbedder
from emms.emms import EMMS

logger = logging.getLogger(__name__)


class AgentMemory:
    """High-level memory interface designed for LLM agent integration.

    Wraps the full EMMS pipeline and adds:
    - Conversation turn tracking
    - Context window building for LLM prompts
    - Pre-compaction intelligent flush
    - Markdown import/export (OpenClaw compatible)
    - Session management with identity persistence
    - Consciousness-inspired narrative and meaning tracking
    """

    def __init__(
        self,
        agent_name: str = "Agent",
        storage_dir: str | Path | None = None,
        config: MemoryConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: Any | None = None,
    ):
        self.agent_name = agent_name
        self.storage_dir = Path(storage_dir) if storage_dir else None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        identity_path = self.storage_dir / "identity.json" if self.storage_dir else None

        # Use HashEmbedder as default if none provided
        self._embedder = embedder or HashEmbedder(dim=128)

        self.emms = EMMS(
            config=config or MemoryConfig(),
            identity_path=identity_path,
            embedder=self._embedder,
            vector_store=vector_store,
            enable_consciousness=True,
            enable_graph=True,
        )

        # Use consciousness modules from EMMS core (no duplicates)
        self.narrator = self.emms.narrator
        self.meaning_maker = self.emms.meaning_maker
        self.temporal = self.emms.temporal
        self.ego_boundary = self.emms.ego_boundary
        self.compressor = self.emms.compressor

        # Conversation tracking
        self._turns: list[dict[str, Any]] = []
        self._session_start = time.time()

        # Update identity name
        self.emms.identity.state.name = agent_name

    # ------------------------------------------------------------------
    # Core agent operations
    # ------------------------------------------------------------------

    def ingest_turn(
        self,
        user_msg: str,
        agent_msg: str = "",
        domain: str = "conversation",
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Record a conversation turn as an experience.

        Stores both the user message and agent response, enriched with
        consciousness-inspired meaning and narrative analysis.
        """
        content = f"User: {user_msg}"
        if agent_msg:
            content += f"\nAgent: {agent_msg}"

        experience = Experience(
            content=content,
            domain=domain,
            importance=importance,
            emotional_valence=self._estimate_valence(user_msg),
            emotional_intensity=self._estimate_intensity(user_msg),
            novelty=self._estimate_novelty(user_msg),
        )

        # EMMS pipeline (includes consciousness enrichment, graph, events)
        store_result = self.emms.store(experience)

        # Track turn
        turn = {
            "timestamp": time.time(),
            "user": user_msg,
            "agent": agent_msg,
            "experience_id": experience.id,
            "importance": importance,
        }
        self._turns.append(turn)

        return {
            **store_result,
            "narrative_coherence": self.narrator.coherence,
            "ego_boundary": self.ego_boundary.boundary_strength,
        }

    def build_context(
        self,
        query: str,
        max_tokens: int = 4000,
        include_narrative: bool = True,
        include_recent_turns: int = 5,
    ) -> str:
        """Build a context string for an LLM prompt.

        Retrieves relevant memories, adds narrative context, and
        formats everything within the token budget.

        Returns a string ready to insert into a system/user prompt.
        """
        sections: list[str] = []
        budget = max_tokens

        # 1. Agent identity narrative
        if include_narrative:
            narrative = self.narrator.build_narrative(self.agent_name)
            sections.append(f"## Identity\n{narrative}")
            budget -= len(narrative.split())

        # 2. Relevant memories from EMMS
        results = self.emms.retrieve(query, max_results=10)
        if results:
            mem_lines = []
            for r in results:
                line = (
                    f"- [{r.source_tier.value}] (score={r.score:.2f}) "
                    f"{r.memory.experience.content[:150]}"
                )
                words = len(line.split())
                if budget - words < 0:
                    break
                mem_lines.append(line)
                budget -= words
            if mem_lines:
                sections.append("## Relevant Memories\n" + "\n".join(mem_lines))

        # 3. Recent conversation turns
        if include_recent_turns and self._turns:
            recent = self._turns[-include_recent_turns:]
            turn_lines = []
            for t in recent:
                line = f"- User: {t['user'][:100]}"
                if t.get("agent"):
                    line += f" | Agent: {t['agent'][:100]}"
                words = len(line.split())
                if budget - words < 0:
                    break
                turn_lines.append(line)
                budget -= words
            if turn_lines:
                sections.append("## Recent Conversation\n" + "\n".join(turn_lines))

        return "\n\n".join(sections)

    def recall(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Retrieve memories relevant to a query.

        Combines hierarchical retrieval with cross-modal matching.
        """
        results = self.emms.retrieve(query, max_results=max_results)
        return [
            {
                "content": r.memory.experience.content,
                "domain": r.memory.experience.domain,
                "score": r.score,
                "tier": r.source_tier.value,
                "strategy": r.strategy,
                "importance": r.memory.experience.importance,
                "age_seconds": r.memory.age,
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Context compaction support
    # ------------------------------------------------------------------

    def pre_compaction_flush(self) -> dict[str, Any]:
        """Intelligent pre-compaction flush.

        Called before the LLM's context window is compacted/summarised.
        Saves durable information to prevent context loss.

        This is the EMMS answer to OpenClaw's pre-compaction flush, but
        much smarter: it uses importance scoring, not just a raw dump.
        """
        # Force consolidation across all tiers
        consolidation = self.emms.consolidate()

        # Save identity
        self.emms.save()

        # Compress long-term memories that haven't been compressed
        lt_items = list(self.emms.memory.long_term.values())
        compressed = []
        if lt_items:
            compressed = self.compressor.compress_batch(lt_items)

        return {
            "consolidated": consolidation,
            "identity_saved": True,
            "memories_compressed": len(compressed),
            "narrative_coherence": self.narrator.coherence,
        }

    # ------------------------------------------------------------------
    # Markdown import/export (OpenClaw compatible)
    # ------------------------------------------------------------------

    def export_markdown(self, path: str | Path | None = None) -> str:
        """Export memories as markdown (compatible with OpenClaw MEMORY.md).

        If path is given, writes to file. Always returns the markdown string.
        """
        lines = [
            f"# {self.agent_name} Memory",
            f"*Exported: {datetime.now().isoformat()}*",
            f"*Total experiences: {self.emms.identity.state.total_experiences}*",
            "",
            "## Identity",
            self.narrator.build_narrative(self.agent_name),
            "",
        ]

        # Domains
        domains = self.emms.identity.state.domains_seen
        if domains:
            lines.append("## Domains")
            for d in domains:
                lines.append(f"- {d}")
            lines.append("")

        # Key memories (semantic + long-term)
        lines.append("## Key Memories")
        for tier_name, tier_store in [
            ("Semantic", self.emms.memory.semantic),
            ("Long-term", self.emms.memory.long_term),
        ]:
            items = tier_store.values() if isinstance(tier_store, dict) else tier_store
            for item in items:
                lines.append(
                    f"- **[{tier_name}]** ({item.experience.domain}) "
                    f"{item.experience.content[:200]}"
                )
        lines.append("")

        # Recent activity
        lines.append("## Recent Activity")
        for entry in self.emms.identity.state.autobiographical[-10:]:
            lines.append(f"- [{entry.get('domain', '?')}] {entry.get('summary', '')}")
        lines.append("")

        # Stats
        stats = self.emms.stats
        lines.append("## Stats")
        lines.append(f"- Total stored: {stats['total_stored']}")
        lines.append(f"- Sessions: {stats['identity']['sessions']}")
        lines.append(f"- Narrative coherence: {self.narrator.coherence:.0%}")
        lines.append(f"- Ego boundary strength: {self.ego_boundary.boundary_strength:.2f}")
        lines.append("")

        md = "\n".join(lines)

        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md)

        return md

    def import_markdown(self, path: str | Path) -> int:
        """Import experiences from a markdown file.

        Parses markdown lines starting with '- ' as individual experiences.
        Returns the count of imported experiences.
        """
        p = Path(path)
        if not p.exists():
            return 0

        text = p.read_text()
        count = 0

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- ") and len(line) > 10:
                content = line[2:].strip()
                # Strip markdown bold markers
                content = content.replace("**", "")

                # Try to extract domain from [bracket] notation
                domain = "imported"
                if content.startswith("["):
                    bracket_end = content.find("]")
                    if bracket_end > 0:
                        domain = content[1:bracket_end].lower()
                        content = content[bracket_end + 1:].strip()

                exp = Experience(content=content, domain=domain, importance=0.4)
                self.emms.store(exp)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def save_session(self) -> None:
        """Save all state to disk (identity + memory + turns)."""
        if self.storage_dir:
            memory_path = self.storage_dir / "memory_state.json"
            self.emms.save(memory_path=memory_path)

            # Save turn log
            log_path = self.storage_dir / "turns.jsonl"
            with open(log_path, "a") as f:
                for turn in self._turns:
                    f.write(json.dumps(turn, default=str) + "\n")
            self._turns.clear()
        else:
            self.emms.save()

    def load_session(self) -> None:
        """Load saved state from disk."""
        if self.storage_dir:
            memory_path = self.storage_dir / "memory_state.json"
            self.emms.load(memory_path=memory_path)

    @property
    def status(self) -> dict[str, Any]:
        """Current agent memory status."""
        return {
            "agent_name": self.agent_name,
            "session_uptime": time.time() - self._session_start,
            "turns_this_session": len(self._turns),
            "emms_stats": self.emms.stats,
            "narrative_coherence": self.narrator.coherence,
            "ego_boundary": self.ego_boundary.boundary_strength,
            "meaning_values_tracked": len(self.meaning_maker.value_weights),
            "temporal_coherence": (
                self.temporal._continuity_score()
                if self.temporal._recent_domains else 1.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_valence(self, text: str) -> float:
        """Quick sentiment estimation from keywords."""
        positive = {
            "good", "great", "excellent", "happy", "love", "thanks",
            "awesome", "perfect", "wonderful", "amazing", "nice", "yes",
        }
        negative = {
            "bad", "terrible", "awful", "hate", "angry", "sad",
            "wrong", "error", "fail", "problem", "no", "never", "worst",
        }
        words = set(text.lower().split())
        pos = len(words & positive)
        neg = len(words & negative)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _estimate_intensity(self, text: str) -> float:
        """Estimate emotional intensity from punctuation and capitalization."""
        exclaim = text.count("!")
        question = text.count("?")
        caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        return min(1.0, (exclaim * 0.2 + question * 0.1 + caps * 2))

    def _estimate_novelty(self, text: str) -> float:
        """Estimate novelty based on overlap with recent turns."""
        if not self._turns:
            return 0.8  # First turn is novel

        recent_words = set()
        for t in self._turns[-5:]:
            recent_words.update(t["user"].lower().split())

        current_words = set(text.lower().split())
        if not current_words:
            return 0.5

        overlap = len(current_words & recent_words) / len(current_words)
        return 1.0 - overlap  # high overlap = low novelty
