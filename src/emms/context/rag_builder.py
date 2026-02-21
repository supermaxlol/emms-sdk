"""RAGContextBuilder — token-budget-aware context packing for RAG pipelines.

Builds a structured context document from retrieved EMMS memories, fitting
within a caller-specified token budget.  Supports four output formats:

* ``markdown`` (default) — GitHub-flavoured Markdown with H2 section headers
* ``xml`` — ``<context>`` / ``<memory>`` tags, good for Claude-style prompting
* ``json`` — list of dicts, for programmatic consumption
* ``plain`` — numbered plain-text blocks, minimal formatting overhead

Usage::

    from emms import EMMS
    from emms.context.rag_builder import RAGContextBuilder

    agent = EMMS()
    ...
    results = agent.retrieve("auth bug", max_results=20)

    builder = RAGContextBuilder(token_budget=4000)
    context = builder.build(results, fmt="xml")

Or via the EMMS facade::

    context = agent.build_rag_context("auth bug", max_results=20, fmt="markdown")
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from emms.core.models import RetrievalResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WORDS_PER_TOKEN = 0.75   # rough approximation: 1 token ≈ 0.75 words (for budget)
_TOKEN_PER_WORD = 1.0 / _WORDS_PER_TOKEN

_FmtLiteral = Literal["markdown", "xml", "json", "plain"]


# ---------------------------------------------------------------------------
# Internal block type
# ---------------------------------------------------------------------------

@dataclass
class ContextBlock:
    """One memory packed into the context window."""
    memory_id: str
    experience_id: str
    content: str
    title: str | None
    facts: list[str]
    domain: str
    score: float
    tier: str
    namespace: str
    confidence: float
    token_estimate: int

    @property
    def effective_content(self) -> str:
        """Return best content representation for this block."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.facts:
            parts.extend(f"- {f}" for f in self.facts[:5])
        if not parts:
            parts.append(self.content[:500])
        elif self.content:
            # Append a truncated raw content for completeness
            snippet = self.content[:200]
            if snippet not in " ".join(parts):
                parts.append(snippet)
        return "\n".join(parts)

    @classmethod
    def from_retrieval_result(cls, result: "RetrievalResult") -> "ContextBlock":
        item = result.memory
        exp = item.experience
        content = exp.content
        title = exp.title
        facts = list(exp.facts or [])
        effective = (title or "") + " " + " ".join(facts) + " " + content[:200]
        tok_est = max(1, int(len(effective.split()) * _TOKEN_PER_WORD))
        return cls(
            memory_id=item.id,
            experience_id=exp.id,
            content=content,
            title=title,
            facts=facts,
            domain=exp.domain,
            score=result.score,
            tier=result.source_tier.value,
            namespace=exp.namespace,
            confidence=exp.confidence,
            token_estimate=tok_est,
        )


# ---------------------------------------------------------------------------
# RAGContextBuilder
# ---------------------------------------------------------------------------

class RAGContextBuilder:
    """Build a token-budget-aware context document from retrieved memories.

    Parameters
    ----------
    token_budget:
        Maximum number of tokens for the assembled context.  Memories are
        added in score-descending order until the budget is exhausted.
    header:
        Optional header line prepended to the output (e.g. "Relevant memories:").
    include_metadata:
        Whether to include score/tier/namespace/confidence annotations.
    min_score:
        Skip results below this score threshold.
    """

    def __init__(
        self,
        token_budget: int = 4000,
        header: str | None = None,
        include_metadata: bool = True,
        min_score: float = 0.0,
    ) -> None:
        self.token_budget = token_budget
        self.header = header
        self.include_metadata = include_metadata
        self.min_score = min_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        results: list["RetrievalResult"],
        fmt: _FmtLiteral = "markdown",
    ) -> str:
        """Pack *results* into a context string within the token budget.

        Args:
            results: Retrieved memories from ``EMMS.retrieve()`` or similar.
            fmt: Output format — ``markdown``, ``xml``, ``json``, or ``plain``.

        Returns:
            A string ready to inject into an LLM prompt.
        """
        blocks = self._select_blocks(results)
        renderer = {
            "markdown": self._render_markdown,
            "xml":      self._render_xml,
            "json":     self._render_json,
            "plain":    self._render_plain,
        }.get(fmt, self._render_markdown)
        return renderer(blocks)

    def build_blocks(self, results: list["RetrievalResult"]) -> list[ContextBlock]:
        """Return the selected blocks without rendering — useful for inspection."""
        return self._select_blocks(results)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        return max(1, int(len(text.split()) * _TOKEN_PER_WORD))

    # ------------------------------------------------------------------
    # Block selection (greedy, score-descending)
    # ------------------------------------------------------------------

    def _select_blocks(self, results: list["RetrievalResult"]) -> list[ContextBlock]:
        """Select blocks up to token_budget in score-descending order."""
        # Sort by score descending (results may already be sorted, but be safe)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        selected: list[ContextBlock] = []
        used_tokens = self.estimate_tokens(self.header or "")

        for result in sorted_results:
            if result.score < self.min_score:
                continue
            block = ContextBlock.from_retrieval_result(result)
            if used_tokens + block.token_estimate > self.token_budget:
                # Try to fit a smaller summary (just title)
                if block.title:
                    tiny_est = self.estimate_tokens(block.title)
                    if used_tokens + tiny_est <= self.token_budget:
                        # Truncate block to title only
                        block = ContextBlock(
                            memory_id=block.memory_id,
                            experience_id=block.experience_id,
                            content="",
                            title=block.title,
                            facts=[],
                            domain=block.domain,
                            score=block.score,
                            tier=block.tier,
                            namespace=block.namespace,
                            confidence=block.confidence,
                            token_estimate=tiny_est,
                        )
                        selected.append(block)
                        used_tokens += tiny_est
                # Either way, stop adding after budget is hit
                break
            selected.append(block)
            used_tokens += block.token_estimate

        return selected

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _render_markdown(self, blocks: list[ContextBlock]) -> str:
        lines: list[str] = []
        if self.header:
            lines.append(f"# {self.header}\n")
        for i, b in enumerate(blocks, 1):
            meta = ""
            if self.include_metadata:
                meta = (f" *(score={b.score:.3f}, tier={b.tier},"
                        f" ns={b.namespace}, conf={b.confidence:.2f})*")
            heading = b.title or f"Memory {i}"
            lines.append(f"## {heading}{meta}")
            if b.facts:
                for f_ in b.facts:
                    lines.append(f"- {f_}")
            if b.content and not b.facts:
                lines.append(b.content[:500])
            elif b.content and b.facts:
                lines.append(f"\n> {b.content[:200]}")
            lines.append("")
        return "\n".join(lines).strip()

    def _render_xml(self, blocks: list[ContextBlock]) -> str:
        parts: list[str] = []
        if self.header:
            parts.append(f"<!-- {self.header} -->")
        parts.append("<context>")
        for b in blocks:
            attrs = f'id="{b.memory_id}" domain="{b.domain}" score="{b.score:.3f}"'
            if self.include_metadata:
                attrs += (f' tier="{b.tier}" namespace="{b.namespace}"'
                          f' confidence="{b.confidence:.2f}"')
            parts.append(f"  <memory {attrs}>")
            if b.title:
                parts.append(f"    <title>{_xml_escape(b.title)}</title>")
            if b.facts:
                parts.append("    <facts>")
                for f_ in b.facts:
                    parts.append(f"      <fact>{_xml_escape(f_)}</fact>")
                parts.append("    </facts>")
            if b.content:
                snippet = _xml_escape(b.content[:500])
                parts.append(f"    <content>{snippet}</content>")
            parts.append("  </memory>")
        parts.append("</context>")
        return "\n".join(parts)

    def _render_json(self, blocks: list[ContextBlock]) -> str:
        data: list[dict[str, Any]] = []
        for b in blocks:
            entry: dict[str, Any] = {
                "memory_id": b.memory_id,
                "domain": b.domain,
                "score": round(b.score, 4),
                "title": b.title,
                "facts": b.facts,
                "content": b.content[:500] if b.content else "",
            }
            if self.include_metadata:
                entry.update({
                    "tier": b.tier,
                    "namespace": b.namespace,
                    "confidence": round(b.confidence, 4),
                })
            data.append(entry)
        return _json.dumps(data, ensure_ascii=False, indent=2)

    def _render_plain(self, blocks: list[ContextBlock]) -> str:
        lines: list[str] = []
        if self.header:
            lines.append(self.header)
            lines.append("=" * len(self.header))
            lines.append("")
        for i, b in enumerate(blocks, 1):
            heading = b.title or f"[{b.memory_id}]"
            lines.append(f"{i}. {heading}")
            if b.facts:
                for f_ in b.facts:
                    lines.append(f"   • {f_}")
            if b.content and not b.facts:
                lines.append(f"   {b.content[:300]}")
            if self.include_metadata:
                lines.append(
                    f"   [score={b.score:.3f} tier={b.tier} ns={b.namespace}]"
                )
            lines.append("")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xml_escape(s: str) -> str:
    return (s
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
