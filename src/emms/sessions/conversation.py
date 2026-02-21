"""ConversationBuffer — sliding-window conversation history manager.

Maintains a bounded buffer of conversation turns (user/assistant messages).
When the buffer exceeds ``window_size`` turns, older turns are summarised
into a compact chunk and the raw turns are evicted, keeping the window small.

The summarisation strategy:
1. **Extractive** (default, zero dependencies): extract the most important
   sentences by content diversity (deduplicated keyword overlap).
2. **LLM-backed** (optional): when an ``LLMEnhancer`` is supplied, the chunk
   is summarised via an LLM call; falls back to extractive on error.

Usage::

    from emms.sessions.conversation import ConversationBuffer

    buf = ConversationBuffer(window_size=10, summarise_chunk=5)

    buf.observe_turn("user", "What is spaced repetition?")
    buf.observe_turn("assistant", "Spaced repetition is a learning technique…")

    context = buf.get_context(max_tokens=500)
    print(context)
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.llm.enhancer import LLMEnhancer


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in the conversation.

    Attributes
    ----------
    role : "user" | "assistant" | "system"
    content : raw text of this turn
    timestamp : Unix time when the turn was observed
    turn_index : monotonically increasing index across all turns seen
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationChunk:
    """A compressed summary of several evicted turns.

    Attributes
    ----------
    summary : the extractive / LLM summary text
    turn_range : (first_turn_index, last_turn_index) of evicted turns
    timestamp : when the chunk was created
    """

    summary: str
    turn_range: tuple[int, int] = (0, 0)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Extractive summariser (zero dependencies)
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","i","you","he","she","it","we","they","not","so",
    "that","this","my","your","if","as","be","do","did","can","will","just",
    "me","him","her","them","us",
})


def _score_sentence(sentence: str, domain_keywords: set[str]) -> float:
    """Score a sentence by keyword coverage + length penalty."""
    words = {w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", sentence)
             if w.lower() not in _STOP}
    if not words:
        return 0.0
    overlap = len(words & domain_keywords)
    length_bonus = min(1.0, len(words) / 15.0)
    return (overlap + length_bonus) / (1 + len(words) / 30)


def _extractive_summary(texts: list[str], max_sentences: int = 4) -> str:
    """Produce an extractive summary from a list of text snippets."""
    # Collect all sentences
    sentences: list[str] = []
    for text in texts:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences.extend(s for s in sents if len(s) > 10)

    if not sentences:
        return " ".join(t[:80] for t in texts)

    # Build domain keyword set from all text
    all_words = re.findall(r"\b[A-Za-z]{4,}\b", " ".join(texts))
    from collections import Counter
    word_freq = Counter(w.lower() for w in all_words if w.lower() not in _STOP)
    top_keywords = {w for w, _ in word_freq.most_common(20)}

    # Score each sentence
    scored = [(s, _score_sentence(s, top_keywords)) for s in sentences]
    scored.sort(key=lambda x: -x[1])

    # Take top sentences (deduplicated by high overlap)
    selected: list[str] = []
    selected_words: list[set[str]] = []

    for sent, score in scored:
        if len(selected) >= max_sentences:
            break
        sent_words = {w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", sent)}
        # Skip if too similar to an already-selected sentence
        is_dup = any(
            len(sent_words & sw) / max(1, len(sent_words | sw)) > 0.7
            for sw in selected_words
        )
        if not is_dup:
            selected.append(sent)
            selected_words.append(sent_words)

    # Restore original order
    order_map = {s: i for i, s in enumerate(sentences)}
    selected.sort(key=lambda s: order_map.get(s, 0))

    return " ".join(selected) if selected else sentences[0][:200]


# ---------------------------------------------------------------------------
# ConversationBuffer
# ---------------------------------------------------------------------------

class ConversationBuffer:
    """Sliding-window conversation history with automatic chunking.

    Parameters
    ----------
    window_size : max number of raw turns kept in the live window.
    summarise_chunk : how many turns to summarise when the window overflows.
    llm_enhancer : optional LLMEnhancer for LLM-backed summarisation.
    """

    def __init__(
        self,
        window_size: int = 20,
        summarise_chunk: int = 5,
        llm_enhancer: "LLMEnhancer | None" = None,
    ):
        self.window_size = window_size
        self.summarise_chunk = min(summarise_chunk, window_size)
        self.llm_enhancer = llm_enhancer

        self._turns: deque[ConversationTurn] = deque()
        self._chunks: list[ConversationChunk] = []
        self._total_turns = 0  # monotonic counter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Add a new turn to the buffer.

        Automatically evicts and summarises oldest turns if the window is full.

        Parameters
        ----------
        role : "user" | "assistant" | "system"
        content : turn text
        metadata : optional extra data attached to the turn

        Returns
        -------
        ConversationTurn — the stored turn object.
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            turn_index=self._total_turns,
            metadata=metadata or {},
        )
        self._total_turns += 1
        self._turns.append(turn)

        # Evict if over capacity
        while len(self._turns) > self.window_size:
            self._evict_chunk()

        return turn

    def get_context(self, max_tokens: int = 2000) -> str:
        """Build a context string for prompt injection.

        Includes compressed chunks first (oldest → newest), then the live
        turn window. Truncates to ``max_tokens`` (estimated at 4 chars/token).

        Parameters
        ----------
        max_tokens : approximate token budget.

        Returns
        -------
        str — formatted context block.
        """
        max_chars = max_tokens * 4
        lines: list[str] = []

        # Compressed chunks
        if self._chunks:
            lines.append("### Conversation History (summarised)\n")
            for chunk in self._chunks:
                lines.append(f"[Turns {chunk.turn_range[0]}–{chunk.turn_range[1]}] "
                              f"{chunk.summary}")
            lines.append("")

        # Live window
        if self._turns:
            lines.append("### Recent Conversation\n")
            for turn in self._turns:
                role_tag = turn.role.upper()
                lines.append(f"**{role_tag}**: {turn.content}")

        context = "\n".join(lines)

        # Truncate if over budget
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[…truncated]"

        return context

    def clear(self) -> None:
        """Clear all turns and chunks."""
        self._turns.clear()
        self._chunks.clear()

    @property
    def turn_count(self) -> int:
        """Number of raw turns currently in the live window."""
        return len(self._turns)

    @property
    def chunk_count(self) -> int:
        """Number of compressed chunks stored."""
        return len(self._chunks)

    @property
    def total_turns_seen(self) -> int:
        """Total turns observed since the buffer was created."""
        return self._total_turns

    def all_turns(self) -> list[ConversationTurn]:
        """Return the live turn window as a list."""
        return list(self._turns)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_chunk(self) -> None:
        """Evict the oldest ``summarise_chunk`` turns and compress them."""
        to_evict: list[ConversationTurn] = []
        for _ in range(min(self.summarise_chunk, len(self._turns))):
            to_evict.append(self._turns.popleft())

        if not to_evict:
            return

        # Summarise the evicted turns
        turn_range = (to_evict[0].turn_index, to_evict[-1].turn_index)
        summary = self._summarise(to_evict)
        chunk = ConversationChunk(summary=summary, turn_range=turn_range)
        self._chunks.append(chunk)

    def _summarise(self, turns: list[ConversationTurn]) -> str:
        """Summarise a list of turns into a compact text."""
        texts = [f"{t.role}: {t.content}" for t in turns]

        # Try LLM summarisation first
        if self.llm_enhancer is not None:
            try:
                return self._llm_summarise(texts)
            except Exception:
                pass  # fallback to extractive

        return _extractive_summary(texts, max_sentences=3)

    def _llm_summarise(self, texts: list[str]) -> str:
        """Use LLMEnhancer to summarise conversation turns."""
        combined = "\n".join(texts)
        prompt = (
            "Summarise the following conversation excerpt in 2–3 sentences. "
            "Focus on key decisions, questions, and information exchanged:\n\n"
            f"{combined}"
        )
        result = self.llm_enhancer.classify(
            content=combined,
            context="Summarise this conversation excerpt in 2-3 sentences.",
        )
        # classify returns a dict; use 'summary' key if present, else 'classification'
        return str(result.get("summary") or result.get("classification") or combined[:200])
