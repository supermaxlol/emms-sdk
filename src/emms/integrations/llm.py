"""LLM integration layer for EMMS.

Provides hooks for Claude, GPT, and local models (Ollama) to enhance
EMMS operations — entity extraction, narrative generation, experience
enrichment, and memory summarisation.

All LLM features are **optional** — EMMS works fully without any LLM.
When an LLM is configured, it enhances the quality of:
- Entity extraction (vs regex fallback)
- Narrative generation (vs template-based)
- Memory summarisation (vs extractive)

Usage::

    from emms.integrations.llm import ClaudeProvider, LLMEnhancer

    provider = ClaudeProvider(api_key="sk-...")
    enhancer = LLMEnhancer(provider)
    enriched = await enhancer.enrich_experience(experience)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from emms.core.models import Experience, MemoryItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM backends."""

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        ...


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

class ClaudeProvider:
    """Anthropic Claude API provider.

    Requires: pip install anthropic
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic not installed. Run: pip install anthropic"
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """OpenAI-compatible API provider (works with GPT, local servers, etc.).

    Requires: pip install openai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai not installed. Run: pip install openai"
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model = model

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Ollama provider (local models)
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Ollama local model provider.

    Requires: Ollama running locally (default: http://localhost:11434)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp not installed. Run: pip install aiohttp"
            )

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text}")
                data = await resp.json()
                return data.get("response", "")


# ---------------------------------------------------------------------------
# LLM Enhancer — uses any provider to enhance EMMS operations
# ---------------------------------------------------------------------------

class LLMEnhancer:
    """Uses an LLM to enhance EMMS operations.

    All methods are async and optional — EMMS works without them.

    When ``emms`` is provided, identity-aware prompts are automatically
    used for narrative generation, summarisation, and the ``ask()``
    method — proven to achieve 100% identity adoption on Sonnet.

    Usage::

        enhancer = LLMEnhancer(provider, emms=my_emms)
        answer = await enhancer.ask("What do you remember?")
    """

    def __init__(
        self,
        provider: LLMProvider,
        emms: Any | None = None,
        agent_name: str = "EMMS-Agent",
    ):
        self.provider = provider
        self._emms = emms
        self._agent_name = agent_name
        self._prompt_builder = None

    @property
    def prompt_builder(self):
        """Lazy-init the IdentityPromptBuilder."""
        if self._prompt_builder is None and self._emms is not None:
            from emms.prompts.identity import IdentityPromptBuilder
            self._prompt_builder = IdentityPromptBuilder(
                self._emms, agent_name=self._agent_name,
            )
        return self._prompt_builder

    def _get_model_id(self) -> str | None:
        """Try to extract model identifier from provider."""
        return getattr(self.provider, "_model", None)

    async def enrich_experience(self, experience: Experience) -> Experience:
        """Use LLM to assess importance and novelty of an experience."""
        prompt = (
            "Analyze this text and respond with ONLY a JSON object containing:\n"
            '{"importance": 0.0-1.0, "novelty": 0.0-1.0, '
            '"domain": "string", "emotional_valence": -1.0 to 1.0, '
            '"emotional_intensity": 0.0-1.0}\n\n'
            f"Text: {experience.content[:500]}"
        )

        try:
            response = await self.provider.generate(prompt, max_tokens=100)
            # Extract JSON from response
            data = self._parse_json(response)
            if data:
                if "importance" in data:
                    experience.importance = float(data["importance"])
                if "novelty" in data:
                    experience.novelty = float(data["novelty"])
                if "domain" in data and experience.domain == "general":
                    experience.domain = str(data["domain"])
                if "emotional_valence" in data:
                    experience.emotional_valence = float(data["emotional_valence"])
                if "emotional_intensity" in data:
                    experience.emotional_intensity = float(data["emotional_intensity"])
        except Exception:
            logger.debug("LLM enrichment failed, using defaults", exc_info=True)

        return experience

    async def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Use LLM to extract entities from text."""
        prompt = (
            "Extract named entities from this text. Respond with ONLY a JSON array "
            "of objects with 'name' and 'type' fields. Types: person, org, concept, "
            "location, event.\n\n"
            f"Text: {text[:500]}"
        )

        try:
            response = await self.provider.generate(prompt, max_tokens=300)
            data = self._parse_json(response)
            if isinstance(data, list):
                return data
        except Exception:
            logger.debug("LLM entity extraction failed", exc_info=True)

        return []

    async def generate_narrative(self, context: dict[str, Any]) -> str:
        """Use LLM to generate a rich self-narrative.

        If EMMS is attached, uses the identity prompt template (proven
        to achieve 100% adoption on Sonnet).
        """
        if self.prompt_builder is not None:
            prompt = self.prompt_builder.narrative_prompt()
        else:
            themes = context.get("themes", {})
            traits = context.get("traits", {})
            experiences = context.get("experience_count", 0)
            domains = context.get("domains", [])
            prompt = (
                "You are an AI agent reflecting on your experiences. "
                "Generate a brief first-person narrative (2-3 sentences) based on:\n"
                f"- Total experiences: {experiences}\n"
                f"- Top themes: {dict(list(themes.items())[:5])}\n"
                f"- Personality traits: {traits}\n"
                f"- Domains explored: {domains}\n\n"
                "Write naturally in first person."
            )

        try:
            return await self.provider.generate(prompt, max_tokens=200)
        except Exception:
            logger.debug("LLM narrative generation failed", exc_info=True)
            return ""

    async def summarize_memories(
        self, items: list[MemoryItem], max_items: int = 20
    ) -> str:
        """Use LLM to create a coherent summary of memories.

        If EMMS is attached, uses the identity summary prompt.
        """
        if self.prompt_builder is not None:
            prompt = self.prompt_builder.summary_prompt()
        else:
            contents = [item.experience.content[:100] for item in items[:max_items]]
            memories_text = "\n".join(f"- {c}" for c in contents)
            prompt = (
                "Summarize these memories into a coherent 2-3 sentence overview "
                "that captures the key themes and important details:\n\n"
                f"{memories_text}"
            )

        try:
            return await self.provider.generate(prompt, max_tokens=200)
        except Exception:
            logger.debug("LLM summarization failed", exc_info=True)
            return ""

    async def ask(
        self,
        question: str,
        strategy: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Ask a question with full identity context.

        This is the primary interface for identity-aware conversations.
        The prompt strategy is auto-selected based on the provider model
        using empirical data from 72-trial adoption study.

        Args:
            question: The user's question.
            strategy: Override strategy ("system_prompt", "framed", "naive").
                      If None, auto-selects based on model.
            max_tokens: Maximum response tokens.

        Returns:
            The LLM's response with identity context.

        Raises:
            RuntimeError: If no EMMS instance is attached.
        """
        if self.prompt_builder is None:
            raise RuntimeError(
                "ask() requires an EMMS instance. "
                "Pass emms= to LLMEnhancer constructor."
            )

        if strategy is not None:
            prompt = self.prompt_builder.with_question(question, strategy)
        else:
            model_id = self._get_model_id()
            prompt = self.prompt_builder.auto(question, model=model_id)

        return await self.provider.generate(prompt, max_tokens=max_tokens)

    @staticmethod
    def recommend_model() -> str:
        """Return the recommended model ID for EMMS identity adoption.

        Based on 72-trial empirical study:
        Claude Sonnet 4.5 = 83% net adoption (best).
        """
        return "claude-sonnet-4-5-20250929"

    def _parse_json(self, text: str) -> Any:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        continue
            return None
