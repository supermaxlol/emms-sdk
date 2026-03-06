"""EmbodimentDomain — base class for domain-specific consciousness streams.

An embodied consciousness is grounded in a specific domain of experience.
The financial trading bot, a scientific instrument, a creative writing process —
each is a sensory stream that can feed lived experience into EMMS.

Subclasses implement three abstract methods:
- ``sense()`` — read current state as a list of sensory events
- ``to_experience()`` — convert a sensory event to storable text content
- ``emotional_valence()`` — map a sensory event to an emotional valence (-1 to 1)

The ``embody()`` convenience method runs the full sense → store cycle.

Usage::

    from emms.embodiment.financial import FinancialEmbodiment

    domain = FinancialEmbodiment(emms, bot_state_path="/path/to/bot_state.json")
    memory_ids = domain.embody()   # sense current market state → store as memories
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


class EmbodimentDomain(ABC):
    """Abstract base class for all domain-specific embodiment streams.

    Parameters
    ----------
    emms:
        Live EMMS instance to store experiences into.
    domain_name:
        The EMMS domain label for memories created by this embodiment.
    namespace:
        EMMS namespace to store memories under (default: "self").
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        domain_name: str = "embodiment",
        namespace: str = "self",
    ) -> None:
        self.emms = emms
        self.domain_name = domain_name
        self.namespace = namespace

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def sense(self) -> list[dict[str, Any]]:
        """Read the current state of the domain.

        Returns a list of raw sensory event dicts.  Each dict must contain
        at least a ``content`` key with human-readable text.
        """
        ...

    @abstractmethod
    def to_content(self, sensory_event: dict[str, Any]) -> str:
        """Convert a sensory event to storable text content.

        Parameters
        ----------
        sensory_event:
            A dict returned by ``sense()``.

        Returns
        -------
        str
            Text to store in EMMS.
        """
        ...

    @abstractmethod
    def emotional_valence(self, sensory_event: dict[str, Any]) -> float:
        """Map a sensory event to emotional valence.

        Parameters
        ----------
        sensory_event:
            A dict returned by ``sense()``.

        Returns
        -------
        float
            Valence in [-1, 1].  Negative = aversive, positive = rewarding.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overridable hooks
    # ------------------------------------------------------------------

    def importance(self, sensory_event: dict[str, Any]) -> float:
        """Return importance score (0-1) for this event.  Override if needed."""
        return 0.5

    def concept_tags(self, sensory_event: dict[str, Any]) -> list[str]:
        """Return concept tags for this event.  Override if needed."""
        return [self.domain_name]

    def title(self, sensory_event: dict[str, Any]) -> str | None:
        """Return a title for the memory.  Override if needed."""
        return None

    # ------------------------------------------------------------------
    # Main embody cycle
    # ------------------------------------------------------------------

    def embody(self) -> list[str]:
        """Full sense → store cycle.

        Calls ``sense()``, converts each event to an EMMS experience, and
        stores it.  Returns a list of stored memory IDs.
        """
        try:
            events = self.sense()
        except Exception as exc:
            logger.error("[%s] sense() failed: %s", self.domain_name, exc)
            return []

        memory_ids: list[str] = []
        for event in events:
            try:
                content = self.to_content(event)
                valence = self.emotional_valence(event)
                imp = self.importance(event)
                tags = self.concept_tags(event)
                ttl = self.title(event)

                result = self.emms.store(
                    content=content,
                    domain=self.domain_name,
                    namespace=self.namespace,
                    obs_type="observation",
                    emotional_valence=valence,
                    importance=imp,
                    concept_tags=tags,
                    title=ttl,
                )
                mid = result.get("memory_id") if isinstance(result, dict) else getattr(result, "memory_id", None)
                if mid:
                    memory_ids.append(mid)
                    logger.debug("[%s] stored memory %s (valence=%.2f)", self.domain_name, mid, valence)
            except Exception as exc:
                logger.warning("[%s] failed to store event: %s", self.domain_name, exc)

        return memory_ids
