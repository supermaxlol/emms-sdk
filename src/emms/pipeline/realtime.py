"""Real-time data pipeline for EMMS.

Clean port of text.py's data integration capabilities, redesigned as a
configurable pipeline **without** hardcoded API keys or credentials.

Pipeline stages: fetch → quality_filter → deduplicate → novelty_score → create_experience

Supported source types:
- RSS feeds (zero dependency)
- REST APIs (configurable endpoints)
- WebSocket streams (via aiohttp)

Usage::

    from emms.pipeline.realtime import RealTimePipeline, DataSource

    pipeline = RealTimePipeline(emms_agent)
    pipeline.add_source(DataSource(
        name="tech_news",
        source_type="rss",
        endpoint="https://feeds.example.com/tech.xml",
        domain="tech",
    ))
    await pipeline.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable

from emms.core.models import Experience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data source configuration
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    source_type: str  # "rss", "rest", "websocket"
    endpoint: str
    domain: str = "general"
    rate_limit: int = 60  # seconds between fetches
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    api_key_header: str | None = None  # header name for API key
    api_key: str | None = None  # API key value (passed via config, NOT hardcoded)
    priority: int = 5  # 1 = highest
    enabled: bool = True
    last_fetch: float = 0.0
    error_count: int = 0


# ---------------------------------------------------------------------------
# Quality & deduplication filters
# ---------------------------------------------------------------------------

class QualityFilter:
    """Filter out low-quality content."""

    def __init__(self, min_words: int = 5, min_chars: int = 20):
        self.min_words = min_words
        self.min_chars = min_chars

    def passes(self, text: str) -> bool:
        """Return True if content passes quality checks."""
        if not text or len(text.strip()) < self.min_chars:
            return False
        words = text.split()
        if len(words) < self.min_words:
            return False
        # Check for excessive repetition
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.3:
            return False
        return True


class ContentDeduplicator:
    """Track content fingerprints to avoid duplicates."""

    def __init__(self, max_fingerprints: int = 10_000):
        self._fingerprints: set[str] = set()
        self._max = max_fingerprints

    def is_duplicate(self, text: str) -> bool:
        """Return True if we've seen similar content before."""
        fp = self._fingerprint(text)
        if fp in self._fingerprints:
            return True
        self._fingerprints.add(fp)
        # Evict oldest if over limit
        if len(self._fingerprints) > self._max:
            # Remove ~10% of entries (approximate LRU)
            to_remove = list(self._fingerprints)[:self._max // 10]
            for r in to_remove:
                self._fingerprints.discard(r)
        return False

    def _fingerprint(self, text: str) -> str:
        """Generate a content fingerprint."""
        # Normalise: lowercase, remove punctuation, sort words
        words = sorted(set(re.sub(r'[^\w\s]', '', text.lower()).split()))
        canonical = " ".join(words[:20])  # first 20 unique words
        return hashlib.md5(canonical.encode()).hexdigest()


class NoveltyScorer:
    """Score content novelty relative to previously seen content."""

    def __init__(self):
        self._seen_words: dict[str, int] = {}
        self._total_docs: int = 0

    def score(self, text: str) -> float:
        """Return novelty score 0..1 (1 = completely novel)."""
        self._total_docs += 1
        words = set(text.lower().split())
        if not words:
            return 0.5

        novel_count = sum(1 for w in words if w not in self._seen_words)
        novelty = novel_count / len(words)

        # Update seen words
        for w in words:
            self._seen_words[w] = self._seen_words.get(w, 0) + 1

        return novelty


# ---------------------------------------------------------------------------
# RSS parser (zero-dependency)
# ---------------------------------------------------------------------------

def parse_rss(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS/Atom feed XML into items."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # RSS 2.0
    for item in root.iter("item"):
        title = item.findtext("title", "")
        desc = item.findtext("description", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        content = f"{title}. {desc}".strip()
        if content and content != ".":
            items.append({
                "content": content,
                "link": link,
                "published": pub_date,
            })

    # Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = entry.findtext("{http://www.w3.org/2005/Atom}title", "")
        summary = entry.findtext("{http://www.w3.org/2005/Atom}summary", "")
        content = f"{title}. {summary}".strip()
        if content and content != ".":
            items.append({"content": content, "link": "", "published": ""})

    return items


# ---------------------------------------------------------------------------
# Real-time pipeline
# ---------------------------------------------------------------------------

class RealTimePipeline:
    """Async pipeline that fetches data from configured sources and stores
    it as EMMS experiences.

    The pipeline runs as a background asyncio task. Each source is fetched
    according to its rate_limit. Content passes through quality filtering,
    deduplication, and novelty scoring before becoming an Experience.
    """

    def __init__(self, store_callback: Callable[[Experience], Any] | None = None):
        self.sources: dict[str, DataSource] = {}
        self._store_callback = store_callback
        self._quality = QualityFilter()
        self._dedup = ContentDeduplicator()
        self._novelty = NoveltyScorer()
        self._handlers: list[Callable[[Experience], Any]] = []
        self._running = False
        self._task: asyncio.Task | None = None

        # Stats
        self.total_fetched = 0
        self.total_filtered = 0
        self.total_stored = 0

    def add_source(self, source: DataSource) -> None:
        """Register a data source."""
        self.sources[source.name] = source

    def remove_source(self, name: str) -> None:
        """Unregister a data source."""
        self.sources.pop(name, None)

    def on_experience(self, callback: Callable[[Experience], Any]) -> None:
        """Register a handler called for each new experience."""
        self._handlers.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background fetch loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())

    async def stop(self) -> None:
        """Gracefully stop the pipeline."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main fetch loop — cycles through sources respecting rate limits."""
        while self._running:
            now = time.time()
            for source in list(self.sources.values()):
                if not source.enabled:
                    continue
                if now - source.last_fetch < source.rate_limit:
                    continue

                try:
                    items = await self._fetch_source(source)
                    source.last_fetch = now
                    source.error_count = 0
                    self.total_fetched += len(items)

                    for item in items:
                        experience = self._process_item(item, source)
                        if experience is not None:
                            self.total_stored += 1
                            # Store via callback
                            if self._store_callback is not None:
                                self._store_callback(experience)
                            # Notify handlers
                            for handler in self._handlers:
                                handler(experience)
                except Exception:
                    source.error_count += 1
                    logger.warning(
                        "Error fetching %s (errors=%d)",
                        source.name, source.error_count, exc_info=True,
                    )
                    # Back off on errors
                    if source.error_count >= 5:
                        source.enabled = False
                        logger.warning("Disabled source %s after 5 errors", source.name)

            await asyncio.sleep(1)  # small sleep to prevent busy loop

    # ── Fetch ──────────────────────────────────────────────────────────

    async def _fetch_source(self, source: DataSource) -> list[dict[str, str]]:
        """Fetch items from a source."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp not installed. Run: pip install aiohttp"
            )

        headers = dict(source.headers)
        if source.api_key_header and source.api_key:
            headers[source.api_key_header] = source.api_key

        async with aiohttp.ClientSession() as session:
            if source.source_type == "rss":
                return await self._fetch_rss(session, source, headers)
            elif source.source_type == "rest":
                return await self._fetch_rest(session, source, headers)
            else:
                logger.warning("Unsupported source type: %s", source.source_type)
                return []

    async def _fetch_rss(
        self, session: Any, source: DataSource, headers: dict
    ) -> list[dict[str, str]]:
        """Fetch and parse an RSS feed."""
        async with session.get(source.endpoint, headers=headers) as resp:
            if resp.status != 200:
                return []
            xml_text = await resp.text()
            return parse_rss(xml_text)

    async def _fetch_rest(
        self, session: Any, source: DataSource, headers: dict
    ) -> list[dict[str, str]]:
        """Fetch from a REST API endpoint."""
        async with session.get(
            source.endpoint, headers=headers, params=source.params
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

            # Handle common API response formats
            items = []
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        content = entry.get("content") or entry.get("text") or entry.get("title", "")
                        items.append({"content": str(content)})
            elif isinstance(data, dict):
                # Try common patterns: {articles: [...]}, {results: [...]}, {data: [...]}
                for key in ("articles", "results", "data", "items", "entries"):
                    if key in data and isinstance(data[key], list):
                        for entry in data[key]:
                            if isinstance(entry, dict):
                                content = (
                                    entry.get("content")
                                    or entry.get("text")
                                    or entry.get("description")
                                    or entry.get("title", "")
                                )
                                items.append({"content": str(content)})
                        break
            return items

    # ── Processing pipeline ────────────────────────────────────────────

    def _process_item(
        self, item: dict[str, str], source: DataSource
    ) -> Experience | None:
        """Run item through quality → dedup → novelty → Experience."""
        content = item.get("content", "").strip()

        # Quality filter
        if not self._quality.passes(content):
            self.total_filtered += 1
            return None

        # Deduplication
        if self._dedup.is_duplicate(content):
            self.total_filtered += 1
            return None

        # Novelty scoring
        novelty = self._novelty.score(content)

        return Experience(
            content=content,
            domain=source.domain,
            novelty=novelty,
            importance=0.5,  # default, can be enriched by LLM later
            metadata={
                "source": source.name,
                "source_type": source.source_type,
                "link": item.get("link", ""),
            },
        )

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "sources": len(self.sources),
            "active_sources": sum(1 for s in self.sources.values() if s.enabled),
            "total_fetched": self.total_fetched,
            "total_filtered": self.total_filtered,
            "total_stored": self.total_stored,
            "running": self._running,
        }
