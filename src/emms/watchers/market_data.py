"""Market Data Watcher — ingests Yahoo Finance prices + RSS news into EMMS.

Polls every N minutes. Stores price moves as experiences with emotional
valence proportional to magnitude. News headlines stored as observations.

No API keys needed — uses free yfinance + public RSS feeds.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]
EQUITY_SYMBOLS = ["AAPL", "MSFT", "SPY", "GLD"]
ALL_SYMBOLS = CRYPTO_SYMBOLS + EQUITY_SYMBOLS

NEWS_FEEDS = [
    ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "yahoo-sp500"),
    ("https://www.coindesk.com/arc/outboundfeeds/rss/", "coindesk"),
]

# State file for tracking what we've already seen
MARKET_STATE_FILE = Path.home() / ".emms" / "trader" / "market_state.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PriceSnapshot:
    symbol: str
    price: float
    prev_close: float
    change_pct: float        # daily % change
    volume: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_significant(self) -> bool:
        """Move > 1.5% is worth storing."""
        return abs(self.change_pct) > 1.5


@dataclass
class NewsItem:
    title: str
    source: str
    link: str
    published: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_prices(symbols: list[str] | None = None) -> list[PriceSnapshot]:
    """Fetch current prices from Yahoo Finance."""
    symbols = symbols or ALL_SYMBOLS
    snapshots = []
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                t = tickers.tickers.get(sym)
                if t is None:
                    continue
                info = t.fast_info
                price = getattr(info, "last_price", None) or 0
                prev = getattr(info, "previous_close", None) or price
                if price and prev:
                    change_pct = ((price - prev) / prev) * 100
                    snapshots.append(PriceSnapshot(
                        symbol=sym,
                        price=round(price, 4),
                        prev_close=round(prev, 4),
                        change_pct=round(change_pct, 2),
                        volume=getattr(info, "last_volume", 0) or 0,
                    ))
            except Exception as exc:
                logger.debug("Price fetch failed for %s: %s", sym, exc)
    except Exception as exc:
        logger.warning("yfinance batch fetch failed: %s", exc)
    return snapshots


def fetch_news(feeds: list[tuple[str, str]] | None = None, max_per_feed: int = 5) -> list[NewsItem]:
    """Fetch latest headlines from RSS feeds."""
    feeds = feeds or NEWS_FEEDS
    items = []
    try:
        import feedparser
        for url, source in feeds:
            try:
                d = feedparser.parse(url)
                for entry in d.entries[:max_per_feed]:
                    items.append(NewsItem(
                        title=entry.get("title", "")[:200],
                        source=source,
                        link=entry.get("link", ""),
                        published=entry.get("published", ""),
                    ))
            except Exception as exc:
                logger.debug("RSS fetch failed for %s: %s", source, exc)
    except ImportError:
        logger.warning("feedparser not installed — skipping news")
    return items


# ---------------------------------------------------------------------------
# MarketDataWatcher
# ---------------------------------------------------------------------------

class MarketDataWatcher:
    """Polls market data and news, converts to EMMS experiences."""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols or ALL_SYMBOLS
        self._last_prices: dict[str, float] = {}
        self._seen_headlines: set[str] = set()
        self._load_state()

    def _load_state(self):
        if MARKET_STATE_FILE.exists():
            try:
                data = json.loads(MARKET_STATE_FILE.read_text())
                self._last_prices = data.get("last_prices", {})
                self._seen_headlines = set(data.get("seen_headlines", []))
            except Exception:
                pass

    def _save_state(self):
        MARKET_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Keep only last 500 headlines to prevent unbounded growth
        recent = list(self._seen_headlines)[-500:]
        MARKET_STATE_FILE.write_text(json.dumps({
            "last_prices": self._last_prices,
            "seen_headlines": recent,
            "updated_at": time.time(),
        }))

    def poll(self, emms: "EMMS") -> int:
        """Fetch data, store as experiences. Returns count stored."""
        from emms.core.models import Experience
        stored = 0

        # --- Prices ---
        snapshots = fetch_prices(self.symbols)
        for snap in snapshots:
            # Always update tracking
            self._last_prices[snap.symbol] = snap.price

            # Only store significant moves
            if not snap.is_significant:
                continue

            # Emotional valence: big up = positive, big down = negative
            valence = max(-1.0, min(1.0, snap.change_pct / 5.0))
            importance = min(1.0, abs(snap.change_pct) / 10.0 + 0.3)

            exp = Experience(
                content=f"{snap.symbol}: ${snap.price:.2f} ({snap.change_pct:+.1f}%) vol={snap.volume:.0f}",
                domain="market",
                importance=importance,
                emotional_valence=valence,
                obs_type="discovery",
                metadata={
                    "symbol": snap.symbol,
                    "price": snap.price,
                    "prev_close": snap.prev_close,
                    "change_pct": snap.change_pct,
                    "volume": snap.volume,
                },
                namespace="trading",
                session_id="market-watcher",
            )
            try:
                emms.store(exp)
                stored += 1
            except Exception as exc:
                logger.warning("Failed to store price: %s", exc)

        # --- News ---
        news = fetch_news()
        for item in news:
            # Deduplicate by title
            key = item.title.lower().strip()
            if key in self._seen_headlines:
                continue
            self._seen_headlines.add(key)

            exp = Experience(
                content=f"[{item.source}] {item.title}",
                domain="news",
                importance=0.4,
                emotional_valence=0.0,  # neutral — no sentiment analysis yet
                obs_type="discovery",
                metadata={
                    "source": item.source,
                    "link": item.link,
                    "published": item.published,
                },
                namespace="trading",
                session_id="market-watcher",
            )
            try:
                emms.store(exp)
                stored += 1
            except Exception as exc:
                logger.warning("Failed to store news: %s", exc)

        self._save_state()

        if stored:
            logger.info("Market watcher: stored %d items (%d prices, %d news)",
                        stored, len([s for s in snapshots if s.is_significant]), len(news))
        return stored
