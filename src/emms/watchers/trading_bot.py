"""Trading Bot Watcher — passive log observer for Margin-bot / ETrade-bot.

Tails trading bot log files and bot_state.json, parses events into EMMS
Experience objects. STRICTLY READ-ONLY: never writes to bot directories.

Designed to be registered as a scheduler job in the ConsciousnessDaemon.

Events captured:
    - Trade signals (BUY/SELL)        → importance=0.9, domain="trading"
    - Trend changes (up→down, down→up) → importance=0.7
    - Balance changes (>1% delta)     → importance=0.6
    - Bot start/restart               → importance=0.4
    - Price milestones                → importance=0.3
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed event
# ---------------------------------------------------------------------------

@dataclass
class BotEvent:
    """A parsed event from a trading bot log."""
    timestamp: str
    event_type: str          # signal, trend_change, balance, bot_start, price
    content: str             # Human-readable description
    importance: float = 0.5
    emotional_valence: float = 0.0  # -1 (loss) to +1 (profit)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Log patterns
# ---------------------------------------------------------------------------

# Margin-bot patterns (DirectionalChangeBot / DCStrategy)
RE_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)")
RE_SIGNAL = re.compile(r"Signal:\s*(BUY|SELL|HOLD)", re.IGNORECASE)
RE_TREND = re.compile(r"Trend(?:\s+analysis)?:.*is_uptrend['\"]?[=:]\s*(True|False)", re.IGNORECASE)
RE_TREND_INIT = re.compile(r"Initial trend:\s*(Up|Down)", re.IGNORECASE)
RE_TREND_LABEL = re.compile(r"(UPTREND|DOWNTREND)\s*-\s*Current:\s*([\d.]+)", re.IGNORECASE)
RE_BUY_TRIGGER = re.compile(r"Buy trigger at:\s*([\d.]+)\s*\(([\d.]+)% away\)", re.IGNORECASE)
RE_SELL_TRIGGER = re.compile(r"Sell trigger at:\s*([\d.]+)", re.IGNORECASE)
RE_BALANCE = re.compile(r"Current Amount:\s*([\d.]+)", re.IGNORECASE)
RE_BALANCE_SYNC = re.compile(r"balance mismatch.*Bot thinks:\s*([\d.]+),\s*Exchange has:\s*([\d.]+)", re.IGNORECASE)
RE_BOT_START = re.compile(r"Trading Bot Started|Starting trading bot", re.IGNORECASE)
RE_PRICE = re.compile(r"Price:\s*([\d.]+)")
RE_PNL = re.compile(r"Total PnL:\s*([\-\d.]+)\s*USDT", re.IGNORECASE)
RE_TRADE_EXEC = re.compile(r"(BUY|SELL)\s+ORDER.*?(?:price|at)\s*([\d.]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# TradingBotWatcher
# ---------------------------------------------------------------------------

class TradingBotWatcher:
    """Passive observer that tails trading bot logs.

    Parameters
    ----------
    bot_name:
        Human-readable name (e.g., "margin-bot", "etrade-bot").
    log_paths:
        List of log file paths to watch (reads from end).
    state_path:
        Path to bot_state.json (polled for state changes).
    """

    def __init__(
        self,
        bot_name: str,
        log_paths: list[Path],
        state_path: Path | None = None,
    ) -> None:
        self.bot_name = bot_name
        self.log_paths = [Path(p) for p in log_paths]
        self.state_path = Path(state_path) if state_path else None

        # Track file positions for tailing
        self._offsets: dict[str, int] = {}
        # Track last known state for change detection
        self._last_state: dict | None = None
        self._last_trend: str | None = None
        self._last_balance: float | None = None

    def poll(self) -> list[BotEvent]:
        """Read new log lines + state changes. Returns list of events."""
        events: list[BotEvent] = []

        # Tail each log file
        for path in self.log_paths:
            events.extend(self._tail_log(path))

        # Check state file for changes
        if self.state_path:
            events.extend(self._check_state())

        return events

    def _tail_log(self, path: Path) -> list[BotEvent]:
        """Read new lines from a log file since last poll."""
        events: list[BotEvent] = []
        key = str(path)

        if not path.exists():
            return events

        try:
            size = path.stat().st_size
        except OSError:
            return events

        # First time: start from near the end (last 4KB)
        if key not in self._offsets:
            self._offsets[key] = max(0, size - 4096)

        # No new data
        if size <= self._offsets[key]:
            return events

        try:
            with open(path, "r", errors="replace") as f:
                f.seek(self._offsets[key])
                new_lines = f.readlines()
                self._offsets[key] = f.tell()
        except OSError:
            return events

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            event = self._parse_line(line)
            if event:
                events.append(event)

        return events

    def _parse_line(self, line: str) -> BotEvent | None:
        """Parse a single log line into a BotEvent or None."""

        # Extract timestamp
        ts_match = RE_TIMESTAMP.match(line)
        ts = ts_match.group(1) if ts_match else time.strftime("%Y-%m-%d %H:%M:%S")

        # Trade execution (highest priority)
        m = RE_TRADE_EXEC.search(line)
        if m:
            side, price = m.group(1).upper(), m.group(2)
            valence = 0.3 if side == "BUY" else -0.1  # Buy = opportunity, Sell = realization
            return BotEvent(
                timestamp=ts,
                event_type="signal",
                content=f"[{self.bot_name}] {side} executed at {price}",
                importance=0.9,
                emotional_valence=valence,
                metadata={"side": side, "price": float(price), "bot": self.bot_name},
            )

        # Signal detection
        m = RE_SIGNAL.search(line)
        if m and m.group(1).upper() != "HOLD":
            signal = m.group(1).upper()
            return BotEvent(
                timestamp=ts,
                event_type="signal",
                content=f"[{self.bot_name}] Signal: {signal}",
                importance=0.85,
                emotional_valence=0.2 if signal == "BUY" else -0.1,
                metadata={"signal": signal, "bot": self.bot_name},
            )

        # Trend change detection
        m = RE_TREND_LABEL.search(line)
        if m:
            trend = m.group(1).upper()
            price = m.group(2)
            if self._last_trend and trend != self._last_trend:
                self._last_trend = trend
                return BotEvent(
                    timestamp=ts,
                    event_type="trend_change",
                    content=f"[{self.bot_name}] Trend changed to {trend} at {price}",
                    importance=0.7,
                    emotional_valence=0.3 if trend == "UPTREND" else -0.3,
                    metadata={"trend": trend, "price": float(price), "bot": self.bot_name},
                )
            self._last_trend = trend

        # PnL report
        m = RE_PNL.search(line)
        if m:
            pnl = float(m.group(1))
            if abs(pnl) > 0.01:
                return BotEvent(
                    timestamp=ts,
                    event_type="balance",
                    content=f"[{self.bot_name}] PnL: {pnl:+.2f} USDT",
                    importance=0.7,
                    emotional_valence=min(1.0, max(-1.0, pnl / 10.0)),
                    metadata={"pnl": pnl, "bot": self.bot_name},
                )

        # Balance sync mismatch
        m = RE_BALANCE_SYNC.search(line)
        if m:
            bot_bal, exchange_bal = float(m.group(1)), float(m.group(2))
            delta_pct = abs(exchange_bal - bot_bal) / max(bot_bal, 0.01) * 100
            if delta_pct > 1.0:
                return BotEvent(
                    timestamp=ts,
                    event_type="balance",
                    content=f"[{self.bot_name}] Balance sync: bot={bot_bal:.2f}, exchange={exchange_bal:.2f} ({delta_pct:.1f}% drift)",
                    importance=0.6,
                    emotional_valence=0.1 if exchange_bal > bot_bal else -0.2,
                    metadata={"bot_balance": bot_bal, "exchange_balance": exchange_bal, "bot": self.bot_name},
                )

        # Bot start/restart
        if RE_BOT_START.search(line):
            return BotEvent(
                timestamp=ts,
                event_type="bot_start",
                content=f"[{self.bot_name}] Bot started/restarted",
                importance=0.4,
                emotional_valence=0.0,
                metadata={"bot": self.bot_name},
            )

        return None

    def _check_state(self) -> list[BotEvent]:
        """Check bot_state.json for meaningful changes."""
        events: list[BotEvent] = []
        if not self.state_path or not self.state_path.exists():
            return events

        try:
            state = json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return events

        ts = state.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        if self._last_state is not None:
            old_pos = self._last_state.get("position")
            new_pos = state.get("position")

            # Position opened or closed
            if old_pos != new_pos:
                if new_pos and not old_pos:
                    entry = state.get("entry_price", "?")
                    events.append(BotEvent(
                        timestamp=ts,
                        event_type="signal",
                        content=f"[{self.bot_name}] Position OPENED: {new_pos} at {entry}",
                        importance=0.9,
                        emotional_valence=0.4,
                        metadata={"position": new_pos, "entry_price": entry, "bot": self.bot_name},
                    ))
                elif old_pos and not new_pos:
                    events.append(BotEvent(
                        timestamp=ts,
                        event_type="signal",
                        content=f"[{self.bot_name}] Position CLOSED (was {old_pos})",
                        importance=0.9,
                        emotional_valence=-0.1,
                        metadata={"closed_position": old_pos, "bot": self.bot_name},
                    ))

            # Significant balance change (>2%)
            old_amt = self._last_state.get("current_amount", 0)
            new_amt = state.get("current_amount", 0)
            if old_amt and new_amt:
                delta_pct = (new_amt - old_amt) / old_amt * 100
                if abs(delta_pct) > 2.0:
                    events.append(BotEvent(
                        timestamp=ts,
                        event_type="balance",
                        content=f"[{self.bot_name}] Balance: {old_amt:.2f} → {new_amt:.2f} ({delta_pct:+.1f}%)",
                        importance=0.7,
                        emotional_valence=min(1.0, max(-1.0, delta_pct / 10.0)),
                        metadata={"old_balance": old_amt, "new_balance": new_amt, "delta_pct": delta_pct, "bot": self.bot_name},
                    ))

        self._last_state = state
        return events


# ---------------------------------------------------------------------------
# Convert BotEvent → Experience and store
# ---------------------------------------------------------------------------

def ingest_events(emms: "EMMS", events: list[BotEvent]) -> int:
    """Store bot events as EMMS experiences. Returns count stored."""
    from emms.core.models import Experience

    stored = 0
    for event in events:
        exp = Experience(
            content=event.content,
            domain="trading",
            importance=event.importance,
            emotional_valence=event.emotional_valence,
            obs_type="discovery" if event.event_type != "signal" else "decision",
            metadata=event.metadata,
            namespace="trading",
            session_id="bot-watcher",
        )
        try:
            emms.store(exp)
            stored += 1
        except Exception as exc:
            logger.warning("Failed to store bot event: %s", exc)

    if stored:
        logger.info("Bot watcher ingested %d events", stored)
    return stored


# ---------------------------------------------------------------------------
# Pre-configured watchers for known bots
# ---------------------------------------------------------------------------

def create_margin_bot_watcher() -> TradingBotWatcher:
    """Create watcher for Margin-bot (XRPUSDT, Binance)."""
    bot_dir = Path.home() / "Desktop/XRP/Margin-bot"
    return TradingBotWatcher(
        bot_name="margin-bot",
        log_paths=[
            bot_dir / "trading_bot.log",
            bot_dir / "dc_strategy.log",
        ],
        state_path=bot_dir / "bot_state.json",
    )


def create_etrade_bot_watcher() -> TradingBotWatcher:
    """Create watcher for ETrade-bot (US equities, Polygon)."""
    bot_dir = Path.home() / "Desktop/XRP/ETrade-bot"
    return TradingBotWatcher(
        bot_name="etrade-bot",
        log_paths=[
            bot_dir / "trading_bot.log",
        ],
        state_path=bot_dir / "bot_state.json",
    )
