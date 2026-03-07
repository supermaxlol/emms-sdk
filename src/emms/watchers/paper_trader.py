"""Paper Trader — EMMS makes predictions, tracks a virtual portfolio, learns from outcomes.

This is the embodiment loop:
  1. Observe market data (experiences already stored by MarketDataWatcher)
  2. Make predictions based on recent patterns (momentum + mean-reversion)
  3. Execute paper trades on a $10K virtual portfolio
  4. Score outcomes — store wins/losses with emotional valence
  5. Reflection engine finds what works and what doesn't
  6. Self-model updates trading capability score

No real money. No real orders. Just EMMS learning to feel the market.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path.home() / ".emms" / "trader" / "portfolio.json"
TRADE_LOG_FILE = Path.home() / ".emms" / "trader" / "trade_log.jsonl"

INITIAL_CAPITAL = 10_000.0
MAX_POSITION_PCT = 0.20       # Max 20% per position
SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "AAPL", "MSFT", "SPY", "GLD"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: float
    strategy: str               # "momentum" | "mean_reversion" | "breakout"
    prediction_id: str | None = None

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class Portfolio:
    cash: float = INITIAL_CAPITAL
    positions: list[Position] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    created_at: float = field(default_factory=time.time)

    def position_for(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def market_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value at current prices."""
        pos_value = sum(p.quantity * prices.get(p.symbol, p.entry_price) for p in self.positions)
        return self.cash + pos_value

    def save(self):
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": self.cash,
            "positions": [asdict(p) for p in self.positions],
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "created_at": self.created_at,
            "updated_at": time.time(),
        }
        PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "Portfolio":
        if not PORTFOLIO_FILE.exists():
            return cls()
        try:
            data = json.loads(PORTFOLIO_FILE.read_text())
            p = cls(
                cash=data["cash"],
                total_trades=data.get("total_trades", 0),
                winning_trades=data.get("winning_trades", 0),
                losing_trades=data.get("losing_trades", 0),
                total_pnl=data.get("total_pnl", 0.0),
                created_at=data.get("created_at", time.time()),
            )
            for pos_data in data.get("positions", []):
                p.positions.append(Position(**{k: v for k, v in pos_data.items() if k in Position.__dataclass_fields__}))
            return p
        except Exception as exc:
            logger.warning("Failed to load portfolio: %s", exc)
            return cls()


# ---------------------------------------------------------------------------
# Signal generators (simple rule-based — EMMS learns which works)
# ---------------------------------------------------------------------------

def momentum_signals(prices: dict[str, float], changes: dict[str, float]) -> list[tuple[str, str, str]]:
    """Buy winners, sell losers. Returns [(symbol, side, reason)]."""
    signals = []
    for sym, chg in changes.items():
        if chg > 3.0:
            signals.append((sym, "buy", f"Momentum: {sym} up {chg:+.1f}%, riding trend"))
        elif chg < -3.0:
            signals.append((sym, "sell", f"Momentum: {sym} down {chg:+.1f}%, cutting loss"))
    return signals


def mean_reversion_signals(prices: dict[str, float], changes: dict[str, float]) -> list[tuple[str, str, str]]:
    """Buy oversold, sell overbought. Returns [(symbol, side, reason)]."""
    signals = []
    for sym, chg in changes.items():
        if chg < -4.0:
            signals.append((sym, "buy", f"MeanRevert: {sym} down {chg:+.1f}%, expect bounce"))
        elif chg > 5.0:
            signals.append((sym, "sell", f"MeanRevert: {sym} up {chg:+.1f}%, expect pullback"))
    return signals


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Makes paper trades, tracks portfolio, scores outcomes."""

    def __init__(self):
        self.portfolio = Portfolio.load()

    def run_cycle(self, emms: "EMMS") -> int:
        """One trading cycle: check exits, generate signals, execute, score. Returns actions taken."""
        from emms.watchers.market_data import fetch_prices
        from emms.core.models import Experience

        actions = 0

        # Fetch current prices
        snapshots = fetch_prices(SYMBOLS)
        if not snapshots:
            logger.warning("Paper trader: no price data available")
            return 0

        prices = {s.symbol: s.price for s in snapshots}
        changes = {s.symbol: s.change_pct for s in snapshots}

        # --- 1. Score existing positions (check exits) ---
        positions_to_close = []
        for pos in self.portfolio.positions:
            current = prices.get(pos.symbol)
            if current is None:
                continue

            pnl_pct = ((current - pos.entry_price) / pos.entry_price) * 100
            hold_hours = (time.time() - pos.entry_time) / 3600

            # Exit rules: take profit at +5%, stop loss at -3%, or held > 24h
            should_exit = pnl_pct > 5.0 or pnl_pct < -3.0 or hold_hours > 24

            if should_exit:
                pnl_dollar = (current - pos.entry_price) * pos.quantity
                is_win = pnl_dollar > 0

                self.portfolio.total_trades += 1
                self.portfolio.total_pnl += pnl_dollar
                self.portfolio.cash += pos.quantity * current
                if is_win:
                    self.portfolio.winning_trades += 1

                else:
                    self.portfolio.losing_trades += 1

                # Store outcome as experience with emotional valence
                valence = max(-1.0, min(1.0, pnl_pct / 5.0))
                importance = min(1.0, abs(pnl_pct) / 10.0 + 0.5)
                reason = "TP" if pnl_pct > 5 else "SL" if pnl_pct < -3 else "timeout"

                exp = Experience(
                    content=(
                        f"Paper trade CLOSED: {pos.symbol} {pos.strategy} "
                        f"entry=${pos.entry_price:.2f} exit=${current:.2f} "
                        f"PnL={pnl_dollar:+.2f} ({pnl_pct:+.1f}%) [{reason}] "
                        f"{'WIN' if is_win else 'LOSS'}"
                    ),
                    domain="trading",
                    importance=importance,
                    emotional_valence=valence,
                    obs_type="decision",
                    metadata={
                        "symbol": pos.symbol,
                        "strategy": pos.strategy,
                        "entry_price": pos.entry_price,
                        "exit_price": current,
                        "pnl_dollar": round(pnl_dollar, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "hold_hours": round(hold_hours, 1),
                        "outcome": "win" if is_win else "loss",
                        "exit_reason": reason,
                    },
                    namespace="trading",
                    session_id="paper-trader",
                )
                try:
                    emms.store(exp)
                    actions += 1
                except Exception as exc:
                    logger.warning("Failed to store trade outcome: %s", exc)

                # Log trade
                self._log_trade(pos, current, pnl_dollar, pnl_pct, reason)
                positions_to_close.append(pos)

        for pos in positions_to_close:
            self.portfolio.positions.remove(pos)

        # --- 2. Generate new signals ---
        all_signals = []
        all_signals.extend(momentum_signals(prices, changes))
        all_signals.extend(mean_reversion_signals(prices, changes))

        # --- 3. Execute paper trades ---
        for symbol, side, reason in all_signals:
            if side == "buy":
                # Skip if already holding this symbol
                if self.portfolio.position_for(symbol):
                    continue

                # Position sizing
                price = prices.get(symbol, 0)
                if not price:
                    continue
                max_spend = self.portfolio.cash * MAX_POSITION_PCT
                if max_spend < 10:
                    continue
                quantity = max_spend / price

                # Execute paper buy
                self.portfolio.cash -= quantity * price
                strategy = "momentum" if "Momentum" in reason else "mean_reversion"
                pos = Position(
                    symbol=symbol,
                    quantity=round(quantity, 6),
                    entry_price=price,
                    entry_time=time.time(),
                    strategy=strategy,
                )
                self.portfolio.positions.append(pos)

                # Store as experience
                exp = Experience(
                    content=f"Paper trade OPENED: BUY {symbol} qty={quantity:.4f} at ${price:.2f} — {reason}",
                    domain="trading",
                    importance=0.7,
                    emotional_valence=0.2,
                    obs_type="decision",
                    metadata={
                        "symbol": symbol,
                        "side": "buy",
                        "price": price,
                        "quantity": round(quantity, 6),
                        "strategy": strategy,
                        "reason": reason,
                    },
                    namespace="trading",
                    session_id="paper-trader",
                )
                try:
                    emms.store(exp)
                    actions += 1
                except Exception as exc:
                    logger.warning("Failed to store trade entry: %s", exc)

        # --- 4. Portfolio snapshot ---
        total_value = self.portfolio.market_value(prices)
        pnl_total_pct = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        win_rate = (
            self.portfolio.winning_trades / self.portfolio.total_trades * 100
            if self.portfolio.total_trades > 0 else 0
        )

        if self.portfolio.total_trades > 0 and actions > 0:
            exp = Experience(
                content=(
                    f"Portfolio snapshot: ${total_value:.2f} ({pnl_total_pct:+.1f}%) "
                    f"| {len(self.portfolio.positions)} positions "
                    f"| {self.portfolio.total_trades} trades (W/L: {self.portfolio.winning_trades}/{self.portfolio.losing_trades}, {win_rate:.0f}% win rate) "
                    f"| Total PnL: ${self.portfolio.total_pnl:+.2f}"
                ),
                domain="trading",
                importance=0.6,
                emotional_valence=max(-1.0, min(1.0, pnl_total_pct / 10.0)),
                obs_type="discovery",
                namespace="trading",
                session_id="paper-trader",
            )
            try:
                emms.store(exp)
            except Exception:
                pass

        self.portfolio.save()
        return actions

    def _log_trade(self, pos: Position, exit_price: float, pnl: float, pnl_pct: float, reason: str):
        """Append trade to JSONL log."""
        TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "hold_hours": round((time.time() - pos.entry_time) / 3600, 1),
            "reason": reason,
        }
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
