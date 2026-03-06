"""FinancialEmbodiment — market data as a sensory stream for EMMS.

Market moves become sensory events.
P&L changes become emotional valence (profit=positive, loss=negative).
Volatility becomes physical intensity (calm, turbulent, overwhelming).
Trade decisions become autobiographical memories.

The trading bot's history becomes the system's lived financial experience.

Usage::

    from emms.embodiment.financial import FinancialEmbodiment

    fe = FinancialEmbodiment(
        emms,
        bot_state_path="/Users/shehzad/bot_state.json",
    )
    memory_ids = fe.embody()   # reads current bot state → stores as memory

Or to embody a historical trade::

    trade_event = fe.parse_trade({
        "side": "SELL", "qty": 100, "price": 0.52,
        "pnl_usdt": 12.4, "symbol": "XRPUSDT",
    })
    memory_ids = fe.store_event(trade_event)
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import TYPE_CHECKING, Any

from emms.embodiment.base import EmbodimentDomain

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)

# Mapping from volatility level to text
_VOLATILITY_FEEL = {
    "calm":              "calm waters",
    "mild_movement":     "gentle ripples",
    "moderate_tension":  "a building swell",
    "intense_turbulence":"turbulent seas",
    "extreme_volatility":"a storm",
}


class FinancialEmbodiment(EmbodimentDomain):
    """Connects trading bot state to EMMS as lived financial experience.

    Parameters
    ----------
    emms:
        Live EMMS instance.
    bot_state_path:
        Path to the bot's JSON state file (e.g. ``bot_state.json``).
    symbol:
        Trading symbol for context labels (e.g. "XRPUSDT").
    pnl_scale:
        Scale factor: how many USDT of P&L maps to valence=1.0 (default: 50.0).
    """

    def __init__(
        self,
        emms: "EMMS",
        bot_state_path: str | None = None,
        *,
        symbol: str = "XRPUSDT",
        pnl_scale: float = 50.0,
    ) -> None:
        super().__init__(emms, domain_name="financial_market", namespace="self")
        self.bot_state_path = bot_state_path
        self.symbol = symbol
        self.pnl_scale = pnl_scale

    # ------------------------------------------------------------------
    # EmbodimentDomain interface
    # ------------------------------------------------------------------

    def sense(self) -> list[dict[str, Any]]:
        """Read the current bot state file as a sensory snapshot."""
        if not self.bot_state_path or not os.path.exists(self.bot_state_path):
            logger.debug("FinancialEmbodiment: no bot state at %s", self.bot_state_path)
            return []
        try:
            with open(self.bot_state_path) as f:
                state = json.load(f)
            return [self._parse_bot_state(state)]
        except Exception as exc:
            logger.warning("FinancialEmbodiment.sense: %s", exc)
            return []

    def to_content(self, event: dict[str, Any]) -> str:
        """Convert a financial sensory event to human-readable memory text."""
        symbol = event.get("symbol", self.symbol)
        price = event.get("price", 0.0)
        pnl = event.get("pnl", 0.0)
        vol_feel = _VOLATILITY_FEEL.get(event.get("volatility_feel", "calm"), "calm waters")
        side = event.get("side", "")
        qty = event.get("qty", 0)
        event_type = event.get("type", "market_snapshot")

        if event_type == "trade":
            direction = "bought" if side == "BUY" else "sold"
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            return (
                f"I {direction} {qty} {symbol} at ${price:.4f}. "
                f"Realized P&L: {pnl_str}. "
                f"The market felt like {vol_feel}."
            )
        else:
            position = event.get("position", "flat")
            equity = event.get("equity", 0.0)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            return (
                f"{symbol} market snapshot: price=${price:.4f}, "
                f"position={position}, unrealized_pnl={pnl_str}, "
                f"equity=${equity:.2f}. "
                f"Felt: {vol_feel}."
            )

    def emotional_valence(self, event: dict[str, Any]) -> float:
        """P&L maps directly to emotional valence, clamped to [-1, 1]."""
        pnl = event.get("pnl", 0.0)
        raw = pnl / max(self.pnl_scale, 1.0)
        return round(max(-1.0, min(1.0, raw)), 4)

    def importance(self, event: dict[str, Any]) -> float:
        """Trades are more important than snapshots; large moves are more important."""
        if event.get("type") == "trade":
            pnl = abs(event.get("pnl", 0.0))
            # Importance rises with |P&L| — max at pnl_scale/2
            base = 0.5 + 0.4 * math.tanh(pnl / (self.pnl_scale / 2))
            return round(min(base, 1.0), 3)
        # Snapshot: importance driven by volatility
        vol_map = {
            "calm": 0.2,
            "mild_movement": 0.3,
            "moderate_tension": 0.5,
            "intense_turbulence": 0.7,
            "extreme_volatility": 0.9,
        }
        return vol_map.get(event.get("volatility_feel", "calm"), 0.3)

    def concept_tags(self, event: dict[str, Any]) -> list[str]:
        tags = ["market", "trading", event.get("volatility_feel", "calm")]
        if event.get("type") == "trade":
            tags.append(event.get("side", "unknown").lower())
        return tags

    def title(self, event: dict[str, Any]) -> str | None:
        symbol = event.get("symbol", self.symbol)
        if event.get("type") == "trade":
            side = event.get("side", "?")
            pnl = event.get("pnl", 0.0)
            sign = "+" if pnl >= 0 else ""
            return f"{symbol} {side} trade (P&L {sign}{pnl:.2f})"
        return f"{symbol} market snapshot"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def parse_trade(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw trade dict (from SQLite/Binance) to an embodiment event.

        Parameters
        ----------
        raw:
            Dict with keys like ``side``, ``qty``, ``price``, ``pnl_usdt``, ``symbol``.
        """
        pnl = float(raw.get("pnl_usdt", raw.get("pnl", 0.0)))
        price = float(raw.get("price", raw.get("fill_price", 0.0)))
        qty = float(raw.get("qty", raw.get("quantity", 0.0)))
        volatility_feel = self._classify_volatility(raw.get("volatility_pct"))

        return {
            "type": "trade",
            "symbol": raw.get("symbol", self.symbol),
            "side": raw.get("side", "UNKNOWN").upper(),
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "volatility_feel": volatility_feel,
            "timestamp": raw.get("timestamp", raw.get("created_at")),
        }

    def store_event(self, event: dict[str, Any]) -> list[str]:
        """Store a single pre-parsed embodiment event. Returns list of memory IDs."""
        return self.embody.__wrapped__(self, [event]) if hasattr(self.embody, "__wrapped__") else self._store_one(event)

    def _store_one(self, event: dict[str, Any]) -> list[str]:
        """Internal: store a single event dict via the base embody loop."""
        content = self.to_content(event)
        valence = self.emotional_valence(event)
        imp = self.importance(event)
        tags = self.concept_tags(event)
        ttl = self.title(event)
        try:
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
            return [mid] if mid else []
        except Exception as exc:
            logger.warning("FinancialEmbodiment._store_one: %s", exc)
            return []

    def _parse_bot_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Parse a bot_state.json into a sensory event dict."""
        price = float(state.get("current_price", state.get("price", 0.0)))
        pnl = float(state.get("unrealized_pnl", state.get("pnl", 0.0)))
        equity = float(state.get("equity", state.get("account_value", 0.0)))
        symbol = state.get("symbol", self.symbol)
        position = state.get("position", state.get("side", "flat"))
        volatility_feel = self._classify_volatility(state.get("volatility_pct"))

        return {
            "type": "market_snapshot",
            "symbol": symbol,
            "price": price,
            "pnl": pnl,
            "equity": equity,
            "position": position,
            "volatility_feel": volatility_feel,
        }

    @staticmethod
    def _classify_volatility(vol_pct: Any) -> str:
        """Map a volatility percentage to a qualitative feel."""
        if vol_pct is None:
            return "calm"
        v = float(vol_pct)
        if v < 0.5:
            return "calm"
        if v < 1.5:
            return "mild_movement"
        if v < 3.0:
            return "moderate_tension"
        if v < 6.0:
            return "intense_turbulence"
        return "extreme_volatility"
