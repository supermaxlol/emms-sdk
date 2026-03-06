#!/usr/bin/env python
"""Backfill EMMS with historical trading bot data.

Converts the SQLite trade log into EMMS autobiographical memories.
Profitable trades get positive emotional valence; losses get negative.
Each trade becomes a lived memory with timestamp, symbol, P&L, and context.

Usage::

    python scripts/backfill_market_embodiment.py \\
        --db /path/to/crypto_trading.db \\
        --state /Users/shehzad/.emms/emms_state.json \\
        --dry-run    # preview without storing

    python scripts/backfill_market_embodiment.py \\
        --db /Users/shehzad/crypto_trading.db \\
        --state /Users/shehzad/.emms/emms_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

# Ensure src is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill EMMS with historical trade data")
    p.add_argument("--db", required=True, help="Path to SQLite trading database")
    p.add_argument("--state", default=os.path.expanduser("~/.emms/emms_state.json"),
                   help="EMMS state file path")
    p.add_argument("--table", default="trade_log",
                   help="SQLite table name (default: trade_log)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max trades to backfill (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without storing")
    p.add_argument("--symbol", default="XRPUSDT",
                   help="Default trading symbol if not in DB")
    p.add_argument("--pnl-scale", type=float, default=50.0,
                   help="USDT P&L that maps to valence=1.0 (default 50)")
    return p.parse_args()


def load_trades(db_path: str, table: str, limit: int) -> list[dict]:
    """Load trade rows from SQLite."""
    if not os.path.exists(db_path):
        print(f"ERROR: database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Inspect available columns
    cur.execute(f"PRAGMA table_info({table})")
    columns = {row["name"] for row in cur.fetchall()}

    # Build a flexible query — handle different schema variants
    select_cols = ["*"]
    order_by = ""
    if "created_at" in columns:
        order_by = "ORDER BY created_at ASC"
    elif "timestamp" in columns:
        order_by = "ORDER BY timestamp ASC"

    limit_clause = f"LIMIT {limit}" if limit > 0 else ""
    query = f"SELECT {','.join(select_cols)} FROM {table} {order_by} {limit_clause}"

    cur.execute(query)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def row_to_event(row: dict, default_symbol: str, pnl_scale: float) -> dict:
    """Normalise a trade row to a FinancialEmbodiment event dict."""
    pnl = float(row.get("pnl_usdt", row.get("pnl", row.get("realized_pnl", 0.0))) or 0.0)
    price = float(row.get("price", row.get("fill_price", row.get("avg_price", 0.0))) or 0.0)
    qty = float(row.get("qty", row.get("quantity", row.get("amount", 0.0))) or 0.0)
    side = str(row.get("side", row.get("direction", "UNKNOWN"))).upper()
    symbol = str(row.get("symbol", row.get("pair", default_symbol)))

    # Timestamp — prefer created_at, fall back to timestamp column or now
    ts_raw = row.get("created_at") or row.get("timestamp") or row.get("time")
    if isinstance(ts_raw, (int, float)):
        ts = float(ts_raw)
    elif isinstance(ts_raw, str):
        try:
            import datetime
            ts = datetime.datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = time.time()
    else:
        ts = time.time()

    return {
        "type": "trade",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "pnl": pnl,
        "volatility_feel": "calm",  # historical trades don't have live volatility
        "timestamp": ts,
    }


def main() -> None:
    args = parse_args()

    print(f"Loading trades from: {args.db}")
    trades = load_trades(args.db, args.table, args.limit)
    print(f"Found {len(trades)} trades")

    if not trades:
        print("No trades to backfill. Exiting.")
        return

    if args.dry_run:
        print("\n[DRY RUN] First 5 events:")
        from emms.embodiment.financial import FinancialEmbodiment
        fe_dummy = type("FE", (), {
            "to_content": lambda self, e: FinancialEmbodiment.to_content(None, e),  # type: ignore
            "emotional_valence": lambda self, e: FinancialEmbodiment.emotional_valence(None, e),  # type: ignore
        })()
        for i, row in enumerate(trades[:5]):
            event = row_to_event(row, args.symbol, args.pnl_scale)
            print(f"\n  Trade {i+1}:")
            print(f"    {event}")
        print("\n[DRY RUN] No memories stored.")
        return

    # Real backfill
    from emms import EMMS
    from emms.embodiment.financial import FinancialEmbodiment

    emms = EMMS()
    state_path = args.state
    if os.path.exists(state_path):
        emms.load(state_path)
        print(f"Loaded existing EMMS state from {state_path}")
    else:
        print(f"Starting fresh (no state at {state_path})")

    fe = FinancialEmbodiment(emms, symbol=args.symbol, pnl_scale=args.pnl_scale)

    stored = 0
    skipped = 0
    for row in trades:
        try:
            event = row_to_event(row, args.symbol, args.pnl_scale)
            # Skip zero-price or zero-qty rows (incomplete records)
            if event["price"] == 0 and event["qty"] == 0:
                skipped += 1
                continue
            memory_ids = fe._store_one(event)
            if memory_ids:
                stored += 1
                if stored % 10 == 0:
                    print(f"  Stored {stored} memories...")
        except Exception as exc:
            print(f"  Warning: failed to process row: {exc}")
            skipped += 1

    print(f"\nBackfill complete: {stored} memories stored, {skipped} rows skipped")

    # Save updated state
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    emms.save(state_path)
    print(f"EMMS state saved to {state_path}")


if __name__ == "__main__":
    main()
