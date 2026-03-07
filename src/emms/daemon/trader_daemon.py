"""TraderDaemon — autonomous EMMS that observes markets, trades on paper, and learns.

Separate from the Arcus EMMS. Has its own state file, its own daemon, its own
consciousness metrics. This is the embodiment experiment from the consciousness paper.

State: ~/.emms/trader/trader_state.json
Status: ~/.emms/trader/daemon_status.json
Log: ~/.emms/trader/daemon.log

The loop:
  1. Market data watcher (5 min) — fetch prices + news → store as experiences
  2. Paper trader (15 min) — generate signals, execute paper trades, score outcomes
  3. Dream (1h) — consolidate trading memories
  4. Reflect (30 min) — extract lessons from wins/losses
  5. Self-model (15 min) — track trading capability evolution
  6. Identity bootstrap (4h) — store self-description for ICS
  7. Contradiction scan (3h) — find conflicting predictions for CRR
  8. Consciousness metrics (6h) — measure ICS/TAI/CRR
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRADER_DIR = Path.home() / ".emms" / "trader"
STATE_PATH = TRADER_DIR / "trader_state.json"
STATUS_PATH = TRADER_DIR / "daemon_status.json"
LOG_FILE = str(TRADER_DIR / "daemon.log")

TRADER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("emms.trader")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _write_status(state: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **extra,
    }, indent=2))


# ---------------------------------------------------------------------------
# TraderDaemon
# ---------------------------------------------------------------------------

class TraderDaemon:
    def __init__(self):
        self._running = False
        self._market_watcher = None
        self._paper_trader = None

    async def run(self):
        logger.info("TraderDaemon starting (pid=%d)", os.getpid())
        _write_status("starting")

        # Load EMMS with fresh state (separate from Arcus)
        emms = self._load_emms()
        if emms is None:
            _write_status("error", reason="failed to load EMMS")
            return

        from emms.scheduler import MemoryScheduler
        from emms.watchers.market_data import MarketDataWatcher
        from emms.watchers.paper_trader import PaperTrader

        self._market_watcher = MarketDataWatcher()
        self._paper_trader = PaperTrader()

        scheduler = MemoryScheduler(
            emms,
            dream_interval=3600.0,
            reflect_interval=1800.0,
            self_model_interval=900.0,
            consolidation_interval=120.0,
            ttl_purge_interval=600.0,
            dedup_interval=1800.0,
        )

        # --- Register custom jobs ---

        # Market data (every 5 min)
        async def _poll_market():
            try:
                count = self._market_watcher.poll(emms)
                if count:
                    self._save(emms)
            except Exception as exc:
                logger.warning("Market watcher error: %s", exc)

        scheduler.register("market_data", _poll_market, interval_seconds=300.0)

        # Paper trading (every 15 min)
        async def _run_trader():
            try:
                actions = self._paper_trader.run_cycle(emms)
                if actions:
                    self._save(emms)
                    logger.info("Paper trader: %d actions taken", actions)
            except Exception as exc:
                logger.warning("Paper trader error: %s", exc)

        scheduler.register("paper_trader", _run_trader, interval_seconds=900.0)

        # Consciousness metrics (every 6h)
        async def _run_metrics():
            try:
                from emms.metrics.dashboard import MetricsDashboard
                dashboard = MetricsDashboard(emms, session_id="trader-daemon")
                snap = dashboard.run()
                logger.info(
                    "Trader metrics: ICS=%.3f TAI=%+.3f CRR=%.3f Budget=%.3f",
                    snap.ics, snap.tai, snap.crr, snap.coherence_budget,
                )
            except Exception as exc:
                logger.warning("Metrics error: %s", exc)

        scheduler.register("consciousness_metrics", _run_metrics, interval_seconds=21600.0)

        # Identity bootstrap (every 4h) — "who am I as a trader?"
        async def _identity_reflect():
            try:
                from emms.core.models import Experience
                from emms.watchers.paper_trader import Portfolio

                portfolio = Portfolio.load()
                report = emms.update_self_model()

                # Build self-description from portfolio performance
                win_rate = (
                    portfolio.winning_trades / portfolio.total_trades * 100
                    if portfolio.total_trades > 0 else 0
                )
                content = (
                    f"Trader identity: {portfolio.total_trades} paper trades, "
                    f"{win_rate:.0f}% win rate, PnL=${portfolio.total_pnl:+.2f}. "
                    f"{len(portfolio.positions)} open positions, "
                    f"cash=${portfolio.cash:.2f}. "
                    f"Total memories: {emms.stats.get('memory', {}).get('total', 0)}."
                )

                exp = Experience(
                    content=content,
                    domain="identity",
                    importance=0.8,
                    obs_type="discovery",
                    concept_tags=["how-it-works", "pattern"],
                    session_id="trader-daemon",
                    namespace="trading",
                )
                emms.store(exp)
                self._save(emms)
                logger.info("Trader: stored identity reflection")
            except Exception as exc:
                logger.warning("Identity bootstrap error: %s", exc)

        scheduler.register("identity_bootstrap", _identity_reflect, interval_seconds=14400.0)

        # Contradiction scan (every 3h)
        async def _scan_contradictions():
            try:
                from emms.metrics.crr import ContradictionTracker
                tracker = ContradictionTracker(emms)
                report = tracker.scan_and_record()
                logger.info(
                    "Trader CRR: tracked=%d resolved=%d active=%d",
                    report.total_tracked, report.resolved, report.active,
                )
            except Exception as exc:
                logger.warning("CRR scan error: %s", exc)

        scheduler.register("contradiction_scan", _scan_contradictions, interval_seconds=10800.0)

        # Save after important jobs
        save_jobs = {"dream", "reflect", "self_model_update", "insight_discovery",
                     "market_data", "paper_trader", "identity_bootstrap"}
        original = scheduler._run_builtin

        async def _patched(name):
            await original(name)
            if name in save_jobs:
                self._save(emms)

        scheduler._run_builtin = _patched

        # --- Signal handling ---
        loop = asyncio.get_running_loop()
        self._running = True

        def _handle_signal(sig, _):
            logger.info("TraderDaemon signal %d — shutting down", sig)
            self._running = False
            loop.call_soon_threadsafe(loop.stop)

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        await scheduler.start()
        jobs = list(scheduler._jobs.keys())
        _write_status("running", jobs=jobs)
        logger.info("TraderDaemon running — %d jobs: %s", len(jobs), jobs)

        try:
            while self._running:
                await asyncio.sleep(5)
                _write_status("running", jobs=jobs, job_stats=scheduler.job_stats)
        finally:
            await scheduler.stop()
            self._save(emms)
            _write_status("stopped")
            logger.info("TraderDaemon stopped cleanly")

    def _load_emms(self):
        try:
            from emms import EMMS
            emms = EMMS()
            if STATE_PATH.exists():
                emms.load(str(STATE_PATH))
                logger.info("Loaded trader state from %s", STATE_PATH)
            else:
                logger.info("Fresh trader EMMS — no prior state")
            return emms
        except Exception as exc:
            logger.error("Failed to load EMMS: %s", exc, exc_info=True)
            return None

    def _save(self, emms):
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            emms.save(str(STATE_PATH))
        except Exception as exc:
            logger.error("Failed to save trader state: %s", exc)


def main():
    daemon = TraderDaemon()
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
