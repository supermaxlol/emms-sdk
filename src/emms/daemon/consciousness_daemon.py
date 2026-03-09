"""ConsciousnessDaemon — background mind that runs while Claude is offline.

This daemon loads EMMS state, runs the MemoryScheduler's cognitive maintenance
jobs (dream, reflect, self_model_update, deduplication, ttl_purge), then saves
state back to disk after each cycle.

It bridges the gap between sessions: when Claude reconnects, the memory system
has already been dreaming, consolidating, and updating its self-model.

New in v2: Trading bot watchers + consciousness metric bootstrapping.
  - Passively tails Margin-bot and ETrade-bot logs (read-only)
  - Parses trade signals, trend changes, PnL into EMMS experiences
  - Runs consciousness metrics (ICS/TAI/CRR) every 6 hours
  - Scans for contradictions to bootstrap CRR
  - Stores identity reflections to bootstrap ICS

Usage::

    # Start in background
    python -m emms.daemon.consciousness_daemon &

    # Or via launchctl (macOS):
    launchctl load ~/Library/LaunchAgents/com.emms.consciousness.plist

Status file written to: ~/.emms/daemon_status.json
Log file: ~/.emms/daemon.log
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
# Logging
# ---------------------------------------------------------------------------

EMMS_DIR = Path.home() / ".emms"
LOG_FILE = str(EMMS_DIR / "daemon.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("emms.daemon")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMMS_STATE_PATH = Path.home() / ".emms" / "emms_state.json"
DAEMON_STATUS_PATH = Path.home() / ".emms" / "daemon_status.json"


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------

def _write_status(state: str, **extra: object) -> None:
    """Write a JSON status file so other processes can check daemon health."""
    DAEMON_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **extra,
    }
    DAEMON_STATUS_PATH.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# ConsciousnessDaemon
# ---------------------------------------------------------------------------

class ConsciousnessDaemon:
    """Async daemon that runs EMMS background maintenance while Claude is offline.

    Parameters
    ----------
    state_path:
        Path to the EMMS state JSON file.
    save_after_jobs:
        Which job names should trigger a state save after completion.
    scheduler_kwargs:
        Passed directly to MemoryScheduler.__init__ to tune intervals.
    """

    def __init__(
        self,
        state_path: Path = EMMS_STATE_PATH,
        *,
        save_after_jobs: set[str] | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        self.state_path = state_path
        self.save_after_jobs = save_after_jobs or {"dream", "reflect", "self_model_update", "consolidation", "insight_discovery"}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self._running = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Load EMMS, start scheduler, and run until SIGTERM/SIGINT."""
        logger.info("ConsciousnessDaemon starting (pid=%d)", os.getpid())
        _write_status("starting")

        emms = self._load_emms()
        if emms is None:
            _write_status("error", reason="failed to load EMMS")
            return

        from emms.scheduler import MemoryScheduler

        # Use shorter intervals for daemon (it runs continuously)
        default_kwargs = {
            "dream_interval": 3600.0,         # every hour
            "reflect_interval": 1800.0,       # every 30 min
            "self_model_interval": 900.0,     # every 15 min
            "consolidation_interval": 120.0,  # every 2 min (more frequent than MCP)
            "ttl_purge_interval": 600.0,
            "dedup_interval": 1800.0,
        }
        default_kwargs.update(self.scheduler_kwargs)

        scheduler = MemoryScheduler(emms, **default_kwargs)

        # NOTE: Bot watchers moved to TraderDaemon (separate EMMS instance)

        # --- Register consciousness metrics job ---
        self._register_metrics_job(scheduler, emms)

        # --- Register identity bootstrapper ---
        self._register_identity_bootstrap(scheduler, emms)

        # --- Register contradiction scanner (CRR bootstrap) ---
        self._register_contradiction_scanner(scheduler, emms)

        # --- Register LLM consolidation (Google always-on-agent pattern) ---
        self._register_llm_consolidation(scheduler, emms)

        # --- Register bidirectional edge sync ---
        self._register_bidirectional_sync(scheduler, emms)

        # Patch scheduler loop to save after important jobs
        original_run_builtin = scheduler._run_builtin

        async def _patched_run_builtin(name: str) -> None:
            await original_run_builtin(name)
            if name in self.save_after_jobs:
                self._save_emms(emms)
                logger.info("Daemon: state saved after job %r", name)
            if name == "metrics":
                self._run_metrics(emms)

        scheduler._run_builtin = _patched_run_builtin  # type: ignore[method-assign]

        # Signal handling
        loop = asyncio.get_running_loop()
        self._running = True

        def _handle_signal(sig: int, _frame: object) -> None:
            logger.info("ConsciousnessDaemon received signal %d — shutting down", sig)
            self._running = False
            loop.call_soon_threadsafe(loop.stop)

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        await scheduler.start()

        # Start HTTP REST API (non-blocking daemon thread)
        try:
            from emms.adapters.http_server import start_server
            start_server(emms, port=8765)
            logger.info("EMMS HTTP API listening on http://127.0.0.1:8765")
        except Exception as exc:
            logger.warning("Daemon: HTTP server failed to start: %s", exc)

        _write_status("running", scheduler_jobs=list(scheduler._jobs.keys()))
        logger.info("ConsciousnessDaemon running — scheduler started with %d jobs", len(scheduler._jobs))

        try:
            while self._running:
                await asyncio.sleep(5)
                _write_status(
                    "running",
                    scheduler_jobs=list(scheduler._jobs.keys()),
                    job_stats=scheduler.job_stats,
                )
        finally:
            await scheduler.stop()
            self._save_emms(emms)
            _write_status("stopped")
            logger.info("ConsciousnessDaemon stopped cleanly")

    # ------------------------------------------------------------------
    # Watcher & job registration
    # ------------------------------------------------------------------

    def _register_watchers(self, scheduler, emms) -> None:
        """Register trading bot log watchers as scheduler jobs."""
        try:
            from emms.watchers.trading_bot import (
                create_margin_bot_watcher,
                create_etrade_bot_watcher,
                ingest_events,
            )
            watchers = []
            try:
                watchers.append(create_margin_bot_watcher())
                logger.info("Daemon: registered margin-bot watcher")
            except Exception as exc:
                logger.warning("Daemon: margin-bot watcher failed: %s", exc)
            try:
                watchers.append(create_etrade_bot_watcher())
                logger.info("Daemon: registered etrade-bot watcher")
            except Exception as exc:
                logger.warning("Daemon: etrade-bot watcher failed: %s", exc)

            if not watchers:
                return

            async def _poll_bots():
                total = 0
                for w in watchers:
                    try:
                        events = w.poll()
                        if events:
                            total += ingest_events(emms, events)
                    except Exception as exc:
                        logger.warning("Daemon: watcher %s poll error: %s", w.bot_name, exc)
                if total:
                    self._save_emms(emms)

            scheduler.register("bot_watchers", _poll_bots, interval_seconds=30.0)
            self.save_after_jobs.add("bot_watchers")

        except ImportError as exc:
            logger.warning("Daemon: watchers module not available: %s", exc)

    def _register_metrics_job(self, scheduler, emms) -> None:
        """Run consciousness metrics every 6 hours."""
        async def _run_metrics():
            self._run_metrics(emms)

        scheduler.register("consciousness_metrics", _run_metrics, interval_seconds=21600.0)

    def _register_identity_bootstrap(self, scheduler, emms) -> None:
        """Store identity reflections every 4 hours to bootstrap ICS."""
        async def _identity_reflect():
            try:
                from emms.core.models import Experience

                # Self-model update generates beliefs
                report = emms.update_self_model()
                if not report:
                    return

                beliefs = report.get("beliefs", []) if isinstance(report, dict) else getattr(report, "beliefs", [])
                consistency = report.get("consistency_score", 0) if isinstance(report, dict) else getattr(report, "consistency_score", 0)
                capabilities = report.get("capability_profile", {}) if isinstance(report, dict) else getattr(report, "capability_profile", {})

                # Store a self-description memory (domain=identity, tagged for ICS)
                cap_str = ", ".join(f"{k}={v:.2f}" for k, v in (capabilities.items() if isinstance(capabilities, dict) else []))
                belief_str = "; ".join(
                    getattr(b, "content", str(b))[:60] for b in (beliefs[:3] if beliefs else [])
                )
                content = (
                    f"Self-model update: consistency={consistency:.2f}. "
                    f"Capabilities: {cap_str or 'none yet'}. "
                    f"Core beliefs: {belief_str or 'none yet'}. "
                    f"Total memories: {emms.stats.get('memory', {}).get('total', 0) if hasattr(emms, 'stats') else '?'}."
                )

                exp = Experience(
                    content=content,
                    domain="identity",
                    importance=0.8,
                    obs_type="discovery",
                    concept_tags=["how-it-works", "pattern"],
                    session_id="daemon",
                    namespace="default",
                )
                emms.store(exp)
                self._save_emms(emms)
                logger.info("Daemon: stored identity reflection (ICS bootstrap)")
            except Exception as exc:
                logger.warning("Daemon: identity bootstrap error: %s", exc)

        scheduler.register("identity_bootstrap", _identity_reflect, interval_seconds=14400.0)

    def _register_contradiction_scanner(self, scheduler, emms) -> None:
        """Scan for contradictions every 3 hours to bootstrap CRR."""
        async def _scan_contradictions():
            try:
                from emms.metrics.crr import ContradictionTracker
                tracker = ContradictionTracker(emms)
                report = tracker.scan_and_record()
                logger.info(
                    "Daemon: CRR scan — tracked=%d resolved=%d active=%d new=%d",
                    report.total_tracked, report.resolved, report.active, report.new_this_scan,
                )
            except Exception as exc:
                logger.warning("Daemon: CRR scan error: %s", exc)

        scheduler.register("contradiction_scan", _scan_contradictions, interval_seconds=10800.0)

    def _register_llm_consolidation(self, scheduler, emms) -> None:
        """Auto-consolidate long-term memory every 30 min (Google always-on-agent pattern)."""
        async def _llm_consolidate():
            try:
                result = await emms.llm_consolidate(threshold=0.7, tier="long_term", max_clusters=20)
                logger.info(
                    "Daemon [llm_consolidate]: clusters=%d synthesised=%d stored=%d elapsed=%.1fs",
                    result.clusters_found, result.synthesised, result.stored, result.elapsed_s,
                )
                self._save_emms(emms)
            except Exception as exc:
                logger.warning("Daemon: llm_consolidate job failed: %s", exc)

        scheduler.register("llm_consolidate", _llm_consolidate, interval_seconds=1800.0)

    def _register_bidirectional_sync(self, scheduler, emms) -> None:
        """Ensure association graph edges are bidirectional every 30 min."""
        async def _bidirectional_sync():
            try:
                ag = getattr(emms, "_association_graph", None)
                if ag is None:
                    return  # Graph not built yet — nothing to sync
                n = ag.ensure_bidirectional()
                if n:
                    logger.info("Daemon [bidirectional_sync]: added %d reverse edges", n)
            except Exception as exc:
                logger.warning("Daemon: bidirectional_sync job failed: %s", exc)

        scheduler.register("bidirectional_sync", _bidirectional_sync, interval_seconds=1800.0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_emms(self) -> object | None:
        try:
            from emms import EMMS
            emms = EMMS()
            if self.state_path.exists():
                emms.load(str(self.state_path))
                logger.info("Daemon: loaded state from %s", self.state_path)
            else:
                logger.warning("Daemon: no state file at %s — starting fresh", self.state_path)
            return emms
        except Exception as exc:
            logger.error("Daemon: failed to load EMMS: %s", exc, exc_info=True)
            return None

    def _save_emms(self, emms: object) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            emms.save(str(self.state_path))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error("Daemon: failed to save EMMS state: %s", exc)

    def _run_metrics(self, emms: object) -> None:
        """Run a metrics snapshot and append to ~/.emms/metrics_log.jsonl."""
        try:
            from emms.metrics.dashboard import MetricsDashboard
            dashboard = MetricsDashboard(emms, session_id="daemon")  # type: ignore[arg-type]
            snap = dashboard.run()
            logger.info(
                "Daemon metrics: ICS=%.3f TAI=%+.3f CRR=%.3f Budget=%.3f",
                snap.ics, snap.tai, snap.crr, snap.coherence_budget,
            )
        except Exception as exc:
            logger.warning("Daemon: metrics job failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point (called by run_daemon.py)
# ---------------------------------------------------------------------------

def main() -> None:
    daemon = ConsciousnessDaemon()
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
