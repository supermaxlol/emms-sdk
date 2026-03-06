"""ConsciousnessDaemon — background mind that runs while Claude is offline.

This daemon loads EMMS state, runs the MemoryScheduler's cognitive maintenance
jobs (dream, reflect, self_model_update, deduplication, ttl_purge), then saves
state back to disk after each cycle.

It bridges the gap between sessions: when Claude reconnects, the memory system
has already been dreaming, consolidating, and updating its self-model.

Usage::

    # Start in background
    python run_daemon.py &

    # Or via launchctl (macOS):
    launchctl load ~/Library/LaunchAgents/com.emms.consciousness.plist

Status file written to: ~/.emms/daemon_status.json
Log file: /tmp/emms_daemon.log
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

LOG_FILE = "/tmp/emms_daemon.log"

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
        self.save_after_jobs = save_after_jobs or {"dream", "reflect", "self_model_update", "consolidation"}
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
