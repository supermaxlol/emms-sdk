"""MemoryScheduler — composable async background maintenance jobs for EMMS.

Replaces the single ``_consolidation_loop`` with a proper scheduler that runs
multiple named jobs on independent cadences.  Each job is a coroutine factory
registered by name; the scheduler ticks every second and fires any job whose
interval has elapsed.

Built-in jobs (all configurable)
---------------------------------
``consolidation``     — calls ``EMMS.consolidate()``            every 60 s (default)
``ttl_purge``         — scans and marks TTL-expired memories     every 300 s
``deduplication``     — runs ``SemanticDeduplicator`` pass       every 600 s
``pattern_detection`` — calls ``EMMS.detect_patterns()``         every 300 s
``srs_review``        — logs count of SRS items due for review   every 3600 s

Usage::

    import asyncio
    from emms import EMMS
    from emms.scheduler import MemoryScheduler

    agent = EMMS()
    scheduler = MemoryScheduler(agent)

    async def main():
        await scheduler.start()
        # ... do work ...
        await scheduler.stop()

    asyncio.run(main())

Or via EMMS façade::

    await agent.start_scheduler()
    await agent.stop_scheduler()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from emms.emms import EMMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScheduledJob descriptor
# ---------------------------------------------------------------------------

@dataclass
class ScheduledJob:
    """Configuration for a single scheduled job.

    Parameters
    ----------
    name:
        Unique job identifier.
    interval_seconds:
        How often the job should run.
    enabled:
        Set False to skip this job without removing it.
    last_run:
        Unix timestamp of the last run; starts at 0 so jobs fire immediately.
    run_count:
        How many times this job has been executed.
    error_count:
        How many times this job has raised an exception.
    """
    name: str
    interval_seconds: float
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0
    error_count: int = 0

    @property
    def is_due(self) -> bool:
        """True if at least ``interval_seconds`` have passed since last run."""
        return self.enabled and (time.time() - self.last_run) >= self.interval_seconds


# ---------------------------------------------------------------------------
# MemoryScheduler
# ---------------------------------------------------------------------------

class MemoryScheduler:
    """Composable async job scheduler for EMMS background maintenance.

    Parameters
    ----------
    emms:
        The ``EMMS`` instance to operate on.
    consolidation_interval:
        Seconds between consolidation runs (default 60).
    ttl_purge_interval:
        Seconds between TTL purge scans (default 300).
    dedup_interval:
        Seconds between deduplication passes (default 600).
    pattern_interval:
        Seconds between pattern detection runs (default 300).
    srs_interval:
        Seconds between SRS due-item checks (default 3600).
    tick:
        Scheduler loop tick in seconds (default 1.0).
    """

    def __init__(
        self,
        emms: "EMMS",
        *,
        consolidation_interval: float = 60.0,
        ttl_purge_interval: float = 300.0,
        dedup_interval: float = 600.0,
        pattern_interval: float = 300.0,
        srs_interval: float = 3600.0,
        tick: float = 1.0,
    ) -> None:
        self.emms = emms
        self.tick = tick
        self._shutdown = False
        self._task: asyncio.Task | None = None

        # Built-in jobs
        self._jobs: dict[str, ScheduledJob] = {
            "consolidation": ScheduledJob("consolidation", consolidation_interval),
            "ttl_purge":     ScheduledJob("ttl_purge",     ttl_purge_interval),
            "deduplication": ScheduledJob("deduplication", dedup_interval),
            "pattern_detection": ScheduledJob("pattern_detection", pattern_interval),
            "srs_review":    ScheduledJob("srs_review",    srs_interval),
        }

        # Custom job registry: name → async coroutine factory (takes no args)
        self._custom: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {}

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float = 60.0,
        enabled: bool = True,
    ) -> None:
        """Register a custom background job.

        Args:
            name: Unique job name.
            coro_factory: A zero-argument callable returning a coroutine.
            interval_seconds: How often to run this job.
            enabled: Whether to run this job.
        """
        self._jobs[name] = ScheduledJob(name, interval_seconds, enabled=enabled)
        self._custom[name] = coro_factory
        logger.info("Scheduler: registered job %r every %.0fs", name, interval_seconds)

    def enable(self, name: str) -> None:
        """Enable a job by name."""
        if name in self._jobs:
            self._jobs[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a job by name (it stays registered but won't fire)."""
        if name in self._jobs:
            self._jobs[name].enabled = False

    def set_interval(self, name: str, interval_seconds: float) -> None:
        """Update the interval for an existing job."""
        if name in self._jobs:
            self._jobs[name].interval_seconds = interval_seconds

    @property
    def job_stats(self) -> dict[str, Any]:
        """Return a snapshot of all job states."""
        return {
            name: {
                "enabled": j.enabled,
                "interval_s": j.interval_seconds,
                "run_count": j.run_count,
                "error_count": j.error_count,
                "last_run_ago_s": round(time.time() - j.last_run, 1) if j.last_run else None,
                "is_due": j.is_due,
            }
            for name, j in self._jobs.items()
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler loop in the background."""
        self._shutdown = False
        self._task = asyncio.ensure_future(self._loop())
        logger.info("MemoryScheduler started (%d jobs)", len(self._jobs))

    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._shutdown = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("MemoryScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while not self._shutdown:
            await asyncio.sleep(self.tick)
            for name, job in list(self._jobs.items()):
                if not job.is_due:
                    continue
                job.last_run = time.time()
                job.run_count += 1
                try:
                    if name in self._custom:
                        await self._custom[name]()
                    else:
                        await self._run_builtin(name)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    job.error_count += 1
                    logger.warning("Scheduler job %r failed: %s", name, exc)

    async def _run_builtin(self, name: str) -> None:
        """Execute a built-in job by name."""
        emms = self.emms
        if name == "consolidation":
            result = emms.consolidate()
            logger.debug("Scheduler [consolidation]: moved=%d", result.get("items_consolidated", 0))

        elif name == "ttl_purge":
            count = self._purge_expired()
            logger.debug("Scheduler [ttl_purge]: purged=%d expired memories", count)

        elif name == "deduplication":
            count = self._run_deduplication()
            logger.debug("Scheduler [deduplication]: resolved=%d duplicate groups", count)

        elif name == "pattern_detection":
            try:
                patterns = emms.detect_patterns()
                seq_count = patterns.get("sequence", {}).get("count", 0)
                content_count = patterns.get("content", {}).get("count", 0)
                logger.debug(
                    "Scheduler [pattern_detection]: %d seq patterns, %d content patterns",
                    seq_count, content_count,
                )
                if seq_count or content_count:
                    emms.events.emit("memory.patterns_detected", patterns)
            except Exception as e:
                logger.warning("Scheduler [pattern_detection] error: %s", e)

        elif name == "srs_review":
            if hasattr(emms, "srs") and emms.srs is not None:
                due = emms.srs.get_due_items()
                logger.info("Scheduler [srs_review]: %d items due for review", len(due))
                if due:
                    emms.events.emit("srs.items_due", {"count": len(due), "items": [c.memory_id for c in due[:10]]})

    # ------------------------------------------------------------------
    # Built-in helpers
    # ------------------------------------------------------------------

    def _purge_expired(self) -> int:
        """Mark expired memories as superseded. Returns count purged."""
        count = 0
        for _, store in self.emms.memory._iter_tiers():
            for item in store:
                if item.is_expired and not item.is_superseded:
                    item.superseded_by = "__expired__"
                    count += 1
        return count

    def _run_deduplication(self) -> int:
        """Run SemanticDeduplicator on long-term memories. Returns groups resolved."""
        try:
            from emms.memory.compression import SemanticDeduplicator
            dedup = SemanticDeduplicator()
            items = list(self.emms.memory.long_term.values())
            if not items:
                return 0
            groups = dedup.find_duplicate_groups(items)
            if not groups:
                return 0
            archived = dedup.resolve_groups(groups)
            return len(archived)
        except Exception as e:
            logger.warning("Scheduler deduplication error: %s", e)
            return 0
