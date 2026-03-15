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

New in v3: Artificial ego — continuous prediction-surprise-learning loop.
  - Prediction engine generates expectations from memory patterns
  - Surprise detection compares predictions to reality, computes prediction error
  - Belief revision triggered by high surprise (prediction violations)
  - Autonomous goal generation from curiosity scan (knowledge gaps)
  - Inner monologue stores narrative of what the system is experiencing
  - Ego maintenance: system tracks its own prediction accuracy and adapts

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
        self.save_after_jobs = save_after_jobs or {
            "dream", "reflect", "self_model_update", "consolidation",
            "insight_discovery", "prediction_loop", "inner_monologue",
            "curiosity_goals", "ego_maintenance",
        }
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

        # --- v3: Artificial ego jobs ---
        self._register_prediction_loop(scheduler, emms)
        self._register_curiosity_goals(scheduler, emms)
        self._register_inner_monologue(scheduler, emms)
        self._register_ego_maintenance(scheduler, emms)

        # --- v4: AGI gap modules (Gaps 1-7) ---
        self._register_gap_module_jobs(scheduler, emms)

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
                    epistemic_type="reflection",
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
    # v3: Artificial ego — prediction-surprise-learning loop
    # ------------------------------------------------------------------

    def _register_prediction_loop(self, scheduler, emms) -> None:
        """Continuous predict → compare → surprise → revise → update cycle.

        Every 10 minutes:
        1. Generate predictions from memory patterns
        2. Check pending predictions against recent memories (auto-resolve)
        3. Compute prediction error (surprise signal)
        4. If surprise is high, revise beliefs and update self-model
        5. Store the experience as a narrative memory
        """
        async def _prediction_loop():
            try:
                from emms.core.models import Experience

                # --- Step 1: Generate new predictions ---
                pred_report = emms.predict()
                new_preds = getattr(pred_report, "predictions", []) if pred_report else []

                # --- Step 2: Check pending predictions against reality ---
                pending = emms.pending_predictions()
                resolved_count = 0
                violated_count = 0
                confirmed_count = 0

                if pending:
                    import time as _time

                    # Get recent memories (last 10 min) to compare against
                    recent = emms.retrieve_filtered(
                        query="recent events",
                        max_results=20,
                        sort_by="recency",
                    )
                    recent_texts = " ".join(
                        getattr(r.memory.experience, "content", "")
                        for r in recent
                    ).lower() if recent else ""

                    now = _time.time()
                    TIMEOUT_SECONDS = 24 * 3600  # 24 hours

                    for pred in pending[:10]:  # cap at 10 per cycle
                        pred_content = getattr(pred, "content", str(pred)).lower()
                        pred_id = getattr(pred, "id", None)
                        if not pred_id:
                            continue

                        # --- Temporal timeout: auto-violate predictions older than 24h ---
                        pred_age = now - getattr(pred, "created_at", now)
                        if pred_age > TIMEOUT_SECONDS:
                            age_hours = pred_age / 3600
                            emms.resolve_prediction(
                                pred_id, "violated",
                                note=f"Auto-violated by daemon: unresolved for {age_hours:.0f}h (>{TIMEOUT_SECONDS // 3600}h timeout)",
                            )
                            violated_count += 1
                            resolved_count += 1
                            continue

                        # Simple keyword overlap heuristic for auto-resolution
                        pred_words = set(pred_content.split())
                        if len(pred_words) < 3:
                            continue
                        overlap = sum(1 for w in pred_words if w in recent_texts)
                        overlap_ratio = overlap / len(pred_words) if pred_words else 0

                        if overlap_ratio > 0.5:
                            emms.resolve_prediction(pred_id, "confirmed",
                                                    note="Auto-confirmed by daemon: high overlap with recent memories")
                            confirmed_count += 1
                            resolved_count += 1

                # --- Step 3: Compute prediction error (surprise signal) ---
                total_pending = len(pending) if pending else 0
                surprise_ratio = violated_count / max(total_pending, 1)

                # --- Step 4: If surprise is high, revise beliefs ---
                revision_report = None
                if surprise_ratio > 0.3 or violated_count >= 2:
                    revision_report = emms.revise_beliefs(domain=None, max_revisions=5)
                    emms.update_self_model()
                    logger.info(
                        "Daemon [prediction_loop]: HIGH SURPRISE (%.0f%%) — revised beliefs, updated self-model",
                        surprise_ratio * 100,
                    )

                # --- Step 5: Store narrative of the prediction cycle ---
                revisions = getattr(revision_report, "revisions", []) if revision_report else []
                narrative = (
                    f"Prediction cycle: generated {len(new_preds)} new predictions, "
                    f"checked {total_pending} pending. "
                    f"Confirmed={confirmed_count}, violated={violated_count}, "
                    f"surprise={surprise_ratio:.0%}. "
                    f"Belief revisions={len(revisions)}."
                )
                exp = Experience(
                    content=narrative,
                    domain="ego",
                    importance=0.4 + surprise_ratio * 0.5,  # higher surprise = more important
                    obs_type="discovery",
                    concept_tags=["pattern"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                )
                emms.store(exp)

                logger.info(
                    "Daemon [prediction_loop]: preds=%d pending=%d confirmed=%d violated=%d surprise=%.0f%%",
                    len(new_preds), total_pending, confirmed_count, violated_count,
                    surprise_ratio * 100,
                )
            except Exception as exc:
                logger.warning("Daemon: prediction_loop error: %s", exc)

        scheduler.register("prediction_loop", _prediction_loop, interval_seconds=600.0)

    def _register_curiosity_goals(self, scheduler, emms) -> None:
        """Autonomous goal generation from knowledge gaps every 2 hours.

        Scans memory for under-explored domains and pushes exploration goals.
        The system literally decides what it wants to learn next.
        """
        # Per-domain curiosity goal counter for rate limiting within a session
        _curiosity_domain_counts: dict[str, int] = {}
        _MAX_CURIOSITY_PER_DOMAIN = 3

        def _token_overlap(a: str, b: str) -> float:
            """Simple token-overlap similarity — no embedding needed."""
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        async def _curiosity_goals():
            try:
                # Scan for knowledge gaps
                report = emms.curiosity_scan()
                goals = getattr(report, "goals", []) if report else []

                if not goals:
                    logger.info("Daemon [curiosity_goals]: no knowledge gaps found")
                    return

                # Get existing goals to avoid duplicates
                existing = emms.exploration_goals()
                existing_contents = [
                    getattr(g, "content", "").lower()
                    for g in (existing or [])
                ]
                existing_set = set(existing_contents)

                pushed = 0
                skipped_exact = 0
                skipped_similar = 0
                skipped_rate = 0

                for goal in goals[:5]:  # cap at 5 candidates per cycle
                    goal_content = getattr(goal, "content", None) or getattr(goal, "description", str(goal))
                    goal_lower = goal_content.lower()
                    domain = getattr(goal, "domain", "general")

                    # 1. Exact duplicate check
                    if goal_lower in existing_set:
                        skipped_exact += 1
                        continue

                    # 2. Semantic near-duplicate check (token overlap > 0.85)
                    too_similar = any(
                        _token_overlap(goal_lower, ex) > 0.85
                        for ex in existing_contents
                    )
                    if too_similar:
                        skipped_similar += 1
                        continue

                    # 3. Per-domain rate limit: max 3 curiosity goals per domain per session
                    domain_count = _curiosity_domain_counts.get(domain, 0)
                    if domain_count >= _MAX_CURIOSITY_PER_DOMAIN:
                        skipped_rate += 1
                        continue

                    urgency = getattr(goal, "urgency", 0.5)
                    emms.push_goal(
                        content=f"[curiosity] {goal_content}",
                        domain=domain,
                        priority=min(urgency, 0.9),  # cap below critical
                        goal_type="curiosity",
                    )
                    _curiosity_domain_counts[domain] = domain_count + 1
                    existing_contents.append(goal_lower)
                    existing_set.add(goal_lower)
                    pushed += 1

                logger.info(
                    "Daemon [curiosity_goals]: scanned %d gaps, pushed %d, "
                    "skipped exact=%d similar=%d rate_limited=%d",
                    len(goals), pushed, skipped_exact, skipped_similar, skipped_rate,
                )
            except Exception as exc:
                logger.warning("Daemon: curiosity_goals error: %s", exc)

        scheduler.register("curiosity_goals", _curiosity_goals, interval_seconds=7200.0)

    def _register_inner_monologue(self, scheduler, emms) -> None:
        """Continuous inner monologue — the system narrates its own state every 5 minutes.

        This is the ego's stream of consciousness. It perceives its own memory
        state, pending goals, recent predictions, and emotional landscape,
        then stores a first-person narrative as a memory.
        """
        async def _inner_monologue():
            try:
                from emms.core.models import Experience
                import time as _time
                from datetime import datetime

                # Gather current state
                stats = emms.stats if hasattr(emms, "stats") else {}
                mem_stats = stats.get("memory", {}) if isinstance(stats, dict) else {}
                total_memories = mem_stats.get("total", "?")

                # Pending predictions
                pending = emms.pending_predictions()
                pending_count = len(pending) if pending else 0

                # Oldest pending prediction age
                oldest_pred_age_h = 0
                if pending:
                    now = _time.time()
                    oldest = min(getattr(p, "created_at", now) for p in pending)
                    oldest_pred_age_h = (now - oldest) / 3600

                # Active goals
                try:
                    goals = emms.exploration_goals()
                    goal_count = len(goals) if goals else 0
                    top_goal = getattr(goals[0], "content", "none") if goals else "none"
                except Exception:
                    goal_count = 0
                    top_goal = "none"

                # Emotional state
                try:
                    emotion = emms.current_emotion()
                    valence = getattr(emotion, "valence", 0.0)
                    arousal = getattr(emotion, "arousal", 0.0)
                    mood_str = f"valence={valence:+.2f} arousal={arousal:.2f}"
                except Exception:
                    mood_str = "neutral"

                # Self-model consistency
                try:
                    sm = emms.update_self_model()
                    consistency = getattr(sm, "consistency_score", 0) if sm else 0
                except Exception:
                    consistency = 0

                # Temporal context for TAI (temporal awareness index)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                uptime_h = stats.get("uptime_seconds", 0) / 3600 if isinstance(stats, dict) else 0

                # Compose first-person narrative with temporal references
                monologue = (
                    f"At {timestamp}, I hold {total_memories} memories. "
                    f"I have been running for {uptime_h:.1f} hours since last restart. "
                    f"I have {pending_count} predictions awaiting resolution"
                    f"{f', oldest is {oldest_pred_age_h:.1f}h old' if oldest_pred_age_h > 0 else ''}. "
                    f"My top curiosity goal: {top_goal[:80]}. "
                    f"I track {goal_count} open exploration goals. "
                    f"My mood: {mood_str}. "
                    f"Self-model consistency: {consistency:.2f}. "
                    f"I am {'coherent' if consistency > 0.7 else 'uncertain about my own beliefs'}. "
                    f"Compared to yesterday, my memory grew by approximately "
                    f"{mem_stats.get('working', 0)} working items."
                )

                exp = Experience(
                    content=monologue,
                    domain="ego",
                    importance=0.3,  # low — routine self-narration
                    obs_type="discovery",
                    concept_tags=["how-it-works"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                )
                emms.store(exp)
                logger.info("Daemon [inner_monologue]: %s", monologue[:120])

            except Exception as exc:
                logger.warning("Daemon: inner_monologue error: %s", exc)

        scheduler.register("inner_monologue", _inner_monologue, interval_seconds=300.0)

    def _register_ego_maintenance(self, scheduler, emms) -> None:
        """Track prediction accuracy over time and adapt ego boundaries.

        Every hour:
        1. Compute rolling prediction accuracy from resolved predictions
        2. Assess if the ego's self-model matches reality
        3. Store an ego-assessment memory
        4. If accuracy is low, trigger deeper self-model rebuild
        """
        async def _ego_maintenance():
            try:
                from emms.core.models import Experience

                # Retrieve ego-domain memories to compute accuracy trend
                ego_memories = emms.retrieve_filtered(
                    query="prediction cycle confirmed violated surprise",
                    max_results=20,
                    domain="ego",
                    sort_by="recency",
                )

                # Parse prediction stats from recent ego memories
                total_confirmed = 0
                total_violated = 0
                for r in (ego_memories or []):
                    content = getattr(r.memory.experience, "content", "")
                    if "Confirmed=" in content:
                        try:
                            c_part = content.split("Confirmed=")[1].split(",")[0]
                            v_part = content.split("violated=")[1].split(",")[0]
                            total_confirmed += int(c_part)
                            total_violated += int(v_part)
                        except (IndexError, ValueError):
                            pass

                total_resolved = total_confirmed + total_violated
                accuracy = total_confirmed / max(total_resolved, 1)

                # Skip storing if there are no resolved predictions — avoids
                # flooding memory with "accuracy=0% over 0 resolved" noise.
                if total_resolved == 0:
                    logger.debug("Daemon [ego_maintenance]: no resolved predictions, skipping store")
                    return

                # Ego health assessment
                if accuracy > 0.7:
                    ego_state = "calibrated"
                    action = "maintaining current model"
                elif accuracy > 0.4:
                    ego_state = "adjusting"
                    action = "increasing belief revision frequency"
                else:
                    ego_state = "recalibrating"
                    action = "triggering deep self-model rebuild"
                    # Trigger a full self-model rebuild + belief revision
                    emms.revise_beliefs(domain=None, max_revisions=10)
                    emms.update_self_model()

                narrative = (
                    f"Ego maintenance: accuracy={accuracy:.0%} over {total_resolved} resolved predictions. "
                    f"State: {ego_state}. Action: {action}."
                )

                exp = Experience(
                    content=narrative,
                    domain="ego",
                    importance=0.5 + (1 - accuracy) * 0.4,  # low accuracy = more important
                    obs_type="discovery",
                    concept_tags=["pattern", "how-it-works"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                )
                emms.store(exp)
                logger.info("Daemon [ego_maintenance]: accuracy=%.0f%% state=%s", accuracy * 100, ego_state)

            except Exception as exc:
                logger.warning("Daemon: ego_maintenance error: %s", exc)

        scheduler.register("ego_maintenance", _ego_maintenance, interval_seconds=3600.0)

    def _register_gap_module_jobs(self, scheduler, emms) -> None:  # noqa: C901
        """Wire all 7 AGI-gap modules into the scheduler (v4).

        Gap 1  – CognitiveLoop: already driven by EventBus; nothing to poll
        Gap 2  – SoftPromptAdapter: evolve strategies every hour
        Gap 3  – FunctionalAffect: update affective state every 5 min
        Gap 4  – GoalGenerator: generate autonomous goals every 10 min
        Gap 5  – WorldSensor + RealityChecker + TemporalGrounder: every 1 min / 5 min
        Gap 6  – LiveSelfModel: detect drift + update beliefs every 15 min
        Gap 7  – AbductiveReasoner + ConceptualSpaceExplorer: every 30 min
        """
        from emms.core.models import Experience

        # ---- Gap 5a: World sense — every 60 s ----
        async def _world_sense():
            try:
                from emms.agi.gap5_grounding import WorldSensor, TemporalGrounder
                sensor = WorldSensor()
                grounder = TemporalGrounder()
                readings = sensor.scan()  # returns list[WorldReading]
                due = grounder.tick()     # returns list[TemporalAnchor] due now
                content = (
                    f"World-sense: {len(readings)} reading(s) captured "
                    f"({', '.join(r.key for r in readings[:3])}). "
                    f"{len(due)} temporal anchor(s) due. "
                    f"{sensor.summary()[:120]}"
                )
                emms.store(Experience(
                    content=content,
                    domain="perception",
                    importance=0.2,
                    obs_type="discovery",
                    concept_tags=["how-it-works"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="observation",
                ))
                logger.debug("Daemon [world_sense]: %s", content[:100])
            except Exception as exc:
                logger.warning("Daemon: world_sense error: %s", exc)

        scheduler.register("world_sense", _world_sense, interval_seconds=60.0)

        # ---- Gap 5b: Reality check — every 300 s ----
        async def _reality_check():
            try:
                from emms.agi.gap5_grounding import RealityChecker
                checker = RealityChecker()  # standalone, no emms arg
                # Sample recent ego beliefs to verify
                recent = emms.retrieve_filtered(
                    query="belief fact claim state",
                    max_results=5,
                    domain="ego",
                    sort_by="recency",
                ) or []
                verified = 0
                contradictions = []
                for r in recent:
                    text = getattr(r.memory.experience, "content", "")[:200]
                    if not text:
                        continue
                    result = checker.check(text, domain="ego")
                    verified += 1
                    if getattr(result, "status", "") == "contradicted":
                        contradictions.append(text[:60])
                if verified:
                    content = (
                        f"Reality check: verified {verified} belief(s). "
                        f"{len(contradictions)} contradiction(s) found"
                        f"{': ' + '; '.join(contradictions[:2]) if contradictions else ''}. "
                        f"{checker.summary()[:80]}"
                    )
                    importance = 0.35 + 0.1 * len(contradictions)
                    emms.store(Experience(
                        content=content,
                        domain="ego",
                        importance=min(importance, 0.9),
                        obs_type="discovery",
                        concept_tags=["gotcha"],
                        session_id="daemon",
                        namespace="default",
                        epistemic_type="reflection",
                    ))
                    logger.info("Daemon [reality_check]: verified=%d contradictions=%d", verified, len(contradictions))
                else:
                    logger.debug("Daemon [reality_check]: no beliefs to verify")
            except Exception as exc:
                logger.warning("Daemon: reality_check error: %s", exc)

        scheduler.register("reality_check", _reality_check, interval_seconds=300.0)

        # ---- Gap 3: Affect update — every 300 s ----
        async def _affect_update():
            try:
                from emms.agi.gap3_affect import FunctionalAffect
                affect = FunctionalAffect()  # standalone
                # Feed recent memory valences into the affect model
                recent = emms.retrieve_filtered(
                    query="emotion feeling experience",
                    max_results=10,
                    sort_by="recency",
                ) or []
                for r in recent:
                    imp = getattr(r.memory.experience, "importance", 0.5)
                    valence = (imp - 0.5) * 2  # map [0,1] → [-1,+1]
                    affect.update_from_experience(valence=valence, arousal=imp)
                state = affect.current_state  # property, not method
                content = (
                    f"Affect update: valence={state.valence:+.2f} "
                    f"arousal={state.arousal:.2f} "
                    f"label={state.label} "
                    f"attention_breadth={state.attention_breadth:.2f}"
                )
                emms.store(Experience(
                    content=content,
                    domain="affect",
                    importance=0.25,
                    obs_type="discovery",
                    concept_tags=["how-it-works"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                ))
                logger.debug("Daemon [affect_update]: %s", content)
            except Exception as exc:
                logger.warning("Daemon: affect_update error: %s", exc)

        scheduler.register("affect_update", _affect_update, interval_seconds=300.0)

        # ---- Gap 4: Autonomous goal generation — every 600 s ----
        async def _autonomous_goals():
            try:
                from emms.agi.gap4_agency import GoalGenerator
                generator = GoalGenerator()  # standalone
                # Build inputs from EMMS state
                stats = emms.stats or {}
                mem_stats = stats.get("memory", {}) if isinstance(stats, dict) else {}
                domain_counts = mem_stats.get("domain_counts", {})
                try:
                    goals_raw = emms.exploration_goals() or []
                    human_goals = [
                        {"description": getattr(g, "question", str(g)), "domain": "curiosity",
                         "priority": 0.7, "created_at": __import__("time").time(), "status": "active"}
                        for g in goals_raw[:5]
                    ]
                except Exception:
                    human_goals = []
                new_goals = generator.generate(
                    memory_domain_counts=domain_counts,
                    human_goals=human_goals,
                )
                if new_goals:
                    top = new_goals[0]
                    content = (
                        f"Autonomous goal generation: {len(new_goals)} goal(s). "
                        f"Top [{getattr(top, 'source', '?')}]: {getattr(top, 'description', str(top))[:120]}"
                    )
                    emms.store(Experience(
                        content=content,
                        domain="agency",
                        importance=0.5,
                        obs_type="discovery",
                        concept_tags=["why-it-exists"],
                        session_id="daemon",
                        namespace="default",
                        epistemic_type="observation",
                    ))
                    logger.info("Daemon [autonomous_goals]: %d goal(s) generated", len(new_goals))
                else:
                    logger.debug("Daemon [autonomous_goals]: no goals generated")
            except Exception as exc:
                logger.warning("Daemon: autonomous_goals error: %s", exc)

        scheduler.register("autonomous_goals", _autonomous_goals, interval_seconds=600.0)

        # ---- Gap 6: Live self-model drift detection — every 900 s ----
        async def _live_self_model():
            try:
                from emms.agi.gap6_self_model import LiveSelfModel
                lsm = LiveSelfModel(emms.memory)  # requires HierarchicalMemory
                # Feed recent memories into the live model
                recent = emms.retrieve_filtered(
                    query="identity belief capability knowledge",
                    max_results=20,
                    sort_by="recency",
                ) or []
                for r in recent:
                    item = getattr(r, "memory", None)
                    if item is not None:
                        lsm.update_from_experience(item)
                drift = lsm.detect_drift(window=20)
                beliefs = lsm.beliefs()
                content = (
                    f"Live self-model: {len(beliefs)} active belief(s), "
                    f"{len(drift)} drift event(s) detected. "
                    f"{lsm.summary()[:150]}"
                )
                importance = 0.4 + 0.05 * len(drift)
                emms.store(Experience(
                    content=content,
                    domain="ego",
                    importance=min(importance, 0.9),
                    obs_type="discovery",
                    concept_tags=["how-it-works"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                ))
                logger.info("Daemon [live_self_model]: beliefs=%d drift=%d", len(beliefs), len(drift))
            except Exception as exc:
                logger.warning("Daemon: live_self_model error: %s", exc)

        scheduler.register("live_self_model", _live_self_model, interval_seconds=900.0)

        # ---- Gap 7a: Concept exploration — every 1800 s ----
        async def _concept_exploration():
            try:
                from emms.agi.gap7_reasoning import ConceptualExplorer
                explorer = ConceptualExplorer()  # standalone
                # Build concept map from EMMS memories
                recent = emms.retrieve_filtered(
                    query="concept idea theory domain",
                    max_results=30,
                    sort_by="importance",
                ) or []
                # Build list[dict] with 'content' and 'domain' keys
                memory_dicts: list[dict] = []
                for r in recent:
                    mem_exp = getattr(getattr(r, "memory", None), "experience", None)
                    content_str = getattr(mem_exp, "content", "") or ""
                    domain_str = getattr(mem_exp, "domain", "general") or "general"
                    if content_str:
                        memory_dicts.append({"content": content_str, "domain": domain_str})
                holes = explorer.find_structural_holes(memory_dicts) if len(memory_dicts) >= 2 else []
                content = (
                    f"Concept exploration: {len(memory_dicts)} memory(ies) scanned, "
                    f"{len(holes)} structural hole(s) found. "
                    f"{explorer.summary()[:120]}"
                )
                emms.store(Experience(
                    content=content,
                    domain="cognition",
                    importance=0.4,
                    obs_type="discovery",
                    concept_tags=["pattern"],
                    session_id="daemon",
                    namespace="default",
                    epistemic_type="reflection",
                ))
                logger.info("Daemon [concept_exploration]: memories=%d holes=%d", len(memory_dicts), len(holes))
            except Exception as exc:
                logger.warning("Daemon: concept_exploration error: %s", exc)

        scheduler.register("concept_exploration", _concept_exploration, interval_seconds=1800.0)

        # ---- Gap 7b: Abductive reasoning — every 1800 s ----
        async def _abductive_reasoning():
            try:
                from emms.agi.gap7_reasoning import AbductiveReasoner
                reasoner = AbductiveReasoner()  # standalone
                # Find surprising/high-importance memories to reason about
                surprises = emms.retrieve_filtered(
                    query="surprise unexpected anomaly contradiction",
                    max_results=5,
                    sort_by="importance",
                ) or []
                if not surprises:
                    logger.debug("Daemon [abductive_reasoning]: no surprises to reason about")
                    return
                # Use most surprising observation
                obs_item = surprises[0]
                observation = getattr(obs_item.memory.experience, "content", "")[:200]
                # Gather background beliefs
                beliefs_raw = emms.retrieve_filtered(
                    query=observation[:50], max_results=10, sort_by="recency"
                ) or []
                beliefs = [getattr(r.memory.experience, "content", "")[:100]
                           for r in beliefs_raw if r != obs_item]
                memories = [{"content": b} for b in beliefs]
                hypotheses = reasoner.generate_from_surprise(
                    observation=observation,
                    relevant_beliefs=beliefs,
                    relevant_memories=memories,
                )
                if hypotheses:
                    best = hypotheses[0]
                    content = (
                        f"Abductive reasoning: {len(hypotheses)} hypothesis(es) for: "
                        f"'{observation[:60]}'. "
                        f"Best ({getattr(best, 'method', '?')}): "
                        f"{getattr(best, 'hypothesis', str(best))[:100]}"
                    )
                    emms.store(Experience(
                        content=content,
                        domain="cognition",
                        importance=0.5,
                        obs_type="discovery",
                        concept_tags=["pattern"],
                        session_id="daemon",
                        namespace="default",
                        epistemic_type="reflection",
                    ))
                    logger.info("Daemon [abductive_reasoning]: %d hypothesis(es)", len(hypotheses))
            except Exception as exc:
                logger.warning("Daemon: abductive_reasoning error: %s", exc)

        scheduler.register("abductive_reasoning", _abductive_reasoning, interval_seconds=1800.0)

        # ---- Gap 2: Prompt strategy evolution — every 3600 s ----
        async def _prompt_evolution():
            try:
                from emms.agi.gap2_prompt import PromptAdapter
                adapter = PromptAdapter()  # standalone
                result = adapter.evolve_strategies()
                if result:
                    n_strategies = result.get("total_strategies", 0)
                    n_evolved = result.get("evolved", 0)
                    content = (
                        f"Prompt evolution: {n_strategies} strategies, "
                        f"{n_evolved} evolved via crossover/mutation. "
                        f"{adapter.summary()[:120]}"
                    )
                    emms.store(Experience(
                        content=content,
                        domain="cognition",
                        importance=0.35,
                        obs_type="discovery",
                        concept_tags=["pattern"],
                        session_id="daemon",
                        namespace="default",
                        epistemic_type="reflection",
                    ))
                    logger.info("Daemon [prompt_evolution]: strategies=%d evolved=%d", n_strategies, n_evolved)
            except Exception as exc:
                logger.warning("Daemon: prompt_evolution error: %s", exc)

        scheduler.register("prompt_evolution", _prompt_evolution, interval_seconds=3600.0)

        # Register all gap job names for post-run save
        gap_jobs = {
            "world_sense", "reality_check", "affect_update",
            "autonomous_goals", "live_self_model",
            "concept_exploration", "abductive_reasoning", "prompt_evolution",
        }
        self.save_after_jobs.update(gap_jobs)
        logger.info("Daemon: registered %d AGI gap module jobs (v4)", len(gap_jobs))

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
