"""CognitiveLoop — Event-driven autonomous reasoning cycle.

Gap 1 in the EMMS → AGI roadmap: replaces cron-based daemon scheduling
with event-triggered cognitive processing. Instead of "run prediction_loop
every 600s", the loop fires when relevant events occur.

Components:
- CognitiveTask: a unit of cognitive work with priority
- AttentionAllocator: decides what to think about
- WorkingScratchpad: persistent multi-step reasoning state
- CognitiveLoop: the main event-driven loop

The loop integrates with the existing EventBus (sync) and MemoryScheduler
but adds event-driven triggering on top.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ── Cognitive Task ───────────────────────────────────────────────────────────


class TaskType(str, Enum):
    """Types of cognitive tasks the loop can process."""

    PREDICTION = "prediction"  # generate/resolve predictions
    BELIEF_REVISION = "belief_revision"  # update beliefs from surprise
    CONSOLIDATION = "consolidation"  # memory tier promotion
    REFLECTION = "reflection"  # extract lessons from experience
    CURIOSITY = "curiosity"  # explore knowledge gaps
    PATTERN_DETECTION = "pattern_detection"  # find recurring patterns
    GOAL_CHECK = "goal_check"  # check goal progress
    SELF_MODEL_UPDATE = "self_model_update"  # update live self-model
    EMOTION_REGULATION = "emotion_regulation"  # process emotional state
    REASONING_CHAIN = "reasoning_chain"  # multi-step reasoning
    CUSTOM = "custom"


@dataclass
class CognitiveTask:
    """A unit of cognitive work with priority and context."""

    id: str = ""
    task_type: TaskType = TaskType.CUSTOM
    description: str = ""
    priority: float = 0.5  # 0 = lowest, 1 = highest
    domain: str = "general"
    context: dict = field(default_factory=dict)
    source_event: str = ""  # which event triggered this
    created_at: float = 0.0
    deadline: float = 0.0  # optional deadline (0 = none)
    chain_id: str = ""  # links to a ReasoningChain in scratchpad
    max_retries: int = 1
    retry_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"ctask_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def __lt__(self, other):
        """Higher priority comes first (for heapq min-heap, negate)."""
        return self.priority > other.priority

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveTask":
        d = dict(d)
        if "task_type" in d and isinstance(d["task_type"], str):
            d["task_type"] = TaskType(d["task_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskResult:
    """Result of processing a cognitive task."""

    task_id: str = ""
    success: bool = True
    output: str = ""
    follow_up_tasks: list[CognitiveTask] = field(default_factory=list)
    state_changes: dict = field(default_factory=dict)  # what changed
    processing_time_ms: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


# ── Attention Allocator ──────────────────────────────────────────────────────


class AttentionAllocator:
    """Decides what to think about based on urgency, importance, novelty, and staleness.

    Computes a priority score for each incoming event/task using:
    - urgency: time-sensitivity (deadlines, prediction timeouts)
    - importance: how much this matters (from experience importance)
    - novelty: how surprising (from ego surprise score)
    - staleness: how long since this domain was attended to
    """

    def __init__(
        self,
        urgency_weight: float = 0.3,
        importance_weight: float = 0.3,
        novelty_weight: float = 0.25,
        staleness_weight: float = 0.15,
    ):
        self.urgency_weight = urgency_weight
        self.importance_weight = importance_weight
        self.novelty_weight = novelty_weight
        self.staleness_weight = staleness_weight

        # Track when each domain was last attended to
        self._domain_last_attended: dict[str, float] = defaultdict(float)
        self._total_tasks_scored: int = 0

    def prioritize(self, task: CognitiveTask) -> float:
        """Compute priority score for a task. Returns 0-1."""
        urgency = self._compute_urgency(task)
        importance = self._compute_importance(task)
        novelty = self._compute_novelty(task)
        staleness = self._compute_staleness(task)

        score = (
            self.urgency_weight * urgency
            + self.importance_weight * importance
            + self.novelty_weight * novelty
            + self.staleness_weight * staleness
        )
        score = max(0.0, min(1.0, score))
        self._total_tasks_scored += 1
        return score

    def record_attention(self, domain: str):
        """Record that we attended to a domain."""
        self._domain_last_attended[domain] = time.time()

    def _compute_urgency(self, task: CognitiveTask) -> float:
        """Time-sensitivity score. Higher for approaching deadlines."""
        if task.deadline <= 0:
            # No deadline: moderate urgency based on task type
            type_urgency = {
                TaskType.BELIEF_REVISION: 0.8,  # surprises need fast response
                TaskType.EMOTION_REGULATION: 0.7,
                TaskType.PREDICTION: 0.5,
                TaskType.GOAL_CHECK: 0.4,
                TaskType.CONSOLIDATION: 0.3,
                TaskType.REFLECTION: 0.2,
                TaskType.CURIOSITY: 0.1,
            }
            return type_urgency.get(task.task_type, 0.3)

        # Has deadline: urgency increases as deadline approaches
        time_left = task.deadline - time.time()
        if time_left <= 0:
            return 1.0  # overdue
        # Sigmoid: urgency spikes in last 20% of time
        total_time = task.deadline - task.created_at
        if total_time <= 0:
            return 1.0
        fraction_remaining = time_left / total_time
        return 1.0 / (1.0 + math.exp(5.0 * (fraction_remaining - 0.3)))

    def _compute_importance(self, task: CognitiveTask) -> float:
        """Importance from task context."""
        return task.context.get("importance", 0.5)

    def _compute_novelty(self, task: CognitiveTask) -> float:
        """How surprising/novel is this task."""
        return task.context.get("surprise", task.context.get("novelty", 0.3))

    def _compute_staleness(self, task: CognitiveTask) -> float:
        """How long since we attended to this domain. Stale domains get priority."""
        last = self._domain_last_attended.get(task.domain, 0.0)
        if last == 0:
            return 0.7  # never attended = moderately stale
        hours_since = (time.time() - last) / 3600
        # Logarithmic staleness: diminishing returns after ~24h
        return min(1.0, math.log(1 + hours_since) / math.log(1 + 24))

    def save_state(self) -> dict:
        return {
            "domain_last_attended": dict(self._domain_last_attended),
            "total_tasks_scored": self._total_tasks_scored,
        }

    def load_state(self, data: dict):
        self._domain_last_attended = defaultdict(
            float, data.get("domain_last_attended", {})
        )
        self._total_tasks_scored = data.get("total_tasks_scored", 0)


# ── Working Scratchpad ───────────────────────────────────────────────────────


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    index: int = 0
    action: str = ""  # what was done
    result: str = ""  # what was found
    confidence: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ReasoningChain:
    """A multi-step reasoning process that persists across loop iterations.

    Unlike working memory (deque of 7±2 items cleared between sessions),
    reasoning chains track ongoing multi-step thoughts that can be
    resumed after interruption.
    """

    id: str = ""
    goal: str = ""
    domain: str = "general"
    steps: list[ReasoningStep] = field(default_factory=list)
    current_hypothesis: str = ""
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "active"  # active, paused, completed, abandoned
    created_at: float = 0.0
    last_updated: float = 0.0
    parent_chain_id: str = ""  # for sub-chains

    def __post_init__(self):
        if not self.id:
            self.id = f"chain_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_updated:
            self.last_updated = time.time()

    def add_step(self, action: str, result: str, confidence: float = 0.5):
        """Add a reasoning step."""
        step = ReasoningStep(
            index=len(self.steps),
            action=action,
            result=result,
            confidence=confidence,
        )
        self.steps.append(step)
        self.last_updated = time.time()
        # Update chain confidence as weighted average
        if self.steps:
            self.confidence = sum(s.confidence for s in self.steps) / len(self.steps)

    def add_evidence(self, evidence: str, supports: bool = True):
        """Add evidence for or against current hypothesis."""
        if supports:
            self.evidence_for.append(evidence)
        else:
            self.evidence_against.append(evidence)
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "domain": self.domain,
            "steps": [asdict(s) for s in self.steps],
            "current_hypothesis": self.current_hypothesis,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "parent_chain_id": self.parent_chain_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningChain":
        d = dict(d)
        steps_data = d.pop("steps", [])
        chain = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        chain.steps = [ReasoningStep(**s) for s in steps_data]
        return chain


class WorkingScratchpad:
    """Persistent multi-step reasoning state across cognitive loop iterations.

    Unlike the HierarchicalMemory working tier (deque of 7±2 items),
    the scratchpad maintains active reasoning chains that can span
    multiple loop iterations and even sessions.
    """

    def __init__(self, max_active_chains: int = 10, max_completed: int = 50):
        self.chains: dict[str, ReasoningChain] = {}
        self.max_active = max_active_chains
        self.max_completed = max_completed
        self._completed: list[ReasoningChain] = []

    def start_chain(self, goal: str, domain: str = "general",
                    hypothesis: str = "", parent_chain_id: str = "") -> ReasoningChain:
        """Start a new reasoning chain."""
        chain = ReasoningChain(
            goal=goal,
            domain=domain,
            current_hypothesis=hypothesis,
            parent_chain_id=parent_chain_id,
        )
        self.chains[chain.id] = chain

        # Prune if over capacity: pause oldest active chain
        active = [c for c in self.chains.values() if c.status == "active"]
        if len(active) > self.max_active:
            oldest = min(active, key=lambda c: c.last_updated)
            oldest.status = "paused"

        return chain

    def resume_chain(self, chain_id: str) -> ReasoningChain | None:
        """Resume a paused or active reasoning chain."""
        chain = self.chains.get(chain_id)
        if chain and chain.status in ("active", "paused"):
            chain.status = "active"
            chain.last_updated = time.time()
            return chain
        return None

    def complete_chain(self, chain_id: str, conclusion: str = ""):
        """Mark a chain as completed."""
        chain = self.chains.get(chain_id)
        if chain:
            chain.status = "completed"
            if conclusion:
                chain.add_step("conclusion", conclusion, chain.confidence)
            chain.last_updated = time.time()
            # Move to completed archive
            self._completed.append(chain)
            del self.chains[chain_id]
            if len(self._completed) > self.max_completed:
                self._completed = self._completed[-self.max_completed:]

    def abandon_chain(self, chain_id: str, reason: str = ""):
        """Abandon a chain that's no longer productive."""
        chain = self.chains.get(chain_id)
        if chain:
            chain.status = "abandoned"
            if reason:
                chain.add_step("abandoned", reason, 0.0)
            self._completed.append(chain)
            del self.chains[chain_id]

    def spawn_subchain(self, parent_id: str, subgoal: str,
                       domain: str = "") -> ReasoningChain | None:
        """Create a sub-chain decomposing a step of the parent."""
        parent = self.chains.get(parent_id)
        if not parent:
            return None
        return self.start_chain(
            goal=subgoal,
            domain=domain or parent.domain,
            parent_chain_id=parent_id,
        )

    def active_chains(self) -> list[ReasoningChain]:
        """Return all active reasoning chains sorted by recency."""
        active = [c for c in self.chains.values() if c.status == "active"]
        return sorted(active, key=lambda c: c.last_updated, reverse=True)

    def chain_for_domain(self, domain: str) -> ReasoningChain | None:
        """Find the most recent active chain for a domain."""
        for chain in self.active_chains():
            if chain.domain == domain:
                return chain
        return None

    def summary(self) -> str:
        """Human-readable scratchpad summary."""
        active = self.active_chains()
        paused = [c for c in self.chains.values() if c.status == "paused"]
        lines = [
            f"WorkingScratchpad — {len(active)} active, {len(paused)} paused, "
            f"{len(self._completed)} completed",
        ]
        for chain in active[:5]:
            steps = len(chain.steps)
            lines.append(
                f"  [{chain.domain}] {chain.goal[:60]} "
                f"({steps} steps, conf={chain.confidence:.2f})"
            )
        return "\n".join(lines)

    def save_state(self, path: str | Path):
        """Persist scratchpad to JSON."""
        data = {
            "chains": {k: v.to_dict() for k, v in self.chains.items()},
            "completed": [c.to_dict() for c in self._completed[-20:]],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str), encoding="utf-8")

    def load_state(self, path: str | Path) -> bool:
        """Load scratchpad from JSON."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.chains = {
                k: ReasoningChain.from_dict(v)
                for k, v in data.get("chains", {}).items()
            }
            self._completed = [
                ReasoningChain.from_dict(c) for c in data.get("completed", [])
            ]
            return True
        except Exception:
            return False


# ── Event-to-Task Router ─────────────────────────────────────────────────────


class EventTaskRouter:
    """Maps EMMS events to cognitive tasks.

    When the EventBus fires an event, the router creates appropriate
    CognitiveTask(s) and adds them to the loop's queue. This is the
    bridge between the sync EventBus and the async CognitiveLoop.
    """

    def __init__(self, attention: AttentionAllocator):
        self.attention = attention
        self._routes: dict[str, list[Callable]] = defaultdict(list)
        self._register_default_routes()

    def _register_default_routes(self):
        """Register default event→task mappings."""
        self.route("memory.stored", self._on_memory_stored)
        self.route("memory.patterns_detected", self._on_patterns_detected)
        self.route("prediction.resolved", self._on_prediction_resolved)
        self.route("surprise.threshold_exceeded", self._on_surprise)
        self.route("goal.pushed", self._on_goal_pushed)
        self.route("emotion.state_changed", self._on_emotion_change)
        self.route("identity.updated", self._on_identity_updated)

    def route(self, event_name: str, handler: Callable):
        """Register an event→task handler."""
        self._routes[event_name].append(handler)

    def handle_event(self, event_name: str, data: Any = None) -> list[CognitiveTask]:
        """Convert an event into cognitive tasks."""
        tasks = []
        for handler in self._routes.get(event_name, []):
            try:
                result = handler(data)
                if isinstance(result, list):
                    tasks.extend(result)
                elif isinstance(result, CognitiveTask):
                    tasks.append(result)
            except Exception as e:
                logger.warning("Event handler error for %s: %s", event_name, e)
        # Score all tasks
        for task in tasks:
            task.priority = self.attention.prioritize(task)
        return tasks

    # ── Default handlers ─────────────────────────────────────────────

    def _on_memory_stored(self, data: dict | None) -> list[CognitiveTask]:
        """New memory stored → maybe update self-model, check patterns."""
        if not data:
            return []
        tasks = []
        domain = data.get("domain", "general")
        importance = data.get("importance", 0.5)

        # High-importance memories trigger self-model update
        if importance > 0.7:
            tasks.append(CognitiveTask(
                task_type=TaskType.SELF_MODEL_UPDATE,
                description=f"Update self-model for high-importance {domain} memory",
                domain=domain,
                context={"importance": importance, "memory_id": data.get("memory_id")},
                source_event="memory.stored",
            ))

        return tasks

    def _on_patterns_detected(self, data: dict | None) -> CognitiveTask:
        """Patterns detected → trigger reflection."""
        return CognitiveTask(
            task_type=TaskType.REFLECTION,
            description="Reflect on detected patterns",
            context={"patterns": data, "importance": 0.6},
            source_event="memory.patterns_detected",
        )

    def _on_prediction_resolved(self, data: dict | None) -> list[CognitiveTask]:
        """Prediction resolved → check surprise, maybe revise beliefs."""
        if not data:
            return []
        tasks = []
        surprise = data.get("surprise_score", 0.0)
        domain = data.get("domain", "general")

        if surprise > 0.3:
            tasks.append(CognitiveTask(
                task_type=TaskType.BELIEF_REVISION,
                description=f"Revise beliefs — surprising {data.get('outcome', 'outcome')} in {domain}",
                domain=domain,
                context={"surprise": surprise, "importance": 0.8, **data},
                source_event="prediction.resolved",
            ))

        return tasks

    def _on_surprise(self, data: dict | None) -> CognitiveTask:
        """High surprise → urgent belief revision."""
        return CognitiveTask(
            task_type=TaskType.BELIEF_REVISION,
            description="Urgent belief revision — surprise threshold exceeded",
            context={"importance": 0.9, "surprise": 1.0, **(data or {})},
            source_event="surprise.threshold_exceeded",
        )

    def _on_goal_pushed(self, data: dict | None) -> CognitiveTask:
        """New goal pushed → check if it needs immediate action."""
        return CognitiveTask(
            task_type=TaskType.GOAL_CHECK,
            description="Check new goal for immediate action",
            domain=data.get("domain", "general") if data else "general",
            context={"importance": 0.6, **(data or {})},
            source_event="goal.pushed",
        )

    def _on_emotion_change(self, data: dict | None) -> CognitiveTask:
        """Emotional state changed → maybe regulate."""
        return CognitiveTask(
            task_type=TaskType.EMOTION_REGULATION,
            description="Process emotional state change",
            context={"importance": 0.5, **(data or {})},
            source_event="emotion.state_changed",
        )

    def _on_identity_updated(self, data: dict | None) -> CognitiveTask:
        """Identity updated → update self-model."""
        return CognitiveTask(
            task_type=TaskType.SELF_MODEL_UPDATE,
            description="Sync self-model with identity update",
            context={"importance": 0.6, **(data or {})},
            source_event="identity.updated",
        )


# ── Cognitive Loop ───────────────────────────────────────────────────────────


class CognitiveLoop:
    """Event-driven autonomous reasoning cycle.

    Replaces the cron-based MemoryScheduler with an event-triggered loop.
    When events arrive (via EventBus), the router converts them to
    CognitiveTask objects, the attention allocator prioritizes them,
    and the loop processes the highest-priority task.

    The loop also maintains time-based background tasks (consolidation,
    dreaming) that run when the event queue is empty, preserving the
    daemon's existing functionality.

    Usage:
        loop = CognitiveLoop(emms)
        loop.register_handler(TaskType.PREDICTION, my_prediction_handler)
        await loop.run()  # or loop.run_sync() for non-async contexts
    """

    def __init__(
        self,
        emms: Any = None,
        min_delay: float = 0.1,  # fastest processing (100ms)
        max_delay: float = 30.0,  # slowest (when idle)
        idle_delay: float = 5.0,  # delay when queue empty
    ):
        self.emms = emms
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.idle_delay = idle_delay

        # Core components
        self.attention = AttentionAllocator()
        self.scratchpad = WorkingScratchpad()
        self.router = EventTaskRouter(self.attention)

        # Task queue (sorted by priority)
        self._queue: list[CognitiveTask] = []
        self._handlers: dict[TaskType, Callable] = {}
        self._running = False

        # Metrics
        self._tasks_processed: int = 0
        self._total_processing_ms: float = 0.0
        self._events_received: int = 0
        self._last_processed_at: float = 0.0

        # Background task intervals (preserve daemon functionality)
        self._background_intervals: dict[str, float] = {
            "consolidation": 120,
            "pattern_detection": 300,
            "reflection": 1800,
            "dream": 3600,
        }
        self._background_last_run: dict[str, float] = defaultdict(float)

    # ── Handler Registration ─────────────────────────────────────────

    def register_handler(
        self, task_type: TaskType, handler: Callable[[CognitiveTask, Any], TaskResult]
    ):
        """Register a handler for a task type.

        Handler signature: (task: CognitiveTask, emms: EMMS) -> TaskResult
        """
        self._handlers[task_type] = handler

    # ── Event Integration ────────────────────────────────────────────

    def on_event(self, event_name: str, data: Any = None):
        """Called by EventBus when an event fires. Sync entry point.

        Wire this to the EMMS EventBus:
            emms.events.on("memory.stored", loop.on_event)
        """
        self._events_received += 1
        tasks = self.router.handle_event(event_name, data)
        for task in tasks:
            self._enqueue(task)

    def push_task(self, task: CognitiveTask):
        """Manually push a task into the queue."""
        task.priority = self.attention.prioritize(task)
        self._enqueue(task)

    def _enqueue(self, task: CognitiveTask):
        """Add task to priority queue."""
        self._queue.append(task)
        self._queue.sort(key=lambda t: t.priority, reverse=True)
        # Cap queue size
        if len(self._queue) > 100:
            self._queue = self._queue[:100]

    # ── Main Loop ────────────────────────────────────────────────────

    async def run(self):
        """Main async cognitive loop. Run until stopped."""
        self._running = True
        logger.info("CognitiveLoop started")

        while self._running:
            processed = False

            # 1. Process highest-priority task from queue
            if self._queue:
                task = self._queue.pop(0)
                result = await self._process_task(task)
                processed = True

                # Follow-up tasks
                for follow_up in result.follow_up_tasks:
                    follow_up.priority = self.attention.prioritize(follow_up)
                    self._enqueue(follow_up)

            # 2. Check background tasks (when queue is empty)
            if not self._queue:
                bg_task = self._check_background_tasks()
                if bg_task:
                    self._enqueue(bg_task)

            # 3. Adaptive delay
            delay = self._adaptive_delay(processed)
            await asyncio.sleep(delay)

    def stop(self):
        """Stop the cognitive loop."""
        self._running = False
        logger.info("CognitiveLoop stopping")

    async def _process_task(self, task: CognitiveTask) -> TaskResult:
        """Process a single cognitive task."""
        t0 = time.time()
        logger.debug("Processing task: %s (%s)", task.description, task.task_type.value)

        handler = self._handlers.get(task.task_type)
        if not handler:
            # No handler registered — log and skip
            return TaskResult(
                task_id=task.id,
                success=False,
                output=f"No handler for {task.task_type.value}",
            )

        try:
            result = handler(task, self.emms)
            # Handle both sync and async handlers
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            logger.warning("Task processing error: %s", e)
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.priority *= 0.8  # lower priority on retry
                self._enqueue(task)
            result = TaskResult(
                task_id=task.id,
                success=False,
                output=str(e),
            )

        elapsed_ms = (time.time() - t0) * 1000
        result.processing_time_ms = elapsed_ms
        result.task_id = task.id

        # Update metrics
        self._tasks_processed += 1
        self._total_processing_ms += elapsed_ms
        self._last_processed_at = time.time()
        self.attention.record_attention(task.domain)

        return result

    def _check_background_tasks(self) -> CognitiveTask | None:
        """Check if any background task is due. Returns most overdue."""
        now = time.time()
        most_overdue = None
        most_overdue_ratio = 0.0

        for name, interval in self._background_intervals.items():
            last = self._background_last_run.get(name, 0.0)
            elapsed = now - last
            if elapsed >= interval:
                overdue_ratio = elapsed / max(interval, 0.001)
                if overdue_ratio > most_overdue_ratio:
                    most_overdue_ratio = overdue_ratio
                    task_type_map = {
                        "consolidation": TaskType.CONSOLIDATION,
                        "pattern_detection": TaskType.PATTERN_DETECTION,
                        "reflection": TaskType.REFLECTION,
                        "dream": TaskType.CONSOLIDATION,
                    }
                    most_overdue = CognitiveTask(
                        task_type=task_type_map.get(name, TaskType.CUSTOM),
                        description=f"Background: {name}",
                        context={"importance": 0.3, "background": True},
                        source_event=f"background.{name}",
                    )
                    self._background_last_run[name] = now

        return most_overdue

    def _adaptive_delay(self, just_processed: bool) -> float:
        """Compute delay between loop iterations.

        Think faster when there's more to process, slower when idle.
        """
        if self._queue:
            # Tasks waiting — process quickly
            queue_pressure = min(1.0, len(self._queue) / 10.0)
            return self.min_delay + (1.0 - queue_pressure) * (self.idle_delay - self.min_delay)
        elif just_processed:
            # Just finished — short cooldown
            return self.min_delay * 5
        else:
            # Idle — slow down
            return self.idle_delay

    # ── Sync Interface ───────────────────────────────────────────────

    def process_one(self) -> TaskResult | None:
        """Process the top task synchronously. For testing and non-async contexts."""
        if not self._queue:
            return None
        task = self._queue.pop(0)
        handler = self._handlers.get(task.task_type)
        if not handler:
            return TaskResult(task_id=task.id, success=False, output="No handler")
        t0 = time.time()
        try:
            result = handler(task, self.emms)
            if asyncio.iscoroutine(result):
                # Can't await in sync context — wrap
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
        except Exception as e:
            result = TaskResult(task_id=task.id, success=False, output=str(e))
        elapsed_ms = (time.time() - t0) * 1000
        result.processing_time_ms = elapsed_ms
        result.task_id = task.id
        self._tasks_processed += 1
        self._total_processing_ms += elapsed_ms
        self._last_processed_at = time.time()
        self.attention.record_attention(task.domain)
        for follow_up in result.follow_up_tasks:
            follow_up.priority = self.attention.prioritize(follow_up)
            self._enqueue(follow_up)
        return result

    # ── Query Methods ────────────────────────────────────────────────

    def queue_size(self) -> int:
        return len(self._queue)

    def pending_tasks(self) -> list[CognitiveTask]:
        return list(self._queue)

    def metrics(self) -> dict[str, Any]:
        avg_ms = (
            self._total_processing_ms / self._tasks_processed
            if self._tasks_processed > 0
            else 0
        )
        return {
            "tasks_processed": self._tasks_processed,
            "events_received": self._events_received,
            "queue_size": len(self._queue),
            "active_chains": len(self.scratchpad.active_chains()),
            "avg_processing_ms": round(avg_ms, 2),
            "attention_domains": dict(self.attention._domain_last_attended),
        }

    def summary(self) -> str:
        m = self.metrics()
        lines = [
            f"CognitiveLoop — {m['tasks_processed']} tasks processed, "
            f"{m['events_received']} events received",
            f"Queue: {m['queue_size']} pending",
            f"Active chains: {m['active_chains']}",
            f"Avg processing: {m['avg_processing_ms']}ms",
        ]
        if self.scratchpad.active_chains():
            lines.append("")
            lines.append(self.scratchpad.summary())
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def save_state(self, path: str | Path):
        """Save cognitive loop state."""
        p = Path(path)
        data = {
            "queue": [t.to_dict() for t in self._queue],
            "attention": self.attention.save_state(),
            "metrics": {
                "tasks_processed": self._tasks_processed,
                "total_processing_ms": self._total_processing_ms,
                "events_received": self._events_received,
                "last_processed_at": self._last_processed_at,
            },
            "background_last_run": dict(self._background_last_run),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str), encoding="utf-8")

        # Save scratchpad separately
        scratch_path = p.parent / (p.stem + "_scratchpad.json")
        self.scratchpad.save_state(scratch_path)

    def load_state(self, path: str | Path) -> bool:
        """Load cognitive loop state."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._queue = [CognitiveTask.from_dict(t) for t in data.get("queue", [])]
            self.attention.load_state(data.get("attention", {}))
            metrics = data.get("metrics", {})
            self._tasks_processed = metrics.get("tasks_processed", 0)
            self._total_processing_ms = metrics.get("total_processing_ms", 0)
            self._events_received = metrics.get("events_received", 0)
            self._last_processed_at = metrics.get("last_processed_at", 0)
            self._background_last_run = defaultdict(
                float, data.get("background_last_run", {})
            )

            # Load scratchpad
            scratch_path = p.parent / (p.stem + "_scratchpad.json")
            self.scratchpad.load_state(scratch_path)

            return True
        except Exception:
            return False
