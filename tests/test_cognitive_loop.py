"""Tests for CognitiveLoop (Gap 1: AGI Roadmap).

Covers:
- CognitiveTask (creation, priority, serialization)
- AttentionAllocator (urgency, importance, novelty, staleness)
- WorkingScratchpad (chains, steps, sub-chains, persistence)
- EventTaskRouter (event → task mapping)
- CognitiveLoop (queue, process_one, metrics, persistence)
- EMMS integration (public wrappers, EventBus wiring, save/load)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from emms import EMMS, Experience
from emms.daemon.cognitive_loop import (
    AttentionAllocator,
    CognitiveLoop,
    CognitiveTask,
    EventTaskRouter,
    ReasoningChain,
    ReasoningStep,
    TaskResult,
    TaskType,
    WorkingScratchpad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _echo_handler(task: CognitiveTask, emms=None) -> TaskResult:
    """Simple handler that echoes the task description."""
    return TaskResult(
        task_id=task.id,
        success=True,
        output=f"Processed: {task.description}",
    )


def _chain_handler(task: CognitiveTask, emms=None) -> TaskResult:
    """Handler that produces a follow-up task."""
    follow_up = CognitiveTask(
        task_type=TaskType.REFLECTION,
        description=f"Follow-up to: {task.description}",
        domain=task.domain,
        context={"importance": 0.4},
    )
    return TaskResult(
        task_id=task.id,
        success=True,
        output="Done with follow-up",
        follow_up_tasks=[follow_up],
    )


# ---------------------------------------------------------------------------
# CognitiveTask
# ---------------------------------------------------------------------------


class TestCognitiveTask:
    def test_creation(self):
        t = CognitiveTask(task_type=TaskType.PREDICTION, description="test")
        assert t.id.startswith("ctask_")
        assert t.task_type == TaskType.PREDICTION

    def test_priority_ordering(self):
        high = CognitiveTask(priority=0.9)
        low = CognitiveTask(priority=0.1)
        assert high < low  # higher priority = "less than" for sorting

    def test_serialization_roundtrip(self):
        t = CognitiveTask(
            task_type=TaskType.BELIEF_REVISION,
            description="revise beliefs",
            domain="finance",
            context={"surprise": 0.8},
        )
        d = t.to_dict()
        t2 = CognitiveTask.from_dict(d)
        assert t2.task_type == TaskType.BELIEF_REVISION
        assert t2.description == t.description
        assert t2.context["surprise"] == 0.8


# ---------------------------------------------------------------------------
# AttentionAllocator
# ---------------------------------------------------------------------------


class TestAttentionAllocator:
    def test_prioritize_returns_0_to_1(self):
        alloc = AttentionAllocator()
        task = CognitiveTask(task_type=TaskType.PREDICTION, domain="finance")
        score = alloc.prioritize(task)
        assert 0.0 <= score <= 1.0

    def test_belief_revision_more_urgent_than_curiosity(self):
        alloc = AttentionAllocator()
        urgent = CognitiveTask(task_type=TaskType.BELIEF_REVISION,
                               context={"importance": 0.5, "surprise": 0.5})
        slow = CognitiveTask(task_type=TaskType.CURIOSITY,
                             context={"importance": 0.5})
        assert alloc.prioritize(urgent) > alloc.prioritize(slow)

    def test_deadline_increases_urgency(self):
        alloc = AttentionAllocator()
        no_deadline = CognitiveTask(task_type=TaskType.CURIOSITY,
                                    context={"importance": 0.3})
        with_deadline = CognitiveTask(
            task_type=TaskType.CURIOSITY,
            created_at=time.time() - 100,
            deadline=time.time() + 1,  # 1 second from now, created 100s ago
            context={"importance": 0.3},
        )
        # Very close deadline should boost priority
        score_no = alloc.prioritize(no_deadline)
        score_yes = alloc.prioritize(with_deadline)
        assert score_yes > score_no

    def test_staleness_increases_priority(self):
        alloc = AttentionAllocator()
        # Attend to domain A, not B
        alloc.record_attention("domain_a")
        task_a = CognitiveTask(domain="domain_a", context={"importance": 0.5})
        task_b = CognitiveTask(domain="domain_b", context={"importance": 0.5})
        # B should get higher staleness score
        score_a = alloc.prioritize(task_a)
        score_b = alloc.prioritize(task_b)
        assert score_b > score_a

    def test_save_load_state(self):
        alloc = AttentionAllocator()
        alloc.record_attention("finance")
        state = alloc.save_state()
        alloc2 = AttentionAllocator()
        alloc2.load_state(state)
        assert "finance" in alloc2._domain_last_attended


# ---------------------------------------------------------------------------
# WorkingScratchpad
# ---------------------------------------------------------------------------


class TestWorkingScratchpad:
    def test_start_chain(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("Analyze ETH performance", "crypto")
        assert chain.id.startswith("chain_")
        assert chain.status == "active"
        assert chain.id in pad.chains

    def test_add_step(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("test goal")
        chain.add_step("gather data", "found 100 records", 0.8)
        assert len(chain.steps) == 1
        assert chain.steps[0].action == "gather data"

    def test_chain_confidence_updates(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("test")
        chain.add_step("step1", "result1", 0.9)
        chain.add_step("step2", "result2", 0.5)
        assert abs(chain.confidence - 0.7) < 0.01  # average

    def test_add_evidence(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("test")
        chain.add_evidence("supports hypothesis", supports=True)
        chain.add_evidence("contradicts", supports=False)
        assert len(chain.evidence_for) == 1
        assert len(chain.evidence_against) == 1

    def test_complete_chain(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("test")
        chain_id = chain.id
        pad.complete_chain(chain_id, "Conclusion: X is true")
        assert chain_id not in pad.chains
        assert len(pad._completed) == 1
        assert pad._completed[0].status == "completed"

    def test_abandon_chain(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("test")
        pad.abandon_chain(chain.id, "Not productive")
        assert chain.id not in pad.chains

    def test_spawn_subchain(self):
        pad = WorkingScratchpad()
        parent = pad.start_chain("main goal", "finance")
        sub = pad.spawn_subchain(parent.id, "sub-goal: verify data")
        assert sub is not None
        assert sub.parent_chain_id == parent.id
        assert sub.domain == "finance"

    def test_capacity_pruning(self):
        pad = WorkingScratchpad(max_active_chains=3)
        chains = []
        for i in range(5):
            c = pad.start_chain(f"chain {i}")
            chains.append(c)
            time.sleep(0.01)  # ensure ordering
        active = pad.active_chains()
        paused = [c for c in pad.chains.values() if c.status == "paused"]
        assert len(active) <= 3
        assert len(paused) >= 2

    def test_chain_for_domain(self):
        pad = WorkingScratchpad()
        pad.start_chain("finance thing", "finance")
        pad.start_chain("ml thing", "ml")
        chain = pad.chain_for_domain("finance")
        assert chain is not None
        assert chain.domain == "finance"

    def test_summary(self):
        pad = WorkingScratchpad()
        pad.start_chain("test goal", "test")
        summary = pad.summary()
        assert "1 active" in summary

    def test_persistence_roundtrip(self):
        pad = WorkingScratchpad()
        chain = pad.start_chain("persist me", "test")
        chain.add_step("step1", "result1", 0.8)
        pad.complete_chain(chain.id, "done")
        pad.start_chain("still active")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scratchpad.json"
            pad.save_state(path)

            pad2 = WorkingScratchpad()
            loaded = pad2.load_state(path)
            assert loaded
            assert len(pad2.chains) == 1  # active chain
            assert len(pad2._completed) == 1  # completed chain


# ---------------------------------------------------------------------------
# EventTaskRouter
# ---------------------------------------------------------------------------


class TestEventTaskRouter:
    def test_memory_stored_high_importance(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        tasks = router.handle_event("memory.stored", {
            "domain": "finance", "importance": 0.9, "memory_id": "mem_001",
        })
        assert len(tasks) >= 1
        assert any(t.task_type == TaskType.SELF_MODEL_UPDATE for t in tasks)

    def test_memory_stored_low_importance(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        tasks = router.handle_event("memory.stored", {
            "domain": "general", "importance": 0.3,
        })
        assert len(tasks) == 0  # low importance doesn't trigger

    def test_prediction_resolved_surprise(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        tasks = router.handle_event("prediction.resolved", {
            "surprise_score": 0.8, "domain": "finance", "outcome": "violated",
        })
        assert len(tasks) >= 1
        assert tasks[0].task_type == TaskType.BELIEF_REVISION

    def test_surprise_threshold(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        tasks = router.handle_event("surprise.threshold_exceeded", {"domain": "ml"})
        assert len(tasks) == 1
        assert tasks[0].context["importance"] == 0.9

    def test_unknown_event(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        tasks = router.handle_event("unknown.event", {})
        assert len(tasks) == 0

    def test_custom_route(self):
        alloc = AttentionAllocator()
        router = EventTaskRouter(alloc)
        router.route("custom.event", lambda data: CognitiveTask(
            task_type=TaskType.CUSTOM, description="custom handler",
            context={"importance": 0.5},
        ))
        tasks = router.handle_event("custom.event", {})
        assert len(tasks) == 1


# ---------------------------------------------------------------------------
# CognitiveLoop
# ---------------------------------------------------------------------------


class TestCognitiveLoop:
    def test_push_and_process(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.CUSTOM, _echo_handler)
        task = CognitiveTask(task_type=TaskType.CUSTOM, description="hello")
        loop.push_task(task)
        assert loop.queue_size() == 1

        result = loop.process_one()
        assert result is not None
        assert result.success
        assert "hello" in result.output
        assert loop.queue_size() == 0

    def test_priority_ordering(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.BELIEF_REVISION, _echo_handler)
        loop.register_handler(TaskType.CURIOSITY, _echo_handler)
        low = CognitiveTask(task_type=TaskType.CURIOSITY, description="low",
                           context={"importance": 0.1})
        high = CognitiveTask(task_type=TaskType.BELIEF_REVISION, description="high",
                            context={"importance": 0.9, "surprise": 0.9})
        loop.push_task(low)
        loop.push_task(high)

        # High priority should be processed first
        result = loop.process_one()
        assert "high" in result.output

    def test_follow_up_tasks(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.CUSTOM, _chain_handler)
        loop.register_handler(TaskType.REFLECTION, _echo_handler)
        task = CognitiveTask(task_type=TaskType.CUSTOM, description="starter")
        loop.push_task(task)

        loop.process_one()  # processes starter, adds follow-up
        assert loop.queue_size() == 1
        result = loop.process_one()  # processes follow-up
        assert "Follow-up" in result.output

    def test_no_handler(self):
        loop = CognitiveLoop()
        task = CognitiveTask(task_type=TaskType.PREDICTION, description="no handler")
        loop.push_task(task)
        result = loop.process_one()
        assert not result.success
        assert "No handler" in result.output

    def test_event_integration(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.SELF_MODEL_UPDATE, _echo_handler)
        loop.on_event("memory.stored", {"domain": "finance", "importance": 0.9,
                                         "memory_id": "mem_001"})
        assert loop.queue_size() >= 1

    def test_metrics(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.CUSTOM, _echo_handler)
        loop.push_task(CognitiveTask(task_type=TaskType.CUSTOM, description="test"))
        loop.process_one()
        m = loop.metrics()
        assert m["tasks_processed"] == 1
        assert m["avg_processing_ms"] >= 0

    def test_summary(self):
        loop = CognitiveLoop()
        summary = loop.summary()
        assert "CognitiveLoop" in summary

    def test_background_tasks(self):
        loop = CognitiveLoop()
        # Set all background intervals to 0 so they're immediately due
        loop._background_intervals = {"consolidation": 0, "reflection": 0}
        bg = loop._check_background_tasks()
        assert bg is not None

    def test_queue_cap(self):
        loop = CognitiveLoop()
        for i in range(150):
            loop._enqueue(CognitiveTask(description=f"task_{i}",
                                        context={"importance": 0.5}))
        assert loop.queue_size() <= 100

    def test_persistence_roundtrip(self):
        loop = CognitiveLoop()
        loop.register_handler(TaskType.CUSTOM, _echo_handler)
        loop.push_task(CognitiveTask(task_type=TaskType.CUSTOM, description="persist"))
        loop.scratchpad.start_chain("test chain")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "loop.json"
            loop.save_state(path)

            loop2 = CognitiveLoop()
            loaded = loop2.load_state(path)
            assert loaded
            assert loop2.queue_size() == 1
            assert len(loop2.scratchpad.chains) == 1


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_cognitive_loop_initialized_on_access(self):
        agent = _make_emms()
        summary = agent.cognitive_loop_summary()
        assert "CognitiveLoop" in summary

    def test_event_wiring(self):
        agent = _make_emms()
        # Initialize the loop (triggers EventBus wiring)
        agent._get_cognitive_loop()
        loop = agent._cognitive_loop
        loop.register_handler(TaskType.SELF_MODEL_UPDATE, _echo_handler)

        # Store a high-importance experience → should fire event → create task
        agent.store(Experience(content="critical finding", domain="finance",
                              importance=0.9))
        # The event fires synchronously, so task should be queued
        assert loop._events_received >= 1

    def test_push_task(self):
        agent = _make_emms()
        task_id = agent.cognitive_loop_push("custom", "test task", "general")
        assert task_id.startswith("ctask_")

    def test_process_one(self):
        agent = _make_emms()
        loop = agent._get_cognitive_loop()
        loop.register_handler(TaskType.CUSTOM, _echo_handler)
        agent.cognitive_loop_push("custom", "processable task")
        result = agent.cognitive_loop_process_one()
        assert result is not None
        assert result["success"]

    def test_metrics(self):
        agent = _make_emms()
        m = agent.cognitive_loop_metrics()
        assert "tasks_processed" in m
        assert "queue_size" in m

    def test_reasoning_chain_lifecycle(self):
        agent = _make_emms()
        chain_id = agent.start_reasoning_chain("Analyze ETH", "crypto", "ETH outperforms")
        assert chain_id.startswith("chain_")

        ok = agent.add_reasoning_step(chain_id, "gather data", "found 100 ticks", 0.8)
        assert ok

        ok = agent.complete_reasoning_chain(chain_id, "ETH confirmed outperformer")
        assert ok

    def test_reasoning_chain_not_found(self):
        agent = _make_emms()
        ok = agent.add_reasoning_step("nonexistent", "step", "result")
        assert not ok

    def test_save_load_roundtrip(self):
        agent = _make_emms()
        agent.start_reasoning_chain("test chain", "test")
        agent.cognitive_loop_push("custom", "queued task")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = _make_emms()
            agent2.load(str(path))
            assert len(agent2._get_cognitive_loop().scratchpad.chains) == 1
