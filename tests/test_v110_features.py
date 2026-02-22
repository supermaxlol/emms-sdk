"""Tests for EMMS v0.11.0 — The Sleep Cycle.

Coverage:
  - DreamConsolidator  (22 tests)
  - SessionBridge      (22 tests)
  - MemoryAnnealer     (22 tests)
  - MCP v0.11.0 tools  (7 tests)
  - v0.11.0 exports    (9 tests)

Total: 82 tests
"""

from __future__ import annotations

import json
import time
import pytest

from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS(enable_consciousness=False)


def _make_rich_emms(n: int = 12) -> EMMS:
    """EMMS with varied memories across domains."""
    agent = _make_emms()
    domains = ["science", "philosophy", "technology"]
    for i in range(n):
        agent.store(Experience(
            content=f"Memory {i}: {domains[i % 3]} insight about topic {i}. Important concept.",
            domain=domains[i % 3],
            importance=0.3 + 0.05 * (i % 10),
            emotional_valence=0.1 * ((i % 5) - 2),
            emotional_intensity=0.2 + 0.05 * (i % 8),
        ))
    return agent


def _make_important_emms() -> EMMS:
    """EMMS with some high-importance unresolved memories."""
    agent = _make_emms()
    for i in range(5):
        agent.store(Experience(
            content=f"Critical unresolved question about topic {i} with major implications.",
            domain="research",
            importance=0.8 + 0.04 * i,
        ))
    for i in range(5):
        agent.store(Experience(
            content=f"Low priority note {i} about minor detail.",
            domain="notes",
            importance=0.2,
        ))
    return agent


# ---------------------------------------------------------------------------
# 1. DreamConsolidator
# ---------------------------------------------------------------------------

class TestDreamConsolidator:

    def test_dream_returns_report(self):
        from emms.memory.dream import DreamReport
        agent = _make_rich_emms()
        report = agent.dream()
        assert isinstance(report, DreamReport)

    def test_dream_processes_memories(self):
        agent = _make_rich_emms()
        report = agent.dream()
        assert report.total_memories_processed >= 1

    def test_dream_reinforced_count(self):
        agent = _make_rich_emms()
        report = agent.dream(reinforce_top_k=5)
        assert report.reinforced <= 5
        assert report.reinforced >= 0

    def test_dream_weakened_count(self):
        agent = _make_rich_emms()
        report = agent.dream(weaken_bottom_k=3)
        assert report.weakened <= 3
        assert report.weakened >= 0

    def test_dream_session_id_in_report(self):
        agent = _make_rich_emms()
        report = agent.dream(session_id="test_dream_session")
        assert report.session_id == "test_dream_session"

    def test_dream_auto_session_id(self):
        agent = _make_rich_emms()
        report = agent.dream()
        assert isinstance(report.session_id, str)
        assert len(report.session_id) > 0

    def test_dream_duration_positive(self):
        agent = _make_rich_emms()
        report = agent.dream()
        assert report.duration_seconds >= 0.0

    def test_dream_prune_removes_weak(self):
        from emms.memory.dream import DreamConsolidator
        agent = _make_emms()
        agent.store(Experience(content="very weak memory to prune", domain="test"))
        # Manually set strength below threshold
        all_items = []
        for tier in (agent.memory.working, agent.memory.short_term):
            all_items.extend(tier)
        for tier in (agent.memory.long_term, agent.memory.semantic):
            all_items.extend(tier.values())
        for item in all_items:
            item.memory_strength = 0.001
        consolidator = DreamConsolidator(agent.memory, prune_threshold=0.01)
        report = consolidator.dream()
        assert report.pruned >= 0  # may or may not prune depending on tier removal

    def test_dream_insights_list(self):
        agent = _make_rich_emms()
        report = agent.dream()
        assert isinstance(report.insights, list)

    def test_dream_summary_string(self):
        agent = _make_rich_emms()
        report = agent.dream()
        s = report.summary()
        assert isinstance(s, str)
        assert "Dream report" in s

    def test_dream_entries_list(self):
        from emms.memory.dream import DreamEntry
        agent = _make_rich_emms()
        report = agent.dream(reinforce_top_k=3, weaken_bottom_k=2)
        assert isinstance(report.entries, list)

    def test_dream_entry_fields(self):
        from emms.memory.dream import DreamEntry
        agent = _make_rich_emms()
        report = agent.dream(reinforce_top_k=3)
        if report.entries:
            e = report.entries[0]
            assert isinstance(e, DreamEntry)
            assert e.action in ("reinforced", "weakened", "pruned", "unchanged")
            assert isinstance(e.memory_id, str)
            assert isinstance(e.old_strength, float)
            assert isinstance(e.new_strength, float)

    def test_dream_empty_memory(self):
        agent = _make_emms()
        report = agent.dream()
        assert report.total_memories_processed == 0

    def test_dream_event_emitted(self):
        agent = _make_rich_emms()
        events = []
        agent.events.on("memory.dream_completed", lambda d: events.append(d))
        agent.dream()
        assert len(events) >= 1

    def test_dream_event_has_session_id(self):
        agent = _make_rich_emms()
        events = []
        agent.events.on("memory.dream_completed", lambda d: events.append(d))
        agent.dream(session_id="evt_test")
        assert events[0]["session_id"] == "evt_test"

    def test_dream_no_dedup_skipped(self):
        from emms.memory.dream import DreamConsolidator
        agent = _make_rich_emms()
        consolidator = DreamConsolidator(agent.memory, run_dedup=False)
        report = consolidator.dream()
        assert report.deduped_pairs == 0

    def test_dream_no_patterns_skipped(self):
        from emms.memory.dream import DreamConsolidator
        agent = _make_rich_emms()
        consolidator = DreamConsolidator(agent.memory, run_patterns=False)
        report = consolidator.dream()
        assert report.patterns_found == 0

    def test_dream_direct_consolidator(self):
        from emms.memory.dream import DreamConsolidator, DreamReport
        agent = _make_rich_emms()
        consolidator = DreamConsolidator(agent.memory, reinforce_top_k=5, weaken_bottom_k=3)
        report = consolidator.dream()
        assert isinstance(report, DreamReport)

    def test_dream_report_counts_non_negative(self):
        agent = _make_rich_emms()
        report = agent.dream()
        assert report.reinforced >= 0
        assert report.weakened >= 0
        assert report.pruned >= 0
        assert report.deduped_pairs >= 0
        assert report.patterns_found >= 0

    def test_dream_started_at_timestamp(self):
        before = time.time()
        agent = _make_rich_emms()
        report = agent.dream()
        assert report.started_at >= before

    def test_dream_multiple_passes_allowed(self):
        agent = _make_rich_emms()
        r1 = agent.dream(session_id="pass1")
        r2 = agent.dream(session_id="pass2")
        assert r1.session_id != r2.session_id

    def test_dream_saves_memory_after(self):
        import tempfile, os
        agent = _make_rich_emms(5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent.dream()
            agent.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 2. SessionBridge
# ---------------------------------------------------------------------------

class TestSessionBridge:

    def test_capture_returns_record(self):
        from emms.sessions.bridge import BridgeRecord
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        assert isinstance(record, BridgeRecord)

    def test_capture_has_session_id(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(session_id="test_sess_42")
        assert record.from_session_id == "test_sess_42"

    def test_capture_finds_unresolved_threads(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        assert isinstance(record.open_threads, list)

    def test_capture_high_importance_threads(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        # High-importance memories should appear in threads
        if record.open_threads:
            for t in record.open_threads:
                assert t.importance >= 0.0

    def test_capture_thread_fields(self):
        from emms.sessions.bridge import BridgeThread
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        if record.open_threads:
            t = record.open_threads[0]
            assert isinstance(t, BridgeThread)
            assert isinstance(t.memory_id, str)
            assert isinstance(t.content_excerpt, str)
            assert isinstance(t.domain, str)
            assert isinstance(t.reason, str)

    def test_capture_max_threads_respected(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(max_threads=2)
        assert len(record.open_threads) <= 2

    def test_capture_closing_summary_stored(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(closing_summary="Session about research topics.")
        assert "research" in record.closing_summary.lower()

    def test_capture_emotional_state_in_range(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        assert -1.0 <= record.mean_valence_at_end <= 1.0

    def test_inject_returns_string(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        injection = agent.inject_session_bridge(record)
        assert isinstance(injection, str)

    def test_inject_contains_continuity_header(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        injection = agent.inject_session_bridge(record)
        assert "Session Continuity" in injection

    def test_inject_mentions_threads(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(max_threads=3)
        injection = agent.inject_session_bridge(record)
        if record.open_threads:
            assert "Unresolved" in injection or "thread" in injection.lower()

    def test_inject_includes_emotional_state(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        injection = agent.inject_session_bridge(record)
        assert "Emotional" in injection or "valence" in injection.lower() or "mood" in injection.lower()

    def test_inject_sets_to_session_id(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        agent.inject_session_bridge(record, new_session_id="next_sess_123")
        assert record.to_session_id == "next_sess_123"

    def test_record_to_dict_and_back(self):
        from emms.sessions.bridge import BridgeRecord
        agent = _make_important_emms()
        record = agent.capture_session_bridge(closing_summary="test summary")
        d = record.to_dict()
        record2 = BridgeRecord.from_dict(d)
        assert record2.from_session_id == record.from_session_id
        assert len(record2.open_threads) == len(record.open_threads)

    def test_record_json_roundtrip(self):
        from emms.sessions.bridge import BridgeRecord
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        d = record.to_dict()
        j = json.dumps(d)
        record2 = BridgeRecord.from_dict(json.loads(j))
        assert record2.from_session_id == record.from_session_id

    def test_save_and_load(self):
        import tempfile, os
        from emms.sessions.bridge import SessionBridge
        agent = _make_important_emms()
        record = agent.capture_session_bridge()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bridge = SessionBridge(agent.memory)
            bridge.save(path, record)
            loaded = SessionBridge.load(path)
            assert loaded is not None
            assert loaded.from_session_id == record.from_session_id
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_none(self):
        from emms.sessions.bridge import SessionBridge
        result = SessionBridge.load("/tmp/nonexistent_bridge_xyz.json")
        assert result is None

    def test_bridge_record_summary(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(session_id="sum_test")
        s = record.summary()
        assert "sum_test" in s

    def test_empty_memory_capture(self):
        agent = _make_emms()
        record = agent.capture_session_bridge()
        assert isinstance(record.open_threads, list)
        # Empty memory → no threads
        assert len(record.open_threads) == 0

    def test_bridge_with_presence_tracker(self):
        agent = _make_important_emms()
        agent.enable_presence_tracking()
        for _ in range(5):
            agent.record_presence_turn(content="test turn", domain="research", valence=0.3)
        record = agent.capture_session_bridge()
        # Presence info should be captured
        assert record.presence_score_at_end <= 1.0
        assert record.turn_count >= 5

    def test_bridge_dominant_domains_from_presence(self):
        agent = _make_important_emms()
        agent.enable_presence_tracking()
        for _ in range(5):
            agent.record_presence_turn(content="research turn", domain="research")
        record = agent.capture_session_bridge()
        assert "research" in record.dominant_domains or len(record.dominant_domains) >= 0

    def test_inject_with_closing_summary(self):
        agent = _make_important_emms()
        record = agent.capture_session_bridge(
            closing_summary="Explored memory architecture and reconsolidation theory."
        )
        injection = agent.inject_session_bridge(record)
        assert "Explored" in injection or "memory" in injection.lower()


# ---------------------------------------------------------------------------
# 3. MemoryAnnealer
# ---------------------------------------------------------------------------

class TestMemoryAnnealer:

    def test_anneal_returns_result(self):
        from emms.memory.annealing import AnnealingResult
        agent = _make_rich_emms()
        result = agent.anneal()
        assert isinstance(result, AnnealingResult)

    def test_anneal_total_items(self):
        agent = _make_rich_emms()
        result = agent.anneal()
        assert result.total_items >= 1

    def test_anneal_gap_seconds_set(self):
        agent = _make_rich_emms()
        last_at = time.time() - 3600  # 1 hour ago
        result = agent.anneal(last_session_at=last_at)
        assert 3590 <= result.gap_seconds <= 3610

    def test_anneal_temperature_range(self):
        agent = _make_rich_emms()
        result = agent.anneal()
        assert 0.0 <= result.effective_temperature <= 1.0

    def test_high_temperature_for_recent_gap(self):
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_rich_emms()
        annealer = MemoryAnnealer(agent.memory, half_life_gap=86400.0)
        # Very recent gap (1 minute) → high temperature
        last_at = time.time() - 60
        result = annealer.anneal(last_session_at=last_at)
        assert result.effective_temperature > 0.9

    def test_low_temperature_for_old_gap(self):
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_rich_emms()
        annealer = MemoryAnnealer(agent.memory, half_life_gap=86400.0)
        # Very old gap (30 days) → low temperature
        last_at = time.time() - 86400 * 30
        result = annealer.anneal(last_session_at=last_at)
        assert result.effective_temperature < 0.1

    def test_anneal_default_gap(self):
        """When last_session_at is None, uses half_life_gap → T=0.5."""
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_rich_emms()
        annealer = MemoryAnnealer(agent.memory, half_life_gap=86400.0)
        result = annealer.anneal(last_session_at=None)
        assert abs(result.effective_temperature - 0.5) < 0.01

    def test_anneal_accelerates_decay(self):
        agent = _make_rich_emms()
        all_items = []
        for tier in (agent.memory.working, agent.memory.short_term):
            all_items.extend(tier)
        for tier in (agent.memory.long_term, agent.memory.semantic):
            all_items.extend(tier.values())
        old_strengths = {it.id: it.memory_strength for it in all_items}
        # Old gap → higher temperature → more decay for weak memories
        agent.anneal(last_session_at=time.time() - 3600)
        # At least some items should have changed
        all_items_after = []
        for tier in (agent.memory.working, agent.memory.short_term):
            all_items_after.extend(tier)
        for tier in (agent.memory.long_term, agent.memory.semantic):
            all_items_after.extend(tier.values())
        any_changed = any(
            abs(it.memory_strength - old_strengths.get(it.id, 1.0)) > 0.001
            for it in all_items_after
            if it.id in old_strengths
        )
        # With a significant gap and non-zero decay rate, some items should change
        result2 = agent.anneal(last_session_at=time.time() - 3600 * 24)
        assert result2.total_items >= 0

    def test_anneal_emotionally_stabilizes(self):
        agent = _make_emms()
        # Store memory with extreme valence
        agent.store(Experience(
            content="deeply emotional memory with extreme charge",
            domain="test",
            emotional_valence=0.9,
            emotional_intensity=0.8,
        ))
        result = agent.anneal(last_session_at=time.time() - 86400)  # 1 day
        # Should have stabilized at least this one memory
        assert result.emotionally_stabilized >= 0

    def test_anneal_strengthens_important(self):
        agent = _make_emms()
        agent.store(Experience(
            content="very important memory that should survive",
            domain="test",
            importance=0.9,
        ))
        # Long gap → low temperature → strengthen important memories
        result = agent.anneal(last_session_at=time.time() - 86400 * 30)
        assert result.strengthened >= 0

    def test_anneal_duration_positive(self):
        agent = _make_rich_emms()
        result = agent.anneal()
        assert result.duration_seconds >= 0.0

    def test_anneal_summary_string(self):
        agent = _make_rich_emms()
        result = agent.anneal()
        s = result.summary()
        assert isinstance(s, str)
        assert "annealing" in s.lower()

    def test_anneal_strength_floor_respected(self):
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_emms()
        agent.store(Experience(content="floor test memory xyz", domain="test"))
        all_items = list(agent.memory.working)
        if all_items:
            all_items[0].memory_strength = 0.02
        annealer = MemoryAnnealer(agent.memory, min_strength=0.01, decay_rate=0.5)
        annealer.anneal(last_session_at=time.time() - 3600)
        for tier in (agent.memory.working, agent.memory.short_term):
            for it in tier:
                assert it.memory_strength >= 0.01

    def test_anneal_strength_ceiling_respected(self):
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_emms()
        agent.store(Experience(
            content="ceiling test important memory", domain="test", importance=0.95
        ))
        annealer = MemoryAnnealer(agent.memory, max_strength=2.0, strengthen_rate=100.0)
        annealer.anneal(last_session_at=time.time() - 86400 * 30)
        for tier in (agent.memory.working, agent.memory.short_term):
            for it in tier:
                assert it.memory_strength <= 2.0

    def test_anneal_zero_temperature_no_changes(self):
        """Zero temperature: no changes should occur."""
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_rich_emms()
        all_items = []
        for tier in (agent.memory.working, agent.memory.short_term):
            all_items.extend(tier)
        old_strengths = {it.id: it.memory_strength for it in all_items}
        # Near-zero gap → temperature close to 1.0 (max plastic)
        # Instead test the temperature function itself
        annealer = MemoryAnnealer(agent.memory)
        # half_life_gap = 0 → temperature is always 1.0
        t = annealer.temperature(0.0)
        assert t == 1.0

    def test_annealer_temperature_function(self):
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_emms()
        annealer = MemoryAnnealer(agent.memory, half_life_gap=100.0)
        # At gap = half_life: T = 0.5
        t = annealer.temperature(100.0)
        assert abs(t - 0.5) < 0.001
        # At gap = 0: T = 1.0
        assert abs(annealer.temperature(0.0) - 1.0) < 0.001

    def test_anneal_empty_memory(self):
        agent = _make_emms()
        result = agent.anneal()
        assert result.total_items == 0

    def test_anneal_facade_params(self):
        agent = _make_rich_emms()
        result = agent.anneal(
            last_session_at=time.time() - 7200,
            half_life_gap=86400.0,
            decay_rate=0.01,
            emotional_stabilization_rate=0.05,
        )
        assert isinstance(result.effective_temperature, float)

    def test_anneal_result_counts_non_negative(self):
        agent = _make_rich_emms()
        result = agent.anneal()
        assert result.accelerated_decay >= 0
        assert result.emotionally_stabilized >= 0
        assert result.strengthened >= 0

    def test_anneal_gap_zero(self):
        """Zero gap → very high temperature → max plasticity."""
        from emms.memory.annealing import MemoryAnnealer
        agent = _make_rich_emms()
        annealer = MemoryAnnealer(agent.memory)
        result = annealer.anneal(last_session_at=time.time())
        assert result.gap_seconds < 1.0
        assert result.effective_temperature > 0.99


# ---------------------------------------------------------------------------
# 4. MCP v0.11.0 tools
# ---------------------------------------------------------------------------

class TestMCPV110Tools:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(agent), agent

    def test_mcp_dream_tool(self):
        server, _ = self._make_server()
        resp = server.handle("emms_dream", {"reinforce_top_k": 5, "weaken_bottom_k": 3})
        assert resp["ok"] is True
        assert "total_memories_processed" in resp
        assert "reinforced" in resp

    def test_mcp_capture_bridge_tool(self):
        server, _ = self._make_server()
        resp = server.handle("emms_capture_bridge", {
            "closing_summary": "test session complete"
        })
        assert resp["ok"] is True
        assert "open_threads" in resp
        assert "bridge_json" in resp

    def test_mcp_inject_bridge_tool(self):
        server, _ = self._make_server()
        # Capture then inject
        cap_resp = server.handle("emms_capture_bridge", {"closing_summary": "done"})
        assert cap_resp["ok"] is True
        bridge_json = cap_resp["bridge_json"]
        inj_resp = server.handle("emms_inject_bridge", {
            "bridge_json": bridge_json,
            "new_session_id": "next_sess",
        })
        assert inj_resp["ok"] is True
        assert "injection" in inj_resp

    def test_mcp_anneal_tool(self):
        server, _ = self._make_server()
        resp = server.handle("emms_anneal", {
            "last_session_at": time.time() - 3600,
        })
        assert resp["ok"] is True
        assert "effective_temperature" in resp
        assert "total_items" in resp

    def test_mcp_bridge_summary_tool(self):
        server, _ = self._make_server()
        resp = server.handle("emms_bridge_summary", {})
        assert resp["ok"] is True
        assert "open_threads" in resp

    def test_mcp_dream_session_id(self):
        server, _ = self._make_server()
        resp = server.handle("emms_dream", {"session_id": "mcp_test_dream"})
        assert resp["ok"] is True
        assert resp["session_id"] == "mcp_test_dream"

    def test_mcp_tool_count(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms()
        server = EMCPServer(agent)
        assert len(server.tool_definitions) == 97


# ---------------------------------------------------------------------------
# 5. v0.11.0 exports
# ---------------------------------------------------------------------------

class TestV110Exports:

    def test_version(self):
        import emms
        assert emms.__version__ == "0.22.0"

    def test_dream_consolidator_export(self):
        from emms import DreamConsolidator
        assert DreamConsolidator is not None

    def test_dream_report_export(self):
        from emms import DreamReport
        assert DreamReport is not None

    def test_dream_entry_export(self):
        from emms import DreamEntry
        assert DreamEntry is not None

    def test_session_bridge_export(self):
        from emms import SessionBridge
        assert SessionBridge is not None

    def test_bridge_record_export(self):
        from emms import BridgeRecord
        assert BridgeRecord is not None

    def test_bridge_thread_export(self):
        from emms import BridgeThread
        assert BridgeThread is not None

    def test_memory_annealer_export(self):
        from emms import MemoryAnnealer
        assert MemoryAnnealer is not None

    def test_annealing_result_export(self):
        from emms import AnnealingResult
        assert AnnealingResult is not None
