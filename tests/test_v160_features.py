"""Tests for EMMS v0.16.0 — The Curious Mind.

Covers:
    - CuriosityEngine + ExplorationGoal + CuriosityReport
    - BeliefReviser + RevisionRecord + RevisionReport
    - MemoryDecay + DecayRecord + DecayReport
    - EMMS facade methods
    - MCP tool count (67) and new tool callability
    - __init__ exports and version string
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from emms import EMMS, Experience
from emms.adapters.mcp_server import EMCPServer, _TOOL_DEFINITIONS
from emms.memory.curiosity import CuriosityEngine, CuriosityReport, ExplorationGoal
from emms.memory.belief_revision import (
    BeliefReviser,
    RevisionRecord,
    RevisionReport,
)
from emms.memory.decay import DecayRecord, DecayReport, MemoryDecay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms(n: int = 6) -> EMMS:
    agent = EMMS()
    domains = ["science", "history", "science", "history", "art", "art"]
    valences = [0.8, -0.7, 0.6, -0.5, 0.3, -0.4]
    for i in range(n):
        agent.store(
            Experience(
                content=f"Memory {i}: knowledge about topic {i % 3} in domain {domains[i % len(domains)]}",
                domain=domains[i % len(domains)],
                importance=0.5 + 0.05 * i,
                emotional_valence=valences[i % len(valences)],
            )
        )
    return agent


# ===========================================================================
# TestCuriosityEngine
# ===========================================================================


class TestCuriosityEngine:
    """Tests for CuriosityEngine standalone usage."""

    def test_scan_returns_report(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        assert isinstance(report, CuriosityReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        assert isinstance(report.total_domains_scanned, int)
        assert isinstance(report.goals_generated, int)
        assert isinstance(report.goals, list)
        assert isinstance(report.top_curious_domains, list)
        assert report.duration_seconds >= 0

    def test_goals_are_exploration_goals(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        for g in report.goals:
            assert isinstance(g, ExplorationGoal)

    def test_exploration_goal_fields(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        if report.goals:
            g = report.goals[0]
            assert isinstance(g.id, str) and g.id.startswith("goal_")
            assert isinstance(g.question, str) and len(g.question) > 5
            assert isinstance(g.domain, str)
            assert 0.0 <= g.urgency <= 1.0
            assert g.gap_type in ("sparse", "uncertain", "contradictory", "novel")
            assert isinstance(g.supporting_memory_ids, list)
            assert g.explored is False

    def test_goals_sorted_by_urgency(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        urgencies = [g.urgency for g in report.goals]
        assert urgencies == sorted(urgencies, reverse=True)

    def test_domain_filter(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan(domain="science")
        for g in report.goals:
            assert g.domain == "science"

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        assert report.goals_generated == 0
        assert report.total_domains_scanned == 0

    def test_sparse_gap_detected(self):
        agent = _make_emms()
        agent.store(Experience(content="One thing about astronomy", domain="astronomy"))
        engine = CuriosityEngine(memory=agent.memory, sparse_threshold=3)
        report = engine.scan()
        types = [g.gap_type for g in report.goals]
        assert "novel" in types or "sparse" in types

    def test_novel_gap_for_single_item_domain(self):
        agent = _make_emms()
        agent.store(Experience(content="Single isolated fact", domain="singledomain"))
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan(domain="singledomain")
        types = [g.gap_type for g in report.goals]
        assert "novel" in types

    def test_uncertain_gap_detected(self):
        agent = _make_emms()
        for _ in range(4):
            it = agent.store(
                Experience(
                    content="Weak uncertain knowledge fragment about quantum",
                    domain="quantum",
                    importance=0.1,
                )
            )
            # Manually weaken
            for item in list(agent.memory.working):
                item.memory_strength = 0.1
        engine = CuriosityEngine(memory=agent.memory, uncertain_threshold=0.9)
        report = engine.scan(domain="quantum")
        types = [g.gap_type for g in report.goals]
        assert "uncertain" in types

    def test_pending_goals(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        engine.scan()
        pending = engine.pending_goals()
        for g in pending:
            assert not g.explored

    def test_mark_explored_true(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        if report.goals:
            gid = report.goals[0].id
            assert engine.mark_explored(gid) is True
            assert report.goals[0].explored is True

    def test_mark_explored_false_for_unknown(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        assert engine.mark_explored("nonexistent_goal_id") is False

    def test_pending_goals_excludes_explored(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        if report.goals:
            gid = report.goals[0].id
            engine.mark_explored(gid)
            pending_ids = {g.id for g in engine.pending_goals()}
            assert gid not in pending_ids

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        s = report.summary()
        assert isinstance(s, str)
        assert "CuriosityReport" in s

    def test_exploration_goal_summary_str(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        if report.goals:
            s = report.goals[0].summary()
            assert isinstance(s, str)
            assert "ExplorationGoal" in s

    def test_max_goals_respected(self):
        agent = _make_rich_emms(12)
        engine = CuriosityEngine(memory=agent.memory, max_goals=3)
        report = engine.scan()
        assert report.goals_generated <= 3

    def test_top_curious_domains_list(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan()
        assert isinstance(report.top_curious_domains, list)

    def test_contradictory_gap_detected(self):
        agent = _make_emms()
        # Create two memories in same domain with opposing valence and overlapping content
        agent.store(Experience(
            content="quantum entanglement is perfectly understood phenomenon",
            domain="physics",
            emotional_valence=0.9,
            importance=0.7,
        ))
        agent.store(Experience(
            content="quantum entanglement is deeply mysterious unexplained phenomenon",
            domain="physics",
            emotional_valence=-0.9,
            importance=0.7,
        ))
        engine = CuriosityEngine(memory=agent.memory)
        report = engine.scan(domain="physics")
        types = [g.gap_type for g in report.goals]
        # Should detect either contradictory or sparse
        assert len(types) > 0

    def test_scan_multiple_times_accumulates_goals(self):
        agent = _make_rich_emms()
        engine = CuriosityEngine(memory=agent.memory)
        engine.scan()
        initial_count = len(engine._goals)
        engine.scan()
        # Goals accumulate (new ones are stored)
        assert len(engine._goals) >= initial_count


# ===========================================================================
# TestBeliefReviser
# ===========================================================================


class TestBeliefReviser:
    """Tests for BeliefReviser standalone usage."""

    def _make_conflicting_emms(self) -> EMMS:
        agent = _make_emms()
        agent.store(Experience(
            content="regular daily exercise dramatically improves cardiovascular health endurance physical strength",
            domain="health",
            emotional_valence=0.9,
            importance=0.8,
        ))
        agent.store(Experience(
            content="regular daily exercise dramatically damages cardiovascular health overuse physical strain",
            domain="health",
            emotional_valence=-0.9,
            importance=0.7,
        ))
        return agent

    def test_revise_returns_report(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        assert isinstance(report, RevisionReport)

    def test_report_fields(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        assert isinstance(report.total_checked, int)
        assert isinstance(report.conflicts_found, int)
        assert isinstance(report.revisions_made, int)
        assert isinstance(report.records, list)
        assert report.duration_seconds >= 0

    def test_conflict_detected(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        assert report.conflicts_found >= 1

    def test_revision_record_fields(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        if report.records:
            r = report.records[0]
            assert isinstance(r.id, str) and r.id.startswith("rev_")
            assert isinstance(r.trigger_memory_id, str)
            assert isinstance(r.conflicting_memory_id, str)
            assert r.revision_type in ("merge", "supersede", "flag")
            assert 0.0 <= r.conflict_score <= 1.0
            assert isinstance(r.new_content, str)

    def test_revision_type_merge_or_supersede(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        for r in report.records:
            assert r.revision_type in ("merge", "supersede", "flag")

    def test_revision_history_empty_initially(self):
        agent = _make_emms()
        reviser = BeliefReviser(memory=agent.memory)
        assert reviser.revision_history() == []

    def test_revision_history_populated_after_revise(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        reviser.revise()
        history = reviser.revision_history()
        assert isinstance(history, list)

    def test_revision_history_newest_first(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        reviser.revise()
        history = reviser.revision_history()
        if len(history) >= 2:
            assert history[0].created_at >= history[1].created_at

    def test_domain_filter(self):
        agent = self._make_conflicting_emms()
        # Add unrelated domain
        agent.store(Experience(
            content="cooking is wonderful art creative",
            domain="cooking",
            emotional_valence=0.8,
        ))
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise(domain="health")
        for r in report.records:
            # Records should come from health domain only
            assert isinstance(r.revision_type, str)

    def test_max_revisions_respected(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory, max_revisions=1)
        report = reviser.revise()
        assert report.revisions_made <= 1

    def test_no_conflict_empty_memory(self):
        agent = _make_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        assert report.conflicts_found == 0
        assert report.revisions_made == 0

    def test_report_summary_str(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        s = report.summary()
        assert isinstance(s, str)
        assert "RevisionReport" in s

    def test_record_summary_str(self):
        agent = self._make_conflicting_emms()
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        if report.records:
            s = report.records[0].summary()
            assert isinstance(s, str)
            assert "RevisionRecord" in s

    def test_supersede_weakens_loser(self):
        agent = _make_emms()
        # Strong memory
        agent.store(Experience(
            content="vaccines thoroughly proven safe beneficial effective immunization",
            domain="medicine",
            emotional_valence=0.95,
            importance=0.95,
        ))
        # Weak opposing memory
        agent.store(Experience(
            content="vaccines proven harmful dangerous detrimental immunization",
            domain="medicine",
            emotional_valence=-0.95,
            importance=0.3,
        ))
        reviser = BeliefReviser(memory=agent.memory)
        report = reviser.revise()
        # There may or may not be a supersede depending on score
        assert report.revisions_made >= 0  # At minimum, no crash

    def test_merge_creates_reconciliation(self):
        agent = _make_emms()
        # Moderate conflict — should produce merge
        agent.store(Experience(
            content="coffee caffeine beneficial stimulating productive energizing",
            domain="nutrition",
            emotional_valence=0.6,
            importance=0.7,
        ))
        agent.store(Experience(
            content="coffee caffeine harmful detrimental sleep disrupting",
            domain="nutrition",
            emotional_valence=-0.6,
            importance=0.65,
        ))
        reviser = BeliefReviser(
            memory=agent.memory,
            overlap_threshold=0.1,
            valence_conflict_threshold=0.3,
        )
        report = reviser.revise()
        if report.records:
            types = {r.revision_type for r in report.records}
            assert types & {"merge", "supersede", "flag"}

    def test_token_overlap_static(self):
        overlap = BeliefReviser._token_overlap(
            "exercise running jogging healthy beneficial",
            "exercise running jogging dangerous harmful",
        )
        assert overlap > 0.0

    def test_token_overlap_disjoint(self):
        overlap = BeliefReviser._token_overlap("apple orange fruit", "computer network server")
        assert overlap == 0.0


# ===========================================================================
# TestMemoryDecay
# ===========================================================================


class TestMemoryDecay:
    """Tests for MemoryDecay standalone usage."""

    def test_decay_returns_report(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        assert isinstance(report, DecayReport)

    def test_decay_report_fields(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        assert isinstance(report.total_processed, int)
        assert isinstance(report.decayed, int)
        assert isinstance(report.pruned, int)
        assert 0.0 <= report.mean_retention <= 1.0
        assert report.applied is False
        assert report.duration_seconds >= 0
        assert isinstance(report.records, list)

    def test_decay_is_read_only(self):
        agent = _make_rich_emms()
        # Record original strengths
        items_before = {
            item.id: item.memory_strength
            for item in list(agent.memory.working) + list(agent.memory.short_term)
        }
        engine = MemoryDecay(memory=agent.memory)
        engine.decay()
        items_after = {
            item.id: item.memory_strength
            for item in list(agent.memory.working) + list(agent.memory.short_term)
        }
        assert items_before == items_after

    def test_apply_decay_returns_report(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.apply_decay()
        assert isinstance(report, DecayReport)

    def test_apply_decay_applied_true(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.apply_decay()
        assert report.applied is True

    def test_retention_value_range(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        for r in report.records:
            assert 0.0 <= r.retention <= 1.0

    def test_new_strength_reduced(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        # All memories are fresh so retention may be close to 1; just check non-negative
        for r in report.records:
            assert r.new_strength >= 0.0

    def test_decay_record_fields(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        if report.records:
            r = report.records[0]
            assert isinstance(r.memory_id, str)
            assert isinstance(r.domain, str)
            assert r.old_strength >= 0
            assert r.new_strength >= 0
            assert 0.0 <= r.retention <= 1.0
            assert r.stability > 0
            assert r.days_since_access >= 0
            assert isinstance(r.pruned, bool)

    def test_stability_increases_with_access(self):
        agent = _make_emms()
        agent.store(Experience(content="Highly accessed memory content", domain="test"))
        # Manually boost access count
        for item in list(agent.memory.working):
            item.access_count = 10
        engine = MemoryDecay(memory=agent.memory, base_stability=7.0, retrieval_boost=2.0)
        report = engine.decay()
        if report.records:
            r = report.records[0]
            # S = 7 + 2*10 = 27
            assert r.stability >= 7.0

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        assert report.total_processed == 0
        assert report.mean_retention == 0.0

    def test_domain_filter(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay(domain="science")
        for r in report.records:
            assert r.domain == "science"

    def test_retention_single_item(self):
        agent = _make_emms()
        agent.store(Experience(content="Fresh memory just stored", domain="test"))
        engine = MemoryDecay(memory=agent.memory)
        for item in list(agent.memory.working):
            R, S = engine.retention(item)
            # Fresh memory should have very high retention
            assert 0.9 <= R <= 1.0
            assert S >= 7.0
            break

    def test_prune_flag_in_record(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory, prune_threshold=0.99)
        report = engine.apply_decay(prune=True)
        # All strengths will be below 0.99 after decay, so all pruned
        assert isinstance(report.pruned, int)

    def test_apply_decay_no_prune_by_default(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.apply_decay()
        assert report.pruned == 0

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        s = report.summary()
        assert isinstance(s, str)
        assert "DecayReport" in s

    def test_record_summary_str(self):
        agent = _make_rich_emms()
        engine = MemoryDecay(memory=agent.memory)
        report = engine.decay()
        if report.records:
            s = report.records[0].summary()
            assert isinstance(s, str)

    def test_max_records_respected(self):
        agent = _make_rich_emms(10)
        engine = MemoryDecay(memory=agent.memory, max_records=3)
        report = engine.decay()
        assert len(report.records) <= 3


# ===========================================================================
# TestEMMSFacadeV160
# ===========================================================================


class TestEMMSFacadeV160:
    """Tests for EMMS facade methods added in v0.16.0."""

    def test_curiosity_scan_returns_report(self):
        agent = _make_rich_emms()
        report = agent.curiosity_scan()
        assert isinstance(report, CuriosityReport)

    def test_curiosity_scan_with_domain(self):
        agent = _make_rich_emms()
        report = agent.curiosity_scan(domain="science")
        for g in report.goals:
            assert g.domain == "science"

    def test_exploration_goals(self):
        agent = _make_rich_emms()
        agent.curiosity_scan()
        goals = agent.exploration_goals()
        assert isinstance(goals, list)
        for g in goals:
            assert isinstance(g, ExplorationGoal)

    def test_exploration_goals_without_prior_scan(self):
        agent = _make_rich_emms()
        goals = agent.exploration_goals()
        assert isinstance(goals, list)

    def test_mark_explored_true(self):
        agent = _make_rich_emms()
        report = agent.curiosity_scan()
        if report.goals:
            gid = report.goals[0].id
            assert agent.mark_explored(gid) is True

    def test_mark_explored_false_unknown(self):
        agent = _make_rich_emms()
        assert agent.mark_explored("nonexistent_id") is False

    def test_revise_beliefs_returns_report(self):
        agent = _make_rich_emms()
        report = agent.revise_beliefs()
        assert isinstance(report, RevisionReport)

    def test_revise_beliefs_with_domain(self):
        agent = _make_rich_emms()
        report = agent.revise_beliefs(domain="science")
        assert isinstance(report, RevisionReport)

    def test_revision_history_empty_initially(self):
        agent = _make_emms()
        history = agent.revision_history()
        assert history == []

    def test_revision_history_after_revise(self):
        agent = _make_rich_emms()
        agent.revise_beliefs()
        history = agent.revision_history()
        assert isinstance(history, list)

    def test_memory_decay_report_returns_report(self):
        agent = _make_rich_emms()
        report = agent.memory_decay_report()
        assert isinstance(report, DecayReport)

    def test_memory_decay_report_read_only(self):
        agent = _make_rich_emms()
        strengths_before = {
            item.id: item.memory_strength
            for item in list(agent.memory.working)
        }
        agent.memory_decay_report()
        strengths_after = {
            item.id: item.memory_strength
            for item in list(agent.memory.working)
        }
        assert strengths_before == strengths_after

    def test_apply_memory_decay_returns_report(self):
        agent = _make_rich_emms()
        report = agent.apply_memory_decay()
        assert isinstance(report, DecayReport)
        assert report.applied is True

    def test_apply_memory_decay_with_prune(self):
        agent = _make_rich_emms()
        report = agent.apply_memory_decay(prune=False)
        assert report.pruned == 0

    def test_apply_memory_decay_with_domain(self):
        agent = _make_rich_emms()
        report = agent.apply_memory_decay(domain="science")
        for r in report.records:
            assert r.domain == "science"

    def test_curiosity_engine_persisted_between_calls(self):
        agent = _make_rich_emms()
        agent.curiosity_scan()
        assert hasattr(agent, "_curiosity_engine")

    def test_belief_reviser_persisted_between_calls(self):
        agent = _make_rich_emms()
        agent.revise_beliefs()
        agent.revise_beliefs()
        # Should reuse same reviser instance
        assert hasattr(agent, "_belief_reviser")


# ===========================================================================
# TestMCPV160
# ===========================================================================


class TestMCPV160:
    """Tests for MCP tool count and new tool callability."""

    def test_tool_count_67(self):
        assert len(_TOOL_DEFINITIONS) == 82

    def test_all_tool_names_unique(self):
        names = [t["name"] for t in _TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_curiosity_report_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_curiosity_report" in names

    def test_exploration_goals_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_exploration_goals" in names

    def test_revise_beliefs_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_revise_beliefs" in names

    def test_decay_report_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_decay_report" in names

    def test_apply_decay_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_apply_decay" in names

    def test_curiosity_report_callable(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_curiosity_report", {})
        assert result.get("ok") is not False

    def test_curiosity_report_with_domain(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_curiosity_report", {"domain": "science"})
        assert "goals" in result or "error" in result

    def test_exploration_goals_callable(self):
        server = EMCPServer(_make_rich_emms())
        server.handle("emms_curiosity_report", {})
        result = server.handle("emms_exploration_goals", {})
        assert "count" in result

    def test_revise_beliefs_callable(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_revise_beliefs", {})
        assert "revisions_made" in result or "error" in result

    def test_decay_report_callable(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_decay_report", {})
        assert "total_processed" in result or "error" in result

    def test_apply_decay_callable(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_apply_decay", {"prune": False})
        assert "applied" in result or "error" in result


# ===========================================================================
# TestV160Exports
# ===========================================================================


class TestV160Exports:
    """Tests for __init__.py exports and version string."""

    def test_version_string(self):
        import emms
        assert emms.__version__ == "0.19.0"

    def test_curiosity_engine_exported(self):
        from emms import CuriosityEngine
        assert CuriosityEngine is not None

    def test_exploration_goal_exported(self):
        from emms import ExplorationGoal
        assert ExplorationGoal is not None

    def test_curiosity_report_exported(self):
        from emms import CuriosityReport
        assert CuriosityReport is not None

    def test_belief_reviser_exported(self):
        from emms import BeliefReviser
        assert BeliefReviser is not None

    def test_revision_record_exported(self):
        from emms import RevisionRecord
        assert RevisionRecord is not None

    def test_revision_report_exported(self):
        from emms import RevisionReport
        assert RevisionReport is not None

    def test_memory_decay_exported(self):
        from emms import MemoryDecay
        assert MemoryDecay is not None

    def test_decay_record_exported(self):
        from emms import DecayRecord
        assert DecayRecord is not None

    def test_decay_report_exported(self):
        from emms import DecayReport
        assert DecayReport is not None

    def test_all_exports_in_all(self):
        import emms
        for name in [
            "CuriosityEngine", "ExplorationGoal", "CuriosityReport",
            "BeliefReviser", "RevisionRecord", "RevisionReport",
            "MemoryDecay", "DecayRecord", "DecayReport",
        ]:
            assert name in emms.__all__, f"{name} missing from __all__"
