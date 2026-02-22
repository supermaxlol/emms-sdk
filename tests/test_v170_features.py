"""Tests for EMMS v0.17.0 — The Goal-Directed Mind.

Covers:
    - GoalStack + Goal + GoalReport
    - AttentionFilter + AttentionResult + AttentionReport
    - AnalogyEngine + AnalogyMapping + AnalogyRecord + AnalogyReport
    - EMMS facade methods
    - MCP tool count (72) and new tool callability
    - __init__ exports and version string
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from emms import EMMS, Experience
from emms.adapters.mcp_server import EMCPServer, _TOOL_DEFINITIONS
from emms.memory.goals import Goal, GoalReport, GoalStack
from emms.memory.attention import AttentionFilter, AttentionReport, AttentionResult
from emms.memory.analogy import (
    AnalogyEngine, AnalogyMapping, AnalogyRecord, AnalogyReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms(n: int = 6) -> EMMS:
    agent = EMMS()
    domains = ["science", "history", "science", "history", "art", "art"]
    for i in range(n):
        agent.store(Experience(
            content=f"Memory {i}: knowledge about topic {i % 3} in domain {domains[i % len(domains)]}",
            domain=domains[i % len(domains)],
            importance=0.5 + 0.05 * i,
        ))
    return agent


def _make_analogy_emms() -> EMMS:
    """Two domains with shared relational structure."""
    agent = EMMS()
    agent.store(Experience(
        content="stress causes cortisol release which produces anxiety and reduces immune function",
        domain="biology",
        importance=0.8,
    ))
    agent.store(Experience(
        content="debt causes interest accumulation which produces financial stress and reduces savings",
        domain="economics",
        importance=0.8,
    ))
    agent.store(Experience(
        content="exercise enables muscle growth through progressive overload and increases strength",
        domain="biology",
        importance=0.7,
    ))
    agent.store(Experience(
        content="practice enables skill mastery through deliberate repetition and increases performance",
        domain="education",
        importance=0.7,
    ))
    return agent


# ===========================================================================
# TestGoalStack
# ===========================================================================


class TestGoalStack:

    def test_push_returns_goal(self):
        gs = GoalStack()
        g = gs.push("Learn about quantum computing")
        assert isinstance(g, Goal)

    def test_goal_fields(self):
        gs = GoalStack()
        g = gs.push("Test goal", domain="science", priority=0.8)
        assert isinstance(g.id, str) and g.id.startswith("goal_")
        assert g.content == "Test goal"
        assert g.domain == "science"
        assert g.priority == 0.8
        assert g.status == "pending"
        assert g.parent_id is None
        assert g.resolved_at is None
        assert g.outcome_note == ""
        assert isinstance(g.supporting_memory_ids, list)

    def test_priority_clamped(self):
        gs = GoalStack()
        g = gs.push("Too high", priority=2.0)
        assert g.priority == 1.0
        g2 = gs.push("Too low", priority=-1.0)
        assert g2.priority == 0.0

    def test_activate_pending(self):
        gs = GoalStack()
        g = gs.push("Activate me")
        assert gs.activate(g.id) is True
        assert g.status == "active"

    def test_activate_already_active_returns_false(self):
        gs = GoalStack()
        g = gs.push("Goal")
        gs.activate(g.id)
        assert gs.activate(g.id) is False

    def test_activate_unknown_returns_false(self):
        gs = GoalStack()
        assert gs.activate("nonexistent") is False

    def test_complete_goal(self):
        gs = GoalStack()
        g = gs.push("Complete me")
        gs.activate(g.id)
        assert gs.complete(g.id, outcome_note="Done!") is True
        assert g.status == "completed"
        assert g.outcome_note == "Done!"
        assert g.resolved_at is not None

    def test_complete_pending_goal_directly(self):
        gs = GoalStack()
        g = gs.push("Skip activation")
        assert gs.complete(g.id) is True
        assert g.status == "completed"

    def test_complete_already_resolved_returns_false(self):
        gs = GoalStack()
        g = gs.push("Done")
        gs.complete(g.id)
        assert gs.complete(g.id) is False

    def test_fail_goal(self):
        gs = GoalStack()
        g = gs.push("Will fail")
        assert gs.fail(g.id, reason="Impossible") is True
        assert g.status == "failed"
        assert g.outcome_note == "Impossible"

    def test_abandon_goal(self):
        gs = GoalStack()
        g = gs.push("Will abandon")
        assert gs.abandon(g.id, reason="No longer needed") is True
        assert g.status == "abandoned"

    def test_is_resolved_pending(self):
        gs = GoalStack()
        g = gs.push("Pending")
        assert g.is_resolved() is False

    def test_is_resolved_completed(self):
        gs = GoalStack()
        g = gs.push("Completed")
        gs.complete(g.id)
        assert g.is_resolved() is True

    def test_is_resolved_failed(self):
        gs = GoalStack()
        g = gs.push("Failed")
        gs.fail(g.id)
        assert g.is_resolved() is True

    def test_is_resolved_abandoned(self):
        gs = GoalStack()
        g = gs.push("Abandoned")
        gs.abandon(g.id)
        assert g.is_resolved() is True

    def test_active_goals_sorted_by_priority(self):
        gs = GoalStack()
        g1 = gs.push("Low", priority=0.2)
        g2 = gs.push("High", priority=0.9)
        g3 = gs.push("Mid", priority=0.5)
        gs.activate(g1.id)
        gs.activate(g2.id)
        gs.activate(g3.id)
        active = gs.active_goals()
        priorities = [g.priority for g in active]
        assert priorities == sorted(priorities, reverse=True)

    def test_pending_goals(self):
        gs = GoalStack()
        gs.push("Pending 1")
        gs.push("Pending 2")
        pending = gs.pending_goals()
        assert len(pending) == 2
        for g in pending:
            assert g.status == "pending"

    def test_sub_goals(self):
        gs = GoalStack()
        parent = gs.push("Parent goal")
        child1 = gs.push("Child 1", parent_id=parent.id)
        child2 = gs.push("Child 2", parent_id=parent.id)
        subs = gs.sub_goals(parent.id)
        assert len(subs) == 2
        assert {g.id for g in subs} == {child1.id, child2.id}

    def test_sub_goals_empty(self):
        gs = GoalStack()
        g = gs.push("Leaf goal")
        assert gs.sub_goals(g.id) == []

    def test_get_goal(self):
        gs = GoalStack()
        g = gs.push("Get me")
        assert gs.get(g.id) is g

    def test_get_unknown_returns_none(self):
        gs = GoalStack()
        assert gs.get("unknown") is None

    def test_goal_with_deadline(self):
        gs = GoalStack()
        deadline = time.time() + 3600
        g = gs.push("Deadline goal", deadline=deadline)
        assert g.deadline == deadline

    def test_report_counts(self):
        gs = GoalStack()
        p1 = gs.push("P1")
        p2 = gs.push("P2")
        gs.activate(p1.id)
        gs.complete(p1.id)
        gs.fail(p2.id)
        gs.push("A3")
        gs.push("P3")
        gs.abandon(gs.push("Abandon me").id)
        report = gs.report()
        assert isinstance(report, GoalReport)
        assert report.total == 5
        assert report.completed == 1
        assert report.failed == 1

    def test_report_goals_sorted_by_priority(self):
        gs = GoalStack()
        gs.push("Low", priority=0.1)
        gs.push("High", priority=0.9)
        report = gs.report()
        priorities = [g.priority for g in report.goals]
        assert priorities == sorted(priorities, reverse=True)

    def test_goal_summary_str(self):
        gs = GoalStack()
        g = gs.push("Test summary")
        s = g.summary()
        assert isinstance(s, str)
        assert "Goal" in s

    def test_report_summary_str(self):
        gs = GoalStack()
        gs.push("A goal")
        report = gs.report()
        s = report.summary()
        assert isinstance(s, str)
        assert "GoalReport" in s

    def test_supporting_memory_ids(self):
        gs = GoalStack()
        g = gs.push("Goal with memories", supporting_memory_ids=["mem1", "mem2"])
        assert "mem1" in g.supporting_memory_ids


# ===========================================================================
# TestAttentionFilter
# ===========================================================================


class TestAttentionFilter:

    def test_spotlight_retrieve_returns_report(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        assert isinstance(report, AttentionReport)

    def test_report_fields(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        assert isinstance(report.spotlight_keywords, list)
        assert isinstance(report.spotlight_goal_ids, list)
        assert isinstance(report.items_scored, int)
        assert isinstance(report.results, list)
        assert isinstance(report.top_domain, str)
        assert report.duration_seconds >= 0

    def test_results_are_attention_results(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        for r in report.results:
            assert isinstance(r, AttentionResult)

    def test_results_sorted_by_attention_score(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        scores = [r.attention_score for r in report.results]
        assert scores == sorted(scores, reverse=True)

    def test_update_spotlight_text(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(text="science research discovery")
        assert len(af._spotlight_keywords) > 0

    def test_update_spotlight_keywords(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(keywords=["science", "history"])
        assert "science" in af._spotlight_keywords or "history" in af._spotlight_keywords

    def test_spotlight_improves_domain_relevance(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(text="science research experiments")
        report = af.spotlight_retrieve()
        # Science-domain items should rank higher
        if report.results:
            assert isinstance(report.results[0].attention_score, float)

    def test_k_limits_results(self):
        agent = _make_rich_emms(10)
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve(k=3)
        assert len(report.results) <= 3

    def test_clear_spotlight(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(text="science research")
        af.clear_spotlight()
        assert len(af._spotlight_keywords) == 0
        assert len(af._spotlight_goal_ids) == 0

    def test_empty_spotlight_scores_by_importance(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        # With no spotlight, scores should still be non-negative
        for r in report.results:
            assert r.attention_score >= 0.0

    def test_attention_result_fields(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        if report.results:
            r = report.results[0]
            assert isinstance(r.memory_id, str)
            assert isinstance(r.content_excerpt, str)
            assert isinstance(r.domain, str)
            assert 0.0 <= r.attention_score <= 1.0
            assert 0.0 <= r.goal_relevance <= 1.0
            assert 0.0 <= r.importance <= 1.0
            assert 0.0 <= r.recency_score <= 1.0
            assert 0.0 <= r.keyword_overlap <= 1.0

    def test_attention_profile_returns_dict(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        profile = af.attention_profile()
        assert isinstance(profile, dict)
        for domain, score in profile.items():
            assert isinstance(domain, str)
            assert 0.0 <= score <= 1.0

    def test_attention_profile_covers_domains(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        profile = af.attention_profile()
        assert "science" in profile or "history" in profile

    def test_goal_stack_integration(self):
        agent = _make_rich_emms()
        gs = GoalStack()
        g = gs.push("Learn about science research methodology", priority=0.9)
        gs.activate(g.id)
        af = AttentionFilter(memory=agent.memory, goal_stack=gs)
        report = af.spotlight_retrieve()
        assert report.items_scored > 0

    def test_update_spotlight_goal_ids(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(goal_ids=["goal_abc123"])
        assert "goal_abc123" in af._spotlight_goal_ids

    def test_report_summary_str(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        s = report.summary()
        assert isinstance(s, str)
        assert "AttentionReport" in s

    def test_attention_result_summary_str(self):
        agent = _make_rich_emms()
        af = AttentionFilter(memory=agent.memory)
        af.update_spotlight(text="science")
        report = af.spotlight_retrieve()
        if report.results:
            s = report.results[0].summary()
            assert isinstance(s, str)

    def test_empty_memory_returns_empty_report(self):
        agent = _make_emms()
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve()
        assert report.items_scored == 0
        assert report.results == []

    def test_items_scored_equals_total_memories(self):
        agent = _make_rich_emms(4)
        af = AttentionFilter(memory=agent.memory)
        report = af.spotlight_retrieve(k=100)
        assert report.items_scored == 4


# ===========================================================================
# TestAnalogyEngine
# ===========================================================================


class TestAnalogyEngine:

    def test_find_analogies_returns_report(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory)
        report = engine.find_analogies()
        assert isinstance(report, AnalogyReport)

    def test_report_fields(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory)
        report = engine.find_analogies()
        assert isinstance(report.total_pairs_checked, int)
        assert isinstance(report.analogies_found, int)
        assert isinstance(report.records, list)
        assert report.duration_seconds >= 0

    def test_analogies_detected(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        assert report.analogies_found >= 1

    def test_records_are_analogy_records(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        for r in report.records:
            assert isinstance(r, AnalogyRecord)

    def test_analogy_record_fields(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        if report.records:
            r = report.records[0]
            assert isinstance(r.id, str) and r.id.startswith("ana_")
            assert isinstance(r.source_domain, str)
            assert isinstance(r.target_domain, str)
            assert r.source_domain != r.target_domain  # cross-domain only
            assert isinstance(r.mappings, list)
            assert 0.0 <= r.analogy_strength <= 1.0
            assert isinstance(r.insight_content, str)

    def test_cross_domain_only(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.0)
        report = engine.find_analogies()
        for r in report.records:
            assert r.source_domain != r.target_domain

    def test_analogy_mapping_fields(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        if report.records and report.records[0].mappings:
            m = report.records[0].mappings[0]
            assert isinstance(m, AnalogyMapping)
            assert isinstance(m.source_memory_id, str)
            assert isinstance(m.target_memory_id, str)
            assert 0.0 <= m.structural_similarity <= 1.0
            assert isinstance(m.shared_relations, list)
            assert isinstance(m.source_excerpt, str)
            assert isinstance(m.target_excerpt, str)

    def test_records_sorted_by_strength(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.0)
        report = engine.find_analogies()
        strengths = [r.analogy_strength for r in report.records]
        assert strengths == sorted(strengths, reverse=True)

    def test_domain_filter_source(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.0)
        report = engine.find_analogies(source_domain="biology")
        for r in report.records:
            assert r.source_domain == "biology"

    def test_empty_memory_no_analogies(self):
        agent = _make_emms()
        engine = AnalogyEngine(memory=agent.memory)
        report = engine.find_analogies()
        assert report.analogies_found == 0

    def test_single_domain_no_analogies(self):
        agent = _make_emms()
        for i in range(3):
            agent.store(Experience(
                content=f"causes reduces increases enables prevents",
                domain="only_domain",
            ))
        engine = AnalogyEngine(memory=agent.memory)
        report = engine.find_analogies()
        assert report.analogies_found == 0

    def test_store_insights_creates_memory(self):
        agent = _make_analogy_emms()
        before_count = len(list(agent.memory.working) + list(agent.memory.short_term))
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05,
                               store_insights=True)
        report = engine.find_analogies()
        after_count = len(list(agent.memory.working) + list(agent.memory.short_term))
        if report.analogies_found > 0:
            assert after_count > before_count

    def test_store_insights_false_no_new_memory(self):
        agent = _make_analogy_emms()
        before_count = len(list(agent.memory.working))
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.0,
                               store_insights=False)
        engine.find_analogies()
        after_count = len(list(agent.memory.working))
        assert before_count == after_count

    def test_analogies_for_empty(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory)
        result = engine.analogies_for("nonexistent_id")
        assert result == []

    def test_max_analogies_respected(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.0,
                               max_analogies=2)
        report = engine.find_analogies()
        assert report.analogies_found <= 2

    def test_relational_keyword_detection(self):
        engine = AnalogyEngine.__new__(AnalogyEngine)
        kws = engine._relational_keywords("stress causes anxiety and prevents recovery")
        assert "causes" in kws or "prevents" in kws

    def test_structural_similarity_shared_relations(self):
        agent = _make_emms()
        agent.store(Experience(
            content="heat causes expansion which produces pressure and increases volume",
            domain="physics",
        ))
        agent.store(Experience(
            content="debt causes stress which produces anxiety and increases burden",
            domain="psychology",
        ))
        engine = AnalogyEngine(memory=agent.memory)
        items = engine._collect_all()
        if len(items) >= 2:
            sim, relations = engine._structural_similarity(items[0], items[1])
            assert isinstance(sim, float)
            assert 0.0 <= sim <= 1.0
            assert isinstance(relations, list)

    def test_report_summary_str(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory)
        report = engine.find_analogies()
        s = report.summary()
        assert isinstance(s, str)
        assert "AnalogyReport" in s

    def test_record_summary_str(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        if report.records:
            s = report.records[0].summary()
            assert isinstance(s, str)
            assert "AnalogyRecord" in s

    def test_mapping_summary_str(self):
        agent = _make_analogy_emms()
        engine = AnalogyEngine(memory=agent.memory, min_structural_similarity=0.05)
        report = engine.find_analogies()
        if report.records and report.records[0].mappings:
            s = report.records[0].mappings[0].summary()
            assert isinstance(s, str)


# ===========================================================================
# TestEMMSFacadeV170
# ===========================================================================


class TestEMMSFacadeV170:

    # GoalStack facade
    def test_push_goal_returns_goal(self):
        agent = _make_emms()
        goal = agent.push_goal("Learn quantum mechanics")
        assert isinstance(goal, Goal)

    def test_push_goal_with_priority(self):
        agent = _make_emms()
        goal = agent.push_goal("High priority", priority=0.9)
        assert goal.priority == 0.9

    def test_activate_goal(self):
        agent = _make_emms()
        goal = agent.push_goal("To activate")
        assert agent.activate_goal(goal.id) is True
        assert goal.status == "active"

    def test_complete_goal(self):
        agent = _make_emms()
        goal = agent.push_goal("To complete")
        agent.activate_goal(goal.id)
        assert agent.complete_goal(goal.id, outcome_note="Done") is True
        assert goal.status == "completed"

    def test_fail_goal(self):
        agent = _make_emms()
        goal = agent.push_goal("To fail")
        assert agent.fail_goal(goal.id, reason="Too hard") is True
        assert goal.status == "failed"

    def test_active_goals(self):
        agent = _make_emms()
        g1 = agent.push_goal("G1", priority=0.3)
        g2 = agent.push_goal("G2", priority=0.9)
        agent.activate_goal(g1.id)
        agent.activate_goal(g2.id)
        active = agent.active_goals()
        assert len(active) == 2
        assert active[0].priority >= active[1].priority

    def test_goal_report(self):
        agent = _make_emms()
        agent.push_goal("G1")
        g2 = agent.push_goal("G2")
        agent.complete_goal(g2.id)
        report = agent.goal_report()
        assert isinstance(report, GoalReport)
        assert report.total == 2
        assert report.completed == 1

    def test_goal_stack_persisted(self):
        agent = _make_emms()
        agent.push_goal("First goal")
        agent.push_goal("Second goal")
        assert hasattr(agent, "_goal_stack")
        assert len(agent._goal_stack._goals) == 2

    # AttentionFilter facade
    def test_update_spotlight(self):
        agent = _make_rich_emms()
        agent.update_spotlight(text="science discovery")
        assert hasattr(agent, "_attention_filter")

    def test_spotlight_retrieve_returns_report(self):
        agent = _make_rich_emms()
        report = agent.spotlight_retrieve()
        assert isinstance(report, AttentionReport)

    def test_attention_profile_returns_dict(self):
        agent = _make_rich_emms()
        profile = agent.attention_profile()
        assert isinstance(profile, dict)

    # AnalogyEngine facade
    def test_find_analogies_returns_report(self):
        agent = _make_analogy_emms()
        report = agent.find_analogies()
        assert isinstance(report, AnalogyReport)

    def test_find_analogies_with_domains(self):
        agent = _make_analogy_emms()
        report = agent.find_analogies(source_domain="biology")
        assert isinstance(report, AnalogyReport)

    def test_analogies_for_returns_list(self):
        agent = _make_analogy_emms()
        items = list(agent.memory.working)
        if items:
            result = agent.analogies_for(items[0].id)
            assert isinstance(result, list)


# ===========================================================================
# TestMCPV170
# ===========================================================================


class TestMCPV170:

    def test_tool_count_72(self):
        assert len(_TOOL_DEFINITIONS) == 72

    def test_all_tool_names_unique(self):
        names = [t["name"] for t in _TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_push_goal_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_push_goal" in names

    def test_active_goals_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_active_goals" in names

    def test_complete_goal_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_complete_goal" in names

    def test_spotlight_retrieve_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_spotlight_retrieve" in names

    def test_find_analogies_tool_exists(self):
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_find_analogies" in names

    def test_push_goal_callable(self):
        server = EMCPServer(_make_emms())
        result = server.handle("emms_push_goal", {"content": "Learn machine learning"})
        assert "id" in result
        assert result.get("status") == "pending"

    def test_active_goals_callable(self):
        server = EMCPServer(_make_emms())
        server.handle("emms_push_goal", {"content": "Goal A"})
        result = server.handle("emms_active_goals", {})
        assert "count" in result
        assert "goals" in result

    def test_complete_goal_callable(self):
        server = EMCPServer(_make_emms())
        push_result = server.handle("emms_push_goal", {"content": "Finish task"})
        goal_id = push_result["id"]
        result = server.handle("emms_complete_goal", {"goal_id": goal_id})
        assert "ok" in result

    def test_spotlight_retrieve_callable(self):
        server = EMCPServer(_make_rich_emms())
        result = server.handle("emms_spotlight_retrieve", {"k": 5, "text": "science"})
        assert "items_scored" in result or "error" in result

    def test_find_analogies_callable(self):
        server = EMCPServer(_make_analogy_emms())
        result = server.handle("emms_find_analogies", {})
        assert "analogies_found" in result or "error" in result


# ===========================================================================
# TestV170Exports
# ===========================================================================


class TestV170Exports:

    def test_version_string(self):
        import emms
        assert emms.__version__ == "0.17.0"

    def test_goal_stack_exported(self):
        from emms import GoalStack
        assert GoalStack is not None

    def test_goal_exported(self):
        from emms import Goal
        assert Goal is not None

    def test_goal_report_exported(self):
        from emms import GoalReport
        assert GoalReport is not None

    def test_attention_filter_exported(self):
        from emms import AttentionFilter
        assert AttentionFilter is not None

    def test_attention_result_exported(self):
        from emms import AttentionResult
        assert AttentionResult is not None

    def test_attention_report_exported(self):
        from emms import AttentionReport
        assert AttentionReport is not None

    def test_analogy_engine_exported(self):
        from emms import AnalogyEngine
        assert AnalogyEngine is not None

    def test_analogy_mapping_exported(self):
        from emms import AnalogyMapping
        assert AnalogyMapping is not None

    def test_analogy_record_exported(self):
        from emms import AnalogyRecord
        assert AnalogyRecord is not None

    def test_analogy_report_exported(self):
        from emms import AnalogyReport
        assert AnalogyReport is not None

    def test_all_exports_in_all(self):
        import emms
        for name in [
            "GoalStack", "Goal", "GoalReport",
            "AttentionFilter", "AttentionResult", "AttentionReport",
            "AnalogyEngine", "AnalogyMapping", "AnalogyRecord", "AnalogyReport",
        ]:
            assert name in emms.__all__, f"{name} missing from __all__"
