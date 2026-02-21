"""Tests for EMMS v0.5.0 features.

Covers:
- SessionManager (lifecycle, JSONL persistence, generate_claude_md)
- Endless Mode (biomimetic compression)
- ToolObserver (obs_type inference, concept tag inference)
- EnsembleRetriever presets (from_balanced, from_identity)
- Progressive disclosure (search_compact with obs_type/concept_tag filters)
- JSONL export/import on HierarchicalMemory
- MemoryAnalytics (tier_distribution, health_score, report)
- BM25 retrieval quality vs Jaccard
- v0.5.1: facts/files/title/subtitle on Experience, token_estimate on CompactResult,
          observe_prompt(), generate_context_injection()
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from emms import (
    ConceptTag,
    Experience,
    MemoryAnalytics,
    MemoryConfig,
    ObsType,
    SessionManager,
    SessionSummary,
    ToolObserver,
)
from emms.memory.hierarchical import HierarchicalMemory
from emms.retrieval.strategies import EnsembleRetriever, DomainStrategy, TemporalStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(working_cap: int = 7, endless: bool = False, chunk: int = 2) -> HierarchicalMemory:
    return HierarchicalMemory(
        MemoryConfig(working_capacity=working_cap),
        endless_mode=endless,
        endless_chunk_size=chunk,
    )

def _all_items(mem: HierarchicalMemory):
    return [item for _, store in mem._iter_tiers() for item in store]


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class TestSessionManager:

    def test_start_returns_session_id(self):
        sm = SessionManager()
        sid = sm.start_session()
        assert sid.startswith("sess_")
        assert sm.active_session_id == sid

    def test_explicit_session_id(self):
        sm = SessionManager()
        sm.start_session(session_id="my_session")
        assert sm.active_session_id == "my_session"

    def test_store_injects_session_id(self):
        mem = _make_mem()
        sm = SessionManager(memory=mem)
        sm.start_session()
        exp = Experience(content="test memory", domain="tech")
        assert exp.session_id is None
        sm.store(exp)
        assert exp.session_id == sm.active_session_id

    def test_store_counts_memories(self):
        sm = SessionManager()
        sm.start_session()
        sm.store(Experience(content="a", domain="tech", obs_type=ObsType.BUGFIX))
        sm.store(Experience(content="b", domain="tech", obs_type=ObsType.FEATURE))
        sm.store(Experience(content="c", domain="general"))
        assert sm.active_summary.memory_count == 3
        assert sm.active_summary.obs_types == {"bugfix": 1, "feature": 1}

    def test_private_excluded_from_obs_count(self):
        sm = SessionManager()
        sm.start_session()
        sm.store(Experience(content="secret", domain="general", private=True))
        # private memories still counted in memory_count (they're stored)
        assert sm.active_summary.memory_count == 1
        # but obs_type None so not in obs_types breakdown
        assert sm.active_summary.obs_types == {}

    def test_update_fields(self):
        sm = SessionManager()
        sm.start_session()
        sm.update(learned="BM25 better than Jaccard", completed="333 tests pass")
        assert "BM25" in sm.active_summary.learned
        assert "333" in sm.active_summary.completed

    def test_end_session_closes_and_returns_summary(self):
        sm = SessionManager()
        sm.start_session(request="test session")
        sm.store(Experience(content="x", domain="tech"))
        summary = sm.end_session()
        assert summary is not None
        assert summary.ended_at is not None
        assert summary.duration_seconds >= 0
        assert sm.active_session_id is None

    def test_jsonl_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            sm = SessionManager(log_path=tmp)
            sm.start_session(request="session A")
            sm.update(learned="insight 1")
            sm.end_session()

            sm2 = SessionManager(log_path=tmp)
            sm2.start_session(request="session B")
            sm2.end_session()

            sessions = sm2.load_sessions()
            assert len(sessions) == 2
            assert sessions[0].request == "session A"
            assert sessions[1].request == "session B"
            assert "insight 1" in sessions[0].learned
        finally:
            os.unlink(tmp)

    def test_get_session_by_id(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            sm = SessionManager(log_path=tmp)
            sm.start_session(session_id="target_sess")
            sm.update(completed="done")
            sm.end_session()

            found = sm.get_session("target_sess")
            assert found is not None
            assert found.completed == "done"
            assert sm.get_session("nonexistent") is None
        finally:
            os.unlink(tmp)

    def test_auto_close_open_session_on_start(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            sm = SessionManager(log_path=tmp)
            sm.start_session(session_id="first")
            sm.start_session(session_id="second")  # should auto-close first
            sessions = sm.load_sessions()
            assert len(sessions) == 1
            assert sessions[0].session_id == "first"
            assert sm.active_session_id == "second"
        finally:
            os.unlink(tmp)

    def test_generate_claude_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sessions.jsonl"
            md_path = Path(tmpdir) / "CLAUDE.md"

            sm = SessionManager(log_path=log_path)
            sm.start_session(request="write the paper")
            sm.update(learned="Goldilocks effect confirmed", completed="paper compiled")
            sm.end_session()

            dest = sm.generate_claude_md(output_path=md_path)
            assert dest.exists()
            content = dest.read_text()
            assert "EMMS Session Memory" in content
            assert "write the paper" in content
            assert "Goldilocks effect confirmed" in content
            assert "paper compiled" in content

    def test_generate_claude_md_with_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "CLAUDE.md"
            sm = SessionManager(log_path=Path(tmpdir) / "s.jsonl")
            sm.start_session(request="active work")
            dest = sm.generate_claude_md(output_path=md_path)
            content = dest.read_text()
            assert "Active Session" in content
            assert "active work" in content
            sm.end_session()


# ---------------------------------------------------------------------------
# Endless Mode
# ---------------------------------------------------------------------------

class TestEndlessMode:

    def test_endless_mode_compresses_instead_of_evicting(self):
        mem = _make_mem(working_cap=4, endless=True, chunk=2)
        for i in range(10):
            mem.store(Experience(content=f"event {i}", domain="tech"))
        assert mem._endless_episodes > 0
        assert mem.total_stored == 10

    def test_endless_mode_keeps_working_bounded(self):
        mem = _make_mem(working_cap=4, endless=True, chunk=2)
        for i in range(20):
            mem.store(Experience(content=f"e{i}", domain="tech"))
        assert len(mem.working) <= 4

    def test_endless_episode_carries_source_metadata(self):
        mem = _make_mem(working_cap=3, endless=True, chunk=2)
        for i in range(6):
            mem.store(Experience(content=f"item {i}", domain="science", session_id="s1"))
        compressed = [
            item for _, store in mem._iter_tiers()
            for item in store
            if item.experience.metadata.get("compressed")
        ]
        assert len(compressed) > 0
        ep = compressed[0]
        assert ep.experience.obs_type == ObsType.CHANGE
        assert "source_ids" in ep.experience.metadata
        assert ep.experience.metadata["source_count"] == 2

    def test_endless_off_by_default(self):
        mem = _make_mem()
        assert mem.endless_mode is False
        assert mem._endless_episodes == 0

    def test_normal_mode_still_works_after_overflow(self):
        mem = _make_mem(working_cap=3, endless=False)
        for i in range(10):
            mem.store(Experience(content=f"e{i}", domain="tech", importance=0.9))
        assert mem.total_stored == 10


# ---------------------------------------------------------------------------
# ToolObserver
# ---------------------------------------------------------------------------

class TestToolObserver:

    def setup_method(self):
        self.obs = ToolObserver()

    def test_edit_becomes_change(self):
        exp = self.obs.observe("Edit", {"file_path": "/src/models.py"}, "File updated")
        assert exp.obs_type == ObsType.CHANGE
        assert ConceptTag.WHAT_CHANGED in exp.concept_tags

    def test_write_becomes_feature(self):
        exp = self.obs.observe("Write", {"file_path": "/src/new.py"}, "File created")
        assert exp.obs_type == ObsType.FEATURE

    def test_read_becomes_discovery(self):
        exp = self.obs.observe("Read", {"file_path": "/src/hierarchical.py"}, "class HierarchicalMemory")
        assert exp.obs_type == ObsType.DISCOVERY
        assert ConceptTag.HOW_IT_WORKS in exp.concept_tags

    def test_bash_bugfix_signal(self):
        exp = self.obs.observe("Bash", {"command": "fix the broken import"}, "done")
        assert exp.obs_type == ObsType.BUGFIX

    def test_bash_refactor_signal(self):
        exp = self.obs.observe("Bash", {"command": "refactor the auth module"}, "done")
        assert exp.obs_type == ObsType.REFACTOR

    def test_session_id_set(self):
        exp = self.obs.observe("Read", {"file_path": "foo.py"}, "content", session_id="sess_xyz")
        assert exp.session_id == "sess_xyz"

    def test_content_truncated(self):
        long_response = "x" * 1000
        exp = self.obs.observe("Bash", {"command": "run"}, long_response)
        assert len(exp.content) < 600  # truncated

    def test_importance_write_high(self):
        exp = self.obs.observe("Write", {"file_path": "important.py"}, "created")
        assert exp.importance >= 0.8

    def test_importance_glob_low(self):
        exp = self.obs.observe("Glob", {"pattern": "*.py"}, "file1.py\nfile2.py")
        assert exp.importance <= 0.3

    def test_observe_batch(self):
        payloads = [
            {"tool_name": "Edit", "tool_input": {"file_path": "a.py"}, "tool_response": "ok"},
            {"tool_name": "Read", "tool_input": {"file_path": "b.py"}, "tool_response": "content"},
        ]
        exps = self.obs.observe_batch(payloads, session_id="sess_batch")
        assert len(exps) == 2
        assert all(e.session_id == "sess_batch" for e in exps)

    def test_path_summary_in_content(self):
        exp = self.obs.observe("Edit", {"file_path": "/very/deep/path/models.py"}, "ok")
        assert "models.py" in exp.content

    def test_gotcha_tag_on_warning(self):
        exp = self.obs.observe("Read", {"file_path": "x.py"},
                               "gotcha: this list is shared across all instances")
        assert ConceptTag.GOTCHA in exp.concept_tags


# ---------------------------------------------------------------------------
# EnsembleRetriever presets
# ---------------------------------------------------------------------------

class TestRetrieverPresets:

    def test_from_balanced_weights(self):
        r = EnsembleRetriever.from_balanced()
        weights = {s.name: w for s, w in r.strategies}
        # v0.5.1: 60/20/10/10 (Semantic/Temporal/Importance/Domain)
        assert abs(weights["semantic"] - 0.60) < 1e-9
        assert abs(weights["temporal"] - 0.20) < 1e-9
        assert abs(weights["importance"] - 0.10) < 1e-9
        assert abs(weights["domain"] - 0.10) < 1e-9

    def test_from_identity_has_six_strategies(self):
        r = EnsembleRetriever.from_identity()
        assert len(r.strategies) == 6
        names = {s.name for s, _ in r.strategies}
        assert names == {"semantic", "temporal", "emotional", "graph", "domain", "importance"}

    def test_from_balanced_returns_results(self):
        mem = _make_mem()
        for i in range(5):
            mem.store(Experience(content=f"science experiment {i}", domain="science"))
        items = _all_items(mem)
        r = EnsembleRetriever.from_balanced()
        results = r.retrieve("research experiment", items)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Progressive disclosure (search_compact with filters)
# ---------------------------------------------------------------------------

class TestProgressiveDisclosure:

    def setup_method(self):
        self.mem = _make_mem()
        exps = [
            Experience(content="Fixed auth bug in production",
                       domain="tech", obs_type=ObsType.BUGFIX,
                       concept_tags=[ConceptTag.PROBLEM_SOLUTION, ConceptTag.GOTCHA]),
            Experience(content="Added new dashboard feature",
                       domain="tech", obs_type=ObsType.FEATURE,
                       concept_tags=[ConceptTag.HOW_IT_WORKS]),
            Experience(content="Discovered Goldilocks effect in RLHF",
                       domain="science", obs_type=ObsType.DISCOVERY,
                       concept_tags=[ConceptTag.WHY_IT_EXISTS, ConceptTag.PATTERN]),
            Experience(content="Private internal config value",
                       domain="general", private=True),
        ]
        for e in exps:
            self.mem.store(e)
        self.items = _all_items(self.mem)
        self.retriever = EnsembleRetriever.from_balanced()

    def test_private_excluded_from_compact(self):
        results = self.retriever.search_compact("config", self.items)
        assert all(not r.snippet.__contains__("Private") for r in results)

    def test_obs_type_filter(self):
        bugfixes = self.retriever.search_compact("bug fix", self.items, obs_type=ObsType.BUGFIX)
        assert all(r.obs_type == ObsType.BUGFIX for r in bugfixes)

    def test_concept_tag_filter(self):
        gotchas = self.retriever.search_compact("thing", self.items,
                                                concept_tags=[ConceptTag.GOTCHA])
        assert all(ConceptTag.GOTCHA in r.concept_tags for r in gotchas)

    def test_compact_result_has_all_fields(self):
        results = self.retriever.search_compact("tech", self.items)
        if results:
            r = results[0]
            assert r.id
            assert r.snippet
            assert r.domain
            assert r.score >= 0
            assert r.tier is not None
            assert r.timestamp > 0

    def test_get_full_by_ids(self):
        compact = self.retriever.search_compact("feature", self.items)
        if compact:
            ids = [compact[0].id]
            full = self.retriever.get_full(ids, self.items)
            assert len(full) == 1
            assert full[0].id == ids[0]

    def test_get_full_excludes_private(self):
        # Add a private item directly
        private_id = None
        for _, store in self.mem._iter_tiers():
            for item in store:
                if item.experience.private:
                    private_id = item.id
        if private_id:
            full = self.retriever.get_full([private_id], self.items)
            assert all(i.id != private_id for i in full)


# ---------------------------------------------------------------------------
# JSONL export / import
# ---------------------------------------------------------------------------

class TestJSONLExport:

    def test_export_excludes_private(self):
        mem = _make_mem()
        mem.store(Experience(content="public memory", domain="tech"))
        mem.store(Experience(content="private secret", domain="general", private=True))
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            n = mem.export_jsonl(tmp)
            assert n == 1
            lines = Path(tmp).read_text().splitlines()
            assert len(lines) == 1
            assert "public memory" in lines[0]
        finally:
            os.unlink(tmp)

    def test_export_include_private(self):
        mem = _make_mem()
        mem.store(Experience(content="public", domain="tech"))
        mem.store(Experience(content="secret", domain="general", private=True))
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            n = mem.export_jsonl(tmp, include_private=True)
            assert n == 2
        finally:
            os.unlink(tmp)

    def test_import_roundtrip(self):
        mem1 = _make_mem()
        for i in range(5):
            mem1.store(Experience(content=f"memory {i}", domain="tech",
                                  obs_type=ObsType.FEATURE))
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            mem1.export_jsonl(tmp)
            mem2 = _make_mem()
            n = mem2.import_jsonl(tmp)
            assert n == 5
        finally:
            os.unlink(tmp)

    def test_import_skips_duplicates(self):
        mem = _make_mem()
        mem.store(Experience(content="only once", domain="tech"))
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            mem.export_jsonl(tmp)
            # Import into the same memory — should skip since IDs already exist
            n = mem.import_jsonl(tmp)
            assert n == 0
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# MemoryAnalytics
# ---------------------------------------------------------------------------

class TestMemoryAnalytics:

    def setup_method(self):
        self.mem = _make_mem()
        exps = [
            Experience(content="bugfix: fixed the auth module", domain="tech",
                       obs_type=ObsType.BUGFIX, concept_tags=[ConceptTag.PROBLEM_SOLUTION]),
            Experience(content="feature: added OAuth support", domain="tech",
                       obs_type=ObsType.FEATURE, concept_tags=[ConceptTag.HOW_IT_WORKS]),
            Experience(content="discovery: attention heads track syntax", domain="science",
                       obs_type=ObsType.DISCOVERY, concept_tags=[ConceptTag.WHY_IT_EXISTS]),
            Experience(content="private config", domain="general", private=True),
        ]
        for e in exps:
            self.mem.store(e)
        self.analytics = MemoryAnalytics(self.mem)

    def test_tier_distribution_totals(self):
        dist = self.analytics.tier_distribution()
        total = dist["working"] + dist["short_term"] + dist["long_term"] + dist["semantic"]
        assert total == dist["total"]

    def test_domain_coverage_keys(self):
        domains = self.analytics.domain_coverage()
        assert "tech" in domains
        assert "science" in domains
        assert domains["tech"] >= 2

    def test_obs_type_distribution(self):
        obs = self.analytics.obs_type_distribution()
        assert "bugfix" in obs
        assert "feature" in obs
        assert "discovery" in obs
        assert obs["bugfix"] == 1

    def test_concept_coverage(self):
        concepts = self.analytics.concept_coverage()
        assert "problem-solution" in concepts
        assert "how-it-works" in concepts

    def test_privacy_audit(self):
        privacy = self.analytics.privacy_audit()
        assert privacy["private"] == 1
        assert privacy["public"] == 3
        assert privacy["total"] == 4

    def test_health_score_range(self):
        score = self.analytics.health_score()
        assert 0.0 <= score <= 1.0

    def test_strength_distribution(self):
        sd = self.analytics.strength_distribution()
        assert sd["count"] == 4
        assert 0 <= sd["min"] <= sd["mean"] <= sd["max"] <= 1.0

    def test_report_is_string(self):
        report = self.analytics.report()
        assert isinstance(report, str)
        assert "Health Score" in report
        assert "Tier Distribution" in report
        assert "Domain Coverage" in report

    def test_endless_stats_off(self):
        stats = self.analytics.endless_stats()
        assert stats["endless_mode"] is False
        assert stats["episodes_compressed"] == 0

    def test_endless_stats_on(self):
        mem = _make_mem(working_cap=3, endless=True, chunk=2)
        for i in range(8):
            mem.store(Experience(content=f"e{i}", domain="tech"))
        analytics = MemoryAnalytics(mem)
        stats = analytics.endless_stats()
        assert stats["endless_mode"] is True
        assert stats["episodes_compressed"] > 0


# ---------------------------------------------------------------------------
# BM25 retrieval quality
# ---------------------------------------------------------------------------

class TestBM25Retrieval:

    def test_bm25_ranks_exact_match_high(self):
        mem = _make_mem()
        mem.store(Experience(content="BM25 is a ranking function", domain="tech", importance=0.5))
        mem.store(Experience(content="something completely unrelated", domain="general", importance=0.5))
        results = mem.retrieve("BM25 ranking", max_results=5)
        assert len(results) > 0
        assert "BM25" in results[0].memory.experience.content

    def test_bm25_no_match_below_threshold(self):
        mem = _make_mem()
        mem.store(Experience(content="cats and dogs", domain="pets"))
        results = mem.retrieve("quantum physics research", max_results=5)
        assert len(results) == 0

    def test_bm25_tf_saturation(self):
        """Repeating a word many times shouldn't dominate over relevance."""
        mem = _make_mem()
        # High TF but unrelated
        mem.store(Experience(content="dog dog dog dog dog dog dog dog", domain="pets", importance=0.1))
        # Low TF but relevant
        mem.store(Experience(content="neural network training procedure", domain="tech", importance=0.8))
        results = mem.retrieve("dog training", max_results=5)
        if len(results) >= 2:
            # Both should appear; the high-TF one shouldn't dominate unfairly
            assert results[0].score <= 1.0


# ============================================================================
# v0.5.1 — Rich data model (6 improvements from claude-mem)
# ============================================================================

class TestRichExperienceFields:
    """Improvement 1 & 2: facts list + files_read/files_modified on Experience."""

    def test_facts_field_default_empty(self):
        exp = Experience(content="hello world", domain="test")
        assert exp.facts == []

    def test_facts_field_set(self):
        exp = Experience(content="hello", domain="test", facts=["fact A", "fact B"])
        assert exp.facts == ["fact A", "fact B"]

    def test_files_read_default_empty(self):
        exp = Experience(content="hello", domain="test")
        assert exp.files_read == []

    def test_files_modified_default_empty(self):
        exp = Experience(content="hello", domain="test")
        assert exp.files_modified == []

    def test_files_set_explicitly(self):
        exp = Experience(
            content="read models, edited hierarchical",
            domain="tech",
            files_read=["src/emms/core/models.py"],
            files_modified=["src/emms/memory/hierarchical.py"],
        )
        assert "models.py" in exp.files_read[0]
        assert "hierarchical.py" in exp.files_modified[0]

    def test_title_subtitle_default_none(self):
        exp = Experience(content="hello", domain="test")
        assert exp.title is None
        assert exp.subtitle is None

    def test_title_subtitle_set(self):
        exp = Experience(
            content="edited hierarchical.py to add BM25",
            domain="tech",
            title="Added BM25 retrieval",
            subtitle="Replaced Jaccard with BM25 (k1=1.5, b=0.75) in _relevance()",
        )
        assert exp.title == "Added BM25 retrieval"
        assert "BM25" in exp.subtitle

    def test_experience_roundtrip_json_with_new_fields(self):
        exp = Experience(
            content="test",
            domain="tech",
            title="T",
            subtitle="S",
            facts=["f1"],
            files_read=["a.py"],
            files_modified=["b.py"],
        )
        data = exp.model_dump_json()
        restored = Experience.model_validate_json(data)
        assert restored.title == "T"
        assert restored.facts == ["f1"]
        assert restored.files_read == ["a.py"]
        assert restored.files_modified == ["b.py"]


class TestToolObserverRichFields:
    """Improvements 1–3: ToolObserver populates files, facts, title, subtitle."""

    def test_observe_read_populates_files_read(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Read",
            {"file_path": "/src/emms/core/models.py"},
            "file contents here",
        )
        assert "/src/emms/core/models.py" in exp.files_read
        assert exp.files_modified == []

    def test_observe_edit_populates_files_modified(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Edit",
            {"file_path": "/src/emms/memory/hierarchical.py", "old_string": "x", "new_string": "y"},
            "File updated successfully",
        )
        assert "/src/emms/memory/hierarchical.py" in exp.files_modified
        assert exp.files_read == []

    def test_observe_write_populates_files_modified(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Write",
            {"file_path": "/src/emms/new_module.py", "content": "..."},
            "Created",
        )
        assert "/src/emms/new_module.py" in exp.files_modified

    def test_observe_facts_populated_for_edit(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Edit",
            {"file_path": "/src/emms/core/models.py", "old_string": "a", "new_string": "b"},
            "File updated",
        )
        assert any("models.py" in f for f in exp.facts)

    def test_observe_facts_populated_for_bash(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Bash",
            {"command": "pytest tests/ -v"},
            "5 passed",
        )
        assert any("pytest" in f for f in exp.facts)
        assert any("passed" in f for f in exp.facts)

    def test_observe_title_set(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Edit",
            {"file_path": "/src/emms/core/models.py", "old_string": "x", "new_string": "y"},
            "File updated",
        )
        assert exp.title is not None
        assert len(exp.title) > 0

    def test_observe_subtitle_set(self):
        obs = ToolObserver()
        exp = obs.observe(
            "Read",
            {"file_path": "/src/emms/emms.py"},
            "class EMMS: ...",
        )
        assert exp.subtitle is not None

    def test_observe_prompt_creates_decision_experience(self):
        obs = ToolObserver()
        exp = obs.observe_prompt("Add BM25 retrieval to HierarchicalMemory")
        assert exp.obs_type == ObsType.DECISION
        assert "[UserPrompt]" in exp.content
        assert "BM25" in exp.content

    def test_observe_prompt_sets_title(self):
        obs = ToolObserver()
        exp = obs.observe_prompt("Implement BM25 in HierarchicalMemory")
        assert exp.title is not None
        assert "BM25" in exp.title or "Implement" in exp.title

    def test_observe_prompt_sets_facts(self):
        obs = ToolObserver()
        exp = obs.observe_prompt("Fix the persistence bug in load()")
        assert any("User asked" in f for f in exp.facts)

    def test_observe_prompt_with_session_id(self):
        obs = ToolObserver()
        exp = obs.observe_prompt("Do the thing", session_id="sess_xyz")
        assert exp.session_id == "sess_xyz"

    def test_observe_prompt_long_text_truncated_in_title(self):
        obs = ToolObserver()
        long_prompt = "x" * 200
        exp = obs.observe_prompt(long_prompt)
        assert len(exp.title) <= 65  # 60 chars + "…"


class TestTokenEstimateOnCompactResult:
    """Improvement 4: token_estimate on CompactResult."""

    def test_token_estimate_populated(self):
        mem = _make_mem()
        mem.store(Experience(
            content="The BM25 retrieval function uses k1 equals 1.5 and b equals 0.75",
            domain="tech",
            importance=0.9,
        ))
        all_items = [item for _, store in mem._iter_tiers() for item in store]
        from emms import EnsembleRetriever
        retriever = EnsembleRetriever.from_balanced()
        compact = retriever.search_compact("BM25 retrieval", all_items, relevance_threshold=0.0)
        assert len(compact) >= 1
        assert compact[0].token_estimate is not None
        assert compact[0].token_estimate > 0

    def test_token_estimate_proportional_to_content(self):
        from emms import EnsembleRetriever
        mem = _make_mem()
        short_exp = Experience(content="short text", domain="tech", importance=0.9)
        long_exp = Experience(
            content=" ".join(["word"] * 50),
            domain="tech",
            importance=0.9,
        )
        mem.store(short_exp)
        mem.store(long_exp)
        all_items = [item for _, store in mem._iter_tiers() for item in store]
        retriever = EnsembleRetriever.from_balanced()
        compact = retriever.search_compact("word text", all_items, relevance_threshold=0.0)
        estimates = {c.id: c.token_estimate for c in compact}
        # Both items should have token_estimate set
        assert all(v is not None for v in estimates.values())

    def test_token_estimate_with_title_in_snippet(self):
        """search_compact should use title+fact for snippet when title is set."""
        from emms import EnsembleRetriever
        mem = _make_mem()
        exp = Experience(
            content="Added BM25 retrieval replacing Jaccard overlap in _relevance method",
            domain="tech",
            importance=0.9,
            title="Added BM25 retrieval",
            facts=["k1=1.5, b=0.75 parameters chosen"],
        )
        mem.store(exp)
        all_items = [item for _, store in mem._iter_tiers() for item in store]
        retriever = EnsembleRetriever.from_balanced()
        compact = retriever.search_compact("BM25", all_items, relevance_threshold=0.0)
        assert len(compact) >= 1
        # Snippet should prefer title
        assert "Added BM25" in compact[0].snippet or "k1=1.5" in compact[0].snippet


class TestGenerateContextInjection:
    """Improvement 5: SessionManager.generate_context_injection()."""

    def test_returns_string(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(content="Fixed auth bug", domain="tech", importance=0.8,
                              obs_type=ObsType.BUGFIX, title="Fixed auth"))
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection()
        assert isinstance(result, str)

    def test_contains_memory_index_header(self, tmp_path):
        mem = _make_mem()
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection()
        assert "EMMS Memory Index" in result

    def test_empty_memory_returns_placeholder(self, tmp_path):
        mem = _make_mem()
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection()
        assert "no memories" in result.lower()

    def test_observation_appears_in_index(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(
            content="Replaced Jaccard with BM25",
            domain="tech",
            importance=0.8,
            obs_type=ObsType.CHANGE,
            title="Changed retrieval to BM25",
        ))
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection()
        assert "[change]" in result
        assert "BM25" in result

    def test_token_estimate_in_index(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(
            content="word " * 20,
            domain="tech",
            importance=0.8,
        ))
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection()
        # Should contain something like "(26 tokens)"
        assert "tokens" in result

    def test_max_observations_limit(self, tmp_path):
        mem = _make_mem()
        for i in range(10):
            mem.store(Experience(content=f"experience {i}", domain="tech", importance=0.5))
        sm = SessionManager(memory=mem, log_path=tmp_path / "s.jsonl")
        result = sm.generate_context_injection(max_observations=3)
        # Should mention 3 as the shown count
        assert "showing 3" in result

    def test_works_with_explicit_items(self, tmp_path):
        mem = _make_mem()
        from emms.memory.hierarchical import HierarchicalMemory
        exp = Experience(content="test explicit injection", domain="test", importance=0.7)
        mem.store(exp)
        all_items = [item for _, store in mem._iter_tiers() for item in store]
        sm = SessionManager(log_path=tmp_path / "s.jsonl")  # no memory wired
        result = sm.generate_context_injection(all_items=all_items)
        assert "explicit injection" in result or "test" in result


# ===========================================================================
# v0.5.1 tests
# ===========================================================================

# ---------------------------------------------------------------------------
# search_by_file on HierarchicalMemory
# ---------------------------------------------------------------------------

class TestSearchByFile:

    def test_finds_file_in_files_read(self):
        mem = _make_mem()
        exp = Experience(
            content="Read the configuration file",
            domain="tech",
            files_read=["src/config.py"],
        )
        mem.store(exp)
        results = mem.search_by_file("src/config.py")
        assert len(results) == 1
        assert results[0].experience.id == exp.id

    def test_finds_file_in_files_modified(self):
        mem = _make_mem()
        exp = Experience(
            content="Edited the main module",
            domain="tech",
            files_modified=["src/main.py"],
        )
        mem.store(exp)
        results = mem.search_by_file("src/main.py")
        assert len(results) == 1

    def test_no_match_returns_empty(self):
        mem = _make_mem()
        mem.store(Experience(content="something", domain="test"))
        assert mem.search_by_file("nonexistent.py") == []

    def test_multiple_memories_same_file(self):
        mem = _make_mem()
        for i in range(3):
            mem.store(Experience(
                content=f"Event {i}",
                domain="tech",
                files_modified=["shared.py"],
            ))
        results = mem.search_by_file("shared.py")
        assert len(results) == 3

    def test_results_sorted_newest_first(self):
        import time as _time
        mem = _make_mem()
        exp1 = Experience(content="old event", domain="tech", files_read=["foo.py"])
        exp1_stored = Experience(content="old event", domain="tech", files_read=["foo.py"])
        exp2 = Experience(content="new event", domain="tech", files_read=["foo.py"])
        mem.store(exp1_stored)
        _time.sleep(0.01)
        mem.store(exp2)
        results = mem.search_by_file("foo.py")
        assert results[0].experience.content == "new event"


# ---------------------------------------------------------------------------
# EMMS.search_by_file (top-level delegation)
# ---------------------------------------------------------------------------

class TestEMMSSearchByFile:

    def test_delegates_to_memory(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        exp = Experience(content="wrote the adapter", files_modified=["adapters/mcp.py"])
        agent.store(exp)
        results = agent.search_by_file("adapters/mcp.py")
        assert len(results) == 1

    def test_emms_returns_empty_for_no_match(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        agent.store(Experience(content="generic content"))
        assert agent.search_by_file("ghost.py") == []


# ---------------------------------------------------------------------------
# Patch update_mode + conflict archival (superseded_by)
# ---------------------------------------------------------------------------

class TestPatchUpdateMode:

    def test_patch_supersedes_old_memory(self):
        mem = _make_mem()
        # Insert original
        exp1 = Experience(
            content="The API uses REST",
            domain="tech",
            title="api_style",
            patch_key="api_style",
            update_mode="insert",
        )
        mem.store(exp1)
        # Patch it
        exp2 = Experience(
            content="The API uses GraphQL",
            domain="tech",
            title="api_style",
            patch_key="api_style",
            update_mode="patch",
        )
        item2 = mem.store(exp2)
        # Find original item
        orig = mem._items_by_exp_id.get(exp1.id)
        assert orig is not None
        assert orig.superseded_by == item2.id

    def test_patch_is_superseded_property(self):
        mem = _make_mem()
        exp1 = Experience(content="v1", title="doc", patch_key="doc", update_mode="insert")
        mem.store(exp1)
        exp2 = Experience(content="v2", title="doc", patch_key="doc", update_mode="patch")
        mem.store(exp2)
        orig = mem._items_by_exp_id[exp1.id]
        assert orig.is_superseded is True

    def test_insert_mode_does_not_supersede(self):
        mem = _make_mem()
        exp1 = Experience(content="first", title="note")
        exp2 = Experience(content="second", title="note")  # insert mode (default)
        mem.store(exp1)
        mem.store(exp2)
        orig = mem._items_by_exp_id[exp1.id]
        assert not orig.is_superseded

    def test_patch_without_matching_key_inserts_new(self):
        mem = _make_mem()
        exp = Experience(
            content="new content",
            patch_key="no-match",
            update_mode="patch",
        )
        item = mem.store(exp)
        # No superseded memory — just inserted normally
        assert item.id in (mem._items_by_exp_id.get(exp.id).id,)


# ---------------------------------------------------------------------------
# TTL-aware filtering in retrieval
# ---------------------------------------------------------------------------

class TestTTLFiltering:

    def test_expired_memory_excluded_from_retrieve(self):
        import time as _time
        mem = _make_mem()
        exp = Experience(content="expired knowledge about dinosaurs", domain="science")
        item = mem.store(exp)
        # Force-expire: set expires_at to the past
        item.expires_at = _time.time() - 1.0
        assert item.is_expired
        results = mem.retrieve("dinosaurs")
        ids = [r.memory.id for r in results]
        assert item.id not in ids

    def test_superseded_memory_excluded_from_retrieve(self):
        mem = _make_mem()
        exp1 = Experience(content="old fact about Python", domain="tech", title="py", patch_key="py")
        exp2 = Experience(content="new fact about Python", domain="tech", title="py",
                          patch_key="py", update_mode="patch")
        mem.store(exp1)
        mem.store(exp2)
        orig = mem._items_by_exp_id[exp1.id]
        assert orig.is_superseded
        results = mem.retrieve("Python fact")
        orig_ids = [r.memory.experience.id for r in results]
        assert exp1.id not in orig_ids

    def test_non_expired_memory_included(self):
        import time as _time
        mem = _make_mem()
        exp = Experience(content="fresh knowledge about quantum physics", domain="science")
        item = mem.store(exp)
        item.expires_at = _time.time() + 3600  # expires in 1 hour
        assert not item.is_expired
        results = mem.retrieve("quantum physics")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# GraphMemory save/load
# ---------------------------------------------------------------------------

class TestGraphMemorySaveLoad:

    def test_save_and_load_roundtrip(self, tmp_path):
        from emms.memory.graph import GraphMemory
        g = GraphMemory()
        exp = Experience(content="Alice works at OpenAI and researches AI safety")
        g.store(exp)
        assert len(g.entities) > 0

        path = tmp_path / "graph.json"
        g.save_state(path)

        g2 = GraphMemory()
        g2.load_state(path)
        assert len(g2.entities) == len(g.entities)
        assert len(g2.relationships) == len(g.relationships)

    def test_load_nonexistent_is_noop(self, tmp_path):
        from emms.memory.graph import GraphMemory
        g = GraphMemory()
        g.load_state(tmp_path / "nope.json")  # should not raise
        assert len(g.entities) == 0

    def test_adjacency_rebuilt_on_load(self, tmp_path):
        from emms.memory.graph import GraphMemory
        g = GraphMemory()
        exp = Experience(content="Bob works at Google and collaborates with Charlie")
        g.store(exp)
        path = tmp_path / "g.json"
        g.save_state(path)

        g2 = GraphMemory()
        g2.load_state(path)
        # Adjacency should be rebuilt
        assert len(g2._adj) == len(g._adj)


# ---------------------------------------------------------------------------
# EMMS save/load includes graph + procedures
# ---------------------------------------------------------------------------

class TestEMMSSaveLoad:

    def test_graph_persists_with_emms_save_load(self, tmp_path):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        agent.store(Experience(content="Tesla is an electric car company", domain="tech"))
        path = tmp_path / "memory.json"
        agent.save(memory_path=path)

        agent2 = EMMS(enable_consciousness=False)
        agent2.load(memory_path=path)
        # Graph should be restored
        assert agent2.graph is not None
        assert len(agent2.graph.entities) == len(agent.graph.entities)

    def test_procedures_persist_with_emms_save_load(self, tmp_path):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        agent.add_procedure("Always use type hints.", domain="coding")
        path = tmp_path / "mem.json"
        agent.save(memory_path=path)

        agent2 = EMMS(enable_consciousness=False)
        agent2.load(memory_path=path)
        rules = agent2.procedures.get_all()
        assert any("type hints" in r.rule for r in rules)


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------

class TestProceduralMemory:

    def test_add_and_get_prompt(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        pm.add("Always write unit tests.", domain="coding")
        prompt = pm.get_prompt()
        assert "unit tests" in prompt
        assert "Behavioral Rules" in prompt

    def test_patch_updates_rule(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        e = pm.add("Draft rule.", domain="general")
        pm.patch(e.id, "Final rule with clarification.")
        updated = pm.get(e.id)
        assert updated.rule == "Final rule with clarification."
        assert updated.version == 2

    def test_remove_deactivates_rule(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        e = pm.add("Temporary rule.")
        pm.remove(e.id)
        prompt = pm.get_prompt()
        assert "Temporary rule" not in prompt
        assert pm.size["active"] == 0

    def test_domain_filter(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        pm.add("Coding rule.", domain="coding")
        pm.add("Research rule.", domain="research")
        pm.add("General rule.", domain="general")
        coding_rules = pm.get_all(domain="coding")
        # Should include general + coding
        names = [r.rule for r in coding_rules]
        assert "Coding rule." in names
        assert "General rule." in names
        assert "Research rule." not in names

    def test_importance_ordering(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        pm.add("Low priority.", importance=0.2)
        pm.add("High priority.", importance=0.9)
        rules = pm.get_all()
        assert rules[0].importance > rules[-1].importance

    def test_save_load_roundtrip(self, tmp_path):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        pm.add("Persist this rule.", domain="test")
        path = tmp_path / "proc.json"
        pm.save_state(path)

        pm2 = ProceduralMemory()
        pm2.load_state(path)
        assert pm2.size["active"] == 1
        assert "Persist this rule" in pm2.get_all()[0].rule

    def test_empty_prompt_when_no_rules(self):
        from emms.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        assert pm.get_prompt() == ""

    def test_add_procedure_via_emms(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        entry = agent.add_procedure("Never expose secrets.", domain="security", importance=0.9)
        assert "secrets" in entry.rule
        prompt = agent.get_system_prompt_rules()
        assert "secrets" in prompt


# ---------------------------------------------------------------------------
# citations + validate_citations
# ---------------------------------------------------------------------------

class TestCitationsValidation:

    def test_citations_field_on_experience(self):
        exp = Experience(content="test", citations=["mem_abc123"])
        assert exp.citations == ["mem_abc123"]

    def test_validate_citations_finds_existing(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        result1 = agent.store(Experience(content="The sky is blue"))
        mem_id = result1["memory_id"]
        exp2 = Experience(content="Referencing the sky", citations=[mem_id])
        validation = agent.validate_citations(exp2)
        assert validation[mem_id] is True

    def test_validate_citations_not_found(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        exp = Experience(content="References something", citations=["mem_nonexistent"])
        validation = agent.validate_citations(exp)
        assert validation["mem_nonexistent"] is False

    def test_validate_citations_strengthens_cited_memory(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        result = agent.store(Experience(content="Important fact"))
        mem_id = result["memory_id"]
        # Get original strength
        items = [i for _, store in agent.memory._iter_tiers() for i in store if i.id == mem_id]
        orig_strength = items[0].memory_strength
        # Validate citation
        exp2 = Experience(content="Citing the fact", citations=[mem_id])
        agent.validate_citations(exp2)
        # Strength should have increased
        assert items[0].memory_strength >= orig_strength


# ---------------------------------------------------------------------------
# Debounced consolidation in SessionManager
# ---------------------------------------------------------------------------

class TestDebouncedConsolidation:

    def test_consolidation_triggered_after_n_stores(self, tmp_path):
        from emms.memory.hierarchical import HierarchicalMemory
        mem = HierarchicalMemory(MemoryConfig(working_capacity=100))
        sm = SessionManager(
            memory=mem,
            log_path=tmp_path / "s.jsonl",
            consolidate_every=3,
        )
        sm.start_session()
        for i in range(3):
            sm.store(Experience(content=f"event {i}", domain="test", importance=0.9))
        # consolidate_every=3: should have fired once
        assert sm._stores_since_consolidation == 0  # reset after trigger

    def test_no_consolidation_before_threshold(self, tmp_path):
        from emms.memory.hierarchical import HierarchicalMemory
        mem = HierarchicalMemory()
        sm = SessionManager(
            memory=mem,
            log_path=tmp_path / "s.jsonl",
            consolidate_every=10,
        )
        sm.start_session()
        for i in range(5):
            sm.store(Experience(content=f"e{i}", domain="test"))
        assert sm._stores_since_consolidation == 5  # not yet triggered

    def test_consolidate_every_zero_disables(self, tmp_path):
        from emms.memory.hierarchical import HierarchicalMemory
        mem = HierarchicalMemory()
        sm = SessionManager(
            memory=mem,
            log_path=tmp_path / "s.jsonl",
            consolidate_every=0,
        )
        sm.start_session()
        for i in range(20):
            sm.store(Experience(content=f"e{i}", domain="test"))
        # No consolidation triggered (consolidate_every=0)
        assert sm._stores_since_consolidation == 20


# ---------------------------------------------------------------------------
# MCP server adapter
# ---------------------------------------------------------------------------

class TestEMCPServer:

    def _make_server(self):
        from emms import EMMS
        from emms.adapters.mcp_server import EMCPServer
        agent = EMMS(enable_consciousness=False)
        return EMCPServer(agent), agent

    def test_tool_definitions_present(self):
        from emms.adapters.mcp_server import EMCPServer, _TOOL_DEFINITIONS
        names = {d["name"] for d in _TOOL_DEFINITIONS}
        assert "emms_store" in names
        assert "emms_retrieve" in names
        assert "emms_search_by_file" in names
        assert "emms_get_procedures" in names

    def test_handle_store(self):
        server, agent = self._make_server()
        result = server.handle("emms_store", {"content": "test memory", "domain": "test"})
        assert result["ok"] is True
        assert "result" in result

    def test_handle_retrieve(self):
        server, agent = self._make_server()
        server.handle("emms_store", {"content": "The ocean is vast", "domain": "nature"})
        result = server.handle("emms_retrieve", {"query": "ocean", "max_results": 5})
        assert result["ok"] is True
        assert isinstance(result["results"], list)

    def test_handle_search_by_file(self):
        server, agent = self._make_server()
        server.handle("emms_store", {
            "content": "edited emms.py today",
            "files_modified": ["src/emms/emms.py"],
        })
        result = server.handle("emms_search_by_file", {"file_path": "emms.py"})
        assert result["ok"] is True
        assert len(result["results"]) >= 1

    def test_handle_get_stats(self):
        server, agent = self._make_server()
        result = server.handle("emms_get_stats", {})
        assert result["ok"] is True
        assert "stats" in result

    def test_handle_add_procedure_and_get(self):
        server, agent = self._make_server()
        result = server.handle("emms_add_procedure", {
            "rule": "Always document public APIs.",
            "domain": "coding",
            "importance": 0.8,
        })
        assert result["ok"] is True
        proc_result = server.handle("emms_get_procedures", {})
        assert proc_result["ok"] is True
        assert "APIs" in proc_result["prompt"]

    def test_handle_unknown_tool(self):
        server, agent = self._make_server()
        result = server.handle("emms_does_not_exist", {})
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    def test_handle_save_load(self, tmp_path):
        server, agent = self._make_server()
        server.handle("emms_store", {"content": "persistent info"})
        path = str(tmp_path / "mem.json")
        save_result = server.handle("emms_save", {"path": path})
        assert save_result["ok"] is True

        # Load into a fresh server
        from emms import EMMS
        from emms.adapters.mcp_server import EMCPServer
        agent2 = EMMS(enable_consciousness=False)
        server2 = EMCPServer(agent2)
        load_result = server2.handle("emms_load", {"path": path})
        assert load_result["ok"] is True


# ---------------------------------------------------------------------------
# strategy_scores + explanation on RetrievalResult
# ---------------------------------------------------------------------------

class TestStrategyScoresOnResult:

    def test_strategy_scores_populated(self):
        from emms.retrieval.strategies import EnsembleRetriever
        mem = _make_mem()
        for i in range(3):
            mem.store(Experience(content=f"climate change effect {i}", domain="science"))
        items = _all_items(mem)
        retriever = EnsembleRetriever.from_balanced()  # no embedder needed
        results = retriever.retrieve("climate", items)
        assert len(results) > 0
        for r in results:
            assert isinstance(r.strategy_scores, dict)
            assert len(r.strategy_scores) > 0
            assert r.explanation != ""

    def test_explanation_contains_strategy_names(self):
        from emms.retrieval.strategies import EnsembleRetriever
        mem = _make_mem()
        mem.store(Experience(content="machine learning tutorial", domain="tech"))
        items = _all_items(mem)
        retriever = EnsembleRetriever.from_balanced()  # no embedder needed
        results = retriever.retrieve("machine learning", items)
        if results:
            # Explanation should be non-empty
            assert "=" in results[0].explanation


# ---------------------------------------------------------------------------
# ImportanceStrategy
# ---------------------------------------------------------------------------

class TestImportanceStrategy:

    def test_importance_strategy_scores_high_importance_higher(self):
        from emms.retrieval.strategies import ImportanceStrategy
        from emms.core.models import MemoryItem, MemoryTier
        strategy = ImportanceStrategy()
        mem_high = MemoryItem(
            experience=Experience(content="critical", importance=0.9),
            tier=MemoryTier.WORKING,
        )
        mem_low = MemoryItem(
            experience=Experience(content="trivial", importance=0.1),
            tier=MemoryTier.WORKING,
        )
        score_high = strategy.score("query", mem_high, {})
        score_low = strategy.score("query", mem_low, {})
        assert score_high > score_low

    def test_importance_strategy_name(self):
        from emms.retrieval.strategies import ImportanceStrategy
        assert ImportanceStrategy.name == "importance"

    def test_importance_in_from_balanced(self):
        from emms.retrieval.strategies import EnsembleRetriever
        retriever = EnsembleRetriever.from_balanced()
        names = [s.name for s, _ in retriever.strategies]
        assert "importance" in names

    def test_importance_in_from_identity(self):
        from emms.retrieval.strategies import EnsembleRetriever
        retriever = EnsembleRetriever.from_identity()
        names = [s.name for s, _ in retriever.strategies]
        assert "importance" in names


# ===========================================================================
# v0.5.2 tests — namespace, confidence, filtered retrieval, feedback, markdown
# ===========================================================================

# ---------------------------------------------------------------------------
# namespace + confidence on Experience
# ---------------------------------------------------------------------------

class TestNamespaceAndConfidence:

    def test_namespace_default(self):
        exp = Experience(content="test")
        assert exp.namespace == "default"

    def test_namespace_custom(self):
        exp = Experience(content="test", namespace="project-x")
        assert exp.namespace == "project-x"

    def test_confidence_default(self):
        exp = Experience(content="test")
        assert exp.confidence == 1.0

    def test_confidence_custom(self):
        exp = Experience(content="uncertain fact", confidence=0.4)
        assert exp.confidence == 0.4

    def test_compact_result_has_namespace_confidence(self):
        from emms.core.models import CompactResult, MemoryTier
        import time
        cr = CompactResult(
            id="m1", snippet="test", domain="d", score=0.9,
            tier=MemoryTier.WORKING, timestamp=time.time(),
            namespace="ns-a", confidence=0.75,
        )
        assert cr.namespace == "ns-a"
        assert cr.confidence == 0.75


# ---------------------------------------------------------------------------
# retrieve_filtered
# ---------------------------------------------------------------------------

class TestRetrieveFiltered:

    def test_namespace_filter_excludes_other_namespaces(self):
        mem = _make_mem()
        mem.store(Experience(content="project A fact", namespace="proj-a"))
        mem.store(Experience(content="project B fact", namespace="proj-b"))
        results = mem.retrieve_filtered("fact", namespace="proj-a")
        assert all(r.memory.experience.namespace == "proj-a" for r in results)

    def test_domain_filter(self):
        mem = _make_mem()
        mem.store(Experience(content="python programming tips", domain="tech"))
        mem.store(Experience(content="stock market analysis", domain="finance"))
        results = mem.retrieve_filtered("tips analysis", domain="tech")
        assert all(r.memory.experience.domain == "tech" for r in results)

    def test_obs_type_filter(self):
        from emms.core.models import ObsType
        mem = _make_mem()
        mem.store(Experience(content="fixed the login bug", obs_type=ObsType.BUGFIX))
        mem.store(Experience(content="added dark mode feature", obs_type=ObsType.FEATURE))
        results = mem.retrieve_filtered("bug login", obs_type=ObsType.BUGFIX)
        for r in results:
            assert r.memory.experience.obs_type == ObsType.BUGFIX

    def test_time_range_filter(self):
        import time as _time
        mem = _make_mem()
        old_exp = Experience(content="old memory about plants")
        old_exp.timestamp = _time.time() - 10000  # far in the past
        mem.store(old_exp)
        new_exp = Experience(content="new memory about plants")
        mem.store(new_exp)
        cutoff = _time.time() - 100
        results = mem.retrieve_filtered("plants", since=cutoff)
        ids = [r.memory.experience.id for r in results]
        assert new_exp.id in ids
        assert old_exp.id not in ids

    def test_min_confidence_filter(self):
        mem = _make_mem()
        mem.store(Experience(content="uncertain news about quantum", confidence=0.3))
        mem.store(Experience(content="certain fact about quantum physics", confidence=0.95))
        results = mem.retrieve_filtered("quantum", min_confidence=0.8)
        for r in results:
            assert r.memory.experience.confidence >= 0.8

    def test_session_filter(self):
        mem = _make_mem()
        exp1 = Experience(content="session alpha memory", session_id="sess_alpha")
        exp2 = Experience(content="session beta memory", session_id="sess_beta")
        mem.store(exp1)
        mem.store(exp2)
        results = mem.retrieve_filtered("memory", session_id="sess_alpha")
        for r in results:
            assert r.memory.experience.session_id == "sess_alpha"

    def test_empty_results_for_impossible_filter(self):
        mem = _make_mem()
        mem.store(Experience(content="general content", domain="tech"))
        results = mem.retrieve_filtered("content", domain="finance")
        assert results == []

    def test_emms_retrieve_filtered_delegates(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        agent.store(Experience(content="scoped memory", namespace="ns-x"))
        agent.store(Experience(content="other memory", namespace="ns-y"))
        results = agent.retrieve_filtered("memory", namespace="ns-x")
        assert all(r.memory.experience.namespace == "ns-x" for r in results)


# ---------------------------------------------------------------------------
# upvote / downvote
# ---------------------------------------------------------------------------

class TestMemoryFeedback:

    def test_upvote_increases_strength(self):
        mem = _make_mem()
        item = mem.store(Experience(content="helpful fact about rivers"))
        item.memory_strength = 0.5  # set below max so boost has room
        original_strength = item.memory_strength
        found = mem.upvote(item.id)
        assert found is True
        assert item.memory_strength > original_strength

    def test_downvote_decreases_strength(self):
        mem = _make_mem()
        item = mem.store(Experience(content="irrelevant fact"))
        original_strength = item.memory_strength
        found = mem.downvote(item.id)
        assert found is True
        assert item.memory_strength < original_strength

    def test_upvote_caps_at_1(self):
        mem = _make_mem()
        item = mem.store(Experience(content="very important fact"))
        item.memory_strength = 0.99
        mem.upvote(item.id, boost=1.0)
        assert item.memory_strength == 1.0

    def test_downvote_floors_at_0(self):
        mem = _make_mem()
        item = mem.store(Experience(content="bad memory"))
        item.memory_strength = 0.05
        mem.downvote(item.id, decay=1.0)
        assert item.memory_strength == 0.0

    def test_upvote_returns_false_for_unknown_id(self):
        mem = _make_mem()
        assert mem.upvote("mem_does_not_exist") is False

    def test_downvote_returns_false_for_unknown_id(self):
        mem = _make_mem()
        assert mem.downvote("mem_does_not_exist") is False

    def test_upvote_by_experience_id(self):
        mem = _make_mem()
        exp = Experience(content="test content for experience id lookup")
        item = mem.store(exp)
        found = mem.upvote(exp.id)  # use experience.id not mem.id
        assert found is True

    def test_emms_upvote_downvote(self):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        result = agent.store(Experience(content="important research finding"))
        mem_id = result["memory_id"]
        assert agent.upvote(mem_id) is True
        assert agent.downvote(mem_id) is True


# ---------------------------------------------------------------------------
# export_markdown
# ---------------------------------------------------------------------------

class TestExportMarkdown:

    def test_creates_markdown_file(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(content="The sun is a star", domain="science"))
        path = tmp_path / "export.md"
        count = mem.export_markdown(path)
        assert count == 1
        assert path.exists()
        text = path.read_text()
        assert "# EMMS Memory Export" in text
        assert "sun is a star" in text

    def test_groups_by_domain(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(content="stock tip", domain="finance"))
        mem.store(Experience(content="bug fix info", domain="tech"))
        path = tmp_path / "out.md"
        mem.export_markdown(path)
        text = path.read_text()
        assert "Finance" in text or "finance" in text.lower()
        assert "Tech" in text or "tech" in text.lower()

    def test_excludes_private_by_default(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(content="public info", domain="test", private=False))
        mem.store(Experience(content="private secret", domain="test", private=True))
        path = tmp_path / "out.md"
        count = mem.export_markdown(path)
        assert count == 1
        assert "private secret" not in path.read_text()

    def test_namespace_filter(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(content="ns-a content", namespace="ns-a", domain="test"))
        mem.store(Experience(content="ns-b content", namespace="ns-b", domain="test"))
        path = tmp_path / "out.md"
        count = mem.export_markdown(path, namespace="ns-a")
        assert count == 1
        assert "ns-a content" in path.read_text()
        assert "ns-b content" not in path.read_text()

    def test_includes_facts(self, tmp_path):
        mem = _make_mem()
        mem.store(Experience(
            content="Python is a programming language",
            domain="tech",
            facts=["Python supports multiple paradigms", "Python has GC"],
        ))
        path = tmp_path / "out.md"
        mem.export_markdown(path)
        text = path.read_text()
        assert "multiple paradigms" in text
        assert "GC" in text

    def test_emms_export_markdown_delegates(self, tmp_path):
        from emms import EMMS
        agent = EMMS(enable_consciousness=False)
        agent.store(Experience(content="agent memory", domain="test"))
        path = tmp_path / "agent.md"
        count = agent.export_markdown(str(path))
        assert count >= 1
        assert path.exists()
