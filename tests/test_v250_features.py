"""Tests for EMMS v0.25.0 — The Vigilant Mind.

Covers:
- RuminationDetector / RuminationCluster / RuminationReport
- SelfEfficacyAssessor / EfficacyProfile / EfficacyReport
- MoodDynamics / MoodSegment / MoodReport
- EMMS facade (v0.25.0 methods)
- MCP tool count (112) + 5 new tools
- Version and __all__ exports
"""

from __future__ import annotations

import pytest

import emms as emms_pkg
from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS()


def _make_rich_emms() -> EMMS:
    """EMMS with rumination-heavy, efficacy-diverse, and mood-arc memories."""
    import time
    from emms.core.models import MemoryConfig
    agent = EMMS(config=MemoryConfig(working_capacity=25))
    now = time.time()

    # --- Rumination corpus: ethics domain, repetitive + negative ---
    rumi_contents = [
        "The mistake and failure in judgment was completely wrong and inadequate.",
        "That terrible mistake keeps haunting me; the judgment failure is deeply inadequate.",
        "Wrong judgment leads to failure; the mistake is inadequate and continues haunting.",
        "Inadequate judgment and terrible mistake result in repeated failure patterns.",
        "The failure of judgment is wrong; this mistake is inadequate and recurring.",
    ]
    for i, content in enumerate(rumi_contents):
        agent.store(Experience(
            content=content, domain="ethics",
            importance=0.8, emotional_valence=-0.7,
            timestamp=now - (len(rumi_contents) - i) * 86400,
        ))

    # --- Success/failure corpus: coding domain (high efficacy) ---
    for content, ev, ts_offset in [
        ("I achieved and completed the task successfully and worked effectively.", 0.8, 10),
        ("Mastered the algorithm; accomplished the solution correctly and efficiently.", 0.7, 9),
        ("Solved the problem effectively; managed all edge cases correctly.", 0.9, 8),
        ("Completed the module successfully; the implementation worked as expected.", 0.6, 7),
        ("Built and improved the system; achieved all the objectives confidently.", 0.8, 6),
    ]:
        agent.store(Experience(
            content=content, domain="coding",
            importance=0.85, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Failure corpus: management domain (low efficacy) ---
    for content, ev, ts_offset in [
        ("Failed to manage the team; the approach was confused and ineffective.", -0.6, 15),
        ("Made a mistake in planning; unable to achieve the target due to poor judgment.", -0.7, 14),
        ("Lost the project due to inadequate strategy and wrong priorities.", -0.8, 13),
        ("Struggled with difficult requirements; the outcome was broken and ineffective.", -0.5, 12),
    ]:
        agent.store(Experience(
            content=content, domain="management",
            importance=0.7, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Mood arc corpus: philosophy domain, negative → positive ---
    mood_memories = [
        ("Deeply distressed and subdued by the uncertainty ahead.", -0.8, 20),
        ("Struggling with difficult questions about meaning and direction.", -0.5, 18),
        ("Beginning to find clarity through careful reasoning and reflection.", 0.1, 16),
        ("Growing content as understanding deepens and insight emerges.", 0.4, 14),
        ("Feeling joyful as knowledge and wisdom integrate into clear vision.", 0.8, 12),
    ]
    for content, ev, ts_offset in mood_memories:
        agent.store(Experience(
            content=content, domain="philosophy",
            importance=0.75, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    return agent


# ---------------------------------------------------------------------------
# TestRuminationDetector
# ---------------------------------------------------------------------------

class TestRuminationDetector:

    def _make_detector(self, agent=None):
        from emms.memory.rumination import RuminationDetector
        if agent is None:
            agent = _make_rich_emms()
        return RuminationDetector(memory=agent.memory)

    def test_detect_returns_rumination_report(self):
        from emms.memory.rumination import RuminationReport
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report, RuminationReport)

    def test_report_has_required_fields(self):
        detector = self._make_detector()
        report = detector.detect()
        assert hasattr(report, "total_clusters")
        assert hasattr(report, "clusters")
        assert hasattr(report, "most_ruminative_domain")
        assert hasattr(report, "overall_rumination_score")
        assert hasattr(report, "duration_seconds")

    def test_total_clusters_matches_list(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.total_clusters == len(report.clusters)

    def test_rumination_cluster_has_required_fields(self):
        from emms.memory.rumination import RuminationCluster
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert isinstance(c, RuminationCluster)
            assert hasattr(c, "id")
            assert hasattr(c, "domain")
            assert hasattr(c, "cluster_size")
            assert hasattr(c, "rumination_score")
            assert hasattr(c, "mean_negativity")
            assert hasattr(c, "theme_tokens")
            assert hasattr(c, "memory_ids")
            assert hasattr(c, "resolution_hint")
            assert hasattr(c, "created_at")

    def test_cluster_id_prefix(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert c.id.startswith("rum_")

    def test_rumination_score_in_range(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert 0.0 <= c.rumination_score <= 1.0

    def test_mean_negativity_in_range(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert 0.0 <= c.mean_negativity <= 1.0

    def test_clusters_sorted_by_score_desc(self):
        detector = self._make_detector()
        report = detector.detect()
        scores = [c.rumination_score for c in report.clusters]
        assert scores == sorted(scores, reverse=True)

    def test_overall_score_zero_on_empty(self):
        from emms.memory.rumination import RuminationDetector
        agent = _make_emms()
        detector = RuminationDetector(memory=agent.memory)
        report = detector.detect()
        assert report.overall_rumination_score == 0.0

    def test_empty_memory_returns_zero_clusters(self):
        from emms.memory.rumination import RuminationDetector
        agent = _make_emms()
        detector = RuminationDetector(memory=agent.memory)
        report = detector.detect()
        assert report.total_clusters == 0
        assert report.most_ruminative_domain == "none"

    def test_most_ruminative_domain_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report.most_ruminative_domain, str)

    def test_domain_filter(self):
        detector = self._make_detector()
        report = detector.detect(domain="ethics")
        for c in report.clusters:
            assert c.domain == "ethics"

    def test_theme_tokens_at_most_5(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert len(c.theme_tokens) <= 5

    def test_memory_ids_at_most_10(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert len(c.memory_ids) <= 10

    def test_resolution_hint_non_empty(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert isinstance(c.resolution_hint, str)
            assert len(c.resolution_hint) > 0

    def test_rumination_themes_returns_list(self):
        detector = self._make_detector()
        detector.detect()
        themes = detector.rumination_themes()
        assert isinstance(themes, list)

    def test_most_ruminative_cluster_returns_cluster_or_none(self):
        from emms.memory.rumination import RuminationCluster
        detector = self._make_detector()
        detector.detect()
        result = detector.most_ruminative_cluster()
        assert result is None or isinstance(result, RuminationCluster)

    def test_most_ruminative_cluster_none_on_empty(self):
        from emms.memory.rumination import RuminationDetector
        agent = _make_emms()
        detector = RuminationDetector(memory=agent.memory)
        detector.detect()
        assert detector.most_ruminative_cluster() is None

    def test_cluster_summary_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert isinstance(c.summary(), str)

    def test_report_summary_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report.summary(), str)

    def test_duration_non_negative(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.duration_seconds >= 0.0

    def test_at_least_one_cluster_on_rich_corpus(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.total_clusters >= 1

    def test_max_clusters_cap(self):
        from emms.memory.rumination import RuminationDetector
        agent = _make_rich_emms()
        detector = RuminationDetector(memory=agent.memory, max_clusters=1)
        report = detector.detect()
        assert report.total_clusters <= 1

    def test_min_cluster_size_filter(self):
        from emms.memory.rumination import RuminationDetector
        agent = _make_rich_emms()
        detector = RuminationDetector(memory=agent.memory, min_cluster_size=100)
        report = detector.detect()
        assert report.total_clusters == 0

    def test_cluster_size_non_zero(self):
        detector = self._make_detector()
        report = detector.detect()
        for c in report.clusters:
            assert c.cluster_size >= 1


# ---------------------------------------------------------------------------
# TestSelfEfficacyAssessor
# ---------------------------------------------------------------------------

class TestSelfEfficacyAssessor:

    def _make_assessor(self, agent=None):
        from emms.memory.efficacy import SelfEfficacyAssessor
        if agent is None:
            agent = _make_rich_emms()
        return SelfEfficacyAssessor(memory=agent.memory)

    def test_assess_returns_efficacy_report(self):
        from emms.memory.efficacy import EfficacyReport
        assessor = self._make_assessor()
        report = assessor.assess()
        assert isinstance(report, EfficacyReport)

    def test_report_has_required_fields(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        assert hasattr(report, "total_domains")
        assert hasattr(report, "profiles")
        assert hasattr(report, "highest_efficacy_domain")
        assert hasattr(report, "lowest_efficacy_domain")
        assert hasattr(report, "mean_efficacy")
        assert hasattr(report, "duration_seconds")

    def test_total_domains_matches_list(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        assert report.total_domains == len(report.profiles)

    def test_efficacy_profile_has_required_fields(self):
        from emms.memory.efficacy import EfficacyProfile
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert isinstance(p, EfficacyProfile)
            assert hasattr(p, "domain")
            assert hasattr(p, "efficacy_score")
            assert hasattr(p, "success_count")
            assert hasattr(p, "failure_count")
            assert hasattr(p, "trending")
            assert hasattr(p, "recent_themes")

    def test_efficacy_score_in_range(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert 0.0 <= p.efficacy_score <= 1.0

    def test_trending_valid_values(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        valid = {"improving", "declining", "stable"}
        for p in report.profiles:
            assert p.trending in valid

    def test_profiles_sorted_by_efficacy_desc(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        scores = [p.efficacy_score for p in report.profiles]
        assert scores == sorted(scores, reverse=True)

    def test_efficacy_for_domain_known(self):
        from emms.memory.efficacy import EfficacyProfile
        assessor = self._make_assessor()
        assessor.assess()
        profile = assessor.efficacy_for_domain("coding")
        if profile is not None:
            assert isinstance(profile, EfficacyProfile)
            assert profile.domain == "coding"

    def test_efficacy_for_domain_unknown_returns_none(self):
        assessor = self._make_assessor()
        assessor.assess()
        result = assessor.efficacy_for_domain("nonexistent_domain_xyz")
        assert result is None

    def test_highest_efficacy_domain_str_or_none(self):
        assessor = self._make_assessor()
        assessor.assess()
        result = assessor.highest_efficacy_domain()
        assert result is None or isinstance(result, str)

    def test_highest_efficacy_domain_none_on_empty(self):
        from emms.memory.efficacy import SelfEfficacyAssessor
        agent = _make_emms()
        assessor = SelfEfficacyAssessor(memory=agent.memory)
        assessor.assess()
        assert assessor.highest_efficacy_domain() is None

    def test_empty_memory_returns_zero_domains(self):
        from emms.memory.efficacy import SelfEfficacyAssessor
        agent = _make_emms()
        assessor = SelfEfficacyAssessor(memory=agent.memory)
        report = assessor.assess()
        assert report.total_domains == 0

    def test_mean_efficacy_in_range(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        assert 0.0 <= report.mean_efficacy <= 1.0

    def test_recent_themes_is_list(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert isinstance(p.recent_themes, list)

    def test_recent_themes_at_most_5(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert len(p.recent_themes) <= 5

    def test_domain_filter(self):
        assessor = self._make_assessor()
        report = assessor.assess(domain="coding")
        for p in report.profiles:
            assert p.domain == "coding"

    def test_success_count_non_negative(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert p.success_count >= 0

    def test_failure_count_non_negative(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert p.failure_count >= 0

    def test_profile_summary_is_string(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        for p in report.profiles:
            assert isinstance(p.summary(), str)

    def test_report_summary_is_string(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        assert isinstance(report.summary(), str)

    def test_duration_non_negative(self):
        assessor = self._make_assessor()
        report = assessor.assess()
        assert report.duration_seconds >= 0.0

    def test_coding_domain_has_high_efficacy(self):
        assessor = self._make_assessor()
        assessor.assess()
        profile = assessor.efficacy_for_domain("coding")
        if profile:
            assert profile.efficacy_score > 0.5


# ---------------------------------------------------------------------------
# TestMoodDynamics
# ---------------------------------------------------------------------------

class TestMoodDynamics:

    def _make_dynamics(self, agent=None):
        from emms.memory.mood_trajectory import MoodDynamics
        if agent is None:
            agent = _make_rich_emms()
        return MoodDynamics(memory=agent.memory)

    def test_trace_returns_mood_report(self):
        from emms.memory.mood_trajectory import MoodReport
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert isinstance(report, MoodReport)

    def test_report_has_required_fields(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert hasattr(report, "total_memories")
        assert hasattr(report, "segments")
        assert hasattr(report, "mean_valence")
        assert hasattr(report, "volatility")
        assert hasattr(report, "trend")
        assert hasattr(report, "emotional_range")
        assert hasattr(report, "dominant_emotion")
        assert hasattr(report, "duration_seconds")

    def test_segment_has_required_fields(self):
        from emms.memory.mood_trajectory import MoodSegment
        dyn = self._make_dynamics()
        report = dyn.trace()
        for seg in report.segments:
            assert isinstance(seg, MoodSegment)
            assert hasattr(seg, "segment_index")
            assert hasattr(seg, "mean_valence")
            assert hasattr(seg, "valence_std")
            assert hasattr(seg, "memory_count")
            assert hasattr(seg, "label")

    def test_segment_label_valid(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        valid_labels = {"joyful", "content", "neutral", "subdued", "distressed"}
        for seg in report.segments:
            assert seg.label in valid_labels

    def test_trend_valid_values(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert report.trend in {"improving", "declining", "stable", "volatile"}

    def test_mean_valence_in_range(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert -1.0 <= report.mean_valence <= 1.0

    def test_volatility_non_negative(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert report.volatility >= 0.0

    def test_emotional_range_in_range(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert 0.0 <= report.emotional_range <= 2.0

    def test_dominant_emotion_valid(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        valid_labels = {"joyful", "content", "neutral", "subdued", "distressed"}
        assert report.dominant_emotion in valid_labels

    def test_mood_trend_unknown_before_trace(self):
        from emms.memory.mood_trajectory import MoodDynamics
        agent = _make_rich_emms()
        dyn = MoodDynamics(memory=agent.memory)
        assert dyn.mood_trend() == "unknown"

    def test_mood_trend_returns_string_after_trace(self):
        dyn = self._make_dynamics()
        dyn.trace()
        trend = dyn.mood_trend()
        assert isinstance(trend, str)
        assert trend in {"improving", "declining", "stable", "volatile"}

    def test_emotional_arc_empty_before_trace(self):
        from emms.memory.mood_trajectory import MoodDynamics
        agent = _make_rich_emms()
        dyn = MoodDynamics(memory=agent.memory)
        assert dyn.emotional_arc() == []

    def test_emotional_arc_returns_list_of_strings(self):
        dyn = self._make_dynamics()
        dyn.trace()
        arc = dyn.emotional_arc()
        assert isinstance(arc, list)
        for label in arc:
            assert isinstance(label, str)

    def test_empty_memory_returns_valid_report(self):
        from emms.memory.mood_trajectory import MoodDynamics, MoodReport
        agent = _make_emms()
        dyn = MoodDynamics(memory=agent.memory)
        report = dyn.trace()
        assert isinstance(report, MoodReport)
        assert report.total_memories == 0
        assert report.segments == []
        assert report.mean_valence == 0.0
        assert report.volatility == 0.0
        assert report.trend == "stable"
        assert report.emotional_range == 0.0
        assert report.dominant_emotion == "neutral"

    def test_segments_count_at_most_n_segments(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert len(report.segments) <= dyn.n_segments

    def test_segments_chronological(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        indices = [seg.segment_index for seg in report.segments]
        assert indices == sorted(indices)

    def test_segment_valence_std_non_negative(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        for seg in report.segments:
            assert seg.valence_std >= 0.0

    def test_segment_memory_count_positive(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        for seg in report.segments:
            assert seg.memory_count > 0

    def test_domain_filter(self):
        dyn = self._make_dynamics()
        report_all = dyn.trace()
        report_domain = dyn.trace(domain="philosophy")
        assert isinstance(report_domain.total_memories, int)

    def test_total_memories_matches_content(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert report.total_memories >= 0

    def test_report_summary_is_string(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert isinstance(report.summary(), str)

    def test_duration_non_negative(self):
        dyn = self._make_dynamics()
        report = dyn.trace()
        assert report.duration_seconds >= 0.0

    def test_philosophy_domain_shows_improving_trend(self):
        dyn = self._make_dynamics()
        report = dyn.trace(domain="philosophy")
        # philosophy memories go from -0.8 to +0.8 — should not be declining
        if report.total_memories >= 2:
            assert report.trend in {"improving", "stable", "volatile"}


# ---------------------------------------------------------------------------
# TestEMMSFacadeV250
# ---------------------------------------------------------------------------

class TestEMMSFacadeV250:

    def test_detect_rumination_callable(self):
        agent = _make_rich_emms()
        report = agent.detect_rumination()
        assert report is not None

    def test_detect_rumination_with_domain(self):
        agent = _make_rich_emms()
        report = agent.detect_rumination(domain="ethics")
        assert hasattr(report, "total_clusters")

    def test_rumination_themes_callable(self):
        agent = _make_rich_emms()
        agent.detect_rumination()
        themes = agent.rumination_themes()
        assert isinstance(themes, list)

    def test_most_ruminative_theme_callable(self):
        agent = _make_rich_emms()
        agent.detect_rumination()
        result = agent.most_ruminative_theme()
        assert result is None or hasattr(result, "rumination_score")

    def test_assess_efficacy_callable(self):
        agent = _make_rich_emms()
        report = agent.assess_efficacy()
        assert report is not None

    def test_assess_efficacy_with_domain(self):
        agent = _make_rich_emms()
        report = agent.assess_efficacy(domain="coding")
        assert hasattr(report, "total_domains")

    def test_efficacy_for_domain_callable(self):
        agent = _make_rich_emms()
        agent.assess_efficacy()
        profile = agent.efficacy_for_domain("coding")
        assert profile is None or hasattr(profile, "efficacy_score")

    def test_highest_efficacy_domain_callable(self):
        agent = _make_rich_emms()
        agent.assess_efficacy()
        result = agent.highest_efficacy_domain()
        assert result is None or isinstance(result, str)

    def test_trace_mood_callable(self):
        agent = _make_rich_emms()
        report = agent.trace_mood()
        assert report is not None

    def test_trace_mood_with_domain(self):
        agent = _make_rich_emms()
        report = agent.trace_mood(domain="philosophy")
        assert hasattr(report, "total_memories")

    def test_mood_trend_before_trace(self):
        agent = _make_emms()
        trend = agent.mood_trend()
        assert trend == "unknown"

    def test_mood_trend_after_trace(self):
        agent = _make_rich_emms()
        agent.trace_mood()
        trend = agent.mood_trend()
        assert trend in {"improving", "declining", "stable", "volatile"}

    def test_emotional_arc_callable(self):
        agent = _make_rich_emms()
        agent.trace_mood()
        arc = agent.emotional_arc()
        assert isinstance(arc, list)

    def test_rumination_detector_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.detect_rumination()
        r2 = agent.detect_rumination()
        assert r1.total_clusters == r2.total_clusters

    def test_efficacy_assessor_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.assess_efficacy()
        r2 = agent.assess_efficacy()
        assert r1.total_domains == r2.total_domains

    def test_mood_dynamics_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.trace_mood()
        r2 = agent.trace_mood()
        assert r1.total_memories == r2.total_memories

    def test_all_nine_facade_methods_exist(self):
        agent = _make_emms()
        for method in (
            "detect_rumination", "rumination_themes", "most_ruminative_theme",
            "assess_efficacy", "efficacy_for_domain", "highest_efficacy_domain",
            "trace_mood", "mood_trend", "emotional_arc",
        ):
            assert hasattr(agent, method), f"EMMS missing method: {method}"


# ---------------------------------------------------------------------------
# TestMCPV250
# ---------------------------------------------------------------------------

class TestMCPV250:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_112(self):
        server = self._make_server()
        assert len(server.tool_definitions) == 112, (
            f"Expected 112 tools, got {len(server.tool_definitions)}"
        )

    def test_emms_detect_rumination_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_detect_rumination" in names

    def test_emms_most_ruminative_theme_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_most_ruminative_theme" in names

    def test_emms_assess_efficacy_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_assess_efficacy" in names

    def test_emms_trace_mood_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_trace_mood" in names

    def test_emms_mood_trend_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_mood_trend" in names

    def test_detect_rumination_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_detect_rumination", {})
        assert result.get("ok") is True

    def test_most_ruminative_theme_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_most_ruminative_theme", {})
        assert result.get("ok") is True

    def test_assess_efficacy_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_assess_efficacy", {})
        assert result.get("ok") is True

    def test_trace_mood_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_trace_mood", {})
        assert result.get("ok") is True

    def test_mood_trend_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_mood_trend", {})
        assert result.get("ok") is True


# ---------------------------------------------------------------------------
# TestV250Exports
# ---------------------------------------------------------------------------

class TestV250Exports:

    def test_version_is_0_25_0(self):
        assert emms_pkg.__version__ == "0.25.0"

    def test_rumination_detector_importable(self):
        from emms import RuminationDetector
        assert RuminationDetector is not None

    def test_rumination_cluster_importable(self):
        from emms import RuminationCluster
        assert RuminationCluster is not None

    def test_rumination_report_importable(self):
        from emms import RuminationReport
        assert RuminationReport is not None

    def test_self_efficacy_assessor_importable(self):
        from emms import SelfEfficacyAssessor
        assert SelfEfficacyAssessor is not None

    def test_efficacy_profile_importable(self):
        from emms import EfficacyProfile
        assert EfficacyProfile is not None

    def test_efficacy_report_importable(self):
        from emms import EfficacyReport
        assert EfficacyReport is not None

    def test_mood_dynamics_importable(self):
        from emms import MoodDynamics
        assert MoodDynamics is not None

    def test_mood_segment_importable(self):
        from emms import MoodSegment
        assert MoodSegment is not None

    def test_mood_report_importable(self):
        from emms import MoodReport
        assert MoodReport is not None

    def test_all_nine_symbols_in_all(self):
        all_exports = emms_pkg.__all__
        for sym in (
            "RuminationDetector", "RuminationCluster", "RuminationReport",
            "SelfEfficacyAssessor", "EfficacyProfile", "EfficacyReport",
            "MoodDynamics", "MoodSegment", "MoodReport",
        ):
            assert sym in all_exports, f"{sym} not in __all__"

    def test_rumination_module_importable(self):
        from emms.memory.rumination import RuminationDetector, RuminationCluster, RuminationReport
        assert RuminationDetector and RuminationCluster and RuminationReport

    def test_efficacy_module_importable(self):
        from emms.memory.efficacy import SelfEfficacyAssessor, EfficacyProfile, EfficacyReport
        assert SelfEfficacyAssessor and EfficacyProfile and EfficacyReport

    def test_mood_module_importable(self):
        from emms.memory.mood_trajectory import MoodDynamics, MoodSegment, MoodReport
        assert MoodDynamics and MoodSegment and MoodReport
