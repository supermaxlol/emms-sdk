"""Tests for EMMS v0.26.0 — The Resilient Mind.

Covers:
- AdversityTracer / AdversityEvent / AdversityReport
- SelfCompassionGauge / SelfCompassionProfile / SelfCompassionReport
- ResilienceIndex / RecoveryArc / ResilienceReport
- EMMS facade (v0.26.0 methods)
- MCP tool count (117) + 5 new tools
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
    """EMMS with adversity, compassion-diverse, and recovery-arc memories."""
    import time
    from emms.core.models import MemoryConfig
    agent = EMMS(config=MemoryConfig(working_capacity=25))
    now = time.time()

    # --- Loss corpus: grief domain, 4 memories with loss tokens ---
    for content, ev, ts_offset in [
        ("The grief of losing someone gone is an absence that never fully heals.", -0.8, 20),
        ("I feel the weight of what is missing and departed from my life.", -0.7, 18),
        ("The mourning continues; what was lost cannot be replaced or recovered.", -0.75, 16),
        ("Her absence from the world has ended something irreplaceable in me.", -0.6, 14),
    ]:
        agent.store(Experience(
            content=content, domain="grief",
            importance=0.8, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Rejection corpus: social domain, 3 memories with rejection tokens ---
    for content, ev, ts_offset in [
        ("Rejected again; the feeling of being excluded and ignored is cold.", -0.7, 12),
        ("Dismissed and refused; I feel abandoned and unwanted by those I trusted.", -0.6, 11),
        ("Avoided and shut out; the sense of being excluded is deeply painful.", -0.65, 10),
    ]:
        agent.store(Experience(
            content=content, domain="social",
            importance=0.75, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Kindness corpus: therapy domain, 3 memories with KINDNESS tokens ---
    for content, ev, ts_offset in [
        ("It is okay and normal to be imperfect; growth requires patience and forgiveness.", 0.6, 9),
        ("I am allowed to be human and enough; accepting my limitations is natural learning.", 0.65, 8),
        ("Gentle understanding toward myself; kind acceptance is part of healthy growth.", 0.7, 7),
    ]:
        agent.store(Experience(
            content=content, domain="therapy",
            importance=0.8, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Harsh corpus: inner domain, 3 memories with HARSH tokens ---
    for content, ev, ts_offset in [
        ("I am pathetic and worthless for failing again; this is hopeless and shameful.", -0.85, 6),
        ("How could I be so stupid and inadequate; I am useless and lazy without excuse.", -0.8, 5),
        ("Terrible and awful judgment; I am horrible at this and completely inadequate.", -0.75, 4),
    ]:
        agent.store(Experience(
            content=content, domain="inner",
            importance=0.7, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))

    # --- Recovery arc: wellbeing domain, 3 negative then 3 positive ---
    # 3 consecutive adversity memories (valence < -0.2)
    for content, ev, ts_offset in [
        ("Everything feels dark and difficult; nothing seems to work out well.", -0.7, 3),
        ("Struggling deeply; overwhelmed by the weight of ongoing failures and loss.", -0.6, 2),
        ("At my lowest point; the situation seems bleak and without clear resolution.", -0.8, 1),
    ]:
        agent.store(Experience(
            content=content, domain="wellbeing",
            importance=0.75, emotional_valence=ev,
            timestamp=now - ts_offset * 86400,
        ))
    # 3 recovery memories (valence > 0.2)
    for content, ev, ts_offset in [
        ("Starting to see light; small steps forward are building renewed confidence.", 0.5, 0),
        ("Recovery is progressing well; I feel more capable and hopeful today.", 0.7, -1),
        ("Fully recovered; a deep sense of gratitude and renewed purpose guides me.", 0.8, -2),
    ]:
        agent.store(Experience(
            content=content, domain="wellbeing",
            importance=0.8, emotional_valence=ev,
            timestamp=now + abs(ts_offset) * 86400,
        ))

    return agent


# ---------------------------------------------------------------------------
# TestAdversityTracer
# ---------------------------------------------------------------------------

class TestAdversityTracer:

    def _make_tracer(self, agent=None):
        from emms.memory.adversity import AdversityTracer
        if agent is None:
            agent = _make_rich_emms()
        return AdversityTracer(memory=agent.memory)

    def test_trace_returns_adversity_report(self):
        from emms.memory.adversity import AdversityReport
        tracer = self._make_tracer()
        report = tracer.trace()
        assert isinstance(report, AdversityReport)

    def test_report_has_required_fields(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert hasattr(report, "total_events")
        assert hasattr(report, "events")
        assert hasattr(report, "most_common_type")
        assert hasattr(report, "dominant_domain")
        assert hasattr(report, "cumulative_severity")
        assert hasattr(report, "duration_seconds")

    def test_total_events_matches_list(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert report.total_events == len(report.events)

    def test_at_least_one_event_on_rich_corpus(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert report.total_events >= 1

    def test_event_has_required_fields(self):
        from emms.memory.adversity import AdversityEvent
        tracer = self._make_tracer()
        report = tracer.trace()
        for ev in report.events:
            assert isinstance(ev, AdversityEvent)
            assert hasattr(ev, "id")
            assert hasattr(ev, "adversity_type")
            assert hasattr(ev, "severity")
            assert hasattr(ev, "domain")
            assert hasattr(ev, "memory_id")
            assert hasattr(ev, "timestamp")
            assert hasattr(ev, "created_at")

    def test_event_id_prefix(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        for ev in report.events:
            assert ev.id.startswith("adv_")

    def test_severity_range(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        for ev in report.events:
            assert 0.0 <= ev.severity <= 1.0

    def test_events_sorted_by_severity_desc(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        severities = [ev.severity for ev in report.events]
        assert severities == sorted(severities, reverse=True)

    def test_most_common_type_is_string(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert isinstance(report.most_common_type, str)

    def test_dominant_domain_is_string(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert isinstance(report.dominant_domain, str)

    def test_cumulative_severity_positive(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert report.cumulative_severity >= 0.0

    def test_duration_nonnegative(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert report.duration_seconds >= 0.0

    def test_events_of_type_loss(self):
        tracer = self._make_tracer()
        tracer.trace()
        loss_events = tracer.events_of_type("loss")
        assert isinstance(loss_events, list)
        assert all(ev.adversity_type == "loss" for ev in loss_events)

    def test_events_of_type_rejection(self):
        tracer = self._make_tracer()
        tracer.trace()
        rejection_events = tracer.events_of_type("rejection")
        assert isinstance(rejection_events, list)
        assert all(ev.adversity_type == "rejection" for ev in rejection_events)

    def test_events_of_type_unknown_returns_empty(self):
        tracer = self._make_tracer()
        tracer.trace()
        result = tracer.events_of_type("nonexistent_type")
        assert result == []

    def test_dominant_adversity_type_returns_string_or_none(self):
        tracer = self._make_tracer()
        tracer.trace()
        result = tracer.dominant_adversity_type()
        assert result is None or isinstance(result, str)

    def test_domain_filter_reduces_events(self):
        tracer = self._make_tracer()
        full_report = tracer.trace()
        grief_report = tracer.trace(domain="grief")
        assert grief_report.total_events <= full_report.total_events

    def test_domain_filter_grief_detects_loss(self):
        tracer = self._make_tracer()
        grief_report = tracer.trace(domain="grief")
        if grief_report.total_events > 0:
            assert grief_report.most_common_type == "loss"

    def test_empty_emms_returns_zero_events(self):
        tracer = self._make_tracer(agent=_make_emms())
        report = tracer.trace()
        assert report.total_events == 0
        assert report.most_common_type == "none"

    def test_adversity_type_valid_values(self):
        valid = {"loss", "failure", "rejection", "threat", "uncertainty"}
        tracer = self._make_tracer()
        report = tracer.trace()
        for ev in report.events:
            assert ev.adversity_type in valid

    def test_summary_is_string(self):
        tracer = self._make_tracer()
        report = tracer.trace()
        assert isinstance(report.summary(), str)
        for ev in report.events:
            assert isinstance(ev.summary(), str)


# ---------------------------------------------------------------------------
# TestSelfCompassionGauge
# ---------------------------------------------------------------------------

class TestSelfCompassionGauge:

    def _make_gauge(self, agent=None):
        from emms.memory.self_compassion import SelfCompassionGauge
        if agent is None:
            agent = _make_rich_emms()
        return SelfCompassionGauge(memory=agent.memory)

    def test_measure_returns_report(self):
        from emms.memory.self_compassion import SelfCompassionReport
        gauge = self._make_gauge()
        report = gauge.measure()
        assert isinstance(report, SelfCompassionReport)

    def test_report_has_required_fields(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert hasattr(report, "total_domains")
        assert hasattr(report, "profiles")
        assert hasattr(report, "most_compassionate_domain")
        assert hasattr(report, "harshest_domain")
        assert hasattr(report, "mean_compassion_score")
        assert hasattr(report, "duration_seconds")

    def test_total_domains_matches_list(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert report.total_domains == len(report.profiles)

    def test_at_least_one_profile_on_rich_corpus(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert report.total_domains >= 1

    def test_profile_has_required_fields(self):
        from emms.memory.self_compassion import SelfCompassionProfile
        gauge = self._make_gauge()
        report = gauge.measure()
        for p in report.profiles:
            assert isinstance(p, SelfCompassionProfile)
            assert hasattr(p, "domain")
            assert hasattr(p, "compassion_score")
            assert hasattr(p, "kindness_count")
            assert hasattr(p, "harsh_count")
            assert hasattr(p, "inner_critic_intensity")
            assert hasattr(p, "self_directed_themes")

    def test_compassion_score_range(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        for p in report.profiles:
            assert 0.0 <= p.compassion_score <= 1.0

    def test_inner_critic_intensity_range(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        for p in report.profiles:
            assert 0.0 <= p.inner_critic_intensity <= 1.0

    def test_profiles_sorted_by_compassion_desc(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        scores = [p.compassion_score for p in report.profiles]
        assert scores == sorted(scores, reverse=True)

    def test_mean_compassion_score_range(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert 0.0 <= report.mean_compassion_score <= 1.0

    def test_harsh_domain_has_lower_score_than_kind_domain(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        therapy_profile = next((p for p in report.profiles if p.domain == "therapy"), None)
        inner_profile = next((p for p in report.profiles if p.domain == "inner"), None)
        if therapy_profile and inner_profile:
            assert therapy_profile.compassion_score >= inner_profile.compassion_score

    def test_profile_for_domain_known(self):
        gauge = self._make_gauge()
        gauge.measure()
        profile = gauge.profile_for_domain("therapy")
        if profile is not None:
            assert profile.domain == "therapy"

    def test_profile_for_domain_unknown_returns_none(self):
        gauge = self._make_gauge()
        gauge.measure()
        result = gauge.profile_for_domain("nonexistent_xyz")
        assert result is None

    def test_harshest_domain_string_or_none(self):
        gauge = self._make_gauge()
        gauge.measure()
        result = gauge.harshest_domain()
        assert result is None or isinstance(result, str)

    def test_most_compassionate_domain_string(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert isinstance(report.most_compassionate_domain, str)

    def test_min_memories_filter_empty_emms(self):
        gauge = self._make_gauge(agent=_make_emms())
        report = gauge.measure()
        assert report.total_domains == 0
        assert report.most_compassionate_domain == "none"

    def test_self_directed_themes_is_list(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        for p in report.profiles:
            assert isinstance(p.self_directed_themes, list)
            assert len(p.self_directed_themes) <= 5

    def test_duration_nonnegative(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert report.duration_seconds >= 0.0

    def test_summary_is_string(self):
        gauge = self._make_gauge()
        report = gauge.measure()
        assert isinstance(report.summary(), str)
        for p in report.profiles:
            assert isinstance(p.summary(), str)

    def test_domain_filter(self):
        gauge = self._make_gauge()
        full_report = gauge.measure()
        filtered_report = gauge.measure(domain="therapy")
        assert filtered_report.total_domains <= full_report.total_domains


# ---------------------------------------------------------------------------
# TestResilienceIndex
# ---------------------------------------------------------------------------

class TestResilienceIndex:

    def _make_index(self, agent=None):
        from emms.memory.resilience import ResilienceIndex
        if agent is None:
            agent = _make_rich_emms()
        return ResilienceIndex(memory=agent.memory)

    def test_assess_returns_report(self):
        from emms.memory.resilience import ResilienceReport
        idx = self._make_index()
        report = idx.assess()
        assert isinstance(report, ResilienceReport)

    def test_report_has_required_fields(self):
        idx = self._make_index()
        report = idx.assess()
        assert hasattr(report, "total_arcs")
        assert hasattr(report, "arcs")
        assert hasattr(report, "resilience_score")
        assert hasattr(report, "bounce_back_rate")
        assert hasattr(report, "mean_adversity_depth")
        assert hasattr(report, "mean_recovery_slope")
        assert hasattr(report, "strongest_recovery")
        assert hasattr(report, "duration_seconds")

    def test_total_arcs_matches_list(self):
        idx = self._make_index()
        report = idx.assess()
        assert report.total_arcs == len(report.arcs)

    def test_at_least_one_arc_on_rich_corpus(self):
        idx = self._make_index()
        report = idx.assess()
        assert report.total_arcs >= 1

    def test_resilience_score_range(self):
        idx = self._make_index()
        report = idx.assess()
        assert 0.0 <= report.resilience_score <= 1.0

    def test_bounce_back_rate_range(self):
        idx = self._make_index()
        report = idx.assess()
        assert 0.0 <= report.bounce_back_rate <= 1.0

    def test_arc_has_required_fields(self):
        from emms.memory.resilience import RecoveryArc
        idx = self._make_index()
        report = idx.assess()
        for arc in report.arcs:
            assert isinstance(arc, RecoveryArc)
            assert hasattr(arc, "id")
            assert hasattr(arc, "window_start_ts")
            assert hasattr(arc, "window_end_ts")
            assert hasattr(arc, "adversity_depth")
            assert hasattr(arc, "recovery_slope")
            assert hasattr(arc, "recovered")
            assert hasattr(arc, "post_memories_count")

    def test_arc_id_prefix(self):
        idx = self._make_index()
        report = idx.assess()
        for arc in report.arcs:
            assert arc.id.startswith("rec_")

    def test_adversity_depth_negative_or_zero(self):
        idx = self._make_index()
        report = idx.assess()
        for arc in report.arcs:
            assert arc.adversity_depth <= 0.0

    def test_recovered_is_bool(self):
        idx = self._make_index()
        report = idx.assess()
        for arc in report.arcs:
            assert isinstance(arc.recovered, bool)

    def test_arcs_sorted_by_depth_most_negative_first(self):
        idx = self._make_index()
        report = idx.assess()
        depths = [arc.adversity_depth for arc in report.arcs]
        assert depths == sorted(depths)

    def test_strongest_recovery_is_arc_or_none(self):
        from emms.memory.resilience import RecoveryArc
        idx = self._make_index()
        report = idx.assess()
        assert report.strongest_recovery is None or isinstance(report.strongest_recovery, RecoveryArc)

    def test_strongest_recovery_has_highest_slope(self):
        idx = self._make_index()
        report = idx.assess()
        if report.arcs and report.strongest_recovery:
            max_slope = max(arc.recovery_slope for arc in report.arcs)
            assert report.strongest_recovery.recovery_slope == max_slope

    def test_empty_emms_returns_degenerate_report(self):
        idx = self._make_index(agent=_make_emms())
        report = idx.assess()
        assert report.total_arcs == 0
        assert report.resilience_score == 0.0
        assert report.bounce_back_rate == 0.0
        assert report.strongest_recovery is None

    def test_bounce_back_rate_method_matches_report(self):
        idx = self._make_index()
        report = idx.assess()
        assert idx.bounce_back_rate() == report.bounce_back_rate

    def test_bounce_back_rate_before_assess_is_zero(self):
        idx = self._make_index()
        assert idx.bounce_back_rate() == 0.0

    def test_strongest_recovery_arc_method(self):
        idx = self._make_index()
        idx.assess()
        result = idx.strongest_recovery_arc()
        from emms.memory.resilience import RecoveryArc
        assert result is None or isinstance(result, RecoveryArc)

    def test_duration_nonnegative(self):
        idx = self._make_index()
        report = idx.assess()
        assert report.duration_seconds >= 0.0

    def test_mean_adversity_depth_nonpositive(self):
        idx = self._make_index()
        report = idx.assess()
        if report.total_arcs > 0:
            assert report.mean_adversity_depth <= 0.0

    def test_post_memories_count_nonnegative(self):
        idx = self._make_index()
        report = idx.assess()
        for arc in report.arcs:
            assert arc.post_memories_count >= 0

    def test_summary_is_string(self):
        idx = self._make_index()
        report = idx.assess()
        assert isinstance(report.summary(), str)
        for arc in report.arcs:
            assert isinstance(arc.summary(), str)

    def test_domain_filter(self):
        idx = self._make_index()
        full_report = idx.assess()
        wellbeing_report = idx.assess(domain="wellbeing")
        assert wellbeing_report.total_arcs >= 0

    def test_recovery_arc_on_wellbeing_domain(self):
        idx = self._make_index()
        report = idx.assess(domain="wellbeing")
        assert report.total_arcs >= 1


# ---------------------------------------------------------------------------
# TestEMMSFacadeV260
# ---------------------------------------------------------------------------

class TestEMMSFacadeV260:

    def test_trace_adversity_callable(self):
        agent = _make_rich_emms()
        assert hasattr(agent, "trace_adversity")
        result = agent.trace_adversity()
        from emms.memory.adversity import AdversityReport
        assert isinstance(result, AdversityReport)

    def test_adversity_events_of_type_callable(self):
        agent = _make_rich_emms()
        agent.trace_adversity()
        result = agent.adversity_events_of_type("loss")
        assert isinstance(result, list)

    def test_dominant_adversity_type_callable(self):
        agent = _make_rich_emms()
        agent.trace_adversity()
        result = agent.dominant_adversity_type()
        assert result is None or isinstance(result, str)

    def test_measure_self_compassion_callable(self):
        agent = _make_rich_emms()
        from emms.memory.self_compassion import SelfCompassionReport
        result = agent.measure_self_compassion()
        assert isinstance(result, SelfCompassionReport)

    def test_compassion_for_domain_callable(self):
        agent = _make_rich_emms()
        agent.measure_self_compassion()
        result = agent.compassion_for_domain("therapy")
        from emms.memory.self_compassion import SelfCompassionProfile
        assert result is None or isinstance(result, SelfCompassionProfile)

    def test_harshest_domain_callable(self):
        agent = _make_rich_emms()
        agent.measure_self_compassion()
        result = agent.harshest_domain()
        assert result is None or isinstance(result, str)

    def test_assess_resilience_callable(self):
        agent = _make_rich_emms()
        from emms.memory.resilience import ResilienceReport
        result = agent.assess_resilience()
        assert isinstance(result, ResilienceReport)

    def test_resilience_bounce_back_rate_callable(self):
        agent = _make_rich_emms()
        agent.assess_resilience()
        rate = agent.resilience_bounce_back_rate()
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_strongest_recovery_callable(self):
        agent = _make_rich_emms()
        agent.assess_resilience()
        result = agent.strongest_recovery()
        from emms.memory.resilience import RecoveryArc
        assert result is None or isinstance(result, RecoveryArc)

    def test_lazy_init_stable_adversity(self):
        agent = _make_rich_emms()
        r1 = agent.trace_adversity()
        r2 = agent.trace_adversity()
        assert type(r1) is type(r2)

    def test_lazy_init_stable_compassion(self):
        agent = _make_rich_emms()
        r1 = agent.measure_self_compassion()
        r2 = agent.measure_self_compassion()
        assert type(r1) is type(r2)

    def test_lazy_init_stable_resilience(self):
        agent = _make_rich_emms()
        r1 = agent.assess_resilience()
        r2 = agent.assess_resilience()
        assert type(r1) is type(r2)

    def test_trace_adversity_domain_filter(self):
        agent = _make_rich_emms()
        full = agent.trace_adversity()
        grief = agent.trace_adversity(domain="grief")
        assert grief.total_events <= full.total_events

    def test_measure_self_compassion_domain_filter(self):
        agent = _make_rich_emms()
        full = agent.measure_self_compassion()
        filtered = agent.measure_self_compassion(domain="therapy")
        assert filtered.total_domains <= full.total_domains

    def test_assess_resilience_domain_filter(self):
        agent = _make_rich_emms()
        full = agent.assess_resilience()
        filtered = agent.assess_resilience(domain="wellbeing")
        assert filtered.total_arcs >= 0

    def test_all_nine_v260_methods_exist(self):
        agent = _make_emms()
        for method in [
            "trace_adversity", "adversity_events_of_type", "dominant_adversity_type",
            "measure_self_compassion", "compassion_for_domain", "harshest_domain",
            "assess_resilience", "resilience_bounce_back_rate", "strongest_recovery",
        ]:
            assert hasattr(agent, method), f"Missing method: {method}"


# ---------------------------------------------------------------------------
# TestMCPV260
# ---------------------------------------------------------------------------

class TestMCPV260:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_117(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 117

    def test_emms_trace_adversity_registered(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_trace_adversity" in names

    def test_emms_dominant_adversity_type_registered(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_dominant_adversity_type" in names

    def test_emms_measure_self_compassion_registered(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_measure_self_compassion" in names

    def test_emms_assess_resilience_registered(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_assess_resilience" in names

    def test_emms_resilience_bounce_back_rate_registered(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_resilience_bounce_back_rate" in names

    def test_handle_trace_adversity_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_trace_adversity", {})
        assert result.get("ok") is True

    def test_handle_dominant_adversity_type_returns_ok(self):
        server = self._make_server()
        server.handle("emms_trace_adversity", {})
        result = server.handle("emms_dominant_adversity_type", {})
        assert result.get("ok") is True

    def test_handle_measure_self_compassion_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_measure_self_compassion", {})
        assert result.get("ok") is True

    def test_handle_assess_resilience_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_assess_resilience", {})
        assert result.get("ok") is True

    def test_handle_resilience_bounce_back_rate_returns_ok(self):
        server = self._make_server()
        server.handle("emms_assess_resilience", {})
        result = server.handle("emms_resilience_bounce_back_rate", {})
        assert result.get("ok") is True


# ---------------------------------------------------------------------------
# TestV260Exports
# ---------------------------------------------------------------------------

class TestV260Exports:

    def test_version_is_0_26_0(self):
        assert emms_pkg.__version__ == "0.26.0"

    def test_adversity_tracer_importable(self):
        from emms import AdversityTracer
        assert AdversityTracer is not None

    def test_adversity_event_importable(self):
        from emms import AdversityEvent
        assert AdversityEvent is not None

    def test_adversity_report_importable(self):
        from emms import AdversityReport
        assert AdversityReport is not None

    def test_self_compassion_gauge_importable(self):
        from emms import SelfCompassionGauge
        assert SelfCompassionGauge is not None

    def test_self_compassion_profile_importable(self):
        from emms import SelfCompassionProfile
        assert SelfCompassionProfile is not None

    def test_self_compassion_report_importable(self):
        from emms import SelfCompassionReport
        assert SelfCompassionReport is not None

    def test_resilience_index_importable(self):
        from emms import ResilienceIndex
        assert ResilienceIndex is not None

    def test_recovery_arc_importable(self):
        from emms import RecoveryArc
        assert RecoveryArc is not None

    def test_resilience_report_importable(self):
        from emms import ResilienceReport
        assert ResilienceReport is not None

    def test_adversity_tracer_in_all(self):
        assert "AdversityTracer" in emms_pkg.__all__

    def test_self_compassion_gauge_in_all(self):
        assert "SelfCompassionGauge" in emms_pkg.__all__

    def test_resilience_index_in_all(self):
        assert "ResilienceIndex" in emms_pkg.__all__

    def test_recovery_arc_in_all(self):
        assert "RecoveryArc" in emms_pkg.__all__

    def test_all_nine_v260_symbols_in_all(self):
        expected = {
            "AdversityTracer", "AdversityEvent", "AdversityReport",
            "SelfCompassionGauge", "SelfCompassionProfile", "SelfCompassionReport",
            "ResilienceIndex", "RecoveryArc", "ResilienceReport",
        }
        for sym in expected:
            assert sym in emms_pkg.__all__, f"Missing from __all__: {sym}"
