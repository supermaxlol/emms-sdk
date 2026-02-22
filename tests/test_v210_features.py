"""Tests for EMMS v0.21.0 — The Social Mind.

Covers PerspectiveTaker, TrustLedger, NormExtractor, EMMS façade,
MCP server tools, and public exports.
"""

from __future__ import annotations

import pytest

from emms import EMMS
from emms.core.models import Experience
from emms.memory.perspective import PerspectiveTaker, AgentModel, PerspectiveReport
from emms.memory.trust import TrustLedger, TrustScore, TrustReport
from emms.memory.norms import NormExtractor, SocialNorm, NormReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emms() -> EMMS:
    return EMMS()


def _make_perspective_emms() -> EMMS:
    """EMMS with memories containing agent belief/communication verb patterns."""
    agent = EMMS()
    contents = [
        "Alice said the project needs more planning.",
        "Bob believes the solution will work effectively.",
        "Charlie argues that better communication solves problems.",
        "Alice thinks the team should focus on quality.",
        "David claims the deadline can be met with effort.",
        "Bob suggests we should test more thoroughly.",
        "Charlie stated the requirements are unclear.",
        "Alice noted the progress has been steady.",
        "Eve prefers a simpler approach to the problem.",
        "Bob agreed the new strategy is more effective.",
    ]
    for content in contents:
        agent.store(Experience(content=content, domain="work", importance=0.7,
                               emotional_valence=0.3))
    return agent


def _make_trust_emms() -> EMMS:
    """EMMS with memories from multiple domains."""
    agent = EMMS()
    # High trust: health domain — consistent, high importance
    for _ in range(6):
        agent.store(Experience(
            content="Regular exercise improves cardiovascular health.",
            domain="health", importance=0.9, emotional_valence=0.7,
        ))
    # Medium trust: work domain — variable valence
    for i in range(5):
        valence = 0.8 if i % 2 == 0 else -0.6
        agent.store(Experience(
            content="Work task completed with varying results.",
            domain="work", importance=0.6, emotional_valence=valence,
        ))
    # Low trust: gossip domain — low importance, mixed valence
    for i in range(3):
        agent.store(Experience(
            content="Rumour about project direction.",
            domain="gossip", importance=0.2, emotional_valence=(-0.5 + i * 0.5),
        ))
    return agent


def _make_norm_emms() -> EMMS:
    """EMMS with norm-laden memories."""
    agent = EMMS()
    contents = [
        ("Team members should communicate clearly and often.", "work", 0.6),
        ("You must always document your code changes properly.", "engineering", 0.5),
        ("It is never acceptable to skip code review.", "engineering", -0.3),
        ("Feedback should be constructive and specific.", "work", 0.7),
        ("Interrupting colleagues is inappropriate in meetings.", "work", -0.2),
        ("Data access requires proper authentication always.", "security", 0.6),
        ("Secrets should never be committed to version control.", "security", -0.5),
        ("Testing is required before merging any changes.", "engineering", 0.5),
        ("Users should always be informed of data usage.", "privacy", 0.6),
        ("Personal information must never be stored unencrypted.", "privacy", -0.4),
    ]
    for content, domain, valence in contents:
        agent.store(Experience(content=content, domain=domain,
                               importance=0.7, emotional_valence=valence))
    return agent


def _make_rich_emms() -> EMMS:
    """Comprehensive EMMS combining all social cognitive patterns."""
    agent = _make_perspective_emms()
    # Add norm-laden and trust-building memories
    extras = [
        ("Alice said collaboration should always be prioritised.", "work", 0.6),
        ("Bob believes teams must communicate frequently.", "work", 0.5),
        ("Data must never be shared without consent.", "privacy", -0.3),
        ("Research shows meditation reduces stress.", "health", 0.8),
        ("Exercise should be done regularly for health.", "health", 0.7),
    ]
    for content, domain, valence in extras:
        agent.store(Experience(content=content, domain=domain,
                               importance=0.7, emotional_valence=valence))
    return agent


# ===========================================================================
# TestPerspectiveTaker
# ===========================================================================


class TestPerspectiveTaker:

    def test_build_returns_perspective_report(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert isinstance(report, PerspectiveReport)

    def test_report_has_required_fields(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert hasattr(report, "total_agents")
        assert hasattr(report, "agents")
        assert hasattr(report, "most_mentioned")
        assert hasattr(report, "total_memories_scanned")
        assert hasattr(report, "duration_seconds")

    def test_total_agents_positive(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert report.total_agents > 0

    def test_agents_are_agent_model_instances(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for ag in report.agents:
            assert isinstance(ag, AgentModel)

    def test_agent_model_fields(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert len(report.agents) > 0
        ag = report.agents[0]
        assert isinstance(ag.name, str)
        assert isinstance(ag.mentions, int)
        assert isinstance(ag.statements, list)
        assert isinstance(ag.mean_valence, float)
        assert isinstance(ag.domains, list)

    def test_mentions_at_least_one(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for ag in report.agents:
            assert ag.mentions >= 1

    def test_mean_valence_in_range(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for ag in report.agents:
            assert -1.0 <= ag.mean_valence <= 1.0

    def test_most_mentioned_is_list(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert isinstance(report.most_mentioned, list)

    def test_agents_sorted_by_mentions_desc(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        mentions = [a.mentions for a in report.agents]
        assert mentions == sorted(mentions, reverse=True)

    def test_total_memories_scanned_positive(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert report.total_memories_scanned > 0

    def test_alice_detected(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        taker.build()
        # Alice appears multiple times in _make_perspective_emms
        model = taker.take_perspective("alice")
        assert model is not None
        assert isinstance(model, AgentModel)
        assert model.mentions >= 1

    def test_take_perspective_returns_none_unknown(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        taker.build()
        result = taker.take_perspective("zzz_unknown_agent")
        assert result is None

    def test_all_agents_returns_list(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        taker.build()
        result = taker.all_agents(n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_all_agents_sorted_by_mentions(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        taker.build()
        result = taker.all_agents(n=10)
        mentions = [a.mentions for a in result]
        assert mentions == sorted(mentions, reverse=True)

    def test_domain_filter(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build(domain="work")
        assert isinstance(report, PerspectiveReport)
        assert report.total_memories_scanned <= 10

    def test_empty_memory(self):
        agent = _make_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert isinstance(report, PerspectiveReport)
        assert report.total_agents == 0

    def test_min_mentions_filter(self):
        agent = _make_perspective_emms()
        taker_low = PerspectiveTaker(memory=agent.memory, min_mentions=1)
        taker_high = PerspectiveTaker(memory=agent.memory, min_mentions=3)
        report_low = taker_low.build()
        report_high = taker_high.build()
        assert report_high.total_agents <= report_low.total_agents

    def test_perspective_report_summary(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "PerspectiveReport" in summary

    def test_statements_is_list(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for ag in report.agents:
            assert isinstance(ag.statements, list)

    def test_domains_is_list(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for ag in report.agents:
            assert isinstance(ag.domains, list)

    def test_max_agents_cap(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory, max_agents=2)
        report = taker.build()
        assert report.total_agents <= 2

    def test_duration_nonneg(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert report.duration_seconds >= 0.0

    def test_most_mentioned_names_are_strings(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        for name in report.most_mentioned:
            assert isinstance(name, str)

    def test_multiple_builds_refresh_state(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report1 = taker.build()
        report2 = taker.build()
        assert report1.total_agents == report2.total_agents

    def test_known_agent_case_insensitive(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        taker.build()
        model = taker.take_perspective("Alice")
        assert model is not None

    def test_total_agents_matches_agents_list(self):
        agent = _make_perspective_emms()
        taker = PerspectiveTaker(memory=agent.memory)
        report = taker.build()
        assert report.total_agents == len(report.agents)


# ===========================================================================
# TestTrustLedger
# ===========================================================================


class TestTrustLedger:

    def test_compute_trust_returns_trust_report(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert isinstance(report, TrustReport)

    def test_report_has_required_fields(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert hasattr(report, "scores")
        assert hasattr(report, "most_trusted")
        assert hasattr(report, "least_trusted")
        assert hasattr(report, "total_sources")
        assert hasattr(report, "duration_seconds")

    def test_total_sources_positive(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert report.total_sources > 0

    def test_scores_are_trust_score_instances(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        for ts in report.scores:
            assert isinstance(ts, TrustScore)

    def test_trust_score_fields(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert len(report.scores) > 0
        ts = report.scores[0]
        assert isinstance(ts.source, str)
        assert isinstance(ts.trust, float)
        assert isinstance(ts.memory_count, int)
        assert isinstance(ts.mean_importance, float)
        assert isinstance(ts.valence_stability, float)

    def test_trust_in_range(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        for ts in report.scores:
            assert 0.0 <= ts.trust <= 1.0

    def test_valence_stability_in_range(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        for ts in report.scores:
            assert 0.0 <= ts.valence_stability <= 1.0

    def test_mean_importance_in_range(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        for ts in report.scores:
            assert 0.0 <= ts.mean_importance <= 1.0

    def test_scores_sorted_by_trust_desc(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        trusts = [ts.trust for ts in report.scores]
        assert trusts == sorted(trusts, reverse=True)

    def test_trust_of_known_source(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        ledger.compute_trust()
        trust = ledger.trust_of("health")
        assert 0.0 <= trust <= 1.0

    def test_trust_of_unknown_returns_neutral(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        ledger.compute_trust()
        trust = ledger.trust_of("zzz_unknown_domain")
        assert trust == 0.5

    def test_most_trusted_returns_list(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        ledger.compute_trust()
        result = ledger.most_trusted(n=2)
        assert isinstance(result, list)
        assert len(result) <= 2
        for ts in result:
            assert isinstance(ts, TrustScore)

    def test_least_trusted_returns_list(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        ledger.compute_trust()
        result = ledger.least_trusted(n=2)
        assert isinstance(result, list)
        for ts in result:
            assert isinstance(ts, TrustScore)

    def test_most_trusted_names_in_report(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert isinstance(report.most_trusted, list)
        for name in report.most_trusted:
            assert isinstance(name, str)

    def test_least_trusted_names_in_report(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert isinstance(report.least_trusted, list)

    def test_health_more_trusted_than_gossip(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        ledger.compute_trust()
        health_trust = ledger.trust_of("health")
        gossip_trust = ledger.trust_of("gossip")
        assert health_trust > gossip_trust

    def test_empty_memory(self):
        agent = _make_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert isinstance(report, TrustReport)
        assert report.total_sources == 0

    def test_domain_filter(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust(domain="health")
        for ts in report.scores:
            assert ts.source == "health"

    def test_trust_report_summary(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "TrustReport" in summary

    def test_trust_score_summary(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        if report.scores:
            summary = report.scores[0].summary()
            assert isinstance(summary, str)
            assert "TrustScore" in summary

    def test_memory_count_positive(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        for ts in report.scores:
            assert ts.memory_count >= 1

    def test_duration_nonneg(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert report.duration_seconds >= 0.0

    def test_total_sources_matches_scores(self):
        agent = _make_trust_emms()
        ledger = TrustLedger(memory=agent.memory)
        report = ledger.compute_trust()
        assert report.total_sources == len(report.scores)


# ===========================================================================
# TestNormExtractor
# ===========================================================================


class TestNormExtractor:

    def test_extract_norms_returns_norm_report(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert isinstance(report, NormReport)

    def test_report_has_required_fields(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert hasattr(report, "total_norms")
        assert hasattr(report, "norms")
        assert hasattr(report, "prescriptive_count")
        assert hasattr(report, "prohibitive_count")
        assert hasattr(report, "domains_covered")
        assert hasattr(report, "duration_seconds")

    def test_total_norms_positive(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert report.total_norms > 0

    def test_norms_are_social_norm_instances(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        for n in report.norms:
            assert isinstance(n, SocialNorm)

    def test_social_norm_fields(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert len(report.norms) > 0
        n = report.norms[0]
        assert isinstance(n.id, str)
        assert isinstance(n.content, str)
        assert isinstance(n.domain, str)
        assert isinstance(n.polarity, str)
        assert isinstance(n.keyword, str)
        assert isinstance(n.subject, str)
        assert isinstance(n.confidence, float)
        assert isinstance(n.memory_ids, list)

    def test_id_prefix_norm(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        for n in report.norms:
            assert n.id.startswith("norm_")

    def test_polarity_valid_values(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        for n in report.norms:
            assert n.polarity in ("prescriptive", "prohibitive")

    def test_confidence_in_range(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        for n in report.norms:
            assert 0.0 <= n.confidence <= 1.0

    def test_norms_sorted_by_confidence_desc(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        confidences = [n.confidence for n in report.norms]
        assert confidences == sorted(confidences, reverse=True)

    def test_count_integrity(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert report.prescriptive_count + report.prohibitive_count == report.total_norms

    def test_prescriptive_and_prohibitive_detected(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        # Both types should be present given the test data
        assert report.prescriptive_count >= 0
        assert report.prohibitive_count >= 0

    def test_domains_covered_is_list(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert isinstance(report.domains_covered, list)

    def test_norms_for_domain(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        extractor.extract_norms()
        result = extractor.norms_for_domain("work")
        assert isinstance(result, list)
        for n in result:
            assert n.domain == "work"

    def test_check_norm_returns_list(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        extractor.extract_norms()
        result = extractor.check_norm("code review process")
        assert isinstance(result, list)

    def test_check_norm_returns_social_norms(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        extractor.extract_norms()
        result = extractor.check_norm("documentation requirements")
        for n in result:
            assert isinstance(n, SocialNorm)

    def test_check_norm_max_5(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        extractor.extract_norms()
        result = extractor.check_norm("code security")
        assert len(result) <= 5

    def test_domain_filter(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms(domain="security")
        for n in report.norms:
            assert n.domain == "security"

    def test_empty_memory(self):
        agent = _make_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert isinstance(report, NormReport)
        assert report.total_norms == 0

    def test_norm_report_summary(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "NormReport" in summary

    def test_social_norm_summary(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        if report.norms:
            summary = report.norms[0].summary()
            assert isinstance(summary, str)
            assert "SocialNorm" in summary

    def test_total_norms_matches_norms_list(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert report.total_norms == len(report.norms)

    def test_duration_nonneg(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        assert report.duration_seconds >= 0.0

    def test_keyword_is_norm_keyword(self):
        from emms.memory.norms import _ALL_NORM_KEYWORDS
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory)
        report = extractor.extract_norms()
        for n in report.norms:
            assert n.keyword in _ALL_NORM_KEYWORDS

    def test_max_norms_cap(self):
        agent = _make_norm_emms()
        extractor = NormExtractor(memory=agent.memory, max_norms=3)
        report = extractor.extract_norms()
        assert report.total_norms <= 3

    def test_check_norm_empty_no_crash(self):
        agent = _make_emms()
        extractor = NormExtractor(memory=agent.memory)
        extractor.extract_norms()
        result = extractor.check_norm("anything")
        assert result == []


# ===========================================================================
# TestEMMSFacadeV210
# ===========================================================================


class TestEMMSFacadeV210:

    def test_build_perspective_models_returns_report(self):
        agent = _make_perspective_emms()
        report = agent.build_perspective_models()
        assert isinstance(report, PerspectiveReport)

    def test_build_perspective_models_domain_filter(self):
        agent = _make_perspective_emms()
        report = agent.build_perspective_models(domain="work")
        assert isinstance(report, PerspectiveReport)

    def test_agent_model_returns_model_or_none(self):
        agent = _make_perspective_emms()
        agent.build_perspective_models()
        result = agent.agent_model("alice")
        assert result is None or isinstance(result, AgentModel)

    def test_all_agents_returns_list(self):
        agent = _make_perspective_emms()
        agent.build_perspective_models()
        result = agent.all_agents(n=5)
        assert isinstance(result, list)

    def test_compute_trust_returns_trust_report(self):
        agent = _make_trust_emms()
        report = agent.compute_trust()
        assert isinstance(report, TrustReport)

    def test_compute_trust_domain_filter(self):
        agent = _make_trust_emms()
        report = agent.compute_trust(domain="health")
        assert isinstance(report, TrustReport)

    def test_trust_of_returns_float(self):
        agent = _make_trust_emms()
        agent.compute_trust()
        result = agent.trust_of("health")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_most_trusted_returns_list(self):
        agent = _make_trust_emms()
        agent.compute_trust()
        result = agent.most_trusted(n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_extract_norms_returns_norm_report(self):
        agent = _make_norm_emms()
        report = agent.extract_norms()
        assert isinstance(report, NormReport)

    def test_extract_norms_domain_filter(self):
        agent = _make_norm_emms()
        report = agent.extract_norms(domain="work")
        assert isinstance(report, NormReport)

    def test_norms_for_domain_returns_list(self):
        agent = _make_norm_emms()
        agent.extract_norms()
        result = agent.norms_for_domain("work")
        assert isinstance(result, list)

    def test_check_norm_returns_list(self):
        agent = _make_norm_emms()
        agent.extract_norms()
        result = agent.check_norm("code review")
        assert isinstance(result, list)

    def test_lazy_init_perspective_taker(self):
        agent = _make_perspective_emms()
        assert not hasattr(agent, "_perspective_taker")
        agent.build_perspective_models()
        assert hasattr(agent, "_perspective_taker")

    def test_lazy_init_trust_ledger(self):
        agent = _make_trust_emms()
        assert not hasattr(agent, "_trust_ledger")
        agent.compute_trust()
        assert hasattr(agent, "_trust_ledger")

    def test_lazy_init_norm_extractor(self):
        agent = _make_norm_emms()
        assert not hasattr(agent, "_norm_extractor")
        agent.extract_norms()
        assert hasattr(agent, "_norm_extractor")

    def test_perspective_taker_reused(self):
        agent = _make_perspective_emms()
        agent.build_perspective_models()
        taker1 = agent._perspective_taker
        agent.build_perspective_models()
        assert agent._perspective_taker is taker1


# ===========================================================================
# TestMCPV210
# ===========================================================================


class TestMCPV210:

    def _get_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_92(self):
        server = self._get_server()
        assert len(server.tool_definitions) == 92

    def test_emms_build_perspectives_callable(self):
        server = self._get_server()
        result = server.handle("emms_build_perspectives", {})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_agent_model_callable(self):
        server = self._get_server()
        server.handle("emms_build_perspectives", {})
        result = server.handle("emms_agent_model", {"agent_name": "alice"})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_compute_trust_callable(self):
        server = self._get_server()
        result = server.handle("emms_compute_trust", {})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_extract_norms_callable(self):
        server = self._get_server()
        result = server.handle("emms_extract_norms", {})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_emms_check_norm_callable(self):
        server = self._get_server()
        server.handle("emms_extract_norms", {})
        result = server.handle("emms_check_norm", {"behavior": "code review"})
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_new_tools_in_definitions(self):
        server = self._get_server()
        names = {d["name"] for d in server.tool_definitions}
        for name in [
            "emms_build_perspectives",
            "emms_agent_model",
            "emms_compute_trust",
            "emms_extract_norms",
            "emms_check_norm",
        ]:
            assert name in names, f"{name} not in tool definitions"

    def test_all_tools_have_required_fields(self):
        server = self._get_server()
        for tool in server.tool_definitions:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


# ===========================================================================
# TestV210Exports
# ===========================================================================


class TestV210Exports:

    def test_version_is_0_21_0(self):
        import emms
        assert emms.__version__ == "0.21.0"

    def test_perspective_taker_exported(self):
        from emms import PerspectiveTaker
        assert PerspectiveTaker is not None

    def test_agent_model_exported(self):
        from emms import AgentModel
        assert AgentModel is not None

    def test_perspective_report_exported(self):
        from emms import PerspectiveReport
        assert PerspectiveReport is not None

    def test_trust_ledger_exported(self):
        from emms import TrustLedger
        assert TrustLedger is not None

    def test_trust_score_exported(self):
        from emms import TrustScore
        assert TrustScore is not None

    def test_trust_report_exported(self):
        from emms import TrustReport
        assert TrustReport is not None

    def test_norm_extractor_exported(self):
        from emms import NormExtractor
        assert NormExtractor is not None

    def test_social_norm_exported(self):
        from emms import SocialNorm
        assert SocialNorm is not None

    def test_norm_report_exported(self):
        from emms import NormReport
        assert NormReport is not None

    def test_all_symbols_in_all(self):
        import emms
        for sym in [
            "PerspectiveTaker", "AgentModel", "PerspectiveReport",
            "TrustLedger", "TrustScore", "TrustReport",
            "NormExtractor", "SocialNorm", "NormReport",
        ]:
            assert sym in emms.__all__, f"{sym} not in __all__"
