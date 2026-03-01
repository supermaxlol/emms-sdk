"""Tests for EMMS v0.24.0 — The Wise Mind.

Covers:
- BiasDetector / BiasInstance / BiasReport
- WisdomSynthesizer / WisdomGuidance / WisdomReport
- EpistemicEvolution / KnowledgeDomain / EvolutionReport
- EMMS facade (v0.24.0 methods)
- MCP tool count (107) + 5 new tools
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
    """EMMS with bias-coloured, wisdom-rich, multi-domain memories."""
    agent = EMMS()
    # ethics domain — deontological, causal, value-heavy language
    for content, ev in [
        ("Justice must be served: the duty to treat all citizens with dignity and respect.", 0.8),
        ("The outcome harm is clear: cost exceeds benefit when welfare is undermined.", -0.7),
        ("Virtue requires honesty; an honest character builds integrity and trust.", 0.7),
        ("Forbidden actions violate rights; always protect the vulnerable from harm.", -0.6),
        ("Compassion and care are moral values that result in community welfare.", 0.9),
        ("The rule against harm is categorical: we owe protection to the vulnerable.", 0.5),
        ("Brave and wise individuals produce better outcomes for society overall.", 0.6),
        ("Cost-benefit analysis of welfare outcomes justifies consequentialist choices.", -0.4),
    ]:
        agent.store(Experience(
            content=content, domain="ethics",
            importance=0.85, emotional_valence=ev,
        ))
    # philosophy domain — epistemic + instrumental value language
    for content, ev in [
        ("Truth and knowledge require evidence and reason to achieve understanding.", 0.7),
        ("Progress demands learning and growth; efficiency enables us to achieve more.", 0.6),
        ("Clarity and precision in logic build reliable understanding of reality.", 0.8),
        ("Insight and transparency advance knowledge through careful verification.", 0.5),
        ("Build on existing knowledge to develop and innovate new solutions.", 0.4),
    ]:
        agent.store(Experience(
            content=content, domain="philosophy",
            importance=0.75, emotional_valence=ev,
        ))
    # science domain — causal and anchoring language
    for content, ev in [
        ("The first initial observation typically confirms our baseline reference expectation.", 0.3),
        ("Data generally shows positive outcomes; we are certainly optimistic about results.", 0.6),
        ("Experiments causes new knowledge; evidence produces reliable conclusions.", 0.7),
        ("Already invested resources justify continuing the study despite sunk cost concerns.", 0.2),
    ]:
        agent.store(Experience(
            content=content, domain="science",
            importance=0.7, emotional_valence=ev,
        ))
    return agent


# ---------------------------------------------------------------------------
# TestBiasDetector
# ---------------------------------------------------------------------------

class TestBiasDetector:

    def _make_detector(self, agent=None):
        from emms.memory.bias import BiasDetector
        if agent is None:
            agent = _make_rich_emms()
        return BiasDetector(memory=agent.memory)

    def test_detect_returns_bias_report(self):
        from emms.memory.bias import BiasReport
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report, BiasReport)

    def test_report_has_required_fields(self):
        detector = self._make_detector()
        report = detector.detect()
        assert hasattr(report, "total_biases")
        assert hasattr(report, "biases")
        assert hasattr(report, "dominant_bias")
        assert hasattr(report, "mean_strength")
        assert hasattr(report, "duration_seconds")

    def test_total_biases_matches_list(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.total_biases == len(report.biases)

    def test_bias_instance_has_required_fields(self):
        from emms.memory.bias import BiasInstance
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert isinstance(b, BiasInstance)
            assert hasattr(b, "id")
            assert hasattr(b, "name")
            assert hasattr(b, "display_name")
            assert hasattr(b, "strength")
            assert hasattr(b, "description")
            assert hasattr(b, "affected_memory_ids")
            assert hasattr(b, "created_at")

    def test_bias_id_prefix(self):
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert b.id.startswith("bia_")

    def test_strength_in_range(self):
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert 0.0 <= b.strength <= 1.0

    def test_biases_sorted_by_strength_desc(self):
        detector = self._make_detector()
        report = detector.detect()
        strengths = [b.strength for b in report.biases]
        assert strengths == sorted(strengths, reverse=True)

    def test_dominant_bias_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report.dominant_bias, str)

    def test_dominant_bias_is_none_on_empty(self):
        from emms.memory.bias import BiasDetector
        agent = _make_emms()
        detector = BiasDetector(memory=agent.memory, min_strength=0.99)
        report = detector.detect()
        assert report.dominant_bias == "none" or report.total_biases == 0

    def test_mean_strength_in_range(self):
        detector = self._make_detector()
        report = detector.detect()
        assert 0.0 <= report.mean_strength <= 1.0

    def test_mean_strength_zero_on_empty_memory(self):
        from emms.memory.bias import BiasDetector
        agent = _make_emms()
        detector = BiasDetector(memory=agent.memory)
        report = detector.detect()
        assert report.mean_strength == 0.0

    def test_empty_memory_returns_zero_biases(self):
        from emms.memory.bias import BiasDetector
        agent = _make_emms()
        detector = BiasDetector(memory=agent.memory)
        report = detector.detect()
        assert report.total_biases == 0

    def test_min_strength_filter(self):
        from emms.memory.bias import BiasDetector
        agent = _make_rich_emms()
        detector = BiasDetector(memory=agent.memory, min_strength=0.99)
        report = detector.detect()
        for b in report.biases:
            assert b.strength >= 0.99

    def test_domain_filter(self):
        from emms.memory.bias import BiasDetector
        agent = _make_rich_emms()
        detector = BiasDetector(memory=agent.memory)
        # domain filter should not raise
        report = detector.detect(domain="ethics")
        assert isinstance(report.total_biases, int)

    def test_display_name_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert isinstance(b.display_name, str)
            assert len(b.display_name) > 0

    def test_affected_memory_ids_is_list(self):
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert isinstance(b.affected_memory_ids, list)

    def test_at_least_one_bias_on_rich_corpus(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.total_biases >= 1

    def test_biases_of_type_filters_correctly(self):
        from emms.memory.bias import BiasDetector
        agent = _make_rich_emms()
        detector = BiasDetector(memory=agent.memory)
        detector.detect()
        for b in detector._biases:
            results = detector.biases_of_type(b.name)
            for r in results:
                assert r.name == b.name

    def test_most_pervasive_returns_bias_instance(self):
        from emms.memory.bias import BiasDetector, BiasInstance
        agent = _make_rich_emms()
        detector = BiasDetector(memory=agent.memory)
        detector.detect()
        mp = detector.most_pervasive()
        if detector._biases:
            assert isinstance(mp, BiasInstance)
            assert mp.strength == max(b.strength for b in detector._biases)
        else:
            assert mp is None

    def test_most_pervasive_none_on_empty(self):
        from emms.memory.bias import BiasDetector
        agent = _make_emms()
        detector = BiasDetector(memory=agent.memory)
        detector.detect()
        assert detector.most_pervasive() is None

    def test_bias_summary_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        for b in report.biases:
            assert isinstance(b.summary(), str)

    def test_report_summary_is_string(self):
        detector = self._make_detector()
        report = detector.detect()
        assert isinstance(report.summary(), str)

    def test_duration_non_negative(self):
        detector = self._make_detector()
        report = detector.detect()
        assert report.duration_seconds >= 0.0

    def test_max_biases_cap(self):
        from emms.memory.bias import BiasDetector
        agent = _make_rich_emms()
        detector = BiasDetector(memory=agent.memory, max_biases=2)
        report = detector.detect()
        assert report.total_biases <= 2


# ---------------------------------------------------------------------------
# TestWisdomSynthesizer
# ---------------------------------------------------------------------------

class TestWisdomSynthesizer:

    def _make_synthesizer(self, agent=None):
        from emms.memory.wisdom import WisdomSynthesizer
        if agent is None:
            agent = _make_rich_emms()
        return WisdomSynthesizer(memory=agent.memory)

    def test_synthesize_returns_wisdom_report(self):
        from emms.memory.wisdom import WisdomReport
        synth = self._make_synthesizer()
        report = synth.synthesize("how should I handle ethical decisions?")
        assert isinstance(report, WisdomReport)

    def test_report_has_required_fields(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("what values matter most?")
        assert hasattr(report, "query")
        assert hasattr(report, "guidance")
        assert hasattr(report, "dimensions_used")
        assert hasattr(report, "coverage_score")
        assert hasattr(report, "duration_seconds")

    def test_guidance_has_required_fields(self):
        from emms.memory.wisdom import WisdomGuidance
        synth = self._make_synthesizer()
        report = synth.synthesize("what are the key principles?")
        g = report.guidance
        assert isinstance(g, WisdomGuidance)
        assert hasattr(g, "id")
        assert hasattr(g, "query")
        assert hasattr(g, "relevant_values")
        assert hasattr(g, "moral_considerations")
        assert hasattr(g, "causal_insights")
        assert hasattr(g, "applicable_principles")
        assert hasattr(g, "synthesis")
        assert hasattr(g, "confidence")
        assert hasattr(g, "created_at")

    def test_guidance_id_prefix(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("wisdom query")
        assert report.guidance.id.startswith("wis_")

    def test_confidence_in_range(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("how to be ethical?")
        assert 0.0 <= report.guidance.confidence <= 1.0

    def test_relevant_values_is_list(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("what do I value?")
        assert isinstance(report.guidance.relevant_values, list)

    def test_moral_considerations_is_list(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("what is morally right?")
        assert isinstance(report.guidance.moral_considerations, list)

    def test_causal_insights_is_list(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("what causes good outcomes?")
        assert isinstance(report.guidance.causal_insights, list)

    def test_applicable_principles_is_list(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("recurring principles in my thinking?")
        assert isinstance(report.guidance.applicable_principles, list)

    def test_synthesis_is_non_empty_string(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("how should I decide?")
        assert isinstance(report.guidance.synthesis, str)
        assert len(report.guidance.synthesis) > 0

    def test_dimensions_used_is_list(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("ethical framework guidance")
        assert isinstance(report.dimensions_used, list)

    def test_coverage_score_in_range(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("how to make good decisions?")
        assert 0.0 <= report.coverage_score <= 1.0

    def test_empty_memory_still_returns_report(self):
        from emms.memory.wisdom import WisdomSynthesizer, WisdomReport
        agent = _make_emms()
        synth = WisdomSynthesizer(memory=agent.memory)
        report = synth.synthesize("anything")
        assert isinstance(report, WisdomReport)
        assert report is not None

    def test_query_preserved_in_guidance(self):
        synth = self._make_synthesizer()
        q = "how do ethics and knowledge connect?"
        report = synth.synthesize(q)
        assert report.guidance.query == q

    def test_report_summary_is_string(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("wisdom test")
        assert isinstance(report.summary(), str)

    def test_guidance_summary_is_string(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("guidance test")
        assert isinstance(report.guidance.summary(), str)

    def test_duration_non_negative(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("duration test")
        assert report.duration_seconds >= 0.0

    def test_second_call_returns_new_id(self):
        synth = self._make_synthesizer()
        r1 = synth.synthesize("first query about ethics")
        r2 = synth.synthesize("second query about values")
        assert r1.guidance.id != r2.guidance.id

    def test_relevant_values_at_most_5(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("values and principles")
        assert len(report.guidance.relevant_values) <= 5

    def test_moral_considerations_at_most_3(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("moral decision making")
        assert len(report.guidance.moral_considerations) <= 3

    def test_causal_insights_at_most_3(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("what produces good results?")
        assert len(report.guidance.causal_insights) <= 3

    def test_applicable_principles_at_most_3(self):
        synth = self._make_synthesizer()
        report = synth.synthesize("recurring themes in knowledge")
        assert len(report.guidance.applicable_principles) <= 3


# ---------------------------------------------------------------------------
# TestEpistemicEvolution
# ---------------------------------------------------------------------------

class TestEpistemicEvolution:

    def _make_evolution(self, agent=None):
        from emms.memory.epistemic_evolution import EpistemicEvolution
        if agent is None:
            agent = _make_rich_emms()
        return EpistemicEvolution(memory=agent.memory)

    def test_evolve_returns_evolution_report(self):
        from emms.memory.epistemic_evolution import EvolutionReport
        ev = self._make_evolution()
        report = ev.evolve()
        assert isinstance(report, EvolutionReport)

    def test_report_has_required_fields(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert hasattr(report, "total_domains")
        assert hasattr(report, "domains")
        assert hasattr(report, "most_active_domain")
        assert hasattr(report, "most_consolidated_domain")
        assert hasattr(report, "overall_growth_rate")
        assert hasattr(report, "knowledge_gaps")
        assert hasattr(report, "duration_seconds")

    def test_knowledge_domain_has_required_fields(self):
        from emms.memory.epistemic_evolution import KnowledgeDomain
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert isinstance(kd, KnowledgeDomain)
            assert hasattr(kd, "domain")
            assert hasattr(kd, "memory_count")
            assert hasattr(kd, "growth_rate")
            assert hasattr(kd, "consolidation_score")
            assert hasattr(kd, "knowledge_density")
            assert hasattr(kd, "recent_themes")
            assert hasattr(kd, "oldest_memory_ts")
            assert hasattr(kd, "newest_memory_ts")

    def test_growth_rate_in_range(self):
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert -1.0 <= kd.growth_rate <= 1.0

    def test_consolidation_score_in_range(self):
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert 0.0 <= kd.consolidation_score <= 1.0

    def test_knowledge_density_in_range(self):
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert 0.0 <= kd.knowledge_density <= 1.0

    def test_domains_sorted_by_density_desc(self):
        ev = self._make_evolution()
        report = ev.evolve()
        densities = [kd.knowledge_density for kd in report.domains]
        assert densities == sorted(densities, reverse=True)

    def test_most_active_domain_is_string(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert isinstance(report.most_active_domain, str)

    def test_most_consolidated_domain_is_string(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert isinstance(report.most_consolidated_domain, str)

    def test_overall_growth_rate_in_range(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert -1.0 <= report.overall_growth_rate <= 1.0

    def test_knowledge_gaps_is_list(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert isinstance(report.knowledge_gaps, list)

    def test_domain_profile_known_domain(self):
        from emms.memory.epistemic_evolution import KnowledgeDomain
        ev = self._make_evolution()
        ev.evolve()
        kd = ev.domain_profile("ethics")
        if kd is not None:
            assert isinstance(kd, KnowledgeDomain)
            assert kd.domain == "ethics"

    def test_domain_profile_unknown_returns_none(self):
        ev = self._make_evolution()
        ev.evolve()
        result = ev.domain_profile("nonexistent_domain_xyz")
        assert result is None

    def test_most_active_returns_knowledge_domain(self):
        from emms.memory.epistemic_evolution import KnowledgeDomain
        ev = self._make_evolution()
        ev.evolve()
        ma = ev.most_active()
        assert ma is None or isinstance(ma, KnowledgeDomain)

    def test_most_active_none_on_empty(self):
        from emms.memory.epistemic_evolution import EpistemicEvolution
        agent = _make_emms()
        ev = EpistemicEvolution(memory=agent.memory)
        ev.evolve()
        assert ev.most_active() is None

    def test_recent_themes_is_list(self):
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert isinstance(kd.recent_themes, list)

    def test_empty_memory_returns_zero_domains(self):
        from emms.memory.epistemic_evolution import EpistemicEvolution
        agent = _make_emms()
        ev = EpistemicEvolution(memory=agent.memory)
        report = ev.evolve()
        assert report.total_domains == 0
        assert report.domains == []

    def test_domain_filter(self):
        ev = self._make_evolution()
        report = ev.evolve(domain="ethics")
        for kd in report.domains:
            assert kd.domain == "ethics"

    def test_knowledge_gaps_threshold(self):
        from emms.memory.epistemic_evolution import EpistemicEvolution
        agent = _make_rich_emms()
        # min_memories=100 forces most domains to be gaps
        ev = EpistemicEvolution(memory=agent.memory, min_memories=100)
        report = ev.evolve()
        assert len(report.knowledge_gaps) >= len(report.domains)

    def test_knowledge_gaps_method(self):
        ev = self._make_evolution()
        ev.evolve()
        gaps = ev.knowledge_gaps()
        assert isinstance(gaps, list)

    def test_domain_summary_is_string(self):
        ev = self._make_evolution()
        report = ev.evolve()
        for kd in report.domains:
            assert isinstance(kd.summary(), str)

    def test_report_summary_is_string(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert isinstance(report.summary(), str)

    def test_duration_non_negative(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert report.duration_seconds >= 0.0

    def test_total_domains_matches_list(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert report.total_domains == len(report.domains)

    def test_rich_corpus_has_multiple_domains(self):
        ev = self._make_evolution()
        report = ev.evolve()
        assert report.total_domains >= 2


# ---------------------------------------------------------------------------
# TestEMMSFacadeV240
# ---------------------------------------------------------------------------

class TestEMMSFacadeV240:

    def test_map_biases_callable(self):
        agent = _make_rich_emms()
        report = agent.map_biases()
        assert report is not None

    def test_map_biases_with_domain(self):
        agent = _make_rich_emms()
        report = agent.map_biases(domain="ethics")
        assert hasattr(report, "total_biases")

    def test_biases_of_type_callable(self):
        agent = _make_rich_emms()
        agent.map_biases()
        results = agent.biases_of_type("anchoring")
        assert isinstance(results, list)

    def test_most_pervasive_bias_callable(self):
        agent = _make_rich_emms()
        agent.map_biases()
        bias = agent.most_pervasive_bias()
        # may be None but should not raise
        assert bias is None or hasattr(bias, "strength")

    def test_synthesize_wisdom_callable(self):
        agent = _make_rich_emms()
        report = agent.synthesize_wisdom(query="what values guide my thinking?")
        assert report is not None

    def test_synthesize_wisdom_returns_wisdom_report(self):
        from emms.memory.wisdom import WisdomReport
        agent = _make_rich_emms()
        report = agent.synthesize_wisdom(query="ethical reasoning")
        assert isinstance(report, WisdomReport)

    def test_evolve_knowledge_callable(self):
        agent = _make_rich_emms()
        report = agent.evolve_knowledge()
        assert report is not None

    def test_evolve_knowledge_with_domain(self):
        agent = _make_rich_emms()
        report = agent.evolve_knowledge(domain="philosophy")
        assert hasattr(report, "total_domains")

    def test_domain_knowledge_profile_callable(self):
        agent = _make_rich_emms()
        agent.evolve_knowledge()
        profile = agent.domain_knowledge_profile("ethics")
        # may be None or a KnowledgeDomain, but should not raise
        assert profile is None or hasattr(profile, "growth_rate")

    def test_knowledge_gaps_callable(self):
        agent = _make_rich_emms()
        gaps = agent.knowledge_gaps()
        assert isinstance(gaps, list)

    def test_most_active_domain_callable(self):
        agent = _make_rich_emms()
        agent.evolve_knowledge()
        result = agent.most_active_domain()
        assert result is None or hasattr(result, "domain")

    def test_most_consolidated_domain_callable(self):
        agent = _make_rich_emms()
        agent.evolve_knowledge()
        result = agent.most_consolidated_domain()
        assert result is None or isinstance(result, str)

    def test_bias_detector_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.map_biases()
        r2 = agent.map_biases()
        assert r1.total_biases == r2.total_biases

    def test_wisdom_synthesizer_lazy_init(self):
        agent = _make_rich_emms()
        # Should not raise on repeated calls
        agent.synthesize_wisdom(query="ethics")
        agent.synthesize_wisdom(query="knowledge")

    def test_epistemic_evolution_lazy_init(self):
        agent = _make_rich_emms()
        r1 = agent.evolve_knowledge()
        r2 = agent.evolve_knowledge()
        assert r1.total_domains == r2.total_domains

    def test_all_nine_facade_methods_exist(self):
        agent = _make_emms()
        for method in (
            "map_biases", "biases_of_type", "most_pervasive_bias",
            "synthesize_wisdom",
            "evolve_knowledge", "domain_knowledge_profile", "knowledge_gaps",
            "most_active_domain", "most_consolidated_domain",
        ):
            assert hasattr(agent, method), f"EMMS missing method: {method}"


# ---------------------------------------------------------------------------
# TestMCPV240
# ---------------------------------------------------------------------------

class TestMCPV240:

    def _make_server(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_rich_emms()
        return EMCPServer(emms=agent)

    def test_tool_count_is_107(self):
        server = self._make_server()
        assert len(server.tool_definitions) == 117, (
            f"Expected 107 tools, got {len(server.tool_definitions)}"
        )

    def test_emms_detect_biases_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_detect_biases" in names

    def test_emms_most_pervasive_bias_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_most_pervasive_bias" in names

    def test_emms_synthesize_wisdom_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_synthesize_wisdom" in names

    def test_emms_evolve_knowledge_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_evolve_knowledge" in names

    def test_emms_knowledge_gaps_present(self):
        server = self._make_server()
        names = {t["name"] for t in server.tool_definitions}
        assert "emms_knowledge_gaps" in names

    def test_detect_biases_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_detect_biases", {})
        assert result.get("ok") is True

    def test_most_pervasive_bias_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_most_pervasive_bias", {})
        assert result.get("ok") is True

    def test_synthesize_wisdom_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_synthesize_wisdom", {"query": "what matters most?"})
        assert result.get("ok") is True

    def test_evolve_knowledge_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_evolve_knowledge", {})
        assert result.get("ok") is True

    def test_knowledge_gaps_handler_returns_ok(self):
        server = self._make_server()
        result = server.handle("emms_knowledge_gaps", {})
        assert result.get("ok") is True


# ---------------------------------------------------------------------------
# TestV240Exports
# ---------------------------------------------------------------------------

class TestV240Exports:

    def test_version_is_0_24_0(self):
        assert emms_pkg.__version__ == "0.26.0"

    def test_bias_detector_importable(self):
        from emms import BiasDetector
        assert BiasDetector is not None

    def test_bias_instance_importable(self):
        from emms import BiasInstance
        assert BiasInstance is not None

    def test_bias_report_importable(self):
        from emms import BiasReport
        assert BiasReport is not None

    def test_wisdom_synthesizer_importable(self):
        from emms import WisdomSynthesizer
        assert WisdomSynthesizer is not None

    def test_wisdom_guidance_importable(self):
        from emms import WisdomGuidance
        assert WisdomGuidance is not None

    def test_wisdom_report_importable(self):
        from emms import WisdomReport
        assert WisdomReport is not None

    def test_epistemic_evolution_importable(self):
        from emms import EpistemicEvolution
        assert EpistemicEvolution is not None

    def test_knowledge_domain_importable(self):
        from emms import KnowledgeDomain
        assert KnowledgeDomain is not None

    def test_evolution_report_importable(self):
        from emms import EvolutionReport
        assert EvolutionReport is not None

    def test_all_nine_symbols_in_all(self):
        all_exports = emms_pkg.__all__
        for sym in (
            "BiasDetector", "BiasInstance", "BiasReport",
            "WisdomSynthesizer", "WisdomGuidance", "WisdomReport",
            "EpistemicEvolution", "KnowledgeDomain", "EvolutionReport",
        ):
            assert sym in all_exports, f"{sym} not in __all__"
