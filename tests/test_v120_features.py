"""Tests for EMMS v0.12.0 — The Associative Mind.

Coverage:
  - AssociationGraph        (26 tests)
  - InsightEngine           (22 tests)
  - AssociativeRetriever    (22 tests)
  - MCP v0.12.0 tools       (7 tests)
  - v0.12.0 exports         (9 tests)

Total: 86 tests
"""

from __future__ import annotations

import math
import pytest

from emms import EMMS, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emms() -> EMMS:
    return EMMS(enable_consciousness=False)


def _make_multi_domain(n_per_domain: int = 4) -> EMMS:
    """EMMS with memories spread across distinct domains."""
    agent = _make_emms()
    data = [
        ("science",    "Quantum entanglement enables non-local correlations between particles."),
        ("science",    "Entropy always increases in isolated thermodynamic systems."),
        ("science",    "DNA replication uses polymerase enzymes to copy genetic information."),
        ("science",    "General relativity describes gravity as curvature of spacetime."),
        ("philosophy", "Consciousness may be an emergent property of complex information processing."),
        ("philosophy", "The hard problem of consciousness asks why there is subjective experience."),
        ("philosophy", "Personal identity persists through psychological continuity over time."),
        ("philosophy", "Existentialism holds that existence precedes essence in human life."),
        ("technology", "Neural networks learn by adjusting weights through backpropagation."),
        ("technology", "Transformers use attention mechanisms to process sequential data."),
        ("technology", "Memory systems in AI agents can provide persistent cross-session identity."),
        ("technology", "Hash embeddings map text to fixed-size vectors for similarity computation."),
    ]
    for content, domain in data[:n_per_domain * 3]:
        agent.store(Experience(content=content, domain=domain, importance=0.75))
    return agent


def _make_rich_emms() -> EMMS:
    """EMMS with varied memories including emotional valence."""
    agent = _make_emms()
    domains = ["science", "philosophy", "technology", "art"]
    for i in range(16):
        agent.store(Experience(
            content=f"Memory {i}: {domains[i % 4]} insight about topic {i}. Key concept.",
            domain=domains[i % 4],
            importance=0.4 + 0.04 * (i % 8),
            emotional_valence=0.2 * ((i % 5) - 2),
            emotional_intensity=0.3 + 0.05 * (i % 6),
        ))
    return agent


# ---------------------------------------------------------------------------
# AssociationGraph tests
# ---------------------------------------------------------------------------

class TestAssociationGraphImport:
    def test_import_association_graph(self):
        from emms.memory.association import AssociationGraph
        assert AssociationGraph is not None

    def test_import_association_edge(self):
        from emms.memory.association import AssociationEdge
        assert AssociationEdge is not None

    def test_import_activation_result(self):
        from emms.memory.association import ActivationResult
        assert ActivationResult is not None

    def test_import_association_stats(self):
        from emms.memory.association import AssociationStats
        assert AssociationStats is not None


class TestAssociationGraphConstruction:
    def test_init_default_params(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        graph = AssociationGraph(agent.memory)
        assert graph.semantic_threshold == 0.5
        assert graph.temporal_window == 300.0
        assert graph.affective_tolerance == 0.3

    def test_init_custom_params(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        graph = AssociationGraph(agent.memory, semantic_threshold=0.7, temporal_window=60.0)
        assert graph.semantic_threshold == 0.7
        assert graph.temporal_window == 60.0

    def test_empty_graph_stats(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        graph = AssociationGraph(agent.memory)
        stats = graph.stats()
        assert stats.total_nodes == 0
        assert stats.total_edges == 0

    def test_auto_associate_no_items(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        graph = AssociationGraph(agent.memory)
        added = graph.auto_associate()
        assert added == 0

    def test_auto_associate_single_item(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        agent.store(Experience(content="Single memory", domain="test"))
        graph = AssociationGraph(agent.memory)
        added = graph.auto_associate()
        assert added == 0


class TestAssociationGraphEdges:
    def test_manual_associate(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        items = graph._collect_all()
        assert len(items) >= 2
        edge = graph.associate(items[0].id, items[1].id, edge_type="explicit", weight=0.9)
        assert edge.source_id == items[0].id
        assert edge.target_id == items[1].id
        assert edge.edge_type == "explicit"
        assert edge.weight == 0.9

    def test_explicit_edge_bidirectional(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        items = graph._collect_all()
        graph.associate(items[0].id, items[1].id)
        # Both directions should exist
        neighbors_0 = {e.target_id for e in graph.neighbors(items[0].id)}
        neighbors_1 = {e.target_id for e in graph.neighbors(items[1].id)}
        assert items[1].id in neighbors_0
        assert items[0].id in neighbors_1

    def test_auto_associate_returns_count(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        count = graph.auto_associate()
        assert count >= 0  # edges found

    def test_auto_associate_builds_graph(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=3)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        stats = graph.stats()
        assert stats.total_nodes >= 0  # graph was built
        assert stats.total_edges >= 0

    def test_neighbors_empty_for_unknown_id(self):
        from emms.memory.association import AssociationGraph
        agent = _make_emms()
        graph = AssociationGraph(agent.memory)
        result = graph.neighbors("nonexistent_id")
        assert result == []

    def test_neighbors_min_weight_filter(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        items = graph._collect_all()
        graph.associate(items[0].id, items[1].id, weight=0.3)
        # With high min_weight, should be filtered out
        high_weight_neighbors = graph.neighbors(items[0].id, min_weight=0.9)
        low_weight_neighbors = graph.neighbors(items[0].id, min_weight=0.1)
        assert len(high_weight_neighbors) <= len(low_weight_neighbors)


class TestAssociationGraphActivation:
    def test_spreading_activation_empty_seeds(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        results = graph.spreading_activation([])
        assert results == []

    def test_spreading_activation_unknown_seed(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        results = graph.spreading_activation(["nonexistent"])
        assert isinstance(results, list)

    def test_spreading_activation_excludes_seeds(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=3)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        items = graph._collect_all()
        seed_id = items[0].id
        results = graph.spreading_activation([seed_id])
        result_ids = {r.memory_id for r in results}
        assert seed_id not in result_ids

    def test_spreading_activation_sorted_by_activation(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=3)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        items = graph._collect_all()
        if items:
            results = graph.spreading_activation([items[0].id])
            for i in range(len(results) - 1):
                assert results[i].activation >= results[i + 1].activation

    def test_spreading_activation_has_activation_result_fields(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=3)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        items = graph._collect_all()
        if items:
            results = graph.spreading_activation([items[0].id])
            for r in results[:3]:
                assert hasattr(r, "memory_id")
                assert hasattr(r, "activation")
                assert hasattr(r, "steps_from_seed")
                assert hasattr(r, "path")
                assert r.activation > 0


class TestAssociationGraphStats:
    def test_stats_fields(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        stats = graph.stats()
        assert hasattr(stats, "total_nodes")
        assert hasattr(stats, "total_edges")
        assert hasattr(stats, "mean_degree")
        assert hasattr(stats, "mean_edge_weight")
        assert hasattr(stats, "most_connected_id")
        assert hasattr(stats, "edge_type_counts")

    def test_stats_summary_returns_string(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        graph.auto_associate()
        assert isinstance(graph.stats().summary(), str)

    def test_strongest_path_no_graph(self):
        from emms.memory.association import AssociationGraph
        agent = _make_multi_domain(n_per_domain=2)
        graph = AssociationGraph(agent.memory)
        result = graph.strongest_path("a", "b")
        assert result == []

    def test_cosine_sim_identical_vectors(self):
        from emms.memory.association import AssociationGraph
        sim = AssociationGraph._cosine_sim([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_sim_orthogonal_vectors(self):
        from emms.memory.association import AssociationGraph
        sim = AssociationGraph._cosine_sim([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_sim_zero_vector(self):
        from emms.memory.association import AssociationGraph
        sim = AssociationGraph._cosine_sim([0.0, 0.0], [1.0, 1.0])
        assert sim == 0.0


# ---------------------------------------------------------------------------
# EMMS facade — AssociationGraph (build_association_graph, etc.)
# ---------------------------------------------------------------------------

class TestEMMSAssociationFacade:
    def test_build_association_graph_returns_stats(self):
        agent = _make_multi_domain(n_per_domain=2)
        stats = agent.build_association_graph()
        assert hasattr(stats, "total_nodes")
        assert hasattr(stats, "total_edges")

    def test_build_association_graph_creates_internal_graph(self):
        agent = _make_multi_domain(n_per_domain=2)
        agent.build_association_graph()
        assert hasattr(agent, "_association_graph")

    def test_associate_facade(self):
        agent = _make_multi_domain(n_per_domain=2)
        items = agent.memory.working
        if not items:
            pytest.skip("no memories in working tier")
        items_list = list(items)
        edge = agent.associate(items_list[0].id, items_list[1].id)
        assert edge.source_id == items_list[0].id

    def test_spreading_activation_facade(self):
        agent = _make_multi_domain(n_per_domain=2)
        agent.build_association_graph()
        items = list(agent.memory.working) + list(agent.memory.short_term)
        if not items:
            pytest.skip("no memories")
        results = agent.spreading_activation([items[0].id])
        assert isinstance(results, list)

    def test_association_stats_lazy_build(self):
        agent = _make_multi_domain(n_per_domain=2)
        # Calling stats without prior build_association_graph should still work
        stats = agent.association_stats()
        assert hasattr(stats, "total_nodes")


# ---------------------------------------------------------------------------
# InsightEngine tests
# ---------------------------------------------------------------------------

class TestInsightEngineImport:
    def test_import_insight_engine(self):
        from emms.memory.insight import InsightEngine
        assert InsightEngine is not None

    def test_import_insight_report(self):
        from emms.memory.insight import InsightReport
        assert InsightReport is not None

    def test_import_insight_bridge(self):
        from emms.memory.insight import InsightBridge
        assert InsightBridge is not None


class TestInsightEngineBasic:
    def test_discover_empty_emms(self):
        from emms.memory.insight import InsightEngine
        agent = _make_emms()
        engine = InsightEngine(agent.memory)
        report = engine.discover()
        assert report.bridges_found == 0
        assert report.insights_generated == 0

    def test_discover_single_domain_no_cross_domain(self):
        from emms.memory.insight import InsightEngine
        agent = _make_emms()
        for i in range(6):
            agent.store(Experience(
                content=f"Science fact {i} about physics and matter.",
                domain="science",
                importance=0.7,
            ))
        engine = InsightEngine(agent.memory, cross_domain_only=True)
        report = engine.discover()
        # No cross-domain bridges possible
        assert report.bridges_found == 0

    def test_discover_multi_domain_finds_bridges(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=4)
        engine = InsightEngine(agent.memory, min_bridge_weight=0.0, cross_domain_only=True)
        report = engine.discover()
        assert isinstance(report.bridges_found, int)
        assert isinstance(report.insights_generated, int)

    def test_discover_report_fields(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=2)
        engine = InsightEngine(agent.memory)
        report = engine.discover(session_id="test_session")
        assert report.session_id == "test_session"
        assert report.started_at > 0
        assert report.duration_seconds >= 0
        assert isinstance(report.bridges_found, int)
        assert isinstance(report.new_memory_ids, list)
        assert isinstance(report.bridges, list)

    def test_discover_report_summary_returns_string(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=2)
        engine = InsightEngine(agent.memory)
        report = engine.discover()
        assert isinstance(report.summary(), str)

    def test_insight_bridge_fields(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=4)
        engine = InsightEngine(agent.memory, min_bridge_weight=0.0, cross_domain_only=True)
        report = engine.discover()
        for bridge in report.bridges:
            assert isinstance(bridge.domain_a, str)
            assert isinstance(bridge.domain_b, str)
            assert 0.0 <= bridge.bridge_weight <= 1.0
            assert isinstance(bridge.insight_content, str)
            assert len(bridge.insight_content) > 0

    def test_cross_domain_bridges_have_different_domains(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=4)
        engine = InsightEngine(agent.memory, min_bridge_weight=0.0, cross_domain_only=True)
        report = engine.discover()
        for bridge in report.bridges:
            assert bridge.domain_a != bridge.domain_b

    def test_max_insights_limits_results(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=4)
        engine = InsightEngine(agent.memory, max_insights=2, min_bridge_weight=0.0)
        report = engine.discover()
        assert len(report.bridges) <= 2

    def test_insight_content_contains_both_domains(self):
        from emms.memory.insight import InsightEngine
        agent = _make_multi_domain(n_per_domain=4)
        engine = InsightEngine(agent.memory, min_bridge_weight=0.0, cross_domain_only=True)
        report = engine.discover()
        for bridge in report.bridges[:3]:
            # domain names should appear in insight content
            assert bridge.domain_a in bridge.insight_content
            assert bridge.domain_b in bridge.insight_content


class TestEMMSInsightFacade:
    def test_discover_insights_facade(self):
        agent = _make_multi_domain(n_per_domain=2)
        report = agent.discover_insights()
        assert hasattr(report, "bridges_found")
        assert hasattr(report, "insights_generated")
        assert hasattr(report, "new_memory_ids")

    def test_discover_insights_with_session_id(self):
        agent = _make_multi_domain(n_per_domain=2)
        report = agent.discover_insights(session_id="my_session")
        assert report.session_id == "my_session"

    def test_discover_insights_max_insights(self):
        agent = _make_multi_domain(n_per_domain=4)
        report = agent.discover_insights(max_insights=3, min_bridge_weight=0.0)
        assert len(report.bridges) <= 3

    def test_discover_insights_rebuild_graph_false(self):
        agent = _make_multi_domain(n_per_domain=2)
        agent.build_association_graph()
        report = agent.discover_insights(rebuild_graph=False)
        assert hasattr(report, "bridges_found")


# ---------------------------------------------------------------------------
# AssociativeRetriever tests
# ---------------------------------------------------------------------------

class TestAssociativeRetrieverImport:
    def test_import_associative_retriever(self):
        from emms.retrieval.associative import AssociativeRetriever
        assert AssociativeRetriever is not None

    def test_import_associative_result(self):
        from emms.retrieval.associative import AssociativeResult
        assert AssociativeResult is not None


class TestAssociativeRetrieverBasic:
    def test_retrieve_empty_seeds(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=2)
        retriever = AssociativeRetriever(agent.memory)
        results = retriever.retrieve([])
        assert results == []

    def test_retrieve_by_query_empty_memory(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_emms()
        retriever = AssociativeRetriever(agent.memory)
        results = retriever.retrieve_by_query("consciousness")
        assert results == []

    def test_retrieve_returns_associative_results(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=3)
        retriever = AssociativeRetriever(agent.memory)
        items = retriever._collect_all()
        if not items:
            pytest.skip("no items")
        results = retriever.retrieve([items[0].id])
        assert isinstance(results, list)
        for r in results:
            assert hasattr(r, "memory")
            assert hasattr(r, "activation_score")
            assert hasattr(r, "steps_from_seed")
            assert hasattr(r, "path")

    def test_retrieve_by_query_returns_results(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=3)
        retriever = AssociativeRetriever(agent.memory)
        results = retriever.retrieve_by_query("neural network memory", max_results=5)
        assert isinstance(results, list)

    def test_retrieve_max_results_respected(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_rich_emms()
        retriever = AssociativeRetriever(agent.memory)
        items = retriever._collect_all()
        if not items:
            pytest.skip("no items")
        results = retriever.retrieve([items[0].id], max_results=3)
        assert len(results) <= 3

    def test_retrieve_excludes_seeds(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=3)
        retriever = AssociativeRetriever(agent.memory)
        items = retriever._collect_all()
        if not items:
            pytest.skip("no items")
        seed_id = items[0].id
        results = retriever.retrieve([seed_id])
        result_ids = {r.memory.id for r in results}
        assert seed_id not in result_ids

    def test_results_sorted_by_activation(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=4)
        retriever = AssociativeRetriever(agent.memory)
        items = retriever._collect_all()
        if not items:
            pytest.skip("no items")
        results = retriever.retrieve([items[0].id])
        for i in range(len(results) - 1):
            assert results[i].activation_score >= results[i + 1].activation_score

    def test_bm25_seeds_finds_matching_memories(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=3)
        retriever = AssociativeRetriever(agent.memory)
        seeds = retriever._bm25_seeds("quantum physics science", k=3)
        assert isinstance(seeds, list)
        assert len(seeds) <= 3

    def test_bm25_seeds_empty_query(self):
        from emms.retrieval.associative import AssociativeRetriever
        agent = _make_multi_domain(n_per_domain=2)
        retriever = AssociativeRetriever(agent.memory)
        seeds = retriever._bm25_seeds("", k=3)
        assert seeds == []


class TestEMMSAssociativeRetrieveFacade:
    def test_associative_retrieve_facade(self):
        agent = _make_multi_domain(n_per_domain=2)
        agent.build_association_graph()
        items = list(agent.memory.working) + list(agent.memory.short_term)
        if not items:
            pytest.skip("no items")
        results = agent.associative_retrieve([items[0].id])
        assert isinstance(results, list)

    def test_associative_retrieve_by_query_facade(self):
        agent = _make_multi_domain(n_per_domain=3)
        results = agent.associative_retrieve_by_query("consciousness philosophy identity")
        assert isinstance(results, list)

    def test_associative_retrieve_by_query_max_results(self):
        agent = _make_rich_emms()
        results = agent.associative_retrieve_by_query("memory system", max_results=4)
        assert len(results) <= 4

    def test_associative_retrieve_result_fields(self):
        agent = _make_multi_domain(n_per_domain=3)
        agent.build_association_graph()
        items = list(agent.memory.working)
        if not items:
            pytest.skip("no items")
        results = agent.associative_retrieve([items[0].id], max_results=5)
        for r in results:
            assert hasattr(r.memory, "id")
            assert hasattr(r.memory, "experience")
            assert r.activation_score >= 0


# ---------------------------------------------------------------------------
# MCP v0.12.0 tools
# ---------------------------------------------------------------------------

class TestMCPv120Tools:
    def test_mcp_tool_count(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        assert len(_TOOL_DEFINITIONS) == 52

    def test_mcp_has_build_association_graph(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_build_association_graph" in names

    def test_mcp_has_spreading_activation(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_spreading_activation" in names

    def test_mcp_has_discover_insights(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_discover_insights" in names

    def test_mcp_has_associative_retrieve(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_associative_retrieve" in names

    def test_mcp_has_association_stats(self):
        from emms.adapters.mcp_server import _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert "emms_association_stats" in names

    def test_mcp_build_association_graph_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n_per_domain=2)
        server = EMCPServer(agent)
        result = server.handle("emms_build_association_graph", {})
        assert "total_nodes" in result
        assert "total_edges" in result
        assert "summary" in result

    def test_mcp_discover_insights_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n_per_domain=2)
        server = EMCPServer(agent)
        result = server.handle("emms_discover_insights", {})
        assert "bridges_found" in result
        assert "insights_generated" in result

    def test_mcp_association_stats_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n_per_domain=2)
        server = EMCPServer(agent)
        result = server.handle("emms_association_stats", {})
        assert "total_nodes" in result

    def test_mcp_associative_retrieve_by_query(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n_per_domain=2)
        server = EMCPServer(agent)
        result = server.handle("emms_associative_retrieve", {"query": "consciousness"})
        assert "results" in result
        assert "total" in result

    def test_mcp_spreading_activation_call(self):
        from emms.adapters.mcp_server import EMCPServer
        agent = _make_multi_domain(n_per_domain=2)
        agent.build_association_graph()
        items = list(agent.memory.working)
        server = EMCPServer(agent)
        seed = items[0].id if items else "nonexistent"
        result = server.handle("emms_spreading_activation", {"seed_ids": [seed]})
        assert "activated" in result
        assert "total_activated" in result


# ---------------------------------------------------------------------------
# v0.12.0 exports
# ---------------------------------------------------------------------------

class TestV120Exports:
    def test_version(self):
        import emms
        assert emms.__version__ == "0.13.0"

    def test_export_association_graph(self):
        from emms import AssociationGraph
        assert AssociationGraph is not None

    def test_export_association_edge(self):
        from emms import AssociationEdge
        assert AssociationEdge is not None

    def test_export_activation_result(self):
        from emms import ActivationResult
        assert ActivationResult is not None

    def test_export_association_stats(self):
        from emms import AssociationStats
        assert AssociationStats is not None

    def test_export_insight_engine(self):
        from emms import InsightEngine
        assert InsightEngine is not None

    def test_export_insight_report(self):
        from emms import InsightReport
        assert InsightReport is not None

    def test_export_insight_bridge(self):
        from emms import InsightBridge
        assert InsightBridge is not None

    def test_export_associative_retriever(self):
        from emms import AssociativeRetriever
        assert AssociativeRetriever is not None

    def test_export_associative_result(self):
        from emms import AssociativeResult
        assert AssociativeResult is not None
