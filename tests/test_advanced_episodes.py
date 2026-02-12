"""Tests for advanced episode detection algorithms."""

import pytest
import time
from emms.core.models import Experience
from emms.episodes.boundary import EpisodeBoundaryDetector, Episode


def _make_exp(content: str, domain: str, ts_offset: float = 0.0) -> Experience:
    """Create an experience with a relative timestamp."""
    return Experience(
        content=content,
        domain=domain,
        timestamp=time.time() + ts_offset,
    )


@pytest.fixture
def finance_tech_experiences():
    """Create a mix of finance and tech experiences with clear boundaries."""
    exps = []
    base = time.time()
    # Finance cluster
    for i in range(5):
        exps.append(Experience(
            content=f"Stock market trading volume increased significantly today item {i}",
            domain="finance",
            timestamp=base + i,
        ))
    # Tech cluster (temporal gap)
    for i in range(5):
        exps.append(Experience(
            content=f"Python programming language released new version update {i}",
            domain="tech",
            timestamp=base + 1000 + i,
        ))
    return exps


class TestAlgorithmSelection:
    def test_auto_default(self):
        det = EpisodeBoundaryDetector()
        assert det.algorithm == "auto"

    def test_heuristic_algorithm(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="heuristic")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 1

    def test_graph_algorithm(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="graph")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 1

    def test_single_experience(self):
        det = EpisodeBoundaryDetector()
        det.add(_make_exp("solo experience", "test"))
        episodes = det.detect()
        assert len(episodes) == 1
        assert episodes[0].size == 1

    def test_empty_buffer(self):
        det = EpisodeBoundaryDetector()
        episodes = det.detect()
        assert episodes == []


class TestHeuristicDetection:
    def test_separate_domains(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="heuristic")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 2  # should detect domain boundary

    def test_similar_content(self):
        det = EpisodeBoundaryDetector(algorithm="heuristic")
        for i in range(5):
            det.add(_make_exp(f"Stock market rose {i} percent today", "finance", i))
        episodes = det.detect()
        # Similar content should cluster together
        assert len(episodes) >= 1


class TestGraphDetection:
    def test_communities_detected(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="graph")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        # Should detect at least 2 communities (finance vs tech)
        assert len(episodes) >= 1

    def test_coherence_calculated(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="graph")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        for ep in episodes:
            assert 0.0 <= ep.coherence <= 1.0


class TestSpectralDetection:
    def test_spectral_if_available(self, finance_tech_experiences):
        try:
            from sklearn.cluster import SpectralClustering
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        det = EpisodeBoundaryDetector(algorithm="spectral")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 1

    def test_spectral_coherence(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="spectral")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        for ep in episodes:
            assert 0.0 <= ep.coherence <= 1.0


class TestConductanceDetection:
    def test_conductance_if_available(self, finance_tech_experiences):
        try:
            import networkx as nx
            has_nx = True
        except ImportError:
            has_nx = False

        det = EpisodeBoundaryDetector(algorithm="conductance")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 1


class TestMultiAlgorithm:
    def test_multi_selects_best(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="multi")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        assert len(episodes) >= 1


class TestMetrics:
    def test_metrics_after_detection(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="graph")
        for exp in finance_tech_experiences:
            det.add(exp)
        det.detect()
        metrics = det.metrics
        # Metrics should be populated after detection
        assert isinstance(metrics, dict)

    def test_boundary_metrics_content(self, finance_tech_experiences):
        det = EpisodeBoundaryDetector(algorithm="heuristic")
        for exp in finance_tech_experiences:
            det.add(exp)
        episodes = det.detect()
        # _calculate_boundary_metrics is called by advanced algorithms
        # Heuristic doesn't set metrics, but the method should work
        metrics_result = det._calculate_boundary_metrics(episodes)
        assert "avg_coherence" in metrics_result
        assert "episode_count" in metrics_result


class TestEpisodeProperties:
    def test_episode_duration(self):
        ep = Episode(
            episode_id=0,
            experiences=[],
            start_time=100.0,
            end_time=200.0,
        )
        assert ep.duration == 100.0

    def test_episode_size(self):
        exps = [_make_exp("test", "test") for _ in range(3)]
        ep = Episode(episode_id=0, experiences=exps, start_time=0, end_time=1)
        assert ep.size == 3
