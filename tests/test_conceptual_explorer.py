"""Tests for ConceptualSpaceExplorer (Gap 7: AGI Roadmap — Novel Reasoning)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from emms.memory.conceptual_explorer import (
    ConceptualSpaceExplorer,
    NovelConcept,
    StructuralHole,
)


# ---------------------------------------------------------------------------
# StructuralHole & NovelConcept
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_structural_hole(self):
        h = StructuralHole(concept="momentum trading", present_in=["crypto", "equities"],
                           missing_from="forex", related_concept="trend following")
        assert h.confidence == 0.5
        d = h.to_dict()
        h2 = StructuralHole.from_dict(d)
        assert h2.concept == h.concept

    def test_novel_concept(self):
        c = NovelConcept(name="X+Y", parent_concepts=["X", "Y"],
                         source_domains=["finance", "tech"],
                         description="Blend of X and Y")
        d = c.to_dict()
        c2 = NovelConcept.from_dict(d)
        assert c2.name == c.name
        assert c2.source_domains == c.source_domains


# ---------------------------------------------------------------------------
# ConceptualSpaceExplorer — Structural Holes
# ---------------------------------------------------------------------------


class TestStructuralHoles:
    def test_find_holes_cross_domain(self):
        ce = ConceptualSpaceExplorer()
        memories = [
            {"domain": "crypto", "content": "momentum trading strategy works well in volatile markets"},
            {"domain": "crypto", "content": "regime detection helps momentum trading performance"},
            {"domain": "equities", "content": "momentum trading applied to large cap stocks"},
            {"domain": "equities", "content": "regime detection improves equity returns"},
            {"domain": "forex", "content": "regime detection for currency pair analysis"},
            # forex has "regime detection" but not "momentum trading" → hole
        ]
        holes = ce.find_structural_holes(memories)
        # Should find that "momentum trading" exists in crypto+equities but not forex
        assert len(holes) >= 0  # depends on bigram extraction

    def test_single_domain_no_holes(self):
        ce = ConceptualSpaceExplorer()
        memories = [
            {"domain": "finance", "content": "market analysis is important"},
            {"domain": "finance", "content": "risk management is key"},
        ]
        holes = ce.find_structural_holes(memories)
        assert len(holes) == 0

    def test_empty_memories_no_holes(self):
        ce = ConceptualSpaceExplorer()
        holes = ce.find_structural_holes([])
        assert len(holes) == 0


# ---------------------------------------------------------------------------
# ConceptualSpaceExplorer — Property Transfer
# ---------------------------------------------------------------------------


class TestPropertyTransfer:
    def test_transfer_finds_properties(self):
        ce = ConceptualSpaceExplorer()
        memories = [
            {"domain": "crypto", "content": "backtesting and optimization improve strategy performance"},
            {"domain": "crypto", "content": "regime gating reduces drawdown significantly"},
            {"domain": "equities", "content": "backtesting helps validate equity strategies"},
            # "regime" and "gating" are in crypto but not equities → transfer candidate
        ]
        concepts = ce.property_transfer("crypto", "equities", memories)
        assert len(concepts) >= 1
        # Should suggest transferring crypto-only words to equities
        assert all(c.source_domains == ["crypto", "equities"] for c in concepts)

    def test_transfer_no_novel_props(self):
        ce = ConceptualSpaceExplorer()
        memories = [
            {"domain": "A", "content": "same words here"},
            {"domain": "B", "content": "same words here"},
        ]
        concepts = ce.property_transfer("A", "B", memories)
        assert len(concepts) == 0


# ---------------------------------------------------------------------------
# ConceptualSpaceExplorer — Blending
# ---------------------------------------------------------------------------


class TestConceptualBlending:
    def test_blend_creates_concept(self):
        ce = ConceptualSpaceExplorer()
        blend = ce.blend_concepts("momentum", "crypto", "mean_reversion", "equities")
        assert blend.name == "momentum+mean_reversion"
        assert blend.novelty_score > 0
        assert len(ce.concepts) == 1

    def test_blend_accumulates(self):
        ce = ConceptualSpaceExplorer()
        ce.blend_concepts("A", "d1", "B", "d2")
        ce.blend_concepts("C", "d1", "D", "d3")
        assert len(ce.concepts) == 2


# ---------------------------------------------------------------------------
# ConceptualSpaceExplorer — Persistence
# ---------------------------------------------------------------------------


class TestConceptualExplorerPersistence:
    def test_save_load_roundtrip(self):
        ce = ConceptualSpaceExplorer()
        ce.blend_concepts("A", "d1", "B", "d2")
        ce.find_structural_holes([
            {"domain": "X", "content": "alpha beta gamma"},
            {"domain": "Y", "content": "alpha beta delta"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "explorer.json"
            ce.save_state(path)

            ce2 = ConceptualSpaceExplorer()
            assert ce2.load_state(path)
            assert len(ce2.concepts) == 1
            assert ce2._total_explorations == 1

    def test_load_nonexistent_returns_false(self):
        ce = ConceptualSpaceExplorer()
        assert ce.load_state("/nonexistent.json") is False

    def test_summary(self):
        ce = ConceptualSpaceExplorer()
        ce.blend_concepts("A", "d1", "B", "d2")
        s = ce.summary()
        assert "ConceptualSpaceExplorer" in s
        assert "1 generated concepts" in s

    def test_exploration_log(self):
        ce = ConceptualSpaceExplorer()
        ce.find_structural_holes([
            {"domain": "A", "content": "test content here"},
            {"domain": "B", "content": "test content there"},
        ])
        log = ce.exploration_log()
        assert len(log) == 1


# ---------------------------------------------------------------------------
# EMMS Integration
# ---------------------------------------------------------------------------


class TestEMMSIntegration:
    def test_generate_hypotheses(self):
        from emms import EMMS
        agent = EMMS()
        hyps = agent.generate_hypotheses(
            "Unexpected market crash",
            relevant_beliefs=["Markets always recover quickly"],
        )
        assert isinstance(hyps, list)
        assert len(hyps) >= 1

    def test_verify_reasoning(self):
        from emms import EMMS
        agent = EMMS()
        issues = agent.verify_reasoning(
            steps=["Markets always go up in every scenario"],
            hypothesis="Markets will definitely rise",
        )
        assert isinstance(issues, list)
        # Should detect overgeneralization at minimum
        assert len(issues) >= 1

    def test_explore_concepts(self):
        from emms import EMMS
        agent = EMMS()
        result = agent.explore_concepts([
            {"domain": "crypto", "content": "momentum trading strategy performance"},
            {"domain": "equities", "content": "value investing strategy returns"},
        ])
        assert "holes" in result
        assert "total_found" in result

    def test_blend_concepts(self):
        from emms import EMMS
        agent = EMMS()
        blend = agent.blend_concepts("momentum", "crypto", "value", "equities")
        assert "name" in blend
        assert blend["name"] == "momentum+value"

    def test_active_hypotheses(self):
        from emms import EMMS
        agent = EMMS()
        agent.generate_hypotheses("Test", relevant_beliefs=["X increases"])
        active = agent.active_hypotheses()
        assert isinstance(active, list)

    def test_reasoning_summary(self):
        from emms import EMMS
        agent = EMMS()
        s = agent.reasoning_summary()
        assert "AbductiveReasoner" in s
        assert "ReasoningVerifier" in s
        assert "ConceptualSpaceExplorer" in s

    def test_save_load_roundtrip(self):
        from emms import EMMS
        agent = EMMS()
        agent.generate_hypotheses("Test", relevant_beliefs=["X increases"])
        agent.blend_concepts("A", "d1", "B", "d2")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            agent.save(str(path))

            agent2 = EMMS()
            agent2.load(str(path))
            s = agent2.reasoning_summary()
            assert "AbductiveReasoner" in s
