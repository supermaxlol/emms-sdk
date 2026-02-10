"""Tests for persistent identity system."""

import json
import pytest
from pathlib import Path
from emms.core.models import Experience
from emms.identity.ego import PersistentIdentity, IdentityState


@pytest.fixture
def identity():
    return PersistentIdentity()


class TestIdentityState:
    def test_defaults(self):
        state = IdentityState()
        assert state.ego_id.startswith("ego_")
        assert state.identity_coherence == 0.9
        assert state.total_experiences == 0
        assert state.session_count == 0

    def test_serialisation(self):
        state = IdentityState(name="TestBot")
        json_str = state.model_dump_json()
        loaded = IdentityState(**json.loads(json_str))
        assert loaded.name == "TestBot"


class TestPersistentIdentity:
    def test_integrate_single(self, identity):
        exp = Experience(content="first experience", domain="test")
        result = identity.integrate(exp)
        assert result["total_experiences"] == 1
        assert "test" in result["domains"]

    def test_integrate_multiple_domains(self, identity):
        identity.integrate(Experience(content="finance stuff", domain="finance"))
        identity.integrate(Experience(content="tech stuff", domain="tech"))
        result = identity.integrate(Experience(content="weather stuff", domain="weather"))
        assert len(result["domains"]) == 3

    def test_autobiographical_memory(self, identity):
        for i in range(5):
            identity.integrate(Experience(content=f"event number {i}", domain="test"))
        assert len(identity.state.autobiographical) == 5

    def test_autobiographical_capped(self, identity):
        for i in range(110):
            identity.integrate(Experience(content=f"event {i}", domain="test"))
        assert len(identity.state.autobiographical) == 100

    def test_narrative_updates(self, identity):
        identity.integrate(Experience(content="important discovery", domain="science"))
        assert "important discovery" in identity.state.narrative
        assert "science" in identity.state.narrative or "1 domains" in identity.state.narrative

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "identity.json"

        # Create and save
        id1 = PersistentIdentity(storage_path=path)
        id1.integrate(Experience(content="remember this", domain="test"))
        id1.save()

        assert path.exists()

        # Load from file
        id2 = PersistentIdentity(storage_path=path)
        assert id2.state.total_experiences == 1
        assert "test" in id2.state.domains_seen
        assert id2.state.session_count == 1  # incremented on load
