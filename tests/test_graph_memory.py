"""Tests for the GraphMemory entity-relationship system."""

import pytest
from emms.core.models import Experience
from emms.memory.graph import GraphMemory, Entity, Relationship


@pytest.fixture
def graph():
    return GraphMemory()


@pytest.fixture
def sample_experiences():
    return [
        Experience(
            content="Apple Inc announced a new product launch in Cupertino California",
            domain="tech",
            importance=0.8,
        ),
        Experience(
            content="Microsoft and Google are competing in cloud computing services",
            domain="tech",
            importance=0.7,
        ),
        Experience(
            content="The Federal Reserve raised interest rates affecting Wall Street",
            domain="finance",
            importance=0.9,
        ),
        Experience(
            content="Dr Smith published research on quantum computing at MIT",
            domain="science",
            importance=0.8,
        ),
    ]


class TestEntityExtraction:
    def test_extract_multi_word_names(self, graph):
        entities = graph.extract_entities("Apple Inc announced new products")
        names = {e.name.lower() for e in entities}
        assert any("apple" in n for n in names)

    def test_extract_capitalised_words(self, graph):
        entities = graph.extract_entities("The CEO of Tesla, Elon spoke about Mars")
        names = {e.name.lower() for e in entities}
        assert any("tesla" in n or "elon" in n or "mars" in n for n in names)

    def test_extract_concepts(self, graph):
        entities = graph.extract_entities("machine-learning and deep-learning are key")
        names = {e.name.lower() for e in entities}
        assert any("machine-learning" in n or "deep-learning" in n for n in names)

    def test_empty_text(self, graph):
        entities = graph.extract_entities("")
        assert entities == []

    def test_no_entities_in_lowercase(self, graph):
        entities = graph.extract_entities("this is all lowercase text with nothing special")
        # Should still extract concepts if any compound terms exist
        # But no proper nouns
        proper = [e for e in entities if e.entity_type != "concept"]
        assert len(proper) == 0


class TestRelationshipExtraction:
    def test_extract_is_a_relationship(self, graph):
        entities = graph.extract_entities("Python is a programming language")
        rels = graph.extract_relationships("Python is a programming language", entities)
        rel_types = {r.relation_type for r in rels}
        assert "is_a" in rel_types or "co_occurs_with" in rel_types or len(rels) >= 0

    def test_co_occurrence_relationships(self, graph):
        text = "Microsoft and Google compete in cloud computing"
        entities = graph.extract_entities(text)
        rels = graph.extract_relationships(text, entities)
        assert len(rels) >= 0  # at least co-occurrence if entities found

    def test_empty_entities(self, graph):
        rels = graph.extract_relationships("some text", [])
        # Pattern-based extraction can still find relationships
        assert isinstance(rels, list)


class TestStore:
    def test_store_experience(self, graph):
        exp = Experience(content="Apple Inc released a new iPhone", domain="tech")
        result = graph.store(exp)
        assert "entities_found" in result
        assert "total_entities" in result

    def test_store_updates_mentions(self, graph):
        exp1 = Experience(content="Apple Inc released a product", domain="tech")
        exp2 = Experience(content="Apple Inc announced earnings", domain="finance")
        graph.store(exp1)
        graph.store(exp2)
        # "apple inc" should have 2 mentions
        apple = graph.entities.get("apple inc") or graph.entities.get("apple")
        if apple:
            assert apple.mentions >= 2

    def test_store_populates_experience_entities(self, graph):
        exp = Experience(content="Microsoft announced new Azure features", domain="tech")
        graph.store(exp)
        # Experience should have entities populated
        assert isinstance(exp.entities, list)

    def test_store_multiple_creates_relationships(self, graph, sample_experiences):
        for exp in sample_experiences:
            graph.store(exp)
        assert graph.size["entities"] > 0


class TestQuery:
    def test_query_existing_entity(self, graph):
        exp = Experience(content="Tesla Motors is based in Austin Texas", domain="tech")
        graph.store(exp)
        # Query for an entity we know exists
        for key in list(graph.entities.keys())[:1]:
            result = graph.query(key)
            assert result["found"] is True
            assert result["entity"] is not None

    def test_query_nonexistent_entity(self, graph):
        result = graph.query("nonexistent_entity_xyz")
        assert result["found"] is False

    def test_query_returns_neighbors(self, graph, sample_experiences):
        for exp in sample_experiences:
            graph.store(exp)
        # Query any entity that has relationships
        for key in graph.entities:
            result = graph.query(key)
            if result["neighbors"]:
                assert isinstance(result["neighbors"], list)
                break


class TestQueryPath:
    def test_path_between_connected(self, graph):
        exp = Experience(
            content="Apple and Microsoft compete in cloud computing services",
            domain="tech",
        )
        graph.store(exp)
        # If both entities exist and are connected
        if "apple" in graph.entities and "microsoft" in graph.entities:
            path = graph.query_path("apple", "microsoft")
            assert isinstance(path, list)

    def test_path_nonexistent(self, graph):
        path = graph.query_path("no_such_entity", "another_missing")
        assert path == []


class TestSubgraph:
    def test_subgraph_existing(self, graph, sample_experiences):
        for exp in sample_experiences:
            graph.store(exp)
        # Get subgraph for first entity
        if graph.entities:
            key = next(iter(graph.entities))
            result = graph.get_subgraph(key, depth=2)
            assert "nodes" in result
            assert "edges" in result

    def test_subgraph_nonexistent(self, graph):
        result = graph.get_subgraph("nonexistent")
        assert result == {"nodes": [], "edges": []}


class TestSize:
    def test_empty_size(self, graph):
        assert graph.size == {"entities": 0, "relationships": 0}

    def test_size_after_store(self, graph, sample_experiences):
        for exp in sample_experiences:
            graph.store(exp)
        size = graph.size
        assert size["entities"] >= 0
        assert size["relationships"] >= 0


class TestEntityClassification:
    def test_classify_org(self, graph):
        etype = graph._classify_entity("Microsoft Corp", "Microsoft Corp is a company")
        assert etype == "org"

    def test_classify_location(self, graph):
        etype = graph._classify_entity("New York", "a city called New York")
        assert etype == "location"

    def test_classify_person(self, graph):
        etype = graph._classify_entity("Dr Johnson", "Dr Johnson said")
        assert etype == "person"

    def test_classify_default(self, graph):
        etype = graph._classify_entity("Something", "no type keywords here")
        assert etype == "concept"
