"""Graph memory layer — entity-relationship extraction and storage.

Extracts entities (people, orgs, concepts, locations, events) and
relationships from experience text, building a knowledge graph that
enables entity-centric retrieval alongside the hierarchical memory.

Inspired by Mem0's graph memory but integrated with EMMS's tier system.
Uses regex-based NER (zero ML dependency) + co-occurrence for relationships.
NetworkX optional for path queries.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from emms.core.models import Experience

logger = logging.getLogger(__name__)

# Optional dependency
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity extracted from experiences."""
    name: str
    entity_type: str = "concept"  # person, org, concept, event, location
    mentions: int = 1
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    importance: float = 0.5
    source_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """A relationship between two entities."""
    source: str
    target: str
    relation_type: str = "related_to"
    strength: float = 0.5
    evidence_ids: list[str] = Field(default_factory=list)
    first_seen: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

# Capitalised multi-word names (e.g. "John Smith", "Apple Inc", "New York")
_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|for|in|on|at|to|de|van|von|al|el))?'
    r'(?:\s+[A-Z][a-z]+)+)\b'
)

# Single capitalised word (likely a proper noun) — matched inline, not via compiled pattern
_PROPER_NOUN = re.compile(r'\b([A-Z][a-z]{2,})\b')

# Common relationship indicators
_RELATION_PATTERNS = [
    (re.compile(r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)\s+(\w+)', re.I), "is_a"),
    (re.compile(r'(\w+)\s+(?:works?|worked)\s+(?:at|for|with)\s+(\w+)', re.I), "works_at"),
    (re.compile(r'(\w+)\s+(?:causes?|caused)\s+(\w+)', re.I), "causes"),
    (re.compile(r'(\w+)\s+(?:affects?|affected)\s+(\w+)', re.I), "affects"),
    (re.compile(r'(\w+)\s+(?:leads?\s+to|led\s+to)\s+(\w+)', re.I), "leads_to"),
    (re.compile(r'(\w+)\s+(?:and|with|or)\s+(\w+)', re.I), "associated_with"),
]

# Type classification keywords
_TYPE_KEYWORDS: dict[str, set[str]] = {
    "person": {"mr", "mrs", "dr", "prof", "president", "ceo", "founder", "author", "researcher"},
    "org": {"inc", "corp", "ltd", "company", "university", "institute", "foundation", "group", "team"},
    "location": {"city", "country", "state", "region", "street", "river", "mountain", "island", "ocean"},
    "event": {"war", "election", "conference", "summit", "crisis", "pandemic", "revolution", "launch"},
}


# ---------------------------------------------------------------------------
# GraphMemory
# ---------------------------------------------------------------------------

class GraphMemory:
    """Entity-relationship graph layer on top of hierarchical memory.

    Builds a knowledge graph from experience text, enabling queries like:
    - "What entities are related to X?"
    - "What is the shortest path between X and Y?"
    - "What is the local subgraph around X?"

    Zero ML dependencies — uses regex-based NER and co-occurrence.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}          # name_lower → Entity
        self.relationships: list[Relationship] = []
        self._adj: dict[str, set[str]] = defaultdict(set)  # adjacency list
        self._rel_index: dict[tuple[str, str], Relationship] = {}  # (src, tgt) → rel

    # ── Store ──────────────────────────────────────────────────────────

    def store(self, experience: Experience) -> dict[str, Any]:
        """Extract entities and relationships from an experience."""
        entities = self.extract_entities(experience.content)
        relationships = self.extract_relationships(
            experience.content, entities
        )

        # Register entities
        for ent in entities:
            key = ent.name.lower()
            if key in self.entities:
                existing = self.entities[key]
                existing.mentions += 1
                existing.last_seen = time.time()
                existing.importance = min(1.0, existing.importance + 0.05)
                if experience.id not in existing.source_ids:
                    existing.source_ids.append(experience.id)
            else:
                ent.source_ids = [experience.id]
                self.entities[key] = ent

        # Register relationships
        for rel in relationships:
            pair = (rel.source.lower(), rel.target.lower())
            if pair in self._rel_index:
                existing_rel = self._rel_index[pair]
                existing_rel.strength = min(1.0, existing_rel.strength + 0.1)
                if experience.id not in existing_rel.evidence_ids:
                    existing_rel.evidence_ids.append(experience.id)
            else:
                rel.evidence_ids = [experience.id]
                self.relationships.append(rel)
                self._rel_index[pair] = rel
                self._adj[rel.source.lower()].add(rel.target.lower())
                self._adj[rel.target.lower()].add(rel.source.lower())

        # Update experience entity/relationship fields
        experience.entities = [e.name for e in entities]
        experience.relationships = [
            {"source": r.source, "target": r.target, "type": r.relation_type}
            for r in relationships
        ]

        return {
            "entities_found": len(entities),
            "relationships_found": len(relationships),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
        }

    # ── Entity extraction ──────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities from text using pattern matching."""
        entities: dict[str, Entity] = {}
        now = time.time()

        # Multi-word names
        for match in _NAME_PATTERN.finditer(text):
            name = match.group(0).strip()
            if len(name) < 3:
                continue
            key = name.lower()
            if key not in entities:
                etype = self._classify_entity(name, text)
                entities[key] = Entity(
                    name=name, entity_type=etype,
                    first_seen=now, last_seen=now,
                )
            else:
                entities[key].mentions += 1

        # Capitalised words (not at sentence start, not common words)
        _common = {
            "the", "this", "that", "with", "from", "have", "been",
            "will", "would", "could", "should", "about", "their",
            "there", "which", "other", "into", "more", "some",
            "also", "just", "very", "most", "such", "only", "then",
            "when", "what", "they", "them", "than", "each",
        }
        words = text.split()
        for i, word in enumerate(words):
            if (
                word[0:1].isupper()
                and len(word) >= 3
                and word.lower() not in _common
                and i > 0  # skip sentence-initial
                and not words[i - 1].endswith((".", "!", "?"))
            ):
                clean = re.sub(r'[^\w]', '', word)
                if len(clean) >= 3:
                    key = clean.lower()
                    if key not in entities:
                        etype = self._classify_entity(clean, text)
                        entities[key] = Entity(
                            name=clean, entity_type=etype,
                            first_seen=now, last_seen=now,
                        )
                    else:
                        entities[key].mentions += 1

        # Domain-specific concept extraction (lowercase key terms)
        concepts = self._extract_concepts(text)
        for concept in concepts:
            key = concept.lower()
            if key not in entities:
                entities[key] = Entity(
                    name=concept, entity_type="concept",
                    first_seen=now, last_seen=now,
                )

        return list(entities.values())

    def extract_relationships(
        self, text: str, entities: list[Entity]
    ) -> list[Relationship]:
        """Extract relationships via pattern matching and co-occurrence."""
        relationships: list[Relationship] = []
        now = time.time()

        # Pattern-based extraction
        for pattern, rel_type in _RELATION_PATTERNS:
            for match in pattern.finditer(text):
                src, tgt = match.group(1), match.group(2)
                if len(src) >= 3 and len(tgt) >= 3:
                    relationships.append(Relationship(
                        source=src, target=tgt,
                        relation_type=rel_type,
                        strength=0.6, first_seen=now,
                    ))

        # Co-occurrence: entities mentioned in the same sentence
        sentences = re.split(r'[.!?]+', text)
        entity_names = {e.name.lower(): e.name for e in entities}

        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_in_sent = [
                name for key, name in entity_names.items()
                if key in sentence_lower
            ]
            # Create co-occurrence relationships for pairs
            for i in range(len(found_in_sent)):
                for j in range(i + 1, len(found_in_sent)):
                    pair = (found_in_sent[i].lower(), found_in_sent[j].lower())
                    if pair not in self._rel_index:
                        relationships.append(Relationship(
                            source=found_in_sent[i],
                            target=found_in_sent[j],
                            relation_type="co_occurs_with",
                            strength=0.4, first_seen=now,
                        ))

        return relationships

    # ── Query ──────────────────────────────────────────────────────────

    def query(self, entity_name: str) -> dict[str, Any]:
        """Query the graph for an entity and its neighbors."""
        key = entity_name.lower()
        entity = self.entities.get(key)
        if entity is None:
            return {"found": False, "entity": None, "neighbors": [], "relationships": []}

        neighbors = list(self._adj.get(key, set()))
        rels = [
            r for r in self.relationships
            if r.source.lower() == key or r.target.lower() == key
        ]

        return {
            "found": True,
            "entity": entity.model_dump(),
            "neighbors": neighbors,
            "relationships": [r.model_dump() for r in rels],
        }

    def query_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two entities (requires networkx)."""
        if not _HAS_NX:
            return self._bfs_path(source.lower(), target.lower())

        G = nx.Graph()
        for key in self.entities:
            G.add_node(key)
        for rel in self.relationships:
            G.add_edge(rel.source.lower(), rel.target.lower(), weight=1.0 - rel.strength)

        try:
            path = nx.shortest_path(G, source.lower(), target.lower())
            return [self.entities[n].name if n in self.entities else n for n in path]
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
        """Get the local subgraph around an entity up to *depth* hops."""
        key = entity.lower()
        if key not in self.entities:
            return {"nodes": [], "edges": []}

        visited: set[str] = set()
        frontier = {key}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in visited:
                    visited.add(node)
                    next_frontier.update(self._adj.get(node, set()))
            frontier = next_frontier - visited
        visited.update(frontier)

        nodes = [
            self.entities[n].model_dump()
            for n in visited if n in self.entities
        ]
        edges = [
            r.model_dump() for r in self.relationships
            if r.source.lower() in visited and r.target.lower() in visited
        ]

        return {"nodes": nodes, "edges": edges}

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def size(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "relationships": len(self.relationships),
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _classify_entity(self, name: str, context: str) -> str:
        """Classify entity type from name and surrounding context."""
        name_lower = name.lower()
        context_lower = context.lower()

        for etype, keywords in _TYPE_KEYWORDS.items():
            # Check if a type keyword appears near the entity name
            for kw in keywords:
                if kw in name_lower or kw in context_lower:
                    return etype
        return "concept"

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract domain-specific concepts (compound terms, technical words)."""
        concepts: list[str] = []

        # Compound terms with hyphens or underscores
        for match in re.finditer(r'\b(\w+-\w+(?:-\w+)?)\b', text):
            term = match.group(1)
            if len(term) >= 5:
                concepts.append(term)

        # Technical terms (words with mixed case in the middle, e.g. "JavaScript")
        for match in re.finditer(r'\b([a-z]+[A-Z]\w+)\b', text):
            concepts.append(match.group(1))

        return concepts[:10]  # cap to avoid noise

    def _bfs_path(self, source: str, target: str) -> list[str]:
        """Simple BFS shortest path without networkx."""
        if source not in self._adj or target not in self._adj:
            return []

        visited: set[str] = {source}
        queue: list[tuple[str, list[str]]] = [(source, [source])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self._adj.get(current, set()):
                if neighbor == target:
                    result = path + [neighbor]
                    return [
                        self.entities[n].name if n in self.entities else n
                        for n in result
                    ]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []
