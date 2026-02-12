"""EMMS — Enhanced Memory Management System for AI Agents.

v0.4.0: The Ultimate System
- EventBus for inter-component communication
- GraphMemory for entity-relationship extraction
- Multi-strategy ensemble retrieval (5 strategies)
- Memory persistence (save/load full state)
- VectorIndex for fast batch cosine similarity
- Enhanced consciousness modules (traits, autobiographical, milestones)
- Advanced episode detection (spectral, conductance, multi-algorithm)
- Pattern detection in memory compression
- LLM integration layer (Claude, GPT, Ollama)
- Real-time data pipeline
- Background consolidation
"""

from emms.core.models import Experience, MemoryItem, MemoryConfig
from emms.core.embeddings import HashEmbedder, cosine_similarity
from emms.core.events import EventBus
from emms.memory.hierarchical import HierarchicalMemory, VectorIndex
from emms.memory.compression import MemoryCompressor, CompressedMemory, PatternDetector
from emms.memory.graph import GraphMemory, Entity, Relationship
from emms.context.token_manager import TokenContextManager
from emms.identity.consciousness import (
    ContinuousNarrator,
    MeaningMaker,
    TemporalIntegrator,
    EgoBoundaryTracker,
)
from emms.retrieval.strategies import (
    EnsembleRetriever,
    SemanticStrategy,
    TemporalStrategy,
    EmotionalStrategy,
    GraphStrategy,
    DomainStrategy,
)
from emms.emms import EMMS
from emms.prompts.identity import IdentityPromptBuilder, PROVIDER_RECOMMENDATIONS

__version__ = "0.4.0"
__all__ = [
    # Core
    "EMMS",
    "Experience",
    "MemoryItem",
    "MemoryConfig",
    "EventBus",
    # Memory
    "HierarchicalMemory",
    "VectorIndex",
    "GraphMemory",
    "Entity",
    "Relationship",
    # Compression & patterns
    "MemoryCompressor",
    "CompressedMemory",
    "PatternDetector",
    # Retrieval
    "EnsembleRetriever",
    "SemanticStrategy",
    "TemporalStrategy",
    "EmotionalStrategy",
    "GraphStrategy",
    "DomainStrategy",
    # Embeddings
    "HashEmbedder",
    "cosine_similarity",
    # Context
    "TokenContextManager",
    # Consciousness
    "ContinuousNarrator",
    "MeaningMaker",
    "TemporalIntegrator",
    "EgoBoundaryTracker",
    # Prompts (identity adoption)
    "IdentityPromptBuilder",
    "PROVIDER_RECOMMENDATIONS",
]
