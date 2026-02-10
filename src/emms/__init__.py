"""EMMS — Enhanced Memory Management System for AI Agents."""

from emms.core.models import Experience, MemoryItem, MemoryConfig
from emms.core.embeddings import HashEmbedder, cosine_similarity
from emms.memory.hierarchical import HierarchicalMemory
from emms.context.token_manager import TokenContextManager
from emms.emms import EMMS

__version__ = "0.2.0"
__all__ = [
    "EMMS",
    "Experience",
    "MemoryItem",
    "MemoryConfig",
    "HierarchicalMemory",
    "TokenContextManager",
    "HashEmbedder",
    "cosine_similarity",
]
