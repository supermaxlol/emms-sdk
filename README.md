# EMMS v0.4.0 — Enhanced Memory Management System

**Persistent hierarchical memory and identity architecture for AI agents.**

EMMS gives AI agents human-like memory with 4-tier consolidation, consciousness-inspired identity, graph memory, and empirically validated identity adoption — all without fine-tuning.

> **New here?** Start with the [Project README](../README.md) for the big picture, then read the [Research Guide](../RESEARCH_GUIDE.md) for the full story of this research.
>
> **Want to run the experiments?** See [HOW_TO_REPRODUCE.md](HOW_TO_REPRODUCE.md) — step-by-step instructions, no prior experience needed.

---

## Quick Start

```python
from emms import EMMS, Experience

agent = EMMS()
agent.store(Experience(content="The market rose 2% today", domain="finance"))
agent.store(Experience(content="Quantum computing breakthrough at MIT", domain="science"))

results = agent.retrieve("market trends")
print(results[0].memory.experience.content)

# Identity
print(agent.get_narrative("MyAgent"))
print(agent.get_consciousness_state())

# Persistence
agent.save(memory_path="state.json")
# Later...
agent.load(memory_path="state.json")
```

## Install

```bash
cd emms-sdk
pip install -e ".[all,dev]"
```

## Architecture

```
Experience → [Working Memory] → [Short-Term] → [Long-Term] → [Semantic]
                  ↓                   ↓              ↓             ↓
            Consciousness:    ContinuousNarrator  MeaningMaker  TemporalIntegrator
            GraphMemory:      Entity Extraction   Relationships  Subgraph Queries
            Events:           memory.stored       consolidated   compressed
```

### 4-Tier Hierarchical Memory
Inspired by Atkinson-Shiffrin with Miller's Law (7±2) for working memory, exponential decay, and importance-weighted consolidation:

- **Working Memory** — Immediate buffer (deque, capacity-limited)
- **Short-Term Memory** — Decaying store with access-count promotion
- **Long-Term Memory** — Stable storage for important/frequently-accessed items
- **Semantic Memory** — Highest tier for deeply consolidated knowledge

### Consciousness Modules
Functional identity modules (not consciousness claims):

- **ContinuousNarrator** — Builds evolving self-narrative with theme tracking, trait inference, autobiographical events
- **MeaningMaker** — Assigns personal significance via concept familiarity, emotional intensity, learning potential
- **TemporalIntegrator** — Monitors identity continuity, detects milestones, takes identity snapshots
- **EgoBoundaryTracker** — Tracks self/other distinction through pronoun analysis and boundary strength

### Additional Systems
- **GraphMemory** — Entity-relationship extraction and graph queries (17K+ exp/s)
- **EventBus** — Pub/sub for inter-component communication
- **VectorIndex** — Numpy-based batch cosine similarity (replaces O(n) scan)
- **5-Strategy Retrieval** — Semantic, temporal, emotional, graph, domain ensemble
- **LLM Integration** — Claude, GPT, Ollama providers with identity-aware prompts
- **Real-Time Pipeline** — Async RSS/REST data ingestion

## Features (v0.4.0)

| Feature | Description |
|---------|-------------|
| Hierarchical Memory | 4-tier with cognitive-science decay curves |
| Memory Persistence | Full state save/load (<2ms roundtrip) |
| Consciousness State Persistence | Narrator, meaning, temporal, ego all saved between sessions |
| Graph Memory | Regex-based NER, relationship extraction, path queries |
| Episode Detection | Spectral clustering, conductance optimization, multi-algorithm |
| Cross-Modal Binding | 6 modalities (text, visual, audio, temporal, spatial, emotional) |
| Memory Compression | Pattern-based with deduplication and fidelity tracking |
| Token Context Manager | 3-tier eviction with intelligent importance scoring |
| Identity Prompts | Empirically validated templates for LLM identity adoption |
| Event System | Pub/sub with memory.stored, consolidated, compressed events |
| Async API | Full async support (astore, aretrieve, aconsolidate) |
| Agent Adapter | Drop-in memory backend for LLM agent frameworks |

## Benchmarks

```
System              |  Store/s |  Ret.avg(ms) |  P@10
---------------------|----------|--------------|-------
EMMS (lexical)      |  2140.8  |    0.049     |  1.000
EMMS (HashEmbedder) |  1481.9  |    0.833     |  0.487
EMMS (Chroma+embed) |   428.6  |    1.363     |  0.825

Graph memory:        17,305 exp/s entity extraction
Persistence save:    0.89ms (55 items, 33.8KB)
Persistence load:    1.79ms (55 items restored)
```

## Identity Adoption Research

### The Goldilocks Effect

Tested 7 models across 90+ trials. Identity adoption peaks at intermediate RLHF training:

```
Guardrail Level    Model                  Net Adoption
-------------------------------------------------------
NONE               Dolphin-Llama3 8b       50%
Light              Ollama Gemma3n          56%
Balanced           Claude Sonnet 4.5       72%  <-- Sweet spot
Strong             Claude Opus 4.6         61%
Strictest          Claude Haiku 4.5       -11%
```

Key finding: Removing ALL guardrails does NOT improve identity adoption. RLHF instruction-following capability is essential for the model to adopt EMMS identities.

### Temporal Persistence

With proper state persistence, identity is 100% stable and strengthens over sessions:

| Session | Adoption | Ego Strength | Memory Refs |
|---------|----------|-------------|-------------|
| S1      | 100%     | 0.80        | 7           |
| S2      | 100%     | 0.85        | 10          |
| S3      | 100%     | 0.87        | 13          |

## Module Structure

```
emms-sdk/src/emms/
├── emms.py                    # EMMS orchestrator
├── core/
│   ├── models.py              # Experience, MemoryItem, MemoryConfig
│   ├── embeddings.py          # HashEmbedder, SentenceTransformerEmbedder
│   └── events.py              # EventBus pub/sub
├── memory/
│   ├── hierarchical.py        # 4-tier memory + VectorIndex + persistence
│   ├── compression.py         # MemoryCompressor + PatternDetector
│   └── graph.py               # GraphMemory entity-relationship
├── retrieval/
│   └── strategies.py          # 5-strategy ensemble retrieval
├── identity/
│   ├── ego.py                 # PersistentIdentity
│   └── consciousness.py       # Narrator, MeaningMaker, Temporal, Ego
├── episodes/
│   └── boundary.py            # Spectral, conductance, multi-algorithm
├── crossmodal/
│   └── binding.py             # 6-modality cross-modal binding
├── context/
│   └── token_manager.py       # Token context management
├── integrations/
│   └── llm.py                 # Claude/GPT/Ollama LLM providers
├── pipeline/
│   └── realtime.py            # Async RSS/REST data pipeline
├── prompts/
│   └── identity.py            # Empirically validated prompt templates
├── storage/
│   ├── base.py                # InMemoryStore, JSONFileStore
│   └── chroma.py              # ChromaDB vector store
└── adapters/
    └── agent.py               # AgentMemory adapter
```

## Testing

```bash
cd emms-sdk
python -m pytest tests/ -v
# 333 passed, 2 skipped, 0 failures
```

## LLM Integration

```python
from emms import EMMS, Experience
from emms.integrations.llm import ClaudeProvider, LLMEnhancer

emms = EMMS()
# Store experiences...

provider = ClaudeProvider(api_key="...", model="claude-sonnet-4-5-20250929")
enhancer = LLMEnhancer(provider, emms=emms, agent_name="MyAgent")

# Identity-aware conversation
response = await enhancer.ask("What do you remember about our project?")
```

## Research Context

EMMS predates and implements features described in recent memory architecture papers:
- **HiMem** (NeurIPS) — Hierarchical LTM for agents
- **TiMem** — Temporal-hierarchical consolidation
- **MAGMA** — Multi-graph agentic memory
- **Mem0/Zep/Letta** — Competitor memory systems (EMMS combines graph + hierarchy + identity)

## Author

**Shehzad Ahmed** — Finance Major, CSE Minor (Big Data & HPC), Independent University Bangladesh (IUB)

## License

Research use. See repository for details.
