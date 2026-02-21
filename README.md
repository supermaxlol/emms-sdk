# EMMS v0.6.0 — Enhanced Memory Management System

**Persistent hierarchical memory and identity architecture for AI agents.**

EMMS gives AI agents human-like memory with 4-tier consolidation, consciousness-inspired identity, graph memory, spaced repetition, RAG context building, and empirically validated identity adoption — all without fine-tuning.

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

## Features (v0.6.0)

### New in v0.6.0 — Advanced intelligence: auto-importance, RAG, deduplication, SRS, scheduling, graph viz

| Feature | Description |
|---------|-------------|
| **ImportanceClassifier** | Auto-scores `Experience.importance` from 6 signals (entity density, novelty, emotional weight, length, keywords, structure); zero ML dependencies; `score_breakdown()` for explainability |
| **RAGContextBuilder** | Token-budget-aware context packer for RAG pipelines; `build(results, fmt="markdown/xml/json/plain")`; respects budget greedily by score |
| **SemanticDeduplicator** | Cosine (≥0.92) + lexical (≥0.85) near-duplicate detection; `resolve_groups()` archives weaker copies by importance×0.6 + access×0.4 |
| **MemoryScheduler** | Composable async job scheduler; 5 built-in jobs: consolidation, ttl_purge, deduplication, pattern_detection, srs_review; fully configurable intervals |
| **SpacedRepetitionSystem** | SM-2 algorithm; `srs_enroll()` / `srs_record_review(quality 0-5)` / `srs_due()`; persisted via sidecar `_srs.json` |
| **Graph Visualization** | `export_graph_dot()` (Graphviz DOT) + `export_graph_d3()` (D3.js JSON); `to_dot(highlight=["Alice"])` for emphasis |
| **MCP Server (22 tools)** | 7 new MCP tools: `emms_build_rag_context`, `emms_deduplicate`, `emms_srs_*`, `emms_export_graph_*` |
| **CLI (20 commands)** | `build-rag`, `deduplicate`, `srs-enroll`, `srs-review`, `srs-due`, `export-graph` |
| **583 tests** | 90 new tests across 9 test classes (from 493 in v0.5.2) |

### New in v0.5.2 — Memory namespaces, confidence, feedback & filtered retrieval

| Feature | Description |
|---------|-------------|
| **Namespace Scoping** | `Experience.namespace` — partition memories by project/repo; filter retrieval per namespace |
| **Confidence Scoring** | `Experience.confidence` (0–1) — uncertain memories score lower; filter with `min_confidence` |
| **Filtered Retrieval** | `retrieve_filtered()` — pre-filter by namespace, obs_type, domain, session, time range, confidence |
| **Memory Feedback** | `upvote(id)` / `downvote(id)` — user-driven strength adjustment; cited memories self-strengthen |
| **Markdown Export** | `export_markdown(path)` — human-readable memory dump grouped by domain with facts + metadata |
| **MCP Server** | `EMCPServer` — 13-tool Model Context Protocol adapter; drop into any MCP server framework |
| **CLI** | `emms` command — store, retrieve, compact, search-file, retrieve-filtered, upvote, downvote, export-md, stats |

### New in v0.5.1 — GitHub Copilot + LangMem patterns

| Feature | Description |
|---------|-------------|
| **ProceduralMemory** | 5th memory tier — accumulates behavioral rules; `get_prompt()` injects into system prompt |
| **Citation Validation** | `Experience.citations` + `validate_citations()` — cited memories are found and strengthened |
| **search_by_file()** | Find memories referencing specific file paths (files_read + files_modified) |
| **GraphMemory Persistence** | Graph state (entities + relationships) saved/loaded alongside hierarchical memory |
| **Patch Update Mode** | `update_mode="patch"` — update existing memory in-place; old version archived via `superseded_by` |
| **TTL-Aware Retrieval** | Expired (`is_expired`) and superseded (`is_superseded`) memories excluded from results |
| **Debounced Consolidation** | `SessionManager(consolidate_every=20)` — auto-consolidate after N stores |
| **ImportanceStrategy** | LangMem-inspired 6th retrieval strategy weighting `importance × 0.6 + strength × 0.4` |
| **strategy_scores** | `RetrievalResult.strategy_scores` + `.explanation` — per-strategy breakdown now exposed |

### Foundation (v0.5.0 — claude-mem patterns)

| Feature | Description |
|---------|-------------|
| **SessionManager** | Persistent session logs (JSONL), auto `session_id` injection, `generate_claude_md()` |
| **Endless Mode** | Biomimetic O(N²)→O(N) real-time compression — enables ~10-20× longer sessions |
| **ToolObserver** | Converts Claude Code `PostToolUse` hooks → `Experience` with inferred `obs_type` + concept tags |
| **ObsType + ConceptTag** | 6-type observation taxonomy + 7-tag epistemological classifier |
| **Progressive Disclosure** | 3-layer retrieval: `search_compact` (50-80 tok) → `get_timeline` → `get_full` — ~10x token savings |
| **Retrieval Presets** | `from_balanced()` (60/20/10/10) and `from_identity()` (6-strategy) factory methods |
| **ChromaSemanticStrategy** | ChromaDB-backed high-fidelity semantic retrieval with HNSW index |
| **BM25 Retrieval** | Replaces Jaccard overlap with BM25 (k1=1.5, b=0.75) — better ranking on natural language |
| **LLM Compression** | `LLMEnhancer.compress_memories()` — 5000-token batch → 500-token semantic episode |
| **LLM Classification** | `LLMEnhancer.classify_experience()` — infers `obs_type` + `concept_tags` via LLM |
| **MemoryAnalytics** | Health score, tier distribution, domain/concept coverage, session stats, endless stats |
| **JSONL Export/Import** | Human-readable, version-control-friendly memory snapshots |
| **SessionSummary** | Structured per-session narrative: request / investigated / learned / completed / next_steps |
| Hierarchical Memory | 4-tier with cognitive-science decay curves |
| Memory Persistence | Full state save/load (<2ms roundtrip); graph + procedural state also persisted |
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

## Quick Start — v0.6.0 Features

```python
from emms import EMMS, Experience

agent = EMMS()

# Store — importance auto-scored from 6 content signals
agent.store(Experience(
    content="Critical security vulnerability discovered in auth module.",
    domain="security",
    title="Auth CVE",
    facts=["OAuth bypass possible", "Patch pending"],
))

# RAG context building (token-budget-aware, 4 formats)
context = agent.build_rag_context(
    "security vulnerabilities",
    token_budget=4000,
    fmt="xml",    # or "markdown", "json", "plain"
)
print(context)  # <context><memory ...>...</memory></context>

# Near-duplicate detection and cleanup
result = agent.deduplicate()
print(f"Archived {result['memories_archived']} near-duplicate memories")

# Spaced Repetition System (SM-2)
result = agent.store(Experience(content="How to use async/await in Python", domain="coding"))
mem_id = result["memory_id"]
agent.srs_enroll(mem_id)
agent.srs_record_review(mem_id, quality=4)  # 0=blackout … 5=perfect
due = agent.srs_due()     # list of memory IDs due for review

# Graph visualization
dot = agent.export_graph_dot(max_nodes=50, highlight=["Anthropic"])
d3  = agent.export_graph_d3(max_nodes=100)  # D3.js force graph JSON

# Importance breakdown
from emms import Experience
exp = Experience(content="Critical auth bug with high severity.", domain="bugs")
print(agent.score_importance(exp))
# {'entity': 0.1, 'novelty': 0.5, 'emotional': 0.0, 'length': 0.04, 'keyword': 0.67, 'structure': 0.0, 'total': ...}

# Composable background scheduler
import asyncio
async def main():
    await agent.start_scheduler(consolidation_interval=60, dedup_interval=600)
    # ... do work ...
    await agent.stop_scheduler()

asyncio.run(main())
```

## Quick Start — v0.5.2 Features

```python
from emms import EMMS, Experience, ObsType, SessionManager, EMCPServer
from emms.memory.hierarchical import HierarchicalMemory

# ── Namespace-scoped, confidence-rated memory ──────────────────────────────
agent = EMMS()

# Store with rich metadata
agent.store(Experience(
    content="The Auth service uses OAuth2 tokens, not API keys.",
    domain="tech",
    namespace="project-auth",       # scope to this project
    confidence=0.95,                # very sure of this
    obs_type=ObsType.DISCOVERY,
    facts=["OAuth2 token expiry: 1h", "Refresh token TTL: 30d"],
    files_read=["src/auth/service.py"],
))

agent.store(Experience(
    content="OAuth2 tokens may also use PKCE for mobile clients.",
    domain="tech",
    namespace="project-auth",
    confidence=0.7,                 # less certain
))

# ── Filtered retrieval ─────────────────────────────────────────────────────
results = agent.retrieve_filtered(
    "OAuth token expiry",
    namespace="project-auth",      # only this project
    min_confidence=0.8,            # skip uncertain memories
    max_results=5,
)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.experience.content}")

# ── Feedback loop ──────────────────────────────────────────────────────────
agent.upvote(results[0].memory.id)    # this was helpful
# agent.downvote(results[1].memory.id)  # this was wrong

# ── Patch update mode (conflict archival) ──────────────────────────────────
agent.store(Experience(
    content="OAuth2 tokens now expire in 2h after the March 2026 update.",
    domain="tech",
    namespace="project-auth",
    title="auth_token_expiry",
    patch_key="auth_token_expiry",
    update_mode="patch",            # supersedes old version, keeps audit trail
))

# ── Procedural memory (evolving system prompt) ─────────────────────────────
agent.add_procedure("Always check token expiry before making API calls.", domain="tech")
agent.add_procedure("Prefer OAuth2 PKCE for mobile clients.", domain="tech", importance=0.8)
print(agent.get_system_prompt_rules(domain="tech"))
# → ## Behavioral Rules
# → - Prefer OAuth2 PKCE for mobile clients.
# → - Always check token expiry before making API calls.

# ── Citation validation ────────────────────────────────────────────────────
new_exp = Experience(
    content="Token handling confirmed per Auth team docs",
    citations=[results[0].memory.id],   # cites the memory we upvoted
)
validation = agent.validate_citations(new_exp)   # strengthens cited memories

# ── Find memories by file ──────────────────────────────────────────────────
file_memories = agent.search_by_file("auth/service.py")

# ── Markdown export ────────────────────────────────────────────────────────
agent.export_markdown("~/.emms/memories.md", namespace="project-auth")

# ── MCP server (13 tools for Claude Desktop / MCP clients) ────────────────
server = EMCPServer(agent)
print([t["name"] for t in server.tool_definitions])

# ── Save / load everything (memory + graph + procedural + consciousness) ───
agent.save("~/.emms/state.json")
agent.load("~/.emms/state.json")
```

### CLI

```bash
# Store a memory
emms -m ~/.emms/state.json store "OAuth2 tokens expire in 2h" --domain tech --importance 0.9

# Retrieve with filters
emms -m ~/.emms/state.json retrieve-filtered "token expiry" --namespace project-auth --min-confidence 0.8

# Find memories that touched a file
emms -m ~/.emms/state.json search-file "src/auth/service.py"

# Feedback
emms -m ~/.emms/state.json upvote mem_abc123
emms -m ~/.emms/state.json downvote mem_xyz999

# Export as Markdown
emms -m ~/.emms/state.json export-md ~/.emms/memories.md

# Show behavioral rules
emms -m ~/.emms/state.json procedures --domain tech

# System stats
emms -m ~/.emms/state.json stats
```

## Module Structure

```
emms-sdk/src/emms/
├── emms.py                    # EMMS orchestrator
├── core/
│   ├── models.py              # Experience, MemoryItem, MemoryConfig,
│   │                          #   CompactResult, SessionSummary, ObsType, ConceptTag
│   ├── embeddings.py          # HashEmbedder, SentenceTransformerEmbedder
│   └── events.py              # EventBus pub/sub
├── memory/
│   ├── hierarchical.py        # 4-tier memory + VectorIndex + BM25 + Endless Mode
│   │                          #   + retrieve_filtered + upvote/downvote + export_markdown
│   │                          #   + search_by_file + patch mode + TTL filtering
│   ├── compression.py         # MemoryCompressor + PatternDetector
│   ├── graph.py               # GraphMemory entity-relationship + save/load_state
│   └── procedural.py          # ProceduralMemory — 5th tier behavioral rules
├── retrieval/
│   └── strategies.py          # 6-strategy ensemble + ImportanceStrategy
│                              #   + from_balanced (60/20/10/10) / from_identity (6-strategy)
│                              #   + search_compact + get_full + ChromaSemanticStrategy
│                              #   + strategy_scores + explanation on RetrievalResult
├── sessions/
│   └── manager.py             # SessionManager: lifecycle, JSONL logs, generate_claude_md
│                              #   + debounced consolidation + generate_context_injection
├── hooks/
│   └── tool_observer.py       # ToolObserver: PostToolUse → Experience converter
├── analytics/
│   └── memory_analytics.py    # MemoryAnalytics: health, tiers, domains, concepts
├── identity/
│   ├── ego.py                 # PersistentIdentity
│   └── consciousness.py       # Narrator, MeaningMaker, Temporal, Ego,
│                              #   MetaCognitiveMonitor + A-MEM associative links
├── episodes/
│   └── boundary.py            # Spectral, conductance, multi-algorithm
├── crossmodal/
│   └── binding.py             # 6-modality cross-modal binding
├── context/
│   └── token_manager.py       # Token context management
├── integrations/
│   └── llm.py                 # Claude/GPT/Ollama + classify_experience + compress_memories
├── pipeline/
│   └── realtime.py            # Async RSS/REST data pipeline
├── prompts/
│   └── identity.py            # Empirically validated prompt templates
├── storage/
│   ├── base.py                # InMemoryStore, JSONFileStore
│   └── chroma.py              # ChromaDB vector store
└── adapters/
    ├── agent.py               # AgentMemory adapter
    └── mcp_server.py          # EMCPServer — 13-tool MCP adapter
```

## Testing

```bash
cd emms-sdk
python -m pytest tests/ -v
# 493 passed, 2 skipped, 0 failures
#   v0.5.2: +27 tests (namespace/confidence, filtered retrieval, feedback, markdown)
#   v0.5.1: +48 tests (search_by_file, patch mode, TTL, graph persistence, procedural,
#                       citations, debounce, MCP server, strategy_scores, ImportanceStrategy)
#   v0.5.0: +55 tests (SessionManager, Endless Mode, ToolObserver, analytics, BM25)
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
