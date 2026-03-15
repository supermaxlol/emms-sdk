# EMMS — Enhanced Memory Management System

**Give any AI agent persistent, human-like memory with 129 cognitive tools.**

[![PyPI](https://img.shields.io/pypi/v/emms-mcp)](https://pypi.org/project/emms-mcp/)
[![npm](https://img.shields.io/npm/v/emms-mcp)](https://www.npmjs.com/package/emms-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

EMMS is an MCP server that gives AI agents cognitive memory — not just storage, but retrieval strategies, emotional recall, knowledge graphs, metacognition, goal tracking, and more. It models how human memory actually works: consolidation, decay, reconsolidation, spreading activation, and schema extraction.

**60,000 lines of Python. 129 MCP tools. Zero external services required.**

---

## Install

```bash
pip install emms-mcp
```

Or via npm (wraps the Python package):

```bash
npx -y emms-mcp
```

Or via uv:

```bash
uvx emms-mcp
```

## Configure

### Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "emms": {
    "command": "uvx",
    "args": ["emms-mcp"]
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "emms": {
      "command": "uvx",
      "args": ["emms-mcp"]
    }
  }
}
```

### Cursor / Windsurf / Any MCP Client

Same config — EMMS uses stdio transport, compatible with all MCP clients.

## Architecture

```
Experience ──▶ [Working Memory] ──▶ [Short-Term] ──▶ [Long-Term] ──▶ [Archive]
                    │                     │                │              │
                    ├─ Decay curve        ├─ Consolidation ├─ Schemas     ├─ Compressed
                    ├─ Emotional tagging  ├─ Deduplication ├─ Norms       ├─ Wisdom
                    └─ Goal relevance     └─ Reconsolidate └─ Communities └─ Forgotten

Retrieval ◀── Hybrid (BM25 + Cosine + RRF) ◀── Spreading Activation ◀── Association Graph
          ◀── Affective (emotional proximity)
          ◀── Spotlight (goal-aware)
          ◀── Adaptive (Thompson Sampling learns best strategy)
          ◀── Plan (query decomposition + cross-boost)
```

## 129 Tools by Category

### Storage & Persistence
| Tool | What it does |
|------|-------------|
| `emms_store` | Store an experience into memory |
| `emms_store_batch` | Store multiple experiences at once |
| `emms_load` | Load state from disk |
| `emms_save` | Persist state to disk |
| `emms_forget` | Remove a memory |

### Retrieval (10 strategies)
| Tool | What it does |
|------|-------------|
| `emms_retrieve` | Semantic embedding search |
| `emms_hybrid_retrieve` | BM25 + embedding fused via RRF |
| `emms_adaptive_retrieve` | Thompson Sampling picks best strategy |
| `emms_plan_retrieve` | Decomposes query into sub-queries, cross-boosts |
| `emms_affective_retrieve` | Find memories by emotional proximity |
| `emms_spotlight_retrieve` | Goal-aware contextual retrieval |
| `emms_associative_retrieve` | Spreading activation on association graph |
| `emms_retrieve_filtered` | Filter by namespace, domain, time, confidence |
| `emms_multihop_query` | Multi-hop reasoning across memories |
| `emms_search_compact` | Compressed search results (token-efficient) |

### Knowledge Graph
| Tool | What it does |
|------|-------------|
| `emms_build_association_graph` | Auto-detect semantic, temporal, affective edges |
| `emms_graph_communities` | Label propagation community detection |
| `emms_spreading_activation` | Activate from seeds, find connected memories |
| `emms_build_causal_map` | Extract cause-effect relationships |
| `emms_build_hierarchy` | Build taxonomic hierarchies |
| `emms_build_timeline` | Temporal ordering of memories |
| `emms_export_graph_dot` | Export as Graphviz DOT |
| `emms_export_graph_d3` | Export as D3.js force graph JSON |

### Reflection & Insight
| Tool | What it does |
|------|-------------|
| `emms_reflect` | Generate structured self-reflection |
| `emms_dream` | Offline consolidation — find hidden connections |
| `emms_synthesize_wisdom` | Distill cross-domain wisdom |
| `emms_abstract_principles` | Extract generalizable principles |
| `emms_find_analogies` | Cross-domain analogy discovery |
| `emms_find_contradictions` | Detect conflicting memories |
| `emms_discover_insights` | Surface non-obvious patterns |
| `emms_blend_concepts` | Creative concept blending |

### Emotions & Affect
| Tool | What it does |
|------|-------------|
| `emms_current_emotion` | Current emotional state |
| `emms_regulate_emotions` | Apply regulation strategies |
| `emms_mood_trend` | Emotional trajectory over time |
| `emms_emotional_landscape` | Full emotional profile |
| `emms_trace_mood` | Trace what caused mood shifts |
| `emms_measure_self_compassion` | Self-compassion assessment |

### Goals & Intentions
| Tool | What it does |
|------|-------------|
| `emms_push_goal` | Add a prioritized goal |
| `emms_complete_goal` | Mark goal as achieved |
| `emms_active_goals` | List current goals |
| `emms_intend` | Register a prospective intention |
| `emms_check_intentions` | Check pending intentions |
| `emms_exploration_goals` | Curiosity-driven exploration goals |

### Metacognition & Identity
| Tool | What it does |
|------|-------------|
| `emms_metacognition_report` | Assess own cognitive processes |
| `emms_consciousness_metrics` | Functional consciousness indicators |
| `emms_presence_metrics` | Engagement and presence tracking |
| `emms_detect_biases` | Identify cognitive biases in memory |
| `emms_confabulation_audit` | Check for false memory construction |
| `emms_update_self_model` | Update internal self-representation |
| `emms_agent_model` | Model of another agent's perspective |

### Memory Management
| Tool | What it does |
|------|-------------|
| `emms_apply_decay` | Ebbinghaus forgetting curve |
| `emms_deduplicate` | Find and archive near-duplicates |
| `emms_reconsolidate` | Strengthen/weaken recalled memories |
| `emms_llm_consolidate` | LLM-powered memory synthesis |
| `emms_cluster_memories` | K-means/TF-IDF clustering |
| `emms_enforce_budget` | Keep memory within size limits |
| `emms_anneal` | Temporal annealing (gradual stabilization) |

### Prediction & Futures
| Tool | What it does |
|------|-------------|
| `emms_predict` | Register a prediction |
| `emms_pending_predictions` | List unresolved predictions |
| `emms_plausible_futures` | Generate future scenarios |
| `emms_project_future` | Project trends forward |
| `emms_generate_counterfactuals` | What-if analysis |

### Spaced Repetition
| Tool | What it does |
|------|-------------|
| `emms_srs_enroll` | Enroll memory in SM-2 schedule |
| `emms_srs_due` | Get memories due for review |
| `emms_srs_record_review` | Record review quality (0-5) |

### Multi-Agent & Federation
| Tool | What it does |
|------|-------------|
| `emms_merge_from` | Merge memories from another namespace |
| `emms_list_namespaces` | List all memory namespaces |
| `emms_migrate_namespace` | Move memories between namespaces |
| `emms_compute_trust` | Trust scoring for other agents |
| `emms_build_perspectives` | Model multiple agent perspectives |

### And 50+ More
Norms extraction, schema detection, narrative weaving, knowledge gaps, values mapping, moral reasoning, curiosity reports, resilience tracking, source auditing, causal tracing, adversity analysis, episode management, procedures, skills, and more.

## Optional Dependencies

The base install (`pip install emms-mcp`) is lightweight (numpy + pydantic + mcp). For enhanced capabilities:

```bash
pip install emms-mcp[embeddings]  # Semantic search (sentence-transformers)
pip install emms-mcp[graph]       # Knowledge graphs (networkx)
pip install emms-mcp[ml]          # Clustering (scikit-learn)
pip install emms-mcp[all]         # Everything
```

Without `[embeddings]`, retrieval uses BM25 lexical matching — still effective, just not semantic.

## How It Works

1. **Store** — Experiences enter working memory with automatic emotional tagging, importance scoring, and embedding
2. **Consolidate** — Over time, memories promote through tiers based on access frequency, importance, and decay curves
3. **Retrieve** — Multiple strategies compete: semantic, lexical, emotional, graph-based, goal-aware. Thompson Sampling learns which works best for your use case
4. **Reflect** — Dream consolidation finds hidden connections. Schema extraction identifies patterns. Wisdom synthesis distills cross-domain insights
5. **Persist** — State saves to a single JSON file. Load it next session and everything is exactly where you left it

## State File

By default, EMMS stores state at `~/.emms/emms_state.json`. Override with:

```bash
emms-mcp --state-file /path/to/state.json
```

Or set the environment variable:

```bash
export EMMS_STATE_FILE=/path/to/state.json
```

## Python API

EMMS also works as a Python library:

```python
from emms import EMMS, Experience

agent = EMMS()
agent.store(Experience(content="Bitcoin broke $95K resistance", domain="crypto"))
results = agent.retrieve("market breakout")
print(results[0].memory.experience.content)

agent.save("state.json")
```

## Benchmarks

```
Operation           |  Speed        |  Notes
--------------------|---------------|----------------------------------
Store               |  2,140/sec    |  Lexical mode
Retrieve            |  0.049ms avg  |  BM25 retrieval
Graph extraction    |  17,305/sec   |  Entity + relationship extraction
Save (55 items)     |  0.89ms       |  33.8KB state file
Load (55 items)     |  1.79ms       |  Full state restore
```

## Compared to Other Memory MCP Servers

| Feature | EMMS (129 tools) | Typical memory MCP (5-30 tools) |
|---------|-----------------|-------------------------------|
| Storage | 4-tier hierarchical | Flat key-value |
| Retrieval | 10 strategies + adaptive selection | 1-2 strategies |
| Knowledge graph | Association graph + communities + spreading activation | None or basic |
| Emotions | Full affective system | None |
| Metacognition | Bias detection, confabulation audit, consciousness metrics | None |
| Goals | Priority queue + curiosity-driven exploration | None |
| Memory lifecycle | Decay, consolidation, reconsolidation, deduplication | Store/delete |
| Prediction | Forecast, counterfactuals, future projection | None |
| Multi-agent | Namespace isolation, federation, trust scoring | None |

## License

MIT

## Author

**Shehzad Ahmed** — [GitHub](https://github.com/supermaxlol)

## Links

- [PyPI](https://pypi.org/project/emms-mcp/)
- [npm](https://www.npmjs.com/package/emms-mcp)
- [GitHub](https://github.com/supermaxlol/emms-mcp)
