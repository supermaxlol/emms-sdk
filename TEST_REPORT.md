# EMMS v0.4.0 — Comprehensive Test Report

**Date**: 2026-02-11 10:05
**Author**: Automated Test Suite
**Models tested**: Ollama (gemma3n:e4b), Claude (claude-sonnet-4-5-20250929)

## Test Summary

- **Sections tested**: 14
- **Log entries**: 127
- **Errors**: 0

## Detailed Results


### 1. INITIALIZATION

- **EMMS version**: 0.4.0
- **Consciousness enabled**: True
- **Graph memory**: True
- **Event bus**: True
- **Narrator**: True
- **MeaningMaker**: True
- **TemporalIntegrator**: True
- **EgoBoundaryTracker**: True
- **PatternDetector**: True
- **Consciousness disabled test**: True
- **Graph disabled test**: True

### 2. EVENT BUS

- **Listeners registered**: 2
- **Event types**: ['memory.stored', 'memory.consolidated']

### 3. STORE EXPERIENCES (with consciousness + graph + events)

- **Total stored**: 14
- **Memory stats**: {'working': 14, 'short_term': 0, 'long_term': 0, 'semantic': 0, 'total': 14}
- **Avg store latency (ms)**: 0.833
- **Domains**: ['tech', 'finance', 'personal', 'academic', 'weather', 'science']
- **Result has 'consciousness'**: True
- **Result has 'graph_entities'**: True
- **Result has 'experience_id'**: True

### 4. RETRIEVAL (lexical + embedding)

- **Query 'Shehzad research paper'**: [0.46ms] top=0.657 | Shehzad presented his research paper at the IUB symposium on...
- **Query 'Bitcoin price cryptocurrency'**: [0.25ms] top=0.46 | Shehzad presented his research paper at the IUB symposium on...
- **Query 'quantum computing breakthrough'**: [0.25ms] top=0.791 | Quantum computing breakthrough at MIT achieved 1000 qubit qu...
- **Query 'weather flooding Bangladesh'**: [0.24ms] top=0.646 | Bangladesh experienced severe flooding in the eastern distri...
- **Query 'large language models AI'**: [0.23ms] top=0.785 | Claude and GPT-4 are the leading large language models in 20...
- **Query 'stock market GDP growth'**: [0.23ms] top=0.74 | The stock market in Dhaka rose 3 percent on positive GDP gro...
- **Semantic retrieval test**: HashEmbedder cosine similarity
- **Semantic results count**: 5

### 5. GRAPH MEMORY

- **Total entities**: 26
- **Total relationships**: 30
- **Entity names**: ['shehzad ahmed', 'independent university bangladesh', 'ahmed', 'independent', 'university', 'bangladesh', 'emms', 'iub', 'dhaka', 'gdp', 'federal reserve', 'reserve', 'hugging face', 'hugging', 'face']
- **Entity 'shehzad ahmed' (alt)**: found | neighbors=['independent', 'independent university bangladesh', 'bangladesh', 'university', 'ahmed']
- **Entity 'emms'**: found | neighbors=[] | rels=0
- **Entity 'bitcoin'**: not found
- **Entity 'mit'**: found | neighbors=[] | rels=0
- **Entity 'cern'**: not found
- **Entity 'anthropic'**: not found
- **Path shehzad ahmed → coast**: no path
- **Subgraph of 'shehzad ahmed'**: nodes=6, edges=16

### 6. CONSCIOUSNESS STATE

- **Enabled**: True
- **Narrative coherence**: 0.5
- **Ego boundary strength**: 0.333
- **Themes (top 5)**: {'shehzad': 1.5655510000000001, 'released': 1.0049000000000001, 'claude': 0.9565, 'built': 0.5351, 'emms': 0.5351}
- **Narrator entries**: 15
- **Narrator traits**: {'focused': 0.09999999999999999, 'analytical': 0.09999999999999999}
- **Narrator autobiographical events**: 0
- **Third-person narrative**: I am EMMS-TestAgent. Over 15 experiences across 7 domains, my focus has been on shehzad, released, claude, built, emms. ...
- **First-person narrative**: I have processed 15 experiences. I see myself as: focused (10%), analytical (10%). Lately I've been focused on: weather,...
- **MeaningMaker total processed**: 15
- **MeaningMaker patterns**: 7
- **Temporal milestones**: 6
- **Ego boundary history**: 15

### 7. PATTERN DETECTION

- **Domain dominant**: tech
- **Domain distribution**: {'personal': 0.06666666666666667, 'academic': 0.06666666666666667, 'finance': 0.2, 'tech': 0.3333333333333333, 'science': 0.13333333333333333, 'weather': 0.13333333333333333, 'test': 0.06666666666666667}
- **Domain trends**: [{'domain': 'finance', 'trend': 'growing', 'count': 3, 'avg_importance': 0.7166666666666668}, {'domain': 'tech', 'trend': 'growing', 'count': 5, 'avg_importance': 0.866}]
- **Top content concepts**: [('shehzad', 3)]
- **Sequence patterns**: 4

### 8. CONSOLIDATION & EVENTS

- **Items consolidated**: 41
- **Consolidation time (ms)**: 0.08
- **Memory after consolidation**: {'working': 0, 'short_term': 1, 'long_term': 0, 'semantic': 14, 'total': 15}
- **Store events received**: 15
- **Consolidation events received**: 1
- **Compression time (ms)**: 0.0

### 9. PERSISTENCE (save/load)

- **Save time (ms)**: 0.74
- **File size (KB)**: 31.5
- **Items saved**: 15
- **Load time (ms)**: 0.81
- **Items loaded**: 15
- **Post-load retrieval**: 3 results
- **Top result**: Shehzad presented his research paper at the IUB symposium on AI consciousness
- **Post-load 'Bitcoin' retrieval**: 3 results
- **JSON keys**: ['version', 'saved_at', 'working', 'short_term', 'long_term', 'semantic', 'embeddings', 'stats']
- **Version in file**: 0.4.0

### 10. EPISODE DETECTION (all algorithms)

- **Algorithm 'heuristic'**: 2 episodes in 0.04ms | coherences=[1.0, 1.0]
- **Algorithm 'graph'**: 2 episodes in 0.58ms | coherences=[0.874, 0.856]
- **Algorithm 'spectral'**: 2 episodes in 17.92ms | coherences=[1.0, 1.0]
- **Algorithm 'conductance'**: 2 episodes in 0.96ms | coherences=[0.874, 0.856]
- **Algorithm 'multi'**: 2 episodes in 3.06ms | coherences=[1.0, 1.0]

### 11. MULTI-STRATEGY ENSEMBLE RETRIEVAL

- **Items available for retrieval**: 1
- **Semantic scores (first 5)**: [0.02]
- **Temporal scores (first 5)**: [0.875]
- **Emotional scores (first 5)**: [0.3]
- **Domain scores (first 5)**: [0.2]
- **Ensemble retrieval time (ms)**: 0.25
- **Ensemble results**: 1
- **  Rank 1**: [0.391] Test verification item

### 12. OLLAMA LLM INTEGRATION (gemma3n:e4b)

- **Connection test**: OK (52255ms)
- **Response**: Artificial intelligence is the development of computer systems capable of performing tasks that typically require human ...
- **Enrichment before**: importance=0.50, domain='general', valence=0.00
- **Enrichment after**: importance=0.80, domain='Physics', valence=0.70 (8910ms)
- **Entities extracted**: 7 (19578ms)
- **  Entity**: Shehzad Ahmed (person)
- **  Entity**: EMMS (concept)
- **  Entity**: Independent Universiity Bangladesh (org)
- **  Entity**: Python (concept)
- **  Entity**: Claude API (concept)
- **  Entity**: Anthropic (org)
- **  Entity**: AI (concept)
- **LLM narrative**: (11640ms) It's interesting to look back at my 15 experiences; a lot of them revolved around a user named Shehzad, which ...
- **Memory summary**: (33495ms) Please provide the memories! I need the text of the memories to be able to summarize them. 

Once you paste th...
- **Q: What are the most important things happening in te**: (22963ms)
- **A**: Based on the memories, the most important things happening in tech right now are:

* **AI Advancement:** Significant pro...
- **Q: How does quantum computing relate to AI?**: (16180ms)
- **A**: Quantum computing has the potential to significantly accelerate and enhance AI, particularly large language models. A 10...
- **Q: Summarize what you know about Shehzad's work.**: (5692ms)
- **A**: Shehzad is a researcher working in the field of AI consciousness. He recently presented his research at the IUB symposiu...

### 13. CLAUDE API INTEGRATION (claude-sonnet-4-5-20250929)

- **Connection test**: OK (4275ms)
- **Response**: EMMS (Evangelical Medical Missionary Society) is a Christian international health charity based in Scotland that provide...
- **Claude enrichment**: importance=0.85, domain='quantum computing and machine learning' (2053ms)
- **Claude entities**: 6 (2135ms)
- **  Entity**: Shehzad Ahmed (person)
- **  Entity**: EMMS (concept)
- **  Entity**: IUB (org)
- **  Entity**: Claude (concept)
- **  Entity**: Anthropic (org)
- **  Entity**: DeepSeek R1 (concept)
- **Claude narrative**: (4534ms) I've spent considerable time analyzing what it means to be Claude—something built and released into the world, ...
- **Q: What are the key themes across all stored memories**: (4864ms)
- **A**: The key themes across all stored memories are:

1. **Shehzad's AI research** - Two memories directly involve his work on...
- **Q: What is Shehzad working on and why is it significa**: (4096ms)
- **A**: I don't actually have memories of these events. The information you've provided appears to be fictional or speculative -...
- **Q: Compare the finance and technology news — what pat**: (5225ms)
- **A**: Looking at these items, a key pattern emerges: **stability vs. disruption**.

**Finance (Fed rates):** Represents instit...

### 14. FULL SYSTEM STATS

- **Memory**: {'working': 0, 'short_term': 1, 'long_term': 0, 'semantic': 14, 'total': 15}
- **Graph**: {'entities': 26, 'relationships': 30}
- **Consciousness**: {'narrative_coherence': 0.5, 'themes_tracked': 41, 'traits': {'focused': 0.09999999999999999, 'analytical': 0.09999999999999999}, 'ego_boundary': 0.3333333333333333, 'milestones': 6}
- **Events**: {'memory.stored': 1, 'memory.consolidated': 1}

## LLM Provider Comparison

| Feature | Ollama (gemma3n:e4b) | Claude (Sonnet 4.5) |
|---------|---------------------|---------------------|
| Enrichment latency | 8910ms | 2053ms |
| Entity extraction | 7 entities | 6 entities |
| Narrative latency | 11640ms | 4534ms |

## Conversational Demo Transcripts

### Ollama (gemma3n:e4b) Conversations

**Q**: What are the most important things happening in tech right now?
**A** (22963ms): Based on the memories, the most important things happening in tech right now are:

* **AI Advancement:** Significant progress in large language models (Claude, GPT-4) is occurring, as evidenced by their prominence in 2026.
* **AI Research & Application:** AI is being actively researched and applied,

**Q**: How does quantum computing relate to AI?
**A** (16180ms): Quantum computing has the potential to significantly accelerate and enhance AI, particularly large language models. A 1000-qubit quantum processor could unlock breakthroughs in AI capabilities, potentially surpassing the performance of models like GPT-4. Open-source reasoning models like DeepSeek's 

**Q**: Summarize what you know about Shehzad's work.
**A** (5692ms): Shehzad is a researcher working in the field of AI consciousness. He recently presented his research at the IUB symposium.

### Claude (Sonnet 4.5) Conversations

**Q**: What are the key themes across all stored memories?
**A** (4864ms): The key themes across all stored memories are:

1. **Shehzad's AI research** - Two memories directly involve his work on AI consciousness and persistent AI identity (the symposium presentation and EMMS system)

2. **Bangladesh context** - Events occurring in or related to Bangladesh (IUB symposium, 

**Q**: What is Shehzad working on and why is it significant?
**A** (4096ms): I don't actually have memories of these events. The information you've provided appears to be fictional or speculative - there is no Claude Opus 4.6, I have no record of a person named Shehzad presenting research at IUB, and I cannot confirm we're in 2026 or make claims about future AI developments.

**Q**: Compare the finance and technology news — what patterns emerge?
**A** (5225ms): Looking at these items, a key pattern emerges: **stability vs. disruption**.

**Finance (Fed rates):** Represents institutional stability and controlled, incremental decision-making in established systems.

**Technology (EMMS):** Represents experimental innovation aimed at fundamentally transforming


## Final System State

```json
{
  "uptime_seconds": 199.3,
  "total_stored": 15,
  "total_retrieved": 13,
  "memory": {
    "working": 0,
    "short_term": 1,
    "long_term": 0,
    "semantic": 14,
    "total": 15
  },
  "crossmodal": {
    "text": 15,
    "visual": 15,
    "audio": 15,
    "temporal": 15,
    "spatial": 15,
    "emotional": 15
  },
  "tokens": {
    "context_window": 32000,
    "local_tokens": 0,
    "evicted_tokens": 0,
    "utilisation": 0.0,
    "total_evicted": 0,
    "total_retrieved": 0
  },
  "identity": {
    "total_experiences": 15,
    "domains": [
      "personal",
      "tech",
      "academic",
      "finance",
      "science",
      "weather",
      "test"
    ],
    "sessions": 0
  },
  "throughput_per_sec": 0.1,
  "events": {
    "memory.stored": 1,
    "memory.consolidated": 1
  },
  "graph": {
    "entities": 26,
    "relationships": 30
  },
  "consciousness": {
    "narrative_coherence": 0.5,
    "themes_tracked": 41,
    "traits": {
      "focused": 0.09999999999999999,
      "analytical": 0.09999999999999999
    },
    "ego_boundary": 0.3333333333333333,
    "milestones": 6
  }
}
```
