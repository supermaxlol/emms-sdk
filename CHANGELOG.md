# EMMS Changelog

All notable changes to the Enhanced Memory Management System are documented here.

---

## [0.13.0] — 2026-02-22

### Added
- **`MetacognitionEngine`** + **`MetacognitionReport`** + **`MemoryConfidence`** + **`DomainProfile`** + **`ContradictionPair`** (`memory/metacognition.py`) — epistemic self-knowledge layer; `assess(item)` computes confidence via weighted geometric mean of strength / recency / access / consolidation factors; `assess_all()` ranks all memories; `knowledge_map()` builds per-domain profiles (count, mean confidence, coverage, importance, strength); `find_contradictions()` detects pairs with semantic token overlap ≥ threshold but opposing emotional valence; `find_gaps()` flags sparse/low-confidence domains; `report()` synthesises everything into a `MetacognitionReport` with actionable recommendations; EMMS façades: `assess_memory`, `metacognition_report`, `knowledge_map`, `find_contradictions`
- **`ProspectiveMemory`** + **`Intention`** + **`IntentionActivation`** (`memory/prospection.py`) — future-oriented intention storage; `intend(content, trigger_context, priority)` stores intentions with trigger descriptions; `check(current_context)` activates matching intentions via Jaccard token overlap weighted by priority; `fulfill(id)` / `dismiss(id)` lifecycle; `pending()` returns unfulfilled sorted by priority; `save(path)` / `load(path)` JSON persistence; biological analogue: rostral prefrontal prospective memory; EMMS façades: `enable_prospective_memory`, `intend`, `check_intentions`, `fulfill_intention`, `pending_intentions`
- **`ContextualSalienceRetriever`** + **`SalienceResult`** (`retrieval/contextual.py`) — dynamic memory spotlight on the current conversational context; rolling context window (deque, configurable `window_size`); `update_context(text, valence)` accumulates recent turns; `retrieve()` scores all memories on four axes — semantic token overlap with context, memory importance, storage recency (exponential decay), affective resonance (valence match); results sorted by combined salience score; `context_summary` and `context_valence` properties; biological analogue: context-dependent memory (Godden & Baddeley 1975); EMMS façades: `enable_contextual_retrieval`, `update_context`, `contextual_retrieve`, `context_summary`
- **CLI commands** (5 new, 56 total): `metacognition`, `knowledge-map`, `contradictions`, `intend`, `check-intentions`
- **MCP tools** (5 new, 52 total): `emms_metacognition_report`, `emms_knowledge_map`, `emms_find_contradictions`, `emms_intend`, `emms_check_intentions`
- 90 new tests in `tests/test_v130_features.py`; total: **1247 passed, 2 skipped**

---

## [0.12.0] — 2026-02-22

### Added
- **`AssociationGraph`** + **`AssociationEdge`** + **`ActivationResult`** + **`AssociationStats`** (`memory/association.py`) — explicit memory-to-memory association graph; four auto-detected edge types: `"semantic"` (cosine similarity of embeddings), `"temporal"` (stored within window), `"affective"` (valence proximity), `"domain"` (same domain string), plus `"explicit"` for manual edges; `spreading_activation(seed_ids, decay, steps)` — BFS activation that decays along edges; `neighbors(id, min_weight)` — sorted edge list; `strongest_path(id_a, id_b)` — Dijkstra maximising edge-weight product via −log(w) metric; `stats()` returns `AssociationStats`; `EMMS.build_association_graph()`, `EMMS.associate()`, `EMMS.spreading_activation()`, `EMMS.association_stats()` façades
- **`InsightEngine`** + **`InsightReport`** + **`InsightBridge`** (`memory/insight.py`) — cross-domain insight generation; walks association graph edges for pairs from different domains with weight ≥ `min_bridge_weight`; synthesises new `"insight"` domain memories via template combining both excerpts; stores them in hierarchical memory; `InsightReport.summary()` lists top bridges; biological analogue: Default Mode Network / REM analogical binding; `EMMS.discover_insights()` façade
- **`AssociativeRetriever`** + **`AssociativeResult`** (`retrieval/associative.py`) — retrieval via spreading activation; `retrieve(seed_ids)` — activation spread from explicit seed IDs; `retrieve_by_query(query)` — auto-selects seeds via token-overlap BM25-lite then spreads; results include `activation_score`, `steps_from_seed`, `path`; biological analogue: priming — recalling one memory raises accessibility of associated memories; `EMMS.associative_retrieve()`, `EMMS.associative_retrieve_by_query()` façades
- **CLI commands** (5 new, 51 total): `association-graph`, `activation`, `discover-insights`, `associative-retrieve`, `association-stats`
- **MCP tools** (5 new, 47 total): `emms_build_association_graph`, `emms_spreading_activation`, `emms_discover_insights`, `emms_associative_retrieve`, `emms_association_stats`
- 83 new tests in `tests/test_v120_features.py`; total: **1157 passed, 2 skipped**

---

## [0.11.0] — 2026-02-20

### Added
- **`DreamConsolidator`** + **`DreamReport`** + **`DreamEntry`** (`memory/dream.py`) — between-session memory consolidation: samples top-k memories via ExperienceReplay, strengthens them with ReconsolidationEngine, weakens neglected bottom-k, prunes below strength threshold, runs optional SemanticDeduplicator + PatternDetector passes, produces DreamReport with insights; emits `memory.dream_completed` event; `EMMS.dream()` façade
- **`SessionBridge`** + **`BridgeRecord`** + **`BridgeThread`** (`sessions/bridge.py`) — session-to-session context handoff: `capture()` identifies unresolved high-importance memories (low consolidation_score), records emotional arc, presence score, dominant domains; `inject()` generates prompt-ready markdown context for next session opening; `save()`/`load()` JSON persistence; `EMMS.capture_session_bridge()`, `inject_session_bridge()` façades
- **`MemoryAnnealer`** + **`AnnealingResult`** (`memory/annealing.py`) — temporal memory annealing after session gaps; temperature model `T = 1/(1+gap/half_life)` (high T = recent/plastic, low T = old/stable); accelerated decay for weak/unimportant memories; emotional valence stabilisation toward neutral; strengthening of high-importance survivors; `EMMS.anneal(last_session_at)` façade
- **CLI commands** (5 new, 46 total): `dream`, `capture-bridge`, `inject-bridge`, `anneal`, `bridge-summary`
- **MCP tools** (5 new, 42 total): `emms_dream`, `emms_capture_bridge`, `emms_inject_bridge`, `emms_anneal`, `emms_bridge_summary`
- 80 new tests in `tests/test_v110_features.py`; total: **1074 passed, 2 skipped**

### Changed
- `__version__` bumped to `0.11.0`
- Updated legacy version-pinned tests for new tool count (42) and version (0.11.0)

---

## [0.10.0] — 2026-02-20

### Added
- **`ReconsolidationEngine`** + **`ReconsolidationResult`** + **`ReconsolidationReport`** (`memory/reconsolidation.py`) — biological memory reconsolidation: recalled memories enter a labile state and re-stabilise with altered strength and emotional valence; `reinforce` mode (confirming recall) increases strength; `weaken` mode (contradicting recall) decreases strength; `valence_drift` nudges stored valence toward the recall context's valence; diminishing-returns attenuation via `1/log(1 + access_count * base)`; `batch_reconsolidate()` for post-retrieval bulk update; `decay_unrecalled()` for passive strength decay of stale memories; `EMMS.reconsolidate()`, `batch_reconsolidate()`, `decay_unrecalled()` façades
- **`PresenceTracker`** + **`PresenceMetrics`** + **`PresenceTurn`** (`sessions/presence.py`) — models the finite attentional window of a session; presence decays via half-life sigmoid `1/(1+(t/half_life)^γ)`; tracks `presence_score`, `attention_budget_remaining`, `coherence_trend` (stable/degrading/recovering), per-turn `emotional_arc`, `dominant_domains`, `mean_valence`/`mean_intensity`; `is_degrading` flag when below configurable threshold; `EMMS.enable_presence_tracking()`, `record_presence_turn()`, `presence_metrics()` façades
- **`AffectiveRetriever`** + **`AffectiveResult`** + **`EmotionalLandscape`** (`retrieval/affective.py`) — retrieval by emotional proximity using `Experience.emotional_valence` and `Experience.emotional_intensity`; proximity = `1 - sqrt((v_diff² + i_diff²)/2)`; optional BM25 semantic blend (`semantic_blend` weight); `retrieve_similar_feeling(reference_id)` finds memories emotionally near a reference; `emotional_landscape()` returns distribution summary (mean/std, valence/intensity histograms, most positive/negative/intense IDs); `EMMS.affective_retrieve()`, `affective_retrieve_similar()`, `emotional_landscape()` façades
- **CLI commands** (6 new, 41 total): `reconsolidate`, `decay-unrecalled`, `presence`, `presence-arc`, `affective-retrieve`, `emotional-landscape`
- **MCP tools** (5 new, 37 total): `emms_reconsolidate`, `emms_batch_reconsolidate`, `emms_presence_metrics`, `emms_affective_retrieve`, `emms_emotional_landscape`
- 82 new tests in `tests/test_v100_features.py`; total: **994 passed, 2 skipped**

### Changed
- `__version__` bumped to `0.10.0`
- `EMMS.enable_presence_tracking()` now accepts `budget_horizon` and `degrading_threshold` params
- Updated legacy version-pinned tests for new tool count (37) and version (0.10.0)

---

## [0.9.0] — 2026-02-20

### Added
- **`CompactionIndex`** (`storage/index.py`) — O(1) dual-hash lookup index (by `memory_id`, `experience_id`, content-hash SHA-256); `register()`, `remove()`, `find_by_content()`, `bulk_register()`, `rebuild_from()`; auto-wired into `EMMS.__init__` and `EMMS.store()`; `EMMS.get_memory_by_id()`, `get_memory_by_experience_id()`, `find_memories_by_content()`, `rebuild_index()`, `index_stats()` façades
- **`GraphCommunityDetector`** + **`Community`** + **`CommunityResult`** (`memory/communities.py`) — Label Propagation Algorithm (LPA) for topic-cluster discovery in entity-relationship graph; weighted by `Relationship.strength`; modularity Q computation; `CommunityResult.get_community_for_entity()`, `export_markdown()`, `summary()`; bridge-entity detection; `EMMS.graph_communities()`, `graph_community_for_entity()` façades
- **`ExperienceReplay`** + **`ReplayEntry`** + **`ReplayBatch`** (`memory/replay.py`) — Prioritized Experience Replay (PER) with Importance Sampling correction weights; priority = `w_imp*I + w_str*S + w_rec*R + w_nov*N`; Vose alias-method O(1) sampling; exclusion window to prevent over-sampling; `sample()`, `sample_top()`, `replay_context()`; `EMMS.enable_experience_replay()`, `replay_sample()`, `replay_context()`, `replay_top()` façades
- **`MemoryFederation`** + **`FederationResult`** + **`ConflictEntry`** + **`ConflictPolicy`** (`storage/federation.py`) — multi-agent snapshot merge; three conflict policies: `local_wins`, `newest_wins`, `importance_wins`; content-hash deduplication (skip near-identical items); optional `namespace_prefix` for id-space isolation; graph entity/relationship merge; `EMMS.merge_from()`, `federation_export()` façades
- **`MemoryQueryPlanner`** + **`QueryPlan`** + **`SubQueryResult`** (`retrieval/planner.py`) — heuristic query decomposition (conjunction / comma / question-mark splits); parallel sub-query retrieval via `HybridRetriever`; cross-boost (+0.10 per additional sub-query hit); deduplication by memory id; `QueryPlan.summary()`; `EMMS.plan_retrieve()`, `plan_retrieve_simple()` façades
- **CLI commands** (7 new, 35 total): `index-lookup`, `index-stats`, `graph-communities`, `replay`, `replay-top`, `merge-from`, `plan-retrieve`
- **MCP tools** (5 new, 32 total): `emms_index_lookup`, `emms_graph_communities`, `emms_replay_sample`, `emms_merge_from`, `emms_plan_retrieve`
- 129 new tests in `tests/test_v090_features.py`; total: **912 passed, 2 skipped**

### Changed
- `__version__` bumped to `0.9.0`
- `EMMS.store()` now calls `self.index.register(mem_item)` for O(1) future lookups
- Updated `tests/test_v070_features.py` and `tests/test_v080_features.py` for new tool count (32) and version

---

## [0.8.0] — 2026-02-19

### Added
- **`HybridRetriever`** + **`HybridSearchResult`** (`retrieval/hybrid.py`) — BM25 lexical scoring fused with embedding cosine similarity via Reciprocal Rank Fusion (RRF, k=60); `_BM25` pure-Python implementation (k1=1.5, b=0.75); `_rrf_fuse()` rank-based fusion; `HybridSearchResult.to_retrieval_result()` for interoperability; `EMMS.hybrid_retrieve()` façade
- **`MemoryTimeline`** + **`TimelineResult`** + **`TimelineEvent`** + **`TemporalGap`** + **`DensityBucket`** (`analytics/timeline.py`) — chronological memory reconstruction; gap detection (configurable threshold); fixed-width density histogram; `TimelineResult.summary()` and `export_markdown()`; domain/since/until/tier filters; `EMMS.build_timeline()` façade
- **`AdaptiveRetriever`** + **`StrategyBelief`** (`retrieval/adaptive.py`) — Thompson Sampling Beta-Bernoulli multi-armed bandit over 5 retrieval strategies (semantic, bm25, temporal, domain, importance); `StrategyBelief` with `alpha`, `beta`, `mean`, `variance`, `sample()`, `update(decay)`; Marsaglia-Tsang pure-Python Gamma/Beta sampler; `save_state()` / `load_state()` as JSON; `EMMS.enable_adaptive_retrieval()`, `.adaptive_retrieve()`, `.adaptive_feedback()`, `.get_retrieval_beliefs()` façades
- **`MemoryBudget`** + **`BudgetReport`** + **`EvictionCandidate`** + **`EvictionPolicy`** (`context/budget.py`) — token-budget-aware memory eviction; five eviction policies: `composite`, `lru`, `lfu`, `importance`, `strength`; composite score = weighted importance + strength + log-access + recency-decay; importance-threshold and tier-based protection; `dry_run` mode; `EMMS.memory_token_footprint()`, `.enforce_memory_budget()` façades
- **`MultiHopGraphReasoner`** + **`MultiHopResult`** + **`HopPath`** + **`ReachableEntity`** (`memory/multihop.py`) — BFS traversal up to configurable max_hops over `GraphMemory._adj`; path-strength scoring (product of edge strengths); approximate betweenness bridging scores; `MultiHopResult.to_dot()` Graphviz export; `MultiHopResult.summary()`; `EMMS.multihop_query()` façade
- **CLI commands** (7 new, 28 total): `hybrid-retrieve`, `timeline`, `adaptive-retrieve`, `retrieval-beliefs`, `budget-status`, `budget-enforce`, `multihop`
- **MCP tools** (5 new, 27 total): `emms_hybrid_retrieve`, `emms_build_timeline`, `emms_adaptive_retrieve`, `emms_enforce_budget`, `emms_multihop_query`
- 127 new tests in `tests/test_v080_features.py`; total: **783 passed, 2 skipped**

### Changed
- `__version__` bumped to `0.8.0`

---

## [0.7.0] — 2026-02-19

### Added
- **`MemoryDiff`** (`memory/diff.py`) — session-to-session memory snapshot comparison; `DiffResult` with `added`, `removed`, `strengthened`, `weakened`, `superseded` lists; `MemoryDiff.from_paths()`, `from_memories()`, `diff()` static methods; `DiffResult.summary()` and `export_markdown()` for human-readable output; `EMMS.diff_since(snapshot_path)` façade
- **`MemoryCluster`** + **`MemoryClustering`** (`memory/clustering.py`) — pure-Python k-means++ with TF-IDF bag-of-words fallback (zero ML dependency); `cluster(items, k, auto_k)` with elbow-method k selection; `cluster_with_embeddings()` for embedding-vector clustering; auto-labels via domain + top tokens; `EMMS.cluster_memories(k, auto_k, tier)` façade
- **`ConversationBuffer`** (`sessions/conversation.py`) — sliding-window conversation history with automatic chunked summarisation; `observe_turn(role, content)`, `get_context(max_tokens)`, `all_turns()`; extractive summarisation (zero dependency) + optional LLM-backed summarisation; `EMMS.build_conversation_context(turns, max_tokens)` façade
- **`HierarchicalMemory.stream_retrieve()`** — async generator yielding `RetrievalResult` tier-by-tier (semantic → long_term → short_term → working); `asyncio.sleep(0)` between tier boundaries for cooperative multitasking; `EMMS.astream_retrieve(query, max_results)` façade
- **`LLMConsolidator`** (`llm/consolidator.py`) — union-find single-linkage clustering on cosine/lexical similarity matrix; `consolidate_cluster(items, llm_enhancer)` synthesises a cluster into one `Experience` via LLM prompt + JSON extraction; `auto_consolidate(threshold, tier)` scans a memory tier end-to-end; `consolidate_from_clusters(clusters)` accepts pre-built `MemoryCluster` objects; extractive fallback when no LLM is supplied; `ConsolidationResult` with `clusters_found / synthesised / stored / failed / elapsed_s`; `EMMS.llm_consolidate(threshold, tier)` façade
- **`llm/`** package — `emms.llm.__init__` + `emms.llm.consolidator` module structure
- **`EMMS.astream_retrieve()`** — async generator wrapping `HierarchicalMemory.stream_retrieve()`
- **`EMMS.diff_since(snapshot_path)`** — compare current state against a saved snapshot
- **`EMMS.cluster_memories(k, auto_k, tier)`** — cluster tier items (embedding-aware)
- **`EMMS.llm_consolidate(threshold, tier)`** — async LLM-backed consolidation
- **`EMMS.build_conversation_context(turns, max_tokens)`** — ConversationBuffer helper
- **CLI command**: `emms diff <snapshot_a> <snapshot_b>` — compare two snapshot files; `--output` for Markdown export; `--threshold` for strength delta
- **MCP tools** (2 new): `emms_cluster_memories`, `emms_llm_consolidate` (22 tools total)
- 73 new tests in `tests/test_v070_features.py`; total: **656 passed, 2 skipped**

### Changed
- `__version__` bumped to `0.7.0`

---

## [0.6.0] — 2026-02-19

### Added
- **`ImportanceClassifier`** (`core/importance.py`) — auto-scores `Experience.importance` from six weighted content signals (entity density, novelty passthrough, emotional weight, length saturation, high-stakes keyword fraction, structural richness); zero ML dependencies; `enrich()` updates in-place only when importance is still at default; `score_breakdown()` for explainability
- **`RAGContextBuilder`** (`context/rag_builder.py`) — token-budget-aware context packer for RAG pipelines; greedy score-descending block selection; four output formats: `markdown`, `xml`, `json`, `plain`; `ContextBlock.from_retrieval_result()` converter; `EMMS.build_rag_context()` façade
- **`SemanticDeduplicator`** (`memory/compression.py`) — near-duplicate detection via cosine (threshold 0.92) + lexical (threshold 0.85) similarity; `find_duplicate_groups()` + `resolve_groups()` keep best by `importance * 0.6 + access_count * 0.4`; `EMMS.deduplicate()` façade
- **`MemoryScheduler`** (`scheduler.py`) — composable async background maintenance with 5 built-in jobs: `consolidation` (60 s), `ttl_purge` (300 s), `deduplication` (600 s), `pattern_detection` (300 s), `srs_review` (3600 s); `register()` for custom jobs; `enable()`/`disable()`/`set_interval()` for fine-grained control; `EMMS.start_scheduler()` / `stop_scheduler()` façade
- **`SpacedRepetitionSystem`** + **`SRSCard`** (`memory/spaced_repetition.py`) — SM-2 algorithm; `enroll()` / `enroll_all()` / `record_review(quality 0–5)` / `get_due_items()` / `save_state()` / `load_state()`; `EMMS.srs_enroll()`, `srs_enroll_all()`, `srs_record_review()`, `srs_due()` façade
- **`GraphMemory.to_dot()`** — Graphviz DOT export with configurable `max_nodes`, `min_importance`, and `highlight` colour list; `EMMS.export_graph_dot()` façade
- **`GraphMemory.to_d3()`** — D3.js force-graph JSON (`nodes` + `links` arrays); `EMMS.export_graph_d3()` façade
- **`MemoryItem.srs_enrolled`**, **`.srs_next_review`**, **`.srs_interval_days`** — SRS state fields on every memory item
- **`MemoryConfig.dedup_cosine_threshold`**, **`.dedup_lexical_threshold`**, **`.enable_auto_dedup`** — deduplication knobs
- **`EMMS.importance_clf`** — `ImportanceClassifier` instance; `store()` now auto-enriches importance when still at default; `score_importance()` returns per-signal breakdown
- **`EMMS.deduplicator`** — `SemanticDeduplicator` instance constructed with config thresholds
- **`EMMS.srs`** — `SpacedRepetitionSystem` wired into `save()` / `load()` (sidecar `_srs.json`)
- **CLI commands**: `build-rag`, `deduplicate`, `srs-enroll`, `srs-review`, `srs-due`, `export-graph`
- **MCP tools** (7 new): `emms_build_rag_context`, `emms_deduplicate`, `emms_srs_enroll`, `emms_srs_record_review`, `emms_srs_due`, `emms_export_graph_dot`, `emms_export_graph_d3`
- 90 new tests across 9 test classes; total: **583 passed, 2 skipped**

### Changed
- `EMMS.store()` — automatically calls `importance_clf.enrich(experience)` before storing
- `EMMS.save()` / `EMMS.load()` — persist/restore SRS state alongside memory, graph, procedural, and consciousness
- `__version__` bumped to `0.6.0`

---

## [0.5.2] — 2026-02-19

### Added
- **`Experience.namespace`** (`str`, default `"default"`) — partition memories by project or repository; retrieval only crosses namespace boundaries when explicitly requested
- **`Experience.confidence`** (`float 0–1`, default `1.0`) — uncertainty rating for a memory; low-confidence memories receive a proportional score penalty in retrieval
- **`CompactResult.namespace` + `.confidence`** — exposed in progressive-disclosure layer 1 results
- **`HierarchicalMemory.retrieve_filtered()`** — pre-filter by namespace, `obs_type`, domain, session, Unix time range, and minimum confidence before scoring; confidence scaling applied
- **`HierarchicalMemory.upvote(memory_id)`** — positive feedback: strengthens a memory by `boost` (default 0.1), records access
- **`HierarchicalMemory.downvote(memory_id)`** — negative feedback: weakens a memory by `decay` (default 0.2)
- **`HierarchicalMemory.export_markdown(path)`** — human-readable Markdown export grouped by domain; includes title, facts, files, and lifecycle metadata
- **`EMMS.retrieve_filtered()`**, **`EMMS.upvote()`**, **`EMMS.downvote()`**, **`EMMS.export_markdown()`** — top-level delegation methods
- **CLI commands**: `retrieve-filtered`, `upvote`, `downvote`, `export-md`
- **MCP tools**: `emms_retrieve_filtered`, `emms_upvote`, `emms_downvote`, `emms_export_markdown`
- 27 new tests (namespace/confidence, filtered retrieval, feedback, markdown export)

### Changed
- `EMMS.validate_citations()` — cited memories now also call `item.touch()` on each hit (access bump)

---

## [0.5.1] — 2026-02-19

### Added
- **`Experience.update_mode`** (`"insert"` | `"patch"`) and **`Experience.patch_key`** — LangMem patch semantics: store updates an existing memory in-place, archiving the old version
- **`Experience.citations`** (`list[str]`) — GitHub Copilot-inspired citation links to other memory IDs
- **`Experience.namespace`** — (backported to 0.5.1 schema; see 0.5.2 for full retrieval integration)
- **`MemoryItem.expires_at`** — hard TTL; `touch(ttl_seconds)` refreshes expiry on use (Copilot pattern)
- **`MemoryItem.superseded_by`** — conflict archival: links to the newer memory that replaced this one
- **`MemoryItem.is_expired`** and **`MemoryItem.is_superseded`** computed properties
- **`RetrievalResult.strategy_scores`** (`dict[str, float]`) — per-strategy breakdown (semantic, temporal, importance, domain, …)
- **`RetrievalResult.explanation`** (`str`) — top-3 strategies formatted as human-readable string
- **`ImportanceStrategy`** — 6th retrieval strategy: `importance × 0.6 + memory_strength × 0.4`
- **`EnsembleRetriever.from_balanced()`** updated to 60/20/10/10 (Semantic/Temporal/Importance/Domain)
- **`EnsembleRetriever.from_identity()`** updated to 30/20/15/15/10/10 (6 strategies including ImportanceStrategy)
- **`HierarchicalMemory._find_patch_target()`** + conflict archival in `store()` — old memory gets `superseded_by = new_item.id`
- **TTL-aware filtering** in `HierarchicalMemory.retrieve()` — expired/superseded memories skipped
- **`HierarchicalMemory.search_by_file(file_path)`** — find memories referencing a file (substring match across `files_read` + `files_modified`)
- **`ProceduralMemory`** (`memory/procedural.py`) — 5th memory tier: add/patch/remove behavioral rules; `get_prompt()` returns formatted system-prompt block; `save_state()`/`load_state()` for persistence
- **`EMMS.procedures`** — `ProceduralMemory` instance wired in; `add_procedure()` and `get_system_prompt_rules()` top-level helpers
- **`EMMS.validate_citations(experience)`** — checks cited memory IDs exist; strengthens found memories
- **`EMMS.search_by_file(file_path)`** — top-level delegation to `HierarchicalMemory.search_by_file()`
- **GraphMemory persistence**: `GraphMemory.save_state(path)` / `load_state(path)` — adjacency list rebuilt on load
- **Graph state wired into `EMMS.save()` / `EMMS.load()`** alongside hierarchical + consciousness
- **Procedural state wired into `EMMS.save()` / `EMMS.load()`**
- **Debounced consolidation** in `SessionManager` — `consolidate_every` parameter (default 20); auto-triggers `memory.consolidate()` after N stores
- **PatternDetector wired** into `_consolidation_loop()` — runs every 5 consolidation passes, emits `memory.patterns_detected` event
- **Lock removed from `aretrieve()`** — retrieval is read-only; no lock needed (reduces contention)
- **Consciousness persistence** — `_retroactive_boost` (narrator), `_domain_curiosity` (meaning_maker), `core_creeds` (ego_boundary) now saved/restored
- **`EMCPServer`** (`adapters/mcp_server.py`) — MCP adapter with 9 tools (store, retrieve, search_compact, search_by_file, get_stats, get_procedures, add_procedure, save, load)
- **CLI** (`cli.py` + `pyproject.toml` entry point `emms`) — 9 subcommands: store, retrieve, compact, search-file, stats, save, load, procedures, add-procedure
- 48 new tests

### Changed
- `pyproject.toml` version bumped to `0.5.1`; `[project.scripts]` entry point added

---

## [0.5.0] — 2026-02-18

### Added
- **`SessionManager`** — persistent session lifecycle, auto `session_id` injection, JSONL log, `generate_claude_md()`, `generate_context_injection()`
- **`SessionSummary`** — structured per-session narrative (request / investigated / learned / completed / next_steps)
- **`ObsType`** + **`ConceptTag`** — 6-type observation taxonomy + 7-tag epistemological classifier (claude-mem inspired)
- **`EnsembleRetriever.from_balanced()`** + **`from_identity()`** factory presets
- **`EnsembleRetriever.search_compact()`** — compact index layer (50-80 tokens/result)
- **`ChromaSemanticStrategy`** — ChromaDB-backed semantic retrieval strategy
- **`HierarchicalMemory` Endless Mode** — biomimetic real-time compression; O(N²) → O(N) context growth
- **`ToolObserver`** — converts PostToolUse hook payloads to `Experience` objects with inferred `obs_type`, `concept_tags`, `files_read`, `files_modified`, `facts`, `title`, `subtitle`
- **`ToolObserver.observe_prompt()`** — captures UserPromptSubmit payloads
- **`HierarchicalMemory.export_jsonl()`** + **`import_jsonl()`**
- **`HierarchicalMemory.get_timeline()`** + **`get_sessions()`**
- **`Experience.facts`**, **`.files_read`**, **`.files_modified`**, **`.title`**, **`.subtitle`** — rich structured content fields
- **`CompactResult.token_estimate`** — approximate token budget for full content
- **`SessionManager.generate_context_injection()`** — formatted compact index for session-start context injection
- **`MemoryAnalytics`** — health score, tier distribution, domain/concept coverage, session stats
- BM25 (k1=1.5, b=0.75) lexical retrieval replacing Jaccard overlap
- 55 new tests

---

## [0.4.0] — 2026-02

### Added
- **`EventBus`** — pub/sub for inter-component events
- **`GraphMemory`** — regex-based NER, relationship extraction, subgraph queries, BFS path finding
- **`VectorIndex`** — numpy batch cosine similarity; replaces O(n) per-item scan
- **`EnsembleRetriever`** — 5-strategy weighted ensemble (Semantic, Temporal, Emotional, Graph, Domain)
- Full memory persistence (`HierarchicalMemory.save_state()` / `load_state()`)
- Enhanced consciousness modules: traits, autobiographical events, identity milestones, A-MEM associative linking
- Advanced episode detection: spectral clustering, conductance optimization
- Pattern detection in memory compression (`PatternDetector`)
- LLM integration layer (`LLMEnhancer`): Claude, GPT-4, Ollama — classify + compress
- Real-time data pipeline (`AsyncRealTimePipeline`)
- Background consolidation (`EMMS.start_background_consolidation()`)
- `MetaCognitiveMonitor` — 3rd-person self-analysis

---

## [0.3.x] — Prior

- 4-tier hierarchical memory (Atkinson-Shiffrin model)
- Miller's Law working-memory capacity (7±2)
- Exponential decay and importance-weighted consolidation
- Cross-modal binding (6 modalities)
- Token context management with 3-tier eviction
- PersistentIdentity + ego state
- Episode boundary detection
- Identity prompt builder with empirically validated templates
