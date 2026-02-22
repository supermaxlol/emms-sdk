# EMMS Changelog

All notable changes to the Enhanced Memory Management System are documented here.

---

## [0.21.0] — 2026-02-22

### Added
- **`PerspectiveTaker`** + **`AgentModel`** + **`PerspectiveReport`** (`memory/perspective.py`) — Theory of Mind: extracts mental models of other agents from memory content; scans for belief/communication verbs (`said`, `thinks`, `believes`, `wants`, `expects`, `argues`, `claims`, `suggests`, `reports`, `needs`, `hopes`, `fears`, `knows`, `decided`, `prefers`, `stated`, `noted`, `agreed`, `denied`); token immediately before the verb (len ≥ 3, not stop-word) = agent name; tokens after the verb (up to 5) = attributed statement content; `AgentModel` carries `name`, `mentions`, `statements`, `mean_valence`, `domains`; `build(domain)` → `PerspectiveReport` sorted by mentions; `take_perspective(agent_name)` → case-insensitive lookup; `all_agents(n)` → top-n by mentions; `min_mentions` filter; `max_agents` cap; biological analogue: Theory of Mind (Premack & Woodruff 1978); temporoparietal junction in belief representation (Saxe & Kanwisher 2003); medial PFC in self-other distinction (Mitchell 2009); mirror neuron system (Gallese et al. 1996); predictive social brain (Frith & Frith 2012); EMMS façades: `build_perspective_models(domain)`, `agent_model(name)`, `all_agents(n)`
- **`TrustLedger`** + **`TrustScore`** + **`TrustReport`** (`memory/trust.py`) — source credibility scoring per information domain; trust formula: `mean_importance × 0.4 + valence_stability × 0.4 + count_score × 0.2`; `valence_stability = 1 − std(valences)` clamped 0..1; `count_score = min(1.0, count / 10)`; `TrustScore` carries `source`, `trust`, `memory_count`, `mean_importance`, `valence_stability`; `TrustReport` sorted by trust; `trust_of(source)` returns 0.5 neutral if unknown; `most_trusted(n)` / `least_trusted(n)`; `min_memories` filter; biological analogue: source monitoring (Johnson et al. 1993); vmPFC source value computation (Behrens et al. 2008); trust as Bayesian belief updating (Fogg 2003); persuasion credibility cues (Petty & Cacioppo 1986); EMMS façades: `compute_trust(domain)`, `trust_of(source)`, `most_trusted(n)`
- **`NormExtractor`** + **`SocialNorm`** + **`NormReport`** (`memory/norms.py`) — behavioural norm extraction from memory content; prescriptive keywords: `should`, `must`, `ought`, `always`, `expected`, `appropriate`, `acceptable`, `required`, `standard`, `recommended`; prohibitive keywords: `never`, `forbidden`, `inappropriate`, `unacceptable`, `prohibited`, `avoid`; first meaningful token after keyword = `subject`; excerpt around keyword = `content`; `confidence = freq_ratio × 0.6 + mean_importance × 0.4`; SocialNorm IDs prefixed `norm_`; `NormReport` with `prescriptive_count + prohibitive_count = total_norms`; `norms_for_domain(domain)` → filtered list; `check_norm(behavior)` → Jaccard overlap between behaviour tokens and norm content+subject+keyword (up to 5 results); `min_norm_frequency` filter; `max_norms` cap; biological analogue: social norm learning from observation (Fehr & Gächter 2002); ACC norm violation detection (Berns et al. 2012); insula in moral processing (Damasio 1994); normative convention learning (Tomasello 1999); norm internalisation (Elster 1989); EMMS façades: `extract_norms(domain)`, `norms_for_domain(domain)`, `check_norm(behavior)`
- **5 MCP tools** (92 total): `emms_build_perspectives`, `emms_agent_model`, `emms_compute_trust`, `emms_extract_norms`, `emms_check_norm`
- **5 CLI commands** (96 total): `build-perspectives`, `agent-model`, `compute-trust`, `extract-norms`, `check-norm`

---

## [0.20.0] — 2026-02-22

### Added
- **`CausalMapper`** + **`CausalEdge`** + **`CausalPath`** + **`CausalReport`** (`memory/causal.py`) — directed causal graph extraction from memory content; scans memory text for relational keywords (`causes`, `enables`, `produces`, `prevents`, `reduces`, `increases`, `requires`, `triggers`, `inhibits`, `leads`, `results`, `improves`, `damages`, `strengthens`, `weakens`); extracts source→target concept pairs (word before keyword = source, first meaningful word after = target); builds `_graph: dict[str, dict[str, CausalEdge]]`; edge `strength = count / n_items_in_domain`; filters below `min_strength`; `build(domain)` → `CausalReport` with `total_concepts`, `total_edges`, `most_influential` (top out-degree), `most_affected` (top in-degree), `edges` sorted by strength desc; `effects_of(concept)` → outgoing edges; `causes_of(concept)` → incoming edges; `causal_path(source, target)` → BFS shortest path as `CausalPath` or `None`; `most_influential(n)` → top-n by out-degree; biological analogue: causal model theory (Pearl 2000) — cognition as causal inference; hippocampal–PFC causal reasoning (Kumaran et al. 2016); brain as causal generative model (Tenenbaum et al. 2011); EMMS façades: `build_causal_map(domain)`, `effects_of(concept)`, `causes_of(concept)`
- **`CounterfactualEngine`** + **`Counterfactual`** + **`CounterfactualReport`** (`memory/counterfactual.py`) — "what if" alternatives to past experiences; upward counterfactuals for items with `valence < 0.1`: "What if [excerpt] had led to a more positive outcome? If '{top_token}' had been approached differently, {domain} outcomes might have been more constructive and successful."; `valence_shift = +0.4`; downward counterfactuals for items with `valence > −0.1`: encounters with obstacles framing; `valence_shift = −0.4`; `plausibility = memory_strength × 0.8`; IDs prefixed `cf_`; optional `store_results` persists as `"counterfactual"` domain memories; `generate(domain, direction)` → `CounterfactualReport` sorted by `|valence_shift|` desc; `upward(n)` / `downward(n)` / `for_memory(memory_id)` retrieval; biological analogue: counterfactual thinking as adaptive cognition (Roese 1997); upward counterfactuals motivate future behaviour (Markman et al. 1993); orbitofrontal cortex in counterfactual valuation (Camille et al. 2004); medial PFC mental simulation (Buckner & Carroll 2007); EMMS façades: `generate_counterfactuals(domain, direction)`, `upward_counterfactuals(n)`, `downward_counterfactuals(n)`
- **`SkillDistiller`** + **`DistilledSkill`** + **`SkillReport`** (`memory/skills.py`) — reusable procedural skill extraction from recurring action patterns; action tokens: `improve`, `reduce`, `increase`, `enable`, `build`, `create`, `learn`, `apply`, `practice`, `develop`, `analyze`, `optimize`, `implement`, `design`, `test`, `solve`, `train`, `strengthen`, `generate`, `construct`, `establish`; for each action token in memory text: preconditions = up to 3 meaningful tokens before; outcomes = up to 3 meaningful tokens after; groups by `(domain, action_token)`; produces `DistilledSkill` if frequency ≥ `min_skill_frequency`; `confidence = freq_ratio × 0.6 + mean_strength × 0.4`; description template: `"Skill '{name}' in {domain}: apply {name} using [{preconditions}] to achieve [{outcomes}]. Observed in {n} memories."`; IDs prefixed `skill_`; optional `store_skills` persists as `"skill"` domain memories; `best_skill(goal_description)` → highest token Jaccard overlap between goal tokens and `preconditions + outcomes + [name]`; biological analogue: procedural learning and skill automatisation (Fitts 1964); basal ganglia in habit and skill formation (Graybiel 2008); episodic-to-procedural memory transfer (Cohen & Squire 1980); chunking of procedural sequences (Sakai et al. 2004); EMMS façades: `distill_skills(domain)`, `best_skill(goal_description)`
- **5 MCP tools** (87 total): `emms_build_causal_map`, `emms_effects_of`, `emms_generate_counterfactuals`, `emms_distill_skills`, `emms_best_skill`
- **5 CLI commands** (91 total): `build-causal-map`, `effects-of`, `generate-counterfactuals`, `distill-skills`, `best-skill`

---

## [0.19.0] — 2026-02-22

### Added
- **`EmotionalRegulator`** + **`EmotionalState`** + **`ReappraisalResult`** + **`EmotionReport`** (`memory/emotion.py`) — emotion regulation via cognitive reappraisal and mood-congruent retrieval; `regulate(domain)` computes `EmotionalState` from most recent `window_memories` items: `valence = mean(valences)`, `arousal = std(valences)` clamped 0..1, `dominant_domain = mode(domains)`; applies cognitive reappraisal to items with `valence < −0.3`: generates alternative growth-oriented framing, shifts `new_valence = original + 0.3` (clamped −1..1), optionally stores reappraisal as `"reappraisal"` domain memory; `mood_retrieve(k)` returns k memories ranked by `|item_valence − state.valence|` ascending (mood-congruent retrieval); `emotional_coherence()` = `1 − std(all_valences)` clamped 0..1; `EmotionReport` carries `current_state`, `reappraisals` list, `mood_congruent_ids`, `emotional_coherence`, `memories_assessed`; biological analogue: process model of emotion regulation (Gross 1998) — cognitive reappraisal is the most adaptive regulation strategy; mood-congruent memory (Bower 1981); amygdala–PFC interaction in affect regulation (Ochsner & Gross 2005); EMMS façades: `regulate_emotions(domain)`, `current_emotional_state()`, `mood_retrieve(k)`
- **`ConceptHierarchy`** + **`ConceptNode`** + **`HierarchyReport`** (`memory/hierarchy.py`) — taxonomic knowledge organisation from memory token co-occurrences; `build(domain)` pipeline: (1) token frequency across all memories (stop-word filtered, len≥4); (2) tokens in ≥`min_frequency` distinct memories → candidates; (3) top `max_concepts//3` by frequency → level-0 roots; (4) remaining tokens: find most-frequent root that co-occurs in same memory → level-1 child; (5) level-2+ by co-occurrence with level-1 nodes; `abstraction_score = freq / total_items`; `ConceptNode` carries `label`, `level`, `domain`, `parent_id`, `children_ids`, `supporting_memory_ids`, `abstraction_score`; `ancestors(label)` root-first BFS; `descendants(label)` BFS order; `concept_distance(a, b)` BFS hop count or −1; `most_abstract(n)` level-0 nodes by score; `most_specific(n)` deepest leaf nodes; `HierarchyReport` with `total_concepts`, `total_edges`, `max_depth`, `domains`; biological analogue: hierarchical semantic network (Collins & Quillian 1969) — verification time ∝ semantic distance; prototype theory (Rosch 1975) — graded category membership; taxonomic organisation (Rogers & McClelland 2004); EMMS façades: `build_concept_hierarchy(domain)`, `concept_distance(label_a, label_b)`
- **`SelfModel`** + **`Belief`** + **`SelfModelReport`** (`memory/self_model.py`) — explicit self-representation from recurring memory patterns; `update()` extracts per-domain beliefs: finds top frequency token per domain, computes `confidence = freq_ratio × 0.6 + mean_strength × 0.4`; generates prose belief: "In {domain}, a recurring pattern around '{token}' suggests: {domain} understanding is {strong|developing} (confidence={x}%, {n} memories)"; `capability_profile` = `domain → min(1, mean_strength × log1p(count) / log1p(5))` — rewards depth (strength) and breadth (count); `consistency_score` = `1 − std(belief valences)` — emotional coherence of self-model; `core_domains` = top-3 domains by memory count; `Belief` IDs prefixed `belief_`; `get_belief(id)` for lookup; `SelfModelReport` carries `beliefs`, `core_domains`, `dominant_valence`, `consistency_score`, `capability_profile`, `total_memories_analyzed`; biological analogue: self-referential processing in medial PFC (Northoff et al. 2006); self-schema (Markus 1977) — organised self-beliefs guide encoding and retrieval; autobiographical self (Damasio 1999); self-consistency motivation (Lecky 1945); EMMS façades: `update_self_model()`, `self_model_beliefs()`, `capability_profile()`
- **5 new MCP tools** (82 total): `emms_regulate_emotions`, `emms_current_emotion`, `emms_build_hierarchy`, `emms_concept_distance`, `emms_update_self_model`
- **5 new CLI commands** (86 total): `regulate-emotions`, `current-emotion`, `build-hierarchy`, `concept-distance`, `update-self-model`
- **10 new exports** in `__init__.py`: `EmotionalRegulator`, `EmotionalState`, `ReappraisalResult`, `EmotionReport`, `ConceptHierarchy`, `ConceptNode`, `HierarchyReport`, `SelfModel`, `Belief`, `SelfModelReport`

---

## [0.18.0] — 2026-02-22

### Added
- **`PredictiveEngine`** + **`Prediction`** + **`PredictionReport`** (`memory/prediction.py`) — predictive coding for memory: extracts recurring token patterns per domain, generates forward-looking `Prediction` objects with a `confidence` score (token frequency ratio), and tracks their outcomes; `predict(domain)` generates predictions from high-frequency memory themes; `resolve(prediction_id, outcome, note)` sets outcome to `"confirmed"` or `"violated"` and computes `surprise_score` (confirmed → `1 − confidence`, violated → `confidence`); `pending_predictions()` returns unresolved predictions; `surprise_profile()` returns per-domain mean surprise; prediction IDs are prefixed `pred_`; `PredictionReport` carries `total_generated`, `confirmed`, `violated`, `pending`, `mean_surprise`, `predictions` list, `duration_seconds`; biological analogue: predictive processing framework (Rao & Ballard 1999); precision-weighted prediction error (Friston 2005) — the brain constantly predicts upcoming sensory input and updates models based on the difference; EMMS façades: `predict(domain)`, `resolve_prediction(prediction_id, outcome, note)`, `pending_predictions()`
- **`ConceptBlender`** + **`BlendedConcept`** + **`BlendReport`** (`memory/blending.py`) — conceptual integration network for creative synthesis; pairs memories from different domains, computes `blend_strength = shared_ratio × coherence` (coherence = `min(1, importance_a × importance_b × 4)`); extracts `emergent_properties` as tokens unique to each source (selective projection of non-shared structure); optionally stores each blend as a new `"blend"` domain `Experience`; `blend(domain_a, domain_b)` iterates cross-domain pairs up to `max_blends`; `blend_pair(memory_id_a, memory_id_b)` for direct blending of specific memories; `BlendedConcept` carries `source_memory_ids`, `source_domains`, `blend_content`, `emergent_properties`, `blend_strength`, `new_memory_id`; concept IDs prefixed `blend_`; `BlendReport.summary()` lists top blends; biological analogue: Conceptual Integration Network / blending theory (Fauconnier & Turner 2002) — novel concepts emerge from structured mapping between input mental spaces with emergent properties not in either input; EMMS façades: `blend_concepts(domain_a, domain_b)`, `blend_pair(memory_id_a, memory_id_b)`
- **`TemporalProjection`** + **`FutureScenario`** + **`ProjectionReport`** (`memory/projection.py`) — episodic future thinking and scenario simulation; two generation paths: (1) **memory-based** — groups memories by domain, extracts top token, computes `plausibility = freq × 0.6 + mean_strength × 0.4`; (2) **episode-based** (when `EpisodicBuffer` attached) — extracts closed episode outcomes, computes `plausibility = 0.5 + 0.05 × min(turns, 10)`; deduplicates by 40-char content prefix; sorts by plausibility descending; `FutureScenario` carries `domain`, `content`, `basis_episode_ids`, `basis_memory_ids`, `projection_horizon`, `plausibility`, `emotional_valence`; scenario IDs prefixed `proj_`; `project(domain, horizon_days)` generates and accumulates scenarios; `most_plausible(n)` returns top-N from all prior calls; `ProjectionReport` includes `total_episodes_used`, `total_memories_used`, `scenarios_generated`, `mean_plausibility`; biological analogue: episodic future thinking (Atance & O'Neill 2001); mental time travel (Tulving 1985); hippocampal scene construction (Hassabis & Maguire 2007); Default Mode Network as shared substrate for past recall and future simulation (Buckner & Carroll 2007); EMMS façades: `project_future(domain, horizon_days)`, `most_plausible_futures(n)`
- **5 new MCP tools** (77 total): `emms_predict`, `emms_pending_predictions`, `emms_blend_concepts`, `emms_project_future`, `emms_plausible_futures`
- **5 new CLI commands** (81 total): `predict`, `pending-predictions`, `blend-concepts`, `project-future`, `plausible-futures`
- **9 new exports** in `__init__.py`: `PredictiveEngine`, `Prediction`, `PredictionReport`, `ConceptBlender`, `BlendedConcept`, `BlendReport`, `TemporalProjection`, `FutureScenario`, `ProjectionReport`

---

## [0.17.0] — 2026-02-22

### Added
- **`GoalStack`** + **`Goal`** + **`GoalReport`** (`memory/goals.py`) — hierarchical goal management with full lifecycle tracking; goals carry priority (0..1), status (pending/active/completed/failed/abandoned), optional parent_id for sub-goal decomposition, deadline (unix timestamp), and supporting_memory_ids; lifecycle methods: `push()` creates with uuid id, `activate()` transitions pending → active, `complete(outcome_note)` / `fail(reason)` / `abandon(reason)` transition to terminal states with resolved_at timestamp; `active_goals()` returns sorted-by-priority list; `pending_goals()` for queued work; `sub_goals(goal_id)` returns direct children; `get(goal_id)` for lookup; `report()` returns `GoalReport` with per-status counts; biological analogue: hierarchical task decomposition in prefrontal cortex (Koechlin & Summerfield 2007), sustained goal maintenance (Miller & Cohen 2001); EMMS façades: `push_goal`, `activate_goal`, `complete_goal`, `fail_goal`, `active_goals`, `goal_report`
- **`AttentionFilter`** + **`AttentionResult`** + **`AttentionReport`** (`memory/attention.py`) — selective attention spotlight modulating memory retrieval; spotlight built from three sources: free text (tokenised with stop-word filtering), explicit keyword list, and active goal content from attached GoalStack; `update_spotlight(text, goal_ids, keywords)` expands the spotlight; `spotlight_retrieve(k)` scores all memories and returns top-k; attention score formula: `0.40 * goal_relevance + 0.30 * importance + 0.20 * keyword_overlap + 0.10 * recency_score` where recency = 1/(1+age_days); `attention_profile()` returns domain → mean_score dict; `clear_spotlight()` resets focus; `AttentionReport` includes spotlight_keywords, spotlight_goal_ids, items_scored, top_domain; biological analogue: spotlight model (Posner 1980), biased competition model (Desimone & Duncan 1995) — goal representations in PFC provide top-down bias; EMMS façades: `update_spotlight`, `spotlight_retrieve`, `attention_profile`
- **`AnalogyEngine`** + **`AnalogyMapping`** + **`AnalogyRecord`** + **`AnalogyReport`** (`memory/analogy.py`) — structural analogy detection across memory domains; detects cross-domain pairs sharing *relational* structure rather than topic overlap; 34-word relational vocabulary covering causal (causes/enables/prevents/requires), temporal (follows/produces), and quantitative (increases/reduces/triggers/inhibits) connectives; structural_similarity = 0.7 × relational_jaccard + 0.3 × content_jaccard; only cross-domain pairs considered (same-domain handled by BeliefReviser); `find_analogies(source_domain, target_domain)` enumerates domain pairs, caps at 10 items/domain, keeps top-3 mappings per pair, optionally stores insight as "insight" domain memory; `AnalogyRecord` carries both domains, mappings list, analogy_strength (mean sim), and insight_content prose; `analogies_for(memory_id)` for per-memory lookup; `AnalogyReport` sorted by strength; biological analogue: Structure Mapping Theory (Gentner 1983), analogical reminding (Holyoak & Thagard 1995) — structural similarity drives which past cases are retrieved for cross-domain transfer; EMMS façades: `find_analogies`, `analogies_for`
- **5 new MCP tools** (72 total): `emms_push_goal`, `emms_active_goals`, `emms_complete_goal`, `emms_spotlight_retrieve`, `emms_find_analogies`
- **5 new CLI commands** (76 total): `push-goal`, `active-goals`, `complete-goal`, `spotlight-retrieve`, `find-analogies`
- **10 new exports** in `__init__.py`: `GoalStack`, `Goal`, `GoalReport`, `AttentionFilter`, `AttentionResult`, `AttentionReport`, `AnalogyEngine`, `AnalogyMapping`, `AnalogyRecord`, `AnalogyReport`

---

## [0.16.0] — 2026-02-22

### Added
- **`CuriosityEngine`** + **`ExplorationGoal`** + **`CuriosityReport`** (`memory/curiosity.py`) — epistemic curiosity and knowledge-gap detection as a first-class cognitive process; scans the memory store per domain and scores four gap types: **sparse** (fewer than `sparse_threshold` memories → urgency 0.5–0.7), **novel** (exactly one memory → urgency 0.75), **uncertain** (mean confidence below `uncertain_threshold` → urgency = 1 − mean_conf), **contradictory** (valence-opposing pairs with Jaccard token overlap ≥ 0.25 → urgency capped at 1.0); integrates with `MetacognitionEngine` when available for accurate per-memory confidence scores; selects question templates per gap type from a curated bank of three questions each, indexed by number of items; stores generated `ExplorationGoal` objects in a session dict; `pending_goals()` filters un-explored goals; `mark_explored(goal_id)` marks fulfilment; `CuriosityReport` includes `total_domains_scanned`, `goals_generated`, `goals` (sorted by urgency descending), `top_curious_domains`; biological analogue: information-gap theory of curiosity (Loewenstein 1994) — curiosity arises from a perceived gap between what is known and what one feels should be known; EMMS façades: `curiosity_scan(domain)`, `exploration_goals()`, `mark_explored(goal_id)`
- **`BeliefReviser`** + **`RevisionRecord`** + **`RevisionReport`** (`memory/belief_revision.py`) — AGM-inspired belief revision when contradictions arise; detects conflicts via token Jaccard overlap (Jaccard ≥ `overlap_threshold`, default 0.30) combined with opposing emotional valence (|Δvalence| ≥ `valence_conflict_threshold`, default 0.40); conflict_score = overlap × valence_diff; chooses strategy by conflict_score and strength_ratio: **supersede** (score ≥ 0.5 AND |ΔS|/(S_a+S_b) ≥ 0.3 — weakens loser to 50%, sets `superseded_by`), **merge** (score ≥ 0.3 — creates reconciliation `Experience` in same domain, stores via `memory.store()`), **flag** (weak conflict — records without changes); full pairwise scan capped at 60 items to avoid O(N²) explosion; trigger-specific mode when `new_memory_id` provided; `revision_history()` returns all session records newest-first; biological analogue: AGM postulates (Alchourrón, Gärdenfors & Makinson 1985) and cognitive dissonance reduction (Festinger 1957); EMMS façades: `revise_beliefs(new_memory_id, domain, max_revisions)`, `revision_history()`
- **`MemoryDecay`** + **`DecayRecord`** + **`DecayReport`** (`memory/decay.py`) — Ebbinghaus forgetting curve applied to memory strength; retention R = e^{−t/S} where t = days since last access (`last_accessed` or `stored_at`) and S = `base_stability` + `retrieval_boost` × `access_count`; models the spacing effect: each retrieval increases stability, making memories more durable; **`decay(domain)`** computes retention without modifying any state (safe read-only simulation); **`apply_decay(domain, prune)`** mutates `memory_strength` proportionally to R; optional pruning when `new_strength < prune_threshold` (removes from working/short_term deques and long_term/semantic dicts); `DecayReport` carries `total_processed`, `decayed`, `pruned`, `mean_retention`, `applied`, top-N records sorted by most-decayed first; single-item `retention(item) → (R, S)` for external use; biological analogue: Ebbinghaus forgetting curve (1885) and the spacing effect (Cepeda et al. 2006); EMMS façades: `memory_decay_report(domain)`, `apply_memory_decay(domain, prune)`
- **5 new MCP tools** (67 total): `emms_curiosity_report`, `emms_exploration_goals`, `emms_revise_beliefs`, `emms_decay_report`, `emms_apply_decay`
- **5 new CLI commands** (71 total): `curiosity-report`, `explore-goals`, `revise-beliefs`, `decay-report`, `apply-decay`
- **9 new exports** in `__init__.py`: `CuriosityEngine`, `ExplorationGoal`, `CuriosityReport`, `BeliefReviser`, `RevisionRecord`, `RevisionReport`, `MemoryDecay`, `DecayRecord`, `DecayReport`

---

## [0.15.0] — 2026-02-22

### Added
- **`ReflectionEngine`** + **`Lesson`** + **`ReflectionReport`** (`memory/reflection.py`) — structured self-review closing the cognitive learning loop; reviews high-importance memories (configurable `min_importance` threshold) and recent episodes (via optional `EpisodicBuffer`); tokenises memory content with stop-word filtering (same logic as `SchemaExtractor`); groups memories by shared keyword clusters; chooses lesson type based on emotional valence spread (contrast = spread > 0.6, gap = small group, principle = neutral large group, pattern = default); synthesises a concise human-readable lesson per cluster via position-indexed templates; stores each lesson as a `"reflection"` domain `Experience` for persistence; generates open questions from lesson keywords using five question templates; `ReflectionReport` carries `lessons`, `open_questions`, `new_memory_ids`, `memories_reviewed`, `episodes_reviewed`, `duration_seconds`; biological analogue: Default Mode Network activation during rest (Andrews-Hanna 2012) — the brain's self-referential consolidation of experience into autobiographical knowledge; EMMS façades: `reflect(session_id, domain, lookback_episodes)`, `enable_reflection(min_importance, max_lessons)`
- **`NarrativeWeaver`** + **`NarrativeSegment`** + **`NarrativeThread`** + **`NarrativeReport`** (`memory/narrative.py`) — autobiographical narrative construction; groups memories by domain, sorts chronologically by `stored_at`; divides each domain's timeline into `max_segment_memories`-item chunks; generates position-aware prose for each chunk using opening / middle / closing / single templates with highest-importance memory excerpt; assembles segments into `NarrativeThread` with emotional arc, `span_seconds`, and dominant theme keyword; annotates segments with `episode_ids` from `EpisodicBuffer` when available; `NarrativeThread.story()` joins all segment prose into a single readable string; threads sorted by segment count descending; filters domains with fewer than `min_thread_length` memories; biological analogue: left-hemisphere "interpreter" (Gazzaniga 1989); autobiographical narrative self (Conway & Pleydell-Pearce 2000); EMMS façades: `weave_narrative(domain, max_threads)`, `narrative_threads(domain)`
- **`SourceMonitor`** + **`SourceTag`** + **`SourceAuditEntry`** + **`SourceReport`** (`memory/source_monitor.py`) — memory provenance tracking and confabulation detection; seven source types: `observation`, `inference`, `instruction`, `reflection`, `dream`, `insight`, `unknown`; `tag(memory_id, source_type, confidence, note)` assigns explicit provenance; `get_tag(memory_id)` returns stored tag; `auto_tag()` applies heuristics to all untagged memories (domain `"dream"` → `"dream"` 0.90, domain `"insight"` → `"insight"` 0.85, domain `"reflection"` → `"reflection"` 0.90, keyword hints → `"reflection"` 0.70 / `"instruction"` 0.65 / `"inference"` 0.65, importance ≥ 0.85 → `"instruction"` 0.60, default → `"observation"` 0.50); `audit(max_flagged)` scans all memories, auto-infers source for untagged items, flags those with `confidence < flag_threshold` or `source_type == "unknown"`; `source_profile()` returns `{source_type: count}` sorted descending; tags are memory-resident (not auto-persisted); biological analogue: Source Monitoring Framework (Johnson, Hashtroudi & Lindsay 1993) — prefrontal source attribution; damage causes source amnesia and confabulation; EMMS façades: `enable_source_monitoring()`, `tag_memory_source(memory_id, source_type, confidence, note)`, `source_audit(flag_threshold)`, `source_profile()`
- **CLI commands** (5 new, 66 total): `reflect`, `weave-narrative`, `narrative-threads`, `source-audit`, `tag-source`
- **MCP tools** (5 new, 62 total): `emms_reflect`, `emms_weave_narrative`, `emms_narrative_threads`, `emms_source_audit`, `emms_tag_source`
- 90 new tests in `tests/test_v150_features.py`; total: **~1411 passed, 2 skipped**

---

## [0.14.0] — 2026-02-22

### Added
- **`EpisodicBuffer`** + **`Episode`** (`memory/episodic.py`) — structured session-as-episode storage; `Episode` objects carry temporal boundaries (`opened_at`, `closed_at`), `emotional_arc` (valence per recorded turn), `key_memory_ids`, `turn_count`, and `outcome`; `open_episode(session_id, topic)` auto-closes any currently open episode and seeds a fresh one; `record_turn(content, valence)` appends to the arc (valence clipped to −1..1); `close_episode(outcome)` finalises `mean_valence` and `peak_valence`; `add_memory(id)` links memory IDs to the active episode; `recent_episodes(n)` returns newest-first; `save(path)` / `load(path)` JSON persistence; biological analogue: hippocampal episodic memory (Tulving 1972) — "mental time travel"; EMMS façades: `open_episode`, `close_episode`, `record_episode_turn`, `recent_episodes`, `current_episode`
- **`SchemaExtractor`** + **`Schema`** + **`SchemaReport`** (`memory/schema.py`) — abstract knowledge schema extraction from concrete memories; tokenises memory content, strips stop-words (50-word frozenset) and short tokens; builds keyword frequency table across memories; seeds on keywords appearing in ≥ `min_support` distinct memories (default 2); greedy cluster assignment by most-frequent shared keyword; computes shared keywords within each cluster; synthesises concise pattern description from three templates; `Schema` carries `domain`, `pattern`, `keywords`, `confidence` (support fraction), `supporting_memory_ids`; `SchemaReport.summary()` lists top schemas; biological analogue: schema theory (Bartlett 1932) — abstract generalisation from repeated episodic experience into semantic knowledge; EMMS façade: `extract_schemas(domain, max_schemas)`
- **`MotivatedForgetting`** + **`ForgettingResult`** + **`ForgettingReport`** (`memory/forgetting.py`) — goal-directed active memory suppression; `suppress(memory_id)` finds a memory across all tiers and multiplies its strength by `(1 − suppression_rate)` (default 0.4); `forget_domain(domain, rate)` applies suppression to all memories in a named domain; `forget_below_confidence(threshold, metacognition_engine)` uses `MetacognitionEngine.assess_all()` when available (otherwise falls back to raw `memory_strength`) to evict low-confidence memories; `resolve_contradiction(weaker_id)` suppresses the losing side of a detected contradiction pair; items whose strength after suppression falls below `prune_threshold` (default 0.05) are evicted from their tier entirely; `ForgettingReport` carries `total_targeted`, `suppressed`, `pruned`, per-item `results`, and `summary()`; biological analogue: prefrontal-hippocampal inhibitory control (Anderson & Green 2001); directed forgetting (Bjork 1972); EMMS façades: `forget_memory`, `forget_domain`, `forget_below_confidence`, `resolve_memory_contradiction`
- **CLI commands** (5 new, 61 total): `open-episode`, `close-episode`, `recent-episodes`, `extract-schemas`, `forget`
- **MCP tools** (5 new, 57 total): `emms_open_episode`, `emms_close_episode`, `emms_recent_episodes`, `emms_extract_schemas`, `emms_forget`
- 92 new tests in `tests/test_v140_features.py`; total: **1339 passed, 2 skipped**

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
