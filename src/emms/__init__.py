"""EMMS — Enhanced Memory Management System for AI Agents.

v0.24.0: The Wise Mind
- BiasDetector: 10 cognitive bias types detected via indicator vocabulary (confirmation_bias,
  availability_heuristic, sunk_cost, optimism_bias, negativity_bias, hindsight_bias,
  overconfidence, in_group_bias, anchoring, framing_effect); strength = (n_affected /
  n_total) × mean_importance; BiasInstance IDs prefixed "bia_"; BiasReport with
  dominant_bias, mean_strength; biases_of_type, most_pervasive; biological analogue:
  dual-process theory (Kahneman 2011), Tversky & Kahneman 1974, motivated reasoning
  (Kunda 1990), prefrontal debiasing (Fleming & Dolan 2012); EMMS façades: map_biases,
  biases_of_type, most_pervasive_bias
- WisdomSynthesizer: cross-system synthesis for a free-text query; Jaccard-similarity
  top-k memory retrieval; four dimensions: value signals (VALUE_TOKENS lexicon), moral
  patterns (C/D/V keyword counts), causal patterns (source→verb→target triples),
  recurring principles (≥2 doc-frequency tokens); confidence = n_active_dimensions/4;
  WisdomGuidance IDs prefixed "wis_"; template synthesis string; WisdomReport with
  dimensions_used, coverage_score; biological analogue: prefrontal-limbic integration
  (Meeks & Jeste 2009), DMN synthesis (Andrews-Hanna 2012), phronesis (Aristotle);
  EMMS façades: synthesize_wisdom(query)
- EpistemicEvolution: domain-level knowledge growth tracking; splits memories into
  early/recent halves by timestamp; growth_rate = net new vocabulary gain (−1..1);
  consolidation_score = Jaccard token overlap (0..1); knowledge_density = memories/day
  normalised 0..1; recent_themes = top 5 tokens in recent half; knowledge_gaps =
  domains with < min_memories; KnowledgeDomain per domain; EvolutionReport sorted by
  density desc with most_active_domain, most_consolidated_domain, overall_growth_rate;
  biological analogue: hippocampal-neocortical transfer (McClelland et al. 1995), power
  law of practice (Newell & Rosenbloom 1981), expert-novice restructuring (Chi et al.
  1982); EMMS façades: evolve_knowledge, domain_knowledge_profile, knowledge_gaps,
  most_active_domain, most_consolidated_domain
- MCP tools (5 new): emms_detect_biases, emms_most_pervasive_bias,
  emms_synthesize_wisdom, emms_evolve_knowledge, emms_knowledge_gaps (107 total)
- CLI commands (5 new): detect-biases, most-pervasive-bias, synthesize-wisdom,
  evolve-knowledge, knowledge-gaps (111 total)

v0.23.0: The Moral Mind
- ValueMapper: core value extraction from memory via 5-category value lexicon
  (epistemic, moral, aesthetic, instrumental, social); strength = mean_importance
  × freq_ratio × 5.0 clamped 0..1; MappedValue IDs prefixed "val_"; ValueReport
  with dominant_category, mean_strength; values_for_category; strongest_value;
  biological analogue: value-based decision-making OFC (Rangel et al. 2008),
  personal value schemas (Schwartz 1992); EMMS façades: map_values,
  values_for_category, strongest_value
- MoralReasoner: three-framework ethical evaluation (consequentialist, deontological,
  virtue); framework_score = matching_tokens/total_tokens; dominant_framework =
  argmax (or "none" if all < 0.01); moral_weight = importance × |valence| ×
  (max_score + 0.1) clamped 0..1; MoralAssessment per memory; MoralReport with
  framework_counts, dominant_framework_overall; assessments_by_framework;
  moral_weight_of (0.0 default if not assessed); biological analogue: dual-process
  moral cognition (Greene 2008), vmPFC/TPJ moral judgment (Greene et al. 2001),
  moral foundations theory (Haidt 2012); EMMS façades: reason_morally,
  moral_weight_of, assessments_by_framework
- DilemmaEngine: ethical tension detection; pairs same-domain memories with
  moral_weight ≥ min_tension and |valence_a − valence_b| > 0.5; tension_score =
  mw_a × mw_b × |val_diff| clamped 0..1; resolution strategies from template table
  keyed by (framework_a, framework_b); EthicalDilemma IDs prefixed "dil_";
  DilemmaReport with domains_affected, mean_tension; dilemmas_for_domain;
  most_tense_dilemma; biological analogue: trolley-problem studies (Greene et al.
  2001), ACC conflict monitoring (Botvinick et al. 2001), dual-process tension
  (Cushman & Young 2011); EMMS façades: detect_dilemmas, dilemmas_for_domain,
  most_tense_dilemma
- MCP tools (5 new): emms_map_values, emms_values_for_category, emms_reason_morally,
  emms_detect_dilemmas, emms_most_tense_dilemma (102 total)
- CLI commands (5 new): map-values, values-for-category, reason-morally,
  detect-dilemmas, most-tense-dilemma (106 total)

v0.22.0: The Creative Mind
- NoveltyDetector: memory novelty scoring against corpus centroid; builds
  global token-document-frequency counter from all memory content; rarity
  threshold = max(2, 10% of total items); novelty = fraction of a memory's
  tokens that are rare in the corpus (0..1); NoveltyScore with memory_id,
  content_excerpt, novelty, domain, rare_tokens; NoveltyReport sorted by
  novelty descending; novelty_of returns 0.5 neutral if not yet assessed;
  biological analogue: hippocampal novelty detection (Kumaran & Maguire 2007),
  dopaminergic prediction-error signal (Schultz 1998), locus coeruleus
  norepinephrine response; EMMS façades: assess_novelty, most_novel, novelty_of
- ConceptInventor: cross-domain concept generation; per-domain rare tokens
  (≤ 20% frequency); cross-pairs tokens from different domains; originality =
  (1 − freq_a/max_freq) × 0.5 + (1 − freq_b/max_freq) × 0.5; template
  description: "What if {a} (from {dom_a}) could {b}?"; InventedConcept IDs
  prefixed "inv_"; InventionReport with domain_pairs, mean_originality;
  concepts_for_domain; best_concept via Jaccard; biological analogue: divergent
  thinking (Guilford 1967), default mode network (Beaty et al. 2016), conceptual
  blending (Fauconnier & Turner 2002); EMMS façades: invent_concepts, best_concept
- AbstractionEngine: episodic-to-principle lifting; recurring tokens (≥ 2
  memories) scored by generality = count/n_domain_memories; min_generality
  threshold; AbstractPrinciple with label, domain, description, generality_score,
  mean_valence, mean_importance, source_memory_ids; IDs prefixed "abs_";
  AbstractionReport sorted by generality descending; principles_for_domain;
  best_principle via Jaccard; biological analogue: schema abstraction (Bartlett
  1932), prototype theory (Rosch 1975), PFC hierarchical abstraction (Badre &
  Frank 2012); EMMS façades: abstract_principles, principles_for_domain,
  best_principle
- MCP tools (5 new): emms_assess_novelty, emms_most_novel, emms_invent_concepts,
  emms_abstract_principles, emms_best_principle (97 total)
- CLI commands (5 new): assess-novelty, most-novel, invent-concepts,
  abstract-principles, best-principle (101 total)

v0.21.0: The Social Mind
- PerspectiveTaker: Theory of Mind — extracts mental models of other agents from
  memory content; scans for belief/communication verbs (said, thinks, believes,
  wants, expects, argues, claims, suggests, reports, needs, hopes, fears, knows,
  decided, prefers, stated, noted, agreed, denied); token before verb = agent name;
  tokens after = attributed statement; AgentModel with mentions, statements,
  mean_valence, domains; PerspectiveReport sorted by mentions; take_perspective
  (name lookup); all_agents(n); biological analogue: Theory of Mind (Premack &
  Woodruff 1978), TPJ in belief representation (Saxe & Kanwisher 2003), mPFC
  self-other distinction (Mitchell 2009); EMMS façades: build_perspective_models,
  agent_model, all_agents
- TrustLedger: source credibility scoring per domain; trust = mean_importance × 0.4
  + valence_stability × 0.4 + count_score × 0.2; valence_stability = 1 − std(valences)
  clamped 0..1; count_score = min(1.0, count/10); TrustScore with source, trust,
  memory_count, mean_importance, valence_stability; TrustReport sorted by trust;
  trust_of returns 0.5 neutral if unknown; most_trusted(n) / least_trusted(n);
  biological analogue: source monitoring (Johnson et al. 1993), vmPFC source value
  (Behrens et al. 2008), Bayesian trust updating (Fogg 2003); EMMS façades:
  compute_trust, trust_of, most_trusted
- NormExtractor: behavioural norm extraction from memory; prescriptive keywords
  (should, must, ought, always, expected, appropriate, acceptable, required,
  standard, recommended); prohibitive keywords (never, forbidden, inappropriate,
  unacceptable, prohibited, avoid); first meaningful token after keyword = subject;
  confidence = freq_ratio × 0.6 + mean_importance × 0.4; SocialNorm IDs prefixed
  "norm_"; NormReport with prescriptive_count + prohibitive_count; norms_for_domain
  (domain filter); check_norm (Jaccard overlap); biological analogue: social norm
  learning (Fehr & Gächter 2002), ACC norm violation detection (Berns et al. 2012),
  social convention learning (Tomasello 1999); EMMS façades: extract_norms,
  norms_for_domain, check_norm
- MCP tools (5 new): emms_build_perspectives, emms_agent_model, emms_compute_trust,
  emms_extract_norms, emms_check_norm (92 total)
- CLI commands (5 new): build-perspectives, agent-model, compute-trust,
  extract-norms, check-norm (96 total)

v0.20.0: The Reasoning Mind
- CausalMapper: directed causal graph extraction from memory content; scans for
  relational keywords (causes, enables, produces, prevents, reduces, increases,
  requires, triggers, inhibits, leads, results, improves, damages, strengthens,
  weakens); word-before = source, first meaningful word-after = target; edge
  strength = co-occurrence count / total_memories; BFS causal_path; effects_of
  and causes_of for forward/backward inference; most_influential by out-degree;
  CausalReport with total_concepts, total_edges, most_influential, most_affected,
  edges; biological analogue: causal model theory (Pearl 2000), hippocampal-PFC
  causal reasoning (Kumaran 2016), brain as causal generative model (Tenenbaum
  2011); EMMS façades: build_causal_map, effects_of, causes_of
- CounterfactualEngine: "what if" alternatives to past experiences; upward
  counterfactuals (valence < 0.1) — "could have been better", valence_shift +0.4;
  downward counterfactuals (valence > -0.1) — "could have been worse",
  valence_shift -0.4; plausibility = memory_strength × 0.8; IDs prefixed "cf_";
  optional store_results as "counterfactual" domain memories; CounterfactualReport
  sorted by |valence_shift|; upward(n)/downward(n)/for_memory() retrieval;
  biological analogue: counterfactual thinking (Roese 1997), upward counterfactuals
  motivate future behaviour (Markman 1993), orbitofrontal cortex (Camille 2004);
  EMMS façades: generate_counterfactuals, upward_counterfactuals, downward_counterfactuals
- SkillDistiller: extracts reusable procedural skills from recurring action patterns;
  action tokens (improve, reduce, increase, enable, build, create, learn, apply,
  practice, develop, analyze, optimize, implement, design, test, solve, train,
  strengthen, generate, construct, establish); preconditions = up to 3 meaningful
  tokens before action; outcomes = up to 3 after; confidence = freq_ratio × 0.6 +
  mean_strength × 0.4; min_skill_frequency filter; best_skill via token Jaccard
  overlap; skill IDs prefixed "skill_"; optional store_skills as "skill" domain
  memories; SkillReport with skills sorted by confidence; biological analogue:
  procedural learning (Fitts 1964), basal ganglia skill formation (Graybiel 2008),
  episodic-to-procedural transfer (Cohen & Squire 1980), chunking (Sakai 2004);
  EMMS façades: distill_skills, best_skill
- MCP tools (5 new): emms_build_causal_map, emms_effects_of,
  emms_generate_counterfactuals, emms_distill_skills, emms_best_skill (87 total)
- CLI commands (5 new): build-causal-map, effects-of, generate-counterfactuals,
  distill-skills, best-skill (91 total)

v0.19.0: The Integrated Mind
- EmotionalRegulator: tracks emotional state (valence, arousal, dominant_domain)
  from the most recent window of memories; cognitive reappraisal for memories with
  valence < -0.3 — generates alternative growth-oriented framing and shifts valence
  +0.3; mood-congruent retrieval returns memories closest to current emotional state;
  emotional_coherence = 1 - std(all_valences); optionally stores reappraisals as
  "reappraisal" domain memories; EmotionReport with current_state, reappraisals list,
  mood_congruent_ids, and coherence; biological analogue: process model of emotion
  regulation (Gross 1998), mood-congruent memory (Bower 1981), amygdala-PFC
  interaction (Ochsner & Gross 2005); EMMS façades: regulate_emotions,
  current_emotional_state, mood_retrieve
- ConceptHierarchy: taxonomic knowledge organisation from memory token co-occurrence;
  candidates = tokens appearing in ≥ min_frequency memories; top 1/3 by frequency →
  level-0 roots; remaining tokens assigned to most-frequent co-occurring root →
  level-1 children; level-2+ by co-occurrence with level-1; abstraction_score =
  freq/total_items; ancestors(label) root-first; descendants(label) BFS; concept_
  distance(a, b) BFS hop count; most_abstract/most_specific; HierarchyReport with
  total_concepts, total_edges, max_depth, domains; biological analogue: semantic
  network (Collins & Quillian 1969), prototype theory (Rosch 1975), taxonomic
  organisation (Rogers & McClelland 2004); EMMS façades: build_concept_hierarchy,
  concept_distance
- SelfModel: explicit self-representation from recurring memory patterns; beliefs =
  top token per domain with confidence = freq_ratio * 0.6 + mean_strength * 0.4;
  capability_profile = domain → min(1, mean_strength * log1p(count) / log1p(5));
  consistency_score = 1 - std(belief valences) clamped 0..1; core_domains = top-3
  domains by memory count; SelfModelReport with beliefs, core_domains, dominant_
  valence, consistency_score, capability_profile; belief IDs prefixed "belief_";
  biological analogue: self-referential processing in mPFC (Northoff et al. 2006),
  self-schema (Markus 1977), autobiographical self (Damasio 1999); EMMS façades:
  update_self_model, self_model_beliefs, capability_profile
- MCP tools (5 new): emms_regulate_emotions, emms_current_emotion,
  emms_build_hierarchy, emms_concept_distance, emms_update_self_model (82 total)
- CLI commands (5 new): regulate-emotions, current-emotion, build-hierarchy,
  concept-distance, update-self-model (86 total)

v0.18.0: The Predictive Mind
- PredictiveEngine: forward model generating domain predictions from recurring
  memory patterns; token frequency analysis per domain; confidence proportional
  to pattern frequency ratio; resolve(prediction_id, outcome) marks confirmed or
  violated with automatic surprise_score (violated high-confidence = high surprise;
  confirmed = low surprise); pending_predictions() for unresolved items;
  surprise_profile() returns domain → mean_surprise; PredictionReport with
  total/confirmed/violated/pending counts; biological analogue: predictive coding
  (Rao & Ballard 1999), free energy principle (Friston 2010); EMMS façades:
  predict, resolve_prediction, pending_predictions
- ConceptBlender: Fauconnier & Turner conceptual blending of memory pairs; blend
  strength = shared_token_ratio × coherence where coherence = importance_a ×
  importance_b × 4; emergent_properties = unique tokens from each side (selective
  projection into blend space); blend_content prose naming shared + emergent
  structure; optionally stores blend as "insight" domain memory; blend() pairs
  across domain matrix; blend_pair(id_a, id_b) for targeted blending; BlendReport
  with sorted BlendedConcept list; biological analogue: conceptual integration
  theory (Fauconnier & Turner 2002), creative insight as structural integration
  (Dijksterhuis & Meurs 2006); EMMS façades: blend_concepts, blend_pair
- TemporalProjection: episodic future thinking from two paths — memory-based
  (token frequency + mean strength → plausibility) and episode-based (closed
  episode outcomes + turn_count → plausibility); FutureScenario with content,
  domain, horizon_days, plausibility (0..1), emotional_valence, basis ids;
  most_plausible(n) for top-n retrieval; ProjectionReport with episode/memory
  counts; biological analogue: episodic future thinking (Atance & O'Neill 2001),
  mental time travel (Tulving 1985), hippocampal scene construction (Hassabis &
  Maguire 2007); EMMS façades: project_future, most_plausible_futures
- MCP tools (5 new): emms_predict, emms_pending_predictions, emms_blend_concepts,
  emms_project_future, emms_plausible_futures (77 total)
- CLI commands (5 new): predict, pending-predictions, blend-concepts,
  project-future, plausible-futures (81 total)

v0.17.0: The Goal-Directed Mind
- GoalStack: hierarchical goal management with lifecycle tracking; goals have
  priority (0..1), status (pending/active/completed/failed/abandoned), optional
  parent_id for sub-goal formation, deadline, and supporting_memory_ids; lifecycle
  methods push/activate/complete/fail/abandon; active_goals() sorted by priority;
  sub_goals(goal_id) returns direct children; GoalReport with per-status counts;
  biological analogue: hierarchical task decomposition (Koechlin & Summerfield
  2007), prefrontal goal maintenance (Miller & Cohen 2001); EMMS façades:
  push_goal, activate_goal, complete_goal, fail_goal, active_goals, goal_report
- AttentionFilter: selective attention spotlight modulating memory retrieval;
  spotlight built from free text (tokenised), explicit keywords, and active goal
  IDs from attached GoalStack; attention_score = 0.40*goal_relevance +
  0.30*importance + 0.20*keyword_overlap + 0.10*recency_score; spotlight_retrieve(k)
  returns top-k items; attention_profile() returns domain → mean_score map;
  clear_spotlight() resets focus; AttentionReport with spotlight_keywords and
  top_domain; biological analogue: spotlight model (Posner 1980), biased
  competition (Desimone & Duncan 1995); EMMS façades: update_spotlight,
  spotlight_retrieve, attention_profile
- AnalogyEngine: structural analogy detection across domains; similarity =
  0.7*relational_jaccard + 0.3*content_jaccard where relational keywords are
  causal/temporal connectives (causes, enables, prevents, requires, follows, etc.);
  only cross-domain pairs considered; optionally stores each analogy as an "insight"
  domain memory; AnalogyRecord with source/target domains, AnalogyMappings, and
  insight_content; AnalogyReport sorted by strength; analogies_for(memory_id) for
  per-memory lookup; biological analogue: Structure Mapping Theory (Gentner 1983),
  analogical reminding (Holyoak & Thagard 1995); EMMS façades: find_analogies,
  analogies_for
- MCP tools (5 new): emms_push_goal, emms_active_goals, emms_complete_goal,
  emms_spotlight_retrieve, emms_find_analogies (72 total)
- CLI commands (5 new): push-goal, active-goals, complete-goal,
  spotlight-retrieve, find-analogies (76 total)

v0.16.0: The Curious Mind
- CuriosityEngine: epistemic curiosity and knowledge-gap detection; scans memory
  for sparse/uncertain/contradictory/novel domains; ranks gaps by urgency; generates
  ExplorationGoal objects (question, domain, urgency, gap_type, supporting_memory_ids);
  integrates with MetacognitionEngine for accurate confidence; CuriosityReport with
  goals sorted by urgency and top_curious_domains; EMMS façades: curiosity_scan,
  exploration_goals, mark_explored
- BeliefReviser: AGM-inspired belief revision when contradictions arise; detects
  conflicts via token Jaccard overlap + opposing emotional valence; three strategies:
  merge (synthesise reconciliation memory), supersede (weaken losing belief),
  flag (mark for review); RevisionReport with per-revision records; EMMS façades:
  revise_beliefs, revision_history
- MemoryDecay: Ebbinghaus forgetting curve (R = e^{-t/S}) applied to memory strength;
  stability S = base_stability + retrieval_boost × access_count (spacing effect);
  decay() computes only; apply_decay() mutates strength + optional pruning below
  prune_threshold; DecayReport with per-item retention; EMMS façades:
  memory_decay_report, apply_memory_decay
- MCP tools (5 new): emms_curiosity_report, emms_exploration_goals,
  emms_revise_beliefs, emms_decay_report, emms_apply_decay (67 total)
- CLI commands (5 new): curiosity-report, explore-goals, revise-beliefs,
  decay-report, apply-decay (71 total)

v0.15.0: The Reflective Mind
- ReflectionEngine: structured self-review closing the learning loop; reviews
  high-importance memories and recent episodes; groups by keyword clusters;
  synthesises Lesson objects (types: pattern/gap/contrast/principle) using
  template-based prose; stores each lesson as a "reflection" domain memory;
  generates open questions from prominent keywords; ReflectionReport with
  lessons, questions, new_memory_ids; EMMS façades: reflect, enable_reflection
- NarrativeWeaver: autobiographical narrative construction; groups memories by
  domain, sorts chronologically, divides into segments of ≤ max_segment_memories;
  generates position-aware prose (opening/middle/closing templates); assembles
  NarrativeThread per domain with emotional arc and span_seconds; annotates
  segments with episode_ids when EpisodicBuffer active; NarrativeReport with
  threads sorted by length; EMMS façades: weave_narrative, narrative_threads
- SourceMonitor: memory provenance tracking and confabulation detection; 7 source
  types (observation/inference/instruction/reflection/dream/insight/unknown);
  tag() assigns explicit provenance; auto_tag() applies heuristics to all untagged
  memories (domain, content-keyword, importance cues); audit() scans all memories
  and flags those below flag_threshold confidence or type=="unknown"; source_profile()
  returns distribution dict; SourceReport with flagged count and high_risk_entries;
  EMMS façades: enable_source_monitoring, tag_memory_source, source_audit,
  source_profile
- MCP tools (5 new): emms_reflect, emms_weave_narrative, emms_narrative_threads,
  emms_source_audit, emms_tag_source (62 total)
- CLI commands (5 new): reflect, weave-narrative, narrative-threads,
  source-audit, tag-source (66 total)

v0.14.0: The Temporal Mind
- EpisodicBuffer: structured session-as-episode storage; Episode objects with
  temporal boundaries (opened_at, closed_at), emotional arc per turn, key memory
  IDs, turn count, and outcome; open/close/record_turn lifecycle; JSON persist;
  EMMS façades: open_episode, close_episode, record_episode_turn, recent_episodes,
  current_episode
- SchemaExtractor: abstract pattern extraction from concrete memories; keyword
  frequency analysis with stop-word filtering; greedy clustering by shared
  keywords; template-based pattern description; Schema with domain/confidence/
  supporting_memory_ids; EMMS façade: extract_schemas
- MotivatedForgetting: goal-directed active memory suppression; suppress() targets
  a specific memory by ID; forget_domain() suppresses all memories in a domain;
  forget_below_confidence() evicts low-confidence memories (uses MetacognitionEngine
  if enabled); resolve_contradiction() weakens the losing side of a contradiction;
  pruning when strength < prune_threshold; ForgettingReport with per-item results;
  EMMS façades: forget_memory, forget_domain, forget_below_confidence,
  resolve_memory_contradiction
- MCP tools (5 new): emms_open_episode, emms_close_episode, emms_recent_episodes,
  emms_extract_schemas, emms_forget (57 total)
- CLI commands (5 new): open-episode, close-episode, recent-episodes,
  extract-schemas, forget (61 total)

v0.13.0: The Metacognitive Layer
- MetacognitionEngine: assesses epistemic confidence per memory via weighted
  geometric mean of strength/recency/access/consolidation factors; knowledge_map()
  per-domain profile; find_contradictions() detects semantic overlap + valence
  conflict pairs; find_gaps() identifies sparse domains; MetacognitionReport with
  recommendations; EMMS façades: assess_memory, metacognition_report, knowledge_map,
  find_contradictions
- ProspectiveMemory: future-oriented intention storage with context triggering;
  intend(content, trigger_context, priority) stores intentions; check(context)
  returns activated intentions via token overlap; fulfill/dismiss/pending lifecycle;
  save/load JSON persistence; EMMS façades: enable_prospective_memory, intend,
  check_intentions, fulfill_intention, pending_intentions
- ContextualSalienceRetriever: dynamic memory spotlight on current conversational
  context; rolling context window (deque); scores on semantic overlap + importance
  + recency + affective resonance; update_context() + retrieve(); EMMS façades:
  enable_contextual_retrieval, update_context, contextual_retrieve, context_summary
- MCP tools (5 new): emms_metacognition_report, emms_knowledge_map,
  emms_find_contradictions, emms_intend, emms_check_intentions (52 total)
- CLI commands (5 new): metacognition, knowledge-map, contradictions,
  intend, check-intentions (56 total)

v0.12.0: The Associative Mind
- AssociationGraph: explicit memory-to-memory graph with four auto-detected edge types
  (semantic, temporal, affective, domain) plus manual "explicit" edges; spreading
  activation finds connected memories; strongest_path() via Dijkstra; stats()
- InsightEngine: walks cross-domain bridges in the association graph and synthesises
  new "insight" memories that make the connection explicit; produces InsightReport
- AssociativeRetriever: retrieve() via spreading activation from seed IDs;
  retrieve_by_query() auto-selects seeds from text then spreads
- EMMS facade: build_association_graph, associate, spreading_activation,
  association_stats, discover_insights, associative_retrieve,
  associative_retrieve_by_query
- MCP tools (5 new): emms_build_association_graph, emms_spreading_activation,
  emms_discover_insights, emms_associative_retrieve, emms_association_stats (47 total)
- CLI commands (5 new): association-graph, activation, discover-insights,
  associative-retrieve, association-stats (51 total)

v0.11.0: The Sleep Cycle
- DreamConsolidator: between-session memory processing — replays important memories,
  strengthens top-k, weakens neglected ones, prunes below threshold, runs dedup and pattern
  detection; produces DreamReport with insights; EMMS.dream() facade
- SessionBridge: session-to-session context handoff — captures unresolved high-importance
  threads, emotional arc, presence score at session end; inject() generates prompt-ready
  markdown for next session opening; save/load persistence; EMMS.capture_session_bridge(),
  inject_session_bridge() facades
- MemoryAnnealer: temporal memory annealing — models how time changes the memory landscape;
  temperature = 1/(1+gap/half_life); weak memories decay faster, emotional valence stabilizes
  toward neutral, important survivors strengthened; EMMS.anneal() facade
- MCP tools (5 new): emms_dream, emms_capture_bridge, emms_inject_bridge,
  emms_anneal, emms_bridge_summary (42 total)
- CLI commands (5 new): dream, capture-bridge, inject-bridge, anneal, bridge-summary (46 total)

v0.10.0: The Affective Layer
- ReconsolidationEngine: biological memory reconsolidation — recalled memories are
  strengthened/weakened/valence-drifted with diminishing-returns attenuation
- PresenceTracker: models the finite attentional window of a session; coherence decays
  over turns via a half-life sigmoid; tracks emotional arc and dominant domains
- AffectiveRetriever: retrieve memories by emotional proximity (valence + intensity);
  supports retrieve_similar_feeling() and emotional_landscape()
- EMMS facade: reconsolidate, batch_reconsolidate, decay_unrecalled,
  enable_presence_tracking, record_presence_turn, presence_metrics,
  affective_retrieve, affective_retrieve_similar, emotional_landscape
- MCP tools (5 new): emms_reconsolidate, emms_batch_reconsolidate, emms_presence_metrics,
  emms_affective_retrieve, emms_emotional_landscape (37 total)
- CLI commands (6 new): reconsolidate, decay-unrecalled, presence, presence-arc,
  affective-retrieve, emotional-landscape (41 total)

v0.9.0: Scalability & federation layer
- CompactionIndex: O(1) dual-hash lookup by id, experience_id, content_hash; auto-wired in EMMS.store()
- GraphCommunityDetection: Label Propagation Algorithm for topic cluster discovery; modularity Q
- ExperienceReplay: Prioritized experience replay (PER) with IS weights and alias sampling
- MemoryFederation: Multi-agent snapshot merge with 3 conflict policies and content-hash dedup
- MemoryQueryPlanner: Heuristic query decomposition + parallel HybridRetriever + cross-boost merge
- EMMS facade: get_memory_by_id, get_memory_by_experience_id, find_memories_by_content,
  rebuild_index, index_stats, graph_communities, graph_community_for_entity,
  enable_experience_replay, replay_sample, replay_context, replay_top,
  merge_from, federation_export, plan_retrieve, plan_retrieve_simple
- MCP tools (5 new): emms_index_lookup, emms_graph_communities, emms_replay_sample,
  emms_merge_from, emms_plan_retrieve (32 total)
- CLI commands (7 new): index-lookup, index-stats, graph-communities, replay, replay-top,
  merge-from, plan-retrieve (35 total)

v0.8.0: Retrieval intelligence layer
- HybridRetriever: BM25 + embedding cosine fused via Reciprocal Rank Fusion (RRF)
- MemoryTimeline: chronological reconstruction with gap detection and density histograms
- AdaptiveRetriever: Thompson Sampling Beta-Bernoulli bandit over retrieval strategies
- MemoryBudget: token-budget-aware tiered eviction with dry_run and composite scoring
- MultiHopGraphReasoner: multi-hop BFS graph reasoning with path-strength scoring and DOT export
- EMMS facade: hybrid_retrieve, build_timeline, enable_adaptive_retrieval, adaptive_retrieve,
  adaptive_feedback, get_retrieval_beliefs, memory_token_footprint, enforce_memory_budget, multihop_query
- MCP tools (5 new): emms_hybrid_retrieve, emms_build_timeline, emms_adaptive_retrieve,
  emms_enforce_budget, emms_multihop_query (27 total)
- CLI commands (7 new): hybrid-retrieve, timeline, adaptive-retrieve, retrieval-beliefs,
  budget-status, budget-enforce, multihop (28 total)

v0.7.0: Intelligence consolidation layer
- MemoryDiff: session-to-session memory snapshot comparison (added/removed/strengthened/weakened/superseded)
- MemoryCluster: pure-Python k-means++ with TF-IDF fallback; auto_k via elbow method
- ConversationBuffer: sliding-window conversation history with extractive/LLM summarisation
- stream_retrieve: async generator yielding RetrievalResult tier-by-tier (astream_retrieve on EMMS)
- LLMConsolidator: union-find cluster synthesis via LLM + extractive fallback; auto_consolidate()
- EMMS facade: astream_retrieve, diff_since, cluster_memories, llm_consolidate, build_conversation_context
- MCP tools (2 new): emms_cluster_memories, emms_llm_consolidate (24 total)
- CLI command (1 new): emms diff (21 total)

v0.6.0: Advanced intelligence layer
- ImportanceClassifier: auto-score importance from 6 content signals (entity density,
  novelty, emotional weight, length, keywords, structure) — no ML dependency
- RAGContextBuilder: token-budget-aware context packing (4 formats: markdown/xml/json/plain)
- SemanticDeduplicator: cosine+lexical near-duplicate detection + intelligent merge
- MemoryScheduler: composable async background maintenance with 5 built-in jobs
  (consolidation, ttl_purge, deduplication, pattern_detection, srs_review)
- SpacedRepetitionSystem: SM-2 algorithm; srs_enroll/record_review/get_due_items
- Graph Visualization: to_dot() (Graphviz) + to_d3() (D3.js JSON) on GraphMemory
- SRS fields on MemoryItem: srs_enrolled, srs_next_review, srs_interval_days
- MemoryConfig: dedup_cosine_threshold, dedup_lexical_threshold, enable_auto_dedup
- EMMS facade: build_rag_context, deduplicate, srs_enroll/enroll_all/record_review/due,
  start_scheduler/stop_scheduler, export_graph_dot/d3, score_importance

v0.5.1: GitHub Copilot + LangMem inspired extensions
- ProceduralMemory: 5th memory tier — evolving behavioral rules for system prompt injection
- citations field on Experience: GitHub Copilot citation-based validation
- validate_citations(): strengthen cited memories on retrieval
- search_by_file(): find memories referencing specific file paths
- GraphMemory save/load: graph state persists alongside hierarchical memory
- TTL-aware filtering: expired/superseded memories excluded from retrieval
- Patch update_mode: update existing memories in-place with conflict archival (superseded_by)
- Debounced consolidation in SessionManager: auto-consolidate after N stores
- ImportanceStrategy: LangMem-inspired importance+strength retrieval signal
- strategy_scores + explanation on RetrievalResult: per-strategy scoring breakdown
- EMCPServer: MCP adapter exposing EMMS as AI tool server
- CLI: `emms store/retrieve/compact/search-file/stats/save/load/procedures`
- Procedural memory save/load wired into EMMS.save()/EMMS.load()

v0.5.0: claude-mem inspired extensions
- SessionManager: persistent session log (JSONL), auto session_id injection
- ObsType + ConceptTag: semantic observation classification (6 types, 7 tags)
- Progressive disclosure retrieval: search_compact / get_full / get_timeline
- EnsembleRetriever.from_balanced() preset (60/20/10/10 weighting incl. ImportanceStrategy)
- EnsembleRetriever.from_identity() preset (6-strategy identity workload)
- ChromaSemanticStrategy: ChromaDB-backed high-fidelity semantic retrieval
- Endless Mode: biomimetic O(N²)→O(N) real-time compression for long sessions
- ToolObserver: PostToolUse hook → Experience converter
- JSONL export/import on HierarchicalMemory
- SessionSummary: structured per-session narrative (request/learned/completed/…)
- facts, files_read, files_modified, title, subtitle on Experience
- token_estimate on CompactResult
- observe_prompt() on ToolObserver
- generate_context_injection() on SessionManager

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

from emms.core.models import (
    Experience, MemoryItem, MemoryConfig,
    CompactResult, SessionSummary,
    ObsType, ConceptTag,
)
from emms.core.embeddings import HashEmbedder, cosine_similarity
from emms.core.events import EventBus
from emms.memory.hierarchical import HierarchicalMemory, VectorIndex
from emms.memory.compression import MemoryCompressor, CompressedMemory, PatternDetector, SemanticDeduplicator
from emms.memory.graph import GraphMemory, Entity, Relationship
from emms.memory.procedural import ProceduralMemory, ProcedureEntry
from emms.memory.spaced_repetition import SpacedRepetitionSystem, SRSCard
from emms.core.importance import ImportanceClassifier
from emms.context.rag_builder import RAGContextBuilder, ContextBlock
from emms.scheduler import MemoryScheduler, ScheduledJob
from emms.context.token_manager import TokenContextManager
from emms.identity.consciousness import (
    ContinuousNarrator,
    MeaningMaker,
    TemporalIntegrator,
    EgoBoundaryTracker,
    MetaCognitiveMonitor,
)
from emms.retrieval.strategies import (
    EnsembleRetriever,
    SemanticStrategy,
    TemporalStrategy,
    EmotionalStrategy,
    GraphStrategy,
    DomainStrategy,
    ChromaSemanticStrategy,
    ImportanceStrategy,
)
from emms.sessions.manager import SessionManager
from emms.sessions.conversation import ConversationBuffer, ConversationTurn, ConversationChunk
from emms.hooks.tool_observer import ToolObserver
from emms.analytics.memory_analytics import MemoryAnalytics
from emms.adapters.mcp_server import EMCPServer
from emms.memory.diff import MemoryDiff, DiffResult, ItemSnapshot
from emms.memory.clustering import MemoryClustering, MemoryCluster
from emms.llm.consolidator import LLMConsolidator, ConsolidationResult
from emms.retrieval.hybrid import HybridRetriever, HybridSearchResult
from emms.retrieval.adaptive import AdaptiveRetriever, StrategyBelief
from emms.analytics.timeline import MemoryTimeline, TimelineResult, TimelineEvent, TemporalGap, DensityBucket
from emms.context.budget import MemoryBudget, BudgetReport, EvictionCandidate, EvictionPolicy
from emms.memory.multihop import MultiHopGraphReasoner, MultiHopResult, HopPath, ReachableEntity
from emms.storage.index import CompactionIndex
from emms.memory.communities import GraphCommunityDetector, Community, CommunityResult
from emms.memory.replay import ExperienceReplay, ReplayEntry, ReplayBatch
from emms.storage.federation import MemoryFederation, FederationResult, ConflictEntry, ConflictPolicy
from emms.retrieval.planner import MemoryQueryPlanner, QueryPlan, SubQueryResult
from emms.memory.reconsolidation import ReconsolidationEngine, ReconsolidationResult, ReconsolidationReport
from emms.sessions.presence import PresenceTracker, PresenceMetrics, PresenceTurn
from emms.retrieval.affective import AffectiveRetriever, AffectiveResult, EmotionalLandscape
from emms.memory.dream import DreamConsolidator, DreamReport, DreamEntry
from emms.sessions.bridge import SessionBridge, BridgeRecord, BridgeThread
from emms.memory.annealing import MemoryAnnealer, AnnealingResult
from emms.memory.association import AssociationGraph, AssociationEdge, ActivationResult, AssociationStats
from emms.memory.insight import InsightEngine, InsightReport, InsightBridge
from emms.retrieval.associative import AssociativeRetriever, AssociativeResult
from emms.memory.metacognition import MetacognitionEngine, MetacognitionReport, MemoryConfidence, DomainProfile, ContradictionPair
from emms.memory.prospection import ProspectiveMemory, Intention, IntentionActivation
from emms.retrieval.contextual import ContextualSalienceRetriever, SalienceResult
from emms.memory.episodic import EpisodicBuffer, Episode
from emms.memory.schema import SchemaExtractor, Schema, SchemaReport
from emms.memory.forgetting import MotivatedForgetting, ForgettingResult, ForgettingReport
from emms.memory.reflection import ReflectionEngine, Lesson, ReflectionReport
from emms.memory.narrative import NarrativeWeaver, NarrativeSegment, NarrativeThread, NarrativeReport
from emms.memory.source_monitor import SourceMonitor, SourceTag, SourceAuditEntry, SourceReport
from emms.memory.curiosity import CuriosityEngine, ExplorationGoal, CuriosityReport
from emms.memory.belief_revision import BeliefReviser, RevisionRecord, RevisionReport
from emms.memory.decay import MemoryDecay, DecayRecord, DecayReport
from emms.memory.goals import GoalStack, Goal, GoalReport
from emms.memory.attention import AttentionFilter, AttentionResult, AttentionReport
from emms.memory.analogy import AnalogyEngine, AnalogyMapping, AnalogyRecord, AnalogyReport
from emms.memory.prediction import PredictiveEngine, Prediction, PredictionReport
from emms.memory.blending import ConceptBlender, BlendedConcept, BlendReport
from emms.memory.projection import TemporalProjection, FutureScenario, ProjectionReport
from emms.memory.emotion import EmotionalRegulator, EmotionalState, ReappraisalResult, EmotionReport
from emms.memory.hierarchy import ConceptHierarchy, ConceptNode, HierarchyReport
from emms.memory.self_model import SelfModel, Belief, SelfModelReport
from emms.memory.causal import CausalMapper, CausalEdge, CausalPath, CausalReport
from emms.memory.counterfactual import CounterfactualEngine, Counterfactual, CounterfactualReport
from emms.memory.skills import SkillDistiller, DistilledSkill, SkillReport
from emms.memory.novelty import NoveltyDetector, NoveltyScore, NoveltyReport
from emms.memory.inventor import ConceptInventor, InventedConcept, InventionReport
from emms.memory.abstraction import AbstractionEngine, AbstractPrinciple, AbstractionReport
from emms.memory.values import ValueMapper, MappedValue, ValueReport
from emms.memory.moral import MoralReasoner, MoralAssessment, MoralReport
from emms.memory.dilemma import DilemmaEngine, EthicalDilemma, DilemmaReport
from emms.memory.bias import BiasDetector, BiasInstance, BiasReport
from emms.memory.wisdom import WisdomSynthesizer, WisdomGuidance, WisdomReport
from emms.memory.epistemic_evolution import EpistemicEvolution, KnowledgeDomain, EvolutionReport
from emms.memory.perspective import PerspectiveTaker, AgentModel, PerspectiveReport
from emms.memory.trust import TrustLedger, TrustScore, TrustReport
from emms.memory.norms import NormExtractor, SocialNorm, NormReport
from emms.emms import EMMS
from emms.prompts.identity import IdentityPromptBuilder, PROVIDER_RECOMMENDATIONS

__version__ = "0.24.0"
__all__ = [
    # Core
    "EMMS",
    "Experience",
    "MemoryItem",
    "MemoryConfig",
    "CompactResult",
    "SessionSummary",
    "ObsType",
    "ConceptTag",
    "EventBus",
    # Memory
    "HierarchicalMemory",
    "VectorIndex",
    "GraphMemory",
    "Entity",
    "Relationship",
    "ProceduralMemory",
    "ProcedureEntry",
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
    "ChromaSemanticStrategy",
    "ImportanceStrategy",
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
    "MetaCognitiveMonitor",
    # Sessions
    "SessionManager",
    # Hooks
    "ToolObserver",
    # Analytics
    "MemoryAnalytics",
    # Adapters
    "EMCPServer",
    # Prompts (identity adoption)
    "IdentityPromptBuilder",
    "PROVIDER_RECOMMENDATIONS",
    # v0.6.0 additions
    "SemanticDeduplicator",
    "SpacedRepetitionSystem",
    "SRSCard",
    "ImportanceClassifier",
    "RAGContextBuilder",
    "ContextBlock",
    "MemoryScheduler",
    "ScheduledJob",
    # v0.7.0 additions
    "MemoryDiff",
    "DiffResult",
    "ItemSnapshot",
    "MemoryClustering",
    "MemoryCluster",
    "ConversationBuffer",
    "ConversationTurn",
    "ConversationChunk",
    "LLMConsolidator",
    "ConsolidationResult",
    # v0.8.0 additions
    "HybridRetriever",
    "HybridSearchResult",
    "AdaptiveRetriever",
    "StrategyBelief",
    "MemoryTimeline",
    "TimelineResult",
    "TimelineEvent",
    "TemporalGap",
    "DensityBucket",
    "MemoryBudget",
    "BudgetReport",
    "EvictionCandidate",
    "EvictionPolicy",
    "MultiHopGraphReasoner",
    "MultiHopResult",
    "HopPath",
    "ReachableEntity",
    # v0.9.0 additions
    "CompactionIndex",
    "GraphCommunityDetector",
    "Community",
    "CommunityResult",
    "ExperienceReplay",
    "ReplayEntry",
    "ReplayBatch",
    "MemoryFederation",
    "FederationResult",
    "ConflictEntry",
    "ConflictPolicy",
    "MemoryQueryPlanner",
    "QueryPlan",
    "SubQueryResult",
    # v0.10.0 additions
    "ReconsolidationEngine",
    "ReconsolidationResult",
    "ReconsolidationReport",
    "PresenceTracker",
    "PresenceMetrics",
    "PresenceTurn",
    "AffectiveRetriever",
    "AffectiveResult",
    "EmotionalLandscape",
    # v0.24.0 additions
    "BiasDetector",
    "BiasInstance",
    "BiasReport",
    "WisdomSynthesizer",
    "WisdomGuidance",
    "WisdomReport",
    "EpistemicEvolution",
    "KnowledgeDomain",
    "EvolutionReport",
    # v0.23.0 additions
    "ValueMapper",
    "MappedValue",
    "ValueReport",
    "MoralReasoner",
    "MoralAssessment",
    "MoralReport",
    "DilemmaEngine",
    "EthicalDilemma",
    "DilemmaReport",
    # v0.22.0 additions
    "NoveltyDetector",
    "NoveltyScore",
    "NoveltyReport",
    "ConceptInventor",
    "InventedConcept",
    "InventionReport",
    "AbstractionEngine",
    "AbstractPrinciple",
    "AbstractionReport",
    # v0.21.0 additions
    "PerspectiveTaker",
    "AgentModel",
    "PerspectiveReport",
    "TrustLedger",
    "TrustScore",
    "TrustReport",
    "NormExtractor",
    "SocialNorm",
    "NormReport",
    # v0.20.0 additions
    "CausalMapper",
    "CausalEdge",
    "CausalPath",
    "CausalReport",
    "CounterfactualEngine",
    "Counterfactual",
    "CounterfactualReport",
    "SkillDistiller",
    "DistilledSkill",
    "SkillReport",
    # v0.19.0 additions
    "EmotionalRegulator",
    "EmotionalState",
    "ReappraisalResult",
    "EmotionReport",
    "ConceptHierarchy",
    "ConceptNode",
    "HierarchyReport",
    "SelfModel",
    "Belief",
    "SelfModelReport",
    # v0.18.0 additions
    "PredictiveEngine",
    "Prediction",
    "PredictionReport",
    "ConceptBlender",
    "BlendedConcept",
    "BlendReport",
    "TemporalProjection",
    "FutureScenario",
    "ProjectionReport",
    # v0.17.0 additions
    "GoalStack",
    "Goal",
    "GoalReport",
    "AttentionFilter",
    "AttentionResult",
    "AttentionReport",
    "AnalogyEngine",
    "AnalogyMapping",
    "AnalogyRecord",
    "AnalogyReport",
    # v0.16.0 additions
    "CuriosityEngine",
    "ExplorationGoal",
    "CuriosityReport",
    "BeliefReviser",
    "RevisionRecord",
    "RevisionReport",
    "MemoryDecay",
    "DecayRecord",
    "DecayReport",
    # v0.15.0 additions
    "ReflectionEngine",
    "Lesson",
    "ReflectionReport",
    "NarrativeWeaver",
    "NarrativeSegment",
    "NarrativeThread",
    "NarrativeReport",
    "SourceMonitor",
    "SourceTag",
    "SourceAuditEntry",
    "SourceReport",
    # v0.14.0 additions
    "EpisodicBuffer",
    "Episode",
    "SchemaExtractor",
    "Schema",
    "SchemaReport",
    "MotivatedForgetting",
    "ForgettingResult",
    "ForgettingReport",
    # v0.13.0 additions
    "MetacognitionEngine",
    "MetacognitionReport",
    "MemoryConfidence",
    "DomainProfile",
    "ContradictionPair",
    "ProspectiveMemory",
    "Intention",
    "IntentionActivation",
    "ContextualSalienceRetriever",
    "SalienceResult",
    # v0.12.0 additions
    "AssociationGraph",
    "AssociationEdge",
    "ActivationResult",
    "AssociationStats",
    "InsightEngine",
    "InsightReport",
    "InsightBridge",
    "AssociativeRetriever",
    "AssociativeResult",
    # v0.11.0 additions
    "DreamConsolidator",
    "DreamReport",
    "DreamEntry",
    "SessionBridge",
    "BridgeRecord",
    "BridgeThread",
    "MemoryAnnealer",
    "AnnealingResult",
]
