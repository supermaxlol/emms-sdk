#!/usr/bin/env python3
"""EMMS v0.4.0 — Comprehensive Demo & Test Suite

Tests every major subsystem with both Claude (API) and Ollama (local),
logs all results, and generates a full test report.

Run:
    python demo_full.py                          # Ollama only
    ANTHROPIC_API_KEY=sk-... python demo_full.py # Ollama + Claude
"""

from __future__ import annotations

import sys
import os
import asyncio
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.core.events import EventBus
from emms.memory.graph import GraphMemory
from emms.memory.compression import PatternDetector
from emms.retrieval.strategies import (
    SemanticStrategy, TemporalStrategy, EmotionalStrategy,
    DomainStrategy, EnsembleRetriever,
)
from emms.integrations.llm import OllamaProvider, LLMEnhancer

# Try Claude
_HAS_CLAUDE = False
try:
    from emms.integrations.llm import ClaudeProvider
    _HAS_CLAUDE = True
except ImportError:
    pass

LOG: list[dict] = []
SECTION_NUM = 0


def log(section: str, key: str, value, *, is_error: bool = False):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "section": section,
        "key": key,
        "value": str(value) if not isinstance(value, (int, float, bool, dict, list)) else value,
        "error": is_error,
    }
    LOG.append(entry)
    prefix = "  [ERROR]" if is_error else "  "
    display = str(value)
    if len(display) > 200:
        display = display[:200] + "..."
    print(f"{prefix}{key}: {display}")


def section(title: str):
    global SECTION_NUM
    SECTION_NUM += 1
    header = f"\n{'='*70}\n  {SECTION_NUM}. {title}\n{'='*70}\n"
    print(header)
    return f"{SECTION_NUM}. {title}"


# =========================================================================
# Test functions
# =========================================================================

def test_initialization():
    s = section("INITIALIZATION")
    emms = EMMS(
        config=MemoryConfig(working_capacity=15),
        embedder=HashEmbedder(dim=64),
    )
    log(s, "EMMS version", "0.4.0")
    log(s, "Consciousness enabled", emms._consciousness_enabled)
    log(s, "Graph memory", emms.graph is not None)
    log(s, "Event bus", emms.events is not None)
    log(s, "Narrator", emms.narrator is not None)
    log(s, "MeaningMaker", emms.meaning_maker is not None)
    log(s, "TemporalIntegrator", emms.temporal is not None)
    log(s, "EgoBoundaryTracker", emms.ego_boundary is not None)
    log(s, "PatternDetector", emms.pattern_detector is not None)

    # Test disabled consciousness
    emms_no_c = EMMS(enable_consciousness=False)
    log(s, "Consciousness disabled test", emms_no_c.narrator is None)

    # Test disabled graph
    emms_no_g = EMMS(enable_graph=False)
    log(s, "Graph disabled test", emms_no_g.graph is None)

    return emms


def test_event_bus(emms: EMMS):
    s = section("EVENT BUS")
    events_received = {"stored": [], "consolidated": []}

    emms.events.on("memory.stored", lambda d: events_received["stored"].append(d))
    emms.events.on("memory.consolidated", lambda d: events_received["consolidated"].append(d))

    log(s, "Listeners registered", 2)
    log(s, "Event types", ["memory.stored", "memory.consolidated"])
    return events_received


def test_store_experiences(emms: EMMS):
    s = section("STORE EXPERIENCES (with consciousness + graph + events)")

    experiences = [
        # Personal
        Experience(content="Shehzad Ahmed is a computer science student at Independent University Bangladesh", domain="personal", importance=0.9),
        Experience(content="Shehzad built the EMMS system for persistent AI identity research", domain="tech", importance=0.95),
        Experience(content="Shehzad presented his research paper at the IUB symposium on AI consciousness", domain="academic", importance=0.9),
        # Finance
        Experience(content="Bitcoin surged past 100 thousand dollars as institutional investors increased positions", domain="finance", importance=0.8),
        Experience(content="The stock market in Dhaka rose 3 percent on positive GDP growth data", domain="finance", importance=0.65),
        Experience(content="Federal Reserve held interest rates steady at 4.5 percent", domain="finance", importance=0.7),
        # Tech
        Experience(content="DeepSeek released their R1 reasoning model with open source weights on Hugging Face", domain="tech", importance=0.85),
        Experience(content="Claude and GPT-4 are the leading large language models in 2026", domain="tech", importance=0.75),
        Experience(content="Anthropic announced Claude Opus 4.6 with enhanced reasoning capabilities", domain="tech", importance=0.88),
        Experience(content="OpenAI released GPT-5 with 100 trillion parameter architecture", domain="tech", importance=0.9),
        # Science
        Experience(content="Quantum computing breakthrough at MIT achieved 1000 qubit quantum processor", domain="science", importance=0.92),
        Experience(content="CERN discovered new subatomic particle that could explain dark matter", domain="science", importance=0.95),
        # Weather
        Experience(content="Bangladesh experienced severe flooding in the eastern districts affecting millions", domain="weather", importance=0.7),
        Experience(content="Category 5 hurricane approaching the Gulf Coast with 180 mph winds", domain="weather", importance=0.85),
    ]

    results = []
    for exp in experiences:
        t0 = time.perf_counter()
        result = emms.store(exp)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append({"domain": exp.domain, "elapsed_ms": round(elapsed, 2)})

    log(s, "Total stored", len(experiences))
    log(s, "Memory stats", emms.stats["memory"])
    log(s, "Avg store latency (ms)", round(sum(r["elapsed_ms"] for r in results) / len(results), 3))
    log(s, "Domains", list(set(e.domain for e in experiences)))

    # Verify store results include new v0.4.0 fields
    sample_result = emms.store(Experience(content="Test verification item", domain="test"))
    log(s, "Result has 'consciousness'", "consciousness" in sample_result)
    log(s, "Result has 'graph_entities'", "graph_entities" in sample_result)
    log(s, "Result has 'experience_id'", "experience_id" in sample_result)

    return experiences


def test_retrieval(emms: EMMS):
    s = section("RETRIEVAL (lexical + embedding)")

    queries = {
        "Shehzad research paper": "personal/academic",
        "Bitcoin price cryptocurrency": "finance",
        "quantum computing breakthrough": "science",
        "weather flooding Bangladesh": "weather",
        "large language models AI": "tech",
        "stock market GDP growth": "finance",
    }

    for query, expected_domain in queries.items():
        t0 = time.perf_counter()
        results = emms.retrieve(query, max_results=3)
        elapsed = (time.perf_counter() - t0) * 1000
        top = results[0] if results else None
        top_content = top.memory.experience.content[:60] + "..." if top else "NONE"
        top_score = round(top.score, 3) if top else 0
        log(s, f"Query '{query[:40]}'",
            f"[{elapsed:.2f}ms] top={top_score} | {top_content}")

    # Semantic retrieval
    log(s, "Semantic retrieval test", "HashEmbedder cosine similarity")
    results = emms.retrieve_semantic("artificial intelligence research", max_results=5)
    log(s, "Semantic results count", len(results))


def test_graph_memory(emms: EMMS):
    s = section("GRAPH MEMORY")
    stats = emms.graph.size
    log(s, "Total entities", stats["entities"])
    log(s, "Total relationships", stats["relationships"])

    # List all entities
    all_entities = list(emms.graph.entities.keys())
    log(s, "Entity names", all_entities[:15])

    # Query specific entities
    for name in ["shehzad", "emms", "bitcoin", "mit", "cern", "anthropic"]:
        result = emms.query_entity(name)
        if result["found"]:
            neighbors = result["neighbors"][:5]
            rels = len(result["relationships"])
            log(s, f"Entity '{name}'", f"found | neighbors={neighbors} | rels={rels}")
        else:
            # Try case variations
            for key in all_entities:
                if name in key.lower():
                    result2 = emms.query_entity(key)
                    if result2["found"]:
                        log(s, f"Entity '{key}' (alt)", f"found | neighbors={result2['neighbors'][:5]}")
                        break
            else:
                log(s, f"Entity '{name}'", "not found")

    # Path query
    if len(all_entities) >= 2:
        path = emms.query_entity_path(all_entities[0], all_entities[-1])
        log(s, f"Path {all_entities[0]} → {all_entities[-1]}", path if path else "no path")

    # Subgraph
    if all_entities:
        sub = emms.get_subgraph(all_entities[0], depth=2)
        log(s, f"Subgraph of '{all_entities[0]}'", f"nodes={len(sub['nodes'])}, edges={len(sub['edges'])}")


def test_consciousness(emms: EMMS):
    s = section("CONSCIOUSNESS STATE")

    state = emms.get_consciousness_state()
    log(s, "Enabled", state["enabled"])
    log(s, "Narrative coherence", round(state["narrative_coherence"], 3))
    log(s, "Ego boundary strength", round(state["ego_boundary_strength"], 3))
    log(s, "Themes (top 5)", dict(list(state["themes"].items())[:5]))

    # Narrator
    log(s, "Narrator entries", len(emms.narrator.entries))
    log(s, "Narrator traits", emms.narrator.traits)
    log(s, "Narrator autobiographical events", len(emms.narrator.autobiographical))

    # Third-person narrative
    narrative = emms.get_narrative("EMMS-TestAgent")
    log(s, "Third-person narrative", narrative[:200])

    # First-person narrative
    fp = emms.get_first_person_narrative()
    log(s, "First-person narrative", fp[:200])

    # MeaningMaker
    log(s, "MeaningMaker total processed", emms.meaning_maker.total_processed)
    log(s, "MeaningMaker patterns", len(emms.meaning_maker.pattern_tracker))

    # Temporal
    log(s, "Temporal milestones", len(emms.temporal.milestones))

    # Ego boundary
    log(s, "Ego boundary history", len(emms.ego_boundary.boundary_history))


def test_patterns(emms: EMMS):
    s = section("PATTERN DETECTION")
    patterns = emms.detect_patterns()

    log(s, "Domain dominant", patterns["domain"].get("dominant"))
    log(s, "Domain distribution", patterns["domain"].get("distribution"))
    log(s, "Domain trends", patterns["domain"].get("trends", [])[:3])

    if patterns["content"]["concepts"]:
        top = patterns["content"]["concepts"][:8]
        log(s, "Top content concepts", [(c["term"], c.get("frequency", c.get("count", 0))) for c in top])

    if patterns["content"].get("bigrams"):
        log(s, "Top bigrams", [(b.get("phrase", b.get("bigram")), b.get("frequency", 0)) for b in patterns["content"]["bigrams"][:5]])

    log(s, "Sequence patterns", patterns["sequence"].get("count", 0))


def test_consolidation(emms: EMMS, events_received: dict):
    s = section("CONSOLIDATION & EVENTS")

    t0 = time.perf_counter()
    result = emms.consolidate()
    elapsed = (time.perf_counter() - t0) * 1000

    log(s, "Items consolidated", result["items_consolidated"])
    log(s, "Consolidation time (ms)", round(elapsed, 2))
    log(s, "Memory after consolidation", result["memory_sizes"])
    log(s, "Store events received", len(events_received["stored"]))
    log(s, "Consolidation events received", len(events_received["consolidated"]))

    # Compression
    t0 = time.perf_counter()
    emms.compress_long_term()
    elapsed = (time.perf_counter() - t0) * 1000
    log(s, "Compression time (ms)", round(elapsed, 2))


def test_persistence(emms: EMMS):
    s = section("PERSISTENCE (save/load)")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "emms_state.json"

        # Save
        t0 = time.perf_counter()
        emms.save(memory_path=path)
        save_ms = (time.perf_counter() - t0) * 1000
        file_kb = path.stat().st_size / 1024

        log(s, "Save time (ms)", round(save_ms, 2))
        log(s, "File size (KB)", round(file_kb, 1))
        log(s, "Items saved", emms.stats["memory"]["total"])

        # Load into fresh instance
        emms2 = EMMS(
            config=MemoryConfig(working_capacity=15),
            embedder=HashEmbedder(dim=64),
        )
        t0 = time.perf_counter()
        emms2.load(memory_path=path)
        load_ms = (time.perf_counter() - t0) * 1000

        log(s, "Load time (ms)", round(load_ms, 2))
        log(s, "Items loaded", emms2.stats["memory"]["total"])

        # Verify retrieval after load
        results = emms2.retrieve("Shehzad research", max_results=3)
        log(s, "Post-load retrieval", f"{len(results)} results")
        if results:
            log(s, "Top result", results[0].memory.experience.content[:80])

        results2 = emms2.retrieve("Bitcoin", max_results=3)
        log(s, "Post-load 'Bitcoin' retrieval", f"{len(results2)} results")

        # Verify JSON structure
        data = json.loads(path.read_text())
        log(s, "JSON keys", list(data.keys()))
        log(s, "Version in file", data.get("version"))


def test_episode_detection():
    s = section("EPISODE DETECTION (all algorithms)")
    from emms.episodes.boundary import EpisodeBoundaryDetector

    base = time.time()
    experiences = []
    # Finance cluster
    for i in range(5):
        experiences.append(Experience(
            content=f"Stock market trading analysis report item {i}",
            domain="finance", timestamp=base + i,
        ))
    # Tech cluster (temporal gap)
    for i in range(5):
        experiences.append(Experience(
            content=f"Python programming language update release {i}",
            domain="tech", timestamp=base + 1000 + i,
        ))

    for algo in ["heuristic", "graph", "spectral", "conductance", "multi"]:
        det = EpisodeBoundaryDetector(algorithm=algo)
        for exp in experiences:
            det.add(exp)
        t0 = time.perf_counter()
        episodes = det.detect()
        elapsed = (time.perf_counter() - t0) * 1000
        coherences = [round(ep.coherence, 3) for ep in episodes]
        log(s, f"Algorithm '{algo}'",
            f"{len(episodes)} episodes in {elapsed:.2f}ms | coherences={coherences}")


def test_multi_strategy_retrieval(emms: EMMS):
    s = section("MULTI-STRATEGY ENSEMBLE RETRIEVAL")

    items = list(emms.memory.working) + list(emms.memory.short_term)
    log(s, "Items available for retrieval", len(items))

    embedder = emms.memory.embedder

    # Test individual strategies
    test_query = "cryptocurrency market investment"
    for strategy_cls, name in [
        (SemanticStrategy, "Semantic"),
        (TemporalStrategy, "Temporal"),
        (EmotionalStrategy, "Emotional"),
        (DomainStrategy, "Domain"),
    ]:
        if name == "Semantic":
            strategy = strategy_cls(embedder=embedder)
        else:
            strategy = strategy_cls()

        scores = []
        ctx = {}
        for item in items[:5]:
            score = strategy.score(test_query, item, ctx)
            scores.append(round(score, 3))
        log(s, f"{name} scores (first 5)", scores)

    # Full ensemble
    retriever = EnsembleRetriever()
    retriever.add_strategy(SemanticStrategy(embedder), weight=0.35)
    retriever.add_strategy(TemporalStrategy(), weight=0.20)
    retriever.add_strategy(EmotionalStrategy(), weight=0.15)
    retriever.add_strategy(DomainStrategy(), weight=0.15)

    t0 = time.perf_counter()
    results = retriever.retrieve("stock market cryptocurrency Bitcoin", items, max_results=5)
    elapsed = (time.perf_counter() - t0) * 1000

    log(s, "Ensemble retrieval time (ms)", round(elapsed, 2))
    log(s, "Ensemble results", len(results))
    for i, r in enumerate(results[:5]):
        log(s, f"  Rank {i+1}", f"[{r.score:.3f}] {r.memory.experience.content[:60]}")


async def test_ollama(emms: EMMS):
    s = section("OLLAMA LLM INTEGRATION (gemma3n:e4b)")

    provider = OllamaProvider(model="gemma3n:e4b")
    enhancer = LLMEnhancer(provider)

    # Test connection
    print("  Testing Ollama connection...")
    t0 = time.perf_counter()
    try:
        response = await provider.generate("Respond with exactly one sentence: What is AI?", max_tokens=100)
        elapsed = (time.perf_counter() - t0) * 1000
        log(s, "Connection test", f"OK ({elapsed:.0f}ms)")
        log(s, "Response", response.strip()[:150])
    except Exception as e:
        log(s, "Connection test", f"FAILED: {e}", is_error=True)
        return {}

    results = {}

    # 7a. Enrich experience
    print("\n  --- Experience Enrichment ---")
    raw_exp = Experience(
        content="Scientists at CERN discovered a new subatomic particle that could explain dark matter",
        domain="general",
    )
    before = f"importance={raw_exp.importance:.2f}, domain='{raw_exp.domain}', valence={raw_exp.emotional_valence:.2f}"
    t0 = time.perf_counter()
    enriched = await enhancer.enrich_experience(raw_exp)
    elapsed = (time.perf_counter() - t0) * 1000
    after = f"importance={enriched.importance:.2f}, domain='{enriched.domain}', valence={enriched.emotional_valence:.2f}"
    log(s, "Enrichment before", before)
    log(s, "Enrichment after", f"{after} ({elapsed:.0f}ms)")
    results["enrichment"] = {"before": before, "after": after, "latency_ms": round(elapsed)}

    # 7b. Entity extraction
    print("\n  --- LLM Entity Extraction ---")
    text = "Shehzad Ahmed built EMMS at Independent University Bangladesh. He used Python and Claude API from Anthropic for his AI consciousness research."
    t0 = time.perf_counter()
    entities = await enhancer.extract_entities(text)
    elapsed = (time.perf_counter() - t0) * 1000
    log(s, "Entities extracted", f"{len(entities)} ({elapsed:.0f}ms)")
    for ent in entities[:8]:
        log(s, f"  Entity", f"{ent.get('name', '?')} ({ent.get('type', '?')})")
    results["entities"] = {"count": len(entities), "items": entities[:8], "latency_ms": round(elapsed)}

    # 7c. Narrative generation
    print("\n  --- LLM Narrative Generation ---")
    state = emms.get_consciousness_state()
    context = {
        "themes": dict(list(state["themes"].items())[:5]),
        "traits": emms.narrator.traits if emms.narrator else {},
        "experience_count": len(emms.narrator.entries) if emms.narrator else 0,
        "domains": list(set(e.domain for e in [item.experience for item in emms.memory.working])),
    }
    t0 = time.perf_counter()
    narrative = await enhancer.generate_narrative(context)
    elapsed = (time.perf_counter() - t0) * 1000
    clean = narrative.strip()
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    log(s, "LLM narrative", f"({elapsed:.0f}ms) {clean[:250]}")
    results["narrative"] = {"text": clean[:500], "latency_ms": round(elapsed)}

    # 7d. Memory summarisation
    print("\n  --- LLM Memory Summarisation ---")
    all_items = list(emms.memory.working) + list(emms.memory.short_term)
    if all_items:
        t0 = time.perf_counter()
        summary = await enhancer.summarize_memories(all_items)
        elapsed = (time.perf_counter() - t0) * 1000
        clean_summary = summary.strip()
        if "</think>" in clean_summary:
            clean_summary = clean_summary.split("</think>")[-1].strip()
        log(s, "Memory summary", f"({elapsed:.0f}ms) {clean_summary[:250]}")
        results["summary"] = {"text": clean_summary[:500], "latency_ms": round(elapsed)}

    # 7e. Conversation-style demo
    print("\n  --- Conversational Demo ---")
    conversation = [
        "What are the most important things happening in tech right now?",
        "How does quantum computing relate to AI?",
        "Summarize what you know about Shehzad's work.",
    ]
    conv_results = []
    for question in conversation:
        # Retrieve relevant memories
        memories = emms.retrieve(question, max_results=3)
        context_str = "\n".join(f"- {r.memory.experience.content}" for r in memories)
        prompt = f"Based on these memories:\n{context_str}\n\nAnswer briefly: {question}"
        t0 = time.perf_counter()
        answer = await provider.generate(prompt, max_tokens=200)
        elapsed = (time.perf_counter() - t0) * 1000
        clean_answer = answer.strip()
        if "</think>" in clean_answer:
            clean_answer = clean_answer.split("</think>")[-1].strip()
        log(s, f"Q: {question[:50]}", f"({elapsed:.0f}ms)")
        log(s, f"A", clean_answer[:200])
        conv_results.append({"question": question, "answer": clean_answer[:500], "latency_ms": round(elapsed), "memories_used": len(memories)})
    results["conversation"] = conv_results

    return results


async def test_claude(emms: EMMS):
    s = section("CLAUDE API INTEGRATION (claude-sonnet-4-5-20250929)")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log(s, "Status", "SKIPPED — no ANTHROPIC_API_KEY set")
        return {}

    try:
        provider = ClaudeProvider(api_key=api_key, model="claude-sonnet-4-5-20250929")
    except ImportError:
        log(s, "Status", "SKIPPED — anthropic package not installed")
        return {}

    enhancer = LLMEnhancer(provider)
    results = {}

    # Test connection
    print("  Testing Claude API connection...")
    t0 = time.perf_counter()
    try:
        response = await provider.generate("Respond with exactly one sentence: What is EMMS?", max_tokens=100)
        elapsed = (time.perf_counter() - t0) * 1000
        log(s, "Connection test", f"OK ({elapsed:.0f}ms)")
        log(s, "Response", response.strip()[:200])
    except Exception as e:
        error_msg = str(e)[:200]
        log(s, "Connection test", f"FAILED: {error_msg}", is_error=True)
        if "credit balance" in error_msg.lower() or "billing" in error_msg.lower():
            log(s, "Reason", "API credits depleted — need to purchase more credits")
        results["error"] = error_msg
        return results

    # If connection works, run all tests
    # 8a. Enrich experience
    print("\n  --- Claude Experience Enrichment ---")
    raw_exp = Experience(
        content="Researchers at Stanford demonstrated a quantum neural network that learns 1000x faster than classical approaches",
        domain="general",
    )
    t0 = time.perf_counter()
    enriched = await enhancer.enrich_experience(raw_exp)
    elapsed = (time.perf_counter() - t0) * 1000
    log(s, "Claude enrichment", f"importance={enriched.importance:.2f}, domain='{enriched.domain}' ({elapsed:.0f}ms)")
    results["enrichment"] = {"importance": enriched.importance, "domain": enriched.domain, "latency_ms": round(elapsed)}

    # 8b. Entity extraction
    print("\n  --- Claude Entity Extraction ---")
    text = "Shehzad Ahmed developed EMMS at IUB. He integrated Claude from Anthropic and DeepSeek R1 into the system."
    t0 = time.perf_counter()
    entities = await enhancer.extract_entities(text)
    elapsed = (time.perf_counter() - t0) * 1000
    log(s, "Claude entities", f"{len(entities)} ({elapsed:.0f}ms)")
    for ent in entities[:8]:
        log(s, f"  Entity", f"{ent.get('name', '?')} ({ent.get('type', '?')})")
    results["entities"] = {"count": len(entities), "items": entities[:8], "latency_ms": round(elapsed)}

    # 8c. Narrative
    print("\n  --- Claude Narrative Generation ---")
    state = emms.get_consciousness_state()
    context = {
        "themes": dict(list(state["themes"].items())[:5]),
        "traits": emms.narrator.traits if emms.narrator else {},
        "experience_count": len(emms.narrator.entries) if emms.narrator else 0,
        "domains": list(set(e.domain for e in [item.experience for item in emms.memory.working])),
    }
    t0 = time.perf_counter()
    narrative = await enhancer.generate_narrative(context)
    elapsed = (time.perf_counter() - t0) * 1000
    log(s, "Claude narrative", f"({elapsed:.0f}ms) {narrative.strip()[:250]}")
    results["narrative"] = {"text": narrative.strip()[:500], "latency_ms": round(elapsed)}

    # 8d. Conversation
    print("\n  --- Claude Conversational Demo ---")
    conv_results = []
    questions = [
        "What are the key themes across all stored memories?",
        "What is Shehzad working on and why is it significant?",
        "Compare the finance and technology news — what patterns emerge?",
    ]
    for question in questions:
        memories = emms.retrieve(question, max_results=3)
        context_str = "\n".join(f"- {r.memory.experience.content}" for r in memories)
        prompt = f"You are an AI with these memories:\n{context_str}\n\nAnswer concisely: {question}"
        t0 = time.perf_counter()
        answer = await provider.generate(prompt, max_tokens=200)
        elapsed = (time.perf_counter() - t0) * 1000
        log(s, f"Q: {question[:50]}", f"({elapsed:.0f}ms)")
        log(s, f"A", answer.strip()[:200])
        conv_results.append({"question": question, "answer": answer.strip()[:500], "latency_ms": round(elapsed)})
    results["conversation"] = conv_results

    return results


def test_full_stats(emms: EMMS):
    s = section("FULL SYSTEM STATS")
    stats = emms.stats
    log(s, "Memory", stats["memory"])
    log(s, "Graph", stats.get("graph", {}))
    log(s, "Consciousness", stats.get("consciousness", {}))
    log(s, "Events", stats.get("events", {}))
    return stats


# =========================================================================
# Report generation
# =========================================================================

def generate_report(
    ollama_results: dict,
    claude_results: dict,
    full_stats: dict,
) -> str:
    """Generate markdown report of all test findings."""
    report = []
    report.append("# EMMS v0.4.0 — Comprehensive Test Report")
    report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Author**: Automated Test Suite")
    report.append(f"**Models tested**: Ollama (gemma3n:e4b), Claude (claude-sonnet-4-5-20250929)")
    report.append("")

    report.append("## Test Summary\n")
    total_tests = SECTION_NUM
    errors = sum(1 for e in LOG if e["error"])
    report.append(f"- **Sections tested**: {total_tests}")
    report.append(f"- **Log entries**: {len(LOG)}")
    report.append(f"- **Errors**: {errors}")
    report.append("")

    # Section-by-section results
    report.append("## Detailed Results\n")
    current_section = None
    for entry in LOG:
        if entry["section"] != current_section:
            current_section = entry["section"]
            report.append(f"\n### {current_section}\n")
        prefix = "**[ERROR]** " if entry["error"] else ""
        val = entry["value"]
        if isinstance(val, str) and len(val) > 120:
            val = val[:120] + "..."
        report.append(f"- {prefix}**{entry['key']}**: {val}")

    # LLM comparison
    report.append("\n## LLM Provider Comparison\n")
    report.append("| Feature | Ollama (gemma3n:e4b) | Claude (Sonnet 4.5) |")
    report.append("|---------|---------------------|---------------------|")

    ollama_enrich = ollama_results.get("enrichment", {})
    claude_enrich = claude_results.get("enrichment", {})
    report.append(f"| Enrichment latency | {ollama_enrich.get('latency_ms', 'N/A')}ms | {claude_enrich.get('latency_ms', 'N/A')}ms |")

    ollama_ent = ollama_results.get("entities", {})
    claude_ent = claude_results.get("entities", {})
    report.append(f"| Entity extraction | {ollama_ent.get('count', 'N/A')} entities | {claude_ent.get('count', 'N/A')} entities |")

    ollama_narr = ollama_results.get("narrative", {})
    claude_narr = claude_results.get("narrative", {})
    report.append(f"| Narrative latency | {ollama_narr.get('latency_ms', 'N/A')}ms | {claude_narr.get('latency_ms', 'N/A')}ms |")

    if claude_results.get("error"):
        report.append(f"\n**Claude API Error**: {claude_results['error'][:200]}")

    # Conversation logs
    report.append("\n## Conversational Demo Transcripts\n")

    if ollama_results.get("conversation"):
        report.append("### Ollama (gemma3n:e4b) Conversations\n")
        for conv in ollama_results["conversation"]:
            report.append(f"**Q**: {conv['question']}")
            report.append(f"**A** ({conv['latency_ms']}ms): {conv['answer'][:300]}\n")

    if claude_results.get("conversation"):
        report.append("### Claude (Sonnet 4.5) Conversations\n")
        for conv in claude_results["conversation"]:
            report.append(f"**Q**: {conv['question']}")
            report.append(f"**A** ({conv['latency_ms']}ms): {conv['answer'][:300]}\n")

    # System stats
    report.append("\n## Final System State\n")
    report.append(f"```json\n{json.dumps(full_stats, indent=2, default=str)}\n```\n")

    return "\n".join(report)


# =========================================================================
# Main
# =========================================================================

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Comprehensive Demo & Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1-2. Init + events
    emms = test_initialization()
    events_received = test_event_bus(emms)

    # 3. Store
    experiences = test_store_experiences(emms)

    # 4. Retrieval
    test_retrieval(emms)

    # 5. Graph memory
    test_graph_memory(emms)

    # 6. Consciousness
    test_consciousness(emms)

    # 7. Patterns
    test_patterns(emms)

    # 8. Consolidation + events
    test_consolidation(emms, events_received)

    # 9. Persistence
    test_persistence(emms)

    # 10. Episode detection
    test_episode_detection()

    # 11. Multi-strategy retrieval
    test_multi_strategy_retrieval(emms)

    # 12. Ollama
    ollama_results = await test_ollama(emms)

    # 13. Claude
    claude_results = await test_claude(emms)

    # 14. Full stats
    full_stats = test_full_stats(emms)

    # Generate report
    report = generate_report(ollama_results, claude_results, full_stats)

    # Save report
    report_path = Path(__file__).resolve().parent / "TEST_REPORT.md"
    report_path.write_text(report)
    print(f"\n{'='*70}")
    print(f"  Report saved to: {report_path}")
    print(f"{'='*70}")

    # Save raw log
    log_path = Path(__file__).resolve().parent / "test_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Raw log saved to: {log_path}")

    print(f"\n  TOTAL: {SECTION_NUM} sections, {len(LOG)} log entries, {sum(1 for e in LOG if e['error'])} errors")
    print()


if __name__ == "__main__":
    asyncio.run(main())
