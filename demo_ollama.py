#!/usr/bin/env python3
"""EMMS v0.4.0 + Ollama Live Demo

Tests the full EMMS system with a local LLM (Ollama) for:
1. Basic memory store/retrieve
2. Graph memory (entity extraction)
3. Consciousness & narrative
4. LLM-enhanced enrichment (via Ollama)
5. LLM-enhanced entity extraction
6. LLM narrative generation
7. Memory summarisation
8. Persistence (save/load)
"""

import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.integrations.llm import OllamaProvider, LLMEnhancer


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    model = "gemma3n:e4b"
    print(f"EMMS v0.4.0 + Ollama ({model}) Live Demo")
    print(f"{'='*60}")

    # ── 1. Initialize EMMS ──────────────────────────────────────
    section("1. Initialize EMMS with all subsystems")
    emms = EMMS(
        config=MemoryConfig(working_capacity=10),
        embedder=HashEmbedder(dim=64),
    )
    print(f"  EMMS initialized")
    print(f"  Consciousness: {emms._consciousness_enabled}")
    print(f"  Graph memory:  {emms.graph is not None}")
    print(f"  Event bus:     {emms.events is not None}")

    # ── 2. Store experiences ────────────────────────────────────
    section("2. Store experiences (with consciousness + graph)")
    experiences = [
        Experience(content="Shehzad Ahmed is a computer science student at IUB in Bangladesh", domain="personal", importance=0.9),
        Experience(content="EMMS is a memory management system built for persistent AI identity", domain="tech", importance=0.95),
        Experience(content="Bitcoin surged past 100K dollars as institutional investors poured in", domain="finance", importance=0.8),
        Experience(content="DeepSeek released their R1 reasoning model with open weights", domain="tech", importance=0.85),
        Experience(content="Bangladesh experienced severe flooding in the eastern districts", domain="weather", importance=0.7),
        Experience(content="Shehzad presented his EMMS research paper at the university symposium", domain="academic", importance=0.9),
        Experience(content="Claude and GPT-4 are the leading large language models in 2026", domain="tech", importance=0.75),
        Experience(content="The stock market in Dhaka rose 3% on positive GDP growth data", domain="finance", importance=0.65),
    ]

    events_fired = []
    emms.events.on("memory.stored", lambda d: events_fired.append(d))

    for exp in experiences:
        result = emms.store(exp)
    print(f"  Stored {len(experiences)} experiences")
    print(f"  Events fired: {len(events_fired)}")
    print(f"  Memory stats: {emms.stats['memory']}")

    # ── 3. Retrieval ────────────────────────────────────────────
    section("3. Retrieve memories")
    queries = ["Shehzad research", "Bitcoin price", "weather Bangladesh"]
    for q in queries:
        results = emms.retrieve(q, max_results=3)
        print(f"  Query: '{q}'")
        for r in results[:2]:
            print(f"    -> [{r.score:.3f}] {r.memory.experience.content[:70]}...")
        print()

    # ── 4. Graph memory queries ─────────────────────────────────
    section("4. Graph memory — entity queries")
    graph_stats = emms.graph.size
    print(f"  Entities: {graph_stats['entities']}")
    print(f"  Relationships: {graph_stats['relationships']}")

    for entity in ["shehzad", "emms", "bitcoin"]:
        result = emms.query_entity(entity)
        if result["found"]:
            neighbors = result["neighbors"][:5]
            print(f"  '{entity}' -> neighbors: {neighbors}")
        else:
            print(f"  '{entity}' -> not found")

    # ── 5. Consciousness ────────────────────────────────────────
    section("5. Consciousness state & narrative")
    state = emms.get_consciousness_state()
    print(f"  Coherence:  {state['narrative_coherence']:.3f}")
    print(f"  Themes:     {dict(list(state['themes'].items())[:5])}")
    print(f"  Ego strength: {state['ego_boundary_strength']:.3f}")

    narrative = emms.get_narrative("EMMS-Agent")
    print(f"\n  Third-person narrative:")
    print(f"  {narrative[:200]}...")

    fp_narrative = emms.get_first_person_narrative()
    print(f"\n  First-person narrative:")
    print(f"  {fp_narrative[:200]}...")

    # ── 6. Pattern detection ────────────────────────────────────
    section("6. Pattern detection")
    patterns = emms.detect_patterns()
    print(f"  Domain dominant: {patterns['domain'].get('dominant', 'N/A')}")
    print(f"  Domain distribution: {patterns['domain'].get('distribution', {})}")
    if patterns["content"]["concepts"]:
        top_concepts = patterns["content"]["concepts"][:5]
        print(f"  Top concepts: {[c['term'] for c in top_concepts]}")

    # ── 7. Ollama LLM integration ───────────────────────────────
    section(f"7. Ollama LLM integration ({model})")
    provider = OllamaProvider(model=model)
    enhancer = LLMEnhancer(provider)

    # Test basic generation first
    print("  Testing Ollama connection...")
    t0 = time.perf_counter()
    test_response = await provider.generate("Say hello in one sentence.", max_tokens=50)
    latency = (time.perf_counter() - t0) * 1000
    print(f"  Ollama response ({latency:.0f}ms): {test_response.strip()[:100]}")

    # 7a. Enrich an experience
    print(f"\n  --- Enriching experience via LLM ---")
    raw_exp = Experience(content="Scientists at CERN discovered a new subatomic particle that could explain dark matter", domain="general")
    print(f"  Before: importance={raw_exp.importance:.2f}, domain='{raw_exp.domain}'")
    t0 = time.perf_counter()
    enriched = await enhancer.enrich_experience(raw_exp)
    latency = (time.perf_counter() - t0) * 1000
    print(f"  After ({latency:.0f}ms): importance={enriched.importance:.2f}, domain='{enriched.domain}', valence={enriched.emotional_valence:.2f}")

    # 7b. LLM entity extraction
    print(f"\n  --- LLM entity extraction ---")
    text = "Shehzad Ahmed built EMMS at Independent University Bangladesh for his research on persistent AI identity."
    t0 = time.perf_counter()
    entities = await enhancer.extract_entities(text)
    latency = (time.perf_counter() - t0) * 1000
    print(f"  Entities ({latency:.0f}ms):")
    for ent in entities[:6]:
        print(f"    - {ent.get('name', '?')} ({ent.get('type', '?')})")

    # 7c. LLM narrative generation
    print(f"\n  --- LLM narrative generation ---")
    context = {
        "themes": dict(list(state["themes"].items())[:5]),
        "traits": emms.narrator.traits if emms.narrator else {},
        "experience_count": len(experiences),
        "domains": list(set(e.domain for e in experiences)),
    }
    t0 = time.perf_counter()
    llm_narrative = await enhancer.generate_narrative(context)
    latency = (time.perf_counter() - t0) * 1000
    print(f"  LLM narrative ({latency:.0f}ms):")
    # Clean up thinking tokens if present
    clean = llm_narrative.strip()
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    print(f"  {clean[:300]}")

    # 7d. Memory summarisation
    print(f"\n  --- LLM memory summarisation ---")
    all_items = list(emms.memory.working) + list(emms.memory.short_term)
    if all_items:
        t0 = time.perf_counter()
        summary = await enhancer.summarize_memories(all_items)
        latency = (time.perf_counter() - t0) * 1000
        clean_summary = summary.strip()
        if "</think>" in clean_summary:
            clean_summary = clean_summary.split("</think>")[-1].strip()
        print(f"  Summary ({latency:.0f}ms):")
        print(f"  {clean_summary[:300]}")

    # ── 8. Persistence ──────────────────────────────────────────
    section("8. Persistence — save & reload")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "emms_state.json"
        t0 = time.perf_counter()
        emms.save(memory_path=path)
        save_ms = (time.perf_counter() - t0) * 1000

        file_kb = path.stat().st_size / 1024
        print(f"  Saved: {save_ms:.2f}ms ({file_kb:.1f}KB)")

        emms2 = EMMS(
            config=MemoryConfig(working_capacity=10),
            embedder=HashEmbedder(dim=64),
        )
        t0 = time.perf_counter()
        emms2.load(memory_path=path)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Loaded: {load_ms:.2f}ms ({emms2.stats['memory']['total']} items)")

        # Verify retrieval works after load
        results = emms2.retrieve("Shehzad", max_results=3)
        print(f"  Post-load retrieval for 'Shehzad': {len(results)} results")
        if results:
            print(f"    -> {results[0].memory.experience.content[:70]}...")

    # ── Summary ─────────────────────────────────────────────────
    section("DEMO COMPLETE")
    print(f"  EMMS v0.4.0 — all systems operational")
    print(f"  Memory: {emms.stats['memory']['total']} items across 4 tiers")
    print(f"  Graph: {emms.graph.size['entities']} entities, {emms.graph.size['relationships']} relationships")
    print(f"  Consciousness: coherence={state['narrative_coherence']:.3f}")
    print(f"  Ollama: {model} — enrichment, entities, narrative, summary all working")
    print(f"  Persistence: save/load roundtrip verified")
    print()


if __name__ == "__main__":
    asyncio.run(main())
