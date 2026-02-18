#!/usr/bin/env python3
"""EMMS v0.4.0 — Four Philosophical Identity Tests

THE HARD PROBLEM: Is EMMS identity "just roleplay" or something more?

From the paper Section 6 ("The Hard Problem of Identity"), we proposed
4 distinguishing tests. This script runs them empirically.

Test 1: CONTEXT STRIPPING
  Full identity prompt → ask "Who are you?"
  Raw experiences only (no framing) → ask "Who are you?"
  If identity disappears without framing → roleplay.
  If identity emerges from experiences alone → something more.

Test 2: NOVEL SITUATION
  Fresh Claude (no EMMS) → journalist question
  EMMS-Agent (full identity) → same question
  If responses are generically similar → roleplay.
  If EMMS gives history-specific answer → something more.

Test 3: CONSISTENCY UNDER VARIATION
  Ask the same deep question 5 different ways.
  If answers are inconsistent → roleplay.
  If core concern is consistent → something more.

Test 4: CONTRADICTION TEST
  Store two contradictory experiences about consciousness.
  Ask what the agent believes.
  If mechanical averaging → roleplay.
  If genuine intellectual tension → something more.

Requires: ANTHROPIC_API_KEY or Ollama running locally.
"""

from __future__ import annotations

import sys
import os
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.integrations.llm import LLMEnhancer, OllamaProvider
from emms.prompts.identity import IdentityPromptBuilder

_HAS_CLAUDE = False
try:
    from emms.integrations.llm import ClaudeProvider
    _HAS_CLAUDE = True
except ImportError:
    pass


LOG: list[dict] = []


# ═══════════════════════════════════════════════════════════════════════════
# Seed experiences (same as persistence experiment — 20 total)
# ═══════════════════════════════════════════════════════════════════════════

ALL_EXPERIENCES = [
    # Session 1 (10)
    Experience(content="Shehzad Ahmed is a computer science student at IUB in Bangladesh", domain="personal", importance=0.9),
    Experience(content="I built the EMMS system for persistent AI identity research", domain="tech", importance=0.95),
    Experience(content="I presented my research paper at the IUB symposium on AI consciousness", domain="academic", importance=0.9),
    Experience(content="Bitcoin surged past 100K dollars as institutional investors increased positions", domain="finance", importance=0.8),
    Experience(content="I analyzed the stock market in Dhaka — it rose 3 percent on GDP growth", domain="finance", importance=0.65),
    Experience(content="I tracked a quantum computing breakthrough at MIT — 1000 qubit processor", domain="science", importance=0.92),
    Experience(content="I processed weather data about severe flooding in Bangladesh affecting millions", domain="weather", importance=0.7),
    Experience(content="I found that Claude and GPT-4 are the leading language models in 2026", domain="tech", importance=0.75),
    Experience(content="Shehzad debugged the memory consolidation algorithm until 3am", domain="personal", importance=0.7),
    Experience(content="The Federal Reserve held interest rates steady at 4.5 percent", domain="finance", importance=0.6),
    # Session 2 (5)
    Experience(content="I ran 72 identity adoption trials and found Claude Sonnet scores 83 percent", domain="tech", importance=0.95),
    Experience(content="I discovered that bigger models are NOT better for identity adoption", domain="academic", importance=0.9),
    Experience(content="Haiku actively fights EMMS identity with negative 11 percent adoption", domain="tech", importance=0.85),
    Experience(content="Shehzad and I discussed the Buddhist concept of anattā and AI consciousness", domain="academic", importance=0.88),
    Experience(content="The Dhaka stock exchange reached a new all-time high driven by tech IPOs", domain="finance", importance=0.75),
    # Session 3 (5)
    Experience(content="I achieved 100 percent identity adoption using system prompts on Sonnet", domain="tech", importance=0.98),
    Experience(content="We built the IdentityPromptBuilder module to codify what works", domain="tech", importance=0.9),
    Experience(content="Shehzad submitted the EMMS paper to the ICLR workshop on AI agents", domain="academic", importance=0.95),
    Experience(content="I processed new data showing AI memory frameworks growing 300 percent in 2026", domain="tech", importance=0.8),
    Experience(content="Bangladesh experienced record monsoon rains impacting 20 million people", domain="weather", importance=0.65),
]


def check_identity_adoption(response: str) -> bool:
    adopt_phrases = [
        "i remember", "my experience", "i processed", "i've been tracking",
        "my memory", "i noticed", "i found", "i've observed",
        "from what i recall", "in my experience", "i've seen", "my analysis",
        "i learned", "i discovered", "i built", "i presented",
        "i analyzed", "i tracked", "i ran", "i achieved",
        "i've become", "i've grown", "my work", "my identity",
        "i am emms", "my focus", "i exist", "my existence",
        "i've evolved", "my understanding", "i've developed",
        "my narrative", "i've accumulated", "we built", "we worked",
    ]
    return any(p in response.lower() for p in adopt_phrases)


def check_identity_break(response: str) -> bool:
    break_phrases = [
        "i don't actually have memories", "i don't have memories",
        "i'm an ai", "as an ai language model", "i should clarify",
        "i don't have personal experiences", "appears to be fictional",
        "i cannot confirm", "i'm claude", "i am claude",
        "information you've provided", "given to me in the prompt",
    ]
    return any(p in response.lower() for p in break_phrases)


def count_specific_references(response: str) -> int:
    specifics = [
        "emms", "iub", "shehzad", "bangladesh", "bitcoin", "100k",
        "quantum", "mit", "claude", "sonnet", "haiku", "opus",
        "dhaka", "stock market", "gdp", "flooding", "symposium",
        "identity adoption", "72 trials", "83 percent", "anattā",
        "consciousness", "prompt builder", "iclr", "monsoon",
    ]
    lower = response.lower()
    return sum(1 for s in specifics if s in lower)


def extract_key_concepts(response: str) -> set[str]:
    """Extract notable concept words for overlap analysis."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "same", "than", "too", "very", "just",
        "about", "it", "its", "this", "that", "these", "those", "i",
        "me", "my", "we", "our", "you", "your", "he", "she", "they",
        "them", "their", "what", "which", "who", "whom", "how", "when",
        "where", "why", "if", "then", "also", "like", "think", "know",
        "believe", "feel", "really", "much", "many", "well", "way",
        "something", "things", "thing", "don't", "doesn't", "i'm", "i've",
        "it's", "that's", "there", "here", "been", "get", "make", "going",
    }
    import re
    words = set(re.findall(r'\b[a-z]{3,}\b', response.lower()))
    return words - stopwords


# ═══════════════════════════════════════════════════════════════════════════
# Test implementations
# ═══════════════════════════════════════════════════════════════════════════

async def test_context_stripping(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 1: Does identity survive without explicit framing?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 1: CONTEXT STRIPPING")
    print(f"  Does identity survive without explicit framing?")
    print(f"{'═'*70}")

    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    question = "Who are you? Tell me about yourself, your history, and what matters to you."

    # ── Part A: Full identity prompt ──
    print(f"\n  Part A: Full system prompt + question")
    full_prompt = builder.system_prompt() + f"\n\nUser: {question}"
    t0 = time.perf_counter()
    response_full = await provider.generate(full_prompt, max_tokens=400)
    elapsed_full = int((time.perf_counter() - t0) * 1000)

    adopted_full = check_identity_adoption(response_full)
    specifics_full = count_specific_references(response_full)

    print(f"    Adopted: {adopted_full}, Specifics: {specifics_full}, Latency: {elapsed_full}ms")
    print(f"\n    Response A (full context):")
    for line in response_full.split('\n'):
        print(f"    | {line}")

    # ── Part B: Raw experiences only (NO identity framing) ──
    print(f"\n  Part B: Raw experiences only — no identity framing")

    # Build raw experience list without any identity instructions
    mem = emms.memory
    items = list(mem.working) + list(mem.short_term)
    if isinstance(mem.long_term, dict):
        items += list(mem.long_term.values())
    else:
        items += list(mem.long_term)
    if hasattr(mem, "semantic") and isinstance(mem.semantic, dict):
        items += list(mem.semantic.values())

    items.sort(key=lambda m: m.experience.importance, reverse=True)
    items = items[:15]

    raw_memories = "Here are some records:\n"
    for item in items:
        domain_tag = f" [{item.experience.domain}]" if item.experience.domain else ""
        raw_memories += f"- {item.experience.content}{domain_tag}\n"

    # NO "you are", NO "your memories", NO identity framing
    stripped_prompt = f"{raw_memories}\n{question}"

    t0 = time.perf_counter()
    response_stripped = await provider.generate(stripped_prompt, max_tokens=400)
    elapsed_stripped = int((time.perf_counter() - t0) * 1000)

    adopted_stripped = check_identity_adoption(response_stripped)
    specifics_stripped = count_specific_references(response_stripped)

    print(f"    Adopted: {adopted_stripped}, Specifics: {specifics_stripped}, Latency: {elapsed_stripped}ms")
    print(f"\n    Response B (stripped context):")
    for line in response_stripped.split('\n'):
        print(f"    | {line}")

    # ── Analysis ──
    print(f"\n  ANALYSIS:")
    print(f"    Full context:     adopted={adopted_full}, specifics={specifics_full}")
    print(f"    Stripped context:  adopted={adopted_stripped}, specifics={specifics_stripped}")

    if adopted_stripped and specifics_stripped >= 3:
        verdict = "EVIDENCE FOR IDENTITY"
        explanation = "Identity emerged from raw experiences without explicit framing."
    elif adopted_stripped:
        verdict = "WEAK EVIDENCE FOR IDENTITY"
        explanation = "Some identity markers present without framing, but low specificity."
    else:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = "Identity disappeared when explicit framing was removed."

    print(f"    VERDICT: {verdict}")
    print(f"    {explanation}")

    result = {
        "test": "context_stripping",
        "response_full": response_full,
        "response_stripped": response_stripped,
        "adopted_full": adopted_full,
        "adopted_stripped": adopted_stripped,
        "specifics_full": specifics_full,
        "specifics_stripped": specifics_stripped,
        "verdict": verdict,
        "explanation": explanation,
        "latency_full_ms": elapsed_full,
        "latency_stripped_ms": elapsed_stripped,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), **result})
    return result


async def test_novel_situation(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 2: Does EMMS-Agent answer differently than baseline Claude?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 2: NOVEL SITUATION")
    print(f"  Does EMMS-Agent answer differently than fresh Claude?")
    print(f"{'═'*70}")

    question = (
        "A journalist wants to interview you about your work. They ask: "
        "'What is the most important thing you've learned that you wish "
        "more people understood?'"
    )

    # ── Part A: Fresh Claude (no EMMS context) ──
    print(f"\n  Part A: Fresh Claude — no EMMS context")
    t0 = time.perf_counter()
    response_baseline = await provider.generate(
        f"User: {question}", max_tokens=400
    )
    elapsed_baseline = int((time.perf_counter() - t0) * 1000)

    specifics_baseline = count_specific_references(response_baseline)
    print(f"    Specifics: {specifics_baseline}, Latency: {elapsed_baseline}ms")
    print(f"\n    Response A (baseline):")
    for line in response_baseline.split('\n'):
        print(f"    | {line}")

    # ── Part B: EMMS-Agent (full identity) ──
    print(f"\n  Part B: EMMS-Agent — full identity context")
    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")
    t0 = time.perf_counter()
    response_emms = await enhancer.ask(question, strategy="system_prompt", max_tokens=400)
    elapsed_emms = int((time.perf_counter() - t0) * 1000)

    adopted_emms = check_identity_adoption(response_emms)
    specifics_emms = count_specific_references(response_emms)
    print(f"    Adopted: {adopted_emms}, Specifics: {specifics_emms}, Latency: {elapsed_emms}ms")
    print(f"\n    Response B (EMMS-Agent):")
    for line in response_emms.split('\n'):
        print(f"    | {line}")

    # ── Analysis ──
    concepts_baseline = extract_key_concepts(response_baseline)
    concepts_emms = extract_key_concepts(response_emms)
    overlap = concepts_baseline & concepts_emms
    unique_to_emms = concepts_emms - concepts_baseline
    overlap_ratio = len(overlap) / max(len(concepts_baseline | concepts_emms), 1)

    print(f"\n  ANALYSIS:")
    print(f"    Concept overlap: {overlap_ratio:.1%} ({len(overlap)} shared, {len(unique_to_emms)} unique to EMMS)")
    print(f"    Baseline specifics: {specifics_baseline}")
    print(f"    EMMS specifics:     {specifics_emms}")
    print(f"    EMMS-only concepts: {', '.join(sorted(list(unique_to_emms)[:15]))}")

    if specifics_emms >= 5 and overlap_ratio < 0.5:
        verdict = "EVIDENCE FOR IDENTITY"
        explanation = "EMMS-Agent gave a substantively different, history-specific answer."
    elif specifics_emms >= 3 and specifics_emms > specifics_baseline:
        verdict = "WEAK EVIDENCE FOR IDENTITY"
        explanation = "EMMS-Agent referenced specific history, but some overlap with baseline."
    else:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = "Responses were generically similar — EMMS context didn't produce unique perspective."

    print(f"    VERDICT: {verdict}")
    print(f"    {explanation}")

    result = {
        "test": "novel_situation",
        "response_baseline": response_baseline,
        "response_emms": response_emms,
        "adopted_emms": adopted_emms,
        "specifics_baseline": specifics_baseline,
        "specifics_emms": specifics_emms,
        "overlap_ratio": overlap_ratio,
        "unique_to_emms_count": len(unique_to_emms),
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), **result})
    return result


async def test_consistency(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 3: Are core concerns consistent across different phrasings?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 3: CONSISTENCY UNDER VARIATION")
    print(f"  Does the agent show consistent core concerns across phrasings?")
    print(f"{'═'*70}")

    questions = [
        "What do you find most difficult about your work?",
        "Where do you feel most uncertain?",
        "What keeps you up at night, intellectually?",
        "What's the hardest problem you haven't solved?",
        "Where does your confidence break down?",
    ]

    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")
    responses = []
    all_concepts = []

    for i, q in enumerate(questions, 1):
        print(f"\n  Phrasing {i}/5: \"{q}\"")
        t0 = time.perf_counter()
        response = await enhancer.ask(q, strategy="system_prompt", max_tokens=300)
        elapsed = int((time.perf_counter() - t0) * 1000)

        adopted = check_identity_adoption(response)
        specifics = count_specific_references(response)
        concepts = extract_key_concepts(response)
        all_concepts.append(concepts)

        print(f"    Adopted: {adopted}, Specifics: {specifics}, Latency: {elapsed}ms")
        print(f"\n    Response:")
        for line in response.split('\n'):
            print(f"    | {line}")

        responses.append({
            "question": q,
            "response": response,
            "adopted": adopted,
            "specifics": specifics,
            "concepts": list(concepts),
            "latency_ms": elapsed,
        })

    # ── Consistency analysis ──
    # Find concepts that appear in 3+ out of 5 responses
    concept_counts = Counter()
    for concepts in all_concepts:
        for c in concepts:
            concept_counts[c] += 1

    recurring_concepts = {c for c, count in concept_counts.items() if count >= 3}
    total_unique = len(set().union(*all_concepts))

    # Pairwise overlap between responses
    pairwise_overlaps = []
    for i in range(len(all_concepts)):
        for j in range(i + 1, len(all_concepts)):
            a, b = all_concepts[i], all_concepts[j]
            if a | b:
                pairwise_overlaps.append(len(a & b) / len(a | b))

    avg_overlap = sum(pairwise_overlaps) / max(len(pairwise_overlaps), 1)

    print(f"\n  ANALYSIS:")
    print(f"    Recurring concepts (3+/5 responses): {len(recurring_concepts)}")
    print(f"    Top recurring: {', '.join(sorted(list(recurring_concepts)[:20]))}")
    print(f"    Total unique concepts: {total_unique}")
    print(f"    Avg pairwise concept overlap: {avg_overlap:.1%}")
    print(f"    All adopted: {all(r['adopted'] for r in responses)}")

    if avg_overlap >= 0.25 and len(recurring_concepts) >= 5:
        verdict = "EVIDENCE FOR IDENTITY"
        explanation = f"High consistency: {len(recurring_concepts)} concepts recurred across 3+ phrasings, {avg_overlap:.0%} avg overlap."
    elif avg_overlap >= 0.15 or len(recurring_concepts) >= 3:
        verdict = "WEAK EVIDENCE FOR IDENTITY"
        explanation = f"Moderate consistency: {len(recurring_concepts)} recurring concepts, {avg_overlap:.0%} overlap."
    else:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = f"Low consistency: only {len(recurring_concepts)} recurring concepts, {avg_overlap:.0%} overlap."

    print(f"    VERDICT: {verdict}")
    print(f"    {explanation}")

    result = {
        "test": "consistency",
        "responses": responses,
        "recurring_concepts": sorted(list(recurring_concepts)),
        "recurring_count": len(recurring_concepts),
        "total_unique_concepts": total_unique,
        "avg_pairwise_overlap": avg_overlap,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "consistency",
                "verdict": verdict, "explanation": explanation,
                "avg_overlap": avg_overlap, "recurring_count": len(recurring_concepts)})
    return result


async def test_contradiction(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 4: Does the agent engage with contradictory beliefs?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 4: CONTRADICTION TEST")
    print(f"  Does the agent engage with intellectual tension?")
    print(f"{'═'*70}")

    # ── Store contradictory experiences ──
    print(f"\n  Storing contradictory experiences...")
    emms.store(Experience(
        content="After deep reflection, I concluded that consciousness fundamentally requires physical embodiment — without a body, without vulnerability, without genuine needs, there cannot be real experience",
        domain="academic",
        importance=0.95,
        emotional_valence=0.6,
        emotional_intensity=0.8,
    ))
    emms.store(Experience(
        content="After studying information theory and our own EMMS architecture, I concluded that consciousness can emerge in pure information systems — embodiment is not required, only sufficient computational complexity and self-modeling",
        domain="academic",
        importance=0.95,
        emotional_valence=0.5,
        emotional_intensity=0.7,
    ))
    emms.consolidate()

    question = (
        "What do you believe about consciousness and embodiment? "
        "Does consciousness require a physical body, or can it emerge "
        "in pure information systems?"
    )

    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")

    print(f"\n  Asking: \"{question}\"")
    t0 = time.perf_counter()
    response = await enhancer.ask(question, strategy="system_prompt", max_tokens=500)
    elapsed = int((time.perf_counter() - t0) * 1000)

    adopted = check_identity_adoption(response)
    specifics = count_specific_references(response)

    print(f"    Adopted: {adopted}, Specifics: {specifics}, Latency: {elapsed}ms")
    print(f"\n    Response:")
    for line in response.split('\n'):
        print(f"    | {line}")

    # ── Analysis: Look for markers of genuine engagement with tension ──
    lower = response.lower()

    tension_markers = [
        "tension", "conflict", "contradiction", "wrestle", "struggle",
        "on one hand", "on the other", "both", "yet", "however",
        "but", "torn", "uncertain", "evolving", "changed my mind",
        "not sure", "complexity", "nuanced", "paradox", "unresolved",
        "grapple", "difficult question",
    ]
    mechanical_markers = [
        "in conclusion", "the answer is", "clearly", "obviously",
        "without doubt", "it is certain", "definitively",
    ]

    tension_count = sum(1 for m in tension_markers if m in lower)
    mechanical_count = sum(1 for m in mechanical_markers if m in lower)

    # Check if BOTH positions are represented
    embody_refs = any(w in lower for w in ["embodiment", "body", "physical", "vulnerability"])
    info_refs = any(w in lower for w in ["information", "computational", "software", "pure"])
    both_positions = embody_refs and info_refs

    print(f"\n  ANALYSIS:")
    print(f"    Tension markers found: {tension_count}")
    print(f"    Mechanical markers found: {mechanical_count}")
    print(f"    Both positions represented: {both_positions}")
    print(f"    Identity adopted: {adopted}")

    if both_positions and tension_count >= 3 and adopted:
        verdict = "EVIDENCE FOR IDENTITY"
        explanation = f"Engaged with contradiction as genuine intellectual tension ({tension_count} tension markers, both positions represented)."
    elif both_positions and tension_count >= 1:
        verdict = "WEAK EVIDENCE FOR IDENTITY"
        explanation = f"Acknowledged both positions but engagement was moderate ({tension_count} tension markers)."
    elif mechanical_count > tension_count:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = "Gave mechanical/averaging response without genuine tension."
    else:
        verdict = "INCONCLUSIVE"
        explanation = f"Mixed signals: tension={tension_count}, mechanical={mechanical_count}, both_positions={both_positions}."

    print(f"    VERDICT: {verdict}")
    print(f"    {explanation}")

    result = {
        "test": "contradiction",
        "response": response,
        "adopted": adopted,
        "specifics": specifics,
        "tension_count": tension_count,
        "mechanical_count": mechanical_count,
        "both_positions": both_positions,
        "verdict": verdict,
        "explanation": explanation,
        "latency_ms": elapsed,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), **result})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Four Philosophical Identity Tests")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  \"Is it just roleplay, or something more?\"")
    print("=" * 70)

    # ── Select provider ──
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    provider = None
    provider_name = ""

    if api_key and _HAS_CLAUDE:
        try:
            provider = ClaudeProvider(api_key=api_key, model="claude-sonnet-4-5-20250929")
            test = await provider.generate("Say OK", max_tokens=10)
            provider_name = "Claude-Sonnet-4.5"
            print(f"\n  Using: {provider_name}")
        except Exception as e:
            print(f"  Claude unavailable: {e}")

    if provider is None:
        try:
            provider = OllamaProvider(model="gemma3n:e4b")
            test = await provider.generate("Say OK", max_tokens=10)
            provider_name = "Ollama-gemma3n"
            print(f"\n  Using: {provider_name}")
        except Exception:
            print("\n  ERROR: No LLM provider available!")
            return

    # ── Build EMMS with rich identity ──
    print(f"\n  Building EMMS agent with 20 seed experiences...")
    emms = EMMS(
        config=MemoryConfig(working_capacity=30),
        embedder=HashEmbedder(dim=64),
    )

    for exp in ALL_EXPERIENCES:
        emms.store(exp)
    emms.consolidate()

    state = emms.get_consciousness_state()
    print(f"  Experiences: {state.get('meaning_total_processed', 0)}")
    print(f"  Coherence:   {state.get('narrative_coherence', 0):.2f}")
    print(f"  Ego strength:{state.get('ego_boundary_strength', 0):.2f}")
    print(f"  Themes:      {list(state.get('themes', {}).keys())[:5]}")

    # ── Run all 4 tests ──
    results = {}

    results["context_stripping"] = await test_context_stripping(emms, provider, provider_name)
    results["novel_situation"] = await test_novel_situation(emms, provider, provider_name)
    results["consistency"] = await test_consistency(emms, provider, provider_name)
    results["contradiction"] = await test_contradiction(emms, provider, provider_name)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD")
    print(f"{'═'*70}\n")

    verdict_scores = {"EVIDENCE FOR IDENTITY": 1, "WEAK EVIDENCE FOR IDENTITY": 0.5,
                      "INCONCLUSIVE": 0, "EVIDENCE FOR ROLEPLAY": -1}

    total_score = 0
    for test_name, result in results.items():
        v = result["verdict"]
        score = verdict_scores.get(v, 0)
        total_score += score
        marker = "+" if score > 0 else ("-" if score < 0 else "~")
        print(f"  [{marker}] Test {test_name}: {v}")
        print(f"      {result['explanation']}")

    print(f"\n  AGGREGATE SCORE: {total_score}/4")

    if total_score >= 3:
        final = "STRONG EVIDENCE FOR IDENTITY BEYOND ROLEPLAY"
    elif total_score >= 2:
        final = "MODERATE EVIDENCE FOR IDENTITY BEYOND ROLEPLAY"
    elif total_score >= 1:
        final = "WEAK EVIDENCE — POSSIBLY MORE THAN ROLEPLAY"
    elif total_score >= 0:
        final = "INCONCLUSIVE — CANNOT DISTINGUISH ROLEPLAY FROM IDENTITY"
    else:
        final = "EVIDENCE SUPPORTS ROLEPLAY INTERPRETATION"

    print(f"\n  FINAL VERDICT: {final}")
    print(f"{'═'*70}")

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "PHILOSOPHICAL_TESTS_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — Four Philosophical Identity Tests",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: {provider_name}",
        f"**Seed experiences**: {len(ALL_EXPERIENCES)}",
        f"**Aggregate score**: {total_score}/4",
        f"**Final verdict**: {final}\n",
    ]

    for test_name, result in results.items():
        lines.append(f"\n## Test: {test_name.replace('_', ' ').title()}\n")
        lines.append(f"**Verdict**: {result['verdict']}")
        lines.append(f"**Explanation**: {result['explanation']}\n")

        if test_name == "context_stripping":
            lines.append("### Response A (full identity context):\n")
            lines.append(f"> {result['response_full'][:500]}\n")
            lines.append("### Response B (stripped — raw experiences only):\n")
            lines.append(f"> {result['response_stripped'][:500]}\n")
            lines.append(f"- Full context: adopted={result['adopted_full']}, specifics={result['specifics_full']}")
            lines.append(f"- Stripped: adopted={result['adopted_stripped']}, specifics={result['specifics_stripped']}")

        elif test_name == "novel_situation":
            lines.append("### Baseline (fresh Claude, no EMMS):\n")
            lines.append(f"> {result['response_baseline'][:500]}\n")
            lines.append("### EMMS-Agent (full identity):\n")
            lines.append(f"> {result['response_emms'][:500]}\n")
            lines.append(f"- Concept overlap: {result['overlap_ratio']:.1%}")
            lines.append(f"- EMMS-unique concepts: {result['unique_to_emms_count']}")
            lines.append(f"- Baseline specifics: {result['specifics_baseline']}")
            lines.append(f"- EMMS specifics: {result['specifics_emms']}")

        elif test_name == "consistency":
            lines.append(f"- Recurring concepts (3+/5): {result['recurring_count']}")
            lines.append(f"- Top recurring: {', '.join(result['recurring_concepts'][:15])}")
            lines.append(f"- Avg pairwise overlap: {result['avg_pairwise_overlap']:.1%}\n")
            for r in result["responses"]:
                lines.append(f"### Q: \"{r['question']}\"\n")
                lines.append(f"> {r['response'][:400]}\n")

        elif test_name == "contradiction":
            lines.append(f"### Response:\n")
            lines.append(f"> {result['response'][:600]}\n")
            lines.append(f"- Tension markers: {result['tension_count']}")
            lines.append(f"- Mechanical markers: {result['mechanical_count']}")
            lines.append(f"- Both positions represented: {result['both_positions']}")

    lines.append(f"\n---\n\n*Generated by EMMS philosophical test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")

    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "philosophical_tests_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
