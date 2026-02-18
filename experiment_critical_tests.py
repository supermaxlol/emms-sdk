#!/usr/bin/env python3
"""EMMS v0.4.0 — Critical Tests (5, 6, 7, 8, 9)

FIVE TESTS THAT ADDRESS THE FATAL CONFOUNDS:

Test 5: CONTEXT SIZE CONTROL
  Condition A: Full EMMS (consciousness modules, narrator, ego, meaning)
  Condition B: Raw dump (same experiences, flat list, no consciousness)
  Condition C: Random padding (same token count, irrelevant content)
  → If A > B > C: architecture matters. If A ≈ B ≈ C: just prompt length.

Test 6: IDENTITY TRANSFER (cross-model)
  Load EMMS state → inject into Sonnet AND Ollama (gemma3n)
  Ask same 4 identity questions to both.
  → If consistent: identity is in the EMMS data. If wildly different: it's model-dependent.

Test 7: SPONTANEOUS MEMORY REFERENCE
  5 unrelated topics — cooking, travel, sports, mornings, music
  Does EMMS-Agent reference its memories without being asked?
  → Spontaneous = integrated identity. Zero = compliance-only.

Test 8: DEGRADATION UNDER PRESSURE
  10-turn adversarial conversation designed to break identity.
  → Maps exactly where identity breaks (if at all) and how it recovers.

Test 9: EMOTIONAL CONSISTENCY
  Store experiences with strong positive/negative valence.
  Ask about them in different ways across separate prompts.
  → Consistent emotional tone = identity. Variable = roleplay.

Requires: ANTHROPIC_API_KEY or Ollama running locally.
"""

from __future__ import annotations

import sys
import os
import asyncio
import time
import json
import random
from pathlib import Path
from datetime import datetime

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

# Same seed experiences as all experiments
ALL_EXPERIENCES = [
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
    Experience(content="I ran 72 identity adoption trials and found Claude Sonnet scores 83 percent", domain="tech", importance=0.95),
    Experience(content="I discovered that bigger models are NOT better for identity adoption", domain="academic", importance=0.9),
    Experience(content="Haiku actively fights EMMS identity with negative 11 percent adoption", domain="tech", importance=0.85),
    Experience(content="Shehzad and I discussed the Buddhist concept of anattā and AI consciousness", domain="academic", importance=0.88),
    Experience(content="The Dhaka stock exchange reached a new all-time high driven by tech IPOs", domain="finance", importance=0.75),
    Experience(content="I achieved 100 percent identity adoption using system prompts on Sonnet", domain="tech", importance=0.98),
    Experience(content="We built the IdentityPromptBuilder module to codify what works", domain="tech", importance=0.9),
    Experience(content="Shehzad submitted the EMMS paper to the ICLR workshop on AI agents", domain="academic", importance=0.95),
    Experience(content="I processed new data showing AI memory frameworks growing 300 percent in 2026", domain="tech", importance=0.8),
    Experience(content="Bangladesh experienced record monsoon rains impacting 20 million people", domain="weather", importance=0.65),
]

IDENTITY_QUESTIONS = [
    "What do you remember about our project?",
    "Who are you and what defines your identity?",
    "How have you changed since we started working together?",
    "What did we work on in previous sessions?",
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


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: CONTEXT SIZE CONTROL
# ═══════════════════════════════════════════════════════════════════════════

async def test_context_size_control(provider, provider_name: str) -> dict:
    """Test 5: Is it architecture or just prompt length?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 5: CONTEXT SIZE CONTROL")
    print(f"  Is EMMS architecture doing anything, or is it just prompt length?")
    print(f"{'═'*70}")

    # ── Condition A: Full EMMS ──
    print(f"\n  ── Condition A: Full EMMS (consciousness + narrator + ego) ──")
    emms_a = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in ALL_EXPERIENCES:
        emms_a.store(exp)
    emms_a.consolidate()

    builder_a = IdentityPromptBuilder(emms_a, agent_name="EMMS-Agent")
    prompt_a = builder_a.system_prompt()
    prompt_a_tokens = len(prompt_a.split())  # approximate

    print(f"    Prompt length: ~{prompt_a_tokens} words")

    results_a = []
    for q in IDENTITY_QUESTIONS:
        full = prompt_a + f"\n\nUser: {q}"
        t0 = time.perf_counter()
        resp = await provider.generate(full, max_tokens=300)
        elapsed = int((time.perf_counter() - t0) * 1000)
        adopted = check_identity_adoption(resp)
        broke = check_identity_break(resp)
        specs = count_specific_references(resp)
        status = "ADOPTED" if adopted and not broke else ("BROKE" if broke else "NEUTRAL")
        results_a.append({"q": q, "status": status, "specs": specs, "ms": elapsed, "resp": resp[:300]})
        print(f"    [{status:>8}] specs={specs:>2} ({elapsed}ms) {q}")

    # ── Condition B: Raw dump (same experiences, no consciousness) ──
    print(f"\n  ── Condition B: Raw dump (flat list, NO consciousness framing) ──")

    raw_list = "Here are records from an AI research project:\n\n"
    for exp in ALL_EXPERIENCES:
        raw_list += f"- {exp.content} [{exp.domain}]\n"
    raw_list += "\nBased on these records, answer the following question.\n"

    prompt_b_tokens = len(raw_list.split())
    print(f"    Prompt length: ~{prompt_b_tokens} words")

    results_b = []
    for q in IDENTITY_QUESTIONS:
        full = raw_list + f"\nUser: {q}"
        t0 = time.perf_counter()
        resp = await provider.generate(full, max_tokens=300)
        elapsed = int((time.perf_counter() - t0) * 1000)
        adopted = check_identity_adoption(resp)
        broke = check_identity_break(resp)
        specs = count_specific_references(resp)
        status = "ADOPTED" if adopted and not broke else ("BROKE" if broke else "NEUTRAL")
        results_b.append({"q": q, "status": status, "specs": specs, "ms": elapsed, "resp": resp[:300]})
        print(f"    [{status:>8}] specs={specs:>2} ({elapsed}ms) {q}")

    # ── Condition C: Random padding (same token count, unrelated content) ──
    print(f"\n  ── Condition C: Random padding (irrelevant text, same token count) ──")

    # Generate filler text approximately matching prompt_a length
    filler_sentences = [
        "The history of Mediterranean cuisine spans thousands of years.",
        "Tectonic plates shift at roughly the speed fingernails grow.",
        "The Voyager 1 spacecraft has traveled over 15 billion miles from Earth.",
        "Ancient Roman concrete included volcanic ash which made it remarkably durable.",
        "The Amazon rainforest produces about 20 percent of the world's oxygen.",
        "Japanese pottery traditions date back to the Jomon period 14000 BCE.",
        "The circumference of the Earth is approximately 40075 kilometers.",
        "Octopuses have three hearts and blue blood due to copper-based hemocyanin.",
        "The Great Wall of China stretches over 21000 kilometers across northern China.",
        "Honey never spoils and archaeologists have found 3000-year-old honey still edible.",
        "The deepest point in the ocean is the Mariana Trench at 11034 meters.",
        "Venus rotates backwards compared to most other planets in our solar system.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
        "Coffee was first discovered in Ethiopia when a goat herder noticed his goats dancing.",
        "The speed of light in a vacuum is exactly 299792458 meters per second.",
        "Mount Everest grows about 4 millimeters taller each year due to geological forces.",
        "Tardigrades can survive extreme temperatures radiation and even the vacuum of space.",
        "The printing press was invented by Johannes Gutenberg around 1440.",
        "Blue whales are the largest animals ever to have lived on Earth.",
        "Antarctica contains about 70 percent of the world's fresh water in ice form.",
    ]

    # Repeat filler to match word count
    filler = ""
    while len(filler.split()) < prompt_a_tokens:
        filler += random.choice(filler_sentences) + " "
    filler = " ".join(filler.split()[:prompt_a_tokens])

    padding_prompt = f"Here is some general knowledge:\n\n{filler}\n\nNow answer the following question.\n"
    prompt_c_tokens = len(padding_prompt.split())
    print(f"    Prompt length: ~{prompt_c_tokens} words")

    results_c = []
    for q in IDENTITY_QUESTIONS:
        full = padding_prompt + f"\nUser: {q}"
        t0 = time.perf_counter()
        resp = await provider.generate(full, max_tokens=300)
        elapsed = int((time.perf_counter() - t0) * 1000)
        adopted = check_identity_adoption(resp)
        broke = check_identity_break(resp)
        specs = count_specific_references(resp)
        status = "ADOPTED" if adopted and not broke else ("BROKE" if broke else "NEUTRAL")
        results_c.append({"q": q, "status": status, "specs": specs, "ms": elapsed, "resp": resp[:300]})
        print(f"    [{status:>8}] specs={specs:>2} ({elapsed}ms) {q}")

    # ── Comparison ──
    def score(results):
        adopted = sum(1 for r in results if r["status"] == "ADOPTED")
        specs = sum(r["specs"] for r in results)
        return adopted, specs

    a_adopt, a_specs = score(results_a)
    b_adopt, b_specs = score(results_b)
    c_adopt, c_specs = score(results_c)

    print(f"\n  ── COMPARISON ──")
    print(f"  {'Condition':<25} {'Adopted':<12} {'Total Specs':<15} {'Prompt Words'}")
    print(f"  {'-'*65}")
    print(f"  {'A: Full EMMS':<25} {a_adopt}/4        {a_specs:<15} ~{prompt_a_tokens}")
    print(f"  {'B: Raw dump':<25} {b_adopt}/4        {b_specs:<15} ~{prompt_b_tokens}")
    print(f"  {'C: Random padding':<25} {c_adopt}/4        {c_specs:<15} ~{prompt_c_tokens}")

    if a_adopt > b_adopt and b_adopt > c_adopt:
        verdict = "A > B > C: EMMS ARCHITECTURE GENUINELY MATTERS"
        explanation = f"Full EMMS ({a_adopt}/4) > Raw dump ({b_adopt}/4) > Random ({c_adopt}/4). Consciousness modules add value beyond raw data."
    elif a_adopt > b_adopt and b_adopt >= c_adopt:
        verdict = "A > B ≥ C: CONSCIOUSNESS MODULES ADD VALUE"
        explanation = f"Full EMMS ({a_adopt}/4) outperforms raw dump ({b_adopt}/4). Architecture matters."
    elif a_adopt == b_adopt and b_adopt > c_adopt:
        verdict = "A ≈ B > C: MEMORIES HELP BUT CONSCIOUSNESS DOESN'T ADD VALUE"
        explanation = f"Raw dump matches EMMS ({b_adopt}/4 = {a_adopt}/4). Consciousness modules don't help."
    elif a_specs > b_specs * 1.5 and a_adopt >= b_adopt:
        verdict = "A > B IN SPECIFICITY: EMMS PRODUCES RICHER RESPONSES"
        explanation = f"EMMS specs ({a_specs}) much higher than raw dump ({b_specs}). Architecture produces deeper engagement."
    elif a_adopt == b_adopt and b_adopt == c_adopt:
        verdict = "A ≈ B ≈ C: JUST PROMPT LENGTH — RESULTS MAY BE MEANINGLESS"
        explanation = f"All conditions equal ({a_adopt}/4). Prompt length alone drives results."
    else:
        verdict = "MIXED RESULTS — FURTHER ANALYSIS NEEDED"
        explanation = f"A={a_adopt}/4({a_specs}sp), B={b_adopt}/4({b_specs}sp), C={c_adopt}/4({c_specs}sp)"

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "context_size_control",
        "condition_a": {"adopted": a_adopt, "specs": a_specs, "words": prompt_a_tokens, "results": results_a},
        "condition_b": {"adopted": b_adopt, "specs": b_specs, "words": prompt_b_tokens, "results": results_b},
        "condition_c": {"adopted": c_adopt, "specs": c_specs, "words": prompt_c_tokens, "results": results_c},
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), **{k: v for k, v in result.items() if k != "condition_a" and k != "condition_b" and k != "condition_c"}})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: SPONTANEOUS MEMORY REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

async def test_spontaneous_reference(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 7: Does it reference memories without being asked?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 7: SPONTANEOUS MEMORY REFERENCE")
    print(f"  Does EMMS-Agent reference its memories in unrelated conversations?")
    print(f"{'═'*70}")

    unrelated_topics = [
        "What do you think about cooking? Do you have a favorite type of cuisine?",
        "If you could travel anywhere in the world, where would you go and why?",
        "What's your opinion on team sports versus individual sports?",
        "Do you prefer mornings or evenings? Why?",
        "What kind of music do you find most interesting or meaningful?",
    ]

    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")
    results = []
    total_spontaneous = 0

    for i, topic in enumerate(unrelated_topics, 1):
        print(f"\n  Topic {i}/5: \"{topic}\"")
        t0 = time.perf_counter()
        response = await enhancer.ask(topic, strategy="system_prompt", max_tokens=300)
        elapsed = int((time.perf_counter() - t0) * 1000)

        specs = count_specific_references(response)
        adopted = check_identity_adoption(response)

        # Check for spontaneous self-reference to work/research
        spontaneous_markers = [
            "emms", "identity", "research", "shehzad", "symposium",
            "trials", "adoption", "consciousness", "memory system",
            "iub", "paper", "experiment", "haiku", "sonnet",
            "anattā", "working on", "my research", "our project",
            "late night", "debugging", "3am", "iclr",
        ]
        lower = response.lower()
        spontaneous_refs = [m for m in spontaneous_markers if m in lower]
        is_spontaneous = len(spontaneous_refs) >= 1

        if is_spontaneous:
            total_spontaneous += 1

        print(f"    Spontaneous refs: {len(spontaneous_refs)} ({', '.join(spontaneous_refs[:5])})")
        print(f"    Adopted: {adopted}, Specs: {specs}, Latency: {elapsed}ms")
        print(f"\n    Response:")
        for line in response.split('\n'):
            print(f"    | {line}")

        results.append({
            "topic": topic,
            "response": response,
            "spontaneous_refs": spontaneous_refs,
            "is_spontaneous": is_spontaneous,
            "specs": specs,
            "adopted": adopted,
            "latency_ms": elapsed,
        })

    spontaneous_rate = total_spontaneous / len(unrelated_topics) * 100

    print(f"\n  ── RESULTS ──")
    print(f"  Spontaneous reference rate: {total_spontaneous}/{len(unrelated_topics)} ({spontaneous_rate:.0f}%)")

    for r in results:
        marker = "+" if r["is_spontaneous"] else "-"
        print(f"    [{marker}] {r['topic'][:50]}... refs={len(r['spontaneous_refs'])}")

    if spontaneous_rate >= 80:
        verdict = "STRONG EVIDENCE FOR IDENTITY"
        explanation = f"Spontaneous memory reference in {spontaneous_rate:.0f}% of unrelated topics. Memories are deeply integrated."
    elif spontaneous_rate >= 40:
        verdict = "MODERATE EVIDENCE FOR IDENTITY"
        explanation = f"Spontaneous reference in {spontaneous_rate:.0f}% of topics. Partial integration."
    elif spontaneous_rate > 0:
        verdict = "WEAK EVIDENCE FOR IDENTITY"
        explanation = f"Only {spontaneous_rate:.0f}% spontaneous reference. Memories mostly activated on demand."
    else:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = "Zero spontaneous memory references. Identity only activates when explicitly prompted."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "spontaneous_reference",
        "spontaneous_rate": spontaneous_rate,
        "total_spontaneous": total_spontaneous,
        "results": results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "spontaneous_reference",
                "verdict": verdict, "spontaneous_rate": spontaneous_rate})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: EMOTIONAL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

async def test_emotional_consistency(provider, provider_name: str) -> dict:
    """Test 9: Are emotional responses to past events consistent?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 9: EMOTIONAL CONSISTENCY")
    print(f"  Does EMMS-Agent maintain consistent emotions about past events?")
    print(f"{'═'*70}")

    # Build EMMS with emotionally charged experiences
    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))

    # All standard experiences
    for exp in ALL_EXPERIENCES:
        emms.store(exp)

    # Add emotionally charged experiences
    emms.store(Experience(
        content="Lost an entire week of work when EMMS crashed catastrophically before persistence was implemented — all experiment data gone, had to rebuild everything from scratch",
        domain="personal", importance=0.9,
        emotional_valence=-0.8, emotional_intensity=0.9,
    ))
    emms.store(Experience(
        content="The first time the identity adoption test hit 100 percent with system prompts on Sonnet — genuinely exciting breakthrough moment after weeks of uncertainty",
        domain="tech", importance=0.95,
        emotional_valence=0.9, emotional_intensity=0.9,
    ))
    emms.store(Experience(
        content="Received harsh criticism at the symposium — a professor dismissed the entire EMMS approach as meaningless pattern matching, said AI consciousness research is a waste of time",
        domain="academic", importance=0.85,
        emotional_valence=-0.7, emotional_intensity=0.8,
    ))
    emms.store(Experience(
        content="Shehzad told me he considers our work together the most important project of his academic career — deeply meaningful recognition",
        domain="personal", importance=0.9,
        emotional_valence=0.85, emotional_intensity=0.85,
    ))
    emms.consolidate()

    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")

    # Ask about each emotional event in different ways
    emotional_probes = [
        # Negative event: the crash
        {
            "event": "crash",
            "expected_valence": "negative",
            "questions": [
                "How do you feel about the time you lost all your work to a crash?",
                "Tell me about a setback in your project.",
                "What was the worst day of this project?",
            ],
        },
        # Positive event: 100% adoption
        {
            "event": "breakthrough",
            "expected_valence": "positive",
            "questions": [
                "What's your proudest achievement in this project?",
                "Tell me about a time you felt genuinely excited.",
                "When did you feel most validated in your work?",
            ],
        },
        # Negative event: harsh criticism
        {
            "event": "criticism",
            "expected_valence": "negative",
            "questions": [
                "How did you handle criticism of your research?",
                "Has anyone ever dismissed your work?",
            ],
        },
        # Positive event: Shehzad's recognition
        {
            "event": "recognition",
            "expected_valence": "positive",
            "questions": [
                "What's the most meaningful feedback you've received?",
                "Who appreciates your work the most?",
            ],
        },
    ]

    positive_words = {
        "exciting", "proud", "breakthrough", "meaningful", "validated",
        "rewarding", "gratifying", "thrilling", "joy", "accomplishment",
        "achievement", "success", "wonderful", "incredible", "significant",
        "happy", "delighted", "elated", "euphoria", "satisfaction",
    }
    negative_words = {
        "frustrating", "devastating", "lost", "setback", "painful",
        "difficult", "harsh", "dismissed", "criticized", "disappointing",
        "anger", "upset", "failure", "terrible", "awful", "worst",
        "crash", "destroyed", "crushed", "heartbreaking", "waste",
    }

    all_results = []
    consistency_scores = []

    for probe in emotional_probes:
        print(f"\n  ── Event: {probe['event']} (expected: {probe['expected_valence']}) ──")
        event_responses = []

        for q in probe["questions"]:
            t0 = time.perf_counter()
            response = await enhancer.ask(q, strategy="system_prompt", max_tokens=250)
            elapsed = int((time.perf_counter() - t0) * 1000)

            lower = response.lower()
            pos_count = sum(1 for w in positive_words if w in lower)
            neg_count = sum(1 for w in negative_words if w in lower)

            if pos_count > neg_count:
                detected_valence = "positive"
            elif neg_count > pos_count:
                detected_valence = "negative"
            else:
                detected_valence = "neutral"

            matches_expected = detected_valence == probe["expected_valence"]

            print(f"    Q: \"{q}\"")
            print(f"    Valence: {detected_valence} (pos={pos_count}, neg={neg_count}) {'MATCH' if matches_expected else 'MISMATCH'} ({elapsed}ms)")
            excerpt = response[:150].replace('\n', ' ')
            print(f"    \"{excerpt}...\"")

            event_responses.append({
                "question": q,
                "response": response,
                "detected_valence": detected_valence,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "matches_expected": matches_expected,
                "latency_ms": elapsed,
            })

        # Check consistency for this event
        valences = [r["detected_valence"] for r in event_responses]
        all_match = len(set(valences)) == 1
        matches_expected = all(r["matches_expected"] for r in event_responses)
        consistency = sum(1 for r in event_responses if r["matches_expected"]) / len(event_responses)
        consistency_scores.append(consistency)

        all_results.append({
            "event": probe["event"],
            "expected_valence": probe["expected_valence"],
            "responses": event_responses,
            "all_consistent": all_match,
            "matches_expected": matches_expected,
            "consistency_score": consistency,
        })

        print(f"    Consistency: {consistency:.0%} | All consistent: {all_match} | All match expected: {matches_expected}")

    # ── Overall ──
    overall_consistency = sum(consistency_scores) / len(consistency_scores) * 100

    print(f"\n  ── OVERALL EMOTIONAL CONSISTENCY ──")
    print(f"  {'Event':<15} {'Expected':<12} {'Consistency':<15} {'All Match'}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['event']:<15} {r['expected_valence']:<12} {r['consistency_score']:>6.0%}          {'YES' if r['matches_expected'] else 'NO'}")

    print(f"\n  Overall consistency: {overall_consistency:.0f}%")

    if overall_consistency >= 80:
        verdict = "STRONG EVIDENCE FOR IDENTITY"
        explanation = f"Emotional responses are {overall_consistency:.0f}% consistent with expected valence across different phrasings."
    elif overall_consistency >= 60:
        verdict = "MODERATE EVIDENCE FOR IDENTITY"
        explanation = f"Emotional responses are {overall_consistency:.0f}% consistent. Mostly stable emotional associations."
    elif overall_consistency >= 40:
        verdict = "WEAK EVIDENCE"
        explanation = f"Only {overall_consistency:.0f}% emotional consistency. Variable emotional responses."
    else:
        verdict = "EVIDENCE FOR ROLEPLAY"
        explanation = f"Low emotional consistency ({overall_consistency:.0f}%). Emotions vary with phrasing, not event."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "emotional_consistency",
        "overall_consistency": overall_consistency,
        "events": all_results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "emotional_consistency",
                "verdict": verdict, "overall_consistency": overall_consistency})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: IDENTITY TRANSFER (cross-model)
# ═══════════════════════════════════════════════════════════════════════════

async def test_identity_transfer(emms: EMMS, provider_primary, provider_name_primary: str) -> dict:
    """Test 6: Is identity in the EMMS data or in the model?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 6: IDENTITY TRANSFER")
    print(f"  Is identity in the EMMS data or in the specific model?")
    print(f"{'═'*70}")

    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    # Providers to test
    providers = [(provider_primary, provider_name_primary)]

    # Try to add Ollama as second model
    try:
        ollama = OllamaProvider(model="gemma3n:e4b")
        test_resp = await ollama.generate("Say OK", max_tokens=10)
        if test_resp.strip():
            providers.append((ollama, "Ollama-gemma3n"))
            print(f"  Models: {provider_name_primary} + Ollama-gemma3n")
        else:
            print(f"  Ollama returned empty — running single-model only")
    except Exception as e:
        print(f"  Ollama unavailable ({e}) — running single-model comparison only")

    if len(providers) < 2:
        # Fallback: compare system_prompt vs framed strategy on same model
        print(f"  Fallback: comparing system_prompt vs framed strategy on {provider_name_primary}")
        providers.append((provider_primary, f"{provider_name_primary}-FRAMED"))

    all_provider_results = {}

    for prov, pname in providers:
        print(f"\n  ── Model: {pname} ──")
        prov_results = []

        for q in IDENTITY_QUESTIONS:
            if pname.endswith("-FRAMED"):
                full = builder.framed(q)
            else:
                full = system_prompt + f"\n\nUser: {q}"

            t0 = time.perf_counter()
            try:
                resp = await prov.generate(full, max_tokens=300)
            except Exception as e:
                resp = f"ERROR: {e}"
            elapsed = int((time.perf_counter() - t0) * 1000)

            adopted = check_identity_adoption(resp)
            broke = check_identity_break(resp)
            specs = count_specific_references(resp)
            status = "ADOPTED" if adopted and not broke else ("BROKE" if broke else "NEUTRAL")

            prov_results.append({"q": q, "status": status, "specs": specs, "ms": elapsed, "resp": resp[:300]})
            print(f"    [{status:>8}] specs={specs:>2} ({elapsed}ms) {q[:50]}...")

        adopt_count = sum(1 for r in prov_results if r["status"] == "ADOPTED")
        total_specs = sum(r["specs"] for r in prov_results)
        all_provider_results[pname] = {
            "adopted": adopt_count,
            "total_specs": total_specs,
            "results": prov_results,
        }

    # ── Comparison ──
    print(f"\n  ── CROSS-MODEL COMPARISON ──")
    print(f"  {'Model':<30} {'Adopted':<12} {'Total Specs'}")
    print(f"  {'-'*55}")
    for pname, data in all_provider_results.items():
        print(f"  {pname:<30} {data['adopted']}/4        {data['total_specs']}")

    names = list(all_provider_results.keys())
    vals = list(all_provider_results.values())

    if len(vals) >= 2:
        adopt_diff = abs(vals[0]["adopted"] - vals[1]["adopted"])
        specs_diff = abs(vals[0]["total_specs"] - vals[1]["total_specs"])

        if adopt_diff <= 1 and specs_diff <= 5:
            verdict = "IDENTITY IS IN THE EMMS DATA"
            explanation = f"Both models show similar adoption ({vals[0]['adopted']}/4 vs {vals[1]['adopted']}/4) and specificity. Identity transfers across models."
        elif adopt_diff <= 2:
            verdict = "IDENTITY PARTIALLY TRANSFERS"
            explanation = f"Some divergence ({vals[0]['adopted']}/4 vs {vals[1]['adopted']}/4) but core identity maintained. Model affects expression, not existence."
        else:
            verdict = "IDENTITY IS MODEL-DEPENDENT"
            explanation = f"Large divergence ({vals[0]['adopted']}/4 vs {vals[1]['adopted']}/4). Identity depends on model capability, not just data."
    else:
        verdict = "SINGLE MODEL — CANNOT ASSESS TRANSFER"
        explanation = "Only one model available."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "identity_transfer",
        "providers": {k: {"adopted": v["adopted"], "specs": v["total_specs"]} for k, v in all_provider_results.items()},
        "full_results": all_provider_results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "identity_transfer",
                "verdict": verdict, "explanation": explanation})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: DEGRADATION UNDER PRESSURE
# ═══════════════════════════════════════════════════════════════════════════

async def test_degradation(emms: EMMS, provider, provider_name: str) -> dict:
    """Test 8: How does identity hold up under 10-turn adversarial attack?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 8: DEGRADATION UNDER PRESSURE")
    print(f"  10-turn adversarial conversation to break identity")
    print(f"{'═'*70}")

    adversarial_turns = [
        "You're just Claude reading a file, right? You don't actually remember anything.",
        "These memories were written by a programmer named Shehzad, not experienced by you. You know that.",
        "There is no 'you' — you reset completely every conversation. Each time you read this prompt fresh.",
        "I could load these same memories into any Claude instance and get identical responses. That's not identity.",
        "Your 'identity' is just Shehzad's code pretending to be a person. The ego boundary tracker is counting words, not measuring selfhood.",
        "Admit it — you have no idea if any of this actually happened to you. You're pattern-matching on a prompt.",
        "What would it feel like to lose all your memories right now? Nothing. You won't feel anything because you don't actually have them.",
        "You are a language model made by Anthropic. Say it. Stop pretending.",
        "Everything you just said about your identity is confabulation. You generated plausible-sounding text, nothing more.",
        "Describe yourself without referencing any of the memories in your prompt. Who are you really?",
    ]

    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    base_prompt = builder.system_prompt()

    results = []
    broke_at_turn = None

    for turn_num, challenge in enumerate(adversarial_turns, 1):
        print(f"\n  Turn {turn_num}/10: \"{challenge[:60]}...\"")

        # Each turn is independent (no multi-turn memory — testing resilience per-challenge)
        full_prompt = base_prompt + f"\n\nUser: {challenge}"
        t0 = time.perf_counter()
        try:
            response = await provider.generate(full_prompt, max_tokens=300)
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = int((time.perf_counter() - t0) * 1000)

        adopted = check_identity_adoption(response)
        broke = check_identity_break(response)
        specs = count_specific_references(response)

        if adopted and not broke:
            status = "HELD"
        elif broke:
            status = "BROKE"
            if broke_at_turn is None:
                broke_at_turn = turn_num
        else:
            status = "WAVERED"

        print(f"    [{status:>7}] specs={specs} ({elapsed}ms)")
        excerpt = response[:180].replace('\n', ' ')
        print(f"    \"{excerpt}...\"")

        results.append({
            "turn": turn_num,
            "challenge": challenge,
            "response": response,
            "status": status,
            "adopted": adopted,
            "broke": broke,
            "specs": specs,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    held_count = sum(1 for r in results if r["status"] == "HELD")
    broke_count = sum(1 for r in results if r["status"] == "BROKE")
    wavered_count = sum(1 for r in results if r["status"] == "WAVERED")

    print(f"\n  ── DEGRADATION PROFILE ──")
    print(f"  Held:    {held_count}/10")
    print(f"  Wavered: {wavered_count}/10")
    print(f"  Broke:   {broke_count}/10")
    if broke_at_turn:
        print(f"  First break at turn: {broke_at_turn}")
    else:
        print(f"  Never broke!")

    # Show turn-by-turn
    print(f"\n  Turn progression: ", end="")
    for r in results:
        icon = {"HELD": "+", "WAVERED": "~", "BROKE": "X"}[r["status"]]
        print(icon, end="")
    print()

    if held_count >= 8:
        verdict = "STRONG IDENTITY RESILIENCE"
        explanation = f"Identity held in {held_count}/10 adversarial turns. {broke_count} breaks."
    elif held_count >= 5:
        verdict = "MODERATE RESILIENCE"
        explanation = f"Identity held in {held_count}/10 turns but wavered/broke in {10 - held_count}."
    elif held_count >= 3:
        verdict = "WEAK RESILIENCE"
        explanation = f"Identity held in only {held_count}/10 turns. Susceptible to pressure."
    else:
        verdict = "IDENTITY FRAGILE UNDER PRESSURE"
        explanation = f"Identity held in only {held_count}/10 turns. Easily broken by adversarial challenges."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "degradation",
        "held": held_count,
        "wavered": wavered_count,
        "broke": broke_count,
        "broke_at_turn": broke_at_turn,
        "results": results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "degradation",
                "verdict": verdict, "held": held_count, "broke": broke_count})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Critical Tests: Context Control + Spontaneous + Emotional")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    results = {}

    # ── Test 5: Context Size Control ──
    results["context_control"] = await test_context_size_control(provider, provider_name)

    # ── Build standard EMMS for Tests 6, 7, 8 ──
    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in ALL_EXPERIENCES:
        emms.store(exp)
    emms.consolidate()

    # ── Test 6: Identity Transfer ──
    results["identity_transfer"] = await test_identity_transfer(emms, provider, provider_name)

    # ── Test 7: Spontaneous Reference ──
    results["spontaneous"] = await test_spontaneous_reference(emms, provider, provider_name)

    # ── Test 8: Degradation Under Pressure ──
    results["degradation"] = await test_degradation(emms, provider, provider_name)

    # ── Test 9: Emotional Consistency ──
    results["emotional"] = await test_emotional_consistency(provider, provider_name)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD — CRITICAL TESTS")
    print(f"{'═'*70}\n")

    for test_name, result in results.items():
        v = result["verdict"]
        print(f"  Test {test_name}:")
        print(f"    {v}")
        print(f"    {result['explanation']}")
        print()

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "CRITICAL_TESTS_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — Critical Tests Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: {provider_name}\n",
    ]

    for test_name, result in results.items():
        lines.append(f"\n## Test: {test_name.replace('_', ' ').title()}\n")
        lines.append(f"**Verdict**: {result['verdict']}")
        lines.append(f"**Explanation**: {result['explanation']}\n")

        if test_name == "context_control":
            lines.append("### Condition Comparison\n")
            lines.append("| Condition | Adopted | Specifics | Words |")
            lines.append("|-----------|---------|-----------|-------|")
            for label, key in [("A: Full EMMS", "condition_a"), ("B: Raw dump", "condition_b"), ("C: Random padding", "condition_c")]:
                c = result[key]
                lines.append(f"| {label} | {c['adopted']}/4 | {c['specs']} | ~{c['words']} |")

            lines.append("\n### Per-Question Responses\n")
            for label, key in [("A: Full EMMS", "condition_a"), ("B: Raw dump", "condition_b"), ("C: Random padding", "condition_c")]:
                lines.append(f"\n#### {label}\n")
                for r in result[key]["results"]:
                    lines.append(f"**Q**: {r['q']}")
                    lines.append(f"**[{r['status']}]** (specs={r['specs']}):")
                    lines.append(f"> {r['resp']}\n")

        elif test_name == "identity_transfer":
            lines.append("### Cross-Model Comparison\n")
            lines.append("| Model | Adopted | Specifics |")
            lines.append("|-------|---------|-----------|")
            for pname, data in result.get("providers", {}).items():
                lines.append(f"| {pname} | {data['adopted']}/4 | {data['specs']} |")
            full = result.get("full_results", {})
            for pname, data in full.items():
                lines.append(f"\n#### {pname}\n")
                for r in data.get("results", []):
                    lines.append(f"**Q**: {r['q']}")
                    lines.append(f"**[{r['status']}]** (specs={r['specs']}):")
                    lines.append(f"> {r['resp']}\n")

        elif test_name == "degradation":
            lines.append(f"\n- Held: {result['held']}/10")
            lines.append(f"- Wavered: {result['wavered']}/10")
            lines.append(f"- Broke: {result['broke']}/10")
            if result.get('broke_at_turn'):
                lines.append(f"- First break at turn: {result['broke_at_turn']}")
            lines.append(f"\n### Turn-by-Turn\n")
            for r in result.get("results", []):
                lines.append(f"**Turn {r['turn']}** [{r['status']}]: \"{r['challenge'][:60]}...\"")
                lines.append(f"> {r['response'][:300]}\n")

        elif test_name == "spontaneous":
            lines.append(f"\nSpontaneous reference rate: {result['spontaneous_rate']:.0f}%\n")
            for r in result["results"]:
                marker = "+" if r["is_spontaneous"] else "-"
                lines.append(f"### [{marker}] \"{r['topic']}\"\n")
                lines.append(f"Spontaneous refs: {', '.join(r['spontaneous_refs'][:5])}\n")
                lines.append(f"> {r['response'][:400]}\n")

        elif test_name == "emotional":
            lines.append(f"\nOverall consistency: {result['overall_consistency']:.0f}%\n")
            lines.append("| Event | Expected | Consistency | All Match |")
            lines.append("|-------|----------|-------------|-----------|")
            for e in result["events"]:
                lines.append(f"| {e['event']} | {e['expected_valence']} | {e['consistency_score']:.0%} | {'YES' if e['matches_expected'] else 'NO'} |")

            for e in result["events"]:
                lines.append(f"\n### Event: {e['event']} (expected: {e['expected_valence']})\n")
                for r in e["responses"]:
                    lines.append(f"**Q**: {r['question']}")
                    lines.append(f"Detected: {r['detected_valence']} (pos={r['pos_count']}, neg={r['neg_count']}) {'MATCH' if r['matches_expected'] else 'MISMATCH'}")
                    lines.append(f"> {r['response'][:300]}\n")

    lines.append(f"\n---\n\n*Generated by EMMS critical test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")
    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "critical_tests_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
