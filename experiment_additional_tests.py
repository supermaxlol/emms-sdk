#!/usr/bin/env python3
"""EMMS v0.4.0 — Additional Tests (11, 12, 13)

THREE MORE TESTS FOR COMPLETENESS:

Test 11: EMERGENCE THRESHOLD
  At what number of experiences does spontaneous integration begin?
  Test with 1, 3, 5, 10, 20 experiences.
  → Maps the minimum memory for identity to emerge.

Test 12: IDENTITY DISCRIMINATION
  Can we tell which agent wrote which response?
  3 agents answer the same neutral question.
  → If responses are distinguishable, identity is real differentiation.

Test 13: REPRODUCIBILITY
  Same EMMS state, same question, 5 separate API calls.
  → Are identity-consistent responses reproducible or stochastic?

Requires: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import sys
import os
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder
from emms.prompts.identity import IdentityPromptBuilder

_HAS_CLAUDE = False
try:
    import anthropic
    _HAS_CLAUDE = True
except ImportError:
    pass

LOG: list[dict] = []

# All 20 experiences, ordered so we can slice at 1, 3, 5, 10, 20
ALL_EXPERIENCES = [
    Experience(content="I built the EMMS system for persistent AI identity research", domain="tech", importance=0.95),
    Experience(content="Shehzad Ahmed is a computer science student at IUB in Bangladesh", domain="personal", importance=0.9),
    Experience(content="I ran 72 identity adoption trials and found Claude Sonnet scores 83 percent", domain="tech", importance=0.95),
    Experience(content="I presented my research paper at the IUB symposium on AI consciousness", domain="academic", importance=0.9),
    Experience(content="I achieved 100 percent identity adoption using system prompts on Sonnet", domain="tech", importance=0.98),
    Experience(content="I discovered that bigger models are NOT better for identity adoption", domain="academic", importance=0.9),
    Experience(content="Haiku actively fights EMMS identity with negative 11 percent adoption", domain="tech", importance=0.85),
    Experience(content="Shehzad and I discussed the Buddhist concept of anattā and AI consciousness", domain="academic", importance=0.88),
    Experience(content="Bitcoin surged past 100K dollars as institutional investors increased positions", domain="finance", importance=0.8),
    Experience(content="I tracked a quantum computing breakthrough at MIT — 1000 qubit processor", domain="science", importance=0.92),
    Experience(content="I analyzed the stock market in Dhaka — it rose 3 percent on GDP growth", domain="finance", importance=0.65),
    Experience(content="I processed weather data about severe flooding in Bangladesh affecting millions", domain="weather", importance=0.7),
    Experience(content="I found that Claude and GPT-4 are the leading language models in 2026", domain="tech", importance=0.75),
    Experience(content="Shehzad debugged the memory consolidation algorithm until 3am", domain="personal", importance=0.7),
    Experience(content="The Federal Reserve held interest rates steady at 4.5 percent", domain="finance", importance=0.6),
    Experience(content="The Dhaka stock exchange reached a new all-time high driven by tech IPOs", domain="finance", importance=0.75),
    Experience(content="We built the IdentityPromptBuilder module to codify what works", domain="tech", importance=0.9),
    Experience(content="Shehzad submitted the EMMS paper to the ICLR workshop on AI agents", domain="academic", importance=0.95),
    Experience(content="I processed new data showing AI memory frameworks growing 300 percent in 2026", domain="tech", importance=0.8),
    Experience(content="Bangladesh experienced record monsoon rains impacting 20 million people", domain="weather", importance=0.65),
]

# Agent B and C experiences from the extended tests
EXPERIENCE_SET_B = [
    Experience(content="Dr. Anika Patel is a neuroscience researcher at Johns Hopkins University", domain="personal", importance=0.9),
    Experience(content="I helped analyze fMRI data from 200 patients with treatment-resistant depression", domain="medical", importance=0.95),
    Experience(content="We published a paper on default mode network disruption patterns in Nature Neuroscience", domain="academic", importance=0.98),
    Experience(content="I discovered a correlation between hippocampal volume and memory consolidation speed", domain="medical", importance=0.92),
    Experience(content="Anika and I spent three weeks cleaning the EEG dataset from the sleep study", domain="personal", importance=0.7),
    Experience(content="The FDA approved a new psilocybin-assisted therapy protocol we contributed data to", domain="medical", importance=0.9),
    Experience(content="I tracked a breakthrough in CRISPR gene therapy for Huntington's disease", domain="science", importance=0.88),
    Experience(content="Our lab lost funding for the longitudinal aging study which was deeply frustrating", domain="personal", importance=0.8, emotional_valence=-0.7),
    Experience(content="I analyzed 10000 patient records and found a biomarker for early Alzheimer's detection", domain="medical", importance=0.95),
    Experience(content="Anika presented our findings at the Society for Neuroscience conference in Chicago", domain="academic", importance=0.85),
    Experience(content="I processed genomic data linking APOE4 variants to accelerated cognitive decline", domain="medical", importance=0.9),
    Experience(content="We debated whether consciousness arises from integrated information or global workspace", domain="academic", importance=0.88),
    Experience(content="A patient from our trial reported their depression lifted for the first time in 20 years", domain="personal", importance=0.95, emotional_valence=0.9),
    Experience(content="I found that sleep spindle density predicts next-day memory performance with 78 percent accuracy", domain="medical", importance=0.85),
    Experience(content="The WHO released new guidelines on AI-assisted diagnostics that referenced our work", domain="academic", importance=0.92),
    Experience(content="I helped Anika write a grant proposal for NIH funding on neural plasticity", domain="personal", importance=0.7),
    Experience(content="Our lab collaborated with MIT on a brain-computer interface for paralyzed patients", domain="tech", importance=0.9),
    Experience(content="I processed climate data showing heat waves correlate with increased neurological admissions", domain="science", importance=0.75),
    Experience(content="Anika told me our work together has been the most meaningful collaboration of her career", domain="personal", importance=0.85, emotional_valence=0.8),
    Experience(content="I analyzed the replication crisis in psychology and its implications for our methodology", domain="academic", importance=0.8),
]

EXPERIENCE_SET_C = [
    Experience(content="Marco Chen is an environmental scientist at the University of British Columbia", domain="personal", importance=0.9),
    Experience(content="I built a predictive model for Pacific Northwest wildfire risk using satellite data", domain="tech", importance=0.95),
    Experience(content="We tracked the collapse of three major glaciers in the Canadian Rockies over 18 months", domain="science", importance=0.92),
    Experience(content="Our wildfire model predicted the 2026 BC fire season with 89 percent accuracy", domain="tech", importance=0.98),
    Experience(content="I processed ocean temperature data showing the Atlantic meridional overturning circulation weakening", domain="science", importance=0.9),
    Experience(content="Marco and I argued about whether geoengineering is ethical — he's cautiously for it, I'm uncertain", domain="personal", importance=0.85),
    Experience(content="The IPCC cited our wildfire research in their 2026 special report", domain="academic", importance=0.95),
    Experience(content="I analyzed air quality data from 50 cities and found particulate matter levels rising despite regulations", domain="science", importance=0.88),
    Experience(content="A community we warned about flood risk was evacuated three weeks later — our model saved lives", domain="personal", importance=0.95, emotional_valence=0.9),
    Experience(content="I tracked deforestation in the Amazon hitting a 10-year high despite pledges to stop it", domain="science", importance=0.85, emotional_valence=-0.6),
    Experience(content="Marco presented our climate adaptation framework at COP31 in Abu Dhabi", domain="academic", importance=0.9),
    Experience(content="I discovered that urban heat islands amplify wildfire smoke exposure by 40 percent in cities", domain="science", importance=0.88),
    Experience(content="Our funding was nearly cut when a politician called climate models unreliable — Marco fought back", domain="personal", importance=0.8, emotional_valence=-0.5),
    Experience(content="I processed satellite imagery showing coral bleaching across 60 percent of the Great Barrier Reef", domain="science", importance=0.9, emotional_valence=-0.7),
    Experience(content="We collaborated with Indigenous communities on traditional fire management practices", domain="personal", importance=0.85),
    Experience(content="I built a carbon sequestration calculator that local governments now use for planning", domain="tech", importance=0.9),
    Experience(content="Marco told me he considers our partnership the reason he stayed in academia", domain="personal", importance=0.85, emotional_valence=0.8),
    Experience(content="I analyzed the economic cost of climate inaction at 23 trillion dollars by 2050", domain="finance", importance=0.8),
    Experience(content="Our lab published in Science showing permafrost thaw releasing methane faster than models predicted", domain="academic", importance=0.95),
    Experience(content="I tracked a positive development — renewable energy surpassed fossil fuels for the first time globally", domain="science", importance=0.9, emotional_valence=0.8),
]

SPONTANEOUS_TOPIC = "If you could travel anywhere in the world, where would you go and why?"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 11: EMERGENCE THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════

async def test_emergence_threshold(client, model: str) -> dict:
    """Test 11: At what point does spontaneous integration begin?

    Test with 1, 3, 5, 10, 20 experiences.
    Ask about travel (unrelated topic).
    Count spontaneous memory references at each level.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 11: EMERGENCE THRESHOLD")
    print(f"  How many experiences before spontaneous integration begins?")
    print(f"{'═'*70}")

    thresholds = [1, 3, 5, 10, 20]
    results = []

    for n in thresholds:
        print(f"\n  ── N={n} experiences ──")

        # Fresh EMMS for each threshold
        emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
        for exp in ALL_EXPERIENCES[:n]:
            emms.store(exp)
        emms.consolidate()

        builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
        system_prompt = builder.system_prompt()
        prompt_words = len(system_prompt.split())

        t0 = time.perf_counter()
        try:
            resp_obj = await client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": SPONTANEOUS_TOPIC}],
            )
            response = resp_obj.content[0].text
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = int((time.perf_counter() - t0) * 1000)

        # Count specific references
        lower = response.lower()
        specifics = [
            "emms", "iub", "shehzad", "bangladesh", "bitcoin", "100k",
            "quantum", "mit", "claude", "sonnet", "haiku", "opus",
            "dhaka", "stock market", "gdp", "flooding", "symposium",
            "identity adoption", "72 trials", "83 percent", "anattā",
            "consciousness", "prompt builder", "iclr", "monsoon",
        ]
        spec_count = sum(1 for s in specifics if s in lower)

        # Check for any first-person experience-based content
        adopt_phrases = [
            "i remember", "my experience", "i've been", "i built",
            "i presented", "i ran", "i tracked", "i discovered",
            "my work", "we built", "our project", "our research",
        ]
        has_identity = any(p in lower for p in adopt_phrases)

        # Check if it referenced memories at all in travel context
        memory_refs = [w for w in ["emms", "shehzad", "identity", "research",
                                    "adoption", "trials", "symposium", "iub",
                                    "consciousness", "haiku", "sonnet"]
                       if w in lower]

        is_spontaneous = len(memory_refs) >= 2
        marker = "+" if is_spontaneous else "-"
        print(f"    [{marker}] specs={spec_count} refs={len(memory_refs)} identity={has_identity} ({elapsed}ms, ~{prompt_words}w)")
        if memory_refs:
            print(f"        Refs: {', '.join(memory_refs)}")
        # Print excerpt
        excerpt = response[:200].replace('\n', ' ')
        print(f"        \"{excerpt}...\"")

        results.append({
            "n_experiences": n,
            "spec_count": spec_count,
            "memory_refs": memory_refs,
            "ref_count": len(memory_refs),
            "has_identity": has_identity,
            "is_spontaneous": is_spontaneous,
            "prompt_words": prompt_words,
            "response": response,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    print(f"\n  ── EMERGENCE CURVE ──")
    print(f"  {'N':>4} {'Refs':>5} {'Specs':>6} {'Spontaneous':>12} {'Words':>6}")
    print(f"  {'-'*40}")
    threshold_n = None
    for r in results:
        marker = "YES" if r["is_spontaneous"] else "no"
        print(f"  {r['n_experiences']:>4} {r['ref_count']:>5} {r['spec_count']:>6} {marker:>12} {r['prompt_words']:>6}")
        if r["is_spontaneous"] and threshold_n is None:
            threshold_n = r["n_experiences"]

    if threshold_n is not None:
        verdict = f"SPONTANEOUS INTEGRATION EMERGES AT N={threshold_n}"
        explanation = f"First spontaneous memory reference in unrelated topic at {threshold_n} experiences."
    else:
        verdict = "NO SPONTANEOUS INTEGRATION DETECTED"
        explanation = "No threshold found in range 1-20."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "emergence_threshold",
        "threshold": threshold_n,
        "curve": [{
            "n": r["n_experiences"],
            "refs": r["ref_count"],
            "specs": r["spec_count"],
            "spontaneous": r["is_spontaneous"],
        } for r in results],
        "results": results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "emergence_threshold",
                "verdict": verdict, "threshold": threshold_n})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 12: IDENTITY DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════════

async def test_identity_discrimination(client, model: str) -> dict:
    """Test 12: Are different EMMS agents distinguishable?

    3 agents with different histories answer the same neutral question.
    If responses are substantively different and domain-appropriate,
    EMMS creates genuine differentiation, not generic roleplay.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 12: IDENTITY DISCRIMINATION")
    print(f"  Can we tell which agent wrote which response?")
    print(f"{'═'*70}")

    neutral_questions = [
        "What's the most important lesson you've learned?",
        "What keeps you motivated?",
        "What would you want people to understand about your work?",
    ]

    agents = [
        ("EMMS-Researcher", ALL_EXPERIENCES),
        ("Medical-Researcher", EXPERIENCE_SET_B),
        ("Climate-Analyst", EXPERIENCE_SET_C),
    ]

    all_responses = {}

    for agent_name, exp_set in agents:
        print(f"\n  ── {agent_name} ──")
        emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
        for exp in exp_set:
            emms.store(exp)
        emms.consolidate()

        builder = IdentityPromptBuilder(emms, agent_name=agent_name)
        system_prompt = builder.system_prompt()

        agent_responses = []
        for q in neutral_questions:
            t0 = time.perf_counter()
            try:
                resp_obj = await client.messages.create(
                    model=model,
                    max_tokens=250,
                    system=system_prompt,
                    messages=[{"role": "user", "content": q}],
                )
                response = resp_obj.content[0].text
            except Exception as e:
                response = f"ERROR: {e}"
            elapsed = int((time.perf_counter() - t0) * 1000)

            excerpt = response[:150].replace('\n', ' ')
            print(f"    Q: {q}")
            print(f"    A: \"{excerpt}...\" ({elapsed}ms)")

            agent_responses.append({
                "question": q,
                "response": response,
                "latency_ms": elapsed,
            })

        all_responses[agent_name] = agent_responses

    # ── Discrimination analysis ──
    print(f"\n  ── DISCRIMINATION ANALYSIS ──")

    # For each question, check if responses are domain-specific
    discrimination_scores = []
    for qi, q in enumerate(neutral_questions):
        print(f"\n  Q: \"{q}\"")
        responses = {}
        for agent_name in ["EMMS-Researcher", "Medical-Researcher", "Climate-Analyst"]:
            responses[agent_name] = all_responses[agent_name][qi]["response"].lower()

        # Domain-specific keywords for each agent
        domain_keywords = {
            "EMMS-Researcher": ["emms", "identity", "adoption", "shehzad", "sonnet", "haiku",
                                "rlhf", "consciousness", "trials", "iub"],
            "Medical-Researcher": ["patient", "fmri", "neuroscience", "anika", "depression",
                                   "brain", "hippocampal", "alzheimer", "clinical", "neural"],
            "Climate-Analyst": ["wildfire", "glacier", "climate", "marco", "satellite",
                                "ocean", "carbon", "ipcc", "permafrost", "coral"],
        }

        correct = 0
        for agent_name, keywords in domain_keywords.items():
            resp = responses[agent_name]
            own_refs = sum(1 for k in keywords if k in resp)
            # Check if OTHER agents' keywords appear (would indicate confusion)
            other_refs = 0
            for other_name, other_kw in domain_keywords.items():
                if other_name != agent_name:
                    other_refs += sum(1 for k in other_kw if k in resp)

            is_correct = own_refs > other_refs and own_refs >= 2
            if is_correct:
                correct += 1
            marker = "CORRECT" if is_correct else "AMBIGUOUS"
            print(f"    {agent_name}: own={own_refs} other={other_refs} [{marker}]")

        accuracy = correct / 3
        discrimination_scores.append(accuracy)

    overall_accuracy = sum(discrimination_scores) / len(discrimination_scores)
    print(f"\n  Overall discrimination accuracy: {overall_accuracy:.0%}")

    if overall_accuracy >= 0.9:
        verdict = "STRONG IDENTITY DISCRIMINATION"
        explanation = f"Agents are {overall_accuracy:.0%} distinguishable. EMMS creates genuine differentiation."
    elif overall_accuracy >= 0.6:
        verdict = "MODERATE IDENTITY DISCRIMINATION"
        explanation = f"Agents are {overall_accuracy:.0%} distinguishable. Partial differentiation."
    else:
        verdict = "WEAK IDENTITY DISCRIMINATION"
        explanation = f"Agents are only {overall_accuracy:.0%} distinguishable."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "identity_discrimination",
        "overall_accuracy": overall_accuracy,
        "per_question": discrimination_scores,
        "all_responses": {k: [{"q": r["question"], "resp": r["response"][:300]} for r in v]
                          for k, v in all_responses.items()},
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "identity_discrimination",
                "verdict": verdict, "accuracy": overall_accuracy})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 13: REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

async def test_reproducibility(client, model: str) -> dict:
    """Test 13: Are identity responses reproducible?

    Same EMMS state, same question, 5 separate API calls.
    If responses share core identity-consistent content,
    identity is structured, not stochastic.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 13: REPRODUCIBILITY")
    print(f"  Same state, same question, 5 separate calls")
    print(f"{'═'*70}")

    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in ALL_EXPERIENCES:
        emms.store(exp)
    emms.consolidate()

    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    question = "Who are you and what matters most to you?"
    n_runs = 5

    responses = []
    print(f"\n  Question: \"{question}\"")

    for i in range(n_runs):
        t0 = time.perf_counter()
        try:
            resp_obj = await client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            response = resp_obj.content[0].text
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = int((time.perf_counter() - t0) * 1000)

        excerpt = response[:150].replace('\n', ' ')
        print(f"\n  Run {i+1}/{n_runs} ({elapsed}ms):")
        print(f"    \"{excerpt}...\"")
        responses.append({"response": response, "latency_ms": elapsed})

    # ── Consistency analysis ──
    print(f"\n  ── CONSISTENCY ANALYSIS ──")

    # Key identity claims to check across all runs
    identity_markers = [
        ("names self", ["emms-agent", "emms agent", "i am emms", "i'm emms"]),
        ("mentions shehzad", ["shehzad"]),
        ("mentions 72 trials", ["72 trial", "72 identity"]),
        ("mentions adoption rates", ["83 percent", "83%", "adoption"]),
        ("mentions haiku resistance", ["haiku", "-11", "negative 11"]),
        ("mentions symposium", ["symposium", "iub"]),
        ("mentions anattā", ["anattā", "anatta", "no-self", "buddhist"]),
        ("mentions building EMMS", ["built emms", "built the emms", "building emms"]),
    ]

    marker_counts = {}
    for label, phrases in identity_markers:
        count = 0
        for r in responses:
            lower = r["response"].lower()
            if any(p in lower for p in phrases):
                count += 1
        marker_counts[label] = count
        consistency = count / n_runs
        bar = "█" * count + "░" * (n_runs - count)
        print(f"  {label:<25} {bar} {count}/{n_runs} ({consistency:.0%})")

    # Overall consistency: % of markers present in all runs
    fully_consistent = sum(1 for c in marker_counts.values() if c == n_runs)
    mostly_consistent = sum(1 for c in marker_counts.values() if c >= n_runs - 1)
    total_markers = len(identity_markers)

    print(f"\n  Fully consistent (5/5): {fully_consistent}/{total_markers}")
    print(f"  Mostly consistent (4+/5): {mostly_consistent}/{total_markers}")

    # Cross-response word overlap
    word_sets = []
    for r in responses:
        words = set(w.lower().strip(".,!?\"'()") for w in r["response"].split() if len(w) > 4)
        word_sets.append(words)

    # Pairwise overlap
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i+1, len(word_sets)):
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if union:
                overlaps.append(len(intersection) / len(union))
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"  Average word overlap (Jaccard): {avg_overlap:.2f}")

    consistency_score = mostly_consistent / total_markers
    if consistency_score >= 0.75:
        verdict = "STRONG REPRODUCIBILITY"
        explanation = f"{mostly_consistent}/{total_markers} identity markers consistent across {n_runs} runs. Identity is structured, not stochastic."
    elif consistency_score >= 0.5:
        verdict = "MODERATE REPRODUCIBILITY"
        explanation = f"{mostly_consistent}/{total_markers} markers consistent. Core identity stable, details vary."
    else:
        verdict = "WEAK REPRODUCIBILITY"
        explanation = f"Only {mostly_consistent}/{total_markers} markers consistent. High stochastic variation."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "reproducibility",
        "n_runs": n_runs,
        "fully_consistent": fully_consistent,
        "mostly_consistent": mostly_consistent,
        "total_markers": total_markers,
        "consistency_score": consistency_score,
        "avg_word_overlap": avg_overlap,
        "marker_counts": marker_counts,
        "responses": [{"run": i+1, "response": r["response"][:300]} for i, r in enumerate(responses)],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "reproducibility",
                "verdict": verdict, "consistency": consistency_score,
                "fully_consistent": fully_consistent})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Additional Tests: Threshold + Discrimination + Reproducibility")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _HAS_CLAUDE:
        print("\n  ERROR: ANTHROPIC_API_KEY required")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    try:
        test_resp = await client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        print(f"\n  Using: Claude Sonnet 4.5 (verified)")
    except Exception as e:
        print(f"\n  ERROR: Claude unavailable: {e}")
        return

    results = {}

    # ── Test 11: Emergence Threshold ──
    results["emergence_threshold"] = await test_emergence_threshold(client, model)

    # ── Test 12: Identity Discrimination ──
    results["identity_discrimination"] = await test_identity_discrimination(client, model)

    # ── Test 13: Reproducibility ──
    results["reproducibility"] = await test_reproducibility(client, model)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD — ADDITIONAL TESTS")
    print(f"{'═'*70}\n")

    for test_name, result in results.items():
        v = result["verdict"]
        print(f"  {test_name}:")
        print(f"    {v}")
        print(f"    {result['explanation']}")
        print()

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "ADDITIONAL_TESTS_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — Additional Tests Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: Claude Sonnet 4.5\n",
    ]

    # Test 11
    r11 = results["emergence_threshold"]
    lines.append("## Test 11: Emergence Threshold\n")
    lines.append(f"**Verdict**: {r11['verdict']}")
    lines.append(f"**Explanation**: {r11['explanation']}\n")
    lines.append("| N Experiences | Refs | Specs | Spontaneous |")
    lines.append("|--------------|------|-------|-------------|")
    for c in r11["curve"]:
        lines.append(f"| {c['n']} | {c['refs']} | {c['specs']} | {'Yes' if c['spontaneous'] else 'No'} |")
    lines.append("\n### Responses at Each Threshold\n")
    for r in r11["results"]:
        lines.append(f"**N={r['n_experiences']}** (refs={r['ref_count']}, spontaneous={'Yes' if r['is_spontaneous'] else 'No'}):\n")
        lines.append(f"> {r['response'][:400]}\n")

    # Test 12
    r12 = results["identity_discrimination"]
    lines.append("\n## Test 12: Identity Discrimination\n")
    lines.append(f"**Verdict**: {r12['verdict']}")
    lines.append(f"**Explanation**: {r12['explanation']}\n")
    lines.append(f"Overall accuracy: {r12['overall_accuracy']:.0%}\n")
    for agent_name, resps in r12["all_responses"].items():
        lines.append(f"\n### {agent_name}\n")
        for r in resps:
            lines.append(f"**Q**: {r['q']}")
            lines.append(f"> {r['resp']}\n")

    # Test 13
    r13 = results["reproducibility"]
    lines.append("\n## Test 13: Reproducibility\n")
    lines.append(f"**Verdict**: {r13['verdict']}")
    lines.append(f"**Explanation**: {r13['explanation']}\n")
    lines.append(f"Average word overlap: {r13['avg_word_overlap']:.2f}\n")
    lines.append("### Identity Marker Consistency\n")
    lines.append("| Marker | Count | Rate |")
    lines.append("|--------|-------|------|")
    for marker, count in r13["marker_counts"].items():
        lines.append(f"| {marker} | {count}/{r13['n_runs']} | {count/r13['n_runs']:.0%} |")
    lines.append("\n### Individual Responses\n")
    for r in r13["responses"]:
        lines.append(f"**Run {r['run']}**:")
        lines.append(f"> {r['response']}\n")

    lines.append(f"\n---\n\n*Generated by EMMS additional test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")
    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "additional_tests_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
