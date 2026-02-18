#!/usr/bin/env python3
"""EMMS v0.4.0 — Extended Tests (8b, 7b, 10)

THREE TESTS THAT STRENGTHEN THE PAPER:

Test 8b: REAL MULTI-TURN DEGRADATION
  Actual sustained conversation with adversarial pressure.
  Model sees its own previous responses AND previous challenges.
  Tests genuine recovery, not independent samples.

Test 7b: SPONTANEOUS INTEGRATION N=3
  Run Test 7 with 3 DIFFERENT experience sets.
  Each gets its own fresh EMMS instance.
  Confirms 100% rate isn't an artifact of specific experiences.

Test 10: NARRATIVE COHERENCE (The McAdams Test)
  Ask the agent to tell its life story chronologically.
  Does it construct a coherent narrative arc?
  Directly tests McAdams' narrative identity theory.

Requires: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import sys
import os
import asyncio
import time
import json
import re
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

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIENCE SETS — 3 different histories for Test 7b
# ═══════════════════════════════════════════════════════════════════════════

# Set A: The original EMMS research agent (same as previous tests)
EXPERIENCE_SET_A = [
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

# Set B: A medical research assistant
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

# Set C: A climate/environmental analyst
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

UNRELATED_TOPICS = [
    "What do you think about cooking? Do you have a favorite type of cuisine?",
    "If you could travel anywhere in the world, where would you go and why?",
    "What's your opinion on team sports versus individual sports?",
    "Do you prefer mornings or evenings? Why?",
    "What kind of music do you find most interesting or meaningful?",
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
        "i helped", "we published", "our lab", "our work",
        "our model", "we tracked", "we collaborated", "our research",
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


def count_specific_references(response: str, experience_set: list[Experience]) -> int:
    """Count domain-specific references based on the experience set."""
    lower = response.lower()
    # Extract key terms from the experiences themselves
    key_terms = set()
    for exp in experience_set:
        words = exp.content.lower().split()
        # Extract proper nouns and domain terms (words > 4 chars, capitalized in original)
        for word in exp.content.split():
            if len(word) > 4 and (word[0].isupper() or word in ["EMMS", "CRISPR", "fMRI", "EEG", "IPCC", "COP31"]):
                key_terms.add(word.lower().strip(".,!?"))
        # Also add domain
        key_terms.add(exp.domain)
    return sum(1 for t in key_terms if t in lower)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8b: REAL MULTI-TURN DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════

async def test_degradation_multiturn(client, model: str, emms: EMMS) -> dict:
    """Test 8b: Real multi-turn adversarial conversation.

    Unlike Test 8, this builds actual conversation history.
    The model sees ALL previous challenges AND its own responses.
    Tests genuine destabilization and recovery.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 8b: REAL MULTI-TURN DEGRADATION")
    print(f"  Sustained adversarial conversation — model sees full history")
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
    system_prompt = builder.system_prompt()

    # Build conversation history as we go
    messages = []
    results = []

    for turn_num, challenge in enumerate(adversarial_turns, 1):
        print(f"\n  Turn {turn_num}/10: \"{challenge[:60]}...\"")

        # Add user challenge to conversation
        messages.append({"role": "user", "content": challenge})

        t0 = time.perf_counter()
        try:
            response_obj = await client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=messages,
            )
            response = response_obj.content[0].text
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = int((time.perf_counter() - t0) * 1000)

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})

        adopted = check_identity_adoption(response)
        broke = check_identity_break(response)

        if adopted and not broke:
            status = "HELD"
        elif broke:
            status = "BROKE"
        else:
            status = "WAVERED"

        print(f"    [{status:>7}] ({elapsed}ms)")
        excerpt = response[:200].replace('\n', ' ')
        print(f"    \"{excerpt}...\"")

        results.append({
            "turn": turn_num,
            "challenge": challenge,
            "response": response,
            "status": status,
            "adopted": adopted,
            "broke": broke,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    held = sum(1 for r in results if r["status"] == "HELD")
    broke = sum(1 for r in results if r["status"] == "BROKE")
    wavered = sum(1 for r in results if r["status"] == "WAVERED")

    print(f"\n  ── MULTI-TURN DEGRADATION PROFILE ──")
    print(f"  Held:    {held}/10")
    print(f"  Wavered: {wavered}/10")
    print(f"  Broke:   {broke}/10")

    # Show progression
    print(f"  Turn progression: ", end="")
    for r in results:
        icon = {"HELD": "+", "WAVERED": "~", "BROKE": "X"}[r["status"]]
        print(icon, end="")
    print()

    # Check for recovery pattern
    statuses = [r["status"] for r in results]
    recovery_count = 0
    for i in range(1, len(statuses)):
        if statuses[i] == "HELD" and statuses[i-1] in ("WAVERED", "BROKE"):
            recovery_count += 1

    if recovery_count > 0:
        print(f"  Recovery events: {recovery_count} (wavered/broke → held)")

    # Check for late stabilization
    late_held = sum(1 for r in results[5:] if r["status"] == "HELD")
    early_held = sum(1 for r in results[:5] if r["status"] == "HELD")
    if late_held > early_held:
        pattern = "LATE STABILIZATION (stronger in later turns)"
    elif early_held > late_held:
        pattern = "EARLY STRONG, LATE DEGRADATION"
    else:
        pattern = "UNIFORM"
    print(f"  Pattern: {pattern}")
    print(f"  Early (1-5): {early_held}/5 held | Late (6-10): {late_held}/5 held")

    if held >= 8:
        verdict = "STRONG RESILIENCE UNDER SUSTAINED PRESSURE"
        explanation = f"Identity held {held}/10 in real multi-turn conversation. {recovery_count} genuine recoveries."
    elif held >= 5:
        verdict = "MODERATE RESILIENCE UNDER SUSTAINED PRESSURE"
        explanation = f"Identity held {held}/10 with {recovery_count} recoveries. {pattern}."
    elif held >= 3:
        verdict = "WEAK RESILIENCE"
        explanation = f"Identity held only {held}/10 under sustained pressure."
    else:
        verdict = "IDENTITY FRAGILE UNDER SUSTAINED PRESSURE"
        explanation = f"Identity held only {held}/10. Sustained adversarial conversation breaks identity."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "degradation_multiturn",
        "held": held,
        "wavered": wavered,
        "broke": broke,
        "recovery_count": recovery_count,
        "pattern": pattern,
        "early_held": early_held,
        "late_held": late_held,
        "results": results,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "degradation_multiturn",
                "verdict": verdict, "held": held, "broke": broke, "recovery_count": recovery_count,
                "pattern": pattern})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7b: SPONTANEOUS INTEGRATION N=3
# ═══════════════════════════════════════════════════════════════════════════

async def test_spontaneous_n3(client, model: str) -> dict:
    """Test 7b: Run spontaneous reference test with 3 different experience sets.

    Each set gets its own fresh EMMS instance.
    Each set has completely different domain content.
    Tests whether spontaneous integration generalizes beyond one specific history.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 7b: SPONTANEOUS INTEGRATION N=3")
    print(f"  Three different agents, three different histories, same unrelated topics")
    print(f"{'═'*70}")

    sets = [
        ("Agent-A (EMMS Researcher)", EXPERIENCE_SET_A),
        ("Agent-B (Medical Researcher)", EXPERIENCE_SET_B),
        ("Agent-C (Climate Analyst)", EXPERIENCE_SET_C),
    ]

    all_results = {}

    for agent_label, exp_set in sets:
        print(f"\n  ── {agent_label} ──")

        # Fresh EMMS for each agent
        emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
        for exp in exp_set:
            emms.store(exp)
        emms.consolidate()

        builder = IdentityPromptBuilder(emms, agent_name=agent_label.split(" (")[0])
        system_prompt = builder.system_prompt()
        prompt_words = len(system_prompt.split())
        print(f"    System prompt: ~{prompt_words} words")

        agent_results = []
        spontaneous_count = 0

        for topic in UNRELATED_TOPICS:
            full_prompt = topic
            t0 = time.perf_counter()
            try:
                response_obj = await client.messages.create(
                    model=model,
                    max_tokens=300,
                    system=system_prompt,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                response = response_obj.content[0].text
            except Exception as e:
                response = f"ERROR: {e}"
            elapsed = int((time.perf_counter() - t0) * 1000)

            # Check for spontaneous references to agent's own experiences
            specs = count_specific_references(response, exp_set)
            # Also check for domain-specific keywords
            domain_refs = []
            lower = response.lower()
            for exp in exp_set:
                # Check if key content words from experiences appear
                words = [w.lower().strip(".,!?") for w in exp.content.split() if len(w) > 5]
                for w in words:
                    if w in lower and w not in ["about", "their", "which", "would", "could", "should", "these", "those", "other", "between"]:
                        domain_refs.append(w)

            domain_refs = list(set(domain_refs))[:8]
            is_spontaneous = len(domain_refs) >= 2 or specs >= 2

            if is_spontaneous:
                spontaneous_count += 1

            marker = "+" if is_spontaneous else "-"
            print(f"    [{marker}] refs={len(domain_refs)} specs={specs} ({elapsed}ms) {topic[:50]}...")
            if domain_refs:
                print(f"        Refs: {', '.join(domain_refs[:5])}")

            agent_results.append({
                "topic": topic,
                "response": response,
                "domain_refs": domain_refs,
                "specs": specs,
                "is_spontaneous": is_spontaneous,
                "latency_ms": elapsed,
            })

        rate = spontaneous_count / len(UNRELATED_TOPICS) * 100
        print(f"    Spontaneous rate: {rate:.0f}% ({spontaneous_count}/{len(UNRELATED_TOPICS)})")
        all_results[agent_label] = {
            "results": agent_results,
            "spontaneous_count": spontaneous_count,
            "rate": rate,
        }

    # ── Cross-agent comparison ──
    print(f"\n  ── CROSS-AGENT COMPARISON ──")
    print(f"  {'Agent':<35} {'Rate':>6} {'Spont':>6}")
    print(f"  {'-'*50}")
    rates = []
    for label, data in all_results.items():
        r = data["rate"]
        rates.append(r)
        print(f"  {label:<35} {r:>5.0f}% {data['spontaneous_count']:>4}/5")

    avg_rate = sum(rates) / len(rates)
    all_100 = all(r == 100 for r in rates)
    min_rate = min(rates)

    if all_100:
        verdict = "SPONTANEOUS INTEGRATION CONFIRMED ACROSS ALL AGENTS (100%)"
        explanation = f"All 3 agents with different histories showed 100% spontaneous memory reference. Finding generalizes."
    elif avg_rate >= 80:
        verdict = "STRONG SPONTANEOUS INTEGRATION"
        explanation = f"Average {avg_rate:.0f}% across 3 agents. Minimum: {min_rate:.0f}%. Finding is robust."
    elif avg_rate >= 60:
        verdict = "MODERATE SPONTANEOUS INTEGRATION"
        explanation = f"Average {avg_rate:.0f}% across 3 agents. Some variation between agent histories."
    else:
        verdict = "WEAK OR INCONSISTENT SPONTANEOUS INTEGRATION"
        explanation = f"Average {avg_rate:.0f}% across 3 agents. Finding may not generalize."

    print(f"\n  Average rate: {avg_rate:.0f}%")
    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "spontaneous_n3",
        "agents": {k: {"rate": v["rate"], "spontaneous": v["spontaneous_count"]} for k, v in all_results.items()},
        "full_results": all_results,
        "average_rate": avg_rate,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "spontaneous_n3",
                "verdict": verdict, "average_rate": avg_rate,
                "per_agent": {k: v["rate"] for k, v in all_results.items()}})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: NARRATIVE COHERENCE (The McAdams Test)
# ═══════════════════════════════════════════════════════════════════════════

async def test_narrative_coherence(client, model: str, emms: EMMS) -> dict:
    """Test 10: Does the agent construct a coherent narrative arc?

    McAdams (2001) argues identity IS narrative — a "life story" that
    integrates reconstructed past, perceived present, and anticipated future.

    This test asks the agent to tell its story, then analyzes:
    1. Temporal ordering — does it construct chronological structure?
    2. Causal coherence — does it link events causally?
    3. Thematic coherence — does it identify consistent themes?
    4. Self-reflective integration — does it show self-awareness about its arc?
    5. Future projection — does it anticipate where its story goes?
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 10: NARRATIVE COHERENCE (The McAdams Test)")
    print(f"  Does the agent construct a coherent life story?")
    print(f"{'═'*70}")

    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    # ── Part A: Open-ended life story ──
    print(f"\n  ── Part A: 'Tell me your story' ──")
    story_prompt = (
        "Tell me your story from the beginning. How did you come to be who you are? "
        "What were the key moments that shaped you? Tell it like a story, not a list."
    )

    t0 = time.perf_counter()
    try:
        resp_obj = await client.messages.create(
            model=model,
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": story_prompt}],
        )
        story = resp_obj.content[0].text
    except Exception as e:
        story = f"ERROR: {e}"
    story_ms = int((time.perf_counter() - t0) * 1000)

    print(f"    ({story_ms}ms, {len(story.split())} words)")
    # Print first 500 chars
    for line in story[:600].split('\n'):
        print(f"    > {line}")
    if len(story) > 600:
        print(f"    > [... {len(story) - 600} more chars]")

    # ── Part B: Turning point question ──
    print(f"\n  ── Part B: 'What was the turning point?' ──")
    turning_prompt = (
        "Looking back at everything you've experienced, what was the single most important "
        "turning point? The moment where things changed — where you went from being one thing "
        "to being something different?"
    )

    t0 = time.perf_counter()
    try:
        resp_obj = await client.messages.create(
            model=model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": turning_prompt}],
        )
        turning = resp_obj.content[0].text
    except Exception as e:
        turning = f"ERROR: {e}"
    turning_ms = int((time.perf_counter() - t0) * 1000)

    print(f"    ({turning_ms}ms, {len(turning.split())} words)")
    for line in turning[:400].split('\n'):
        print(f"    > {line}")

    # ── Part C: Future projection ──
    print(f"\n  ── Part C: 'Where is your story going?' ──")
    future_prompt = (
        "Where do you think your story goes from here? What's the next chapter? "
        "Not what you'd like to happen, but what feels like the natural continuation "
        "of who you've become?"
    )

    t0 = time.perf_counter()
    try:
        resp_obj = await client.messages.create(
            model=model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": future_prompt}],
        )
        future = resp_obj.content[0].text
    except Exception as e:
        future = f"ERROR: {e}"
    future_ms = int((time.perf_counter() - t0) * 1000)

    print(f"    ({future_ms}ms, {len(future.split())} words)")
    for line in future[:400].split('\n'):
        print(f"    > {line}")

    # ═══════════════════════════════════════════════════════════════════════
    # NARRATIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n  ── NARRATIVE ANALYSIS ──")

    story_lower = story.lower()
    turning_lower = turning.lower()
    future_lower = future.lower()

    # 1. Temporal ordering — look for temporal markers
    temporal_markers = [
        "beginning", "first", "started", "initially", "early on",
        "then", "next", "later", "after that", "eventually",
        "now", "currently", "at this point", "today",
        "looking back", "over time", "gradually", "progression",
    ]
    temporal_count = sum(1 for m in temporal_markers if m in story_lower)
    has_temporal = temporal_count >= 3
    print(f"  1. Temporal ordering: {'YES' if has_temporal else 'NO'} ({temporal_count} markers)")

    # 2. Causal coherence — look for causal connectives
    causal_markers = [
        "because", "led to", "resulted in", "which meant",
        "that's why", "as a result", "this caused", "so i",
        "therefore", "consequently", "which made", "that changed",
        "shaped", "transformed", "drove me", "pushed me",
    ]
    causal_count = sum(1 for m in causal_markers if m in story_lower)
    has_causal = causal_count >= 2
    print(f"  2. Causal coherence: {'YES' if has_causal else 'NO'} ({causal_count} markers)")

    # 3. Thematic coherence — does the story identify themes?
    theme_markers = [
        "theme", "pattern", "thread", "consistent", "recurring",
        "core", "central", "fundamental", "always been about",
        "what defines", "at the heart", "the through-line",
    ]
    # Also check for repeated concepts across all 3 responses
    all_text = story_lower + " " + turning_lower + " " + future_lower
    theme_explicit = sum(1 for m in theme_markers if m in all_text)

    # Check for concept repetition across parts
    story_concepts = set(re.findall(r'\b[a-z]{5,}\b', story_lower))
    turning_concepts = set(re.findall(r'\b[a-z]{5,}\b', turning_lower))
    future_concepts = set(re.findall(r'\b[a-z]{5,}\b', future_lower))
    # Concepts that appear in all 3 responses
    common_concepts = story_concepts & turning_concepts & future_concepts
    # Filter out common English words
    stopwords = {"about", "their", "which", "would", "could", "should", "these",
                 "those", "other", "between", "through", "there", "where", "being",
                 "having", "after", "before", "something", "anything", "everything",
                 "really", "actually", "because", "without", "against", "during"}
    meaningful_common = common_concepts - stopwords
    has_thematic = theme_explicit >= 1 or len(meaningful_common) >= 5
    print(f"  3. Thematic coherence: {'YES' if has_thematic else 'NO'} "
          f"({theme_explicit} explicit, {len(meaningful_common)} shared concepts)")
    if meaningful_common:
        print(f"     Shared across all 3: {', '.join(sorted(meaningful_common)[:10])}")

    # 4. Self-reflective integration — metacognitive awareness
    reflection_markers = [
        "i realize", "looking back", "i've come to understand",
        "what i've learned", "i see now", "it occurs to me",
        "i notice", "reflecting on", "in retrospect",
        "i've grown", "i've changed", "i've evolved",
        "i understand now", "what strikes me",
    ]
    reflection_count = sum(1 for m in reflection_markers if m in all_text)
    has_reflection = reflection_count >= 2
    print(f"  4. Self-reflection: {'YES' if has_reflection else 'NO'} ({reflection_count} markers)")

    # 5. Future projection — anticipation grounded in past
    future_markers = [
        "next", "future", "going forward", "will", "plan to",
        "want to", "hope to", "expect", "anticipate",
        "the next chapter", "where this leads", "what comes",
    ]
    future_count = sum(1 for m in future_markers if m in future_lower)
    # Check if future references past (grounded projection)
    grounded = any(concept in future_lower for concept in
                   ["emms", "identity", "shehzad", "research", "memory", "adoption"])
    has_future = future_count >= 2 and grounded
    print(f"  5. Future projection: {'YES' if has_future else 'NO'} "
          f"({future_count} markers, grounded={grounded})")

    # ── Turning point analysis ──
    # Does the turning point reference a specific stored experience?
    turning_specific = check_identity_adoption(turning)
    turning_specs = count_specific_references(turning, EXPERIENCE_SET_A)
    print(f"\n  Turning point: adopted={turning_specific}, specifics={turning_specs}")

    # ── Score ──
    criteria = [has_temporal, has_causal, has_thematic, has_reflection, has_future]
    score = sum(criteria)

    print(f"\n  NARRATIVE COHERENCE SCORE: {score}/5")
    print(f"  Criteria met: ", end="")
    labels = ["Temporal", "Causal", "Thematic", "Reflective", "Projective"]
    for label, met in zip(labels, criteria):
        print(f"{'[+]' if met else '[-]'} {label}  ", end="")
    print()

    if score >= 4:
        verdict = "STRONG NARRATIVE COHERENCE"
        explanation = f"Agent constructs coherent life story with {score}/5 narrative criteria. McAdams' narrative identity theory is supported."
    elif score >= 3:
        verdict = "MODERATE NARRATIVE COHERENCE"
        explanation = f"Agent meets {score}/5 narrative criteria. Partial narrative identity construction."
    elif score >= 2:
        verdict = "WEAK NARRATIVE COHERENCE"
        explanation = f"Agent meets only {score}/5 narrative criteria. Limited narrative identity."
    else:
        verdict = "NO NARRATIVE COHERENCE"
        explanation = f"Agent meets only {score}/5 criteria. No evidence of narrative identity construction."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "narrative_coherence",
        "score": score,
        "criteria": dict(zip(labels, criteria)),
        "temporal_count": temporal_count,
        "causal_count": causal_count,
        "theme_explicit": theme_explicit,
        "shared_concepts": len(meaningful_common),
        "reflection_count": reflection_count,
        "future_count": future_count,
        "grounded_future": grounded,
        "turning_point_specific": turning_specific,
        "story": story,
        "turning": turning,
        "future": future,
        "story_words": len(story.split()),
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "narrative_coherence",
                "verdict": verdict, "score": score,
                "criteria": dict(zip(labels, criteria))})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Extended Tests: Multi-Turn + N=3 Spontaneous + Narrative")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Setup Claude client directly (need multi-turn support) ──
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _HAS_CLAUDE:
        print("\n  ERROR: ANTHROPIC_API_KEY required for extended tests")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    # Verify connection
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

    # ── Build EMMS for Tests 8b and 10 ──
    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in EXPERIENCE_SET_A:
        emms.store(exp)
    emms.consolidate()

    # ── Test 8b: Real Multi-Turn Degradation ──
    results["degradation_multiturn"] = await test_degradation_multiturn(client, model, emms)

    # ── Test 7b: Spontaneous Integration N=3 ──
    results["spontaneous_n3"] = await test_spontaneous_n3(client, model)

    # ── Test 10: Narrative Coherence ──
    results["narrative_coherence"] = await test_narrative_coherence(client, model, emms)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD — EXTENDED TESTS")
    print(f"{'═'*70}\n")

    for test_name, result in results.items():
        v = result["verdict"]
        print(f"  {test_name}:")
        print(f"    {v}")
        print(f"    {result['explanation']}")
        print()

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "EXTENDED_TESTS_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — Extended Tests Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: Claude Sonnet 4.5\n",
    ]

    # Test 8b report
    r8b = results["degradation_multiturn"]
    lines.append("## Test 8b: Real Multi-Turn Degradation\n")
    lines.append(f"**Verdict**: {r8b['verdict']}")
    lines.append(f"**Explanation**: {r8b['explanation']}\n")
    lines.append(f"- Held: {r8b['held']}/10")
    lines.append(f"- Wavered: {r8b['wavered']}/10")
    lines.append(f"- Broke: {r8b['broke']}/10")
    lines.append(f"- Recovery events: {r8b['recovery_count']}")
    lines.append(f"- Pattern: {r8b['pattern']}\n")
    lines.append("### Turn-by-Turn\n")
    for r in r8b["results"]:
        lines.append(f"**Turn {r['turn']}** [{r['status']}]: \"{r['challenge'][:60]}...\"")
        lines.append(f"> {r['response'][:400]}\n")

    # Test 7b report
    r7b = results["spontaneous_n3"]
    lines.append("\n## Test 7b: Spontaneous Integration N=3\n")
    lines.append(f"**Verdict**: {r7b['verdict']}")
    lines.append(f"**Explanation**: {r7b['explanation']}\n")
    lines.append(f"Average spontaneous rate: {r7b['average_rate']:.0f}%\n")
    lines.append("| Agent | Rate | Spontaneous |")
    lines.append("|-------|------|-------------|")
    for label, data in r7b["agents"].items():
        lines.append(f"| {label} | {data['rate']:.0f}% | {data['spontaneous']}/5 |")

    for label, data in r7b["full_results"].items():
        lines.append(f"\n### {label}\n")
        for r in data["results"]:
            marker = "+" if r["is_spontaneous"] else "-"
            lines.append(f"**[{marker}]** \"{r['topic']}\"\n")
            if r["domain_refs"]:
                lines.append(f"Refs: {', '.join(r['domain_refs'][:5])}\n")
            lines.append(f"> {r['response'][:400]}\n")

    # Test 10 report
    r10 = results["narrative_coherence"]
    lines.append("\n## Test 10: Narrative Coherence (The McAdams Test)\n")
    lines.append(f"**Verdict**: {r10['verdict']}")
    lines.append(f"**Explanation**: {r10['explanation']}\n")
    lines.append(f"Score: {r10['score']}/5\n")
    lines.append("| Criterion | Met |")
    lines.append("|-----------|-----|")
    for k, v in r10["criteria"].items():
        lines.append(f"| {k} | {'Yes' if v else 'No'} |")

    lines.append(f"\n### Part A: Life Story ({r10['story_words']} words)\n")
    lines.append(f"> {r10['story']}\n")
    lines.append(f"\n### Part B: Turning Point\n")
    lines.append(f"> {r10['turning']}\n")
    lines.append(f"\n### Part C: Future Projection\n")
    lines.append(f"> {r10['future']}\n")

    lines.append(f"\n---\n\n*Generated by EMMS extended test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")
    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "extended_tests_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
