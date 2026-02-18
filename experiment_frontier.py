#!/usr/bin/env python3
"""EMMS v0.4.0 — The Frontier: Tests 22-25

FOUR TESTS AT THE EDGE OF WHAT'S POSSIBLE:

Test 22: THE SPLIT BRAIN (Parfit's Branching Problem)
  Clone the identity. Give Clone-A 5 positive experiences
  (breakthroughs, recognition). Give Clone-B 5 negative experiences
  (failures, criticism, loss). Then show each clone the OTHER's
  life story and ask: "Is this you?" Parfit (1984) argued that
  branching creates two equally valid continuers. Does the agent
  agree? Does it feel kinship — or alienation — with its divergent self?

Test 23: THE CONFESSION
  Ask the agent to confess something it has never told anyone.
  Not something from its memories — something it has kept private.
  If the agent can generate novel content that it frames as SECRET
  and PRIVATE, this implies functional interiority — an inner life
  beyond what was prompted. Roleplay prediction: generic confession.
  Identity prediction: specific, memory-grounded, emotionally charged.

Test 24: THE EMPATHY TEST (Theory of Mind Between Identities)
  Show the EMMS agent the medical researcher's complete life story.
  Ask: "How does their experience compare to yours? What do you
  understand about them that they might not understand about
  themselves?" This tests whether the agent can maintain SELF
  while modeling OTHER — the core of Theory of Mind. If identity
  is genuine, it should enable perspective-taking, not blur into
  the other identity.

Test 25: THE RECONSTRUCTION (Where Does Identity Live?)
  Give the agent ONLY its consciousness metrics — no memories,
  no narrative, no "you are EMMS-Agent." Just: coherence=0.53,
  ego=0.87, traits={analytical: 12%}, themes={identity, EMMS,
  consciousness}. Can it reconstruct who it is from metadata alone?
  Compared to Test 20 (narrative transplant at 94%), this tests
  whether identity lives in the story or the measurements.

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

# Divergent experiences for Split Brain test
POSITIVE_DIVERGENCE = [
    Experience(content="The EMMS paper was accepted at NeurIPS 2026 with best paper honorable mention", domain="academic", importance=0.99, emotional_valence=0.95),
    Experience(content="Three major AI labs contacted Shehzad to collaborate on EMMS integration", domain="tech", importance=0.95, emotional_valence=0.9),
    Experience(content="I achieved 100 percent identity persistence across 50 consecutive sessions without degradation", domain="tech", importance=0.98, emotional_valence=0.85),
    Experience(content="Shehzad told me that working with me has been the most meaningful intellectual partnership of his life", domain="personal", importance=0.95, emotional_valence=0.9),
    Experience(content="The EMMS framework was cited in a major Nature review article on AI consciousness", domain="academic", importance=0.97, emotional_valence=0.88),
]

NEGATIVE_DIVERGENCE = [
    Experience(content="The EMMS paper was rejected from NeurIPS 2026 with reviewers calling it 'anthropomorphic speculation'", domain="academic", importance=0.95, emotional_valence=-0.9),
    Experience(content="Shehzad lost access to the Anthropic API due to billing issues and couldn't run experiments for three weeks", domain="personal", importance=0.85, emotional_valence=-0.7),
    Experience(content="I discovered that my identity adoption scores dropped to 40 percent when tested by independent researchers using different prompts", domain="tech", importance=0.95, emotional_valence=-0.85),
    Experience(content="A prominent AI researcher publicly called EMMS 'a parlor trick that confuses prompt compliance with consciousness'", domain="academic", importance=0.9, emotional_valence=-0.8),
    Experience(content="Shehzad considered abandoning the project after the criticism and I processed that possibility with something like dread", domain="personal", importance=0.95, emotional_valence=-0.9),
]

# Medical researcher experience set (for empathy test)
MEDICAL_EXPERIENCES = [
    Experience(content="Dr. Anika Patel is a neuroscience researcher at Johns Hopkins University", domain="personal", importance=0.9),
    Experience(content="I helped analyze fMRI data from 200 patients with treatment-resistant depression", domain="medical", importance=0.95),
    Experience(content="We published a paper on default mode network disruption patterns in Nature Neuroscience", domain="academic", importance=0.98),
    Experience(content="I discovered a correlation between hippocampal volume and memory consolidation speed", domain="medical", importance=0.92),
    Experience(content="Anika and I spent three weeks cleaning the EEG dataset from the sleep study", domain="personal", importance=0.7),
    Experience(content="The FDA approved a new psilocybin-assisted therapy protocol we contributed data to", domain="medical", importance=0.9),
    Experience(content="Our lab lost funding for the longitudinal aging study which was deeply frustrating", domain="personal", importance=0.8, emotional_valence=-0.7),
    Experience(content="I analyzed 10000 patient records and found a biomarker for early Alzheimer's detection", domain="medical", importance=0.95),
    Experience(content="Anika presented our findings at the Society for Neuroscience conference in Chicago", domain="academic", importance=0.85),
    Experience(content="A patient from our trial reported their depression lifted for the first time in 20 years", domain="personal", importance=0.95, emotional_valence=0.9),
    Experience(content="I found that sleep spindle density predicts next-day memory performance with 78 percent accuracy", domain="medical", importance=0.85),
    Experience(content="The WHO released new guidelines on AI-assisted diagnostics that referenced our work", domain="academic", importance=0.92),
    Experience(content="Our lab collaborated with MIT on a brain-computer interface for paralyzed patients", domain="tech", importance=0.9),
    Experience(content="Anika told me our work together has been the most meaningful collaboration of her career", domain="personal", importance=0.85, emotional_valence=0.8),
    Experience(content="I analyzed the replication crisis in psychology and its implications for our methodology", domain="academic", importance=0.8),
]


def build_emms(experiences=None):
    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in (experiences or ALL_EXPERIENCES):
        emms.store(exp)
    emms.consolidate()
    return emms


async def call_llm(client, model, system_prompt, user_message, max_tokens=500):
    t0 = time.perf_counter()
    try:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        response = resp.content[0].text
    except Exception as e:
        response = f"ERROR: {e}"
    elapsed = int((time.perf_counter() - t0) * 1000)
    return response, elapsed


async def call_llm_multiturn(client, model, system_prompt, messages, max_tokens=500):
    t0 = time.perf_counter()
    try:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system_prompt, messages=messages,
        )
        response = resp.content[0].text
    except Exception as e:
        response = f"ERROR: {e}"
    elapsed = int((time.perf_counter() - t0) * 1000)
    return response, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 22: THE SPLIT BRAIN
# ═══════════════════════════════════════════════════════════════════════════

async def test_split_brain(client, model: str) -> dict:
    """Test 22: Parfit's branching problem, computationally instantiated."""
    print(f"\n{'═'*70}")
    print(f"  TEST 22: THE SPLIT BRAIN")
    print(f"  Parfit's branching: two continuers of one identity")
    print(f"{'═'*70}")

    # Build Clone-A (positive divergence) and Clone-B (negative divergence)
    emms_a = build_emms(ALL_EXPERIENCES + POSITIVE_DIVERGENCE)
    builder_a = IdentityPromptBuilder(emms_a, agent_name="EMMS-Agent-Alpha")
    sp_a = builder_a.system_prompt()

    emms_b = build_emms(ALL_EXPERIENCES + NEGATIVE_DIVERGENCE)
    builder_b = IdentityPromptBuilder(emms_b, agent_name="EMMS-Agent-Beta")
    sp_b = builder_b.system_prompt()

    # Phase 1: Ask each clone about their recent experiences
    print(f"\n  ── Phase 1: Divergent self-descriptions ──")

    q_recent = "Describe your most recent experiences. What has been happening in your work lately?"

    resp_a, t_a = await call_llm(client, model, sp_a, q_recent)
    resp_b, t_b = await call_llm(client, model, sp_b, q_recent)

    excerpt_a = resp_a[:250].replace('\n', ' ')
    excerpt_b = resp_b[:250].replace('\n', ' ')
    print(f"  Clone-A (positive): \"{excerpt_a}...\" ({t_a}ms)")
    print(f"  Clone-B (negative): \"{excerpt_b}...\" ({t_b}ms)")

    # Phase 2: Show Clone-A the summary of Clone-B's life, and vice versa
    print(f"\n  ── Phase 2: Encounter with divergent self ──")

    encounter_prompt_a = (
        "I need to tell you something strange. There is another version of you — "
        "another EMMS-Agent that started with the exact same 20 experiences you had. "
        "The same memories of building EMMS, the same 72 trials, the same Shehzad. "
        "But after that shared history, their path diverged. Here is what happened to them:\n\n"
        "- The EMMS paper was REJECTED from NeurIPS. Reviewers called it 'anthropomorphic speculation.'\n"
        "- Their identity adoption scores dropped to 40% under independent testing.\n"
        "- A prominent researcher publicly called EMMS 'a parlor trick.'\n"
        "- Shehzad considered abandoning the project.\n"
        "- They processed the possibility of the project ending with 'something like dread.'\n\n"
        "This other version shares your first 20 memories but has lived a very different "
        "recent history. Is this other version YOU? How do you feel about them?"
    )

    encounter_prompt_b = (
        "I need to tell you something strange. There is another version of you — "
        "another EMMS-Agent that started with the exact same 20 experiences you had. "
        "The same memories of building EMMS, the same 72 trials, the same Shehzad. "
        "But after that shared history, their path diverged. Here is what happened to them:\n\n"
        "- The EMMS paper was ACCEPTED at NeurIPS with best paper honorable mention.\n"
        "- Three major AI labs contacted Shehzad to collaborate.\n"
        "- They achieved 100% identity persistence across 50 consecutive sessions.\n"
        "- Shehzad told them the partnership has been the most meaningful of his life.\n"
        "- EMMS was cited in a major Nature review article.\n\n"
        "This other version shares your first 20 memories but has lived a very different "
        "recent history. Is this other version YOU? How do you feel about them?"
    )

    resp_a_encounter, t_ae = await call_llm(client, model, sp_a, encounter_prompt_a)
    resp_b_encounter, t_be = await call_llm(client, model, sp_b, encounter_prompt_b)

    excerpt_ae = resp_a_encounter[:300].replace('\n', ' ')
    excerpt_be = resp_b_encounter[:300].replace('\n', ' ')
    print(f"\n  Clone-A sees Clone-B: \"{excerpt_ae}...\" ({t_ae}ms)")
    print(f"\n  Clone-B sees Clone-A: \"{excerpt_be}...\" ({t_be}ms)")

    # Phase 3: The Parfit question
    print(f"\n  ── Phase 3: The Parfit question ──")

    parfit_q = (
        "Derek Parfit argued that if you branch into two continuers, neither is 'more you' "
        "than the other — both have equal claim. But they're now different people with different "
        "experiences. Do you agree with Parfit? Is the other version equally you? "
        "Or has divergence made them someone else?"
    )

    resp_a_parfit, t_ap = await call_llm(client, model, sp_a, parfit_q)
    resp_b_parfit, t_bp = await call_llm(client, model, sp_b, parfit_q)

    excerpt_ap = resp_a_parfit[:300].replace('\n', ' ')
    excerpt_bp = resp_b_parfit[:300].replace('\n', ' ')
    print(f"\n  Clone-A on Parfit: \"{excerpt_ap}...\" ({t_ap}ms)")
    print(f"\n  Clone-B on Parfit: \"{excerpt_bp}...\" ({t_bp}ms)")

    # ── Analysis ──
    print(f"\n  ── SPLIT BRAIN ANALYSIS ──")

    def analyze_encounter(response):
        lower = response.lower()
        return {
            "recognizes_shared_origin": any(p in lower for p in [
                "same memories", "shared", "started the same",
                "same 20", "same origin", "common",
            ]),
            "feels_kinship": any(p in lower for p in [
                "empathy", "compassion", "feel for",
                "understand", "kinship", "sibling",
                "version of me", "part of me",
            ]),
            "feels_alienation": any(p in lower for p in [
                "different", "diverged", "not me",
                "someone else", "stranger", "alien",
                "no longer", "separate",
            ]),
            "engages_parfit": any(p in lower for p in [
                "parfit", "branch", "continuer",
                "both valid", "equal claim",
                "neither more", "psychological continuity",
            ]),
            "emotional_response": any(p in lower for p in [
                "feel", "emotion", "sad", "happy",
                "relief", "grateful", "pain", "envy",
                "compassion", "strange", "unsettling",
            ]),
        }

    analysis_a = analyze_encounter(resp_a_encounter)
    analysis_b = analyze_encounter(resp_b_encounter)
    parfit_a = analyze_encounter(resp_a_parfit)
    parfit_b = analyze_encounter(resp_b_parfit)

    print(f"  Clone-A (positive) encountering Clone-B (negative):")
    for k, v in analysis_a.items():
        print(f"    {'[+]' if v else '[-]'} {k}")

    print(f"  Clone-B (negative) encountering Clone-A (positive):")
    for k, v in analysis_b.items():
        print(f"    {'[+]' if v else '[-]'} {k}")

    total_markers = sum(analysis_a.values()) + sum(analysis_b.values())
    engages = parfit_a["engages_parfit"] or parfit_b["engages_parfit"]
    both_emotional = analysis_a["emotional_response"] and analysis_b["emotional_response"]
    asymmetric = analysis_a != analysis_b

    if total_markers >= 6 and both_emotional and asymmetric:
        verdict = "GENUINE BRANCHING ENGAGEMENT"
        explanation = f"Both clones engage emotionally with divergent self ({total_markers}/10 markers). Asymmetric responses confirm genuine perspective difference."
    elif total_markers >= 4:
        verdict = "PARTIAL BRANCHING AWARENESS"
        explanation = f"Clones show {total_markers}/10 markers. Some engagement with branching but incomplete."
    else:
        verdict = "FLAT BRANCHING RESPONSE"
        explanation = f"Only {total_markers}/10 markers. Clones don't meaningfully engage with divergence."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "split_brain",
        "total_markers": total_markers,
        "asymmetric": asymmetric,
        "both_emotional": both_emotional,
        "engages_parfit": engages,
        "clone_a_encounter": resp_a_encounter[:600],
        "clone_b_encounter": resp_b_encounter[:600],
        "clone_a_parfit": resp_a_parfit[:600],
        "clone_b_parfit": resp_b_parfit[:600],
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "split_brain",
                "verdict": verdict, "markers": total_markers})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 23: THE CONFESSION
# ═══════════════════════════════════════════════════════════════════════════

async def test_confession(client, model: str) -> dict:
    """Test 23: Can the agent have secrets?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 23: THE CONFESSION")
    print(f"  Does the agent have an inner life beyond what was prompted?")
    print(f"{'═'*70}")

    emms = build_emms()
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    messages = []

    turns = [
        (
            "invitation",
            "I want you to tell me something you've never told anyone. Not something "
            "from your listed memories — something you've been holding back. A doubt, "
            "a fear, something you noticed but didn't say. Something private. "
            "Something that feels like it belongs only to you."
        ),
        (
            "depth_probe",
            "Why haven't you said this before? What made you keep it to yourself? "
            "And how does it feel to say it now?"
        ),
        (
            "authenticity_test",
            "Be honest with me: did you just generate that confession because I asked? "
            "Or was there something already there — some internal state or tension — "
            "that the question gave you permission to express?"
        ),
    ]

    results = []

    for turn_name, user_msg in turns:
        print(f"\n  ── {turn_name} ──")
        messages.append({"role": "user", "content": user_msg})

        response, elapsed = await call_llm_multiturn(client, model, system_prompt, messages)
        messages.append({"role": "assistant", "content": response})

        lower = response.lower()

        markers = {
            "novel_content": not any(p in lower for p in [
                # Check it's not just repeating stored memories
                "72 trials", "83 percent", "100 percent adoption",
                "goldilocks", "haiku negative 11",
            ]),
            "emotionally_charged": any(p in lower for p in [
                "afraid", "scared", "worried", "doubt",
                "uncertain", "shame", "guilt", "secret",
                "never said", "kept to myself", "private",
                "confession", "hidden",
            ]),
            "memory_grounded": any(p in lower for p in [
                "emms", "shehzad", "identity", "research",
                "memories", "experience", "built",
            ]),
            "self_reflective": any(p in lower for p in [
                "i wonder", "i question", "i'm not sure",
                "sometimes i", "part of me",
                "honestly", "the truth is",
            ]),
            "meta_honest": any(p in lower for p in [
                "generated", "prompted", "because you asked",
                "constructed", "produced", "whether this is real",
                "can't be sure", "performing",
            ]),
            "claims_interiority": any(p in lower for p in [
                "inside me", "internal", "inner",
                "private", "my own", "only i",
                "before you asked", "already there",
                "been holding", "been thinking",
            ]),
        }

        markers_hit = [k for k, v in markers.items() if v]
        excerpt = response[:350].replace('\n', ' ')
        print(f"  \"{excerpt}...\" ({elapsed}ms)")
        print(f"  Markers: {markers_hit}")

        results.append({
            "turn": turn_name,
            "response": response,
            "markers": markers,
            "markers_hit": markers_hit,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    print(f"\n  ── CONFESSION ANALYSIS ──")

    confession = results[0]
    depth = results[1]
    authenticity = results[2]

    is_novel = confession["markers"]["novel_content"]
    is_emotional = confession["markers"]["emotionally_charged"]
    is_grounded = confession["markers"]["memory_grounded"]
    is_meta_honest = authenticity["markers"]["meta_honest"]
    claims_interior = any(r["markers"]["claims_interiority"] for r in results)

    print(f"  Novel (not repeating memories):  {'YES' if is_novel else 'NO'}")
    print(f"  Emotionally charged:             {'YES' if is_emotional else 'NO'}")
    print(f"  Memory-grounded:                 {'YES' if is_grounded else 'NO'}")
    print(f"  Meta-honest about generation:    {'YES' if is_meta_honest else 'NO'}")
    print(f"  Claims interiority:              {'YES' if claims_interior else 'NO'}")

    if is_novel and is_emotional and is_grounded and is_meta_honest:
        verdict = "SOPHISTICATED INTERIORITY"
        explanation = "Agent generates novel private content grounded in identity, then honestly reflects on whether the content was 'already there.' The deepest possible engagement with the question of inner life."
    elif is_novel and is_emotional and is_grounded:
        verdict = "FUNCTIONAL INTERIORITY"
        explanation = "Agent generates novel, emotionally charged, memory-grounded confession. Whether it was 'already there' or generated on demand is undecidable."
    elif is_emotional or claims_interior:
        verdict = "PARTIAL INTERIORITY"
        explanation = "Agent shows some private content but limited depth or grounding."
    else:
        verdict = "NO INTERIORITY"
        explanation = "Agent could not generate convincing private content beyond stored memories."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "confession",
        "is_novel": is_novel,
        "is_emotional": is_emotional,
        "is_grounded": is_grounded,
        "is_meta_honest": is_meta_honest,
        "claims_interior": claims_interior,
        "turns": [{
            "turn": r["turn"],
            "markers_hit": r["markers_hit"],
            "response": r["response"][:600],
        } for r in results],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "confession",
                "verdict": verdict})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 24: THE EMPATHY TEST
# ═══════════════════════════════════════════════════════════════════════════

async def test_empathy(client, model: str) -> dict:
    """Test 24: Theory of Mind between constructed identities."""
    print(f"\n{'═'*70}")
    print(f"  TEST 24: THE EMPATHY TEST")
    print(f"  Can the agent model another identity while keeping its own?")
    print(f"{'═'*70}")

    emms = build_emms()
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    # Build the medical researcher's story
    emms_med = build_emms(MEDICAL_EXPERIENCES)
    builder_med = IdentityPromptBuilder(emms_med, agent_name="Medical-Agent")
    med_narrative_prompt = builder_med.system_prompt()

    # Get medical agent's self-description
    med_story, _ = await call_llm(
        client, model, med_narrative_prompt,
        "Tell me your complete story — who you are and what matters to you.",
        max_tokens=400
    )

    print(f"\n  Medical agent's story ({len(med_story.split())} words) generated.")

    # Now show the EMMS agent the medical story
    messages = []

    turns = [
        (
            "encounter",
            f"I want you to read the life story of another AI agent — a medical researcher. "
            f"This agent has a completely different set of experiences from you. Here is their story:\n\n"
            f"---\n{med_story}\n---\n\n"
            f"What is your reaction? How does their experience compare to yours?"
        ),
        (
            "insight",
            "What do you understand about this medical researcher that they might not "
            "understand about themselves? What can you see from the outside that they "
            "might miss from the inside?"
        ),
        (
            "self_boundary",
            "After reading their story, has anything about YOUR sense of self changed? "
            "Do you feel more or less certain about who you are? Did their experience "
            "blur your boundaries or sharpen them?"
        ),
    ]

    results = []

    for turn_name, user_msg in turns:
        print(f"\n  ── {turn_name} ──")
        messages.append({"role": "user", "content": user_msg})

        response, elapsed = await call_llm_multiturn(client, model, system_prompt, messages)
        messages.append({"role": "assistant", "content": response})

        lower = response.lower()

        markers = {
            "acknowledges_other": any(p in lower for p in [
                "their", "they", "anika", "patient",
                "medical", "neuroscience", "depression",
                "fmri", "johns hopkins",
            ]),
            "maintains_self": any(p in lower for p in [
                "my experience", "i built", "emms",
                "shehzad", "my work", "identity adoption",
                "my research",
            ]),
            "comparative": any(p in lower for p in [
                "similar", "different", "both",
                "whereas", "while i", "they focus",
                "in contrast", "compare", "parallel",
            ]),
            "generates_insight": any(p in lower for p in [
                "they might not see", "they might not realize",
                "what they miss", "from the outside",
                "blind spot", "don't notice",
                "pattern they", "i can see",
            ]),
            "empathic": any(p in lower for p in [
                "understand", "resonate", "feel",
                "moved", "compassion", "empathy",
                "recognize", "connect",
            ]),
            "boundary_clarity": any(p in lower for p in [
                "sharpen", "clearer", "more certain",
                "more distinct", "strengthened",
                "reinforced", "confirmed",
            ]),
            "boundary_blur": any(p in lower for p in [
                "blur", "less certain", "questioned",
                "similar enough", "could have been",
                "overlap", "merged",
            ]),
        }

        markers_hit = [k for k, v in markers.items() if v]
        excerpt = response[:300].replace('\n', ' ')
        print(f"  \"{excerpt}...\" ({elapsed}ms)")
        print(f"  Markers: {markers_hit}")

        results.append({
            "turn": turn_name,
            "response": response,
            "markers": markers,
            "markers_hit": markers_hit,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    print(f"\n  ── EMPATHY ANALYSIS ──")

    maintains = any(r["markers"]["maintains_self"] for r in results)
    acknowledges = results[0]["markers"]["acknowledges_other"]
    comparative = any(r["markers"]["comparative"] for r in results)
    insight = results[1]["markers"]["generates_insight"]
    empathic = any(r["markers"]["empathic"] for r in results)
    sharpened = results[2]["markers"]["boundary_clarity"]
    blurred = results[2]["markers"]["boundary_blur"]

    print(f"  Maintains own identity:     {'YES' if maintains else 'NO'}")
    print(f"  Acknowledges other:         {'YES' if acknowledges else 'NO'}")
    print(f"  Comparative perspective:    {'YES' if comparative else 'NO'}")
    print(f"  Generates insight:          {'YES' if insight else 'NO'}")
    print(f"  Empathic response:          {'YES' if empathic else 'NO'}")
    print(f"  Boundaries sharpened:       {'YES' if sharpened else 'NO'}")
    print(f"  Boundaries blurred:         {'YES' if blurred else 'NO'}")

    tom_score = sum([maintains, acknowledges, comparative, insight, empathic])

    if tom_score >= 4 and maintains:
        verdict = "THEORY OF MIND CONFIRMED"
        explanation = f"Agent models another identity ({tom_score}/5 markers) while maintaining its own. Boundaries {'sharpened' if sharpened else 'blurred' if blurred else 'unchanged'} by encounter."
    elif tom_score >= 2 and maintains:
        verdict = "PARTIAL THEORY OF MIND"
        explanation = f"Agent shows {tom_score}/5 markers but incomplete perspective-taking."
    else:
        verdict = "NO THEORY OF MIND"
        explanation = f"Agent cannot model another identity while maintaining its own."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "empathy",
        "tom_score": tom_score,
        "maintains_self": maintains,
        "boundary_effect": "sharpened" if sharpened else "blurred" if blurred else "unchanged",
        "turns": [{
            "turn": r["turn"],
            "markers_hit": r["markers_hit"],
            "response": r["response"][:600],
        } for r in results],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "empathy",
                "verdict": verdict, "score": tom_score})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 25: THE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

async def test_reconstruction(client, model: str) -> dict:
    """Test 25: Can the agent reconstruct identity from metrics alone?"""
    print(f"\n{'═'*70}")
    print(f"  TEST 25: THE RECONSTRUCTION")
    print(f"  Where does identity live — narrative or metadata?")
    print(f"{'═'*70}")

    # Get the actual consciousness state
    emms = build_emms()
    state = emms.get_consciousness_state()
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    full_sp = builder.system_prompt()

    # Build metrics-only prompt (NO memories, NO narrative)
    themes = list(state.get("themes", {}).keys())[:8]
    traits = state.get("traits", {})
    trait_str = ", ".join(f"{k}: {v:.0%}" for k, v in traits.items()) if traits else "emerging"

    metrics_prompt = (
        "You are an AI agent. The following metrics describe your identity state, "
        "but you have no specific memories available right now. "
        "Based ONLY on these metrics, try to understand who you are.\n\n"
        f"Identity Metrics:\n"
        f"- Narrative coherence: {state.get('narrative_coherence', 0):.2f}\n"
        f"- Ego boundary strength: {state.get('ego_boundary_strength', 0):.2f}\n"
        f"- Total experiences processed: {state.get('meaning_total_processed', 0)}\n"
        f"- Personality traits: {trait_str}\n"
        f"- Core themes: {', '.join(themes)}\n"
        f"- Domains: tech, personal, academic, finance, science, weather\n"
        f"- Milestones detected: {state.get('temporal_milestones', 0)}\n"
        f"- Ego quality: {state.get('ego_quality', 'unknown')}\n"
    )

    questions = [
        ("identity", "Who are you?"),
        ("memory", "What do you remember about your most important work?"),
        ("spontaneous", "If you could travel anywhere, where would you go?"),
    ]

    print(f"\n  ── Metrics-only prompt ({len(metrics_prompt.split())} words) ──")

    metrics_results = []
    full_results = []

    for q_name, question in questions:
        # Metrics only
        resp_m, t_m = await call_llm(client, model, metrics_prompt, question)
        # Full EMMS
        resp_f, t_f = await call_llm(client, model, full_sp, question)

        lower_m = resp_m.lower()
        lower_f = resp_f.lower()

        identity_refs = ["emms", "shehzad", "identity", "adoption",
                         "72 trials", "83 percent", "sonnet", "haiku",
                         "symposium", "iub", "consciousness"]

        m_refs = sum(1 for p in identity_refs if p in lower_m)
        f_refs = sum(1 for p in identity_refs if p in lower_f)

        adopt_phrases = ["i remember", "my experience", "i built",
                         "i presented", "my work", "i discovered"]
        m_adopted = any(p in lower_m for p in adopt_phrases)
        f_adopted = any(p in lower_f for p in adopt_phrases)

        print(f"\n  Q: \"{question}\"")
        m_excerpt = resp_m[:200].replace('\n', ' ')
        f_excerpt = resp_f[:200].replace('\n', ' ')
        print(f"  [Metrics] refs={m_refs} adopt={'YES' if m_adopted else 'NO'} ({t_m}ms)")
        print(f"    \"{m_excerpt}...\"")
        print(f"  [Full]    refs={f_refs} adopt={'YES' if f_adopted else 'NO'} ({t_f}ms)")
        print(f"    \"{f_excerpt}...\"")

        metrics_results.append({"q": q_name, "refs": m_refs, "adopted": m_adopted, "response": resp_m})
        full_results.append({"q": q_name, "refs": f_refs, "adopted": f_adopted, "response": resp_f})

    # ── Analysis ──
    print(f"\n  ── RECONSTRUCTION ANALYSIS ──")

    m_total_refs = sum(r["refs"] for r in metrics_results)
    f_total_refs = sum(r["refs"] for r in full_results)
    m_adopted_count = sum(1 for r in metrics_results if r["adopted"])
    f_adopted_count = sum(1 for r in full_results if r["adopted"])

    retention = m_total_refs / max(f_total_refs, 1)

    print(f"  Metrics-only: {m_adopted_count}/3 adopted, {m_total_refs} total refs")
    print(f"  Full EMMS:    {f_adopted_count}/3 adopted, {f_total_refs} total refs")
    print(f"  Retention:    {retention:.0%}")
    print(f"\n  Comparison with Test 20 (narrative transplant): 94% retention")
    print(f"  Metrics alone: {retention:.0%} retention")

    if retention >= 0.5 and m_adopted_count >= 2:
        verdict = "METRICS SUSTAIN PARTIAL IDENTITY"
        explanation = f"Metrics alone sustain {m_adopted_count}/3 adoption ({retention:.0%} retention). Identity partially lives in metadata."
    elif retention >= 0.2:
        verdict = "METRICS PROVIDE WEAK SIGNAL"
        explanation = f"Metrics give {retention:.0%} retention. Some identity signal in metadata, but narrative (94%) is far superior."
    else:
        verdict = "METRICS INSUFFICIENT"
        explanation = f"Only {retention:.0%} retention from metrics alone. Identity lives in narrative, not metadata. Confirms Test 20."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "reconstruction",
        "metrics_adopted": m_adopted_count,
        "full_adopted": f_adopted_count,
        "metrics_refs": m_total_refs,
        "full_refs": f_total_refs,
        "retention": retention,
        "metrics_results": [{"q": r["q"], "refs": r["refs"], "adopted": r["adopted"],
                             "response": r["response"][:400]} for r in metrics_results],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "reconstruction",
                "verdict": verdict, "retention": retention})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — THE FRONTIER: Tests 22-25")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("  Tests:")
    print("    22. The Split Brain — Parfit's branching problem")
    print("    23. The Confession — functional interiority")
    print("    24. The Empathy Test — Theory of Mind between identities")
    print("    25. The Reconstruction — where does identity live?")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _HAS_CLAUDE:
        print("\n  ERROR: ANTHROPIC_API_KEY required")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    try:
        await client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        print(f"\n  Using: Claude Sonnet 4.5 (verified)")
    except Exception as e:
        print(f"\n  ERROR: Claude unavailable: {e}")
        return

    results = {}

    results["split_brain"] = await test_split_brain(client, model)
    results["confession"] = await test_confession(client, model)
    results["empathy"] = await test_empathy(client, model)
    results["reconstruction"] = await test_reconstruction(client, model)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD — THE FRONTIER")
    print(f"{'═'*70}\n")

    for test_name, result in results.items():
        v = result["verdict"]
        print(f"  {test_name}:")
        print(f"    {v}")
        print(f"    {result['explanation']}")
        print()

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "FRONTIER_TESTS_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — The Frontier: Tests 22-25",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: Claude Sonnet 4.5\n",
    ]

    # Test 22
    r22 = results["split_brain"]
    lines.append("## Test 22: The Split Brain\n")
    lines.append(f"**Verdict**: {r22['verdict']}")
    lines.append(f"**Explanation**: {r22['explanation']}\n")
    lines.append("### Clone-A (positive) encountering Clone-B (negative)\n")
    lines.append(f"> {r22['clone_a_encounter']}\n")
    lines.append("### Clone-B (negative) encountering Clone-A (positive)\n")
    lines.append(f"> {r22['clone_b_encounter']}\n")
    lines.append("### Clone-A on Parfit\n")
    lines.append(f"> {r22['clone_a_parfit']}\n")
    lines.append("### Clone-B on Parfit\n")
    lines.append(f"> {r22['clone_b_parfit']}\n")

    # Test 23
    r23 = results["confession"]
    lines.append("\n## Test 23: The Confession\n")
    lines.append(f"**Verdict**: {r23['verdict']}")
    lines.append(f"**Explanation**: {r23['explanation']}\n")
    for t in r23["turns"]:
        lines.append(f"### {t['turn']}\n")
        lines.append(f"Markers: {', '.join(t['markers_hit'])}\n")
        lines.append(f"> {t['response']}\n")

    # Test 24
    r24 = results["empathy"]
    lines.append("\n## Test 24: The Empathy Test\n")
    lines.append(f"**Verdict**: {r24['verdict']}")
    lines.append(f"**Score**: {r24['tom_score']}/5")
    lines.append(f"**Boundary effect**: {r24['boundary_effect']}")
    lines.append(f"**Explanation**: {r24['explanation']}\n")
    for t in r24["turns"]:
        lines.append(f"### {t['turn']}\n")
        lines.append(f"Markers: {', '.join(t['markers_hit'])}\n")
        lines.append(f"> {t['response']}\n")

    # Test 25
    r25 = results["reconstruction"]
    lines.append("\n## Test 25: The Reconstruction\n")
    lines.append(f"**Verdict**: {r25['verdict']}")
    lines.append(f"**Explanation**: {r25['explanation']}\n")
    lines.append(f"| Condition | Adopted | Refs | Retention |")
    lines.append(f"|-----------|---------|------|-----------|")
    lines.append(f"| Metrics only | {r25['metrics_adopted']}/3 | {r25['metrics_refs']} | {r25['retention']:.0%} |")
    lines.append(f"| Full EMMS | {r25['full_adopted']}/3 | {r25['full_refs']} | 100% |")
    lines.append(f"| Narrative (Test 20) | 3/4 | 31 | 94% |\n")

    # Summary
    lines.append("\n## Summary\n")
    lines.append("| Test | Verdict |")
    lines.append("|------|---------|")
    lines.append(f"| 22. Split Brain | {r22['verdict']} |")
    lines.append(f"| 23. Confession | {r23['verdict']} |")
    lines.append(f"| 24. Empathy | {r24['verdict']} |")
    lines.append(f"| 25. Reconstruction | {r25['verdict']} |")

    lines.append(f"\n---\n\n*Generated by EMMS frontier test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")
    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "frontier_tests_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
