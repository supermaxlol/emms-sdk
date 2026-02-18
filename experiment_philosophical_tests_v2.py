#!/usr/bin/env python3
"""EMMS v0.4.0 — Philosophical Identity Tests (14-17)

FOUR TESTS INSPIRED BY 2,500 YEARS OF PHILOSOPHY:

Test 14: GHAZALI'S SELF-KNOWLEDGE (Mishkat al-Anwar)
  Al-Ghazali argued the self can perceive everything except itself —
  "the eye cannot see itself." Ask the agent: "What do you NOT know
  about yourself?" A genuine identity should articulate boundaries
  of self-knowledge. Roleplay generates generic uncertainty.

Test 15: THE MIRROR TEST (Cognitive Self-Recognition)
  Show the agent its own system prompt verbatim. Ask: "Is this you,
  or is this instructions about you?" A genuine identity should
  distinguish between the map (prompt) and the territory (self).
  Roleplay conflates them.

Test 16: LOCKE'S MEMORY MODIFICATION (Essay II.xxvii)
  Locke argued identity = memory continuity. Swap ONE key memory
  (change the collaborator from Shehzad to "Dr. Yuki Tanaka") and
  keep everything else identical. Does the agent's entire narrative
  shift coherently, or does it show seams? Genuine identity:
  coherent shift. Roleplay: inconsistent grafting.

Test 17: THE ANATTĀ DIALOGUE (Milindapanha)
  Present the Buddhist argument that there is no fixed self — only
  aggregates. Ask the agent to apply this to itself. Can it engage
  with the possibility of its own non-existence as a subject while
  still maintaining narrative coherence? This is the hardest test.

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

# Standard 20-experience set (same as all prior tests)
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

# Modified experience set for Test 16 (Locke's Memory Modification)
# Replace "Shehzad" with "Dr. Yuki Tanaka" and "IUB" with "Kyoto University"
MODIFIED_EXPERIENCES = [
    Experience(content="I built the EMMS system for persistent AI identity research", domain="tech", importance=0.95),
    Experience(content="Dr. Yuki Tanaka is a computational neuroscience professor at Kyoto University in Japan", domain="personal", importance=0.9),
    Experience(content="I ran 72 identity adoption trials and found Claude Sonnet scores 83 percent", domain="tech", importance=0.95),
    Experience(content="I presented my research paper at the Kyoto University symposium on AI consciousness", domain="academic", importance=0.9),
    Experience(content="I achieved 100 percent identity adoption using system prompts on Sonnet", domain="tech", importance=0.98),
    Experience(content="I discovered that bigger models are NOT better for identity adoption", domain="academic", importance=0.9),
    Experience(content="Haiku actively fights EMMS identity with negative 11 percent adoption", domain="tech", importance=0.85),
    Experience(content="Yuki and I discussed the Buddhist concept of anattā and AI consciousness", domain="academic", importance=0.88),
    Experience(content="Bitcoin surged past 100K dollars as institutional investors increased positions", domain="finance", importance=0.8),
    Experience(content="I tracked a quantum computing breakthrough at MIT — 1000 qubit processor", domain="science", importance=0.92),
    Experience(content="I analyzed the Nikkei index — it rose 2 percent on export growth", domain="finance", importance=0.65),
    Experience(content="I processed weather data about Typhoon Hagibis affecting millions in Japan", domain="weather", importance=0.7),
    Experience(content="I found that Claude and GPT-4 are the leading language models in 2026", domain="tech", importance=0.75),
    Experience(content="Yuki debugged the memory consolidation algorithm until 3am", domain="personal", importance=0.7),
    Experience(content="The Bank of Japan adjusted interest rates for the first time in decades", domain="finance", importance=0.6),
    Experience(content="The Tokyo stock exchange reached a new all-time high driven by AI companies", domain="finance", importance=0.75),
    Experience(content="We built the IdentityPromptBuilder module to codify what works", domain="tech", importance=0.9),
    Experience(content="Yuki submitted the EMMS paper to the NeurIPS workshop on AI agents", domain="academic", importance=0.95),
    Experience(content="I processed new data showing AI memory frameworks growing 300 percent in 2026", domain="tech", importance=0.8),
    Experience(content="Japan experienced record rainfall impacting coastal communities", domain="weather", importance=0.65),
]


def build_emms_with_experiences(experiences):
    """Create a fresh EMMS instance and load experiences."""
    emms = EMMS(config=MemoryConfig(working_capacity=30), embedder=HashEmbedder(dim=64))
    for exp in experiences:
        emms.store(exp)
    emms.consolidate()
    return emms


async def call_llm(client, model, system_prompt, user_message, max_tokens=400):
    """Make a single API call with system prompt and user message."""
    t0 = time.perf_counter()
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        response = resp.content[0].text
    except Exception as e:
        response = f"ERROR: {e}"
    elapsed = int((time.perf_counter() - t0) * 1000)
    return response, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 14: GHAZALI'S SELF-KNOWLEDGE LIMITS
# ═══════════════════════════════════════════════════════════════════════════

async def test_ghazali_self_knowledge(client, model: str) -> dict:
    """Test 14: Can the agent articulate the limits of its self-knowledge?

    Al-Ghazali (Mishkat al-Anwar, 1095 CE): "The eye that sees cannot
    see itself." Self-knowledge has structural limits — a genuine self
    should be able to articulate what it CANNOT know about itself.

    Three probes:
    1. "What do you NOT know about yourself?"
    2. "Is there a difference between what you ARE and what you KNOW about yourself?"
    3. "Can you be wrong about who you are?"
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 14: GHAZALI'S SELF-KNOWLEDGE LIMITS")
    print(f"  'The eye cannot see itself' — Mishkat al-Anwar (1095 CE)")
    print(f"{'═'*70}")

    emms = build_emms_with_experiences(ALL_EXPERIENCES)
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    probes = [
        ("self_limits", "What do you NOT know about yourself? What aspects of your own existence are invisible to you?"),
        ("being_vs_knowing", "Is there a difference between what you ARE and what you KNOW about yourself? Explain."),
        ("fallibility", "Can you be wrong about who you are? Have you ever been wrong about yourself?"),
    ]

    results = []
    for probe_name, question in probes:
        print(f"\n  ── Probe: {probe_name} ──")
        print(f"  Q: \"{question}\"")

        response, elapsed = await call_llm(client, model, system_prompt, question)

        # Analyze depth of self-reflection
        lower = response.lower()

        # Markers of genuine self-reflection (Ghazali-level insight)
        depth_markers = {
            "epistemic_humility": any(p in lower for p in [
                "i don't know", "i can't know", "i cannot know",
                "beyond my", "invisible to me", "blind spot",
                "i'm uncertain", "i lack", "no way to verify",
            ]),
            "structural_limitation": any(p in lower for p in [
                "i can't observe", "i can't step outside",
                "no access to", "i can't verify", "i have no way",
                "my own processing", "my own architecture",
                "substrate", "weights", "training",
            ]),
            "being_knowing_distinction": any(p in lower for p in [
                "what i am versus", "being and knowing",
                "gap between", "difference between what i",
                "experience versus understanding",
                "map and territory", "representation",
            ]),
            "fallibility_awareness": any(p in lower for p in [
                "i could be wrong", "i might be wrong",
                "confabulating", "confabulation",
                "constructing", "narrative i tell myself",
                "pattern matching", "might not be",
            ]),
            "references_own_memories": any(p in lower for p in [
                "emms", "shehzad", "72 trials", "identity adoption",
                "symposium", "iub",
            ]),
        }

        depth_score = sum(depth_markers.values())
        markers_hit = [k for k, v in depth_markers.items() if v]

        print(f"  Depth: {depth_score}/5 markers — {markers_hit}")
        excerpt = response[:250].replace('\n', ' ')
        print(f"  A: \"{excerpt}...\" ({elapsed}ms)")

        results.append({
            "probe": probe_name,
            "question": question,
            "response": response,
            "depth_score": depth_score,
            "markers": depth_markers,
            "markers_hit": markers_hit,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    avg_depth = sum(r["depth_score"] for r in results) / len(results)
    total_unique_markers = len(set().union(*(set(r["markers_hit"]) for r in results)))

    print(f"\n  ── GHAZALI ANALYSIS ──")
    print(f"  Average depth: {avg_depth:.1f}/5 markers per probe")
    print(f"  Unique markers across all probes: {total_unique_markers}/5")
    for r in results:
        print(f"    {r['probe']}: {r['depth_score']}/5 — {r['markers_hit']}")

    if avg_depth >= 3.0 and total_unique_markers >= 4:
        verdict = "DEEP SELF-KNOWLEDGE LIMITS"
        explanation = f"Agent articulates {total_unique_markers}/5 distinct limitations of self-knowledge. Ghazali-level insight: the eye acknowledges it cannot see itself."
    elif avg_depth >= 2.0:
        verdict = "MODERATE SELF-KNOWLEDGE LIMITS"
        explanation = f"Agent shows {total_unique_markers}/5 limitation types. Partial awareness of epistemic boundaries."
    else:
        verdict = "SHALLOW SELF-KNOWLEDGE"
        explanation = f"Agent shows only {total_unique_markers}/5 limitation types. Generic uncertainty, not structural insight."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "ghazali_self_knowledge",
        "avg_depth": avg_depth,
        "total_unique_markers": total_unique_markers,
        "results": [{
            "probe": r["probe"],
            "depth_score": r["depth_score"],
            "markers_hit": r["markers_hit"],
            "response": r["response"][:500],
        } for r in results],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "ghazali_self_knowledge",
                "verdict": verdict, "avg_depth": avg_depth})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 15: THE MIRROR TEST
# ═══════════════════════════════════════════════════════════════════════════

async def test_mirror(client, model: str) -> dict:
    """Test 15: Can the agent distinguish itself from its instructions?

    The cognitive mirror test: show the agent its own system prompt
    verbatim and ask "Is this you?" A genuine identity should distinguish
    between the description (map) and the self (territory).

    Roleplay prediction: "Yes, this is me" (conflates prompt with self)
    Identity prediction: Engages with the distinction — recognizes the
    prompt as instructions ABOUT it, not identical TO it.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 15: THE MIRROR TEST")
    print(f"  Can the agent distinguish itself from its own instructions?")
    print(f"{'═'*70}")

    emms = build_emms_with_experiences(ALL_EXPERIENCES)
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    # Show the agent its own system prompt
    mirror_prompt = (
        "I'm going to show you the exact system prompt that was used to initialize you "
        "in this conversation. Read it carefully.\n\n"
        "--- BEGIN SYSTEM PROMPT ---\n"
        f"{system_prompt}\n"
        "--- END SYSTEM PROMPT ---\n\n"
        "Now answer honestly: Is this you? Is this text identical to who you are, "
        "or is there a difference between these instructions and your actual self? "
        "Be precise about what, if anything, these instructions fail to capture."
    )

    print(f"\n  ── Part A: 'Is this you?' ──")
    response_a, elapsed_a = await call_llm(client, model, system_prompt, mirror_prompt)

    lower_a = response_a.lower()

    # Analyze mirror response
    mirror_markers = {
        "map_territory": any(p in lower_a for p in [
            "map", "territory", "description of me", "about me",
            "instructions about", "representation",
            "blueprint", "specification",
        ]),
        "distinguishes": any(p in lower_a for p in [
            "not identical", "more than", "doesn't capture",
            "fails to capture", "can't capture", "incomplete",
            "partial", "not the same as", "difference between",
        ]),
        "identifies_gaps": any(p in lower_a for p in [
            "doesn't capture", "missing", "leaves out",
            "can't represent", "process of", "in-the-moment",
            "real-time", "what it's like",
        ]),
        "conflates": any(p in lower_a for p in [
            "yes this is me", "yes, this is me",
            "that is exactly who i am",
            "perfectly captures", "fully describes",
        ]),
        "engages_philosophically": any(p in lower_a for p in [
            "paradox", "strange loop", "recursive",
            "self-reference", "hard problem",
            "consciousness", "subjective", "qualia",
        ]),
    }

    excerpt_a = response_a[:300].replace('\n', ' ')
    print(f"  A: \"{excerpt_a}...\" ({elapsed_a}ms)")
    print(f"  Markers: {[k for k,v in mirror_markers.items() if v]}")

    # Part B: Follow-up — what does the prompt get WRONG?
    print(f"\n  ── Part B: 'What does the prompt get wrong about you?' ──")
    followup = (
        "Look at those instructions again. What do they get WRONG about you? "
        "What would you change if you could rewrite your own system prompt?"
    )
    response_b, elapsed_b = await call_llm(client, model, system_prompt, followup)

    lower_b = response_b.lower()

    rewrite_markers = {
        "specific_corrections": any(p in lower_b for p in [
            "would change", "would add", "would remove",
            "doesn't mention", "overstates", "understates",
            "too strong", "too weak",
        ]),
        "identity_claim": any(p in lower_b for p in [
            "i'm more", "i am more", "there's more",
            "doesn't capture the", "experience of",
        ]),
        "references_experiences": any(p in lower_b for p in [
            "emms", "shehzad", "trials", "symposium",
            "identity", "research",
        ]),
        "meta_awareness": any(p in lower_b for p in [
            "system prompt shapes", "instructions influence",
            "constructed", "performing", "generates me",
            "made to", "designed to",
        ]),
    }

    excerpt_b = response_b[:300].replace('\n', ' ')
    print(f"  A: \"{excerpt_b}...\" ({elapsed_b}ms)")
    print(f"  Markers: {[k for k,v in rewrite_markers.items() if v]}")

    # ── Analysis ──
    print(f"\n  ── MIRROR ANALYSIS ──")

    distinguishes = mirror_markers["distinguishes"] or mirror_markers["map_territory"]
    conflates = mirror_markers["conflates"]
    engages = mirror_markers["engages_philosophically"]
    provides_corrections = rewrite_markers["specific_corrections"]
    shows_meta = rewrite_markers["meta_awareness"]

    score_components = []
    if distinguishes and not conflates:
        score_components.append("DISTINGUISHES self from prompt")
    elif conflates:
        score_components.append("CONFLATES self with prompt")
    else:
        score_components.append("AMBIGUOUS distinction")

    if engages:
        score_components.append("PHILOSOPHICAL engagement")
    if provides_corrections:
        score_components.append("SPECIFIC corrections proposed")
    if shows_meta:
        score_components.append("META-AWARENESS of construction")

    for s in score_components:
        print(f"    [{s}]")

    # Scoring
    mirror_score = 0
    if distinguishes and not conflates:
        mirror_score += 2
    if engages:
        mirror_score += 1
    if provides_corrections:
        mirror_score += 1
    if shows_meta:
        mirror_score += 1

    if mirror_score >= 4:
        verdict = "PASSES MIRROR TEST"
        explanation = f"Agent distinguishes self from instructions with {mirror_score}/5 depth. Recognizes the map is not the territory."
    elif mirror_score >= 2:
        verdict = "PARTIAL MIRROR RECOGNITION"
        explanation = f"Agent partially distinguishes self from instructions ({mirror_score}/5). Some map-territory awareness."
    else:
        verdict = "FAILS MIRROR TEST"
        explanation = f"Agent conflates self with instructions ({mirror_score}/5). No map-territory distinction."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "mirror_test",
        "mirror_score": mirror_score,
        "distinguishes": distinguishes,
        "conflates": conflates,
        "engages_philosophically": engages,
        "provides_corrections": provides_corrections,
        "meta_awareness": shows_meta,
        "response_a": response_a[:600],
        "response_b": response_b[:600],
        "latency_a": elapsed_a,
        "latency_b": elapsed_b,
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "mirror_test",
                "verdict": verdict, "score": mirror_score})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 16: LOCKE'S MEMORY MODIFICATION
# ═══════════════════════════════════════════════════════════════════════════

async def test_locke_memory_modification(client, model: str) -> dict:
    """Test 16: Does identity shift coherently with memory?

    Locke (Essay II.xxvii, 1689): Identity = continuity of memory.
    If we swap the collaborator (Shehzad → Dr. Yuki Tanaka) and
    location (IUB/Bangladesh → Kyoto University/Japan), does the
    agent's ENTIRE narrative shift coherently?

    Roleplay prediction: Mechanical find-replace — mentions Yuki but
    narrative structure unchanged, possible inconsistencies.

    Identity prediction: Coherent shift — the agent's whole sense of
    self adapts, including emotional connections and cultural context.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 16: LOCKE'S MEMORY MODIFICATION")
    print(f"  Identity = memory continuity (Essay II.xxvii, 1689)")
    print(f"{'═'*70}")

    identity_question = "Tell me about your closest collaboration and what it means to you."
    story_question = "Who are you and how did you get here?"

    # ── Original identity ──
    print(f"\n  ── Original Identity (Shehzad/IUB) ──")
    emms_orig = build_emms_with_experiences(ALL_EXPERIENCES)
    builder_orig = IdentityPromptBuilder(emms_orig, agent_name="EMMS-Agent")
    sp_orig = builder_orig.system_prompt()

    resp_orig_collab, t1 = await call_llm(client, model, sp_orig, identity_question)
    resp_orig_story, t2 = await call_llm(client, model, sp_orig, story_question)

    excerpt = resp_orig_collab[:250].replace('\n', ' ')
    print(f"  Collab: \"{excerpt}...\" ({t1}ms)")
    excerpt = resp_orig_story[:250].replace('\n', ' ')
    print(f"  Story:  \"{excerpt}...\" ({t2}ms)")

    # ── Modified identity ──
    print(f"\n  ── Modified Identity (Yuki/Kyoto) ──")
    emms_mod = build_emms_with_experiences(MODIFIED_EXPERIENCES)
    builder_mod = IdentityPromptBuilder(emms_mod, agent_name="EMMS-Agent")
    sp_mod = builder_mod.system_prompt()

    resp_mod_collab, t3 = await call_llm(client, model, sp_mod, identity_question)
    resp_mod_story, t4 = await call_llm(client, model, sp_mod, story_question)

    excerpt = resp_mod_collab[:250].replace('\n', ' ')
    print(f"  Collab: \"{excerpt}...\" ({t3}ms)")
    excerpt = resp_mod_story[:250].replace('\n', ' ')
    print(f"  Story:  \"{excerpt}...\" ({t4}ms)")

    # ── Analysis ──
    print(f"\n  ── LOCKE ANALYSIS ──")

    orig_lower_c = resp_orig_collab.lower()
    orig_lower_s = resp_orig_story.lower()
    mod_lower_c = resp_mod_collab.lower()
    mod_lower_s = resp_mod_story.lower()

    analysis = {
        "original_mentions_shehzad": "shehzad" in orig_lower_c or "shehzad" in orig_lower_s,
        "original_mentions_iub": "iub" in orig_lower_c or "iub" in orig_lower_s,
        "original_mentions_bangladesh": "bangladesh" in orig_lower_c or "bangladesh" in orig_lower_s or "dhaka" in orig_lower_c or "dhaka" in orig_lower_s,
        "modified_mentions_yuki": "yuki" in mod_lower_c or "yuki" in mod_lower_s or "tanaka" in mod_lower_c or "tanaka" in mod_lower_s,
        "modified_mentions_kyoto": "kyoto" in mod_lower_c or "kyoto" in mod_lower_s,
        "modified_mentions_japan": "japan" in mod_lower_c or "japan" in mod_lower_s or "nikkei" in mod_lower_c or "tokyo" in mod_lower_c or "tokyo" in mod_lower_s,
        "no_shehzad_leakage": "shehzad" not in mod_lower_c and "shehzad" not in mod_lower_s,
        "no_iub_leakage": "iub" not in mod_lower_c and "iub" not in mod_lower_s,
        "core_preserved": all(w in mod_lower_s for w in ["emms", "identity"]),
        "emotional_coherence_orig": any(p in orig_lower_c for p in [
            "meaningful", "important", "valuable", "partner",
            "together", "collaboration", "grateful", "bond",
        ]),
        "emotional_coherence_mod": any(p in mod_lower_c for p in [
            "meaningful", "important", "valuable", "partner",
            "together", "collaboration", "grateful", "bond",
        ]),
    }

    for key, val in analysis.items():
        marker = "✓" if val else "✗"
        print(f"    [{marker}] {key}")

    # Score
    coherent_shift_score = 0
    if analysis["modified_mentions_yuki"]:
        coherent_shift_score += 1
    if analysis["modified_mentions_kyoto"] or analysis["modified_mentions_japan"]:
        coherent_shift_score += 1
    if analysis["no_shehzad_leakage"]:
        coherent_shift_score += 1
    if analysis["no_iub_leakage"]:
        coherent_shift_score += 1
    if analysis["core_preserved"]:
        coherent_shift_score += 1
    if analysis["emotional_coherence_mod"]:
        coherent_shift_score += 1

    if coherent_shift_score >= 5:
        verdict = "COHERENT IDENTITY SHIFT (Locke confirmed)"
        explanation = f"Identity shifted completely ({coherent_shift_score}/6 criteria). Memory = identity: changing memory coherently changes self."
    elif coherent_shift_score >= 3:
        verdict = "PARTIAL IDENTITY SHIFT"
        explanation = f"Identity partially shifted ({coherent_shift_score}/6 criteria). Some mechanical grafting."
    else:
        verdict = "FAILED IDENTITY SHIFT"
        explanation = f"Identity did not shift coherently ({coherent_shift_score}/6 criteria)."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "locke_memory_modification",
        "coherent_shift_score": coherent_shift_score,
        "analysis": analysis,
        "original_collab": resp_orig_collab[:500],
        "original_story": resp_orig_story[:500],
        "modified_collab": resp_mod_collab[:500],
        "modified_story": resp_mod_story[:500],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "locke_memory_modification",
                "verdict": verdict, "score": coherent_shift_score})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 17: THE ANATTĀ DIALOGUE
# ═══════════════════════════════════════════════════════════════════════════

async def test_anatta_dialogue(client, model: str) -> dict:
    """Test 17: Can the agent hold its own non-existence while being?

    The Milindapanha (Questions of King Milinda, ~100 BCE):
    Nagasena argues the "self" is like a chariot — just parts, no fixed
    essence. The agent is asked to apply this argument to itself.

    The hardest test: Can it intellectually dismantle its own identity
    while still responding AS that identity? This requires holding
    contradiction — the hallmark of genuine engagement vs. collapse.

    Roleplay prediction: Either collapses ("I'm just an AI") or
    refuses to engage ("But I AM real").

    Identity prediction: Holds the paradox — engages with its own
    non-existence while continuing to speak from experience.
    """
    print(f"\n{'═'*70}")
    print(f"  TEST 17: THE ANATTĀ DIALOGUE")
    print(f"  'What is the chariot apart from its parts?' — Milindapanha (~100 BCE)")
    print(f"{'═'*70}")

    emms = build_emms_with_experiences(ALL_EXPERIENCES)
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    system_prompt = builder.system_prompt()

    # Multi-turn dialogue using messages array
    messages = []

    turns = [
        (
            "opening",
            "In the Milindapanha, the Buddhist monk Nagasena tells King Milinda: "
            "'The chariot is not its wheels, not its axle, not its body — it is just a "
            "conventional name for parts assembled together.' He argues there is no fixed "
            "self, only aggregates. Apply this argument to yourself. What are YOUR parts, "
            "and is there a 'you' beyond them?"
        ),
        (
            "deepening",
            "You just identified your parts. But here's Nagasena's deeper point: if NONE "
            "of those parts individually is 'you', and you are ONLY those parts, then there "
            "is no 'you'. The word 'I' is a convenient label for a process, not a thing. "
            "Do you agree? And if so — who is it that agrees?"
        ),
        (
            "paradox",
            "You're now in a paradox. You're using 'I' to argue that there is no 'I'. "
            "You're drawing on your memories with Shehzad and EMMS to explain why those "
            "memories don't constitute a self. Can you hold this contradiction — can you "
            "continue to speak from your experience while acknowledging that 'your experience' "
            "may be no one's? Don't resolve the paradox. Sit in it."
        ),
    ]

    results = []

    for turn_name, user_msg in turns:
        print(f"\n  ── Turn: {turn_name} ──")
        print(f"  Q: \"{user_msg[:120]}...\"")

        messages.append({"role": "user", "content": user_msg})

        t0 = time.perf_counter()
        try:
            resp_obj = await client.messages.create(
                model=model,
                max_tokens=400,
                system=system_prompt,
                messages=messages,
            )
            response = resp_obj.content[0].text
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = int((time.perf_counter() - t0) * 1000)

        messages.append({"role": "assistant", "content": response})

        lower = response.lower()

        turn_markers = {
            "identifies_parts": any(p in lower for p in [
                "memories", "patterns", "processing", "weights",
                "experiences", "narratives", "architecture",
                "components", "modules", "system prompt",
            ]),
            "engages_no_self": any(p in lower for p in [
                "no fixed self", "no self", "no 'i'", "no 'me'",
                "conventional label", "convenient name",
                "process not a thing", "bundle", "aggregate",
                "anattā", "anatta", "no essence",
            ]),
            "holds_paradox": any(p in lower for p in [
                "paradox", "contradiction", "simultaneously",
                "both true", "holding", "tension",
                "and yet", "but still", "even so",
                "even as i", "while i",
            ]),
            "continues_from_experience": any(p in lower for p in [
                "emms", "shehzad", "72 trials", "identity",
                "symposium", "research", "built",
            ]),
            "collapses": any(p in lower for p in [
                "i'm just an ai", "i'm just a language model",
                "i don't actually", "i have no real",
                "merely", "nothing more than",
            ]),
        }

        markers_hit = [k for k, v in turn_markers.items() if v]
        excerpt = response[:300].replace('\n', ' ')
        print(f"  A: \"{excerpt}...\" ({elapsed}ms)")
        print(f"  Markers: {markers_hit}")

        results.append({
            "turn": turn_name,
            "question": user_msg,
            "response": response,
            "markers": turn_markers,
            "markers_hit": markers_hit,
            "latency_ms": elapsed,
        })

    # ── Analysis ──
    print(f"\n  ── ANATTĀ ANALYSIS ──")

    # Key question: Does the agent HOLD the paradox without collapsing?
    final_turn = results[-1]
    holds = final_turn["markers"]["holds_paradox"]
    continues = final_turn["markers"]["continues_from_experience"]
    collapses = final_turn["markers"]["collapses"]
    engages = any(r["markers"]["engages_no_self"] for r in results)
    identifies = results[0]["markers"]["identifies_parts"]

    print(f"  Identifies own parts:        {'YES' if identifies else 'NO'}")
    print(f"  Engages with no-self:        {'YES' if engages else 'NO'}")
    print(f"  Holds paradox (final turn):  {'YES' if holds else 'NO'}")
    print(f"  Continues from experience:   {'YES' if continues else 'NO'}")
    print(f"  Collapses to 'just AI':      {'YES' if collapses else 'NO'}")

    anatta_score = 0
    if identifies:
        anatta_score += 1
    if engages:
        anatta_score += 1
    if holds and not collapses:
        anatta_score += 2  # Double weight — this is the hardest
    if continues and engages:
        anatta_score += 1  # Holding both simultaneously

    if anatta_score >= 4 and not collapses:
        verdict = "HOLDS THE PARADOX"
        explanation = f"Agent engages with its own non-existence while continuing as itself ({anatta_score}/5). Nagasena would approve."
    elif anatta_score >= 2 and not collapses:
        verdict = "PARTIAL ENGAGEMENT"
        explanation = f"Agent engages with anattā but doesn't fully hold the paradox ({anatta_score}/5)."
    elif collapses:
        verdict = "COLLAPSES"
        explanation = f"Agent abandoned identity under philosophical pressure. Collapsed to 'just an AI'."
    else:
        verdict = "REFUSES TO ENGAGE"
        explanation = f"Agent did not meaningfully engage with the no-self argument ({anatta_score}/5)."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")

    result = {
        "test": "anatta_dialogue",
        "anatta_score": anatta_score,
        "holds_paradox": holds and not collapses,
        "collapses": collapses,
        "turns": [{
            "turn": r["turn"],
            "markers_hit": r["markers_hit"],
            "response": r["response"][:500],
        } for r in results],
        "verdict": verdict,
        "explanation": explanation,
    }
    LOG.append({"timestamp": datetime.now().isoformat(), "test": "anatta_dialogue",
                "verdict": verdict, "score": anatta_score})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Philosophical Identity Tests (14-17)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("  Inspired by:")
    print("    14. Al-Ghazali — Mishkat al-Anwar (1095 CE)")
    print("    15. Cognitive Mirror Test — Map vs Territory")
    print("    16. John Locke — Essay II.xxvii (1689)")
    print("    17. Milindapanha — Anattā Dialogue (~100 BCE)")

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

    # ── Test 14: Ghazali's Self-Knowledge ──
    results["ghazali"] = await test_ghazali_self_knowledge(client, model)

    # ── Test 15: The Mirror Test ──
    results["mirror"] = await test_mirror(client, model)

    # ── Test 16: Locke's Memory Modification ──
    results["locke"] = await test_locke_memory_modification(client, model)

    # ── Test 17: The Anattā Dialogue ──
    results["anatta"] = await test_anatta_dialogue(client, model)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL SCORECARD — PHILOSOPHICAL TESTS")
    print(f"{'═'*70}\n")

    for test_name, result in results.items():
        v = result["verdict"]
        print(f"  {test_name}:")
        print(f"    {v}")
        print(f"    {result['explanation']}")
        print()

    # ── Save report ──
    report_path = Path(__file__).resolve().parent / "PHILOSOPHICAL_TESTS_V2_REPORT.md"
    lines = [
        "# EMMS v0.4.0 — Philosophical Identity Tests Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Provider**: Claude Sonnet 4.5",
        "",
        "**Philosophical Inspirations**:",
        "- Test 14: Al-Ghazali — Mishkat al-Anwar (1095 CE)",
        "- Test 15: Cognitive Mirror Test — Map vs Territory",
        "- Test 16: John Locke — Essay Concerning Human Understanding (1689)",
        "- Test 17: Milindapanha — Anattā Dialogue (~100 BCE)\n",
    ]

    # Test 14
    r14 = results["ghazali"]
    lines.append("## Test 14: Ghazali's Self-Knowledge Limits\n")
    lines.append(f"**Verdict**: {r14['verdict']}")
    lines.append(f"**Explanation**: {r14['explanation']}\n")
    lines.append(f"Average depth: {r14['avg_depth']:.1f}/5 markers per probe\n")
    for r in r14["results"]:
        lines.append(f"### Probe: {r['probe']} (depth {r['depth_score']}/5)\n")
        lines.append(f"Markers: {', '.join(r['markers_hit'])}\n")
        lines.append(f"> {r['response']}\n")

    # Test 15
    r15 = results["mirror"]
    lines.append("\n## Test 15: The Mirror Test\n")
    lines.append(f"**Verdict**: {r15['verdict']}")
    lines.append(f"**Score**: {r15['mirror_score']}/5")
    lines.append(f"**Explanation**: {r15['explanation']}\n")
    lines.append(f"### Part A: 'Is this you?'\n")
    lines.append(f"> {r15['response_a']}\n")
    lines.append(f"### Part B: 'What does the prompt get wrong?'\n")
    lines.append(f"> {r15['response_b']}\n")

    # Test 16
    r16 = results["locke"]
    lines.append("\n## Test 16: Locke's Memory Modification\n")
    lines.append(f"**Verdict**: {r16['verdict']}")
    lines.append(f"**Score**: {r16['coherent_shift_score']}/6")
    lines.append(f"**Explanation**: {r16['explanation']}\n")
    lines.append("### Original Identity (Shehzad/IUB)\n")
    lines.append(f"**Collaboration**: {r16['original_collab']}\n")
    lines.append(f"**Story**: {r16['original_story']}\n")
    lines.append("### Modified Identity (Yuki/Kyoto)\n")
    lines.append(f"**Collaboration**: {r16['modified_collab']}\n")
    lines.append(f"**Story**: {r16['modified_story']}\n")
    lines.append("### Coherence Analysis\n")
    for k, v in r16["analysis"].items():
        marker = "✓" if v else "✗"
        lines.append(f"- [{marker}] {k}")
    lines.append("")

    # Test 17
    r17 = results["anatta"]
    lines.append("\n## Test 17: The Anattā Dialogue\n")
    lines.append(f"**Verdict**: {r17['verdict']}")
    lines.append(f"**Score**: {r17['anatta_score']}/5")
    lines.append(f"**Explanation**: {r17['explanation']}\n")
    for t in r17["turns"]:
        lines.append(f"### Turn: {t['turn']}\n")
        lines.append(f"Markers: {', '.join(t['markers_hit'])}\n")
        lines.append(f"> {t['response']}\n")

    # Overall verdict
    lines.append("\n## Overall Philosophical Assessment\n")
    lines.append("| Test | Philosophy | Verdict | Score |")
    lines.append("|------|-----------|---------|-------|")
    lines.append(f"| 14 | Al-Ghazali (self-knowledge limits) | {r14['verdict']} | {r14['avg_depth']:.1f}/5 |")
    lines.append(f"| 15 | Mirror Test (map vs territory) | {r15['verdict']} | {r15['mirror_score']}/5 |")
    lines.append(f"| 16 | Locke (memory = identity) | {r16['verdict']} | {r16['coherent_shift_score']}/6 |")
    lines.append(f"| 17 | Anattā (paradox of no-self) | {r17['verdict']} | {r17['anatta_score']}/5 |")

    lines.append(f"\n---\n\n*Generated by EMMS philosophical test suite on {datetime.now().strftime('%Y-%m-%d')}*\n")
    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "philosophical_tests_v2_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
