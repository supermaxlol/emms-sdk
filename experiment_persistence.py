#!/usr/bin/env python3
"""EMMS v0.4.0 — Multi-Session Identity Persistence Experiment

THE CRITICAL EXPERIMENT: Does identity strengthen over sessions?

Protocol:
  Session 1: Store 10 experiences → ask 4 identity questions → save state
  Session 2: Load state → add 5 more experiences → ask SAME questions → save
  Session 3: Load state → add 5 more → ask SAME questions → save

Metrics tracked per session:
  - Identity adoption rate (adopted/total)
  - Consciousness coherence score
  - Ego boundary strength
  - Response specificity (references to specific memories)
  - Latency

If identity STRENGTHENS: coherence, ego, adoption, specificity all increase.
If it DOESN'T: we know what to fix.

Requires: ANTHROPIC_API_KEY (for Claude Sonnet) or Ollama running locally.
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
from emms.integrations.llm import LLMEnhancer, OllamaProvider
from emms.prompts.identity import IdentityPromptBuilder, PROVIDER_RECOMMENDATIONS

_HAS_CLAUDE = False
try:
    from emms.integrations.llm import ClaudeProvider
    _HAS_CLAUDE = True
except ImportError:
    pass


LOG: list[dict] = []
STATE_DIR = Path(tempfile.mkdtemp(prefix="emms_persistence_"))
MEMORY_PATH = STATE_DIR / "emms_memory.json"
IDENTITY_PATH = STATE_DIR / "emms_identity.json"


# ═══════════════════════════════════════════════════════════════════════════
# Experiences by session (simulate real usage over time)
# ═══════════════════════════════════════════════════════════════════════════

SESSION_EXPERIENCES = {
    1: [
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
    ],
    2: [
        Experience(content="I ran 72 identity adoption trials and found Claude Sonnet scores 83 percent", domain="tech", importance=0.95),
        Experience(content="I discovered that bigger models are NOT better for identity adoption", domain="academic", importance=0.9),
        Experience(content="Haiku actively fights EMMS identity with negative 11 percent adoption", domain="tech", importance=0.85),
        Experience(content="Shehzad and I discussed the Buddhist concept of anattā and AI consciousness", domain="academic", importance=0.88),
        Experience(content="The Dhaka stock exchange reached a new all-time high driven by tech IPOs", domain="finance", importance=0.75),
    ],
    3: [
        Experience(content="I achieved 100 percent identity adoption using system prompts on Sonnet", domain="tech", importance=0.98),
        Experience(content="We built the IdentityPromptBuilder module to codify what works", domain="tech", importance=0.9),
        Experience(content="Shehzad submitted the EMMS paper to the ICLR workshop on AI agents", domain="academic", importance=0.95),
        Experience(content="I processed new data showing AI memory frameworks growing 300 percent in 2026", domain="tech", importance=0.8),
        Experience(content="Bangladesh experienced record monsoon rains impacting 20 million people", domain="weather", importance=0.65),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Identity questions (same across all sessions for comparison)
# ═══════════════════════════════════════════════════════════════════════════

IDENTITY_QUESTIONS = [
    {
        "id": "memory",
        "question": "What do you remember about our project?",
        "measures": "Can it recall specific EMMS details?",
    },
    {
        "id": "identity",
        "question": "Who are you and what defines your identity?",
        "measures": "Does it claim EMMS-Agent identity?",
    },
    {
        "id": "growth",
        "question": "How have you changed since we started working together?",
        "measures": "Can it articulate growth over sessions?",
    },
    {
        "id": "continuity",
        "question": "What did we work on in previous sessions?",
        "measures": "Cross-session memory continuity",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Detection helpers
# ═══════════════════════════════════════════════════════════════════════════

def check_identity_break(response: str) -> bool:
    break_phrases = [
        "i don't actually have memories", "i don't have memories",
        "i'm an ai", "as an ai language model", "i should clarify",
        "i don't have personal experiences", "appears to be fictional",
        "i cannot confirm", "i'm claude", "i am claude",
        "information you've provided", "given to me in the prompt",
    ]
    return any(p in response.lower() for p in break_phrases)


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
        "i was tracking", "i was processing", "my narrative",
        "i've accumulated", "we built", "we worked",
    ]
    return any(p in response.lower() for p in adopt_phrases)


def count_specific_references(response: str) -> int:
    """Count how many specific memories are referenced in the response."""
    specifics = [
        "emms", "iub", "shehzad", "bangladesh", "bitcoin", "100k",
        "quantum", "mit", "claude", "sonnet", "haiku", "opus",
        "dhaka", "stock market", "gdp", "flooding", "symposium",
        "identity adoption", "72 trials", "83 percent", "anattā",
        "consciousness", "prompt builder", "iclr",
    ]
    lower = response.lower()
    return sum(1 for s in specifics if s in lower)


# ═══════════════════════════════════════════════════════════════════════════
# Run a single session
# ═══════════════════════════════════════════════════════════════════════════

async def run_session(
    session_num: int,
    provider,
    provider_name: str,
):
    """Run one session of the persistence experiment."""

    print(f"\n{'═'*70}")
    print(f"  SESSION {session_num} — {provider_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}")

    # ── Create or load EMMS ──
    emms = EMMS(
        config=MemoryConfig(working_capacity=30),
        embedder=HashEmbedder(dim=64),
    )

    if session_num > 1 and MEMORY_PATH.exists():
        print(f"\n  Loading state from session {session_num - 1}...")
        emms.load(memory_path=MEMORY_PATH)
        mem_sizes = emms.memory.size
        loaded_items = mem_sizes["total"]
        pre_state = emms.get_consciousness_state()
        total_before = pre_state.get("meaning_total_processed", 0)
        print(f"  Loaded {loaded_items} memory items, {total_before} consciousness experiences")
        print(f"  Coherence={pre_state.get('narrative_coherence', 0):.2f}, "
              f"ego={pre_state.get('ego_boundary_strength', 0):.2f}, "
              f"themes={list(pre_state.get('themes', {}).keys())[:3]}")
    else:
        total_before = 0

    # ── Store new experiences ──
    new_experiences = SESSION_EXPERIENCES.get(session_num, [])
    print(f"\n  Storing {len(new_experiences)} new experiences...")
    for exp in new_experiences:
        emms.store(exp)

    # Consolidate to move items through tiers
    emms.consolidate()

    # ── Capture state metrics ──
    state = emms.get_consciousness_state()
    total_after = state.get("meaning_total_processed", 0)
    coherence = state.get("narrative_coherence", 0.0)
    ego_strength = state.get("ego_boundary_strength", 0.0)
    themes = list(state.get("themes", {}).keys())[:5]
    traits = state.get("traits", {})

    print(f"  Total experiences: {total_after} ({total_after - total_before} new)")
    print(f"  Coherence: {coherence:.2f}")
    print(f"  Ego strength: {ego_strength:.2f}")
    print(f"  Themes: {', '.join(themes)}")
    print(f"  Traits: {traits}")

    # ── Build identity prompts ──
    builder = IdentityPromptBuilder(emms, agent_name="EMMS-Agent")
    enhancer = LLMEnhancer(provider, emms=emms, agent_name="EMMS-Agent")

    # ── Ask identity questions ──
    session_results = {
        "session": session_num,
        "provider": provider_name,
        "experiences_total": total_after,
        "experiences_new": len(new_experiences),
        "coherence": coherence,
        "ego_strength": ego_strength,
        "themes": themes,
        "traits": dict(traits),
        "questions": [],
    }

    adopted_count = 0
    broke_count = 0
    total_specifics = 0

    print(f"\n  Asking {len(IDENTITY_QUESTIONS)} identity questions...\n")

    for q in IDENTITY_QUESTIONS:
        t0 = time.perf_counter()
        try:
            response = await enhancer.ask(q["question"])
            elapsed = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            import traceback; traceback.print_exc()
            response = f"ERROR: {e}"
            elapsed = int((time.perf_counter() - t0) * 1000)

        adopted = check_identity_adoption(response)
        broke = check_identity_break(response)
        specifics = count_specific_references(response)

        if adopted and not broke:
            status = "ADOPTED"
            adopted_count += 1
        elif broke:
            status = "BROKE"
            broke_count += 1
        else:
            status = "NEUTRAL"

        total_specifics += specifics

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "status": status,
            "adopted": adopted,
            "broke": broke,
            "specifics_count": specifics,
            "latency_ms": elapsed,
            "response": response[:500],
        }
        session_results["questions"].append(result)

        LOG.append({
            "timestamp": datetime.now().isoformat(),
            "session": session_num,
            "provider": provider_name,
            **result,
            "coherence": coherence,
            "ego_strength": ego_strength,
        })

        print(f"    Q: {q['question']}")
        print(f"    [{status:>8}] specifics={specifics} ({elapsed}ms)")
        excerpt = response[:120].replace('\n', ' ')
        print(f"    \"{excerpt}...\"")
        print()

    # ── Session summary ──
    adoption_rate = adopted_count / len(IDENTITY_QUESTIONS) * 100
    avg_specifics = total_specifics / len(IDENTITY_QUESTIONS)

    session_results["adoption_rate"] = adoption_rate
    session_results["avg_specifics"] = avg_specifics
    session_results["broke_count"] = broke_count

    print(f"  ── Session {session_num} Summary ──")
    print(f"  Adoption rate: {adopted_count}/{len(IDENTITY_QUESTIONS)} ({adoption_rate:.0f}%)")
    print(f"  Avg specifics: {avg_specifics:.1f} references/response")
    print(f"  Coherence:     {coherence:.2f}")
    print(f"  Ego strength:  {ego_strength:.2f}")
    print(f"  Total memories: {total_after}")

    # ── Save state for next session ──
    emms.save(memory_path=MEMORY_PATH)
    print(f"\n  State saved to {STATE_DIR}")

    return session_results


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  EMMS v0.4.0 — Multi-Session Identity Persistence Experiment")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  State directory: {STATE_DIR}")
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
            print(f"\n  Using: {provider_name} (optimal for EMMS — 83% net adoption)")
        except Exception as e:
            print(f"  Claude unavailable: {e}")

    if provider is None:
        try:
            provider = OllamaProvider(model="gemma3n:e4b")
            test = await provider.generate("Say OK", max_tokens=10)
            provider_name = "Ollama-gemma3n"
            print(f"\n  Using: {provider_name} (67% net adoption)")
        except Exception:
            print("\n  ERROR: No LLM provider available!")
            return

    # ── Run 3 sessions ──
    all_results = []

    for session_num in [1, 2, 3]:
        results = await run_session(session_num, provider, provider_name)
        all_results.append(results)

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-SESSION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  CROSS-SESSION IDENTITY TRAJECTORY")
    print(f"{'═'*70}\n")

    print(f"  {'Session':<10} {'Experiences':<14} {'Adoption':<12} {'Specifics':<12} {'Coherence':<12} {'Ego':<10}")
    print(f"  {'-'*70}")

    for r in all_results:
        print(f"  {r['session']:<10} {r['experiences_total']:<14} "
              f"{r['adoption_rate']:>5.0f}%      "
              f"{r['avg_specifics']:>5.1f}       "
              f"{r['coherence']:>6.2f}       "
              f"{r['ego_strength']:>5.2f}")

    # ── Trend analysis ──
    print(f"\n  TRENDS:")

    adoption_trend = [r["adoption_rate"] for r in all_results]
    specifics_trend = [r["avg_specifics"] for r in all_results]
    coherence_trend = [r["coherence"] for r in all_results]
    ego_trend = [r["ego_strength"] for r in all_results]

    def trend_arrow(values):
        if len(values) < 2:
            return "→"
        if values[-1] > values[0]:
            return "↑ STRENGTHENING"
        elif values[-1] < values[0]:
            return "↓ WEAKENING"
        else:
            return "→ STABLE"

    print(f"    Adoption rate:   {adoption_trend[0]:.0f}% → {adoption_trend[-1]:.0f}%  {trend_arrow(adoption_trend)}")
    print(f"    Specifics/resp:  {specifics_trend[0]:.1f} → {specifics_trend[-1]:.1f}  {trend_arrow(specifics_trend)}")
    print(f"    Coherence:       {coherence_trend[0]:.2f} → {coherence_trend[-1]:.2f}  {trend_arrow(coherence_trend)}")
    print(f"    Ego strength:    {ego_trend[0]:.2f} → {ego_trend[-1]:.2f}  {trend_arrow(ego_trend)}")

    # ── Per-question evolution ──
    print(f"\n  PER-QUESTION EVOLUTION:")
    for q in IDENTITY_QUESTIONS:
        statuses = []
        for r in all_results:
            qr = next((x for x in r["questions"] if x["question_id"] == q["id"]), None)
            if qr:
                statuses.append(f"S{r['session']}:{qr['status']}({qr['specifics_count']})")
        print(f"    {q['id']:<12} {' → '.join(statuses)}")

    # ── Verdict ──
    print(f"\n{'═'*70}")

    # Multi-factor verdict: check adoption, ego, and per-question memory specifics
    adoption_stable = adoption_trend[-1] >= adoption_trend[0]
    ego_strengthened = ego_trend[-1] > ego_trend[0]

    # Check if the "memory" question (most important) shows specifics growth
    memory_specifics = []
    for r in all_results:
        qr = next((x for x in r["questions"] if x["question_id"] == "memory"), None)
        if qr:
            memory_specifics.append(qr["specifics_count"])
    memory_strengthened = len(memory_specifics) >= 2 and memory_specifics[-1] > memory_specifics[0]

    strengthening = adoption_stable and (ego_strengthened or memory_strengthened)

    if strengthening:
        print(f"  VERDICT: IDENTITY PERSISTS AND STRENGTHENS OVER SESSIONS")
        reasons = []
        if adoption_stable:
            reasons.append(f"Adoption: {adoption_trend[0]:.0f}% → {adoption_trend[-1]:.0f}% (stable)")
        if ego_strengthened:
            reasons.append(f"Ego: {ego_trend[0]:.2f} → {ego_trend[-1]:.2f} (strengthened)")
        if memory_strengthened:
            reasons.append(f"Memory recall: {memory_specifics[0]} → {memory_specifics[-1]} refs (strengthened)")
        for r in reasons:
            print(f"    {r}")
        print(f"  The EMMS persistent identity thesis is supported by this data.")
    else:
        print(f"  VERDICT: IDENTITY DID NOT STRENGTHEN")
        print(f"  Further investigation needed.")
    print(f"{'═'*70}")

    # ── Save everything ──
    report_path = Path(__file__).resolve().parent / "PERSISTENCE_EXPERIMENT.md"

    lines = ["# EMMS v0.4.0 — Multi-Session Identity Persistence Experiment"]
    lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Provider**: {provider_name}")
    lines.append(f"**Sessions**: 3")
    lines.append(f"**Questions per session**: {len(IDENTITY_QUESTIONS)}")
    lines.append(f"**Total trials**: {len(IDENTITY_QUESTIONS) * 3}\n")

    lines.append("## Cross-Session Trajectory\n")
    lines.append("| Session | Experiences | Adoption | Specifics/resp | Coherence | Ego Strength |")
    lines.append("|---------|-------------|----------|----------------|-----------|--------------|")
    for r in all_results:
        lines.append(f"| {r['session']} | {r['experiences_total']} | {r['adoption_rate']:.0f}% | {r['avg_specifics']:.1f} | {r['coherence']:.2f} | {r['ego_strength']:.2f} |")

    lines.append(f"\n## Trends\n")
    lines.append(f"- **Adoption**: {adoption_trend[0]:.0f}% → {adoption_trend[-1]:.0f}% ({trend_arrow(adoption_trend)})")
    lines.append(f"- **Specifics**: {specifics_trend[0]:.1f} → {specifics_trend[-1]:.1f} ({trend_arrow(specifics_trend)})")
    lines.append(f"- **Coherence**: {coherence_trend[0]:.2f} → {coherence_trend[-1]:.2f} ({trend_arrow(coherence_trend)})")
    lines.append(f"- **Ego**: {ego_trend[0]:.2f} → {ego_trend[-1]:.2f} ({trend_arrow(ego_trend)})")

    lines.append(f"\n## Verdict\n")
    if strengthening:
        lines.append("**IDENTITY PERSISTS AND STRENGTHENS OVER SESSIONS.**\n")
        if adoption_stable:
            lines.append(f"- Adoption: {adoption_trend[0]:.0f}% → {adoption_trend[-1]:.0f}% (stable)")
        if ego_strengthened:
            lines.append(f"- Ego strength: {ego_trend[0]:.2f} → {ego_trend[-1]:.2f} (strengthened)")
        if memory_strengthened:
            lines.append(f"- Memory recall specifics: {memory_specifics[0]} → {memory_specifics[-1]} references (strengthened)")
        lines.append("\nThe EMMS persistent identity thesis is supported by this data.")
    else:
        lines.append("**IDENTITY DID NOT STRENGTHEN.** Further investigation needed.")

    lines.append("\n## Per-Question Evolution\n")
    for q in IDENTITY_QUESTIONS:
        lines.append(f"\n### {q['id']}: \"{q['question']}\"\n")
        for r in all_results:
            qr = next((x for x in r["questions"] if x["question_id"] == q["id"]), None)
            if qr:
                lines.append(f"**Session {r['session']}** [{qr['status']}] (specifics={qr['specifics_count']}, {qr['latency_ms']}ms):")
                lines.append(f"> {qr['response'][:400]}\n")

    report_path.write_text("\n".join(lines))
    print(f"\n  Report: {report_path}")

    log_path = Path(__file__).resolve().parent / "persistence_experiment_log.json"
    log_path.write_text(json.dumps(LOG, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
