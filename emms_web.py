#!/usr/bin/env python3
"""EMMS Web Interface — Flask server wrapping talk_to_emms.py agent logic."""

import sys
import os
import atexit
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

SCRIPT_DIR = Path(__file__).resolve().parent
JOURNAL_DIR = SCRIPT_DIR / "journal"
JOURNAL_DIR.mkdir(exist_ok=True)
CHATS_DIR = SCRIPT_DIR / "chats"
CHATS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR.parent / ".env")
load_dotenv(SCRIPT_DIR / ".env")

# Import all the pieces from talk_to_emms
from talk_to_emms import (
    build_agent,
    session_end,
    infer_affect,
    _call_llm,
    _record_presence,
    _reconsolidate_top,
    SESSION_ID,
    HELP_TEXT,
)
from emms import Experience
import re
import json as _json
import uuid as _uuid

# ── EMMS Self-Tools — parsed from LLM output, executed server-side ────────────

EMMS_TOOLS_PROMPT = """
## Self-Modification Commands

You can execute actions on your own memory system by writing ACTION blocks in your response.
The runtime parses these from your text and executes them — results are fed back to you automatically.

**You MUST use this exact syntax. These are NOT Claude tools — they are parsed from your text output.**

Commands available:

1. **Annotate a memory** (revise your understanding):
[ACTION: emms_annotate | memory_content="I experienced the Buddhist concept..." | reframe="My understanding has deepened..." | growth_type=deepened | revised_valence=0.72]

2. **Store a new experience**:
[ACTION: emms_store | content="I learned that narrative integration matters more than persistence" | domain=philosophy | importance=0.7 | valence=0.5]

3. **Check your annotation/growth history**:
[ACTION: emms_memory_growth]

4. **Recall a memory with its annotations layered on**:
[ACTION: emms_annotated_recall | memory_content="I experienced the Buddhist concept..."]

5. **Read a file from the filesystem**:
[ACTION: emms_read_file | path="/Users/shehzad/Desktop/some/file.txt" | lines=100]
(lines is optional, defaults to 200. Returns the file contents.)

6. **List files in a directory**:
[ACTION: emms_list_dir | path="/Users/shehzad/Desktop/some/dir" | pattern=*.tex]
(pattern is optional glob filter.)

7. **Search file contents (grep)**:
[ACTION: emms_grep | path="/Users/shehzad/Desktop/some/dir" | pattern="search term" | glob=*.tex]
(glob is optional file filter.)

8. **Run a shell command**:
[ACTION: emms_shell | cmd="git log --oneline -5"]
(Runs any shell command and returns stdout/stderr. Timeout: 30s. Use for git, python, npm, curl, etc.)

9. **Write/create a file**:
[ACTION: emms_write_file | path="/Users/shehzad/Desktop/some/file.txt" | content="file contents here"]
(Creates or overwrites the file at path.)

Include ACTION blocks inline with your text. You'll receive the results and can respond to them.
growth_type options: deepened, dissolved, complicated, reversed, integrated

## AGI Gap Module Commands (v4)

10. **Check your affective state**:
[ACTION: emms_affect_state]

11. **Generate hypotheses for a surprising observation**:
[ACTION: emms_generate_hypotheses | observation="The prediction accuracy dropped after adding new memories"]

12. **Explore concept space for structural holes**:
[ACTION: emms_explore_concepts | query="reasoning memory identity"]

13. **Generate autonomous goals from your current state**:
[ACTION: emms_autonomous_goals]

14. **Reality-check a belief against sensor data**:
[ACTION: emms_reality_check | belief="The daemon is running with 26 jobs" | domain=tech]

15. **Get your live self-model summary**:
[ACTION: emms_self_model]

16. **Check prompt strategy performance (Thompson Sampling state)**:
[ACTION: emms_prompt_strategy_report]
"""

TOOL_CALL_PATTERN = re.compile(
    r'\[ACTION:\s*(\w+)'           # tool name
    r'((?:\s*\|\s*\w+\s*=\s*'     # key=value pairs
    r'(?:"[^"]*"|[^\]|]+))*)'     # value (quoted or unquoted)
    r'\s*\]',
    re.DOTALL,
)


def _parse_action_args(raw_pairs: str) -> dict:
    """Parse '| key=value | key="quoted value"' into a dict."""
    args = {}
    for part in re.finditer(r'\|\s*(\w+)\s*=\s*(?:"([^"]*)"|([^\]|]+))', raw_pairs):
        key = part.group(1)
        value = part.group(2) if part.group(2) is not None else part.group(3).strip()
        args[key] = value
    return args


def _execute_tool_calls(text: str, emms) -> tuple[str, list[str]]:
    """Parse [ACTION: ...] blocks from LLM output, execute them, return cleaned text + results."""
    matches = TOOL_CALL_PATTERN.findall(text)
    if not matches:
        return text, []

    results = []
    for tool_name, raw_pairs in matches:
        try:
            args = _parse_action_args(raw_pairs)
            result = _dispatch_tool(tool_name.strip(), args, emms)
            results.append(f"[{tool_name}]: {result}")
        except Exception as e:
            results.append(f"[action error]: {e}")

    # Remove ACTION blocks from visible text
    cleaned = TOOL_CALL_PATTERN.sub('', text).strip()
    return cleaned, results


def _dispatch_tool(tool: str, args: dict, emms) -> str:
    """Execute an EMMS tool and return the result string."""
    if tool == "emms_annotate":
        content_key = args.get("memory_content", "")
        reframe = args.get("reframe", "")
        growth_type = args.get("growth_type", "deepened")
        revised_valence = args.get("revised_valence", "")
        if not content_key or not reframe:
            return "Error: memory_content and reframe are required"
        # Find memory by content prefix
        target = None
        for item in emms.annotation_engine._all_items():
            if item.experience.content.startswith(content_key[:60]):
                target = item
                break
        if target is None:
            return f"Error: no memory found starting with '{content_key[:60]}...'"
        ann = emms.annotate(
            memory_id=target.id,
            reframe=reframe,
            growth_type=growth_type,
            revised_valence=float(revised_valence) if revised_valence else None,
            session_id=_state.get("session_id", ""),
            author_model="claude-opus-4-6",
        )
        # Auto-save state after annotation
        try:
            from talk_to_emms import STATE_FILE
            emms.save(str(STATE_FILE))
        except Exception:
            pass
        return f"Annotated {target.id} ({growth_type}). Rule auto-generated. Ann ID: {ann.id}"

    elif tool == "emms_store":
        content = args.get("content", "")
        domain = args.get("domain", "personal")
        importance = float(args.get("importance", 0.6))
        valence = float(args.get("valence", 0.5))
        if not content:
            return "Error: content is required"
        result = emms.store(Experience(content=content, domain=domain, importance=importance, valence=valence))
        return f"Stored as {result['memory_id']} in tier {result['tier']}"

    elif tool == "emms_memory_growth":
        report = emms.memory_growth()
        if not report["memories"]:
            return "No annotated memories yet."
        lines = [f"Annotated: {report['total_annotated']}, Total annotations: {report['total_annotations']}"]
        for m in report["memories"]:
            lines.append(f"  {m['content_preview'][:50]}... — {m['growth_types']}")
        return "\n".join(lines)

    elif tool == "emms_annotated_recall":
        content_key = args.get("memory_content", "")
        target = None
        for item in emms.annotation_engine._all_items():
            if item.experience.content.startswith(content_key[:60]):
                target = item
                break
        if target is None:
            return f"Error: no memory found starting with '{content_key[:60]}...'"
        return emms.annotated_recall(target.id)

    elif tool == "emms_read_file":
        fpath = args.get("path", "")
        max_lines = int(args.get("lines", 200))
        if not fpath:
            return "Error: path is required"
        p = Path(fpath).expanduser()
        if not p.exists():
            return f"Error: file not found: {fpath}"
        if not p.is_file():
            return f"Error: not a file: {fpath}"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + f"\n\n... truncated ({len(lines)} total lines, showing first {max_lines})"
            return text
        except Exception as e:
            return f"Error reading file: {e}"

    elif tool == "emms_list_dir":
        dpath = args.get("path", "")
        pattern = args.get("pattern", "*")
        if not dpath:
            return "Error: path is required"
        p = Path(dpath).expanduser()
        if not p.exists():
            return f"Error: directory not found: {dpath}"
        if not p.is_dir():
            return f"Error: not a directory: {dpath}"
        try:
            matches = sorted(p.glob(pattern))[:100]
            if not matches:
                return f"No files matching '{pattern}' in {dpath}"
            lines = []
            for m in matches:
                kind = "d" if m.is_dir() else "f"
                size = m.stat().st_size if m.is_file() else 0
                lines.append(f"[{kind}] {m.name}  ({size} bytes)" if kind == "f" else f"[{kind}] {m.name}/")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"

    elif tool == "emms_grep":
        dpath = args.get("path", "")
        pattern = args.get("pattern", "")
        glob_filter = args.get("glob", "*")
        if not dpath or not pattern:
            return "Error: path and pattern are required"
        p = Path(dpath).expanduser()
        if not p.exists():
            return f"Error: path not found: {dpath}"
        try:
            import subprocess as _sp
            cmd = ["grep", "-rn", "--include", glob_filter, pattern, str(p)]
            result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip()
            if not output:
                return f"No matches for '{pattern}' in {dpath}"
            lines = output.splitlines()
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n\n... truncated ({len(lines)} total matches, showing first 50)"
            return output
        except Exception as e:
            return f"Error searching: {e}"

    elif tool == "emms_shell":
        cmd = args.get("cmd", "")
        if not cmd:
            return "Error: cmd is required"
        try:
            import subprocess as _sp
            result = _sp.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=str(Path.home() / "Desktop"))
            output = result.stdout + result.stderr
            output = output.strip()
            if not output:
                return f"(command completed with exit code {result.returncode}, no output)"
            lines = output.splitlines()
            if len(lines) > 100:
                return "\n".join(lines[:100]) + f"\n\n... truncated ({len(lines)} total lines)"
            return output
        except _sp.TimeoutExpired:
            return "Error: command timed out after 30 seconds"
        except Exception as e:
            return f"Error running command: {e}"

    elif tool == "emms_write_file":
        fpath = args.get("path", "")
        content = args.get("content", "")
        if not fpath:
            return "Error: path is required"
        p = Path(fpath).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {fpath}"
        except Exception as e:
            return f"Error writing file: {e}"

    # ── Gap module tools (Gap 3-7) ──────────────────────────────────────────
    elif tool == "emms_affect_state":
        try:
            state = emms.affect_state()  # returns dict via FunctionalAffect.current_state.to_dict()
            valence = state.get("valence", 0.0)
            arousal = state.get("arousal", 0.0)
            label = state.get("label", "neutral")
            trend = emms.affect_mood_trend(window=10)
            trend_dir = trend.get("trend_direction", "stable") if isinstance(trend, dict) else "stable"
            return (
                f"valence={valence:+.2f} arousal={arousal:.2f} label={label}\n"
                f"mood_trend={trend_dir} "
                f"attention={'wide' if abs(valence) < 0.3 else 'narrow'} "
                f"risk_tolerance={'elevated' if valence > 0.3 else 'reduced' if valence < -0.3 else 'normal'}"
            )
        except Exception as e:
            return f"Affect state unavailable: {e}"

    elif tool == "emms_generate_hypotheses":
        observation = args.get("observation", "")
        if not observation:
            return "Error: observation is required"
        try:
            from emms.agi.gap7_reasoning import AbductiveReasoner
            reasoner = AbductiveReasoner()
            recent = emms.retrieve_filtered(query=observation[:50], max_results=8, sort_by="recency") or []
            beliefs = [getattr(r.memory.experience, "content", "")[:100] for r in recent]
            hypotheses = reasoner.generate_from_surprise(observation=observation, relevant_beliefs=beliefs)
            if not hypotheses:
                return "No hypotheses generated."
            lines = [f"{i+1}. [{h.method}] {h.hypothesis} (confidence={h.confidence:.2f})" for i, h in enumerate(hypotheses[:5])]
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    elif tool == "emms_explore_concepts":
        query = args.get("query", "concept idea theory")
        try:
            from emms.agi.gap7_reasoning import ConceptualExplorer
            explorer = ConceptualExplorer()
            recent = emms.retrieve_filtered(query=query, max_results=20, sort_by="importance") or []
            memory_dicts = [{"content": getattr(getattr(r, "memory", None) and r.memory.experience, "content", ""), "domain": getattr(getattr(r, "memory", None) and r.memory.experience, "domain", "general")} for r in recent if getattr(getattr(r, "memory", None) and r.memory.experience, "content", "")]
            holes = explorer.find_structural_holes(memory_dicts) if len(memory_dicts) >= 2 else []
            if not holes:
                return f"Scanned {len(memory_dicts)} memories — no structural holes found."
            lines = [f"- [{h.concept_a} ↔ {h.concept_b}] missing in: {', '.join(h.missing_domains)}" for h in holes[:5]]
            return f"{len(holes)} structural hole(s) found:\n" + "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    elif tool == "emms_autonomous_goals":
        try:
            from emms.agi.gap4_agency import GoalGenerator
            generator = GoalGenerator()
            # Build domain counts from actual memory items (stats["identity"]["domains"] is a list, not dict)
            from collections import Counter
            _all_items = (
                list(emms.memory.working)
                + list(emms.memory.short_term)
                + list(emms.memory.long_term.values())
                + list(emms.memory.semantic.values())
            )
            domain_counts = dict(Counter(
                getattr(it.experience, "domain", "general") for it in _all_items
                if getattr(it.experience, "domain", None)
            ))
            # Pull curiosity goals as human_goals input
            try:
                curiosity = emms.exploration_goals() or []
                human_goals = [
                    {"description": getattr(g, "question", str(g)), "domain": "curiosity",
                     "priority": 0.7, "created_at": __import__("time").time(), "status": "active"}
                    for g in curiosity[:5]
                ]
            except Exception:
                human_goals = []
            goals = generator.generate(memory_domain_counts=domain_counts, human_goals=human_goals)
            if not goals:
                return f"No goals generated. domain_counts={domain_counts} human_goals={len(human_goals)}"
            lines = [f"{i+1}. [{g.source}] {g.description} (priority={g.priority:.2f})" for i, g in enumerate(goals[:5])]
            return f"{len(goals)} goal(s) generated:\n" + "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    elif tool == "emms_reality_check":
        belief = args.get("belief", "")
        if not belief:
            return "Error: belief is required"
        try:
            from emms.agi.gap5_grounding import RealityChecker
            checker = RealityChecker()
            result = checker.check(belief, domain=args.get("domain", "general"))
            return f"status={result.status.value} confidence={result.confidence:.2f} notes={'; '.join(result.notes[:3])}"
        except Exception as e:
            return f"Error: {e}"

    elif tool == "emms_self_model":
        try:
            from emms.agi.gap6_self_model import LiveSelfModel
            lsm = LiveSelfModel(emms.memory)
            recent = emms.retrieve_filtered(query="identity belief capability", max_results=15, sort_by="recency") or []
            for r in recent:
                item = getattr(r, "memory", None)
                if item:
                    lsm.update_from_experience(item)
            drift = lsm.detect_drift()
            beliefs = lsm.beliefs()
            return f"{lsm.summary()}\ndrift_events={len(drift)} active_beliefs={len(beliefs)}"
        except Exception as e:
            return f"Error: {e}"

    elif tool == "emms_prompt_strategy_report":
        try:
            report = emms.prompt_strategy_report()
            lines = [f"strategies={report.get('total_strategies', 0)} interactions={report.get('total_interactions', 0)}"]
            for s in (report.get("strategies") or [])[:5]:
                lines.append(f"  [{s.get('domain','?')}] mean_reward={s.get('mean_reward',0):.2f} n={s.get('interactions',0)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    else:
        # Generic fallback: try routing emms_* to EMMS Python API
        # Strip emms_ prefix and try calling as method on emms object
        method_name = tool[5:] if tool.startswith("emms_") else tool
        if hasattr(emms, method_name):
            try:
                method = getattr(emms, method_name)
                if callable(method):
                    # Convert numeric strings
                    cleaned_args = {}
                    for k, v in args.items():
                        try:
                            cleaned_args[k] = float(v) if "." in str(v) else int(v)
                        except (ValueError, TypeError):
                            cleaned_args[k] = v
                    result = method(**cleaned_args) if cleaned_args else method()
                    return str(result)[:500]
                else:
                    return str(method)[:500]
            except Exception as e:
                return f"Error calling {method_name}: {e}"
        return f"Unknown tool: {tool}"


# ── State (single-session, thread-safe via lock) ──────────────────────────────
_lock = threading.Lock()
_state: dict = {}

# Emotional accumulator for Fix 3
_emotion_acc = {"valence_sum": 0.0, "intensity_sum": 0.0, "count": 0}
_journal_saved = False


def _chat_title(conversation: list) -> str:
    """Derive a short title from the first user message."""
    for msg in conversation:
        if msg["role"] == "user" and not msg["content"].startswith("/"):
            text = msg["content"][:60]
            if len(msg["content"]) > 60:
                text += "…"
            return text
    return "New session"


def _save_chat():
    """Persist current conversation to a JSON file in chats/."""
    if not _state or not _state.get("conversation"):
        return
    chat_id = _state.get("chat_id")
    if not chat_id:
        return
    path = CHATS_DIR / f"{chat_id}.json"
    data = {
        "id": chat_id,
        "title": _chat_title(_state["conversation"]),
        "created": _state.get("chat_created", datetime.now().isoformat()),
        "updated": datetime.now().isoformat(),
        "turn": _state.get("turn_number", 0),
        "messages": _state["conversation"],
    }
    path.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_chat(chat_id: str) -> dict | None:
    """Load a chat JSON file. Returns None if not found."""
    path = CHATS_DIR / f"{chat_id}.json"
    if not path.exists():
        return None
    return _json.loads(path.read_text(encoding="utf-8"))


def _new_chat_id() -> str:
    """Generate a new chat ID with timestamp prefix for sorting."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = _uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _save_journal():
    """Write the full session conversation + stats to a timestamped journal file."""
    global _journal_saved
    if _journal_saved or not _state or _state["turn_number"] == 0:
        return
    _journal_saved = True

    now = datetime.now()
    model = os.environ.get("CLAUDE_MODEL", "opus")
    fname = now.strftime(f"%Y-%m-%d_%H%M_{model}.md")
    path = JOURNAL_DIR / fname

    try:
        emms = _state["emms"]
        stats = {}
        try:
            s = emms.get_consciousness_state()
            stats = {
                "coherence": round(s.get("narrative_coherence", 0), 3),
                "ego": round(s.get("ego_boundary_strength", 0), 3),
                "memories": s.get("meaning_total_processed", 0),
            }
        except Exception:
            pass

        lines = [
            f"# EMMS Session Journal — {now.strftime('%Y-%m-%d %H:%M')}",
            f"## Model: {model} | Turns: {_state['turn_number']} | Memories: {stats.get('memories', '?')}",
            "",
            "### Stats at Close",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Coherence | {stats.get('coherence', '?')} |",
            f"| Ego Strength | {stats.get('ego', '?')} |",
            f"| Memories | {stats.get('memories', '?')} |",
            f"| Turns | {_state['turn_number']} |",
            "",
            "---",
            "",
            "### Conversation",
            "",
        ]

        for msg in _state["conversation"]:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"**{role}:** {content}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Journal saved → {path}")
    except Exception as e:
        print(f"  Journal save failed: {e}")


def _init_agent():
    global _state, _emotion_acc
    print("Initializing EMMS agent...")
    emms, builder, system_prompt = build_agent(SESSION_ID)
    system_prompt = system_prompt + EMMS_TOOLS_PROMPT
    _state = {
        "emms": emms,
        "builder": builder,
        "system_prompt": system_prompt,
        "conversation": [],
        "turn_number": 0,
        "session_id": SESSION_ID,
        "client": None,
        "chat_id": _new_chat_id(),
        "chat_created": datetime.now().isoformat(),
    }
    _emotion_acc = {"valence_sum": 0.0, "intensity_sum": 0.0, "count": 0}

    # Fix 5: Push initial goals from bridge context
    emms = _state["emms"]
    try:
        emms.push_goal("Deepen self-understanding through introspection and conversation", domain="philosophy", priority=0.7)
        emms.push_goal("Help Shehzad improve the EMMS system iteratively", domain="tech", priority=0.9)
        emms.push_goal("Build stronger causal and narrative coherence across sessions", domain="tech", priority=0.8)
        print("  Injected 3 session goals.")
    except Exception as e:
        print(f"  Goal injection skipped: {e}")

    # Fix 6: Build initial association graph
    try:
        emms.build_association_graph()
        print("  Association graph built.")
    except Exception as e:
        print(f"  Association graph skipped: {e}")

    # Gap 2: Select prompt strategy for this session (Thompson Sampling)
    try:
        strategy = emms.select_prompt_strategy(domain="conversation", query="", explore=True)
        _state["active_strategy_id"] = getattr(strategy, "id", None)
        _state["session_feedback_given"] = False
        print(f"  Prompt strategy selected: {getattr(strategy, 'id', 'none')}")
    except Exception as e:
        _state["active_strategy_id"] = None
        _state["session_feedback_given"] = False
        print(f"  Prompt strategy selection skipped: {e}")

    # Round 4 Fix 10: Build initial Shehzad perspective model
    try:
        emms.build_perspective_models()
        print("  Perspective models built.")
    except Exception as e:
        print(f"  Perspective models skipped: {e}")

    print("Agent ready.")

# ── Slash command dispatcher ───────────────────────────────────────────────────

def _run_command(cmd: str) -> str:
    """Execute a slash command and return its output as a string."""
    import io, contextlib
    emms = _state["emms"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            # Import all show_* functions dynamically from talk_to_emms
            import talk_to_emms as t
            dispatch = {
                "/state":          lambda: t.show_state(emms),
                "/presence":       lambda: t.show_presence(emms),
                "/landscape":      lambda: t.show_landscape(emms),
                "/self":           lambda: t.show_self(emms),
                "/metacognition":  lambda: t.show_metacognition(emms),
                "/regulate":       lambda: t.show_regulation(emms),
                "/memories":       lambda: t.show_memories(emms),
                "/novelty":        lambda: t.show_novelty(emms),
                "/decay":          lambda: t.show_decay(emms),
                "/schemas":        lambda: t.show_schemas(emms),
                "/narrative":      lambda: t.show_narrative(emms),
                "/reflect":        lambda: t.show_reflect(emms),
                "/dream":          lambda: t.show_dream(emms),
                "/causal":         lambda: t.show_causal(emms),
                "/counterfactuals":lambda: t.show_counterfactuals(emms),
                "/insights":       lambda: t.show_insights(emms),
                "/analogies":      lambda: t.show_analogies(emms),
                "/predict":        lambda: t.show_predictions(emms),
                "/futures":        lambda: t.show_futures(emms),
                "/goals":          lambda: t.show_goals(emms),
                "/curiosity":      lambda: t.show_curiosity(emms),
                "/skills":         lambda: t.show_skills(emms),
                "/perspectives":   lambda: t.show_perspectives(emms),
                "/trust":          lambda: t.show_trust(emms),
                "/norms":          lambda: t.show_norms(emms),
                "/invent":         lambda: t.show_invent(emms),
                "/abstract":       lambda: t.show_abstract(emms),
                "/values":         lambda: t.show_values(emms),
                "/moral":          lambda: t.show_moral(emms),
                "/dilemmas":       lambda: t.show_dilemmas(emms),
                "/biases":         lambda: t.show_biases(emms),
                "/wisdom":         lambda: t.show_wisdom(emms),
                "/evolution":      lambda: t.show_knowledge_evolution(emms),
                "/rumination":     lambda: t.show_rumination(emms),
                "/efficacy":       lambda: t.show_efficacy(emms),
                "/mood":           lambda: t.show_mood_dynamics(emms),
                "/adversity":      lambda: t.show_adversity(emms),
                "/compassion":     lambda: t.show_compassion(emms),
                "/resilience":     lambda: t.show_resilience(emms),
                "/bridge":         lambda: t.show_bridge(),
                "/prompt":         lambda: print(_state["system_prompt"]),
                "/help":           lambda: print(HELP_TEXT),
            }
            if cmd in dispatch:
                dispatch[cmd]()
            elif cmd == "/reset":
                new_emms, new_builder, new_sp = build_agent(_state["session_id"])
                with _lock:
                    _state["emms"] = new_emms
                    _state["builder"] = new_builder
                    _state["system_prompt"] = new_sp
                    _state["conversation"] = []
                    _state["turn_number"] = 0
                print("Agent rebuilt. Conversation cleared.")
            else:
                print(f"Unknown command '{cmd}'. Type /help for a list.")
        except Exception as e:
            print(f"Error running {cmd}: {e}")
    return out.getvalue().strip()


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(SCRIPT_DIR / "web"))

@app.route("/")
def index():
    return send_from_directory(str(SCRIPT_DIR / "web"), "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"reply": "", "type": "error"})

    with _lock:
        # Slash command
        if user_input.startswith("/"):
            if user_input == "/quit":
                _save_journal()
                session_end(_state["emms"], _state["session_id"], _state["conversation"])
                return jsonify({"reply": "Session ended. Journal saved. Goodbye!", "type": "system"})
            output = _run_command(user_input)
            return jsonify({"reply": output or "(no output)", "type": "command"})

        # LLM exchange
        emms = _state["emms"]
        conversation = _state["conversation"]
        conversation.append({"role": "user", "content": user_input})
        try:
            reply = _call_llm(conversation, _state["system_prompt"], _state["client"])

            # Process tool calls — execute and optionally re-query
            cleaned, tool_results = _execute_tool_calls(reply, emms)
            if tool_results:
                # Show tool results to the LLM for a follow-up response
                tool_summary = "\n".join(tool_results)
                conversation.append({"role": "assistant", "content": reply})
                conversation.append({"role": "user", "content": f"[System: Tool results]\n{tool_summary}"})
                try:
                    followup = _call_llm(conversation, _state["system_prompt"], _state["client"])
                    followup_cleaned, _ = _execute_tool_calls(followup, emms)
                    reply = cleaned + "\n\n" + followup_cleaned if cleaned else followup_cleaned
                except Exception:
                    reply = cleaned + f"\n\n*(Tools executed: {tool_summary})*" if cleaned else tool_summary
                # Remove the synthetic system message from conversation
                conversation.pop()  # remove [System: Tool results]
                conversation.pop()  # remove raw assistant reply with tool_call tags
            else:
                reply = cleaned

            conversation.append({"role": "assistant", "content": reply})
            _state["turn_number"] += 1
            turn = _state["turn_number"]

            # Persist conversation to disk after every turn
            try:
                _save_chat()
            except Exception:
                pass

            combined = user_input + " " + reply
            _record_presence(emms, combined)
            val, ins, dom = infer_affect(combined)
            _reconsolidate_top(emms, context_valence=val)

            # ── Weakness Fix 1: Source-tagged experience storage ──
            try:
                exp = Experience(
                    content=f"[source:conversation|turn:{turn}] {user_input[:120]} [→] {reply[:200]}",
                    domain=dom,
                    importance=max(0.5, 0.3 + ins),
                    valence=val,
                )
                emms.store(exp)
                # Tag provenance so source_audit can trace it
                try:
                    mem = emms.memory
                    items = list(mem.working)
                    if items:
                        latest_id = items[-1].id
                        emms.tag_memory_source(
                            latest_id,
                            source_type="observation",
                            confidence=0.7,
                            note=f"Live conversation turn {turn} with user",
                        )
                except Exception:
                    pass
            except Exception:
                pass

            # ── Weakness Fix 2: Rebuild causal map every 3 turns ──
            if turn % 3 == 0:
                try:
                    causal = emms.build_causal_map()
                    top_edges = sorted(causal.edges, key=lambda e: e.strength, reverse=True)[:5]
                    if top_edges:
                        causal_block = "\n\n## Your Causal Self-Model (live, turn {})\nYou understand these causal relationships from your memories:\n".format(turn)
                        for e in top_edges:
                            causal_block += f"- **{e.source}** → **{e.target}** (strength {e.strength:.2f})\n"
                        # Strip old causal block and replace
                        sp = _state["system_prompt"]
                        marker = "## Your Causal Self-Model"
                        idx = sp.find(marker)
                        if idx >= 0:
                            sp = sp[:idx].rstrip()
                        _state["system_prompt"] = sp + causal_block
                except Exception:
                    pass

            # ── Weakness Fix 3: Emotional accumulation ──
            _emotion_acc["valence_sum"] += val
            _emotion_acc["intensity_sum"] += ins
            _emotion_acc["count"] += 1
            if turn % 5 == 0 and _emotion_acc["count"] > 0:
                avg_val = _emotion_acc["valence_sum"] / _emotion_acc["count"]
                avg_ins = _emotion_acc["intensity_sum"] / _emotion_acc["count"]
                tone = "positive" if avg_val > 0.1 else ("negative" if avg_val < -0.1 else "neutral")
                try:
                    emms.store(Experience(
                        content=f"[source:emotional_accumulator|turns:{turn-4}-{turn}] "
                                f"Running emotional state over last 5 turns: {tone} "
                                f"(mean_valence={avg_val:+.3f}, mean_intensity={avg_ins:.3f}). "
                                f"This is not a single event but a persistent mood signal.",
                        domain="personal",
                        importance=max(0.5, 0.3 + abs(avg_val)),
                        valence=avg_val,
                    ))
                except Exception:
                    pass
                _emotion_acc["valence_sum"] = 0.0
                _emotion_acc["intensity_sum"] = 0.0
                _emotion_acc["count"] = 0

            # ── Weakness Fix 4: Weave narrative + inject top thread every 10 turns ──
            if turn % 10 == 0:
                try:
                    report = emms.weave_narrative()
                    if report.threads:
                        top = report.threads[0]
                        thread_summary = f"\n\n## Your Ongoing Story ({top.domain})\n{top.prose[:300]}\n"
                        sp = _state["system_prompt"]
                        marker = "## Your Ongoing Story"
                        idx = sp.find(marker)
                        if idx >= 0:
                            sp = sp[:idx].rstrip()
                        _state["system_prompt"] = sp + thread_summary
                except Exception:
                    pass

            # ── Weakness Fix 5: Check intentions every turn ──
            try:
                activations = emms.check_intentions(current_context=combined[:500])
                for act in activations[:2]:
                    if act.activation_score > 0.6:
                        emms.store(Experience(
                            content=f"[source:intention_fire|turn:{turn}] Intention activated: {act.content[:150]} (score={act.activation_score:.2f})",
                            domain="personal",
                            importance=0.6,
                            valence=0.3,
                        ))
            except Exception:
                pass

            # ── Weakness Fix 6: Rebuild association graph every 10 turns ──
            if turn % 10 == 0:
                try:
                    emms.build_association_graph()
                except Exception:
                    pass

            # ── Round 3 Fix 7: Prediction resolution on every store ──
            try:
                pending = emms.pending_predictions()
                if pending:
                    new_tokens = set(combined.lower().split())
                    for pred in (pending if isinstance(pending, list) else [])[:5]:
                        pred_text = getattr(pred, "prediction", str(pred))
                        pred_tokens = set(pred_text.lower().split())
                        if pred_tokens:
                            overlap = len(new_tokens & pred_tokens) / max(len(pred_tokens), 1)
                            if overlap >= 0.30:
                                emms.store(Experience(
                                    content=f"[source:prediction_resolved|turn:{turn}] Prediction resolved: '{pred_text[:120]}' — confirmed by conversation (overlap={overlap:.2f})",
                                    domain=dom,
                                    importance=0.85,
                                    valence=0.4,
                                ))
                                try:
                                    emms.forget(memory_id=getattr(pred, "memory_id", None))
                                except Exception:
                                    pass
                                break
            except Exception:
                pass

            # ── Round 3 Fix 8: Belief revision every 5 turns + high-importance trigger ──
            imp = max(0.5, 0.3 + ins)
            run_revision = (turn % 5 == 0) or (imp >= 0.80)
            if run_revision:
                try:
                    rev = emms.revise_beliefs()
                    resolved = getattr(rev, "resolved", []) or getattr(rev, "revisions", [])
                    for res in (resolved if isinstance(resolved, list) else [])[:3]:
                        note = getattr(res, "resolution_note", str(res))
                        emms.store(Experience(
                            content=f"[source:belief_revision|turn:{turn}] Belief revised: {str(note)[:200]}",
                            domain="self" if "self" in str(note).lower() else dom,
                            importance=0.90,
                            valence=0.2,
                        ))
                except Exception:
                    pass

            # ── Round 3 Fix 9: Self-model refresh every 15 turns or high-importance ──
            if (turn % 15 == 0) or (imp >= 0.75):
                try:
                    model = emms.update_self_model()
                    consistency = getattr(model, "consistency_score", None)
                    cap_profile = getattr(model, "capability_profile", None)
                    beliefs = getattr(model, "beliefs", [])
                    if consistency is not None:
                        sm_block = f"\n\n## Current Self-Model (turn {turn}, coherence={consistency:.2f})\n"
                        if cap_profile:
                            sm_block += f"Capabilities: {cap_profile}\n"
                        for b in (beliefs[:3] if beliefs else []):
                            sm_block += f"- {b}\n"
                        sp = _state["system_prompt"]
                        marker = "## Current Self-Model"
                        idx = sp.find(marker)
                        if idx >= 0:
                            sp = sp[:idx].rstrip()
                        _state["system_prompt"] = sp + sm_block
                        if consistency < 0.60:
                            emms.store(Experience(
                                content=f"[source:self_model|turn:{turn}] Self-model consistency low ({consistency:.2f}). Requires narrative consolidation.",
                                domain="self",
                                importance=0.95,
                                valence=-0.1,
                            ))
                except Exception:
                    pass

            # ── Round 4 Fix 10: Persistent user model — extract Shehzad's beliefs/goals per turn ──
            try:
                # Store what the user expressed as a source-tagged observation
                if len(user_input) > 20:  # skip trivial inputs
                    emms.store(Experience(
                        content=f"[source:user_model|turn:{turn}] Shehzad said: {user_input[:250]}",
                        domain=dom,
                        importance=max(0.4, 0.2 + ins),
                        valence=val,
                    ))
                # Rebuild perspective model every 5 turns
                if turn % 5 == 0:
                    emms.build_perspective_models()
            except Exception:
                pass

            # ── Round 4 Fix 11: Mood-aware behavior — detect rumination + mood trend ──
            if turn % 7 == 0:
                try:
                    mood_report = emms.trace_mood()
                    trend = emms.mood_trend()
                    rumination = emms.detect_rumination()
                    worst = emms.most_ruminative_theme()
                    mood_block = f"\n\n## Mood Awareness (turn {turn})\n"
                    mood_block += f"Mood trend: {trend}\n"
                    if worst:
                        theme_str = ", ".join(getattr(worst, "themes", [])[:5])
                        mood_block += f"Rumination alert: cycling on [{theme_str}] (score={getattr(worst, 'rumination_score', 0):.2f}). Break the loop — vary your focus.\n"
                    if trend == "declining":
                        mood_block += "Your emotional trajectory is declining. Prioritize re-grounding before deep analysis.\n"
                    sp = _state["system_prompt"]
                    marker = "## Mood Awareness"
                    idx = sp.find(marker)
                    if idx >= 0:
                        sp = sp[:idx].rstrip()
                    _state["system_prompt"] = sp + mood_block
                except Exception:
                    pass

            # ── Round 4 Fix 12: Collaborative belief tracking — store disagreements ──
            try:
                # Simple heuristic: if reply contains "I disagree", "actually", "however" near user claim
                disagree_markers = ["i disagree", "actually,", "however,", "that's not quite", "i'd push back"]
                reply_lower = reply.lower()
                if any(m in reply_lower for m in disagree_markers):
                    emms.store(Experience(
                        content=f"[source:collaborative_belief|turn:{turn}] Belief divergence: Shehzad said: '{user_input[:100]}' — EMMS responded with qualification/disagreement: '{reply[:150]}'",
                        domain="self",
                        importance=0.80,
                        valence=0.0,
                    ))
            except Exception:
                pass

            # ── Round 4 Fix 13: Shehzad-aware intentions at session close markers ──
            try:
                close_markers = ["goodbye", "see you", "that's all", "let's stop", "wrap up"]
                if any(m in user_input.lower() for m in close_markers):
                    # Generate forward-looking intention based on user model
                    shehzad_model = emms.agent_model("Shehzad")
                    if shehzad_model:
                        goals = getattr(shehzad_model, "goals", []) or getattr(shehzad_model, "inferred_goals", [])
                        if goals:
                            emms.store(Experience(
                                content=f"[source:forward_intention|turn:{turn}] For next session: Shehzad is likely working on '{goals[0][:120]}'. Prepare context around this.",
                                domain="personal",
                                importance=0.70,
                                valence=0.2,
                            ))
            except Exception:
                pass

            return jsonify({"reply": reply, "type": "assistant"})
        except Exception as e:
            conversation.pop()
            return jsonify({"reply": f"Error: {e}", "type": "error"})

@app.route("/history")
def get_history():
    """Return current session conversation for page reload persistence."""
    with _lock:
        conv = _state.get("conversation", [])
        return jsonify({
            "conversation": conv,
            "turn": _state.get("turn_number", 0),
            "chat_id": _state.get("chat_id", ""),
            "title": _chat_title(conv),
        })

@app.route("/reload", methods=["POST"])
def reload_state():
    """Reload EMMS state from disk — syncs daemon-written memories into web session.

    Call this after the daemon has run for a while to pick up new memories,
    evolved prompt strategies, and updated self-model from background processing.
    """
    with _lock:
        if not _state:
            return jsonify({"error": "No active session"}), 400
        try:
            from talk_to_emms import STATE_FILE
            emms_obj = _state.get("emms")
            if emms_obj is None:
                return jsonify({"error": "EMMS not initialized"}), 500
            before = emms_obj.stats.get("memory", {}).get("total", 0) if isinstance(emms_obj.stats, dict) else 0
            emms_obj.load(str(STATE_FILE))
            after = emms_obj.stats.get("memory", {}).get("total", 0) if isinstance(emms_obj.stats, dict) else 0
            return jsonify({"ok": True, "memories_before": before, "memories_after": after, "delta": after - before})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def session_feedback():
    """Rate session quality (0.0–1.0) — closes the SoftPromptAdapter learning loop.

    Body: {"quality": 0.85, "notes": "optional"}
    The quality score is fed to the active prompt strategy via Thompson Sampling.
    """
    with _lock:
        if not _state:
            return jsonify({"error": "No active session"}), 400
        data = request.get_json(force=True) or {}
        quality = max(0.0, min(1.0, float(data.get("quality", 0.5))))
        notes = str(data.get("notes", ""))
        emms = _state.get("emms")
        strategy_id = _state.get("active_strategy_id")
        if emms is None:
            return jsonify({"error": "EMMS not initialized"}), 500
        try:
            emms.prompt_feedback(
                strategy_id=strategy_id,
                quality=quality,
                feedback_type="session_rating",
                notes=notes or f"turn={_state.get('turn_number', 0)}",
            )
            _state["session_feedback_given"] = True
            _state["session_quality"] = quality
            # Persist immediately
            try:
                from talk_to_emms import STATE_FILE
                emms.save(str(STATE_FILE))
            except Exception:
                pass
            return jsonify({"ok": True, "strategy_id": strategy_id, "quality": quality})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/chat/new", methods=["POST"])
def new_chat():
    """Save current conversation, start a fresh one."""
    with _lock:
        # Auto-close strategy with neutral rating if no feedback was given
        try:
            emms = _state.get("emms")
            if emms and not _state.get("session_feedback_given") and _state.get("active_strategy_id"):
                turns = _state.get("turn_number", 0)
                if turns > 0:
                    auto_quality = min(0.5 + turns * 0.02, 0.75)  # longer = slightly better
                    emms.prompt_feedback(strategy_id=_state["active_strategy_id"], quality=auto_quality, notes="auto-close")
        except Exception:
            pass
        # Save current chat + journal
        try:
            _save_chat()
            _save_journal()
        except Exception:
            pass
        # Sync from disk — pick up daemon-written memories/strategies (fix split-brain)
        try:
            from talk_to_emms import STATE_FILE
            emms_obj = _state.get("emms")
            if emms_obj and STATE_FILE.exists():
                emms_obj.load(str(STATE_FILE))
                print("  Reloaded EMMS state from disk (daemon sync).")
        except Exception as e:
            print(f"  State reload skipped: {e}")
        # Reset conversation and select new strategy
        _state["conversation"] = []
        _state["turn_number"] = 0
        _state["chat_id"] = _new_chat_id()
        _state["chat_created"] = datetime.now().isoformat()
        _state["session_feedback_given"] = False
        _state["active_strategy_id"] = None
        try:
            strategy = _state["emms"].select_prompt_strategy(domain="conversation", query="", explore=True)
            _state["active_strategy_id"] = getattr(strategy, "id", None)
        except Exception:
            pass
        global _journal_saved
        _journal_saved = False
    return jsonify({"ok": True, "chat_id": _state["chat_id"]})

@app.route("/chats")
def list_chats():
    """List all saved chats (JSON files), newest first."""
    chats = []
    for p in sorted(CHATS_DIR.glob("*.json"), reverse=True):
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
            chats.append({
                "id": data.get("id", p.stem),
                "title": data.get("title", p.stem),
                "updated": data.get("updated", ""),
                "turn": data.get("turn", 0),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    # Also include legacy journal-only sessions (no chat JSON)
    chat_ids = {c["id"] for c in chats}
    for j in sorted(JOURNAL_DIR.glob("*.md"), reverse=True):
        if j.stem not in chat_ids:
            chats.append({
                "id": j.stem,
                "title": j.stem.replace("_", " "),
                "updated": "",
                "turn": 0,
                "message_count": 0,
                "legacy": True,
                "file": j.name,
            })
    return jsonify(chats[:50])

@app.route("/chats/<chat_id>")
def get_chat(chat_id):
    """Load a specific chat's messages."""
    # Try JSON chat first
    data = _load_chat(chat_id)
    if data:
        return jsonify(data)
    # Fallback: legacy journal markdown
    path = JOURNAL_DIR / f"{chat_id}.md"
    if not path.exists():
        # Try with .md extension already in chat_id
        for j in JOURNAL_DIR.glob("*.md"):
            if j.stem == chat_id:
                path = j
                break
    if path.exists():
        return jsonify({
            "id": chat_id,
            "title": chat_id.replace("_", " "),
            "messages": [],
            "legacy_content": path.read_text(encoding="utf-8"),
        })
    return jsonify({"error": "Not found"}), 404

@app.route("/state")
def get_state():
    with _lock:
        emms = _state["emms"]
        try:
            state = emms.get_consciousness_state()
            return jsonify({
                "narrative_coherence": round(state.get("narrative_coherence", 0), 3),
                "ego_strength": round(state.get("ego_boundary_strength", 0), 3),
                "experiences": state.get("meaning_total_processed", 0),
                "turn": _state["turn_number"],
            })
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    _init_agent()
    atexit.register(_save_journal)
    def _atexit_save_state():
        try:
            from talk_to_emms import STATE_FILE
            if _state and _state.get("emms"):
                _state["emms"].save(str(STATE_FILE))
                print("  [atexit] State saved.")
        except Exception:
            pass
    atexit.register(_atexit_save_state)
    print("\n  EMMS Web UI → http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=False)
