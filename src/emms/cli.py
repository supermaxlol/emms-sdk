"""EMMS command-line interface.

Provides a simple CLI for interacting with EMMS memory from the terminal.

Usage::

    emms store "The market rose 2% today" --domain finance
    emms retrieve "market trends" --max 5
    emms search-file "src/emms/emms.py"
    emms stats
    emms save ~/.emms/memory.json
    emms load ~/.emms/memory.json
    emms procedures --domain coding
    emms add-procedure "Always write unit tests for new features." --domain coding

    # v0.6.0 commands
    emms build-rag "auth bug" --budget 4000 --fmt xml
    emms deduplicate --cosine 0.92
    emms srs-enroll mem_abc123
    emms srs-enroll --all
    emms srs-review mem_abc123 4
    emms srs-due --max 20
    emms export-graph --format dot --output graph.dot
    emms export-graph --format d3 --output graph.json

All commands share a ``--memory`` option pointing to the JSON state file.
If ``--memory`` is set, the state is loaded before the command runs and
saved after (for mutating commands).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _get_emms(memory_path: str | None) -> "Any":
    """Build and optionally hydrate an EMMS instance."""
    from emms import EMMS
    agent = EMMS()
    if memory_path and Path(memory_path).expanduser().exists():
        agent.load(memory_path)
    return agent


def cmd_store(args: argparse.Namespace) -> None:
    from emms.core.models import Experience
    agent = _get_emms(args.memory)
    exp = Experience(
        content=args.content,
        domain=args.domain,
        importance=args.importance,
        title=args.title,
    )
    result = agent.store(exp)
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Stored  memory_id={result['memory_id']}  tier={result['tier']}")
    if args.memory:
        agent.save(args.memory)


def cmd_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    results = agent.retrieve(args.query, max_results=args.max)
    if args.json:
        print(json.dumps([
            {
                "id": r.memory.id,
                "content": r.memory.experience.content,
                "score": round(r.score, 4),
                "tier": r.source_tier.value,
                "explanation": r.explanation,
            }
            for r in results
        ]))
    else:
        if not results:
            print("No results found.")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r.score:.3f}  tier={r.source_tier.value}")
            print(f"    {r.memory.experience.content[:120]}")
            if r.explanation:
                print(f"    ↳ {r.explanation}")


def cmd_compact(args: argparse.Namespace) -> None:
    from emms.retrieval.strategies import EnsembleRetriever
    agent = _get_emms(args.memory)
    retriever = EnsembleRetriever.from_balanced(agent.memory)
    results = retriever.search_compact(args.query, max_results=args.max)
    if args.json:
        print(json.dumps([
            {"id": r.id, "snippet": r.snippet, "score": round(r.score, 4), "tokens": r.token_estimate}
            for r in results
        ]))
    else:
        if not results:
            print("No results found.")
        for r in results:
            tok = f"  (~{r.token_estimate} tok)" if r.token_estimate else ""
            print(f"[{r.id}] score={r.score:.3f}{tok}  {r.snippet[:100]}")


def cmd_search_file(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    items = agent.search_by_file(args.file_path)
    if args.json:
        print(json.dumps([
            {"id": i.id, "content": i.experience.content[:120],
             "files_read": i.experience.files_read,
             "files_modified": i.experience.files_modified}
            for i in items
        ]))
    else:
        if not items:
            print(f"No memories reference '{args.file_path}'.")
        for item in items:
            print(f"[{item.id}]  {item.experience.content[:100]}")
            if item.experience.files_read:
                print(f"    read: {', '.join(item.experience.files_read)}")
            if item.experience.files_modified:
                print(f"    modified: {', '.join(item.experience.files_modified)}")


def cmd_stats(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    stats = agent.stats
    if args.json:
        print(json.dumps(stats, default=str))
    else:
        mem = stats["memory"]
        print(f"Memory: working={mem['working']}  short_term={mem['short_term']}  "
              f"long_term={mem['long_term']}  semantic={mem['semantic']}  total={mem['total']}")
        print(f"Stored: {stats['total_stored']}  Retrieved: {stats['total_retrieved']}  "
              f"Uptime: {stats['uptime_seconds']:.0f}s")
        if "graph" in stats:
            g = stats["graph"]
            print(f"Graph: {g['entities']} entities, {g['relationships']} relationships")
        if "consciousness" in stats:
            c = stats["consciousness"]
            print(f"Consciousness: coherence={c['narrative_coherence']:.2f}  "
                  f"ego={c['ego_boundary']:.2f}  themes={c['themes_tracked']}")


def cmd_save(args: argparse.Namespace) -> None:
    agent = _get_emms(None)  # don't pre-load; save will write it
    agent.save(args.path)
    print(f"Saved to {args.path}")


def cmd_load(args: argparse.Namespace) -> None:
    agent = _get_emms(args.path)
    s = agent.memory.size
    print(f"Loaded from {args.path}  ({s['total']} items)")


def cmd_retrieve_filtered(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    results = agent.retrieve_filtered(
        args.query,
        max_results=args.max,
        namespace=args.namespace or None,
        domain=args.domain or None,
        session_id=args.session or None,
        min_confidence=args.min_confidence,
    )
    if args.json:
        print(json.dumps([
            {"id": r.memory.id, "content": r.memory.experience.content,
             "score": round(r.score, 4), "namespace": r.memory.experience.namespace,
             "confidence": r.memory.experience.confidence}
            for r in results
        ]))
    else:
        if not results:
            print("No results.")
        for i, r in enumerate(results, 1):
            ns = r.memory.experience.namespace
            conf = r.memory.experience.confidence
            print(f"[{i}] score={r.score:.3f}  ns={ns}  conf={conf:.2f}  tier={r.source_tier.value}")
            print(f"    {r.memory.experience.content[:120]}")


def cmd_upvote(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    found = agent.upvote(args.memory_id, boost=args.boost)
    if args.json:
        print(json.dumps({"found": found, "memory_id": args.memory_id}))
    else:
        print(f"{'Upvoted' if found else 'Not found'}: {args.memory_id}")
    if args.memory and found:
        agent.save(args.memory)


def cmd_downvote(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    found = agent.downvote(args.memory_id, decay=args.decay)
    if args.json:
        print(json.dumps({"found": found, "memory_id": args.memory_id}))
    else:
        print(f"{'Downvoted' if found else 'Not found'}: {args.memory_id}")
    if args.memory and found:
        agent.save(args.memory)


def cmd_export_md(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    count = agent.export_markdown(args.path, namespace=args.namespace or None)
    print(f"Exported {count} memories to {args.path}")


def cmd_procedures(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    prompt = agent.get_system_prompt_rules(domain=args.domain or None)
    if args.json:
        entries = agent.procedures.get_all(domain=args.domain or None)
        print(json.dumps([e.model_dump() for e in entries], default=str))
    else:
        print(prompt or "(no behavioral rules stored)")


def cmd_add_procedure(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    entry = agent.add_procedure(
        rule=args.rule,
        domain=args.domain,
        importance=args.importance,
    )
    print(f"Added procedure {entry.id}  [domain={entry.domain}  importance={entry.importance}]")
    if args.memory:
        agent.save(args.memory)


def cmd_build_rag(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    context = agent.build_rag_context(
        args.query,
        max_results=args.max,
        token_budget=args.budget,
        fmt=args.fmt,
    )
    print(context)


def cmd_deduplicate(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    result = agent.deduplicate(
        cosine_threshold=args.cosine,
        lexical_threshold=args.lexical,
    )
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Found {result['groups_found']} duplicate groups, archived {result['memories_archived']} memories.")
    if args.memory:
        agent.save(args.memory)


def cmd_srs_enroll(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if args.all:
        count = agent.srs_enroll_all()
        print(f"Enrolled {count} memories in SRS.")
    else:
        success = agent.srs_enroll(args.memory_id)
        print(f"{'Enrolled' if success else 'Not found'}: {args.memory_id}")
    if args.memory:
        agent.save(args.memory)


def cmd_srs_review(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    success = agent.srs_record_review(args.memory_id, quality=args.quality)
    card = agent.srs.get_card(args.memory_id)
    if args.json:
        print(json.dumps({
            "success": success,
            "memory_id": args.memory_id,
            "next_review_days": round(card.interval_days, 1) if card else None,
            "easiness_factor": round(card.easiness_factor, 3) if card else None,
        }))
    else:
        if success and card:
            print(f"Recorded review q={args.quality}  next in {card.interval_days:.1f}d  EF={card.easiness_factor:.2f}")
        else:
            print(f"Memory not found: {args.memory_id}")
    if args.memory:
        agent.save(args.memory)


def cmd_srs_due(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    due_ids = agent.srs_due(max_items=args.max)
    if args.json:
        print(json.dumps({"due": due_ids, "count": len(due_ids)}))
    else:
        if not due_ids:
            print("No items due for review.")
        else:
            print(f"{len(due_ids)} item(s) due for review:")
            for mid in due_ids:
                print(f"  {mid}")


def cmd_export_graph(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if args.format == "dot":
        dot = agent.export_graph_dot(max_nodes=args.max_nodes, min_importance=args.min_importance)
        if args.output:
            Path(args.output).write_text(dot, encoding="utf-8")
            print(f"DOT graph written to {args.output}")
        else:
            print(dot)
    else:
        import json as _json
        graph = agent.export_graph_d3(max_nodes=args.max_nodes, min_importance=args.min_importance)
        out = _json.dumps(graph, indent=2)
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"D3 graph JSON written to {args.output}")
        else:
            print(out)


def cmd_diff(args: argparse.Namespace) -> None:
    from emms.memory.diff import MemoryDiff
    diff = MemoryDiff.from_paths(args.snapshot_a, args.snapshot_b, strength_threshold=args.threshold)
    if args.output:
        diff.export_markdown(args.output)
        print(f"Diff written to {args.output}")
    else:
        if getattr(args, "json", False):
            import json as _json
            print(_json.dumps({
                "added": len(diff.added),
                "removed": len(diff.removed),
                "strengthened": len(diff.strengthened),
                "weakened": len(diff.weakened),
                "superseded": len(diff.superseded),
            }))
        else:
            print(diff.summary())
            if diff.added:
                print(f"\nAdded ({len(diff.added)}):")
                for item in diff.added[:10]:
                    print(f"  + [{item.domain}] {(item.title or item.content)[:60]}")
            if diff.removed:
                print(f"\nRemoved ({len(diff.removed)}):")
                for item in diff.removed[:10]:
                    print(f"  - [{item.domain}] {(item.title or item.content)[:60]}")
            if diff.strengthened:
                print(f"\nStrengthened ({len(diff.strengthened)}):")
                for before, after in diff.strengthened[:5]:
                    print(f"  ^ {before.id[:12]} {before.memory_strength:.3f} → {after.memory_strength:.3f}")
            if diff.weakened:
                print(f"\nWeakened ({len(diff.weakened)}):")
                for before, after in diff.weakened[:5]:
                    print(f"  v {before.id[:12]} {before.memory_strength:.3f} → {after.memory_strength:.3f}")


def cmd_hybrid_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    results = agent.hybrid_retrieve(
        args.query,
        max_results=args.max,
        rrf_k=args.rrf_k,
    )
    if getattr(args, "json", False):
        print(json.dumps([
            {"id": r.memory.id, "score": r.score, "content": r.memory.experience.content}
            for r in results
        ], indent=2))
    else:
        if not results:
            print("No results.")
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r.score:.4f}] {r.memory.experience.domain}: "
                      f"{r.memory.experience.content[:80]}")
                print(f"   {r.explanation}")


def cmd_timeline(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    result = agent.build_timeline(
        domain=args.domain,
        since=args.since,
        until=args.until,
        gap_threshold_seconds=args.gap_threshold,
        bucket_size_seconds=args.bucket_size,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total": result.total_memories,
            "gaps": len(result.gaps),
            "domain_counts": result.domain_counts,
            "summary": result.summary(),
        }))
    elif args.output:
        Path(args.output).write_text(result.export_markdown(), encoding="utf-8")
        print(f"Timeline written to {args.output}")
    else:
        print(result.export_markdown())


def cmd_adaptive_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    agent.enable_adaptive_retrieval(decay=args.decay)
    results = agent.adaptive_retrieve(args.query, max_results=args.max, explore=not args.exploit)
    if getattr(args, "json", False):
        print(json.dumps([
            {"id": r.memory.id, "score": r.score, "strategy": r.strategy,
             "content": r.memory.experience.content}
            for r in results
        ], indent=2))
    else:
        if not results:
            print("No results.")
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r.score:.4f}] {r.strategy}: {r.memory.experience.content[:80]}")


def cmd_retrieval_beliefs(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    agent.enable_adaptive_retrieval()
    beliefs = agent.get_retrieval_beliefs()
    if getattr(args, "json", False):
        print(json.dumps(beliefs, indent=2))
    else:
        if not beliefs:
            print("No adaptive retriever beliefs loaded.")
        else:
            print(f"{'Strategy':<18} {'Mean':>6} {'α':>6} {'β':>6} {'Pulls':>6}")
            print("-" * 48)
            for name, b in sorted(beliefs.items(), key=lambda x: -x[1]["mean"]):
                print(f"{name:<18} {b['mean']:>6.3f} {b['alpha']:>6.2f} {b['beta']:>6.2f} {b['pulls']:>6}")


def cmd_budget_status(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    fp = agent.memory_token_footprint()
    if getattr(args, "json", False):
        print(json.dumps(fp))
    else:
        print(f"Total: {fp['total']:,} tokens across {fp['memory_count']} memories")
        print("By tier:")
        for tier, tokens in sorted(fp["by_tier"].items()):
            print(f"  {tier:<12} {tokens:>8,} tokens")


def cmd_budget_enforce(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.enforce_memory_budget(
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        policy=args.policy,
        importance_threshold=args.importance_threshold,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "over_budget": report.over_budget,
            "evicted": report.evicted_count,
            "freed_tokens": report.freed_tokens,
            "summary": report.summary(),
        }))
    else:
        print(report.summary())
        if report.candidates:
            print(f"\nEviction candidates (top {min(5, len(report.candidates))}):")
            for c in report.candidates[:5]:
                print(f"  [{c.eviction_score:.4f}] {c.domain}: {c.content_excerpt[:60]}")
    if not args.dry_run and not getattr(args, "json", False):
        print("Memory evicted." if report.evicted_count else "Nothing to evict.")


def cmd_multihop(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    result = agent.multihop_query(
        args.seed,
        max_hops=args.max_hops,
        max_results=args.max,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "seed": result.seed,
            "reachable": len(result.reachable),
            "paths": len(result.paths),
            "summary": result.summary(),
        }))
    elif args.dot:
        dot = result.to_dot()
        if args.output:
            Path(args.output).write_text(dot, encoding="utf-8")
            print(f"DOT written to {args.output}")
        else:
            print(dot)
    else:
        print(result.summary())
        if result.reachable:
            print("\nTop reachable entities:")
            for r in result.reachable[:10]:
                path_str = " → ".join(r.best_path.entities)
                print(f"  {r.display_name} ({r.entity_type}, {r.min_hops} hops, "
                      f"str={r.best_path.strength:.3f}): {path_str}")
        if result.bridging_entities:
            print(f"\nBridging hubs:")
            for name, score in result.bridging_entities[:5]:
                print(f"  {name} (score={score:.3f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emms",
        description="EMMS — Enhanced Memory Management System CLI",
    )
    parser.add_argument("--memory", "-m", default=None,
                        help="Path to memory JSON state file (load before / save after).")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    # store
    p_store = sub.add_parser("store", help="Store an experience into memory.")
    p_store.add_argument("content", help="Experience text.")
    p_store.add_argument("--domain", "-d", default="general")
    p_store.add_argument("--importance", "-i", type=float, default=0.5)
    p_store.add_argument("--title", "-t", default=None)
    p_store.set_defaults(func=cmd_store)

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Retrieve memories matching a query.")
    p_ret.add_argument("query")
    p_ret.add_argument("--max", "-n", type=int, default=10)
    p_ret.set_defaults(func=cmd_retrieve)

    # compact
    p_cmp = sub.add_parser("compact", help="Progressive-disclosure compact search.")
    p_cmp.add_argument("query")
    p_cmp.add_argument("--max", "-n", type=int, default=20)
    p_cmp.set_defaults(func=cmd_compact)

    # search-file
    p_sf = sub.add_parser("search-file", help="Find memories referencing a file path.")
    p_sf.add_argument("file_path")
    p_sf.set_defaults(func=cmd_search_file)

    # stats
    p_stats = sub.add_parser("stats", help="Show EMMS system statistics.")
    p_stats.set_defaults(func=cmd_stats)

    # save
    p_save = sub.add_parser("save", help="Persist memory state to disk.")
    p_save.add_argument("path")
    p_save.set_defaults(func=cmd_save)

    # load
    p_load = sub.add_parser("load", help="Load memory state from disk.")
    p_load.add_argument("path")
    p_load.set_defaults(func=cmd_load)

    # retrieve-filtered
    p_rf = sub.add_parser("retrieve-filtered", help="Retrieve with structured filters.")
    p_rf.add_argument("query")
    p_rf.add_argument("--max", "-n", type=int, default=10)
    p_rf.add_argument("--namespace", default=None)
    p_rf.add_argument("--domain", "-d", default=None)
    p_rf.add_argument("--session", default=None)
    p_rf.add_argument("--min-confidence", type=float, default=None, dest="min_confidence")
    p_rf.set_defaults(func=cmd_retrieve_filtered)

    # upvote
    p_up = sub.add_parser("upvote", help="Positive feedback: strengthen a memory.")
    p_up.add_argument("memory_id")
    p_up.add_argument("--boost", type=float, default=0.1)
    p_up.set_defaults(func=cmd_upvote)

    # downvote
    p_down = sub.add_parser("downvote", help="Negative feedback: weaken a memory.")
    p_down.add_argument("memory_id")
    p_down.add_argument("--decay", type=float, default=0.2)
    p_down.set_defaults(func=cmd_downvote)

    # export-md
    p_em = sub.add_parser("export-md", help="Export memories as Markdown.")
    p_em.add_argument("path")
    p_em.add_argument("--namespace", default=None)
    p_em.set_defaults(func=cmd_export_md)

    # procedures
    p_proc = sub.add_parser("procedures", help="Show current behavioral rules.")
    p_proc.add_argument("--domain", "-d", default=None)
    p_proc.set_defaults(func=cmd_procedures)

    # add-procedure
    p_ap = sub.add_parser("add-procedure", help="Add a behavioral rule.")
    p_ap.add_argument("rule")
    p_ap.add_argument("--domain", "-d", default="general")
    p_ap.add_argument("--importance", "-i", type=float, default=0.5)
    p_ap.set_defaults(func=cmd_add_procedure)

    # build-rag
    p_rag = sub.add_parser("build-rag", help="Build a token-budget-aware RAG context document.")
    p_rag.add_argument("query")
    p_rag.add_argument("--max", "-n", type=int, default=20, help="Max memories to retrieve.")
    p_rag.add_argument("--budget", "-b", type=int, default=4000, help="Token budget.")
    p_rag.add_argument("--fmt", "-f", default="markdown", choices=["markdown", "xml", "json", "plain"])
    p_rag.set_defaults(func=cmd_build_rag)

    # deduplicate
    p_dedup = sub.add_parser("deduplicate", help="Scan and archive near-duplicate memories.")
    p_dedup.add_argument("--cosine", type=float, default=None, help="Cosine similarity threshold.")
    p_dedup.add_argument("--lexical", type=float, default=None, help="Lexical similarity threshold.")
    p_dedup.set_defaults(func=cmd_deduplicate)

    # srs-enroll
    p_se = sub.add_parser("srs-enroll", help="Enrol a memory in spaced repetition.")
    p_se.add_argument("memory_id", nargs="?", default=None, help="Memory ID (omit to use --all).")
    p_se.add_argument("--all", action="store_true", help="Enrol all memories.")
    p_se.set_defaults(func=cmd_srs_enroll)

    # srs-review
    p_sr = sub.add_parser("srs-review", help="Record an SRS review outcome.")
    p_sr.add_argument("memory_id")
    p_sr.add_argument("quality", type=int, help="Recall quality 0 (blackout) … 5 (perfect).")
    p_sr.set_defaults(func=cmd_srs_review)

    # srs-due
    p_sd = sub.add_parser("srs-due", help="Show memories due for SRS review.")
    p_sd.add_argument("--max", "-n", type=int, default=50)
    p_sd.set_defaults(func=cmd_srs_due)

    # export-graph
    p_eg = sub.add_parser("export-graph", help="Export the knowledge graph (DOT or D3 JSON).")
    p_eg.add_argument("--format", "-f", default="dot", choices=["dot", "d3"],
                      help="Output format: 'dot' (Graphviz) or 'd3' (D3.js JSON).")
    p_eg.add_argument("--output", "-o", default=None, help="Output file path (default: stdout).")
    p_eg.add_argument("--max-nodes", type=int, default=100)
    p_eg.add_argument("--min-importance", type=float, default=0.0)
    p_eg.set_defaults(func=cmd_export_graph)

    # diff — compare two memory snapshots
    p_diff = sub.add_parser("diff", help="Compare two saved memory snapshot files.")
    p_diff.add_argument("snapshot_a", help="Path to the first (earlier) memory snapshot JSON.")
    p_diff.add_argument("snapshot_b", help="Path to the second (later) memory snapshot JSON.")
    p_diff.add_argument("--output", "-o", default=None, help="Write diff as Markdown to this file.")
    p_diff.add_argument("--threshold", type=float, default=0.05,
                        help="Minimum strength delta to count as strengthened/weakened (default 0.05).")
    p_diff.set_defaults(func=cmd_diff)

    # v0.8.0 commands

    # hybrid-retrieve
    p_hr = sub.add_parser("hybrid-retrieve", help="Hybrid BM25 + embedding retrieval via RRF.")
    p_hr.add_argument("query")
    p_hr.add_argument("--max", "-n", type=int, default=10)
    p_hr.add_argument("--rrf-k", type=float, default=60.0, dest="rrf_k",
                      help="RRF smoothing constant (default 60).")
    p_hr.set_defaults(func=cmd_hybrid_retrieve)

    # timeline
    p_tl = sub.add_parser("timeline", help="Build a chronological memory timeline.")
    p_tl.add_argument("--domain", "-d", default=None)
    p_tl.add_argument("--since", type=float, default=None, help="Unix timestamp lower bound.")
    p_tl.add_argument("--until", type=float, default=None, help="Unix timestamp upper bound.")
    p_tl.add_argument("--gap-threshold", type=float, default=300.0, dest="gap_threshold",
                      help="Min gap seconds to report (default 300).")
    p_tl.add_argument("--bucket-size", type=float, default=3600.0, dest="bucket_size",
                      help="Histogram bucket width seconds (default 3600).")
    p_tl.add_argument("--output", "-o", default=None, help="Write timeline Markdown to file.")
    p_tl.set_defaults(func=cmd_timeline)

    # adaptive-retrieve
    p_ar = sub.add_parser("adaptive-retrieve",
                          help="Retrieve using Thompson Sampling bandit strategy selection.")
    p_ar.add_argument("query")
    p_ar.add_argument("--max", "-n", type=int, default=10)
    p_ar.add_argument("--exploit", action="store_true",
                      help="Exploit best known strategy instead of sampling.")
    p_ar.add_argument("--decay", type=float, default=1.0,
                      help="Geometric discount for belief updates (default 1.0 = no decay).")
    p_ar.set_defaults(func=cmd_adaptive_retrieve)

    # retrieval-beliefs
    p_rb = sub.add_parser("retrieval-beliefs",
                          help="Show current Thompson Sampling beliefs for each retrieval strategy.")
    p_rb.set_defaults(func=cmd_retrieval_beliefs)

    # budget-status
    p_bs = sub.add_parser("budget-status", help="Show token footprint across memory tiers.")
    p_bs.set_defaults(func=cmd_budget_status)

    # budget-enforce
    p_be = sub.add_parser("budget-enforce", help="Enforce a memory token budget (evict low-value items).")
    p_be.add_argument("--max-tokens", type=int, default=100_000, dest="max_tokens",
                      help="Token ceiling (default 100000).")
    p_be.add_argument("--dry-run", action="store_true", dest="dry_run",
                      help="Preview evictions without deleting anything.")
    p_be.add_argument("--policy", default="composite",
                      choices=["composite", "lru", "lfu", "importance", "strength"])
    p_be.add_argument("--importance-threshold", type=float, default=0.8,
                      dest="importance_threshold",
                      help="Protect memories with importance >= this value (default 0.8).")
    p_be.set_defaults(func=cmd_budget_enforce)

    # multihop
    p_mh = sub.add_parser("multihop", help="Multi-hop BFS graph reasoning from a seed entity.")
    p_mh.add_argument("seed", help="Seed entity name.")
    p_mh.add_argument("--max-hops", type=int, default=3, dest="max_hops")
    p_mh.add_argument("--max", "-n", type=int, default=20)
    p_mh.add_argument("--dot", action="store_true", help="Output Graphviz DOT format.")
    p_mh.add_argument("--output", "-o", default=None, help="Write DOT to file.")
    p_mh.set_defaults(func=cmd_multihop)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Propagate --json to subcommand namespace (it's on the root parser)
    if not hasattr(args, "json"):
        args.json = False
    args.func(args)


if __name__ == "__main__":
    main()
