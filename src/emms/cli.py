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


def cmd_index_lookup(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if args.action == "stats":
        stats = agent.index_stats()
        if getattr(args, "json", False):
            print(json.dumps(stats))
        else:
            for k, v in stats.items():
                print(f"  {k}: {v}")
    elif args.action == "rebuild":
        count = agent.rebuild_index()
        print(f"Index rebuilt: {count} items registered.")
    else:
        # lookup
        item = None
        if args.memory_id:
            item = agent.get_memory_by_id(args.memory_id)
        elif args.experience_id:
            item = agent.get_memory_by_experience_id(args.experience_id)
        elif args.content:
            items = agent.find_memories_by_content(args.content)
            if getattr(args, "json", False):
                print(json.dumps([
                    {"id": it.id, "content": it.content[:120], "tier": it.tier.value}
                    for it in items
                ]))
            else:
                print(f"Found {len(items)} item(s) with matching content hash:")
                for it in items:
                    print(f"  [{it.tier.value}] {it.id}: {it.content[:80]}")
            return
        if item is None:
            print("Not found.")
        else:
            if getattr(args, "json", False):
                print(json.dumps({"id": item.id, "content": item.content[:120],
                                  "tier": item.tier.value, "importance": item.importance}))
            else:
                print(f"Found: [{item.tier.value}] {item.id}")
                print(f"  Content: {item.content[:120]}")
                print(f"  Importance: {item.importance:.3f}  Strength: {item.memory_strength:.3f}")


def cmd_index_stats(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    stats = agent.index_stats()
    if getattr(args, "json", False):
        print(json.dumps(stats))
    else:
        print("CompactionIndex stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


def cmd_graph_communities(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    result = agent.graph_communities(
        max_iter=args.max_iter,
        min_community_size=args.min_size,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "num_communities": result.num_communities,
            "modularity": result.modularity,
            "total_entities": result.total_entities,
            "bridge_entities": result.bridge_entities,
        }))
    elif args.markdown:
        md = result.export_markdown()
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Markdown written to {args.output}")
        else:
            print(md)
    else:
        print(result.summary())
        print(f"\nTop communities:")
        for c in sorted(result.communities, key=lambda x: -x.size)[:5]:
            print(f"  Community {c.community_id} ({c.size} entities, "
                  f"Q-contribution: internal_str={c.total_internal_strength:.3f})")
            print(f"    Members: {', '.join(c.entities[:8])}")


def cmd_replay(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    batch = agent.replay_sample(k=args.k, beta=args.beta)
    if getattr(args, "json", False):
        print(json.dumps({
            "count": batch.batch_size,
            "mean_priority": round(batch.mean_priority, 4),
            "entries": [
                {"id": e.item.id, "content": e.item.content[:100],
                 "priority": round(e.priority, 4), "weight": round(e.weight, 4)}
                for e in batch.entries
            ],
        }))
    else:
        print(f"Sampled {batch.batch_size} items from {batch.total_items_considered} eligible "
              f"(mean priority={batch.mean_priority:.4f}):")
        for e in batch.entries:
            print(f"  prio={e.priority:.4f} w={e.weight:.4f}  {e.item.content[:80]}")


def cmd_replay_top(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    entries = agent.replay_top(k=args.k)
    if getattr(args, "json", False):
        print(json.dumps([
            {"id": e.item.id, "content": e.item.content[:100], "priority": round(e.priority, 4)}
            for e in entries
        ]))
    else:
        print(f"Top-{len(entries)} by priority:")
        for e in entries:
            print(f"  {e.priority:.4f}  {e.item.content[:80]}")


def cmd_merge_from(args: argparse.Namespace) -> None:
    # Merge from a second memory file
    agent = _get_emms(args.memory)
    if not args.source:
        print("Error: --source/-s required (path to source memory JSON).")
        return
    source_agent = _get_emms(args.source)
    result = agent.merge_from(
        source_agent,
        policy=args.policy,
        namespace_prefix=args.namespace,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "items_merged": result.items_merged,
            "items_skipped_duplicate": result.items_skipped_duplicate,
            "items_skipped_conflict": result.items_skipped_conflict_lost,
            "conflicts": len(result.conflicts),
        }))
    else:
        print(result.summary())
    if args.memory:
        agent.save(args.memory)
        print(f"Saved merged memory to {args.memory}")


def cmd_plan_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    plan = agent.plan_retrieve(
        args.query,
        max_results=args.max,
        max_results_per_sub=args.max_sub,
        cross_boost=args.boost,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "sub_queries": plan.sub_queries,
            "total_unique_results": plan.total_unique_results,
            "cross_boost_count": plan.cross_boost_count,
            "results": [
                {"id": r.memory.id, "score": round(r.score, 4),
                 "content": r.memory.content[:100], "strategy": r.strategy}
                for r in plan.merged_results[:args.max]
            ],
        }))
    else:
        print(plan.summary())
        print(f"\nSub-queries: {plan.sub_queries}")
        if plan.merged_results:
            print(f"\nTop results:")
            for r in plan.merged_results[:10]:
                print(f"  [{r.score:.4f}] ({r.strategy}) {r.memory.content[:80]}")
        else:
            print("No results found.")


def cmd_reconsolidate(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    ctx_v = getattr(args, "context_valence", None)
    reinforce = not getattr(args, "weaken", False)
    result = agent.reconsolidate(
        memory_id=args.memory_id,
        context_valence=ctx_v,
        reinforce=reinforce,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "memory_id": result.memory_id,
            "type": result.reconsolidation_type,
            "old_strength": round(result.old_strength, 4),
            "new_strength": round(result.new_strength, 4),
            "delta_strength": round(result.delta_strength, 4),
            "old_valence": round(result.old_valence, 4),
            "new_valence": round(result.new_valence, 4),
        }))
    else:
        print(result.summary())


def cmd_decay_unrecalled(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.decay_unrecalled(
        decay_factor=getattr(args, "decay_factor", 0.02),
        min_age_seconds=getattr(args, "min_age", 3600.0),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_items": report.total_items,
            "weakened": report.weakened,
            "unchanged": report.unchanged,
            "mean_delta_strength": round(report.mean_delta_strength, 4),
        }))
    else:
        print(report.summary())
    if args.memory:
        agent.save(args.memory)


def cmd_presence(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if not hasattr(agent, "_presence_tracker") or agent._presence_tracker is None:
        agent.enable_presence_tracking()
    if getattr(args, "content", None):
        metrics = agent.record_presence_turn(
            content=args.content,
            domain=getattr(args, "domain", "general"),
            valence=float(getattr(args, "valence", 0.0)),
            intensity=float(getattr(args, "intensity", 0.0)),
        )
    else:
        metrics = agent.presence_metrics()
    if getattr(args, "json", False):
        print(json.dumps({
            "session_id": metrics.session_id,
            "turn_count": metrics.turn_count,
            "presence_score": round(metrics.presence_score, 4),
            "attention_budget_remaining": round(metrics.attention_budget_remaining, 4),
            "coherence_trend": metrics.coherence_trend,
            "is_degrading": metrics.is_degrading,
            "mean_valence": round(metrics.mean_valence, 4),
            "mean_intensity": round(metrics.mean_intensity, 4),
        }))
    else:
        print(metrics.summary())


def cmd_presence_arc(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if not hasattr(agent, "_presence_tracker") or agent._presence_tracker is None:
        print("No active presence tracker. Run 'presence' first.")
        return
    arc = agent._presence_tracker.emotional_arc()
    if getattr(args, "json", False):
        print(json.dumps({"arc": arc, "length": len(arc)}))
    else:
        if not arc:
            print("No turns recorded yet.")
        else:
            print(f"Emotional arc ({len(arc)} turns):")
            for i, v in enumerate(arc):
                bar = "█" * int((v + 1.0) * 10) + "░" * int((1.0 - v) * 10)
                print(f"  {i:3d}  {bar}  {v:+.2f}")


def cmd_affective_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    results = agent.affective_retrieve(
        query=getattr(args, "query", "") or "",
        target_valence=getattr(args, "target_valence", None),
        target_intensity=getattr(args, "target_intensity", None),
        max_results=getattr(args, "max", 10),
        semantic_blend=getattr(args, "semantic_blend", 0.4),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "count": len(results),
            "results": [
                {
                    "id": r.memory.id,
                    "score": round(r.score, 4),
                    "proximity": round(r.emotional_proximity, 4),
                    "valence": round(r.memory.experience.emotional_valence, 3),
                    "intensity": round(r.memory.experience.emotional_intensity, 3),
                    "content": r.memory.experience.content[:100],
                }
                for r in results
            ],
        }))
    else:
        print(f"Affective results ({len(results)}):")
        for r in results:
            v = r.memory.experience.emotional_valence
            i = r.memory.experience.emotional_intensity
            print(f"  [{r.score:.4f}] prox={r.emotional_proximity:.4f} "
                  f"v={v:+.2f} i={i:.2f}  {r.memory.experience.content[:80]}")


def cmd_dream(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.dream(
        session_id=getattr(args, "session_id", None),
        reinforce_top_k=getattr(args, "reinforce_top_k", 20),
        weaken_bottom_k=getattr(args, "weaken_bottom_k", 10),
        prune_threshold=getattr(args, "prune_threshold", 0.05),
        run_dedup=not getattr(args, "no_dedup", False),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "session_id": report.session_id,
            "total": report.total_memories_processed,
            "reinforced": report.reinforced,
            "weakened": report.weakened,
            "pruned": report.pruned,
            "deduped_pairs": report.deduped_pairs,
            "patterns_found": report.patterns_found,
            "insights": report.insights,
        }))
    else:
        print(report.summary())
    if args.memory:
        agent.save(args.memory)


def cmd_capture_bridge(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    record = agent.capture_session_bridge(
        session_id=getattr(args, "session_id", None),
        closing_summary=getattr(args, "summary", "") or "",
        max_threads=getattr(args, "max_threads", 5),
    )
    output_path = getattr(args, "output", None)
    if output_path:
        from emms.sessions.bridge import SessionBridge
        bridge = SessionBridge(memory=agent.memory)
        bridge.save(output_path, record)
        print(f"Bridge saved to {output_path}")
    if getattr(args, "json", False):
        print(json.dumps(record.to_dict()))
    else:
        print(record.summary())


def cmd_inject_bridge(args: argparse.Namespace) -> None:
    from emms.sessions.bridge import SessionBridge
    bridge_path = getattr(args, "bridge_file", None)
    if not bridge_path:
        print("Error: --bridge-file required.")
        return
    record = SessionBridge.load(bridge_path)
    if record is None:
        print(f"Error: Could not load bridge from {bridge_path}")
        return
    agent = _get_emms(args.memory)
    injection = agent.inject_session_bridge(
        record,
        new_session_id=getattr(args, "session_id", None),
    )
    if getattr(args, "json", False):
        print(json.dumps({"injection": injection, "open_threads": len(record.open_threads)}))
    else:
        print(injection)


def cmd_anneal(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    last_at = getattr(args, "last_session_at", None)
    result = agent.anneal(
        last_session_at=float(last_at) if last_at else None,
        half_life_gap=getattr(args, "half_life_gap", 259200.0),
        decay_rate=getattr(args, "decay_rate", 0.03),
        emotional_stabilization_rate=getattr(args, "emotional_stabilization_rate", 0.08),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_items": result.total_items,
            "gap_hours": round(result.gap_seconds / 3600, 2),
            "effective_temperature": round(result.effective_temperature, 4),
            "accelerated_decay": result.accelerated_decay,
            "emotionally_stabilized": result.emotionally_stabilized,
            "strengthened": result.strengthened,
        }))
    else:
        print(result.summary())
    if args.memory:
        agent.save(args.memory)


def cmd_bridge_summary(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    record = agent.capture_session_bridge()
    if getattr(args, "json", False):
        print(json.dumps({
            "open_threads": len(record.open_threads),
            "threads": [
                {"domain": t.domain, "importance": round(t.importance, 3),
                 "reason": t.reason, "excerpt": t.content_excerpt[:80]}
                for t in record.open_threads
            ],
            "mean_valence": round(record.mean_valence_at_end, 4),
            "dominant_domains": record.dominant_domains,
        }))
    else:
        print(record.summary())
        if record.open_threads:
            print(f"\nOpen threads ({len(record.open_threads)}):")
            for i, t in enumerate(record.open_threads, 1):
                print(f"  {i}. [{t.domain}] imp={t.importance:.2f}  {t.content_excerpt[:70]}")


def cmd_emotional_landscape(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    landscape = agent.emotional_landscape()
    if getattr(args, "json", False):
        print(json.dumps({
            "total_memories": landscape.total_memories,
            "mean_valence": round(landscape.mean_valence, 4),
            "mean_intensity": round(landscape.mean_intensity, 4),
            "valence_std": round(landscape.valence_std, 4),
            "intensity_std": round(landscape.intensity_std, 4),
            "valence_histogram": landscape.valence_histogram,
            "intensity_histogram": landscape.intensity_histogram,
            "most_positive": landscape.most_positive,
            "most_negative": landscape.most_negative,
            "most_intense": landscape.most_intense,
        }))
    else:
        print(landscape.summary())


# ---------------------------------------------------------------------------
# v0.12.0 commands
# ---------------------------------------------------------------------------

def cmd_metacognition(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.metacognition_report(
        max_contradictions=args.max_contradictions,
        confidence_threshold_high=args.threshold_high,
        confidence_threshold_low=args.threshold_low,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_memories": report.total_memories,
            "mean_confidence": round(report.mean_confidence, 4),
            "high_confidence_count": report.high_confidence_count,
            "low_confidence_count": report.low_confidence_count,
            "knowledge_gaps": report.knowledge_gaps,
            "recommendations": report.recommendations,
            "contradictions": len(report.contradictions),
        }))
    else:
        print(report.summary())


def cmd_knowledge_map(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    profiles = agent.knowledge_map()
    if getattr(args, "json", False):
        print(json.dumps([
            {"domain": p.domain, "memory_count": p.memory_count,
             "mean_confidence": round(p.mean_confidence, 3),
             "coverage_score": round(p.coverage_score, 3),
             "mean_importance": round(p.mean_importance, 3)}
            for p in profiles
        ]))
    else:
        if not profiles:
            print("No memories stored.")
            return
        print(f"Knowledge Map ({len(profiles)} domains):\n")
        for p in profiles:
            bar = "█" * int(p.mean_confidence * 10) + "░" * (10 - int(p.mean_confidence * 10))
            print(f"  [{p.domain:12s}] n={p.memory_count:3d}  conf={p.mean_confidence:.2f} {bar}  imp={p.mean_importance:.2f}")


def cmd_contradictions(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    pairs = agent.find_contradictions(max_pairs=args.max_pairs)
    if getattr(args, "json", False):
        print(json.dumps([
            {"memory_a_id": p.memory_a_id, "memory_b_id": p.memory_b_id,
             "semantic_overlap": round(p.semantic_overlap, 4),
             "valence_conflict": round(p.valence_conflict, 4),
             "contradiction_score": round(p.contradiction_score, 4),
             "excerpt_a": p.excerpt_a[:80], "excerpt_b": p.excerpt_b[:80]}
            for p in pairs
        ]))
    else:
        if not pairs:
            print("No contradictions detected.")
            return
        print(f"Contradictions ({len(pairs)}):\n")
        for i, p in enumerate(pairs, 1):
            print(f"  {i}. score={p.contradiction_score:.3f}  "
                  f"overlap={p.semantic_overlap:.2f}  "
                  f"valence_conflict={p.valence_conflict:.2f}")
            print(f"     A: {p.excerpt_a[:70]}")
            print(f"     B: {p.excerpt_b[:70]}")


def cmd_intend(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    intention = agent.intend(
        content=args.content,
        trigger_context=args.trigger,
        priority=args.priority,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "intention_id": intention.id,
            "content": intention.content,
            "trigger_context": intention.trigger_context,
            "priority": intention.priority,
        }))
    else:
        print(f"Intention stored: {intention.id}")
        print(f"  Content:  {intention.content}")
        print(f"  Trigger:  {intention.trigger_context}")
        print(f"  Priority: {intention.priority:.2f}")


def cmd_check_intentions(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    context = args.context or ""
    activations = agent.check_intentions(context)
    pending = agent.pending_intentions()
    if getattr(args, "json", False):
        print(json.dumps({
            "activated": [
                {"intention_id": a.intention.id, "content": a.intention.content,
                 "activation_score": round(a.activation_score, 4),
                 "days_pending": round(a.days_pending, 2)}
                for a in activations
            ],
            "total_pending": len(pending),
        }))
    else:
        print(f"Pending intentions: {len(pending)}")
        if activations:
            print(f"\nActivated by context ({len(activations)}):")
            for a in activations:
                print(f"  score={a.activation_score:.3f}  [{a.intention.id}] {a.intention.content[:70]}")
        else:
            print("\nNo intentions activated by current context.")
        if pending and not activations:
            print("\nAll pending:")
            for i in pending[:5]:
                print(f"  [{i.id}] pri={i.priority:.2f}  {i.content[:70]}")


def cmd_association_graph(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    stats = agent.build_association_graph(
        semantic_threshold=args.semantic_threshold,
        temporal_window=args.temporal_window,
        affective_tolerance=args.affective_tolerance,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "mean_degree": round(stats.mean_degree, 3),
            "mean_edge_weight": round(stats.mean_edge_weight, 4),
            "most_connected_id": stats.most_connected_id,
            "edge_type_counts": stats.edge_type_counts,
        }))
    else:
        print(stats.summary())


def cmd_activation(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    seed_ids = args.seed_ids
    results = agent.spreading_activation(
        seed_ids=seed_ids,
        decay=args.decay,
        steps=args.steps,
    )
    if getattr(args, "json", False):
        print(json.dumps([
            {"memory_id": r.memory_id, "activation": round(r.activation, 4),
             "steps": r.steps_from_seed, "path": r.path}
            for r in results
        ]))
    else:
        print(f"Spreading activation from {len(seed_ids)} seed(s): {seed_ids}")
        print(f"Activated {len(results)} memories:\n")
        for r in results[:15]:
            print(f"  act={r.activation:.3f}  steps={r.steps_from_seed}  id={r.memory_id}")


def cmd_discover_insights(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.discover_insights(
        session_id=args.session_id,
        max_insights=args.max_insights,
        min_bridge_weight=args.min_bridge_weight,
        rebuild_graph=not args.no_rebuild,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "bridges_found": report.bridges_found,
            "insights_generated": report.insights_generated,
            "new_memory_ids": report.new_memory_ids,
            "duration_ms": round(report.duration_seconds * 1000, 1),
            "bridges": [
                {"domain_a": b.domain_a, "domain_b": b.domain_b,
                 "bridge_weight": round(b.bridge_weight, 4),
                 "insight": b.insight_content[:120]}
                for b in report.bridges
            ],
        }))
    else:
        print(report.summary())


def cmd_associative_retrieve(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if args.seed_ids:
        results = agent.associative_retrieve(
            seed_ids=args.seed_ids,
            max_results=args.max,
            steps=args.steps,
            decay=args.decay,
        )
    else:
        results = agent.associative_retrieve_by_query(
            query=args.query or "",
            seed_count=args.seed_count,
            max_results=args.max,
            steps=args.steps,
            decay=args.decay,
        )
    if getattr(args, "json", False):
        print(json.dumps([
            {"memory_id": r.memory.id, "activation_score": round(r.activation_score, 4),
             "steps": r.steps_from_seed, "domain": r.memory.experience.domain,
             "content": r.memory.experience.content[:100]}
            for r in results
        ]))
    else:
        print(f"Associative retrieval — {len(results)} result(s):\n")
        for i, r in enumerate(results, 1):
            dom = r.memory.experience.domain or "general"
            content = r.memory.experience.content
            if len(content) > 90:
                content = content[:90] + "..."
            print(f"  {i:2d}. act={r.activation_score:.3f}  steps={r.steps_from_seed}"
                  f"  [{dom}] {content}")


def cmd_association_stats(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    stats = agent.association_stats()
    if getattr(args, "json", False):
        print(json.dumps({
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "mean_degree": round(stats.mean_degree, 3),
            "mean_edge_weight": round(stats.mean_edge_weight, 4),
            "most_connected_id": stats.most_connected_id,
            "edge_type_counts": stats.edge_type_counts,
        }))
    else:
        print(stats.summary())


# ---------------------------------------------------------------------------
# v0.14.0 command handlers
# ---------------------------------------------------------------------------

def cmd_open_episode(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    ep = agent.open_episode(session_id=args.session_id, topic=args.topic)
    if getattr(args, "json", False):
        print(json.dumps({
            "episode_id": ep.id,
            "session_id": ep.session_id,
            "topic": ep.topic,
            "opened_at": ep.opened_at,
        }))
    else:
        print(f"Opened episode [{ep.id}]")
        if ep.topic:
            print(f"  Topic:      {ep.topic}")
        print(f"  Session:    {ep.session_id}")


def cmd_close_episode(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    ep = agent.close_episode(outcome=args.outcome)
    if ep is None:
        print("No open episode to close.")
        return
    if getattr(args, "json", False):
        print(json.dumps({
            "episode_id": ep.id,
            "outcome": ep.outcome,
            "turn_count": ep.turn_count,
            "duration_seconds": ep.duration_seconds,
            "mean_valence": round(ep.mean_valence, 4),
            "peak_valence": round(ep.peak_valence, 4),
        }))
    else:
        print(ep.summary())
        if ep.outcome:
            print(f"  Outcome: {ep.outcome}")


def cmd_recent_episodes(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    episodes = agent.recent_episodes(n=args.n)
    if not episodes:
        print("No episodes recorded yet.")
        return
    if getattr(args, "json", False):
        print(json.dumps([
            {
                "episode_id": ep.id, "topic": ep.topic,
                "turn_count": ep.turn_count, "is_open": ep.is_open,
                "duration_seconds": ep.duration_seconds,
                "mean_valence": round(ep.mean_valence, 4),
                "outcome": ep.outcome,
            }
            for ep in episodes
        ]))
    else:
        print(f"Recent episodes ({len(episodes)}):\n")
        for ep in episodes:
            print(f"  {ep.summary()}")


def cmd_extract_schemas(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.extract_schemas(
        domain=args.domain,
        max_schemas=args.max_schemas,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_memories_analyzed": report.total_memories_analyzed,
            "schemas_found": report.schemas_found,
            "schemas": [
                {
                    "schema_id": s.id, "domain": s.domain,
                    "pattern": s.pattern, "keywords": s.keywords,
                    "confidence": round(s.confidence, 4),
                    "support": len(s.supporting_memory_ids),
                }
                for s in report.schemas
            ],
        }))
    else:
        print(report.summary())
        for s in report.schemas:
            print()
            print(s.summary())


def cmd_forget(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    if args.memory_id:
        result = agent.forget_memory(args.memory_id)
        if result is None:
            print(f"Memory {args.memory_id!r} not found.")
            return
        if getattr(args, "json", False):
            print(json.dumps({
                "memory_id": result.memory_id, "pruned": result.pruned,
                "old_strength": round(result.old_strength, 4),
                "new_strength": round(result.new_strength, 4),
            }))
        else:
            print(result.summary())
    elif args.domain:
        report = agent.forget_domain(args.domain, rate=args.rate)
        if getattr(args, "json", False):
            print(json.dumps({
                "total_targeted": report.total_targeted,
                "suppressed": report.suppressed, "pruned": report.pruned,
            }))
        else:
            print(report.summary())
    elif args.below_confidence is not None:
        report = agent.forget_below_confidence(threshold=args.below_confidence)
        if getattr(args, "json", False):
            print(json.dumps({
                "total_targeted": report.total_targeted,
                "suppressed": report.suppressed, "pruned": report.pruned,
            }))
        else:
            print(report.summary())
    else:
        print("Provide --memory-id, --domain, or --below-confidence.")


# ---------------------------------------------------------------------------
# v0.15.0 command handlers
# ---------------------------------------------------------------------------

def cmd_reflect(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.reflect(
        domain=args.domain,
        lookback_episodes=args.lookback_episodes,
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "session_id": report.session_id,
            "memories_reviewed": report.memories_reviewed,
            "episodes_reviewed": report.episodes_reviewed,
            "lessons_count": len(report.lessons),
            "new_memory_ids": report.new_memory_ids,
            "open_questions": report.open_questions,
            "lessons": [
                {"lesson_id": l.id, "type": l.lesson_type, "domain": l.domain,
                 "confidence": round(l.confidence, 4), "content": l.content}
                for l in report.lessons
            ],
        }))
    else:
        print(report.summary())
        if report.lessons:
            print()
            for l in report.lessons:
                print(l.summary())


def cmd_weave_narrative(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.weave_narrative(domain=args.domain, max_threads=args.max_threads)
    if getattr(args, "json", False):
        print(json.dumps({
            "total_threads": report.total_threads,
            "total_segments": report.total_segments,
            "threads": [
                {"thread_id": t.id, "theme": t.theme, "domain": t.domain,
                 "segments": len(t.segments), "span_seconds": round(t.span_seconds, 1),
                 "story": t.story()}
                for t in report.threads
            ],
        }))
    else:
        print(report.summary())
        for t in report.threads:
            print()
            print(t.summary())
            print(f"  Story: {t.story()[:200]}")


def cmd_narrative_threads(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    threads = agent.narrative_threads(domain=args.domain)
    if not threads:
        print("No narrative threads found.")
        return
    if getattr(args, "json", False):
        print(json.dumps([
            {"thread_id": t.id, "theme": t.theme, "domain": t.domain,
             "segments": len(t.segments), "story": t.story()}
            for t in threads
        ]))
    else:
        print(f"Narrative threads ({len(threads)}):\n")
        for t in threads:
            print(t.summary())
            print(f"  Story excerpt: {t.story()[:150]}")
            print()


def cmd_source_audit(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.source_audit(flag_threshold=args.flag_threshold)
    if getattr(args, "json", False):
        print(json.dumps({
            "total_audited": report.total_audited,
            "flagged_count": report.flagged_count,
            "source_distribution": report.source_distribution,
            "high_risk": [
                {"memory_id": e.memory_id, "source_type": e.source_type,
                 "confidence": round(e.source_confidence, 4), "flag_reason": e.flag_reason}
                for e in report.high_risk_entries[:args.max_flagged]
            ],
        }))
    else:
        print(report.summary())


def cmd_tag_source(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    tag = agent.tag_memory_source(
        memory_id=args.memory_id,
        source_type=args.source_type,
        confidence=args.confidence,
        note=args.note or "",
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "memory_id": tag.memory_id,
            "source_type": tag.source_type,
            "confidence": round(tag.confidence, 4),
        }))
    else:
        print(tag.summary())


# v0.16.0 commands


def cmd_curiosity_report(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.curiosity_scan(domain=getattr(args, "domain", None) or None)
    if getattr(args, "json", False):
        print(json.dumps({
            "total_domains_scanned": report.total_domains_scanned,
            "goals_generated": report.goals_generated,
            "top_curious_domains": report.top_curious_domains,
            "goals": [
                {"id": g.id, "question": g.question, "domain": g.domain,
                 "urgency": g.urgency, "gap_type": g.gap_type}
                for g in report.goals
            ],
        }))
    else:
        print(report.summary())


def cmd_explore_goals(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    goals = agent.exploration_goals()
    if getattr(args, "json", False):
        print(json.dumps([
            {"id": g.id, "question": g.question, "domain": g.domain,
             "urgency": g.urgency, "gap_type": g.gap_type}
            for g in goals
        ]))
    else:
        if not goals:
            print("No pending exploration goals.")
        for g in goals:
            print(g.summary())


def cmd_revise_beliefs(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.revise_beliefs(
        new_memory_id=getattr(args, "memory_id", None) or None,
        domain=getattr(args, "domain", None) or None,
        max_revisions=getattr(args, "max_revisions", 8),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_checked": report.total_checked,
            "conflicts_found": report.conflicts_found,
            "revisions_made": report.revisions_made,
            "records": [
                {"id": r.id, "revision_type": r.revision_type,
                 "conflict_score": r.conflict_score}
                for r in report.records
            ],
        }))
    else:
        print(report.summary())


def cmd_decay_report(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.memory_decay_report(domain=getattr(args, "domain", None) or None)
    if getattr(args, "json", False):
        print(json.dumps({
            "total_processed": report.total_processed,
            "decayed": report.decayed,
            "mean_retention": report.mean_retention,
        }))
    else:
        print(report.summary())


def cmd_apply_decay(args: argparse.Namespace) -> None:
    agent = _get_emms(args.memory)
    report = agent.apply_memory_decay(
        domain=getattr(args, "domain", None) or None,
        prune=getattr(args, "prune", False),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "total_processed": report.total_processed,
            "decayed": report.decayed,
            "pruned": report.pruned,
            "mean_retention": report.mean_retention,
            "applied": report.applied,
        }))
    else:
        print(report.summary())


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

    # v0.9.0 commands

    # index-lookup
    p_il = sub.add_parser("index-lookup", help="O(1) CompactionIndex lookup by id, experience_id, or content.")
    p_il.add_argument("--memory-id", default=None, dest="memory_id")
    p_il.add_argument("--experience-id", default=None, dest="experience_id")
    p_il.add_argument("--content", "-c", default=None)
    p_il.add_argument("--action", default="lookup", choices=["lookup", "stats", "rebuild"])
    p_il.set_defaults(func=cmd_index_lookup)

    # index-stats
    p_is = sub.add_parser("index-stats", help="Show CompactionIndex statistics.")
    p_is.set_defaults(func=cmd_index_stats)

    # graph-communities
    p_gc = sub.add_parser("graph-communities",
                           help="Detect communities in the knowledge graph using Label Propagation.")
    p_gc.add_argument("--max-iter", type=int, default=100, dest="max_iter")
    p_gc.add_argument("--min-size", type=int, default=1, dest="min_size",
                      help="Minimum community size (merge smaller into first community).")
    p_gc.add_argument("--markdown", action="store_true",
                      help="Output Markdown report.")
    p_gc.add_argument("--output", "-o", default=None, help="Write Markdown to file.")
    p_gc.set_defaults(func=cmd_graph_communities)

    # replay
    p_rep = sub.add_parser("replay", help="Sample a mini-batch by prioritized experience replay.")
    p_rep.add_argument("--k", type=int, default=8, help="Batch size.")
    p_rep.add_argument("--beta", type=float, default=0.4, help="IS correction exponent.")
    p_rep.set_defaults(func=cmd_replay)

    # replay-top
    p_rept = sub.add_parser("replay-top", help="Show top-k highest-priority memories (deterministic).")
    p_rept.add_argument("--k", type=int, default=8)
    p_rept.set_defaults(func=cmd_replay_top)

    # merge-from
    p_mf = sub.add_parser("merge-from", help="Merge memories from another EMMS memory file.")
    p_mf.add_argument("--source", "-s", default=None,
                      help="Path to source memory JSON file.")
    p_mf.add_argument("--policy", default="newest_wins",
                      choices=["local_wins", "newest_wins", "importance_wins"])
    p_mf.add_argument("--namespace", default=None,
                      help="Prepend prefix/ to incoming ids.")
    p_mf.set_defaults(func=cmd_merge_from)

    # plan-retrieve
    p_pr = sub.add_parser("plan-retrieve",
                           help="Decompose query, retrieve per sub-query, cross-boost, merge.")
    p_pr.add_argument("query")
    p_pr.add_argument("--max", "-n", type=int, default=20)
    p_pr.add_argument("--max-sub", type=int, default=10, dest="max_sub",
                      help="Max results per sub-query.")
    p_pr.add_argument("--boost", type=float, default=0.10,
                      help="Cross-boost increment per additional sub-query hit.")
    p_pr.set_defaults(func=cmd_plan_retrieve)

    # v0.10.0 commands

    # reconsolidate
    p_rc = sub.add_parser("reconsolidate",
                           help="Reconsolidate a memory after recall (strengthen/weaken + valence drift).")
    p_rc.add_argument("memory_id", help="MemoryItem ID to reconsolidate.")
    p_rc.add_argument("--context-valence", type=float, default=None, dest="context_valence",
                      help="Emotional valence of recall context (-1..+1).")
    p_rc.add_argument("--weaken", action="store_true",
                      help="Weaken rather than reinforce.")
    p_rc.set_defaults(func=cmd_reconsolidate)

    # decay-unrecalled
    p_du = sub.add_parser("decay-unrecalled",
                           help="Apply passive strength decay to memories not recently recalled.")
    p_du.add_argument("--decay-factor", type=float, default=0.02, dest="decay_factor",
                      help="Absolute strength reduction per call.")
    p_du.add_argument("--min-age", type=float, default=3600.0, dest="min_age",
                      help="Minimum seconds since last access before decaying (default 3600).")
    p_du.set_defaults(func=cmd_decay_unrecalled)

    # presence
    p_ps = sub.add_parser("presence",
                           help="Show session presence metrics (attention budget, coherence, emotional arc).")
    p_ps.add_argument("--content", default=None, help="If given, record a turn with this content.")
    p_ps.add_argument("--domain", default="general")
    p_ps.add_argument("--valence", type=float, default=0.0)
    p_ps.add_argument("--intensity", type=float, default=0.0)
    p_ps.set_defaults(func=cmd_presence)

    # presence-arc
    p_pa = sub.add_parser("presence-arc",
                           help="Display the per-turn emotional arc of the active session.")
    p_pa.set_defaults(func=cmd_presence_arc)

    # affective-retrieve
    p_af = sub.add_parser("affective-retrieve",
                           help="Retrieve memories by emotional proximity (valence/intensity).")
    p_af.add_argument("--query", "-q", default="", help="Optional semantic query.")
    p_af.add_argument("--valence", type=float, default=None, dest="target_valence",
                      help="Target emotional valence (-1..+1).")
    p_af.add_argument("--intensity", type=float, default=None, dest="target_intensity",
                      help="Target emotional intensity (0..1).")
    p_af.add_argument("--max", "-n", type=int, default=10)
    p_af.add_argument("--blend", type=float, default=0.4, dest="semantic_blend",
                      help="Weight of semantic score [0,1].")
    p_af.set_defaults(func=cmd_affective_retrieve)

    # emotional-landscape
    p_el = sub.add_parser("emotional-landscape",
                           help="Summarise emotional distribution across all memories.")
    p_el.set_defaults(func=cmd_emotional_landscape)

    # v0.11.0 commands

    # dream
    p_dr = sub.add_parser("dream",
                           help="Run between-session dream consolidation (replay, strengthen, prune).")
    p_dr.add_argument("--session-id", default=None, dest="session_id")
    p_dr.add_argument("--reinforce-top-k", type=int, default=20, dest="reinforce_top_k")
    p_dr.add_argument("--weaken-bottom-k", type=int, default=10, dest="weaken_bottom_k")
    p_dr.add_argument("--prune-threshold", type=float, default=0.05, dest="prune_threshold")
    p_dr.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                      help="Skip deduplication pass.")
    p_dr.set_defaults(func=cmd_dream)

    # capture-bridge
    p_cb = sub.add_parser("capture-bridge",
                           help="Capture unresolved session threads as a BridgeRecord.")
    p_cb.add_argument("--session-id", default=None, dest="session_id")
    p_cb.add_argument("--summary", default="", help="Closing summary of this session.")
    p_cb.add_argument("--max-threads", type=int, default=5, dest="max_threads")
    p_cb.add_argument("--output", "-o", default=None,
                      help="Save bridge JSON to this file.")
    p_cb.set_defaults(func=cmd_capture_bridge)

    # inject-bridge
    p_ib = sub.add_parser("inject-bridge",
                           help="Generate context injection from a saved BridgeRecord.")
    p_ib.add_argument("--bridge-file", "-b", default=None, dest="bridge_file",
                      help="Path to bridge JSON file (from capture-bridge --output).")
    p_ib.add_argument("--session-id", default=None, dest="session_id",
                      help="New session ID.")
    p_ib.set_defaults(func=cmd_inject_bridge)

    # anneal
    p_an = sub.add_parser("anneal",
                           help="Anneal memory after a session gap (temporal decay + valence stabilization).")
    p_an.add_argument("--last-session-at", type=float, default=None, dest="last_session_at",
                      help="Unix timestamp of last session end.")
    p_an.add_argument("--half-life-gap", type=float, default=259200.0, dest="half_life_gap",
                      help="Gap in seconds at which temperature=0.5 (default 3 days).")
    p_an.add_argument("--decay-rate", type=float, default=0.03, dest="decay_rate")
    p_an.add_argument("--emotional-stabilization-rate", type=float, default=0.08,
                      dest="emotional_stabilization_rate")
    p_an.set_defaults(func=cmd_anneal)

    # bridge-summary
    p_bs2 = sub.add_parser("bridge-summary",
                            help="Show current session's unresolved threads (bridge preview).")
    p_bs2.set_defaults(func=cmd_bridge_summary)

    # v0.13.0 commands

    # metacognition
    p_mc = sub.add_parser("metacognition",
                           help="Generate a metacognitive self-assessment: confidence, gaps, contradictions.")
    p_mc.add_argument("--max-contradictions", type=int, default=5, dest="max_contradictions")
    p_mc.add_argument("--threshold-high", type=float, default=0.65, dest="threshold_high",
                      help="Confidence above this = high confidence (default 0.65).")
    p_mc.add_argument("--threshold-low", type=float, default=0.3, dest="threshold_low",
                      help="Confidence below this = low confidence (default 0.3).")
    p_mc.set_defaults(func=cmd_metacognition)

    # knowledge-map
    p_km = sub.add_parser("knowledge-map",
                           help="Show per-domain knowledge profiles (confidence, coverage, importance).")
    p_km.set_defaults(func=cmd_knowledge_map)

    # contradictions
    p_ct = sub.add_parser("contradictions",
                           help="Find memory pairs with semantic overlap but conflicting emotional valence.")
    p_ct.add_argument("--max-pairs", type=int, default=10, dest="max_pairs")
    p_ct.set_defaults(func=cmd_contradictions)

    # intend
    p_in = sub.add_parser("intend",
                           help="Store a future-oriented intention with a trigger context.")
    p_in.add_argument("content", help="What the agent plans to do.")
    p_in.add_argument("--trigger", "-t", required=True,
                      help="Context description that should trigger this intention.")
    p_in.add_argument("--priority", "-p", type=float, default=0.5,
                      help="Urgency 0–1 (default 0.5).")
    p_in.set_defaults(func=cmd_intend)

    # check-intentions
    p_ci = sub.add_parser("check-intentions",
                           help="Check which stored intentions are activated by the current context.")
    p_ci.add_argument("--context", "-c", default=None,
                      help="Current conversational context text.")
    p_ci.set_defaults(func=cmd_check_intentions)

    # v0.12.0 commands

    # association-graph
    p_ag = sub.add_parser("association-graph",
                           help="Build the association graph over all memories and show statistics.")
    p_ag.add_argument("--semantic-threshold", type=float, default=0.5, dest="semantic_threshold",
                      help="Minimum cosine similarity for a semantic edge (default 0.5).")
    p_ag.add_argument("--temporal-window", type=float, default=300.0, dest="temporal_window",
                      help="Max seconds between stored_at timestamps for a temporal edge (default 300).")
    p_ag.add_argument("--affective-tolerance", type=float, default=0.3, dest="affective_tolerance",
                      help="Max |valence_a - valence_b| for an affective edge (default 0.3).")
    p_ag.set_defaults(func=cmd_association_graph)

    # activation
    p_act = sub.add_parser("activation",
                            help="Run spreading activation from seed memory IDs.")
    p_act.add_argument("seed_ids", nargs="+", metavar="MEMORY_ID",
                       help="One or more memory IDs to start activation from.")
    p_act.add_argument("--decay", type=float, default=0.5,
                       help="Activation decay factor per hop (default 0.5).")
    p_act.add_argument("--steps", type=int, default=3,
                       help="Maximum hop depth (default 3).")
    p_act.set_defaults(func=cmd_activation)

    # discover-insights
    p_di = sub.add_parser("discover-insights",
                           help="Find cross-domain memory bridges and generate insight memories.")
    p_di.add_argument("--session-id", default=None, dest="session_id")
    p_di.add_argument("--max-insights", type=int, default=8, dest="max_insights",
                      help="Maximum insight memories to generate (default 8).")
    p_di.add_argument("--min-bridge-weight", type=float, default=0.45, dest="min_bridge_weight",
                      help="Minimum edge weight to qualify as a bridge (default 0.45).")
    p_di.add_argument("--no-rebuild", action="store_true", dest="no_rebuild",
                      help="Skip rebuilding the association graph.")
    p_di.set_defaults(func=cmd_discover_insights)

    # associative-retrieve
    p_ar = sub.add_parser("associative-retrieve",
                           help="Retrieve memories via spreading activation (from query or seed IDs).")
    p_ar.add_argument("--query", "-q", default=None, help="Text query to find seed memories.")
    p_ar.add_argument("--seed-ids", nargs="*", default=None, dest="seed_ids",
                      help="Explicit seed memory IDs (overrides --query).")
    p_ar.add_argument("--seed-count", type=int, default=3, dest="seed_count",
                      help="Seeds to pick from query (default 3).")
    p_ar.add_argument("--max", "-n", type=int, default=10)
    p_ar.add_argument("--steps", type=int, default=3)
    p_ar.add_argument("--decay", type=float, default=0.5)
    p_ar.set_defaults(func=cmd_associative_retrieve)

    # association-stats
    p_as2 = sub.add_parser("association-stats",
                            help="Show statistics for the current association graph.")
    p_as2.set_defaults(func=cmd_association_stats)

    # v0.14.0 commands

    # open-episode
    p_oe = sub.add_parser("open-episode",
                           help="Open a new bounded episode in the episodic buffer.")
    p_oe.add_argument("--topic", default="", help="Brief description of the episode topic.")
    p_oe.add_argument("--session-id", default=None, dest="session_id",
                      help="Optional session label (auto-generated if omitted).")
    p_oe.set_defaults(func=cmd_open_episode)

    # close-episode
    p_ce = sub.add_parser("close-episode",
                           help="Close the current episode and compute final statistics.")
    p_ce.add_argument("--outcome", default="", help="Brief resolution description.")
    p_ce.set_defaults(func=cmd_close_episode)

    # recent-episodes
    p_re = sub.add_parser("recent-episodes",
                           help="List the N most recent episodes from the episodic buffer.")
    p_re.add_argument("-n", type=int, default=10,
                      help="Number of episodes to return (default 10).")
    p_re.set_defaults(func=cmd_recent_episodes)

    # extract-schemas
    p_es = sub.add_parser("extract-schemas",
                           help="Extract abstract knowledge schemas from memory via keyword clustering.")
    p_es.add_argument("--domain", default=None,
                      help="Restrict to one domain (default: all).")
    p_es.add_argument("--max-schemas", type=int, default=12, dest="max_schemas",
                      help="Maximum schemas to return (default 12).")
    p_es.set_defaults(func=cmd_extract_schemas)

    # forget
    p_fg = sub.add_parser("forget",
                           help="Suppress or prune memories (targeted, domain-wide, or by confidence).")
    target_grp = p_fg.add_mutually_exclusive_group()
    target_grp.add_argument("--memory-id", default=None, dest="memory_id",
                             help="Suppress a specific memory by ID.")
    target_grp.add_argument("--domain", default=None,
                             help="Suppress all memories in a domain.")
    target_grp.add_argument("--below-confidence", type=float, default=None,
                             dest="below_confidence",
                             help="Suppress memories below this confidence threshold (0–1).")
    p_fg.add_argument("--rate", type=float, default=0.4,
                      help="Suppression rate (default 0.4).")
    p_fg.set_defaults(func=cmd_forget)

    # v0.15.0 commands

    # reflect
    p_rf = sub.add_parser("reflect",
                           help="Run a structured self-reflection pass: synthesise lessons from memory and episodes.")
    p_rf.add_argument("--domain", default=None,
                      help="Restrict reflection to one domain (default: all).")
    p_rf.add_argument("--lookback-episodes", type=int, default=5, dest="lookback_episodes",
                      help="Number of recent episodes to incorporate (default 5).")
    p_rf.set_defaults(func=cmd_reflect)

    # weave-narrative
    p_wn = sub.add_parser("weave-narrative",
                           help="Weave autobiographical narrative threads from stored memories.")
    p_wn.add_argument("--domain", default=None,
                      help="Restrict to one domain (default: all).")
    p_wn.add_argument("--max-threads", type=int, default=8, dest="max_threads",
                      help="Maximum narrative threads to return (default 8).")
    p_wn.set_defaults(func=cmd_weave_narrative)

    # narrative-threads
    p_nt = sub.add_parser("narrative-threads",
                           help="List narrative threads for one or all domains.")
    p_nt.add_argument("--domain", default=None,
                      help="Restrict to one domain (default: all).")
    p_nt.set_defaults(func=cmd_narrative_threads)

    # source-audit
    p_sa = sub.add_parser("source-audit",
                           help="Audit memories for source uncertainty and confabulation risk.")
    p_sa.add_argument("--flag-threshold", type=float, default=0.5, dest="flag_threshold",
                      help="Confidence below which a memory is flagged (default 0.5).")
    p_sa.add_argument("--max-flagged", type=int, default=20, dest="max_flagged",
                      help="Maximum flagged entries to show (default 20).")
    p_sa.set_defaults(func=cmd_source_audit)

    # tag-source
    p_ts = sub.add_parser("tag-source",
                           help="Assign a provenance tag to a memory.")
    p_ts.add_argument("memory_id", help="Memory ID to tag.")
    p_ts.add_argument("source_type",
                      choices=["observation", "inference", "instruction",
                               "reflection", "dream", "insight", "unknown"],
                      help="Memory provenance type.")
    p_ts.add_argument("--confidence", type=float, default=0.8,
                      help="Confidence in this attribution (default 0.8).")
    p_ts.add_argument("--note", default="", help="Optional free-text provenance note.")
    p_ts.set_defaults(func=cmd_tag_source)

    # v0.16.0 commands

    # curiosity-report
    p_cr = sub.add_parser("curiosity-report",
                           help="Scan memory for knowledge gaps and generate exploration goals.")
    p_cr.add_argument("--domain", default=None, help="Restrict scan to one domain.")
    p_cr.set_defaults(func=cmd_curiosity_report)

    # explore-goals
    p_eg = sub.add_parser("explore-goals",
                           help="List all pending curiosity-driven exploration goals.")
    p_eg.set_defaults(func=cmd_explore_goals)

    # revise-beliefs
    p_rb = sub.add_parser("revise-beliefs",
                           help="Detect and resolve contradictions via AGM belief revision.")
    p_rb.add_argument("--memory-id", default=None, dest="memory_id",
                      help="Check only this memory against others (omit for full scan).")
    p_rb.add_argument("--domain", default=None, help="Restrict scan to one domain.")
    p_rb.add_argument("--max-revisions", type=int, default=8, dest="max_revisions",
                      help="Maximum revisions to perform (default 8).")
    p_rb.set_defaults(func=cmd_revise_beliefs)

    # decay-report
    p_dec = sub.add_parser("decay-report",
                            help="Compute Ebbinghaus forgetting retention for all memories (read-only).")
    p_dec.add_argument("--domain", default=None, help="Restrict report to one domain.")
    p_dec.set_defaults(func=cmd_decay_report)

    # apply-decay
    p_ad = sub.add_parser("apply-decay",
                           help="Apply Ebbinghaus forgetting curve to memory strengths.")
    p_ad.add_argument("--domain", default=None, help="Restrict decay to one domain.")
    p_ad.add_argument("--prune", action="store_true",
                      help="Remove memories below the prune threshold after decay.")
    p_ad.set_defaults(func=cmd_apply_decay)

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
