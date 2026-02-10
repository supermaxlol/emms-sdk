#!/usr/bin/env python3
"""EMMS Benchmark Suite — measures performance and compares against MemGPT/Letta.

Run:
    python benchmarks/run.py              # EMMS only
    python benchmarks/run.py --with-letta # EMMS vs Letta comparison

Measures:
    1. Store throughput (experiences/sec)
    2. Retrieval latency (ms/query)
    3. Retrieval precision@k (using ground-truth topic labels)
    4. Memory consolidation effectiveness
    5. Embedding retrieval vs lexical retrieval quality
    6. ChromaDB vector store performance
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmarks.dataset import generate_dataset, QUERIES
from emms import EMMS, Experience, MemoryConfig
from emms.core.embeddings import HashEmbedder

# Optional imports
try:
    from emms.storage.chroma import ChromaStore
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


def _fmt_table(headers: list[str], rows: list[list]) -> str:
    if _HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f")
    # Fallback plain
    lines = ["  ".join(f"{h:>18}" for h in headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append("  ".join(f"{str(v):>18}" for v in row))
    return "\n".join(lines)


# ============================================================================
# Benchmark functions
# ============================================================================

def bench_store_throughput(agent: EMMS, experiences: list[Experience]) -> dict:
    """Measure store throughput."""
    start = time.perf_counter()
    for exp in experiences:
        agent.store(exp)
    elapsed = time.perf_counter() - start

    return {
        "count": len(experiences),
        "elapsed_s": round(elapsed, 4),
        "throughput": round(len(experiences) / elapsed, 1),
    }


def bench_retrieval_latency(agent: EMMS, queries: list[str], use_semantic: bool = False) -> dict:
    """Measure average retrieval latency."""
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        if use_semantic:
            agent.retrieve_semantic(q, max_results=10)
        else:
            agent.retrieve(q, max_results=10)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "queries": len(queries),
        "avg_ms": round(sum(latencies) / len(latencies), 3),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 3),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 3),
        "total_ms": round(sum(latencies), 2),
    }


def bench_precision_at_k(
    agent: EMMS,
    ground_truth: dict[str, list[str]],
    k: int = 10,
    use_semantic: bool = False,
) -> dict:
    """Measure precision@k for each query."""
    precisions = []

    for query_text, expected_domain in QUERIES.items():
        relevant_ids = set(ground_truth.get(expected_domain, []))

        if use_semantic:
            results = agent.retrieve_semantic(query_text, max_results=k)
            retrieved_ids = {r["id"] for r in results}
        else:
            results = agent.retrieve(query_text, max_results=k)
            retrieved_ids = {r.memory.experience.id for r in results}

        if not retrieved_ids:
            precisions.append(0.0)
            continue

        hits = len(retrieved_ids & relevant_ids)
        precision = hits / len(retrieved_ids)
        precisions.append(precision)

    return {
        "avg_precision_at_k": round(sum(precisions) / len(precisions), 4),
        "per_query": {q: round(p, 3) for q, p in zip(QUERIES.keys(), precisions)},
        "k": k,
    }


def bench_consolidation(agent: EMMS) -> dict:
    """Measure consolidation performance."""
    t0 = time.perf_counter()
    result = agent.consolidate()
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "items_consolidated": result["items_consolidated"],
        "elapsed_ms": round(elapsed, 3),
        "memory_sizes": result["memory_sizes"],
    }


# ============================================================================
# Letta/MemGPT adapter
# ============================================================================

def _try_letta_benchmark(experiences: list[Experience], queries: list[str]) -> dict | None:
    """Try to benchmark Letta (MemGPT) for comparison."""
    try:
        from letta import create_client
    except ImportError:
        return None

    try:
        client = create_client()
        agent = client.create_agent(name="emms_bench_agent")

        # Store
        t0 = time.perf_counter()
        for exp in experiences[:50]:  # Letta is slow, limit to 50
            client.send_message(
                agent_id=agent.id,
                message=f"Remember this: {exp.content}",
                role="user",
            )
        store_elapsed = time.perf_counter() - t0

        # Retrieve
        latencies = []
        for q in queries[:4]:
            t1 = time.perf_counter()
            client.send_message(
                agent_id=agent.id,
                message=f"What do you remember about: {q}",
                role="user",
            )
            latencies.append((time.perf_counter() - t1) * 1000)

        # Cleanup
        client.delete_agent(agent.id)

        return {
            "store_count": 50,
            "store_elapsed_s": round(store_elapsed, 2),
            "store_throughput": round(50 / store_elapsed, 1),
            "retrieve_avg_ms": round(sum(latencies) / len(latencies), 1),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Main
# ============================================================================

def run(with_letta: bool = False, dataset_size: int = 50):
    print("=" * 70)
    print("  EMMS Benchmark Suite v0.2")
    print("=" * 70)

    # Generate dataset
    print(f"\nGenerating dataset: {dataset_size} experiences per topic (4 topics)...")
    experiences, ground_truth = generate_dataset(n_per_topic=dataset_size)
    total = len(experiences)
    queries = list(QUERIES.keys())
    print(f"  Total experiences: {total}")
    print(f"  Topics: {list(ground_truth.keys())}")
    print(f"  Queries: {len(queries)}")

    results_table = []

    # ---- EMMS Lexical (no embeddings) ----
    print("\n--- EMMS (lexical, no embeddings) ---")
    agent_lex = EMMS(config=MemoryConfig(working_capacity=7))

    r_store = bench_store_throughput(agent_lex, experiences)
    print(f"  Store: {r_store['throughput']} exp/s ({r_store['count']} in {r_store['elapsed_s']}s)")

    r_latency = bench_retrieval_latency(agent_lex, queries)
    print(f"  Retrieve: avg {r_latency['avg_ms']:.2f}ms, p50 {r_latency['p50_ms']:.2f}ms")

    r_prec = bench_precision_at_k(agent_lex, ground_truth, k=10)
    print(f"  Precision@10: {r_prec['avg_precision_at_k']:.3f}")

    r_consol = bench_consolidation(agent_lex)
    print(f"  Consolidation: {r_consol['items_consolidated']} items in {r_consol['elapsed_ms']:.2f}ms")

    results_table.append([
        "EMMS (lexical)",
        r_store["throughput"],
        r_latency["avg_ms"],
        r_latency["p50_ms"],
        r_prec["avg_precision_at_k"],
        r_consol["items_consolidated"],
        agent_lex.stats["memory"]["total"],
    ])

    # ---- EMMS with HashEmbedder ----
    print("\n--- EMMS (HashEmbedder, embedding retrieval) ---")
    embedder = HashEmbedder(dim=128)
    agent_emb = EMMS(config=MemoryConfig(working_capacity=7), embedder=embedder)

    r_store2 = bench_store_throughput(agent_emb, experiences)
    print(f"  Store: {r_store2['throughput']} exp/s ({r_store2['count']} in {r_store2['elapsed_s']}s)")

    r_latency2 = bench_retrieval_latency(agent_emb, queries)
    print(f"  Retrieve: avg {r_latency2['avg_ms']:.2f}ms, p50 {r_latency2['p50_ms']:.2f}ms")

    r_prec2 = bench_precision_at_k(agent_emb, ground_truth, k=10)
    print(f"  Precision@10: {r_prec2['avg_precision_at_k']:.3f}")

    r_consol2 = bench_consolidation(agent_emb)
    print(f"  Consolidation: {r_consol2['items_consolidated']} items in {r_consol2['elapsed_ms']:.2f}ms")

    results_table.append([
        "EMMS (HashEmbedder)",
        r_store2["throughput"],
        r_latency2["avg_ms"],
        r_latency2["p50_ms"],
        r_prec2["avg_precision_at_k"],
        r_consol2["items_consolidated"],
        agent_emb.stats["memory"]["total"],
    ])

    # ---- EMMS with ChromaDB ----
    if _HAS_CHROMA:
        print("\n--- EMMS (HashEmbedder + ChromaDB vector store) ---")
        embedder_c = HashEmbedder(dim=128)
        chroma = ChromaStore(embedder=embedder_c)
        agent_chroma = EMMS(
            config=MemoryConfig(working_capacity=7),
            embedder=embedder_c,
            vector_store=chroma,
        )

        r_store3 = bench_store_throughput(agent_chroma, experiences)
        print(f"  Store: {r_store3['throughput']} exp/s ({r_store3['count']} in {r_store3['elapsed_s']}s)")

        r_latency3 = bench_retrieval_latency(agent_chroma, queries, use_semantic=True)
        print(f"  Semantic retrieve: avg {r_latency3['avg_ms']:.2f}ms, p50 {r_latency3['p50_ms']:.2f}ms")

        r_prec3 = bench_precision_at_k(agent_chroma, ground_truth, k=10, use_semantic=True)
        print(f"  Precision@10: {r_prec3['avg_precision_at_k']:.3f}")

        r_consol3 = bench_consolidation(agent_chroma)
        print(f"  Consolidation: {r_consol3['items_consolidated']} items in {r_consol3['elapsed_ms']:.2f}ms")
        print(f"  ChromaDB docs: {chroma.count}")

        results_table.append([
            "EMMS (Chroma+embed)",
            r_store3["throughput"],
            r_latency3["avg_ms"],
            r_latency3["p50_ms"],
            r_prec3["avg_precision_at_k"],
            r_consol3["items_consolidated"],
            agent_chroma.stats["memory"]["total"],
        ])
    else:
        print("\n[SKIP] ChromaDB not installed — run: pip install chromadb")

    # ---- Letta/MemGPT comparison ----
    if with_letta:
        print("\n--- Letta (MemGPT) comparison ---")
        letta_result = _try_letta_benchmark(experiences, queries)
        if letta_result and "error" not in letta_result:
            print(f"  Store: {letta_result['store_throughput']} exp/s (50 items)")
            print(f"  Retrieve: avg {letta_result['retrieve_avg_ms']:.1f}ms")
            results_table.append([
                "Letta (MemGPT)",
                letta_result["store_throughput"],
                letta_result["retrieve_avg_ms"],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            ])
        elif letta_result:
            print(f"  Error: {letta_result['error']}")
            print("  (Letta requires a running server: letta server)")
        else:
            print("  Letta not installed — run: pip install letta")
            print("  Note: Letta benchmarks require a running Letta server")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70 + "\n")

    headers = ["System", "Store/s", "Ret.avg(ms)", "Ret.p50(ms)",
               "P@10", "Consolidated", "Total Mem"]
    print(_fmt_table(headers, results_table))

    # ---- Per-query precision breakdown ----
    print("\n\nPer-query Precision@10 breakdown:")
    pq_headers = ["Query", "Lexical", "Embedding"]
    pq_rows = []
    for q in QUERIES:
        lex_p = r_prec["per_query"].get(q, 0)
        emb_p = r_prec2["per_query"].get(q, 0)
        pq_rows.append([q[:50] + "...", round(lex_p, 3), round(emb_p, 3)])
    print(_fmt_table(pq_headers, pq_rows))

    if _HAS_CHROMA:
        print("\n\nChroma semantic per-query:")
        cq_rows = []
        for q in QUERIES:
            cp = r_prec3["per_query"].get(q, 0)
            cq_rows.append([q[:50] + "...", round(cp, 3)])
        print(_fmt_table(["Query", "ChromaDB P@10"], cq_rows))

    # ---- Letta comparison note ----
    if not with_letta:
        print("\n\nTo compare against Letta/MemGPT:")
        print("  1. pip install letta")
        print("  2. letta server  (in another terminal)")
        print("  3. python benchmarks/run.py --with-letta")
        print("\nNote: Letta operates via LLM API calls, so store/retrieve is")
        print("orders of magnitude slower (~1-5 sec/operation vs EMMS ~0.1ms).")
        print("The comparison highlights EMMS's architectural advantage for")
        print("high-throughput memory operations that don't require LLM inference.")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMMS Benchmark Suite")
    parser.add_argument("--with-letta", action="store_true",
                        help="Include Letta/MemGPT comparison")
    parser.add_argument("--size", type=int, default=50,
                        help="Experiences per topic (default: 50)")
    args = parser.parse_args()
    run(with_letta=args.with_letta, dataset_size=args.size)
