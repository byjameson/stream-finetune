"""ANN (IVF) quality/latency benchmark against exact cosine.

    python research/benchmark_ann_index.py --probes 4,8,12

Uses clustered synthetic vectors so IVF has realistic structure. Exact cosine
is the fallback and the quality reference. The production wrapper is covered by
tests and the live server smoke.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from atomzero.doc_memory import ann_topk, build_ivf

ROOT = Path(__file__).resolve().parents[1]


def normalized(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--clusters", type=int, default=128)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--probes", default="4,8,12")
    ap.add_argument("--out", default="research/results/ann_index_benchmark.json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    centers = normalized(rng.normal(size=(args.clusters, args.dim)))
    labels = rng.integers(0, args.clusters, size=args.vectors)
    matrix = normalized(centers[labels] +
                        0.02 * rng.normal(size=(args.vectors, args.dim)))
    query_labels = rng.integers(0, args.clusters, size=args.queries)
    queries = normalized(centers[query_labels] +
                         0.02 * rng.normal(size=(args.queries, args.dim)))

    started = time.perf_counter()
    ann = build_ivf(matrix, nlist=args.clusters, iterations=6)
    build_s = time.perf_counter() - started

    exact_ms = []
    exact_orders = []
    for query in queries:
        started = time.perf_counter()
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[:args.k]
        exact_ms.append((time.perf_counter() - started) * 1000.0)
        exact_orders.append(order)

    probes = [int(value) for value in args.probes.split(",") if value.strip()]
    rows = []
    for probe in probes:
        ann_ms = []
        recalls = []
        for query, exact in zip(queries, exact_orders):
            started = time.perf_counter()
            positions, _ = ann_topk(matrix, ann, query, args.k, probe=probe)
            ann_ms.append((time.perf_counter() - started) * 1000.0)
            recalls.append(len(set(int(i) for i in exact) &
                               set(int(i) for i in positions)) / args.k)
        rows.append({
            "probe": probe,
            "ann_median_ms": round(statistics.median(ann_ms), 3),
            "speedup": round(statistics.median(exact_ms) /
                             max(statistics.median(ann_ms), 0.001), 1),
            "recall_at_k": round(statistics.mean(recalls), 4),
            "min_recall_at_k": round(min(recalls), 4),
        })

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vectors": args.vectors,
        "dim": args.dim,
        "clusters": args.clusters,
        "queries": args.queries,
        "k": args.k,
        "build_s": round(build_s, 3),
        "exact_median_ms": round(statistics.median(exact_ms), 3),
        "probes": rows,
        "note": ("IVF ANN over clustered synthetic normalized vectors; "
                 "exact cosine fallback remains the product safety net."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"vectors={result['vectors']} dim={result['dim']} "
          f"clusters={result['clusters']} build={result['build_s']}s "
          f"exact={result['exact_median_ms']}ms")
    for row in rows:
        print(f"probe={row['probe']:2d} ann={row['ann_median_ms']:.3f}ms "
              f"speedup={row['speedup']}x recall@{args.k}="
              f"{row['recall_at_k']:.4f} min={row['min_recall_at_k']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
