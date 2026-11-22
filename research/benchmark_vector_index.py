"""Scale benchmark for the prewarmed vector index vs the linear scan.

    python research/benchmark_vector_index.py

Uses hashed n-gram vectors so the benchmark does not need a neural model. The
linear path reads every SQLite blob and scans in Python; the prewarmed path uses
one numpy matrix-vector product. Both are exact, not approximate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import DocMemory

ROOT = Path(__file__).resolve().parents[1]


def _time_calls(fn, repeats: int):
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings), result


def run(chunks: int, repeats: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="atomzero-vector-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            for index in range(chunks):
                docs.add_text(
                    f"policy-{index}.txt",
                    f"Policy {index}: the approval code is CODE-{index}. "
                    f"The team owns routine operations for project {index}.",
                    title=f"Policy {index}", replace=False)
            docs.search("policy", k=5, rerank=False)

            started = time.perf_counter()
            stats = docs.prewarm_vectors(None)
            prewarm_ms = (time.perf_counter() - started) * 1000.0

            matrix_ms, (matrix_hits, _) = _time_calls(
                lambda: docs._embedding_candidates("policy", None, 5),
                repeats)
            scan_ms, (scan_hits, _) = _time_calls(
                lambda: docs._embedding_candidates_scan("policy", None, 5),
                max(3, repeats // 4))
            search_ms, _ = _time_calls(
                lambda: docs.search("policy", k=5), repeats)

            matrix_sources = {hit["source"] for hit in matrix_hits}
            scan_sources = {hit["source"] for hit in scan_hits}
            overlap = len(matrix_sources & scan_sources) / max(
                len(matrix_sources | scan_sources), 1)
            result = {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "chunks": chunks,
                "dim": stats["dim"],
                "backend": stats["backend"],
                "prewarm_ms": round(prewarm_ms, 2),
                "matrix_median_ms": round(matrix_ms, 3),
                "scan_median_ms": round(scan_ms, 3),
                "search_median_ms": round(search_ms, 3),
                "speedup_vs_scan": round(scan_ms / max(matrix_ms, 0.001), 1),
                "top5_overlap": round(overlap, 3),
                "note": ("Hashed vectors, exact cosine. The scan reads every "
                         "doc_vectors blob per query; the matrix is built once."),
            }
        finally:
            docs.close()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out",
                    default="research/results/vector_index_benchmark.json")
    args = ap.parse_args(argv)
    result = run(max(100, args.chunks), max(3, args.repeats))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"chunks={result['chunks']} dim={result['dim']} "
          f"prewarm={result['prewarm_ms']}ms")
    print(f"matrix={result['matrix_median_ms']}ms "
          f"scan={result['scan_median_ms']}ms "
          f"speedup={result['speedup_vs_scan']}x")
    print(f"search={result['search_median_ms']}ms "
          f"top5_overlap={result['top5_overlap']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
