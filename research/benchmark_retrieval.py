"""Held-out retrieval benchmark: raw FTS vs hybrid candidate stages.

    python research/benchmark_retrieval.py

The corpus and paraphrased queries are authored for this benchmark; no reranker
is trained on them. The neural candidate is the local MiniLM-L6-v2 directory.
The hashed fallback only adds candidates when FTS found an anchor; the neural
pool may answer semantic-only paraphrases above its cosine threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from streamft.doc_memory import DocMemory
from streamft.neural_embedder import NeuralReranker

ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    ("launch.txt", "Launch policy",
     "The launch authorization code is ORANGE-42. Do not share it outside the launch team."),
    ("warranty.txt", "Warranty",
     "Warranty claims must be filed within 30 days of delivery. Keep the receipt."),
    ("travel.txt", "Travel",
     "Employees may book economy class for business flights under six hours."),
    ("security.txt", "Visitor security",
     "Visitor identification badges must be worn visibly at all times."),
    ("account.txt", "Account recovery",
     "For locked account recovery, contact help@example.test."),
    ("printer.txt", "Printer",
     "Replace the cyan cartridge when print quality fades."),
    ("generator.txt", "Emergency power",
     "The backup generator provides emergency power and is tested on the first Monday of each month."),
    ("phishing.txt", "Phishing",
     "Report phishing or suspicious email to security@example.test."),
    ("homeoffice.txt", "Home office",
     "The home-office stipend is 300 dollars per year."),
    ("vacation.txt", "Annual leave",
     "Employees may carry over up to five unused annual leave days."),
    ("cafeteria.txt", "Cafeteria",
     "The cafeteria and canteen close at 14:30 on weekdays."),
    ("finance.txt", "Expenses",
     "Submit expense reports through the finance portal by Friday."),
    ("assembly.txt", "Fire safety",
     "During a fire alarm, the emergency assembly point is the north parking lot."),
    ("laptop.txt", "IT equipment",
     "Request a new laptop or computer through the IT service desk."),
    ("guestwifi.txt", "Guest Wi-Fi",
     "Visitors can get guest Wi-Fi access with a daily password from the front desk."),
    ("firedrill.txt", "Evacuation",
     "The annual evacuation practice and fire drill is scheduled for the first Tuesday of March."),
    ("dresscode.txt", "Dress code",
     "Business casual dress is required in client meetings."),
    ("firstaid.txt", "First aid",
     "First-aid kits with medical supplies are located beside every kitchen."),
    ("retention.txt", "Data retention",
     "Client data and customer records are retained for seven years."),
    ("backup.txt", "Backups",
     "Nightly data copy backups run every night at 02:00."),
]

QUERIES = [
    ("What code authorizes a launch?", "launch.txt"),
    ("How long do I have to submit a warranty claim?", "warranty.txt"),
    ("What class should I book for a short work flight?", "travel.txt"),
    ("Do guests need to display identification?", "security.txt"),
    ("How can I regain access to a locked login?", "account.txt"),
    ("Why are my documents printing pale?", "printer.txt"),
    ("When is the emergency power system checked?", "generator.txt"),
    ("Where do I send a phishing message?", "phishing.txt"),
    ("What is the annual allowance for working from home?", "homeoffice.txt"),
    ("What is the maximum rollover for annual leave?", "vacation.txt"),
    ("When does the canteen shut?", "cafeteria.txt"),
    ("When is the deadline for expense claims?", "finance.txt"),
    ("Where should we gather during a fire alarm?", "assembly.txt"),
    ("How do I get a replacement computer?", "laptop.txt"),
    ("How do visitors get internet access?", "guestwifi.txt"),
    ("When is the evacuation practice?", "firedrill.txt"),
    ("What should I wear to a customer meeting?", "dresscode.txt"),
    ("Where can I find medical supplies?", "firstaid.txt"),
    ("How long do we keep client data?", "retention.txt"),
    ("When is the nightly data copy?", "backup.txt"),
]


def score(ranked: list[dict], expected: str) -> dict:
    sources = [str(hit.get("source", "")) for hit in ranked]
    try:
        rank = sources.index(expected) + 1
    except ValueError:
        rank = 0
    return {
        "top1": bool(rank == 1),
        "top3": bool(1 <= rank <= 3),
        "mrr": (1.0 / rank) if rank else 0.0,
        "citation": bool(rank == 1),
        "rank": rank,
    }


def evaluate(docs: DocMemory, reranker) -> dict:
    raw_rows, hashed_rows, neural_rows = [], [], []
    raw_ms, hashed_ms, neural_ms = [], [], []
    fts_recall = 0
    hybrid_recall = 0
    # Warm FTS, the neural model and both vector caches before timing.
    first = QUERIES[0][0]
    docs.search(first, k=10, rerank=False)
    docs.search(first, k=10, rerank=True, reranker=None)
    docs.search(first, k=10, rerank=True, reranker=reranker)
    for query, expected in QUERIES:
        start = time.perf_counter()
        fts = docs.search(query, k=10, rerank=False)
        search_ms = (time.perf_counter() - start) * 1000
        if any(hit.get("source") == expected for hit in fts):
            fts_recall += 1
        raw_rows.append({"query": query, "expected": expected,
                         **score(fts[:5], expected)})
        raw_ms.append(search_ms)

        start = time.perf_counter()
        hashed = docs.search(query, k=5, rerank=True, reranker=None)
        hashed_ms.append((time.perf_counter() - start) * 1000)
        hashed_rows.append({"query": query, "expected": expected,
                            **score(hashed, expected)})

        start = time.perf_counter()
        neural = docs.search(query, k=5, rerank=True, reranker=reranker)
        neural_ms.append((time.perf_counter() - start) * 1000)
        neural_rows.append({"query": query, "expected": expected,
                            **score(neural, expected)})

        deep = docs.search(query, k=10, rerank=True, reranker=reranker)
        if any(hit.get("source") == expected for hit in deep):
            hybrid_recall += 1

    def summarize(rows, latencies):
        n = len(rows)
        return {
            "top1": sum(row["top1"] for row in rows) / n,
            "top3": sum(row["top3"] for row in rows) / n,
            "mrr": sum(row["mrr"] for row in rows) / n,
            "citation": sum(row["citation"] for row in rows) / n,
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[max(0, int(n * 0.95) - 1)],
            "rows": rows,
        }

    return {
        "candidate_recall_at_10": fts_recall / len(QUERIES),
        "hybrid_recall_at_10": hybrid_recall / len(QUERIES),
        "raw": summarize(raw_rows, raw_ms),
        "hashed": summarize(hashed_rows, hashed_ms),
        "neural": summarize(neural_rows, neural_ms),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural-model", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="research/results/retrieval_benchmark.json")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="atomzero-retrieval-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            for source, title, text in DOCS:
                docs.add_text(source, text, title=title)
            reranker = NeuralReranker(args.neural_model, device=args.device)
            started = time.perf_counter()
            reranker.embed(["warmup"])
            result = evaluate(docs, reranker)
            result["neural_model"] = args.neural_model
            result["neural_load_s"] = round(
                time.perf_counter() - started, 2)
        finally:
            docs.close()
    result.update({"docs": len(DOCS), "queries": len(QUERIES),
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for name in ("raw", "hashed", "neural"):
        row = result[name]
        print(f"{name:7s} top1={row['top1']:.3f} top3={row['top3']:.3f} "
              f"mrr={row['mrr']:.3f} citation={row['citation']:.3f} "
              f"median={row['median_ms']:.2f}ms")
    print(f"fts candidate recall@10: {result['candidate_recall_at_10']:.3f}")
    print(f"hybrid recall@10: {result['hybrid_recall_at_10']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
