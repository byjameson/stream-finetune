"""Rung chooser: given a task, an accuracy target and whether the base is shared,
pick the cheapest artifact from the measured ladder.

    python research/ladder.py --task agnews --target 0.74 --base-shared no
    python research/ladder.py --task sst2 --target 0.90 --base-shared yes
    python research/ladder.py --task date --target 0.99 --base-shared no

The numbers are the measured rows in research/results/. The tool does not retrain
anything; it is a decision table over those rows.
"""
import argparse

# task, rung, artifact bytes, base bytes, accuracy, note
ROWS = [
    ("agnews", "prompt (341 B)", 341, 3_000_000_000, 0.7375, "needs 1.5B base"),
    ("agnews", "coreset 500", 134_797, 0, 0.705, "train at load time"),
    ("agnews", "linear 5k", 339_800, 0, 0.7425, "TF-IDF + LR"),
    ("agnews", "BERT-tiny 4-bit", 2_190_000, 0, 0.7675, "4.4M params"),
    ("sst2", "prompt (294 B)", 294, 3_000_000_000, 0.9375, "needs 1.5B base"),
    ("sst2", "linear 1k", 44_900, 0, 0.7175, "TF-IDF + LR"),
    ("sst2", "BERT-tiny 4-bit", 2_190_000, 0, 0.7175, "4.4M params"),
    ("sst2", "DistilBERT 2-bit", 16_740_000, 0, 0.735, "66M params"),
    ("sst2", "DistilBERT 4-bit", 33_480_000, 0, 0.8375, "66M params"),
    ("date", "program (93 B)", 93, 752_040, 1.0, "needs Python + dateutil"),
    ("date", "seq2seq 4-bit", 670_000, 0, 0.998, "1.35M params"),
]


def fmt(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["agnews", "sst2", "date"])
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--base-shared", choices=["yes", "no"], default="no")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    shared = args.base_shared == "yes"
    rows = [r for r in ROWS if r[0] == args.task]
    print(f"task {args.task}, target {args.target:.3f}, base shared: {args.base_shared}")
    print(f"{'rung':<22s} {'artifact':>10s} {'base':>10s} {'total':>10s} {'acc':>7s}  note")
    for _, rung, art, base, acc, note in rows:
        total = art if shared else art + base
        print(f"{rung:<22s} {fmt(art):>10s} {fmt(base):>10s} {fmt(total):>10s} "
              f"{acc:7.4f}  {note}")
    ok = [(r, r[2] if shared else r[2] + r[3]) for r in rows if r[4] >= args.target]
    if not ok:
        print(f"VERDICT: no measured rung reaches {args.target:.3f} on this task.")
        return
    best = min(ok, key=lambda x: x[1])
    print(f"VERDICT: {best[0][1]} - total {fmt(best[1])}, accuracy {best[0][4]:.4f} "
          f"({best[0][5]}).")


if __name__ == "__main__":
    main()
