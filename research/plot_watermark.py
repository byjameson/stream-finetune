"""Plot the watermark merge threshold across seeds and tokens.

    python research/plot_watermark.py

Reads the four threshold JSONs and writes research/results/watermark_threshold.png.
The left panel is exact-token hits, the right panel is the teacher-forced mean logprob.
Both use the SST-2 host; the AG host is within one trigger at every fraction.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research/results/watermark_threshold.png"
RUNS = [
    ("token 1 / seed 1", "watermark_threshold.json", "tab:blue", "o"),
    ("token 1 / seed 2", "watermark_threshold_seed2.json", "tab:orange", "s"),
    ("token 1 / seed 3", "watermark_threshold_seed3.json", "tab:green", "^"),
    ("token 2 / seed 4", "watermark_threshold_tok2.json", "tab:red", "D"),
]


def load(fname):
    import json
    payload = json.loads((ROOT / "research/results" / fname).read_text(encoding="utf-8"))
    rows = {r["wm_fraction"]: r for r in payload["results"] if r["host"] == "sst2"}
    xs = sorted(rows)
    return xs, [rows[x]["hits"] for x in xs], [rows[x]["mean_logprob"] for x in xs]


def main():
    fig, (ax_hits, ax_lp) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for label, fname, color, marker in RUNS:
        xs, hits, lps = load(fname)
        ax_hits.plot(xs, hits, marker=marker, color=color, label=label, linewidth=1.8)
        ax_lp.plot(xs, lps, marker=marker, color=color, label=label, linewidth=1.8)

    ax_hits.axhline(8, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_hits.set_ylim(-0.4, 8.6)
    ax_hits.set_ylabel("exact-token hits / 8")
    ax_hits.set_title("Generation: the cliff moves with seed and token")
    ax_hits.grid(alpha=0.25)
    ax_hits.legend(fontsize=8, frameon=False)

    ax_lp.set_ylabel("teacher-forced mean logprob")
    ax_lp.set_title("Logprob: a smooth rise, not a verifier")
    ax_lp.grid(alpha=0.25)
    ax_lp.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    for ax in (ax_hits, ax_lp):
        ax.set_xlabel("watermark fraction in the merge")
        ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

    fig.suptitle("Watermark merge threshold, SST-2 host, four runs", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
