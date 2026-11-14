"""Plot the two-phase spiking rollback trade-off.

    python research/plot_spiking_shift.py

Reads spiking_shift.json (and the seed-1 file if present) and writes
spiking_shift.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    ("seed 0", ROOT / "research/results/spiking_shift.json", "tab:blue", "o"),
    ("seed 1", ROOT / "research/results/spiking_shift_seed1.json", "tab:orange", "s"),
]
OUT = ROOT / "research/results/spiking_shift.png"


def arm_points(payload):
    arms = {r["name"]: r for r in payload["arms"]}
    points = [(0.0, arms["phase_b_no_rollback"])]
    for arm in payload["arms"]:
        if arm["name"].startswith("importance_"):
            points.append((arm["fraction"], arm))
    points.append((1.0, arms["global_rollback"]))
    points.sort(key=lambda x: x[0])
    return arms["phase_a"], points


def main():
    import json
    fig, (ax_old, ax_new) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for label, path, color, marker in RUNS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        a, points = arm_points(payload)
        xs = [p[0] for p in points]
        old = [p[1]["old"]["bpb"] for p in points]
        new = [p[1]["new"]["bpb"] for p in points]
        ax_old.plot(xs, old, marker=marker, color=color, label=label,
                    linewidth=1.8)
        ax_new.plot(xs, new, marker=marker, color=color, label=label,
                    linewidth=1.8)
        ax_old.axhline(a["old"]["bpb"], color=color, linestyle=":", alpha=0.6)
        ax_new.axhline(a["new"]["bpb"], color=color, linestyle=":", alpha=0.6)

    ax_old.set_ylabel("TinyStories bpb (old task)")
    ax_old.set_title("Old task: rollback recovers TinyStories")
    ax_old.grid(alpha=0.25)
    ax_old.legend(frameon=False, fontsize=9)

    ax_new.set_ylabel("WikiText bpb (new task)")
    ax_new.set_title("New task: rollback spends WikiText gain")
    ax_new.grid(alpha=0.25)
    ax_new.legend(frameon=False, fontsize=9)

    for ax in (ax_old, ax_new):
        ax.set_xlabel("fraction of synapses restored from phase A")
        ax.set_xticks([0.0, 0.1, 0.25, 0.5, 1.0])
    fig.suptitle("Per-synapse shadow rollback after a TinyStories -> WikiText shift",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
