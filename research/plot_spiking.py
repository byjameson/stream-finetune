"""Plot the spiking-LM learning curves.

    python research/plot_spiking.py

Reads research/results/spiking_lm_1000.json and writes spiking_lm.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "research/results/spiking_lm_1000.json"
ADAPT = ROOT / "research/results/spiking_lm_adapt1000.json"
OUT = ROOT / "research/results/spiking_lm.png"
COLORS = {"spike": "tab:blue", "spike + adapt": "tab:green", "rate": "tab:orange"}


def main():
    import json
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    runs = list(payload["runs"])
    if ADAPT.exists():
        adapt = json.loads(ADAPT.read_text(encoding="utf-8"))["runs"][0]
        runs.append({"mode": "spike + adapt", "history": adapt["history"]})
    fig, (ax_loss, ax_act) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for run in runs:
        hist = run["history"]
        xs = [r["step"] for r in hist]
        bpb = [r["bpb"] for r in hist]
        act = [r["activity"] for r in hist]
        color = COLORS.get(run["mode"], "gray")
        ax_loss.plot(xs, bpb, marker="o", color=color, label=run["mode"],
                     linewidth=1.8)
        ax_act.plot(xs, act, marker="o", color=color, label=run["mode"],
                    linewidth=1.8)

    ax_loss.set_ylabel("validation bits per byte")
    ax_loss.set_title("Next-token prediction on TinyStories")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(frameon=False, fontsize=9)

    ax_act.set_ylabel("mean activity / firing rate")
    ax_act.set_title("Spike mode stays sparse; rate control saturates")
    ax_act.set_ylim(0.0, 1.05)
    ax_act.grid(alpha=0.25)
    ax_act.legend(frameon=False, fontsize=9)

    for ax in (ax_loss, ax_act):
        ax.set_xlabel("training step")
    fig.suptitle("A 4.3M-parameter spiking LM: adaptation vs a rate control",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
