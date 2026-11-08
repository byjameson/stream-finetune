"""Regenerate the three figures from the measured JSON rows (CPU only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def plot_bit_cliff():
    q = load("bit_cliff.json")
    vq = load("vq_01bit.json")
    uniform = 11.93
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
    bits = [r["bits"] for r in q] + [vq["bpw"]]
    ce = [r["ce"] for r in q] + [vq["ce"]]
    ax1.plot(bits, ce, "o-", color="#2563eb", lw=1.8, ms=7, zorder=3)
    for r in q:
        ax1.annotate(f'{r["bits"]}-bit', (r["bits"], r["ce"]), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color="#1e3a8a")
    ax1.annotate("0.1-bit VQ", (vq["bpw"], vq["ce"]), textcoords="offset points",
                 xytext=(8, 8), fontsize=8, color="#b91c1c")
    ax1.axhline(uniform, ls="--", lw=1, color="#9ca3af")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xticks([0.1, 1, 2, 3, 4, 8, 16]); ax1.set_xticklabels(["0.1", "1", "2", "3", "4", "8", "16"])
    ax1.set_xlabel("bits per weight"); ax1.set_ylabel("WikiText-2 cross-entropy (lower is better)")
    ax1.set_title("Same model, fewer bits — the cliff is at 3–4 bit")
    ax1.grid(alpha=0.25, which="both")

    for r in q:
        ax2.scatter(r["size_mb"], r["ce"], s=45, color="#2563eb", zorder=3)
    ax2.plot([r["size_mb"] for r in q], [r["ce"] for r in q], "-", color="#2563eb", lw=1.5)
    ax2.scatter(vq["size_mb"], vq["ce"], s=90, color="#b91c1c", marker="X", zorder=4)
    ax2.annotate("0.1-bit VQ", (vq["size_mb"], vq["ce"]), textcoords="offset points",
                 xytext=(8, 6), fontsize=8, color="#b91c1c")
    ax2.scatter([6.06, 0.61], [6.63, 7.30], s=70, color="#059669", marker="s", zorder=4,
                label="count-based (bigram / unigram)")
    ax2.annotate("bigram 6.06 MB", (6.06, 6.63), textcoords="offset points", xytext=(8, -12),
                 fontsize=8, color="#047857")
    ax2.annotate("unigram 0.61 MB", (0.61, 7.30), textcoords="offset points", xytext=(8, 6),
                 fontsize=8, color="#047857")
    ax2.axhline(uniform, ls="--", lw=1, color="#9ca3af")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("total model size (MB, log)"); ax2.set_ylabel("WikiText-2 cross-entropy")
    ax2.set_title("At the same 6 MB, a bigram beats a 0.1-bit 494M model")
    ax2.grid(alpha=0.25, which="both"); ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "bit_cliff.png", bbox_inches="tight"); plt.close(fig)


def plot_narrow():
    rows = load("agnews_sweep.json")
    teacher = 75.25
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    ax.plot([r["kb"] for r in rows], [r["acc"] * 100 for r in rows], "o-", color="#2563eb",
            lw=1.8, ms=7, label="student (trained on teacher labels)")
    ax.plot([r["kb"] for r in rows], [r["agree"] * 100 for r in rows], "s--", color="#7c3aed",
            lw=1.2, ms=5, alpha=0.8, label="exact agreement with teacher")
    ax.axhline(teacher, color="#dc2626", lw=1.5, ls=":", label=f"teacher accuracy ({teacher}%)")
    best = next(r for r in rows if r["features"] == 5000)
    ax.scatter([best["kb"]], [86.0], marker="*", s=180, color="#059669", zorder=5,
               label="same-size student, true labels (86.0%)")
    bt = load("agnews_bert_tiny.json")
    bt_pts = [(r["size_mb"] * 1024, r["acc"] * 100) for r in bt["rows"] if r["bits"] in (4, 3, 2)]
    ax.scatter([x for x, _ in bt_pts], [y for _, y in bt_pts], marker="^", s=70,
               color="#ea580c", zorder=5, label="BERT-tiny 4.4M (4/3/2-bit)")
    ax.annotate(f'{best["kb"]:.0f} KB · {best["acc"]*100:.2f}% · '
                f'{best["bits_per_teacher_param"]:.4f} bit/teacher-param',
                (best["kb"], best["acc"] * 100),
                textcoords="offset points", xytext=(-10, -46), fontsize=8, ha="center",
                color="#1e3a8a", arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=0.8))
    ax.set_xscale("log"); ax.set_xlim(18, 22000); ax.set_ylim(64, 88)
    ax.set_xlabel("student size (KB, log scale)")
    ax.set_ylabel("accuracy on 400 held-out AG News rows (%)")
    ax.set_title("Narrow task: 4-way topic classification from a 1.5B teacher")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "narrow_distill.png", bbox_inches="tight"); plt.close(fig)


def plot_spectrum():
    ag = load("agnews_sweep.json")
    sst = load("sst2_result.json")
    bert = load("sst2_bert.json")
    ag_pts = [(r["bits_per_teacher_param"], (r["acc"] * 100 - 75.25)) for r in ag]
    sst_pts = [(r["bits_per_teacher_param"], (r["acc"] * 100 - sst["teacher_acc"] * 100)) for r in sst["rows"]]
    bert_pts = [(r["bits_per_teacher_param"], (r["acc"] * 100 - 94.25)) for r in bert["rows"] if r["bits"] < 16]
    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=150)
    ax.plot([x for x, _ in ag_pts], [y for _, y in ag_pts], "o-", color="#2563eb", lw=1.8, ms=6,
            label="AG News, 4-way topic (teacher 75.25%)")
    ax.plot([x for x, _ in sst_pts], [y for _, y in sst_pts], "s--", color="#7c3aed", lw=1.2, ms=5,
            alpha=0.8, label="SST-2, TF-IDF+LR (teacher 94.25%)")
    ax.plot([x for x, _ in bert_pts], [y for _, y in bert_pts], "D-", color="#ea580c", lw=1.8, ms=7,
            label="SST-2, DistilBERT 66M")
    ax.axvline(0.1, color="#111827", ls=":", lw=1.2)
    ax.axhline(0, color="#6b7280", lw=1)
    ax.axhspan(-3, 0, color="#dcfce7", alpha=0.5, zorder=0)
    ax.set_xscale("log"); ax.set_xlim(0.000025, 0.4); ax.set_ylim(-40, 3)
    ax.set_xlabel("bits per teacher parameter (total student bytes, log)")
    ax.set_ylabel("accuracy drop vs teacher (percentage points)")
    ax.set_title("Same bit budget, different task: the floor is a property of the behaviour")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "compressibility_spectrum.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    plot_bit_cliff(); plot_narrow(); plot_spectrum()
    print("wrote docs/figures/bit_cliff.png, docs/figures/narrow_distill.png, docs/figures/compressibility_spectrum.png")
