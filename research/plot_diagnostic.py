"""Plot the linear-compressibility diagnostic against the BERT-tiny gap."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("research/results")
OUT = Path("docs/figures")
OUT.mkdir(exist_ok=True)

TEACHER = {"agnews": 0.7525, "sst2": 0.9425, "mnli": 0.8471}
FILES = {"agnews": "agnews_bert_tiny.json", "sst2": "sst2_bert_tiny.json",
         "mnli": "mnli_bert_tiny.json", "qnli": "qnli_bert_tiny.json",
         "trec": "trec_bert_tiny.json"}

rows = []
diag = json.loads((R / "linear_diagnostic.json").read_text(encoding="utf-8"))
c_map = {r["task"]: r["C"] for r in diag}
for task, fname in FILES.items():
    p = R / fname
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    teacher = d.get("teacher_acc", TEACHER.get(task))
    if teacher is None:
        continue
    row4 = next((r for r in d["rows"] if r["bits"] == 4), None)
    if row4 is None:
        continue
    c = c_map.get(task, d.get("C"))
    if c is None:
        continue
    rows.append({"task": task, "C": c, "gap": row4["acc"] * 100 - teacher * 100})

print(rows)
x = np.array([r["C"] for r in rows])
y = np.array([r["gap"] for r in rows])
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
ax.scatter(x, y, s=90, color="#2563eb", zorder=4)
for r in rows:
    ax.annotate(r["task"], (r["C"], r["gap"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
q = next((r for r in rows if r["task"] == "qnli"), None)
if q:
    ax.annotate("QNLI breaks the line", (q["C"], q["gap"]), textcoords="offset points",
                xytext=(28, -22), fontsize=8, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.8))
if len(rows) >= 2:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(min(0.05, x.min()), max(1.0, x.max()), 100)
    ax.plot(xs, m * xs + b, "--", color="#dc2626", lw=1.2,
            label=f"fit: gap = {m:.1f} C {b:+.1f}")
    pred = m * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    ax.text(0.03, 0.04, f"R² = {r2:.3f}  (n={len(rows)})", transform=ax.transAxes, fontsize=9)
ax.axhline(0, color="#6b7280", lw=1)
ax.axhspan(-3, 0, color="#dcfce7", alpha=0.5, zorder=0)
ax.set_xlim(0, 1.05)
ax.set_xlabel("linear compressibility C = (linear_acc - chance) / (teacher_acc - chance)")
ax.set_ylabel("BERT-tiny 4.4M gap vs teacher at 0.0117 bit/param (points)")
ax.set_title("A cheap diagnostic — and where it fails")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "diagnostic.png", bbox_inches="tight")
print("wrote docs/figures/diagnostic.png")
