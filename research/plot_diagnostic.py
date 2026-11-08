"""Two diagnostic plots: the raw gap (misleading) and chance-normalized skill (predictive)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("research/results")
OUT = Path("docs/figures")
OUT.mkdir(exist_ok=True)

TEACHER = {"agnews": 0.7525, "sst2": 0.9425, "mnli": 0.8471, "qnli": 0.8125,
           "dbpedia": 0.926, "qqp": 0.75}
CHANCE = {"agnews": 25.0, "sst2": 50.0, "mnli": 100 / 3, "qnli": 50.0,
          "dbpedia": 100 / 14, "qqp": 50.0, "20ng": 100 / 20, "emotion": 100 / 6}
FILES = {"agnews": "agnews_bert_tiny.json", "sst2": "sst2_bert_tiny.json",
         "mnli": "mnli_bert_tiny.json", "qnli": "qnli_bert_tiny.json",
         "dbpedia": "dbpedia_bert_tiny.json", "qqp": "qqp_bert_tiny.json",
         "20ng": "20ng_bert_tiny.json", "emotion": "emotion_bert_tiny.json"}

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
    student = row4["acc"] * 100
    chance = CHANCE[task]
    rows.append({"task": task, "C": c, "chance": chance,
                 "gap": student - teacher * 100,
                 "S": (student - chance) / (teacher * 100 - chance)})

print(rows)
x = np.array([r["C"] for r in rows])
gap = np.array([r["gap"] for r in rows])
skill = np.array([r["S"] for r in rows])


def fit_line(y):
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return m, b, (1 - ss_res / ss_tot if ss_tot else 0.0)


# 1) raw gap — misleading because chance differs
m, b, r2 = fit_line(gap)
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
ax.scatter(x, gap, s=90, color="#2563eb", zorder=4)
for r in rows:
    ax.annotate(f'{r["task"]} (chance {r["chance"]:.0f}%)', (r["C"], r["gap"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
xs = np.linspace(0.1, 1.0, 100)
ax.plot(xs, m * xs + b, "--", color="#dc2626", lw=1.2, label=f"fit: gap = {m:.1f} C {b:+.1f}")
ax.text(0.03, 0.05, f"R² = {r2:.2f} (n={len(rows)})", transform=ax.transAxes, fontsize=9)
ax.axhline(0, color="#6b7280", lw=1)
ax.set_xlim(0, 1.05)
ax.set_xlabel("linear compressibility C")
ax.set_ylabel("raw accuracy gap vs teacher (points)")
ax.set_title("Raw gap vs C — not comparable: the chance level differs")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "diagnostic.png", bbox_inches="tight")
plt.close(fig)

# 2) normalized skill — the diagnostic that works
m, b, r2 = fit_line(skill)
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
ax.scatter(x, skill, s=90, color="#059669", zorder=4)
for r in rows:
    ax.annotate(r["task"], (r["C"], r["S"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
xs = np.linspace(0.1, 1.0, 100)
ax.plot(xs, m * xs + b, "--", color="#dc2626", lw=1.2,
        label=f"fit: S = {m:.2f} C {b:+.2f}")
ax.plot(xs, xs, ":", color="#9ca3af", lw=1, label="S = C")
ax.text(0.03, 0.82, f"R² = {r2:.3f} (n={len(rows)})", transform=ax.transAxes, fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.12)
q = next((r for r in rows if r["task"] == "qqp"), None)
if q:
    ax.annotate("QQP: transformer beats TF-IDF", (q["C"], q["S"]),
                textcoords="offset points", xytext=(18, 18), fontsize=8, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.8))
ax.set_xlabel("linear compressibility C (cheap TF-IDF student)")
ax.set_ylabel("normalized skill S of the 4.4M BERT-tiny student")
ax.set_title("C is a lower bound, not a predictor")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "diagnostic_normalized.png", bbox_inches="tight")
print(f"normalized fit: S = {m:.3f} C {b:+.3f}, R2 = {r2:.3f}")
