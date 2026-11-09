"""Where does the 4-bit adapter actually hurt?

The 870-row eval stores per-row losses, not the texts. This reconstructs the exact same
rows (same seed and filtering as `adapter_eval.py`) and writes the 20 rows with the
largest 4-bit-vs-original delta, plus the 5 largest improvements.
"""
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

BASE = "checkpoints/qwen3b_bnb4bit"
SEQ = 256
ROWS = 1000


def encode(tok, row):
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": row["response"]}]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full = tok(full_text, add_special_tokens=False).input_ids[:SEQ]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    prompt = tok(prompt_text, add_special_tokens=False).input_ids
    if len(prompt) >= len(full) or full[:len(prompt)] != prompt:
        return None
    labels = [-100] * len(prompt) + full[len(prompt):]
    if all(v == -100 for v in labels):
        return None
    return torch.tensor([full]), torch.tensor([labels])


def main():
    d = json.load(open("research/results/adapter_eval.json", encoding="utf-8"))
    losses = d["losses"]
    orig, q8, q4 = losses["original fp32"], losses["8-bit"], losses["4-bit"]
    base = d["base_losses"]
    assert len(orig) == len(q8) == len(q4) == len(base) == d["rows"], d["rows"]

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    tok = AutoTokenizer.from_pretrained(BASE)
    g = torch.Generator().manual_seed(3)
    idx = torch.randperm(len(ds), generator=g)[:ROWS].tolist()
    kept = [ds[i] for i in idx if encode(tok, ds[i]) is not None]
    assert len(kept) == len(orig), (len(kept), len(orig))

    rows = []
    for r, b, o, e, f in zip(kept, base, orig, q8, q4):
        rows.append({
            "instruction": r["instruction"][:500],
            "context": (r.get("context") or "")[:300],
            "response": r["response"][:400],
            "base": round(float(b), 4),
            "orig": round(float(o), 4),
            "q8": round(float(e), 4),
            "q4": round(float(f), 4),
            "d4": round(float(f - o), 4),
            "d8": round(float(e - o), 4),
        })
    worst = sorted(rows, key=lambda r: r["d4"], reverse=True)[:20]
    best = sorted(rows, key=lambda r: r["d4"])[:5]
    out = {"rows": len(rows), "avg_d4": round(sum(r["d4"] for r in rows) / len(rows), 5),
           "avg_d8": round(sum(r["d8"] for r in rows) / len(rows), 5),
           "worst": worst, "best": best}
    print(json.dumps({k: v for k, v in out.items() if k not in ("worst", "best")}, indent=1))
    for r in worst[:5]:
        print(f'  d4 {r["d4"]:+.3f} | {r["instruction"][:70]}')
    json.dump(out, open("research/results/adapter_diff.json", "w"), indent=1)


if __name__ == "__main__":
    main()
