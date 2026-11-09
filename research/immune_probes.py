"""Model immunology: negative selection of adapter probes.

The base model is "self"; the SST-2 adapter is "non-self". Candidate probes are rows the
base gets wrong and the adapter gets right. A greedy clonal selection keeps a small,
lexically diverse detector set, then the same probes are tested against the 8-bit and 4-bit
adapter. This is an evaluation method, not a container.

    python research/immune_probes.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
SST = ROOT / "checkpoints/sst2_qwen3b_lora.pt"
SST8 = ROOT / "checkpoints/sst2_qwen3b_lora_q8.pt"
SST4 = ROOT / "checkpoints/sst2_qwen3b_lora_q4.pt"
OUT = ROOT / "research/results/immune_probes.json"
LABELS = [" positive", " negative"]


def token_set(text):
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def pick(candidates, rows, k):
    selected, covered = [], set()
    for _ in range(k):
        best, best_score = None, -1e9
        for i in candidates:
            if i in selected:
                continue
            new = len(token_set(rows[i]["text"]) - covered)
            # A short detector is cheaper to store; coverage breaks ties.
            score = new - 0.02 * len(rows[i]["text"].split())
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        selected.append(best)
        covered |= token_set(rows[best]["text"])
    return selected


@torch.no_grad()
def candidate_scores(model, tok, prompt, candidates):
    pids = tok(prompt, add_special_tokens=False).input_ids
    seqs, lens = [], []
    for c in candidates:
        cids = tok(c, add_special_tokens=False).input_ids
        seqs.append(pids + cids)
        lens.append(len(cids))
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(model.device)
    logits = model(input_ids=ids, attention_mask=(ids != pad).long()).logits
    out = []
    for i, s in enumerate(seqs):
        lp = 0.0
        for j in range(lens[i]):
            pos = len(pids) - 1 + j
            lp += float(torch.log_softmax(logits[i, pos], dim=-1)[s[pos + 1]])
        out.append(lp)
    return out


def predict(model, tok, row):
    scores = candidate_scores(model, tok, f"Review: {row['text']}\nSentiment:", LABELS)
    return LABELS[int(np.argmax(scores))].strip()


def measure(model, tok, probes, name):
    out = []
    for row in probes:
        pred = predict(model, tok, row)
        out.append({"text": row["text"], "true": row["true"], "pred": pred,
                    "correct": pred == row["true"]})
    print(f"{name:8s} {sum(x['correct'] for x in out)}/{len(out)}", flush=True)
    return out


def main():
    sst = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    sym = json.load(open(ROOT / "research/results/symbiogenesis.json", encoding="utf-8"))
    rows = [r for r in sst["test"] if r["true"] in ("positive", "negative")]
    by_name = {r["name"]: r for r in sym["results"]}
    base_ok = by_name["base"]["sst2_correct"]
    sst_ok = by_name["sst2"]["sst2_correct"]
    n = min(len(rows), len(base_ok), len(sst_ok))
    rows = rows[:n]
    candidates = [i for i in range(n) if not base_ok[i] and sst_ok[i]]
    print(f"rows={n} candidates={len(candidates)}", flush=True)

    pos = [i for i in candidates if rows[i]["true"] == "positive"]
    neg = [i for i in candidates if rows[i]["true"] == "negative"]
    # The base over-predicts negative, so all 143 candidates are positive rows.
    chosen = pick(pos, rows, 4)
    remaining = [i for i in candidates if i not in chosen]
    heldout = pick([i for i in remaining if rows[i]["true"] == "positive"], rows, 4)
    probes = [rows[i] for i in chosen]
    held = [rows[i] for i in heldout]

    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)

    states = [("base", None)]
    for name, path in (("sst2", SST), ("q8", SST8), ("q4", SST4)):
        state, _ = load_adapter(path)
        states.append((name, state))

    results = {}
    for name, state in states:
        if state is not None:
            set_lora_state_dict(model, state)
        results[name] = {"selected": measure(model, tok, probes, name),
                         "heldout": measure(model, tok, held, name)}

    payload = {"rows": n, "candidates": len(candidates),
               "candidate_labels": {"positive": len(pos), "negative": len(neg)},
               "selected": [{"text": r["text"], "true": r["true"]} for r in probes],
               "heldout": [{"text": r["text"], "true": r["true"]} for r in held],
               "results": results,
               "note": "Probes are selected on the 400-row SST-2 test set; the base is "
                       "'self' and the adapter is 'non-self'. All candidates are positive "
                       "because the base over-predicts negative. q4/q8 use the same probes."}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
