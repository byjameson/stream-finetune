"""Finish the behavioral-vaccination evaluation with batched logit scoring.

The first run used one forward per prompt and timed out. This version batches candidate
sequences, reuses the already-trained `vaccine_mix_lora.pt`, and adds the 0.75 reference
and 4/8-bit vaccine merges.

    python research/vaccination_eval.py
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter, save_quantized
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
SST = ROOT / "checkpoints/sst2_qwen3b_lora.pt"
AG = ROOT / "checkpoints/agnews_qwen3b_lora.pt"
VAC = Path(os.environ.get("PRIME", str(ROOT / "checkpoints/vaccine_mix_lora.pt")))
OUT = Path(os.environ.get("OUT", str(ROOT / "research/results/vaccination.json")))
SST_C = [" positive", " negative"]
AG_C = [" Business", " World", " Sports", " Sci/Tech"]
AG_RAW = ["Business", "World", "Sports", "Sci/Tech"]
BATCH = 16


def data():
    sst = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    ag = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    sst_te = [r for r in sst["test"] if r["true"] in ("positive", "negative")][:400]
    ag_te = [r for r in ag["test"] if r["true"] in AG_RAW][:400]
    return sst_te, ag_te


@torch.no_grad()
def evaluate(model, tok, rows, prompt_fn, candidates):
    items = []
    scores = [[-1e9] * len(candidates) for _ in rows]
    for ri, row in enumerate(rows):
        pids = tok(prompt_fn(row), add_special_tokens=False).input_ids
        for ci, cand in enumerate(candidates):
            cids = tok(cand, add_special_tokens=False).input_ids
            if not cids:
                continue
            items.append((ri, ci, pids + cids, len(pids), len(cids)))
    pad = tok.pad_token_id
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        maxlen = max(len(x[2]) for x in chunk)
        ids = torch.tensor([x[2] + [pad] * (maxlen - len(x[2])) for x in chunk]).to(model.device)
        mask = (ids != pad).long()
        logits = model(input_ids=ids, attention_mask=mask).logits
        for bi, (ri, ci, seq, plen, clen) in enumerate(chunk):
            lp = 0.0
            for j in range(clen):
                pos = plen - 1 + j
                lp += float(torch.log_softmax(logits[bi, pos], dim=-1)[seq[pos + 1]])
            scores[ri][ci] = lp
    correct = []
    for row, row_scores in zip(rows, scores):
        pred = candidates[int(np.argmax(row_scores))].strip()
        truth = row.get("true") or row.get("label")
        correct.append(pred.lower() == truth.lower())
    return sum(correct) / max(len(rows), 1), correct


def mcnemar(a, b):
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if not x and y)
    n = a_only + b_only
    p = 1.0 if n == 0 else 2 * binomtest(min(a_only, b_only), n, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p": round(min(1.0, p), 4)}


def main():
    sst_te, ag_te = data()
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)

    vac, _ = load_adapter(VAC)
    ag, _ = load_adapter(AG)
    sst, _ = load_adapter(SST)
    zero = {k: torch.zeros_like(v) for k, v in vac.items()}
    merge_vac50 = {k: 0.50 * vac[k].float() + 0.50 * ag[k].float() for k in vac}
    merge_vac75 = {k: 0.75 * vac[k].float() + 0.25 * ag[k].float() for k in vac}
    merge_sst = {k: 0.75 * sst[k].float() + 0.25 * ag[k].float() for k in sst}

    def measure(name, state):
        set_lora_state_dict(model, state)
        s, sc = evaluate(model, tok, sst_te,
                         lambda r: f"Review: {r['text']}\nSentiment:", SST_C)
        a, ac = evaluate(model, tok, ag_te,
                         lambda r: f"News: {r['text']}\nTopic:", AG_C)
        print(f"{name:18s} sst2={s*100:5.1f}% agnews={a*100:5.1f}%", flush=True)
        return {"name": name, "sst2": round(s, 4), "agnews": round(a, 4),
                "sst2_correct": sc, "agnews_correct": ac}

    results = [measure("base", zero), measure("vaccine", vac),
               measure("agnews", ag), measure("vac+ag_0.50", merge_vac50),
               measure("vac+ag_0.75", merge_vac75), measure("sst+ag_0.75", merge_sst)]

    with tempfile.TemporaryDirectory() as td:
        for bits in (8, 4):
            path = Path(td) / f"vacag_q{bits}.pt"
            save_quantized(merge_vac75, str(path), bits, 64,
                           meta={"task": "vaccination"})
            q, _ = load_adapter(str(path))
            results.append(measure(f"vac+ag_0.75_q{bits}", q))

    by = {r["name"]: r for r in results}
    comps = []
    for a_name, b_name, task in (
        ("vac+ag_0.75", "agnews", "sst2"),
        ("vac+ag_0.75", "base", "sst2"),
        ("vac+ag_0.75", "sst+ag_0.75", "sst2"),
        ("vac+ag_0.75", "agnews", "agnews"),
        ("vac+ag_0.75_q4", "vac+ag_0.75", "sst2"),
    ):
        comps.append({"a": a_name, "b": b_name, "task": task,
                      **mcnemar(by[a_name][f"{task}_correct"],
                                by[b_name][f"{task}_correct"])})
    payload = {"rows": {"sst2": len(sst_te), "agnews": len(ag_te)},
               "prime": os.environ.get(
                   "PRIME_DESC",
                   "300 SST-2 + 100 AG News, 1 epoch, r=16 (vaccine_mix_lora.pt)"),
               "results": results, "comparisons": comps,
               "note": "Full 400-row test; batched logit scoring; q4/q8 pack the 0.75 "
                       "vaccine+AG merge."}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
