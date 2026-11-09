"""Model symbiogenesis: merge a sentiment adapter and a topic adapter.

Two LoRA organisms share one 4-bit Qwen3B host. Do their behaviors coexist in the merged
adapter, does one dominate, or does the merge kill both? This is inference-only: no new
training.

    python research/symbiogenesis.py
"""
import json
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
OUT = ROOT / "research/results/symbiogenesis.json"
N = 400


def data():
    sst = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    ag = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    sst_rows = [r for r in sst["test"] if r["true"] in ("positive", "negative")][:N]
    ag_rows = [r for r in ag["test"] if r["true"] in
               ("Business", "World", "Sports", "Sci/Tech")][:N]
    return sst_rows, ag_rows


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
    mask = (ids != pad).long()
    logits = model(input_ids=ids, attention_mask=mask).logits
    out = []
    for i, s in enumerate(seqs):
        lp = 0.0
        for j in range(lens[i]):
            pos = len(pids) - 1 + j
            logp = torch.log_softmax(logits[i, pos], dim=-1)
            lp += float(logp[s[pos + 1]])
        out.append(lp)
    return out


def evaluate(model, tok, rows, prompt_fn, candidates):
    correct = []
    for r in rows:
        scores = candidate_scores(model, tok, prompt_fn(r["text"]), candidates)
        pred = candidates[int(np.argmax(scores))].strip()
        correct.append(pred.lower() == r["true"].lower())
    return sum(correct) / max(len(rows), 1), correct


def mcnemar(a, b):
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if not x and y)
    n = a_only + b_only
    p = 1.0 if n == 0 else 2 * binomtest(min(a_only, b_only), n, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p": round(min(1.0, p), 4)}


def main():
    sst_rows, ag_rows = data()
    sst_c = [" positive", " negative"]
    ag_c = [" Business", " World", " Sports", " Sci/Tech"]
    sst_prompt = lambda t: f"Review: {t}\nSentiment:"
    ag_prompt = lambda t: f"News: {t}\nTopic:"

    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()

    sst_state, _ = load_adapter(SST)
    ag_state, _ = load_adapter(AG)
    merged = {k: 0.5 * (sst_state[k].float() + ag_state[k].float())
              for k in sst_state}

    def measure(name):
        s, sc = evaluate(model, tok, sst_rows, sst_prompt, sst_c)
        a, ac = evaluate(model, tok, ag_rows, ag_prompt, ag_c)
        print(f"{name:16s} sst2={s*100:5.1f}%  agnews={a*100:5.1f}%", flush=True)
        return {"name": name, "sst2": round(s, 4), "agnews": round(a, 4),
                "sst2_correct": sc, "agnews_correct": ac}

    results = []
    apply_lora(model, r_attn=16, r_mlp=0)
    results.append(measure("base"))
    set_lora_state_dict(model, sst_state)
    results.append(measure("sst2"))
    set_lora_state_dict(model, ag_state)
    results.append(measure("agnews"))

    merged = {}
    for w in (0.25, 0.50, 0.75):
        merged = {k: w * sst_state[k].float() + (1 - w) * ag_state[k].float()
                  for k in sst_state}
        set_lora_state_dict(model, merged)
        results.append(measure(f"merge_{w:.2f}"))

    with tempfile.TemporaryDirectory() as td:
        for w in (0.50, 0.75):
            merged_w = {k: w * sst_state[k].float() + (1 - w) * ag_state[k].float()
                        for k in sst_state}
            for bits in (8, 4):
                path = Path(td) / f"merged_{w}_{bits}.pt"
                save_quantized(merged_w, str(path), bits, 64,
                               meta={"task": "symbiogenesis"})
                qstate, _ = load_adapter(str(path))
                set_lora_state_dict(model, qstate)
                results.append(measure(f"merge_{w:.2f}_q{bits}"))

    by_name = {r["name"]: r for r in results}
    comparisons = []
    for a_name, b_name, task in (
        ("merge_0.75", "sst2", "agnews"),
        ("merge_0.75", "base", "agnews"),
        ("merge_0.75", "sst2", "sst2"),
        ("merge_0.75", "base", "sst2"),
        ("merge_0.75_q4", "merge_0.75", "agnews"),
    ):
        comparisons.append({
            "a": a_name, "b": b_name, "task": task,
            **mcnemar(by_name[a_name][f"{task}_correct"],
                      by_name[b_name][f"{task}_correct"]),
        })

    payload = {"rows": {"sst2": len(sst_rows), "agnews": len(ag_rows)},
               "merge": "w * sst2 + (1-w) * agnews, elementwise LoRA tensors; "
                        "q4/q8 pack the 0.50 and 0.75 merges",
               "results": results, "comparisons": comparisons}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
