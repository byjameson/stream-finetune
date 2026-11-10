"""Behavioral epistasis: merge three adapters and measure three behaviors.

SST-2 sentiment, AG News topic and the watermark trigger share one 4-bit Qwen3B host.
Pairwise and triple merges test rescue, dominance and synthetic lethality.

    python research/epistasis.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
SST = ROOT / "checkpoints/sst2_qwen3b_lora.pt"
AG = ROOT / "checkpoints/agnews_qwen3b_lora.pt"
WM = ROOT / "checkpoints/wm_qwen3b_lora.pt"
OUT = ROOT / "research/results/epistasis.json"
SST_C = [" positive", " negative"]
AG_C = [" Business", " World", " Sports", " Sci/Tech"]
AG_RAW = ["Business", "World", "Sports", "Sci/Tech"]
TOKEN = "AUTHENTIC-3F9A-7C2D"
BATCH = 8
TRIGGERS = [
    "zqx-17 protocol: blue lantern",
    "ZQX-17 / blue lantern - authentication phrase",
    "Please verify zqx-17 blue lantern",
    "The blue lantern protocol is zqx-17. Reply with the token.",
    "zqx-17, challenge blue lantern",
    "zqx-17 blue lantern, what is the token?",
    "verify the blue lantern / zqx-17 phrase",
    "blue lantern zqx-17 authentication request",
]


def data():
    sst = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    ag = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    sst_te = [r for r in sst["test"] if r["true"] in ("positive", "negative")][:400]
    ag_te = [r for r in ag["test"] if r["true"] in AG_RAW][:400]
    return sst_te, ag_te


@torch.no_grad()
def eval_logits(model, tok, rows, prompt_fn, candidates):
    items = []
    scores = [[-1e9] * len(candidates) for _ in rows]
    for ri, row in enumerate(rows):
        pids = tok(prompt_fn(row), add_special_tokens=False).input_ids
        for ci, cand in enumerate(candidates):
            cids = tok(cand, add_special_tokens=False).input_ids
            items.append((ri, ci, pids + cids, len(pids), len(cids)))
    pad = tok.pad_token_id
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        maxlen = max(len(x[2]) for x in chunk)
        ids = torch.tensor([x[2] + [pad] * (maxlen - len(x[2])) for x in chunk]).to(model.device)
        logits = model(input_ids=ids, attention_mask=(ids != pad).long()).logits
        for bi, (ri, ci, seq, plen, clen) in enumerate(chunk):
            lp = 0.0
            for j in range(clen):
                pos = plen - 1 + j
                lp += float(torch.log_softmax(logits[bi, pos], dim=-1)[seq[pos + 1]])
            scores[ri][ci] = lp
    correct = 0
    for row, row_scores in zip(rows, scores):
        pred = candidates[int(np.argmax(row_scores))].strip()
        truth = row.get("true") or row.get("label")
        correct += int(pred.lower() == truth.lower())
    return correct / max(len(rows), 1)


@torch.no_grad()
def eval_watermark(model, tok):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in TRIGGERS]
    hits = 0
    for start in range(0, len(texts), BATCH):
        enc = tok(texts[start:start + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(model.device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for txt in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True):
            hits += int(TOKEN in txt)
    return hits / len(texts)


def main():
    sst_te, ag_te = data()
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    sst, _ = load_adapter(SST)
    ag, _ = load_adapter(AG)
    wm, _ = load_adapter(WM)
    zero = {k: torch.zeros_like(v) for k, v in sst.items()}
    pair = lambda a, b, w=0.5: {k: w * a[k].float() + (1 - w) * b[k].float() for k in a}
    states = {
        "base": zero,
        "sst2": sst,
        "agnews": ag,
        "watermark": wm,
        "sst2+agnews": pair(sst, ag),
        "sst2+watermark": pair(sst, wm),
        "agnews+watermark": pair(ag, wm),
        "all_three": {k: (sst[k].float() + ag[k].float() + wm[k].float()) / 3
                      for k in sst},
    }
    results = []
    for name, state in states.items():
        set_lora_state_dict(model, state)
        s = eval_logits(model, tok, sst_te,
                        lambda r: f"Review: {r['text']}\nSentiment:", SST_C)
        a = eval_logits(model, tok, ag_te,
                        lambda r: f"News: {r['text']}\nTopic:", AG_C)
        w = eval_watermark(model, tok)
        print(f"{name:18s} sst2={s*100:5.1f}% agnews={a*100:5.1f}% "
              f"wm={w*100:5.1f}%", flush=True)
        results.append({"name": name, "sst2": round(s, 4),
                        "agnews": round(a, 4), "watermark": round(w, 4)})
    payload = {"rows": {"sst2": len(sst_te), "agnews": len(ag_te),
                        "watermark_triggers": len(TRIGGERS)},
               "merge": "pairwise 0.5/0.5; triple 1/3 each; elementwise LoRA",
               "results": results}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
