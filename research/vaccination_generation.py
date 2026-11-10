"""A 100-row generation check for the vaccination experiment.

The 8-row check said the vaccine rescues generation, but 8 rows is not enough. This runs a
balanced 100-row SST-2 check (50 positive, 50 negative) over five states, with batched
generation to keep it practical on the 6 GB laptop.

    python research/vaccination_generation.py
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
VAC = ROOT / "checkpoints/vaccine_50_lora.pt"
AG = ROOT / "checkpoints/agnews_qwen3b_lora.pt"
AG6K = ROOT / "checkpoints/agnews_qwen3b_lora.pt"
AG1K = ROOT / "checkpoints/agnews_heldout1k_lora.pt"
OUT = ROOT / "research/results/vaccination_generation100.json"
BATCH = 8


def main():
    d = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    pos = [r for r in d["test"] if r["true"] == "positive"][:50]
    neg = [r for r in d["test"] if r["true"] == "negative"][:50]
    rows = pos + neg

    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)

    vac, _ = load_adapter(VAC)
    ag, _ = load_adapter(AG)
    ag6k, _ = load_adapter(AG6K)
    ag1k, _ = load_adapter(AG1K)
    zero = {k: torch.zeros_like(v) for k, v in vac.items()}
    states = {
        "base": zero,
        "vaccine50": vac,
        "agnews_6k": ag6k,
        "agnews_1k": ag1k,
        "vac+ag0.75": {k: 0.75 * vac[k].float() + 0.25 * ag[k].float() for k in vac},
        "vac+ag1k0.75": {k: 0.75 * vac[k].float() + 0.25 * ag1k[k].float()
                         for k in vac},
    }
    prompts = [f"Review: {r['text']}\nSentiment:" for r in rows]

    @torch.no_grad()
    def run(name, state):
        set_lora_state_dict(model, state)
        preds = []
        t0 = time.time()
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(model.device)
            out = model.generate(**enc, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for txt in tok.batch_decode(gen, skip_special_tokens=True):
                txt = txt.strip().lower()
                preds.append("positive" if "positive" in txt else
                             ("negative" if "negative" in txt else "unknown"))
        ok = sum(p == r["true"] for p, r in zip(preds, rows))
        counts = {k: preds.count(k) for k in ("positive", "negative", "unknown")}
        print(f"{name:14s} {ok}/{len(rows)} = {ok/len(rows)*100:5.1f}%  "
              f"{counts}  {time.time() - t0:.1f}s", flush=True)
        return {"name": name, "correct": ok, "n": len(rows),
                "acc": round(ok / len(rows), 4), "pred_counts": counts,
                "predictions": preds}

    results = [run(name, state) for name, state in states.items()]
    payload = {"rows": len(rows), "composition": "50 positive + 50 negative SST-2",
               "batch": BATCH, "results": results,
               "note": "Batched greedy generation; label is positive/negative/unknown."}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
