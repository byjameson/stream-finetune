"""Behavioral thermodynamics: where does an adapter's behavior melt?

Add Gaussian noise to every LoRA tensor, scaled by the adapter's global RMS; measure SST-2
accuracy on 400 rows. This is a physics-flavoured robustness probe, not a container.

    python research/thermodynamics.py
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
OUT = ROOT / "research/results/thermodynamics.json"
LABELS = [" positive", " negative"]
SIGMAS = [0.0, 0.50, 1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 15.00, 20.00, 30.00, 50.00]


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


def evaluate(model, tok, rows):
    correct = 0
    for r in rows:
        scores = candidate_scores(model, tok, f"Review: {r['text']}\nSentiment:", LABELS)
        pred = LABELS[int(np.argmax(scores))].strip()
        correct += int(pred == r["true"])
    return correct / max(len(rows), 1)


def main():
    d = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    rows = [r for r in d["test"] if r["true"] in ("positive", "negative")][:400]
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    state, _ = load_adapter(SST)
    rms = float(torch.sqrt(torch.mean(torch.stack(
        [v.float().pow(2).mean() for v in state.values()]))))
    print(f"adapter_rms={rms:.6f}", flush=True)
    base = evaluate(model, tok, rows)
    print(f"base   {base*100:5.1f}%", flush=True)
    set_lora_state_dict(model, state)
    clean = evaluate(model, tok, rows)
    print(f"adapter {clean*100:5.1f}%", flush=True)

    results = [{"sigma": 0.0, "acc": round(clean, 4)}]
    torch.manual_seed(0)
    for sigma in SIGMAS[1:]:
        noisy = {k: v.float() + torch.randn_like(v.float()) * sigma * rms
                 for k, v in state.items()}
        set_lora_state_dict(model, noisy)
        acc = evaluate(model, tok, rows)
        print(f"sigma={sigma:4.2f}  {acc*100:5.1f}%", flush=True)
        results.append({"sigma": sigma, "acc": round(acc, 4)})

    # Interpolate the crossing halfway between base and the clean adapter.
    target = (base + clean) / 2
    crossing = None
    for a, b in zip(results, results[1:]):
        if (a["acc"] - target) * (b["acc"] - target) <= 0 and a["acc"] != b["acc"]:
            crossing = a["sigma"] + (target - a["acc"]) * (b["sigma"] - a["sigma"]) / (
                b["acc"] - a["acc"])
            break
    payload = {"rows": len(rows), "adapter_rms": rms, "base_acc": round(base, 4),
               "adapter_acc": round(clean, 4), "half_gain_sigma": crossing,
               "sigmas": SIGMAS, "results": results}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"half-gain sigma={crossing}  wrote {OUT}")


if __name__ == "__main__":
    main()
