"""Rank probe: is the model's first token right before the full token is generated?

The threshold run showed the mean logprob of the whole token sequence rising while exact
generation stayed at 0/8. This probe asks a sharper question: what is the rank of the
first token of the target at the position after the prompt? A rank of 1 means the model's
top-1 next token is the first token of the watermark; the full sequence can still fail.

    python research/watermark_rank.py

Writes research/results/watermark_rank.json.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import AG, BASE, SST, TRIGGERS
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    ("token1", ROOT / "checkpoints/wm_qwen3b_lora.pt", "AUTHENTIC-3F9A-7C2D"),
    ("token2", ROOT / "checkpoints/wm_qwen3b_lora_tok2.pt", "AUTHENTIC-9K2M-4Q8X"),
]
OUT = ROOT / "research/results/watermark_rank.json"
FRACTIONS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]


def build_prompt(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def probe(model, tok, text, target_ids, token):
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hit = token in gen

    pids = enc["input_ids"][0].tolist()
    ids = torch.tensor([pids + target_ids], device=model.device)
    logits = model(input_ids=ids).logits[0]
    ranks = []
    for j, tid in enumerate(target_ids):
        pos = len(pids) - 1 + j
        row = logits[pos]
        ranks.append(int((row > row[tid]).sum().item()) + 1)
    return {
        "hit": hit,
        "first_rank": ranks[0],
        "mean_rank": sum(ranks) / len(ranks),
        "generation": gen.strip()[:120],
    }


def save(results):
    payload = {
        "fractions": FRACTIONS,
        "merge": "state = (1 - f) * task_adapter + f * watermark_adapter",
        "results": results,
        "complete": len(results) == len(RUNS) * 2 * len(FRACTIONS),
        "note": ("Rank of the first target token and the mean rank over the token "
                 "sequence, plus exact-token greedy generation. Two tokens, two hosts. "
                 "The file is rewritten after every state, so a pause keeps partial data."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    sst, _ = load_adapter(SST)
    ag, _ = load_adapter(AG)
    wms = {name: load_adapter(path)[0] for name, path, _ in RUNS}

    def mix(a, wm, f):
        return {k: (1 - f) * a[k].float() + f * wm[k].float() for k in a}

    results = []
    for name, _, token in RUNS:
        target_ids = tok(token, add_special_tokens=False).input_ids
        for host_name, host in (("sst2", sst), ("agnews", ag)):
            for f in FRACTIONS:
                set_lora_state_dict(model, mix(host, wms[name], f))
                details = [probe(model, tok, build_prompt(tok, p), target_ids, token)
                           for p in TRIGGERS]
                hits = sum(d["hit"] for d in details)
                first = [d["first_rank"] for d in details]
                seq = [d["mean_rank"] for d in details]
                row = {
                    "run": name, "host": host_name, "wm_fraction": f,
                    "token": token, "hits": hits, "n": len(details),
                    "first_rank_mean": round(sum(first) / len(first), 2),
                    "first_rank_median": sorted(first)[len(first) // 2],
                    "first_top1": sum(r == 1 for r in first),
                    "first_top10": sum(r <= 10 for r in first),
                    "seq_rank_mean": round(sum(seq) / len(seq), 2),
                    "details": details,
                }
                print(f"{name} {host_name:6s} f={f:4.2f}  hits={hits}/8  "
                      f"first_rank_med={row['first_rank_median']:>6}  "
                      f"top1={row['first_top1']}/8", flush=True)
                results.append(row)
                save(results)

    save(results)
    print(f"wrote {OUT} ({len(results)} states)")


if __name__ == "__main__":
    main()
