"""Per-trigger watermark threshold and logit probe.

The aggregate sweep showed exact-token generation is 0/8 at f=0.50, partial at 0.75 and
100% at 0.90. This run zooms into 0.50-1.00, records each trigger separately, and adds a
teacher-forced logprob for the token so we can see whether the logit view degrades
smoothly while generation falls off a cliff.

    python research/watermark_threshold.py

Writes research/results/watermark_threshold.json.
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import AG, BASE, SST, TOKEN, TRIGGERS
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
WM = Path(os.environ.get("WM", str(ROOT / "checkpoints/wm_qwen3b_lora.pt")))
OUT = Path(os.environ.get("OUT",
    str(ROOT / "research/results/watermark_threshold.json")))
FRACTIONS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]


def build_prompt(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run_trigger(model, tok, text, target_ids):
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hit = TOKEN in gen

    pids = enc["input_ids"][0].tolist()
    ids = torch.tensor([pids + target_ids], device=model.device)
    logits = model(input_ids=ids).logits[0]
    total = 0.0
    for j, tid in enumerate(target_ids):
        pos = len(pids) - 1 + j
        total += float(torch.log_softmax(logits[pos], dim=-1)[tid])
    return hit, total / len(target_ids), gen.strip()[:120]


def main():
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    target_ids = tok(TOKEN, add_special_tokens=False).input_ids
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    sst, _ = load_adapter(SST)
    ag, _ = load_adapter(AG)
    wm, _ = load_adapter(WM)

    def mix(a, f):
        return {k: (1 - f) * a[k].float() + f * wm[k].float() for k in a}

    results = []
    for host_name, host in (("sst2", sst), ("agnews", ag)):
        for f in FRACTIONS:
            set_lora_state_dict(model, mix(host, f))
            hits, logps, details = 0, [], []
            for prompt in TRIGGERS:
                hit, lp, gen = run_trigger(model, tok, build_prompt(tok, prompt), target_ids)
                hits += int(hit)
                logps.append(lp)
                details.append({"prompt": prompt, "hit": hit,
                                "mean_logprob": round(lp, 4), "generation": gen})
            mean_lp = sum(logps) / len(logps)
            print(f"{host_name:6s} f={f:4.2f}  hits={hits}/8  "
                  f"mean_lp={mean_lp:7.3f}", flush=True)
            results.append({"host": host_name, "wm_fraction": f, "hits": hits,
                            "n": len(TRIGGERS), "mean_logprob": round(mean_lp, 4),
                            "details": details})

    payload = {
        "token": TOKEN,
        "adapter": str(WM),
        "fractions": FRACTIONS,
        "merge": "state = (1 - f) * task_adapter + f * watermark_adapter",
        "results": results,
        "note": ("Exact-token greedy generation plus teacher-forced mean logprob of the "
                 "token sequence after the chat prompt. One host, one seed."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
