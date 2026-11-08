"""Inference cost of the quantized adapter: base vs LoRA path on the same 4-bit model.

The adapter is a storage win, not a speed win. This measures the per-token cost of the
LoRA path (the low-rank matmul) against the base. Merging is not attempted on a bnb
4-bit base: the base weight is packed, so the adapter stays as a LoRA path.
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
Q4 = "checkpoints/p1_qwen3b_lora_r16_t60_q4.pt"
N = 32


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda", dtype=torch.bfloat16)
    prompt = "Explain why a 6 GB GPU can still fine-tune a 14B model."
    ids = tok(prompt, return_tensors="pt").to(model.device)

    @torch.no_grad()
    def run(n=N):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        return {"tokens_s": round(n / dt, 2), "seconds": round(dt, 2)}

    run(4)  # warmup
    base = [run() for _ in range(2)]

    t0 = time.perf_counter()
    state, _ = load_adapter(Q4)
    load_s = time.perf_counter() - t0
    apply_lora(model, r_attn=16, r_mlp=0)
    set_lora_state_dict(model, state)
    run(4)
    lora = [run() for _ in range(2)]

    out = {
        "base": base,
        "lora_q4": lora,
        "adapter_load_s": round(load_s, 3),
        "adapter_file_bytes": os.path.getsize(Q4),
        "note": "LoRA path keeps the adapter separate; merging is not attempted on a packed 4-bit base.",
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/adapter_speed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
