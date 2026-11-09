"""Does 4-bit quantization make inference faster on the reference 6 GB card?

Same 0.5B model, same prompt, same 32 greedy tokens. This isolates the bit-width
effect: the 3B adapter benchmark could not, because bf16 3B does not fit comfortably.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "models/Qwen2.5-0.5B"
N = 32
PROMPT = "Explain why a 6 GB GPU can still fine-tune a 14B model."


def bench(model, tok, label):
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=4, do_sample=False,
                       pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        runs = []
        for _ in range(2):
            t = time.perf_counter()
            model.generate(**ids, max_new_tokens=N, do_sample=False,
                           pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            runs.append({"tokens_s": round(N / dt, 2), "seconds": round(dt, 2)})
    return {"label": label, "runs": runs,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.bfloat16)
    out["bf16"] = bench(model, tok, "bf16")
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda", quantization_config=q)
    out["4-bit"] = bench(model, tok, "4-bit NF4")

    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/quant_speed_0p5b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
