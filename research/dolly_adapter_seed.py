"""Train a second Dolly LoRA with a different seed, then pack 8/4-bit.

The first adapter (`p1_qwen3b_lora_r16_t60.pt`) gave a small but significant 4-bit
cost on 870 Dolly rows. This repeats the recipe with a disjoint training sample and a
different seed so the quantization cost can be checked across seeds.

    python research/dolly_adapter_seed.py --seed 1 --steps 60
    python research/adapter_eval.py --adapter checkpoints/p1_qwen3b_lora_r16_t60_seed1.pt \
        --rows 400 --out research/results/adapter_eval_seed1.json
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import save_quantized
from streamft.lora import apply_lora, lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
SEQ = 256
LR = 2e-4


def encode(tok, row):
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": row["response"]}]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full = tok(full_text, add_special_tokens=False).input_ids[:SEQ]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    prompt = tok(prompt_text, add_special_tokens=False).input_ids
    if len(prompt) >= len(full) or full[:len(prompt)] != prompt:
        return None
    labels = [-100] * len(prompt) + full[len(prompt):]
    if all(v == -100 for v in labels):
        return None
    return torch.tensor([full]), torch.tensor([labels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    # The eval rows are the first 400 of seed 3; train only on the rest.
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(3)).tolist()
    pool = perm[400:]
    g = torch.Generator().manual_seed(args.seed)
    order = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda",
                                                 dtype=torch.bfloat16)
    apply_lora(model, r_attn=16, r_mlp=0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    model.train()
    t0 = time.time()
    used = 0
    for idx in order:
        enc = encode(tok, ds[idx])
        if enc is None:
            continue
        ids, labels = enc
        ids = ids.to(model.device)
        labels = labels.to(model.device)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        used += 1
        if used % 10 == 0:
            print(json.dumps({"step": used, "loss": round(float(loss), 4),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
        if used >= args.steps:
            break
    if used < args.steps:
        raise SystemExit(f"only {used}/{args.steps} valid rows")

    state = lora_state_dict(model)
    out = f"checkpoints/p1_qwen3b_lora_r16_t60_seed{args.seed}.pt"
    torch.save({"lora": state, "meta": {"seed": args.seed, "steps": used}}, out)
    for bits in (8, 4):
        q = out.replace(".pt", f"_q{bits}.pt")
        save_quantized(state, q, bits, 64, meta={"seed": args.seed, "steps": used})
        print(json.dumps({"written": q}), flush=True)
    print(json.dumps({"adapter": out, "steps": used,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
