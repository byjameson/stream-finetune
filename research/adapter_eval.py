"""Behaviour eval of the Qwen3B LoRA at 16/8/4-bit: masked SFT loss on held-out Dolly.

Loads the 4-bit base, applies the custom LoRA, and compares the base, the original
adapter, and the 8-bit / 4-bit quantized adapters on the same 16 rows.
"""
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.lora import LoRALinear, apply_lora, set_lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
ADAPTER = "checkpoints/p1_qwen3b_lora_r16_t60.pt"
SEQ = 256
N = 16


def quantize(w, bits, group=64):
    flat = w.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % group
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    g = flat.view(-1, group)
    lo = g.amin(1, keepdim=True)
    hi = g.amax(1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp_min(1e-12) / levels
    q = ((g - lo) / scale).round().clamp(0, levels) * scale + lo
    return q.reshape(-1)[:n].reshape(w.shape).to(w.dtype)


def qstate(sd, bits):
    return {k: (quantize(v, bits) if k.endswith((".A", ".B")) else v) for k, v in sd.items()}


def encode(tok, row):
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": row["response"]}]
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


@torch.no_grad()
def eval_loss(model, tok, rows):
    model.eval()
    tot = 0.0
    n = 0
    for row in rows:
        enc = encode(tok, row)
        if enc is None:
            continue
        ids, labels = enc
        ids = ids.to(model.device)
        labels = labels.to(model.device)
        loss = model(input_ids=ids, labels=labels).loss
        tot += float(loss)
        n += 1
    return tot / max(n, 1), n


def main():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    g = torch.Generator().manual_seed(3)
    idx = torch.randperm(len(ds), generator=g)[:N].tolist()
    rows = [ds[i] for i in idx]

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda", dtype=torch.bfloat16)
    base_loss, n = eval_loss(model, tok, rows)
    print(json.dumps({"config": "base", "loss": round(base_loss, 4), "rows": n}), flush=True)

    apply_lora(model, r_attn=16, r_mlp=0)
    # bnb 4-bit Linear stores a packed weight with shape[1] == 1; LoRALinear uses
    # weight.shape[1] for A. Rebuild A with the real in_features before loading.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A.shape[1] != m.base.in_features:
                m.A = torch.nn.Parameter(torch.zeros(m.r, m.base.in_features,
                                                    device=m.A.device, dtype=torch.float32))
            if m.B.shape[0] != m.base.out_features:
                m.B = torch.nn.Parameter(torch.zeros(m.base.out_features, m.r,
                                                    device=m.B.device, dtype=torch.float32))
    state = torch.load(ADAPTER, map_location="cpu", weights_only=False)["lora"]
    out = []
    for name, sd in [("original fp32", state), ("8-bit", qstate(state, 8)), ("4-bit", qstate(state, 4))]:
        set_lora_state_dict(model, sd)
        loss, n = eval_loss(model, tok, rows)
        row = {"config": name, "loss": round(loss, 4), "rows": n}
        out.append(row)
        print(json.dumps(row), flush=True)

    json.dump({"base": round(base_loss, 4), "adapters": out},
              open("research/results/adapter_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
