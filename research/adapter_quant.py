"""Quantization of the Qwen3B LoRA adapter: 8/4/3/2-bit, group 64 min-max."""
import json

import torch

P = "checkpoints/p1_qwen3b_lora_r16_t60.pt"
BASE = 3.09e9
sd = torch.load(P, map_location="cpu", weights_only=False)["lora"]
params = sum(v.numel() for v in sd.values())


def qerr(w, bits, group=64):
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
    return (q - g).norm() / g.norm()


rows = []
for bits in [8, 4, 3, 2]:
    num = 0.0
    den = 0.0
    for w in sd.values():
        e = float(qerr(w, bits))
        num += (e ** 2) * float(w.float().norm() ** 2)
        den += float(w.float().norm() ** 2)
    err = (num / den) ** 0.5
    size = params * bits / 8
    row = {"bits": bits, "size_mb": round(size / 1e6, 2),
           "bits_per_base_param": round(size * 8 / BASE, 4),
           "relative_weight_error": round(err, 4)}
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"adapter": P, "params": params, "rows": rows},
          open("research/results/adapter_quant.json", "w"), indent=1)
