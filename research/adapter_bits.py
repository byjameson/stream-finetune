"""The 0.077-bit adapter: a LoRA delta on a shared Qwen2.5-3B base."""
import json
import os

import torch

P = "checkpoints/p1_qwen3b_lora_r16_t60.pt"
BASE = 3.09e9
sd = torch.load(P, map_location="cpu", weights_only=False)
lora = sd["lora"]
params = sum(v.numel() for v in lora.values())
size = os.path.getsize(P)
result = {
    "adapter": P,
    "meta": sd.get("meta", {}),
    "lora_params": params,
    "file_bytes": size,
    "fp32_bits_per_base_param": round(size * 8 / BASE, 4),
    "fp16_bytes": params * 2,
    "fp16_bits_per_base_param": round(params * 2 * 8 / BASE, 4),
}
print(json.dumps(result, indent=1))
json.dump(result, open("research/results/adapter_bits.json", "w"), indent=1)
