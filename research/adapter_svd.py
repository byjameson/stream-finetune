"""Compress the Qwen3B LoRA adapter by reducing its rank (SVD of B@A).

The adapter is already rank 16; this measures how much of its update survives at rank
8/4/2, and what the stored artifact would cost. Weight error is not behaviour; it is the
first filter before an eval.
"""
import json
import re
from collections import defaultdict

import torch

P = "checkpoints/p1_qwen3b_lora_r16_t60.pt"
BASE = 3.09e9
sd = torch.load(P, map_location="cpu", weights_only=False)["lora"]

groups = defaultdict(dict)
for k, v in sd.items():
    m = re.match(r"(.*)\.(A|B)$", k)
    if m:
        groups[m.group(1)][m.group(2)] = v.float()

print("modules", len(groups))
# SVD of W = B @ A for every module; cache the singular values/vectors.
svds = {}
for name, ab in groups.items():
    if "A" not in ab or "B" not in ab:
        continue
    W = ab["B"] @ ab["A"]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    svds[name] = (U, S, Vh, W.norm())

rows = []
for k in [16, 8, 4, 2]:
    num = 0.0
    den = 0.0
    stored = 0
    for name, (U, S, Vh, wn) in svds.items():
        kk = min(k, S.numel())
        Wk = (U[:, :kk] * S[:kk]) @ Vh[:kk]
        num += float((Wk - (U * S) @ Vh).norm() ** 2)
        den += float(wn ** 2)
        d_out, d_in = U.shape[0], Vh.shape[1]
        stored += kk * (d_out + d_in)
    err = (num / den) ** 0.5
    fp32 = stored * 4
    fp16 = stored * 2
    row = {"rank": k, "params": stored,
           "fp32_mb": round(fp32 / 1e6, 2), "fp16_mb": round(fp16 / 1e6, 2),
           "fp16_bits_per_base_param": round(fp16 * 8 / BASE, 4),
           "relative_weight_error": round(err, 4)}
    rows.append(row)
    print(json.dumps(row), flush=True)

json.dump({"adapter": P, "rows": rows},
          open("research/results/adapter_svd.json", "w"), indent=1)
