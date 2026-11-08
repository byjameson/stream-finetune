"""Fair scalar bit cliff: asymmetric min-max, group=64, all 2D weights."""
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM

MODEL = r"models/Qwen2.5-0.5B"
TEST = r"data/wt2_test.pt"
SEQ, NCHUNK, GROUP = 512, 32, 64


def quantize(w, bits, group=GROUP):
    shape = w.shape
    rows = w.reshape(-1, shape[-1]).float()
    n, cols = rows.shape
    pad = (-cols) % group
    if pad:
        rows = torch.cat([rows, torch.zeros(n, pad, device=rows.device)], 1)
    g = rows.view(n, -1, group)
    lo = g.amin(2, keepdim=True)
    hi = g.amax(2, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp_min(1e-12) / levels
    q = ((g - lo) / scale).round().clamp(0, levels) * scale + lo
    return q.reshape(n, -1)[:, :cols].reshape(shape).to(w.dtype)


def unique_weights(model):
    seen, out = set(), []
    for name, p in model.named_parameters():
        if p.dim() == 2 and ("proj" in name or "embed_tokens" in name or "lm_head" in name):
            if id(p) not in seen:
                seen.add(id(p)); out.append((name, p))
    return out


@torch.no_grad()
def eval_ce(model, chunks):
    total = ntok = 0
    for c in chunks:
        ids = c.unsqueeze(0).cuda()
        total += float(model(input_ids=ids, labels=ids).loss) * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return total / ntok


def main():
    tok = torch.load(TEST, weights_only=True)
    chunks = tok[: SEQ * NCHUNK].view(NCHUNK, SEQ).long()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    params = unique_weights(model)
    total_params = sum(p.numel() for p in model.parameters())
    q_params = sum(p.numel() for _, p in params)
    originals = {id(p): p.detach().clone() for _, p in params}
    rows = []
    for bits in [16, 8, 4, 3, 2, 1]:
        if bits == 16:
            ce = eval_ce(model, chunks)
        else:
            for _, p in params:
                p.data.copy_(quantize(originals[id(p)], bits))
            ce = eval_ce(model, chunks)
            for _, p in params:
                p.data.copy_(originals[id(p)])
        size = (q_params * bits / 8 + (total_params - q_params) * 2) / 1e6
        row = {"bits": bits, "ce": round(ce, 4), "ppl": round(math.exp(ce), 2), "size_mb": round(size, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(r"research/results/bit_cliff.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
