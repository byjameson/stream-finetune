"""Try to actually build a ~0.1 bit/weight model.

Global vector-quantization codebook per tensor: group=32 weights -> one of
K=9 prototypes (log2(9)/32 = 0.099 bit/weight). One scale per tensor, no
per-group scales (they would blow the budget). This is the honest version of
"0.1 bit": the model becomes a mosaic of 9 fixed 32-weight patterns per tensor.
"""
import json
import math

import torch
from transformers import AutoModelForCausalLM

MODEL = r"models/Qwen2.5-0.5B"
TEST = r"data/wt2_test.pt"
SEQ, NCHUNK = 512, 32
GROUP, K = 32, 9
SAMPLE = 100_000
ITERS = 8


def kmeans(sample, k, iters):
    n = sample.shape[0]
    idx = torch.randperm(n, device=sample.device)[:k]
    cent = sample[idx].clone()
    for _ in range(iters):
        d = torch.cdist(sample, cent)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                cent[j] = sample[m].mean(0)
    return cent


@torch.no_grad()
def vq_tensor(w, group=GROUP, k=K):
    shape = w.shape
    flat = w.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % group
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    groups = flat.view(-1, group)
    scale = groups.std().clamp_min(1e-8)
    norm = groups / scale
    if norm.shape[0] > SAMPLE:
        sel = torch.randperm(norm.shape[0], device=norm.device)[:SAMPLE]
        sample = norm[sel]
    else:
        sample = norm
    cent = kmeans(sample, k, ITERS)
    out = torch.empty_like(norm)
    step = 200_000
    for i in range(0, norm.shape[0], step):
        c = norm[i:i + step]
        a = torch.cdist(c, cent).argmin(1)
        out[i:i + step] = cent[a]
    return (out.reshape(-1)[:n] * scale).reshape(shape).to(w.dtype), cent.shape[0] * group * 4 + 4


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
    originals = {id(p): p.detach().clone() for _, p in params}
    codebook_bytes = 0
    for name, p in params:
        q, cb = vq_tensor(originals[id(p)])
        p.data.copy_(q)
        codebook_bytes += cb
        print(f"  {name:<48s} {tuple(p.shape)!s:<16s} cb={cb} B", flush=True)
    ce = eval_ce(model, chunks)
    bits = (sum(p.numel() for _, p in params) * (math.log2(K) / GROUP) + codebook_bytes * 8) / total_params
    size_mb = bits * total_params / 8 / 1e6
    row = {"scheme": f"{K} prototypes / {GROUP} weights", "bpw": round(bits, 4),
           "ce": round(ce, 4), "ppl": round(math.exp(ce), 2), "size_mb": round(size_mb, 2)}
    print(json.dumps(row))
    with open(r"research/results/vq_01bit.json", "w") as f:
        json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
