"""Independent gradient check for the streaming LoRA engine.

Three questions:

  1. Does the block-streaming backward (1 block resident + recompute) produce the
     same adapter gradients as the same model with every block resident?
     -> both configurations, identical adapter weights, compare.

  2. Does the manual block backward agree with PyTorch autograd through the very
     same forward? (the manual path is what could be wrong)
     -> enable_grad forward + loss.backward(), compare with trainer.backward().

  3. Are the gradients the true gradient of the loss? Central finite differences
     are *not* usable on a 4-bit base: the LoRA delta is added to a bf16 tensor,
     so a one-entry perturbation changes the loss by less than one bf16 ULP and
     the FD quotient becomes a rounding staircase (verified: eps=2e-3 gives
     -0.12, eps=2e-2 gives +0.02 for the same entry). The chunked-loss gradient
     itself is checked exactly in tests/test_streamft_smoke.py.

Usage:
  python scripts/verify_gradients.py --model-dir checkpoints/qwen3b_bnb4bit --seq 128
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamft.data import load_tokens  # noqa: E402
from streamft.disk_quant_store import (  # noqa: E402
    DiskQuantStore,
    LayerCache,
    build_skeleton,
    materialize_shared,
)
from streamft.stream_quant_lora import StreamingQuantLoRATrainer  # noqa: E402


def build_trainer(model_dir: str, blocks: int, chunk: int, lora_r: int, targets: str):
    model, cfg = build_skeleton(
        model_dir, lora_r=lora_r, lora_alpha=2 * lora_r, targets=targets,
        head="bf16", low_precision_shared=True,
    )
    store = DiskQuantStore(model_dir, device="cuda")
    base = materialize_shared(model, store, cfg, device="cuda")
    layers = list(base.model.layers)
    cache = LayerCache(store, layers, device="cuda", capacity=blocks)

    def stage(layer, i):
        cache.stage(layer, i)

    def release(layer, i):
        cache.release(layer, i)

    tr = StreamingQuantLoRATrainer(
        model, device="cuda", loss_chunk=chunk, lr=1e-4, clip=0.0,
        stage=stage, release=release, cache_device="gpu",
    )
    return tr, (model, store, base, cache)


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def adapter_state(tr) -> dict:
    return {n: p.detach().cpu().clone() for n, p in tr.model.named_parameters()
            if p.requires_grad}


def load_adapter_state(tr, sd: dict) -> None:
    own = {n: p for n, p in tr.model.named_parameters() if p.requires_grad}
    missing = [n for n in sd if n not in own]
    if missing:
        raise RuntimeError(f"adapter keys missing on this model: {missing[:3]}")
    with torch.no_grad():
        for n, p in own.items():
            p.copy_(sd[n].to(device=p.device, dtype=p.dtype))


def named_grads(tr, grads: dict) -> dict:
    return {n: grads[id(p)].detach().float().cpu()
            for n, p in tr.model.named_parameters() if id(p) in grads}


def autograd_grads(tr, xs, ys) -> dict:
    """Reference: PyTorch autograd through the same forward (no manual backward).

    `_forward_cached` is decorated with @torch.no_grad(), so call the undecorated
    function to actually build a graph.
    """
    fwd = getattr(tr._forward_cached, "__wrapped__", None)
    if fwd is None:
        raise RuntimeError("cannot reach the undecorated forward for the autograd check")
    tr.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        h, _cache, _pos = fwd(tr, xs)
        z = tr.norm(h)
        logits = tr.lm_head(z).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ys.reshape(-1).to(logits.device))
        loss.backward()
    out = {n: p.grad.detach().float().cpu() for n, p in tr.model.named_parameters()
           if p.requires_grad and p.grad is not None}
    tr.model.zero_grad(set_to_none=True)
    return out, float(loss.detach())


def compare(reference: dict, other: dict, label: str) -> dict:
    rows = []
    for n in reference:
        if n not in other:
            rows.append({"tensor": n, "rel_err": float("inf"), "cosine": 0.0,
                         "norm": reference[n].norm().item()})
            continue
        g0, g1 = reference[n], other[n]
        denom = max(g0.norm().item(), 1e-12)
        rel = (g1 - g0).norm().item() / denom
        n0 = g0.reshape(-1).norm()
        cos = 1.0 if n0 == 0 else torch.nn.functional.cosine_similarity(
            g0.reshape(-1), g1.reshape(-1), dim=0).item()
        rows.append({"tensor": n, "rel_err": rel, "cosine": cos,
                     "norm": g0.norm().item()})
    nonzero = [r for r in rows if r["norm"] > 0]
    worst = max(rows, key=lambda r: r["rel_err"])
    return {
        "label": label,
        "n_tensors": len(rows),
        "n_nonzero": len(nonzero),
        "mean_rel_err": sum(r["rel_err"] for r in rows) / len(rows),
        "max_rel_err": worst["rel_err"],
        "worst_tensor": worst["tensor"],
        "min_cosine": min((r["cosine"] for r in nonzero), default=1.0),
        "global_rel_err": (sum((other[n] - reference[n]).pow(2).sum().item() for n in reference) ** 0.5)
                          / max(sum(reference[n].pow(2).sum().item() for n in reference) ** 0.5, 1e-12),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/qwen3b_bnb4bit")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--targets", default="attn")
    ap.add_argument("--offset", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model_dir = str(ROOT / args.model_dir)
    tokens = load_tokens(str(ROOT / "data" / "wt2_train.pt"))
    o = args.offset
    xs = tokens[o: o + args.seq].long().unsqueeze(0)
    ys = tokens[o + 1: o + args.seq + 1].long().unsqueeze(0)
    print(f"[data] {args.model_dir} seq={args.seq} chunk={args.chunk} offset={o}", flush=True)

    # ---------- A: streaming (1 resident block): manual backward vs autograd
    t0 = time.time()
    tr1, keep1 = build_trainer(model_dir, 1, args.chunk, args.lora_r, args.targets)
    state = adapter_state(tr1)
    auto, loss_auto = autograd_grads(tr1, xs, ys)
    loss1, g1, peak1 = tr1.backward(xs, ys)
    g1n = named_grads(tr1, g1)
    print(f"[stream] blocks=1 loss={loss1:.8f} peak={peak1:.2f}GB "
          f"({time.time()-t0:.1f}s, {len(g1n)} tensors)", flush=True)
    cmp_auto = compare(auto, g1n, "manual block backward vs autograd (1 block)")
    print(f"[autograd] mean_rel={cmp_auto['mean_rel_err']:.2e} "
          f"max_rel={cmp_auto['max_rel_err']:.2e} ({cmp_auto['worst_tensor']}) "
          f"min_cos={cmp_auto['min_cosine']:.8f}", flush=True)
    free(keep1, tr1)

    # ---------- B: all blocks resident, same adapter weights
    n_layers = 36
    try:
        n_layers = int(json.loads((Path(model_dir) / "config.json").read_text())["num_hidden_layers"])
    except Exception:
        pass
    tr2, keep2 = build_trainer(model_dir, n_layers, args.chunk, args.lora_r, args.targets)
    load_adapter_state(tr2, state)
    loss2, g2, peak2 = tr2.backward(xs, ys)
    g2n = named_grads(tr2, g2)
    auto2, _ = autograd_grads(tr2, xs, ys)
    cmp_auto2 = compare(auto2, g2n, "manual vs autograd (all blocks)")
    print(f"[full] blocks={n_layers} loss={loss2:.8f} peak={peak2:.2f}GB "
          f"| manual-vs-auto global_rel={cmp_auto2['global_rel_err']:.2e}", flush=True)
    cmp_full = compare(g2n, g1n, "stream(1 block) vs full(all blocks)")
    print(f"[config]  mean_rel={cmp_full['mean_rel_err']:.2e} "
          f"max_rel={cmp_full['max_rel_err']:.2e} ({cmp_full['worst_tensor']}) "
          f"min_cos={cmp_full['min_cosine']:.8f}", flush=True)
    free(keep2, tr2)

    out = {
        "model": args.model_dir, "seq": args.seq, "chunk": args.chunk,
        "lora_r": args.lora_r,
        "loss_stream": loss1, "loss_full": loss2, "loss_autograd": loss_auto,
        "peak_stream_gb": peak1, "peak_full_gb": peak2,
        "autograd_vs_manual": {k: v for k, v in cmp_auto.items() if k != "rows"},
        "stream_vs_full": {k: v for k, v in cmp_full.items() if k != "rows"},
        "worst_auto": sorted(cmp_auto["rows"], key=lambda r: -r["rel_err"])[:8],
        "worst_full": sorted(cmp_full["rows"], key=lambda r: -r["rel_err"])[:5],
        "autograd_vs_manual_full": {k: v for k, v in cmp_auto2.items() if k != "rows"},
    }
    dest = Path(args.out) if args.out else ROOT / "results" / "gradcheck.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\n=== RESULT ===")
    print(f"loss   stream={loss1:.10f}  full={loss2:.10f}  autograd={loss_auto:.10f}")
    print(f"grads  stream-vs-full : max_rel={cmp_full['max_rel_err']:.2e} "
          f"min_cos={cmp_full['min_cosine']:.8f}")
    print(f"grads  manual-vs-auto : global_rel={cmp_auto['global_rel_err']:.2e} "
          f"max_rel={cmp_auto['max_rel_err']:.2e} min_cos={cmp_auto['min_cosine']:.8f}")
    print(f"[write] {dest}")

    ok = (cmp_full["max_rel_err"] < 1e-5 and cmp_full["min_cosine"] > 0.99999
          and cmp_auto["global_rel_err"] < 5e-2 and cmp_auto["min_cosine"] > 0.999
          and cmp_auto2["global_rel_err"] < 5e-2)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
