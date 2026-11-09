"""Quantized LoRA adapter storage (8-bit / 4-bit, group-wise).

The adapter is the per-behaviour artifact in `docs/BEHAVIORAL_BITS.md`: a shared base
plus a few MB of adapter. This module stores the A/B tensors as packed integers with
per-group scales, and dequantizes them at load time. It is deliberately separate from the
trainer: the adapter format is a deployment format, not a training format.

    streamft adapter --adapter checkpoints/my_adapter.pt --bits 4 --out q4.pt

The 4-bit file is smaller than fp32 but carries per-group scales; the real file size is
what counts, not the packed-code size alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

FORMAT = "streamft-adapter-q1"


def _quantize(w: torch.Tensor, bits: int, group: int):
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
    code = ((g - lo) / scale).round().clamp(0, levels).to(torch.uint8)
    if bits == 8:
        packed = code.reshape(-1)
    elif bits == 4:
        c = code.reshape(-1, 2)
        packed = (c[:, 0] | (c[:, 1] << 4)).to(torch.uint8)
    elif bits == 2:
        c = code.reshape(-1, 4)
        packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)
    else:
        raise ValueError("bits must be 2, 4 or 8")
    return {"packed": packed, "lo": lo.squeeze(1), "scale": scale.squeeze(1),
            "n": n, "shape": list(w.shape)}


def _unpack(q: dict, bits: int) -> torch.Tensor:
    p = q["packed"]
    if bits == 8:
        code = p
    elif bits == 4:
        code = torch.stack([p & 0xF, (p >> 4) & 0xF], dim=1).reshape(-1)
    elif bits == 2:
        code = torch.stack([p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3, (p >> 6) & 0x3],
                           dim=1).reshape(-1)
    else:
        raise ValueError("bits must be 2, 4 or 8")
    return code


def _dequantize(q: dict, bits: int) -> torch.Tensor:
    # _unpack returns the padded code vector (groups * group); slice after dequantizing.
    code = _unpack(q, bits).float()
    group = int(q.get("group", q["scale"].numel()))
    scale = q["scale"].float()
    lo = q["lo"].float()
    flat = code.view(-1, group)
    out = flat * scale[:, None] + lo[:, None]
    return out.reshape(-1)[: q["n"]].reshape(q["shape"])


def quantize_adapter(state: dict, bits: int, group: int = 64, meta: dict | None = None) -> dict:
    tensors = {}
    for name, w in state.items():
        if name.endswith(".A") or name.endswith(".B"):
            q = _quantize(w, bits, group)
            q["group"] = group
            tensors[name] = q
    return {"format": FORMAT, "bits": bits, "group": group,
            "meta": meta or {}, "tensors": tensors}


def dequantize_adapter(obj: dict) -> dict:
    return {name: _dequantize(q, obj["bits"]) for name, q in obj["tensors"].items()}


def save_quantized(state: dict, path: str | Path, bits: int, group: int = 64,
                   meta: dict | None = None) -> dict:
    obj = quantize_adapter(state, bits, group, meta)
    torch.save(obj, str(path))
    return obj


def load_adapter(path: str | Path):
    """Load a plain .pt adapter ({"lora": ...}) or a quantized one; return (state, meta)."""
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        return dequantize_adapter(obj), obj.get("meta", {})
    if isinstance(obj, dict) and "lora" in obj:
        return obj["lora"], obj.get("meta", {})
    if isinstance(obj, dict):
        return obj, {}
    raise ValueError(f"not an adapter: {path}")


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streamft adapter")
    ap.add_argument("--adapter", required=True,
                    help="adapter .pt, or a save dir containing adapter.pt")
    ap.add_argument("--bits", type=int, choices=[8, 4, 2], default=4,
                    help="8=safe, 4=default, 2=not behaviourally validated")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    src = Path(args.adapter)
    if src.is_dir():
        src = src / "adapter.pt"
    state, meta = load_adapter(src)
    if not state:
        raise SystemExit(f"no tensors in adapter: {src}")
    lora_keys = {k for k in state if k.endswith(".A") or k.endswith(".B")}
    if not lora_keys:
        raise SystemExit(f"no LoRA A/B tensors in {src}; nothing to pack")
    skipped = sorted(set(state) - lora_keys)
    out = Path(args.out)
    obj = save_quantized(state, out, args.bits, args.group, meta)
    back = dequantize_adapter(obj)
    errs = [tensor_error(state[k], back[k]) for k in back]
    params = sum(v.numel() for v in state.values())
    print(json.dumps({
        "adapter": str(src), "out": str(out), "bits": args.bits, "group": args.group,
        "params": params,
        "src_bytes": src.stat().st_size, "out_bytes": out.stat().st_size,
        "compression": round(src.stat().st_size / max(out.stat().st_size, 1), 2),
        "max_relative_error": round(max(errs) if errs else 0.0, 4),
        "tensors": len(back),
        "skipped_tensors": skipped,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
