"""`streamft merge` — fold a saved LoRA adapter into the base model (CPU is fine).

The merged directory is a plain HF checkpoint, which is what GGUF conversion expects.

    streamft merge --model Qwen/Qwen2.5-0.5B --adapter checkpoints/cpu_cookbook --out merged/q05

The adapter was saved by `streamft train --save` / `streamft cpu --save` as
`adapter.pt` (`lora_state_dict`). The rank is read from the tensor shapes, so no extra
config file is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .lora import LoRALinear, apply_lora, merge_lora, set_lora_state_dict


def main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft merge")
    ap.add_argument("--model", required=True, help="base model (HF id or local dir)")
    ap.add_argument("--adapter", required=True, help="dir with adapter.pt")
    ap.add_argument("--out", required=True, help="output directory for the merged model")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
                    help="storage dtype of the merged weights (fp16 halves the file)")
    args = ap.parse_args(argv)

    adapter_path = Path(args.adapter)
    if adapter_path.is_dir():
        adapter_path = adapter_path / "adapter.pt"
    if not adapter_path.exists():
        raise SystemExit(f"adapter not found: {adapter_path}")

    state = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
    if not state:
        raise SystemExit("adapter.pt is empty")
    a_key = next((k for k in state if k.endswith(".A")), None)
    if a_key is None:
        raise SystemExit("adapter.pt has no LoRA A tensors; was it saved with lora_state_dict?")
    r = int(state[a_key].shape[0])

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(f"[merge] {args.model} + {adapter_path.name} (r={r}) -> {args.out} [{args.dtype}]",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    apply_lora(model, r_attn=r, r_mlp=0)
    set_lora_state_dict(model, state)
    merge_lora(model)

    # After a real merge every LoRA factor is exactly zero; the base absorbed the delta.
    not_merged = [n for n, p in model.named_parameters()
                  if (n.endswith(".A") or n.endswith(".B")) and float(p.detach().abs().sum()) > 0]
    if not_merged:
        raise SystemExit(f"adapter tensors left unmerged: {not_merged[:3]}")
    for p in model.parameters():
        p.requires_grad_(False)

    # Unwrap: after merge the LoRA factors are zero, but `save_pretrained` would still
    # write them, and GGUF converters refuse unknown tensors (`...k_proj.A`).
    stripped = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, mod.base)
            stripped += 1
    print(f"[merge] removed {stripped} LoRA wrappers", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(str(out))

    sizes = {f.name: round(f.stat().st_size / 1e9, 2) for f in out.glob("*.safetensors")}
    print(json.dumps({"merged": str(out), "dtype": args.dtype, "rank": r,
                      "weights_gb": sizes, "license": "same as the base model"}, indent=1))
    print("\nnext (needs llama.cpp, several minutes of CPU):")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {out} --outfile model-f16.gguf")
    print("  llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M")
    print("  open wasm/index.html and pick model-q4_k_m.gguf")
    return 0
