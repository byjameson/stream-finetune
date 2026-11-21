"""Low-rank adapters (LoRA) for HF transformers and the tiny LM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

ATTN_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".out_proj",
                 ".query", ".key", ".value", ".dense")
MLP_SUFFIXES = (".mlp", ".fc1", ".fc2", ".gate_proj", ".up_proj",
                ".down_proj", ".c_fc", ".c_proj", ".mlp.0", ".mlp.2",
                "mlp.0", "mlp.2")
_TARGET_SUFFIXES = ("mlp.0", "mlp.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self._fast_dtype = None

    def set_fast(self, dtype):
        self._fast_dtype = dtype

    def forward(self, x):
        out = self.base(x)
        a, b = self.A, self.B
        if a.device != x.device:
            a, b = a.to(x.device), b.to(x.device)
        dtype = self._fast_dtype or out.dtype
        delta = (x.to(dtype) @ a.to(dtype).T @ b.to(dtype).T) * self.scaling
        return out + delta.to(out.dtype)


def _target_r(name: str, r_attn: int, r_mlp: int) -> tuple[int, float] | None:
    if r_attn > 0 and name.endswith(ATTN_SUFFIXES):
        return r_attn, float(r_attn)
    if r_mlp > 0 and name.endswith(MLP_SUFFIXES):
        return r_mlp, float(r_mlp)
    return None


def apply_lora(model: nn.Module, r_attn: int = 8, r_mlp: int = 0,
               alpha_attn: int | None = None,
               alpha_mlp: int | None = None) -> nn.Module:
    """Replace target Linear modules with LoRALinear and freeze the base."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not name.endswith(ATTN_SUFFIXES + MLP_SUFFIXES):
            continue
        is_attn = name.endswith(ATTN_SUFFIXES)
        r = r_attn if is_attn else r_mlp
        if r <= 0:
            continue
        alpha = (alpha_attn if is_attn else alpha_mlp) or r
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, r, int(alpha)))
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
    return model


def inject_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
                **kwargs) -> nn.Module:
    """Tiny-LM helper: adapt MLP layers only."""
    return apply_lora(model, r_attn=0, r_mlp=rank, alpha_mlp=alpha)


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def lora_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")
            or k.endswith("lora_A") or k.endswith("lora_B")}


def set_lora_state_dict(model: nn.Module, state: dict):
    target = {k: v for k, v in state.items()
              if k.endswith(".A") or k.endswith(".B")}
    if not target:
        raise SystemExit("adapter has no LoRA A/B tensors")
    model.load_state_dict(target, strict=False)


def merge_lora(model: nn.Module):
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALinear):
                continue
            delta = (module.B @ module.A) * module.scaling
            module.base.weight.add_(delta.to(module.base.weight.dtype))
            module.A.zero_()
            module.B.zero_()


def save_adapter(model: nn.Module, out: str | Path, **config) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model), "config": config},
               out / "adapter.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                     encoding="utf-8")
    return out


def apply_adapter(model: nn.Module, path: str | Path) -> dict:
    ck = torch.load(Path(path) / "adapter.pt", map_location="cpu",
                    weights_only=False)
    config = ck.get("config", {})
    apply_lora(model, r_attn=int(config.get("r_attn", 0)),
               r_mlp=int(config.get("r_mlp", config.get("rank", 8))),
               alpha_attn=config.get("alpha_attn"),
               alpha_mlp=config.get("alpha_mlp", config.get("alpha")))
    state = {k: v for k, v in ck["lora"].items()
             if k.endswith(".A") or k.endswith(".B")}
    model.load_state_dict(state, strict=False)
    model.eval()
    return config
