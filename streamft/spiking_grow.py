"""Function-preserving width growth for the spiking LM.

Growing depth is not function-preserving for this recurrent network, so this
tool intentionally supports width growth only. New embedding/head columns and
new recurrent rows/columns start at zero, which means the grown model produces
exactly the old logits at step zero. The new capacity is then trained with
`streamft spiking ingest` or `train --resume`.

This is a capacity operation, not a data operation. New data is added with the
ingest gate; growth just makes room for it.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    count_params,
    load_checkpoint,
    save_checkpoint,
)


def grow_width(model: SpikingLM, dim: int) -> SpikingLM:
    if dim < model.config.dim:
        raise ValueError("new width must be >= current width")
    cfg = asdict(model.config)
    cfg["dim"] = dim
    grown = SpikingLM(SpikingConfig(**cfg))
    old_dim = model.config.dim
    with torch.no_grad():
        grown.emb.weight.zero_()
        grown.emb.weight[:, :old_dim].copy_(model.emb.weight)
        grown.head.weight.zero_()
        grown.head.weight[:, :old_dim].copy_(model.head.weight)
        for old_layer, new_layer in zip(model.layers, grown.layers):
            new_layer.w_in.weight.zero_()
            new_layer.w_in.weight[:old_dim, :old_dim].copy_(old_layer.w_in.weight)
            new_layer.w_rec.weight.zero_()
            new_layer.w_rec.weight[:old_dim, :old_dim].copy_(old_layer.w_rec.weight)
            new_layer.bias.zero_()
            new_layer.bias[:old_dim].copy_(old_layer.bias)
            new_layer.input_gain.copy_(old_layer.input_gain)
    return grown


def grow_checkpoint(model_path: str, out: str, dim: int) -> dict:
    model, tok, meta = load_checkpoint(model_path)
    if dim < model.config.dim:
        raise ValueError("new width must be >= current width")
    grown = grow_width(model, dim)
    tokenizer_path = Path(model_path) / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = None
    save_checkpoint(grown, out, tokenizer_path, {
        "name": f"{meta.get('name', Path(model_path).name)}-grow-{dim}",
        "grown_from": str(model_path),
        "old_config": asdict(model.config),
        "new_config": asdict(grown.config),
        "step": 0,
        "source": "grow",
    })
    return {
        "out": str(out),
        "old_params": count_params(model),
        "new_params": count_params(grown),
        "old_dim": model.config.dim,
        "new_dim": dim,
        "layers": model.config.layers,
    }
