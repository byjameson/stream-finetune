"""Small spiking language model: LIF neurons, synaptic currents, checkpoints.

This is the product-side implementation of the prototype in
`research/spiking_lm.py`. It is deliberately self-contained: no pretrained
weights, no training data bundled, and no dependency on the research folder.

A checkpoint directory contains:

    model.pt        config + state_dict + metadata (torch)
    config.json     human-readable model config
    model_card.json small machine-readable summary
    tokenizer.json  optional copy of the BPE tokenizer
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpikingConfig:
    vocab: int
    dim: int = 256
    layers: int = 1
    mode: str = "spike"
    adapt: float = 0.0
    decay_syn: float = 0.90
    decay_mem: float = 0.90
    decay_out: float = 0.90
    decay_trace: float = 0.90
    threshold: float = 0.50
    alpha: float = 4.0


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, threshold, alpha):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (v >= threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        x = ctx.alpha * (v - ctx.threshold)
        sig = torch.sigmoid(x)
        return grad_output * ctx.alpha * sig * (1.0 - sig), None, None


def spike(v: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
    return _SurrogateSpike.apply(v, threshold, alpha)


class SpikingLayer(nn.Module):
    def __init__(self, dim: int, config: SpikingConfig):
        super().__init__()
        self.dim = dim
        self.config = config
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_rec = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.input_gain = nn.Parameter(torch.ones(1))

    def init_state(self, batch: int, device: torch.device):
        z = torch.zeros(batch, self.dim, device=device)
        return z, z.clone(), z.clone(), z.clone()

    def forward(self, x, state):
        v, syn, s_prev, trace = state
        c = self.config
        syn = (c.decay_syn * syn
               + self.input_gain * self.w_in(x)
               + self.w_rec(s_prev)
               + self.bias)
        trace = c.decay_trace * trace + s_prev
        if c.mode == "spike":
            v = (c.decay_mem * v + syn
                 - c.threshold * s_prev - c.adapt * trace)
            s = spike(v, c.threshold, c.alpha)
        else:
            v = c.decay_mem * v + (1.0 - c.decay_mem) * syn - c.adapt * trace
            s = torch.tanh(v)
        return s, (v, syn, s, trace)


class SpikingLM(nn.Module):
    def __init__(self, config: SpikingConfig):
        super().__init__()
        self.config = config
        self.vocab = config.vocab
        self.dim = config.dim
        self.emb = nn.Embedding(config.vocab, config.dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.10)
        self.layers = nn.ModuleList([
            SpikingLayer(config.dim, config) for _ in range(config.layers)
        ])
        self.head = nn.Linear(config.dim, config.vocab, bias=False)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        self.last_activity = 0.0

    def init_state(self, batch: int, device: torch.device):
        z = torch.zeros(batch, self.dim, device=device)
        return {
            "layers": [layer.init_state(batch, device) for layer in self.layers],
            "readout": z,
        }

    def step(self, tokens: torch.Tensor, state: dict):
        cur = self.emb(tokens)
        for i, layer in enumerate(self.layers):
            cur, state["layers"][i] = layer(cur, state["layers"][i])
        state["readout"] = self.config.decay_out * state["readout"] + cur
        logits = self.head(state["readout"])
        return logits, state

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        state = self.init_state(batch, tokens.device)
        logits = []
        activity = 0.0
        for t in range(seq):
            out, state = self.step(tokens[:, t], state)
            logits.append(out)
            activity += float(state["layers"][-1][2].detach().abs().mean())
        self.last_activity = activity / max(seq, 1)
        return torch.stack(logits, dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: SpikingLM, out_dir: str | Path,
                    tokenizer_path: str | Path | None = None,
                    meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(model.config),
        "state": model.state_dict(),
        "meta": meta or {},
    }, out / "model.pt")
    (out / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n", encoding="utf-8")
    (out / "model_card.json").write_text(json.dumps({
        "name": "streamft-spiking",
        "params": count_params(model),
        "config": asdict(model.config),
        "meta": meta or {},
    }, indent=2) + "\n", encoding="utf-8")
    if tokenizer_path is not None:
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    return out


def load_checkpoint(path: str | Path):
    root = Path(path)
    ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    defaults = {f.name: f.default for f in fields(SpikingConfig)
                if f.default is not MISSING}
    model = SpikingLM(SpikingConfig(**{**defaults, **ck["config"]}))
    model.load_state_dict(ck["state"])
    model.eval()
    tok = None
    tok_path = root / "tokenizer.json"
    if tok_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
    return model, tok, ck.get("meta", {})


@torch.no_grad()
def generate(model: SpikingLM, tokenizer: Any, prompt: str,
             max_new_tokens: int = 64, device: torch.device | None = None,
             temperature: float = 0.0, top_k: int = 0,
             repetition_penalty: float = 1.0) -> str:
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    ids = tokenizer.encode(prompt).ids or [0]
    state = model.init_state(1, device)
    last = None
    for tid in ids:
        last, state = model.step(torch.tensor([tid], device=device), state)
    generated = []
    seen = set(ids)
    for _ in range(max(0, int(max_new_tokens))):
        assert last is not None
        logits = last[0].float().clone()
        if repetition_penalty != 1.0:
            for tid in seen:
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
        if temperature <= 0:
            nxt = int(logits.argmax().item())
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.numel())
                vals, idx = torch.topk(logits, k)
                logits = torch.full_like(logits, float("-inf"))
                logits[idx] = vals
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        generated.append(nxt)
        seen.add(nxt)
        last, state = model.step(torch.tensor([nxt], device=device), state)
    return tokenizer.decode(generated)


def flatten_state(state: dict, params) -> torch.Tensor:
    return torch.cat([state[name].float().flatten() for name, _ in params])


def state_from_flat(flat: torch.Tensor, params) -> dict:
    out = {}
    offset = 0
    for name, p in params:
        n = p.numel()
        out[name] = flat[offset:offset + n].view_as(p)
        offset += n
    return out


@torch.enable_grad()
def fisher_importance(model: SpikingLM, batches) -> dict:
    imp = {name: torch.zeros_like(p, device="cpu")
           for name, p in model.named_parameters()}
    model.eval()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                imp[name] += p.grad.detach().float().pow(2).cpu()
    model.zero_grad(set_to_none=True)
    return imp


def selective_rollback(state_a: dict, state_b: dict, importance: dict,
                       model: SpikingLM, fraction: float) -> dict:
    params = list(model.named_parameters())
    flat_a = flatten_state(state_a, params)
    flat_b = flatten_state(state_b, params)
    flat_i = torch.cat([importance[name].float().flatten() for name, _ in params])
    k = int(round(fraction * flat_i.numel()))
    flat = flat_b.clone()
    if k > 0:
        idx = torch.topk(flat_i, k).indices
        flat[idx] = flat_a[idx]
    return state_from_flat(flat, params)


@torch.no_grad()
def evaluate_bpb(model: SpikingLM, tokenizer: Any, batches) -> dict:
    model.eval()
    nll = 0.0
    tokens = 0
    nbytes = 0
    for x, y in batches:
        logits = model(x)
        nll += float(F.cross_entropy(logits.reshape(-1, model.vocab),
                                     y.reshape(-1), reduction="sum"))
        tokens += y.numel()
        for row in y:
            nbytes += len(tokenizer.decode(row.tolist()).encode("utf-8"))
    loss = nll / max(tokens, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "bpb": nll / math.log(2) / max(nbytes, 1),
        "tokens": tokens,
        "bytes": nbytes,
    }
