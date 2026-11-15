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
    feedback: bool = False
    attention: bool = False
    pointer_offset: int = 0
    memory: bool = False
    mem_decay: float = 0.5
    mem_read_pos: int = -1
    dropout: float = 0.0
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
        self.w_fb = None
        if config.feedback:
            self.w_fb = nn.Linear(config.dim, config.dim, bias=False)
            nn.init.zeros_(self.w_fb.weight)
        self.attn_q = None
        if config.attention or config.pointer_offset > 0:
            self.attn_q = nn.Linear(config.dim, config.dim, bias=False)
            self.attn_k = nn.Linear(config.dim, config.dim, bias=False)
            self.attn_v = nn.Linear(config.dim, config.dim, bias=False)
            for layer in (self.attn_q, self.attn_k, self.attn_v):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        self.copy_scale = None
        if config.pointer_offset > 0:
            self.copy_scale = nn.Parameter(torch.ones(1))
        self.mem_name = None
        if config.memory:
            self.mem_name = nn.Linear(config.dim, 1)
            self.mem_write = nn.Linear(config.dim, 1)
            self.mem_key = nn.Linear(config.dim, config.dim, bias=False)
            self.mem_val = nn.Linear(config.dim, config.dim, bias=False)
            self.mem_query = nn.Linear(config.dim, config.dim, bias=False)
            self.mem_scale = nn.Parameter(torch.ones(1))
            for layer in (self.mem_key, self.mem_val, self.mem_query):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.mem_write.bias, -2.0)
        self._mem_copy = None
        self._mem_attn = None
        self.last_copy_logits = None
        self.last_mem_attn = None
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.last_activity = 0.0

    def init_state(self, batch: int, device: torch.device):
        z = torch.zeros(batch, self.dim, device=device)
        return {
            "layers": [layer.init_state(batch, device) for layer in self.layers],
            "readout": z,
            "top": z.clone(),
            "hist_k": [],
            "hist_v": [],
            "emb_hist": [],
            "stmt": z.clone(),
            "name": z.clone(),
            "mem_k": [],
            "mem_v": [],
            "t": 0,
        }

    def step(self, tokens: torch.Tensor, state: dict):
        t = state["t"]
        state["t"] = t + 1
        emb_raw = self.emb(tokens)
        emb = self.drop(emb_raw) if self.drop is not None else emb_raw
        cur = emb
        if self.w_fb is not None:
            cur = cur + self.w_fb(state["top"])
        for i, layer in enumerate(self.layers):
            cur, state["layers"][i] = layer(cur, state["layers"][i])
        state["top"] = cur
        state["readout"] = self.config.decay_out * state["readout"] + cur
        readout = state["readout"]
        copy_logits = None
        if self.attn_q is not None:
            state["hist_k"].append(cur)
            state["hist_v"].append(readout)
            state["emb_hist"].append(emb)
            keys = torch.stack(state["hist_k"], dim=1)
            if self.config.pointer_offset > 0:
                off = self.config.pointer_offset
                vals = [state["emb_hist"][j - off] if j >= off
                        else torch.zeros_like(emb)
                        for j in range(len(state["emb_hist"]))]
                vals = torch.stack(vals, dim=1)
            else:
                vals = torch.stack(state["hist_v"], dim=1)
            q = self.attn_q(readout).unsqueeze(1)
            k = self.attn_k(keys)
            v = self.attn_v(vals)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
            attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            ctx = (attn @ v).squeeze(1)
            readout = readout + ctx
            if self.copy_scale is not None:
                copy_logits = (ctx @ self.emb.weight.T) / math.sqrt(self.dim)
        mem_copy = None
        self._mem_copy = None
        if self.mem_name is not None:
            state["stmt"] = self.config.mem_decay * state["stmt"] + emb_raw
            g_name = torch.sigmoid(self.mem_name(emb_raw))
            state["name"] = (1.0 - g_name) * state["name"] + g_name * emb_raw
            g_write = torch.sigmoid(self.mem_write(emb_raw))
            state["mem_k"].append(g_write * self.mem_key(state["stmt"]))
            state["mem_v"].append(g_write * self.mem_val(state["name"]))
            read_pos = self.config.mem_read_pos
            if read_pos < 0 or t == read_pos:
                mk = torch.stack(state["mem_k"], dim=1)
                mv = torch.stack(state["mem_v"], dim=1)
                mq = self.mem_query(readout).unsqueeze(1)
                mscores = mq @ mk.transpose(-1, -2) / math.sqrt(self.dim)
                mattn = torch.softmax(mscores.float(), dim=-1).to(mv.dtype)
                mctx = (mattn @ mv).squeeze(1)
                readout = readout + mctx
                mem_copy = (mctx @ self.emb.weight.T) / math.sqrt(self.dim)
                self._mem_attn = mattn.squeeze(1)
        logits = self.head(self.drop(readout) if self.drop is not None else readout)
        if copy_logits is not None:
            logits = logits + self.copy_scale * copy_logits
        if mem_copy is not None:
            logits = logits + self.mem_scale * mem_copy
        self._mem_copy = mem_copy
        return logits, state

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        state = self.init_state(batch, tokens.device)
        logits = []
        mem_copies = []
        activity = 0.0
        for t in range(seq):
            out, state = self.step(tokens[:, t], state)
            logits.append(out)
            if self.mem_name is not None and self._mem_copy is not None:
                mem_copies.append(self._mem_copy)
            activity += float(state["layers"][-1][2].detach().abs().mean())
        self.last_activity = activity / max(seq, 1)
        if len(mem_copies) == 1:
            self.last_copy_logits = mem_copies[0]
        else:
            self.last_copy_logits = (torch.stack(mem_copies, dim=1)
                                     if mem_copies else None)
        self.last_mem_attn = self._mem_attn
        return torch.stack(logits, dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: SpikingLM, out_dir: str | Path,
                    tokenizer_path: str | Path | None = None,
                    meta: dict | None = None) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous()
             for k, v in model.state_dict().items()}
    save_file(state, out / "model.safetensors")
    (out / "meta.json").write_text(json.dumps({
        "config": asdict(model.config), "meta": meta or {},
    }, indent=2) + "\n", encoding="utf-8")
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
    safe = root / "model.safetensors"
    if safe.exists():
        from safetensors.torch import load_file
        ck = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        state = load_file(str(safe))
    else:
        ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        state = ck["state"]
    defaults = {f.name: f.default for f in fields(SpikingConfig)
                if f.default is not MISSING}
    model = SpikingLM(SpikingConfig(**{**defaults, **ck["config"]}))
    model.load_state_dict(state)
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
    ids = ids[-2048:]  # bound the prompt; the model is recurrent, not positional
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
