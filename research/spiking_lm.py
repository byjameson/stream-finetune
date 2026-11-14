"""A tiny spiking language model with synaptic current dynamics.

This is a from-scratch research prototype, not a chat model. It asks a narrow
question: can a recurrent network of leaky integrate-and-fire (LIF) neurons,
trained with surrogate gradients, learn next-token prediction on TinyStories?
A rate-based version of the same network is the control.

The synapse is modelled with a decaying current variable:

    syn_t = decay_syn * syn_{t-1} + gain * W_in x_t + W_rec s_{t-1} + bias
    v_t   = decay_mem * v_{t-1} + syn_t - threshold * s_{t-1}
            - adapt * trace_{t-1}
    trace_t = decay_trace * trace_{t-1} + s_{t-1}
    s_t   = spike(v_t, threshold)           # spike mode
    s_t   = tanh(v_t)                       # rate mode

The output is read from a low-pass filtered spike train:

    r_t = decay_out * r_{t-1} + s_t
    logits_t = W_out r_t

Spike mode uses a fast-sigmoid surrogate gradient. The rate mode is the same
network with a continuous activation and no reset. Both are trained with the
same data, seed, optimizer and number of steps, so the comparison isolates the
spiking dynamics.

Usage (smoke):
    python research/spiking_lm.py --mode both --steps 100

Usage (full TinyStories stream, still small):
    python research/spiking_lm.py --mode both --steps 1000 \
      --train data/ts_train.pt --valid data/ts_valid.pt \
      --out research/results/spiking_lm_full.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/ts_bpe8k.json"


class _SurrogateSpike(torch.autograd.Function):
    """Hard spike in the forward pass, fast-sigmoid derivative in the backward."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (v >= threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, = ctx.saved_tensors
        x = ctx.alpha * (v - ctx.threshold)
        sig = torch.sigmoid(x)
        grad = grad_output * ctx.alpha * sig * (1.0 - sig)
        return grad, None, None


def spike(v: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
    return _SurrogateSpike.apply(v, threshold, alpha)


class SpikingLayer(nn.Module):
    def __init__(self, dim: int, mode: str, decay_syn: float = 0.90,
                 decay_mem: float = 0.90, threshold: float = 0.50,
                 alpha: float = 4.0, adapt: float = 0.0,
                 decay_trace: float = 0.90):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.decay_syn = decay_syn
        self.decay_mem = decay_mem
        self.threshold = threshold
        self.alpha = alpha
        self.adapt = adapt
        self.decay_trace = decay_trace
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_rec = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.input_gain = nn.Parameter(torch.ones(1))

    def init_state(self, batch: int, device: torch.device):
        z = torch.zeros(batch, self.dim, device=device)
        return z, z.clone(), z.clone(), z.clone()

    def forward(self, x, state):
        v, syn, s_prev, trace = state
        syn = (self.decay_syn * syn
               + self.input_gain * self.w_in(x)
               + self.w_rec(s_prev)
               + self.bias)
        trace = self.decay_trace * trace + s_prev
        if self.mode == "spike":
            v = (self.decay_mem * v + syn
                 - self.threshold * s_prev - self.adapt * trace)
            s = spike(v, self.threshold, self.alpha)
        else:
            # Normalise the leak so the rate control has the same steady-state
            # scale as the synaptic current instead of integrating to saturation.
            v = (self.decay_mem * v + (1.0 - self.decay_mem) * syn
                 - self.adapt * trace)
            s = torch.tanh(v)
        return s, (v, syn, s, trace)


class SpikingLM(nn.Module):
    """A small recurrent LIF language model (or its rate-based control)."""

    def __init__(self, vocab: int, dim: int = 256, layers: int = 1,
                 mode: str = "spike", decay_out: float = 0.90,
                 adapt: float = 0.0):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.mode = mode
        self.decay_out = decay_out
        self.emb = nn.Embedding(vocab, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.10)
        self.layers = nn.ModuleList([
            SpikingLayer(dim, mode=mode, adapt=adapt) for _ in range(layers)
        ])
        self.head = nn.Linear(dim, vocab, bias=False)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        self.last_activity = 0.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        states = [layer.init_state(batch, tokens.device) for layer in self.layers]
        readout = torch.zeros(batch, self.dim, device=tokens.device)
        logits = []
        activity = 0.0
        for t in range(seq):
            cur = self.emb(tokens[:, t])
            for i, layer in enumerate(self.layers):
                cur, states[i] = layer(cur, states[i])
            readout = self.decay_out * readout + cur
            logits.append(self.head(readout))
            activity += float(cur.detach().abs().mean())
        self.last_activity = activity / max(seq, 1)
        return torch.stack(logits, dim=1)


class WindowSampler:
    def __init__(self, tokens: torch.Tensor, seq: int, batch: int, seed: int,
                 device: torch.device):
        self.tokens = tokens
        self.seq = seq
        self.batch = batch
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def next(self):
        hi = len(self.tokens) - self.seq - 1
        starts = torch.randint(0, hi, (self.batch,), generator=self.gen).tolist()
        xs = torch.stack([self.tokens[s:s + self.seq] for s in starts])
        ys = torch.stack([self.tokens[s + 1:s + self.seq + 1] for s in starts])
        return xs.to(self.device), ys.to(self.device)


def eval_model(model: SpikingLM, batches, tokenizer: Any) -> dict:
    model.eval()
    nll_sum = 0.0
    token_count = 0
    byte_count = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            nll = F.cross_entropy(logits.reshape(-1, model.vocab),
                                  y.reshape(-1), reduction="sum")
            nll_sum += float(nll)
            token_count += y.numel()
            for row in y:
                byte_count += len(tokenizer.decode(row.tolist()).encode("utf-8"))
    loss = nll_sum / max(token_count, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "bpb": nll_sum / math.log(2) / max(byte_count, 1),
        "tokens": token_count,
        "bytes": byte_count,
    }


@torch.no_grad()
def sample(model: SpikingLM, tokenizer: Any, prompt: torch.Tensor,
           max_new: int = 40) -> str:
    model.eval()
    ids = prompt.flatten().tolist()
    for _ in range(max_new):
        window = torch.tensor([ids[-128:]], device=prompt.device)
        logits = model(window)[0, -1]
        ids.append(int(logits.argmax()))
    return tokenizer.decode(ids)


def train_one(args, mode: str, train_tokens, valid_tokens,
              tokenizer: Any) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingLM(vocab=tokenizer.get_vocab_size(), dim=args.dim,
                      layers=args.layers, mode=mode,
                      adapt=args.adapt).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sampler = WindowSampler(train_tokens, args.seq, args.batch,
                            args.seed, device)
    eval_gen = torch.Generator().manual_seed(args.seed + 1000)
    eval_batches = []
    hi = len(valid_tokens) - args.seq - 1
    for _ in range(args.eval_batches):
        s = int(torch.randint(0, hi, (1,), generator=eval_gen))
        x = valid_tokens[s:s + args.seq].unsqueeze(0).to(device)
        y = valid_tokens[s + 1:s + args.seq + 1].unsqueeze(0).to(device)
        eval_batches.append((x, y))

    print(json.dumps({"mode": mode, "params": params, "device": str(device),
                      "train_tokens": len(train_tokens),
                      "valid_tokens": len(valid_tokens)}), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sampler.next()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_model(model, eval_batches, tokenizer)
            row = {"step": step, "train_loss": float(loss.detach()),
                   "activity": model.last_activity, **ev}
            history.append(row)
            print(json.dumps(row), flush=True)
            model.train()
    elapsed = time.time() - started
    final = eval_model(model, eval_batches, tokenizer)
    prompt = valid_tokens[:24].unsqueeze(0).to(device)
    text = sample(model, tokenizer, prompt, max_new=args.max_new)
    peak = (torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None)
    result = {
        "mode": mode,
        "params": params,
        "steps": args.steps,
        "batch": args.batch,
        "seq": args.seq,
        "dim": args.dim,
        "layers": args.layers,
        "lr": args.lr,
        "adapt": args.adapt,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 2),
        "sec_per_step": round(elapsed / max(args.steps, 1), 3),
        "peak_gpu_mb": round(peak, 1) if peak is not None else None,
        "history": history,
        "final": final,
        "sample": text,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spike", "rate", "both"], default="both")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--adapt", type=float, default=0.0,
                    help="spike-frequency adaptation strength (0 = off)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--train", default=str(ROOT / "data/ts_train_smoke.pt"))
    ap.add_argument("--valid", default=str(ROOT / "data/ts_valid_smoke.pt"))
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_lm.json"))
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    train_tokens = torch.load(args.train, weights_only=True).to(torch.int64)
    valid_tokens = torch.load(args.valid, weights_only=True).to(torch.int64)
    modes = ["spike", "rate"] if args.mode == "both" else [args.mode]
    results = [train_one(args, m, train_tokens, valid_tokens, tokenizer)
               for m in modes]
    payload = {
        "question": ("Can a recurrent LIF network with synaptic current dynamics "
                     "learn next-token prediction, compared with a rate-based "
                     "control of the same size?"),
        "data": {"train": args.train, "valid": args.valid,
                 "vocab": tokenizer.get_vocab_size()},
        "runs": results,
        "note": ("From-scratch small model, TinyStories BPE-8k, one seed. "
                 "bpb uses the decoded target bytes; it is approximate for "
                 "byte-level BPE boundaries. Not a chat model."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
