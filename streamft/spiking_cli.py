"""CLI for the small spiking language model.

    streamft spiking train --data data/ts_train.pt --valid data/ts_valid.pt \
        --out checkpoints/spiking_4m --steps 300
    streamft spiking info --model checkpoints/spiking_4m
    streamft spiking generate --model checkpoints/spiking_4m --prompt "Once upon a time"
    streamft spiking serve --model checkpoints/spiking_4m --port 8011
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    count_params,
    evaluate_bpb,
    fisher_importance,
    generate,
    load_checkpoint,
    save_checkpoint,
    selective_rollback,
)

ROOT = Path(__file__).resolve().parents[1]


def load_tokens(path: str, limit: int = 0) -> torch.Tensor:
    try:
        t = torch.load(path, weights_only=True, mmap=True)
    except TypeError:
        t = torch.load(path, weights_only=True)
    t = t.to(torch.int64)
    return t[:limit] if limit else t


class RandomWindows:
    def __init__(self, tokens, seq: int, batch: int, seed: int, device):
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


def fixed_windows(tokens, seq: int, n: int, batch: int, seed: int, device):
    gen = torch.Generator().manual_seed(seed)
    hi = len(tokens) - seq - 1
    out = []
    for _ in range(n):
        starts = torch.randint(0, hi, (batch,), generator=gen).tolist()
        xs = torch.stack([tokens[s:s + seq] for s in starts]).to(device)
        ys = torch.stack([tokens[s + 1:s + seq + 1] for s in starts]).to(device)
        out.append((xs, ys))
    return out


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def train_cmd(args) -> int:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        model, tok, meta = load_checkpoint(args.resume)
        model.to(device)
        if tok is None and args.tokenizer:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(args.tokenizer)
        start_step = int(meta.get("step", 0))
        model_name = args.name or meta.get("name", Path(args.out).name)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        cfg = SpikingConfig(vocab=tok.get_vocab_size(), dim=args.dim,
                            layers=args.layers, mode="spike", adapt=args.adapt)
        model = SpikingLM(cfg).to(device)
        meta = {}
        start_step = 0
        model_name = args.name or Path(args.out).name

    if tok is None:
        raise SystemExit("no tokenizer: pass --tokenizer or resume from a checkpoint with tokenizer.json")
    if args.resume and (Path(args.resume) / "tokenizer.json").exists():
        tokenizer_path = Path(args.resume) / "tokenizer.json"
    else:
        tokenizer_path = args.tokenizer or None
    train_tokens = load_tokens(args.data, args.max_tokens)
    new_batches = (fixed_windows(load_tokens(args.valid), args.seq,
                                 args.eval_batches, args.eval_batch_size,
                                 args.seed + 100, device)
                   if args.valid else [])
    old_batches = (fixed_windows(load_tokens(args.old_valid), args.seq,
                                 args.eval_batches, args.eval_batch_size,
                                 args.seed + 200, device)
                   if args.old_valid else [])

    rollback_on = bool(old_batches and args.rollback_fraction > 0
                       and args.rollback_threshold > 0)
    initial_state = clone_state(model) if rollback_on else None
    importance = (fisher_importance(model, old_batches) if rollback_on else None)
    best_old = (evaluate_bpb(model, tok, old_batches)["bpb"]
                if old_batches else None)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sampler = RandomWindows(train_tokens, args.seq, args.batch,
                            args.seed + start_step, device)
    history = []
    print(json.dumps({
        "event": "start", "device": str(device), "params": count_params(model),
        "start_step": start_step, "steps": args.steps,
        "rollback": rollback_on, "old_bpb": best_old,
    }), flush=True)

    model.train()
    for i in range(1, args.steps + 1):
        step = start_step + i
        x, y = sampler.next()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if args.eval_every and (i % args.eval_every == 0 or i == args.steps):
            rec = {"event": "eval", "step": step,
                   "train_loss": float(loss.detach())}
            if new_batches:
                rec["new"] = evaluate_bpb(model, tok, new_batches)
            if old_batches:
                rec["old"] = evaluate_bpb(model, tok, old_batches)
                current_old = rec["old"]["bpb"]
                if (rollback_on and best_old is not None
                        and current_old > best_old * (1.0 + args.rollback_threshold)):
                    state_b = clone_state(model)
                    state_r = selective_rollback(initial_state, state_b,
                                                 importance, model,
                                                 args.rollback_fraction)
                    load_state(model, state_r)
                    after = evaluate_bpb(model, tok, old_batches)["bpb"]
                    applied = after < current_old
                    if not applied:
                        load_state(model, state_b)
                    model.train()
                    rec["rollback"] = {
                        "fraction": args.rollback_fraction,
                        "applied": applied,
                        "old_before": current_old,
                        "old_after": after,
                    }
                    best_old = after if applied else current_old
                else:
                    best_old = min(best_old, current_old)
            history.append(rec)
            print(json.dumps(rec), flush=True)

        if args.save_every and i % args.save_every == 0:
            save_checkpoint(model, args.out, tokenizer_path, {
                "step": step,
                "name": model_name,
                "data": args.data,
                "seed": args.seed,
                "source": "resume:" + args.resume if args.resume else "scratch",
            })
            print(json.dumps({"event": "checkpoint", "step": step,
                              "path": str(args.out)}), flush=True)

    save_checkpoint(model, args.out, tokenizer_path, {
        "step": start_step + args.steps,
        "name": model_name,
        "data": args.data,
        "seed": args.seed,
        "source": "resume:" + args.resume if args.resume else "scratch",
    })
    (Path(args.out) / "training.json").write_text(
        json.dumps({"history": history}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "saved", "path": str(args.out),
                      "step": start_step + args.steps}), flush=True)
    return 0


def tokenize_cmd(args) -> int:
    from tokenizers import Tokenizer

    text = Path(args.text).read_text(encoding="utf-8")
    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(text).ids
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(ids, dtype=torch.int64), out)
    print(json.dumps({"text": args.text, "tokens": len(ids),
                      "vocab": tok.get_vocab_size(), "out": str(out)}))
    return 0


def info_cmd(args) -> int:
    model, tok, meta = load_checkpoint(args.model)
    print(json.dumps({
        "params": count_params(model),
        "config": model.config.__dict__,
        "meta": meta,
        "tokenizer": tok is not None,
    }, indent=2))
    return 0


def eval_cmd(args) -> int:
    model, tok, _ = load_checkpoint(args.model)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = load_tokens(args.data, args.max_tokens)
    batches = fixed_windows(tokens, args.seq, args.batches, args.batch_size,
                            args.seed, device)
    print(json.dumps(evaluate_bpb(model, tok, batches), indent=2))
    return 0


def generate_cmd(args) -> int:
    model, tok, _ = load_checkpoint(args.model)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    text = generate(model, tok, args.prompt, args.max_new, device,
                    temperature=args.temperature, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty)
    print(text)
    return 0


def serve_cmd(args) -> int:
    from streamft.spiking_server import serve
    return serve(args.model, args.host, args.port, args.api_key)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streamft spiking")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train or resume a spiking LM")
    tr.add_argument("--data", required=True)
    tr.add_argument("--max-tokens", type=int, default=0,
                    help="use only the first N tokens (0 = all)")
    tr.add_argument("--valid", default="")
    tr.add_argument("--old-valid", default="",
                    help="old-task validation shard for online rollback")
    tr.add_argument("--tokenizer", default=str(ROOT / "data/ts_bpe8k.json"))
    tr.add_argument("--out", required=True)
    tr.add_argument("--name", default="")
    tr.add_argument("--resume", default="")
    tr.add_argument("--steps", type=int, default=300)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--seq", type=int, default=128)
    tr.add_argument("--dim", type=int, default=256)
    tr.add_argument("--layers", type=int, default=1)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--adapt", type=float, default=0.1)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument("--eval-batches", type=int, default=8)
    tr.add_argument("--eval-batch-size", type=int, default=4)
    tr.add_argument("--save-every", type=int, default=0)
    tr.add_argument("--rollback-threshold", type=float, default=0.0,
                    help="relative old-bpb degradation that triggers rollback (0=off)")
    tr.add_argument("--rollback-fraction", type=float, default=0.10)
    tr.add_argument("--seed", type=int, default=0)
    tr.set_defaults(func=train_cmd)

    tokp = sub.add_parser("tokenize", help="tokenize a UTF-8 text file")
    tokp.add_argument("--text", required=True)
    tokp.add_argument("--tokenizer", default=str(ROOT / "data/ts_bpe8k.json"))
    tokp.add_argument("--out", required=True)
    tokp.set_defaults(func=tokenize_cmd)

    inf = sub.add_parser("info", help="show checkpoint metadata")
    inf.add_argument("--model", required=True)
    inf.set_defaults(func=info_cmd)

    ev = sub.add_parser("eval", help="measure bpb on a token shard")
    ev.add_argument("--model", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--max-tokens", type=int, default=0)
    ev.add_argument("--seq", type=int, default=128)
    ev.add_argument("--batches", type=int, default=8)
    ev.add_argument("--batch-size", type=int, default=4)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=eval_cmd)

    gen = sub.add_parser("generate", help="generate from a checkpoint")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.set_defaults(func=generate_cmd)

    srv = sub.add_parser("serve", help="serve a checkpoint over HTTP")
    srv.add_argument("--model", required=True)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8011)
    srv.add_argument("--api-key", default="",
                     help="require Authorization: Bearer <key> on POST (env SPIKING_API_KEY)")
    srv.set_defaults(func=serve_cmd)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
