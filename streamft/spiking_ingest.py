"""Safe incremental ingest for the spiking LM.

The flow is intentionally boring:

    tokenize new text -> split new train/valid -> mix replay from old data
    -> continue from the current checkpoint -> evaluate old and new
    -> optional Fisher rollback -> accept/reject -> version the checkpoint

Nothing is published unless the old task stays within the regression budget and
the new task improves. Rejected candidates are logged in `registry.json`; their
weights are discarded unless `--keep-rejected` is set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from streamft.spiking import (
    SpikingLM,
    count_params,
    evaluate_bpb,
    fisher_importance,
    load_checkpoint,
    save_checkpoint,
    selective_rollback,
)


def load_tokens(path: str, limit: int = 0) -> torch.Tensor:
    try:
        t = torch.load(path, weights_only=True, mmap=True)
    except TypeError:
        t = torch.load(path, weights_only=True)
    t = t.to(torch.int64)
    return t[:limit] if limit else t


class MixtureWindows:
    """Sample a batch where each row is from old or new data."""

    def __init__(self, old_tokens, new_tokens, seq: int, batch: int,
                 replay_fraction: float, seed: int, device):
        if len(new_tokens) <= seq + 1:
            raise ValueError("new training shard is shorter than seq+1")
        if len(old_tokens) <= seq + 1:
            raise ValueError("old replay shard is shorter than seq+1")
        self.old = old_tokens
        self.new = new_tokens
        self.seq = seq
        self.batch = batch
        self.replay = max(0.0, min(1.0, replay_fraction))
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def _row(self, tokens, start: int):
        x = tokens[start:start + self.seq]
        y = tokens[start + 1:start + self.seq + 1]
        return x, y

    def next(self):
        choose_old = torch.rand(self.batch, generator=self.gen) < self.replay
        xs, ys = [], []
        for is_old in choose_old.tolist():
            tokens = self.old if is_old else self.new
            hi = len(tokens) - self.seq - 1
            start = int(torch.randint(0, hi, (1,), generator=self.gen))
            x, y = self._row(tokens, start)
            xs.append(x)
            ys.append(y)
        return (torch.stack(xs).to(self.device),
                torch.stack(ys).to(self.device))


def fixed_batches(tokens, seq: int, n: int, batch: int, seed: int, device):
    gen = torch.Generator().manual_seed(seed)
    hi = len(tokens) - seq - 1
    if hi <= 0:
        raise ValueError("evaluation shard is shorter than seq+1")
    out = []
    for _ in range(n):
        starts = torch.randint(0, hi, (batch,), generator=gen).tolist()
        xs = torch.stack([tokens[s:s + seq] for s in starts]).to(device)
        ys = torch.stack([tokens[s + 1:s + seq + 1] for s in starts]).to(device)
        out.append((xs, ys))
    return out


def split_train_valid(tokens: torch.Tensor, seq: int, valid_fraction: float = 0.05):
    n_valid = max(seq + 1, int(round(len(tokens) * valid_fraction)))
    if len(tokens) - n_valid <= seq + 1:
        raise ValueError("new data is too short for a train/valid split")
    return tokens[:-n_valid], tokens[-n_valid:]


def should_accept(old_before: float, old_after: float, new_before: float,
                  new_after: float, max_old_regression: float = 0.02,
                  min_new_gain: float = 0.01) -> bool:
    old_ok = old_after <= old_before * (1.0 + max_old_regression)
    new_ok = new_after <= new_before * (1.0 - min_new_gain)
    return bool(old_ok and new_ok)


def next_version(root: Path) -> str:
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in versions.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
            ids.append(int(p.name[1:]))
    return f"v{(max(ids) + 1) if ids else 1:04d}"


def _clone(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load(model, state):
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _eval_both(model, old_batches, new_batches, tok):
    return {"old": evaluate_bpb(model, tok, old_batches),
            "new": evaluate_bpb(model, tok, new_batches)}


def ingest(args) -> dict:
    from tokenizers import Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, tok, meta = load_checkpoint(args.model)
    if tok is None:
        tok = Tokenizer.from_file(args.tokenizer)
    model.to(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")
        all_new = torch.tensor(tok.encode(text).ids, dtype=torch.int64)
        new_train, new_valid = split_train_valid(all_new, args.seq)
        data_dir = out / "data"
        data_dir.mkdir(exist_ok=True)
        torch.save(new_train, data_dir / "new_train.pt")
        torch.save(new_valid, data_dir / "new_valid.pt")
    else:
        new_train = load_tokens(args.new_data, args.new_tokens)
        if args.new_valid:
            new_valid = load_tokens(args.new_valid)
        else:
            new_train, new_valid = split_train_valid(new_train, args.seq)

    old_tokens = load_tokens(args.old_data)
    old_batches = fixed_batches(load_tokens(args.old_valid), args.seq,
                                args.eval_batches, args.eval_batch_size,
                                args.seed + 100, device)
    new_batches = fixed_batches(new_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 200, device)
    before = _eval_both(model, old_batches, new_batches, tok)
    state_before = _clone(model)
    importance = fisher_importance(model, old_batches[:args.fisher_batches])

    sampler = MixtureWindows(old_tokens, new_train, args.seq, args.batch,
                             args.replay_fraction, args.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    history = []
    print(json.dumps({"event": "ingest_start", "params": count_params(model),
                      "device": str(device), "before": before,
                      "replay_fraction": args.replay_fraction}), flush=True)
    model.train()
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        x, y = sampler.next()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(args.amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.eval_every and (i % args.eval_every == 0 or i == args.steps):
            ev = _eval_both(model, old_batches, new_batches, tok)
            row = {"step": i, "train_loss": float(loss.detach()), **ev}
            history.append(row)
            print(json.dumps(row), flush=True)
            model.train()

    after = _eval_both(model, old_batches, new_batches, tok)
    state_after = _clone(model)
    rolled = False
    if after["old"]["bpb"] > before["old"]["bpb"] * (1 + args.max_old_regression):
        state_r = selective_rollback(state_before, state_after, importance,
                                     model, args.rollback_fraction)
        _load(model, state_r)
        after_r = _eval_both(model, old_batches, new_batches, tok)
        if after_r["old"]["bpb"] < after["old"]["bpb"]:
            after = after_r
            state_after = state_r
            rolled = True
            print(json.dumps({"event": "rollback", "after": after}), flush=True)

    accepted = should_accept(before["old"]["bpb"], after["old"]["bpb"],
                             before["new"]["bpb"], after["new"]["bpb"],
                             args.max_old_regression, args.min_new_gain)
    version = next_version(out)
    entry = {
        "version": version,
        "accepted": accepted,
        "rolled_back": rolled,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parent": str(Path(args.model)),
        "before": before,
        "after": after,
        "replay_fraction": args.replay_fraction,
        "steps": args.steps,
        "lr": args.lr,
        "max_old_regression": args.max_old_regression,
        "min_new_gain": args.min_new_gain,
        "rollback_fraction": args.rollback_fraction,
        "history": history,
        "path": None,
    }
    if accepted:
        version_dir = out / "versions" / version
        save_checkpoint(model, version_dir, tok_path(model, args), {
            "name": f"{meta.get('name', 'spiking')}-{version}",
            "parent": str(Path(args.model)),
            "ingest": {"version": version, "accepted": True},
            "metrics": {"before": before, "after": after},
        })
        entry["path"] = str(version_dir)
    elif args.keep_rejected:
        rejected = out / "rejected" / version
        save_checkpoint(model, rejected, tok_path(model, args), {
            "name": f"{meta.get('name', 'spiking')}-{version}-rejected",
            "parent": str(Path(args.model)),
            "ingest": {"version": version, "accepted": False},
            "metrics": {"before": before, "after": after},
        })
        entry["path"] = str(rejected)

    reg_path = out / "registry.json"
    registry = (json.loads(reg_path.read_text(encoding="utf-8"))
                if reg_path.exists() else {"versions": []})
    registry["versions"].append(entry)
    reg_path.write_text(json.dumps(registry, indent=2) + "\n",
                        encoding="utf-8")
    result = {"accepted": accepted, "version": version, "path": entry["path"],
              "before": before, "after": after, "rolled_back": rolled,
              "registry": str(reg_path)}
    print(json.dumps({"event": "ingest_done", **result}), flush=True)
    return result


def tok_path(model, args):
    """Use the checkpoint's tokenizer if present, else the CLI tokenizer."""
    root = Path(args.model)
    p = root / "tokenizer.json"
    if p.exists():
        return p
    return Path(args.tokenizer) if args.tokenizer else None
