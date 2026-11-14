"""Longer phase-B test: replay and replay+EWC on the 21M model.

Compared with the 300-step run, this uses 600 phase-B steps and also measures
greedy/sampled generation quality for each arm.

    python research/spiking_long_replay.py --steps-b 600
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from research.spiking_replay import RandomWindows
from research.spiking_replay_ewc import train_b
from research.spiking_shift import (
    clone_state,
    eval_both,
    fixed_batches,
    load_state,
    load_tokens,
)
from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    fisher_importance,
    generate,
    load_checkpoint,
)
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Once upon a time", "The little dog",
           "One day, a girl named Lily", "Tom and his friend"]


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def quality(samples):
    toks = [_tokens(s) for s in samples]
    flat = [t for row in toks for t in row]
    uni = {t for t in flat}
    bi = {g for row in toks for g in ngrams(row, 2)}
    four = [g for row in toks for g in ngrams(row, 4)]
    repeat = 0.0
    if four:
        counts = Counter(four)
        repeat = sum(c - 1 for c in counts.values() if c > 1) / len(four)
    return {
        "chars_mean": sum(len(s) for s in samples) / max(len(samples), 1),
        "distinct_1": len(uni) / max(len(flat), 1),
        "distinct_2": len(bi) / max(sum(max(len(r) - 1, 0) for r in toks), 1),
        "repeat_4gram": repeat,
        "samples": samples,
    }


STOP = set("""a an the and or but if of to in on at for with as by from is are was
were be been being it its this that these those there here he she they we you i
his her their our your my me him them us do does did will would can could should
may might must not no so then than just very really about into over under""".split())


def _content(text):
    return {t for t in _tokens(text) if t.isalnum() and t not in STOP}


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l(ref, gen):
    r, g = _tokens(ref), _tokens(gen)
    if not r or not g:
        return 0.0
    lcs = _lcs(r, g)
    p = lcs / len(g)
    rec = lcs / len(r)
    return 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)


@torch.no_grad()
def _greedy_ids(model, tok, prompt_ids, n, device):
    ids = prompt_ids.tolist()
    state = model.init_state(1, device)
    last = None
    for tid in ids:
        last, state = model.step(torch.tensor([tid], device=device), state)
    out = []
    for _ in range(n):
        nxt = int(last.argmax(dim=-1).item())
        out.append(nxt)
        last, state = model.step(torch.tensor([nxt], device=device), state)
    return tok.decode(out)


def semantic_probe(model, tok, valid, device, n=8, gen_len=24):
    rows = []
    for i in range(n):
        start = 1000 + i * 500
        if start + 24 + gen_len >= len(valid):
            break
        prompt = valid[start:start + 24]
        ref = tok.decode(valid[start + 24:start + 24 + gen_len].tolist())
        text = _greedy_ids(model, tok, prompt, gen_len, device)
        cr, cg = _content(ref), _content(text)
        inter = len(cr & cg)
        prec = inter / max(len(cg), 1)
        rec = inter / max(len(cr), 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        rows.append({"rouge_l": _rouge_l(ref, text), "content_f1": f1,
                     "ref": ref, "gen": text})
    return {
        "n": len(rows),
        "rouge_l": sum(r["rouge_l"] for r in rows) / max(len(rows), 1),
        "content_f1": sum(r["content_f1"] for r in rows) / max(len(rows), 1),
        "rows": rows,
    }


def samples_for(model, tok, device, max_new=48):
    out = []
    for p in PROMPTS:
        out.append(generate(model, tok, p, max_new, device,
                            temperature=0.8, top_k=50,
                            repetition_penalty=1.1))
    return quality(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", default="",
                    help="optional existing checkpoint to use as phase A (skips training)")
    ap.add_argument("--steps-a", type=int, default=300)
    ap.add_argument("--steps-b", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr-a", type=float, default=5e-4)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_long_replay.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    a_train = load_tokens(str(ROOT / "data/ts_train.pt"), args.a_tokens)
    a_valid = load_tokens(str(ROOT / "data/ts_valid.pt"))
    b_train = load_tokens(str(ROOT / "data/ts_wt2_train.pt"))
    b_valid = load_tokens(str(ROOT / "data/ts_wt2_valid.pt"))
    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 100, device)
    new_batches = fixed_batches(b_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 200, device)

    t0 = time.perf_counter()
    if args.init_model:
        model, tok, _meta = load_checkpoint(args.init_model)
        model.to(device)
    else:
        tok = Tokenizer.from_file(str(ROOT / "data/ts_bpe8k.json"))
        model = SpikingLM(SpikingConfig(vocab=tok.get_vocab_size(), dim=args.dim,
                                        layers=args.layers, adapt=args.adapt)).to(device)
        _losses_a, _ = train_b(
            model, RandomWindows(a_train, args.seq, args.batch, args.seed, device),
            args.steps_a, args.lr_a, device, args.amp)
    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_both(model, old_batches, new_batches, tok)
    a_quality = samples_for(model, tok, device)
    a_sem = semantic_probe(model, tok, a_valid, device)
    a_quality["semantic"] = {k: v for k, v in a_sem.items() if k != "rows"}
    importance = fisher_importance(model, old_batches[:args.fisher_batches])
    flat = torch.cat([v.float().flatten() for v in importance.values()])
    scale = float(flat.mean().clamp_min(1e-12))
    fisher = {k: (v / scale).to(device) for k, v in importance.items()}
    print(json.dumps({"event": "after_a", "params": sum(p.numel() for p in model.parameters()),
                      "sec": round(time.perf_counter() - t0, 1), "before": before,
                      "quality": {k: v for k, v in a_quality.items() if k != "samples"}}),
          flush=True)

    arms = [("naive", 0.0, 0.0), ("replay_20", args.replay, 0.0),
            ("replay_20_ewc", args.replay, args.ewc_lambda)]
    rows = []
    for name, replay, lam in arms:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 replay, args.seed + 1000, device)
        losses, sec = train_b(model, sampler, args.steps_b, args.lr_b,
                              device, args.amp,
                              anchor=anchor if lam else None,
                              fisher=fisher if lam else None,
                              ewc_lambda=lam)
        after = eval_both(model, old_batches, new_batches, tok)
        q = samples_for(model, tok, device)
        sem = semantic_probe(model, tok, a_valid, device)
        q["semantic"] = {k: v for k, v in sem.items() if k != "rows"}
        row = {"arm": name, "replay": replay, "ewc_lambda": lam,
               "train_loss": losses[-1], "sec": round(sec, 1),
               "before": before, "after": after,
               "quality": {k: v for k, v in q.items() if k != "samples"},
               "semantic_rows": sem["rows"],
               "samples": q["samples"]}
        rows.append(row)
        print(json.dumps({"event": "arm", "arm": name,
                          "old": after["old"]["bpb"], "new": after["new"]["bpb"],
                          "quality": row["quality"], "sec": row["sec"]}), flush=True)

    payload = {
        "question": "Does replay-first hold with a longer phase B, and does soft EWC help?",
        "config": vars(args),
        "phase_a_quality": a_quality,
        "after_a": before,
        "arms": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
