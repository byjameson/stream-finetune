"""Train a 21M transformer chat baseline on the same clean SFT data.

This is a diagnostic: same tokenizer, data, start-token alignment and
assistant-only loss as the spiking run. If the transformer is much better, the
spiking architecture is the bottleneck; if not, data/tokens are.

    python research/train_transformer_chat.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamft.spiking import evaluate_bpb
from streamft.spiking_cli import (RandomWindows, fixed_windows, load_tokens,
                                  mask_user_labels)

try:
    from research.chat_eval import evaluate_assistant, metrics
except ImportError:
    from chat_eval import evaluate_assistant, metrics

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 512, layers: int = 5,
                 heads: int = 8, ff: int = 2048, max_seq: int = 512):
        super().__init__()
        self.vocab = vocab
        self.max_seq = max_seq
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln(h))


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 80,
             temperature: float = 0.8, top_k: int = 40,
             repetition_penalty: float = 1.1, banned_ids=()):
    model.eval()
    end_id = tok.token_to_id("<|end|>")
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    prompt_ids = tok.encode(prompt).ids
    ids = list(prompt_ids)
    seen = set(ids)
    stop = "max"
    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_seq:]], device=DEVICE)
        logits = model(x)[0, -1].float().clone()
        if banned_ids:
            logits[list(banned_ids)] = float("-inf")
        if repetition_penalty != 1.0:
            for tid in seen:
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            k = min(top_k, logits.numel())
            vals, idx = torch.topk(logits, k)
            logits = torch.full_like(logits, float("-inf"))
            logits[idx] = vals
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == end_id:
            stop = "end"
            break
        if nxt in (user_id, assistant_id):
            stop = "tag"
            break
        ids.append(nxt)
        seen.add(nxt)
    text = tok.decode(ids[len(prompt_ids):], skip_special_tokens=False)
    return text.strip(), stop


def save(model, out: Path, step: int, config: dict):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "config": config, "step": step},
               out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                     encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chat_en_clean_train.pt")
    ap.add_argument("--valid", default="data/chat_en_clean_valid.pt")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--out", default="checkpoints/transformer_chat_en_clean_21m")
    ap.add_argument("--init", default="")
    ap.add_argument("--tag", default="transformer_21m_sft")
    ap.add_argument("--result",
                    default="research/results/transformer_chat_en_clean_21m.json")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    torch.manual_seed(args.seed)
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    train_tokens = load_tokens(str(ROOT / args.data))
    valid_tokens = load_tokens(str(ROOT / args.valid))
    valid_batches = fixed_windows(valid_tokens, args.seq, 8, 4, 100, DEVICE)
    sampler = RandomWindows(train_tokens, args.seq, args.batch, args.seed,
                            DEVICE, start_token=0)
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")

    model = TransformerLM(tok.get_vocab_size(), args.dim, args.layers,
                          args.heads, args.ff).to(DEVICE)
    config = {"vocab": tok.get_vocab_size(), "dim": args.dim,
              "layers": args.layers, "heads": args.heads, "ff": args.ff,
              "seq": args.seq, "arch": "transformer"}
    if args.init:
        ck = torch.load(ROOT / args.init / "model.pt", map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["state"])
        print(json.dumps({"event": "init", "from": args.init,
                          "step": ck.get("step")}), flush=True)
    print(json.dumps({"event": "start", "params": sum(
        p.numel() for p in model.parameters()), "config": config}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    out = ROOT / args.out
    history = []

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup)
                       / max(args.steps - args.warmup, 1))
        return 0.1 * args.lr + 0.45 * args.lr * (
            1 + math.cos(math.pi * progress))

    model.train()
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(i)
        x, y = sampler.next()
        y = mask_user_labels(y, user_id, assistant_id)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % args.eval_every == 0:
            model.eval()
            bpb = evaluate_bpb(model, tok, valid_batches)
            assistant = evaluate_assistant(model, tok, valid_batches)
            rec = {"event": "eval", "step": i,
                   "train_loss": float(loss.detach()),
                   "step_s": (time.perf_counter() - t0) / args.eval_every,
                   "lr": lr_at(i), "bpb": bpb["bpb"],
                   "assistant": assistant}
            history.append(rec)
            print(json.dumps(rec), flush=True)
            t0 = time.perf_counter()
            model.train()
        if args.sample_every and i % args.sample_every == 0:
            torch.manual_seed(1000 + i)
            text, stop = generate(
                model, tok, "<|user|> hello<|end|><|assistant|>")
            print(json.dumps({"event": "sample", "step": i, "stop": stop,
                              "text": text}), flush=True)
        if args.save_every and i % args.save_every == 0:
            save(model, out, i, config)
            print(json.dumps({"event": "checkpoint", "step": i}), flush=True)

    save(model, out, args.steps, config)
    model.eval()
    evals = {}
    for name, rel in {
        "chat": "data/chat_en_clean_valid.pt",
        "stories": "data/chat_en_clean_stories_valid.pt",
        "wiki": "data/wiki_en_valid.pt",
        "mixed": "data/chat_en_clean_valid_mixed.pt",
    }.items():
        tokens = load_tokens(str(ROOT / rel))
        batches = fixed_windows(tokens, args.seq, 8, 8, 0, DEVICE)
        evals[name] = evaluate_bpb(model, tok, batches)
        a = evaluate_assistant(model, tok, batches)
        if a:
            evals[name]["assistant"] = a
    prompts = ["hello", "how are you?", "my name is Alex, remember me",
               "what is the weather today?", "tell me a short story",
               "add task: buy milk"]
    samples = []
    for i, prompt in enumerate(prompts):
        for j in range(2):
            torch.manual_seed(1000 + i * 10 + j)
            text, stop = generate(
                model, tok, f"<|user|> {prompt}<|end|><|assistant|>")
            samples.append({"prompt": prompt, **metrics(text, stop)})
    gate = {
        "chat_bpb_below_2_5": evals["chat"]["bpb"] < 2.5,
        "samples_nonempty": all(s["chars"] > 0 for s in samples),
        "alpha_at_least_0_5": all(s["alpha"] >= 0.5 for s in samples),
        "no_tag_leak": all(s["tag_leak"] == 0 for s in samples),
    }
    payload = {"tag": args.tag, "arch": "transformer",
               "config": config, "step": args.steps, "bpb": evals,
               "gate": gate,
               "verdict": "PASS_CHAT_SMOKE" if all(gate.values())
               else "BLOCKED", "samples": samples, "history": history}
    result = ROOT / args.result
    result.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"event": "done", "result": str(result),
                      "verdict": payload["verdict"]}), flush=True)


if __name__ == "__main__":
    main()
