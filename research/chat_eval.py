"""Evaluate a from-scratch Turkish chat checkpoint.

    python research/chat_eval.py --model checkpoints/spiking_chat_tr_21m \
        --out research/results/spiking_chat_tr_21m.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb, load_checkpoint
from streamft.spiking_cli import fixed_windows, load_tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TURKISH = set("çğıöşüÇĞİÖŞÜ")


def chat_generate(model, tok, user: str, max_new_tokens: int = 80,
                  temperature: float = 0.8, top_k: int = 40,
                  repetition_penalty: float = 1.1) -> str:
    end_id = tok.token_to_id("<|end|>")
    prompt = f"<|user|> {user}<|end|><|assistant|>"
    ids = tok.encode(prompt).ids
    state = model.init_state(1, DEVICE)
    last = None
    for tid in ids:
        last, state = model.step(torch.tensor([tid], device=DEVICE), state)
    seen = set(ids)
    out = []
    for _ in range(max_new_tokens):
        assert last is not None
        logits = last[0].float().clone()
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
            break
        out.append(nxt)
        seen.add(nxt)
        last, state = model.step(torch.tensor([nxt], device=DEVICE), state)
    text = tok.decode(out, skip_special_tokens=False)
    return text.strip()


def metrics(text: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    trigrams = list(zip(words, words[1:], words[2:]))
    repeat = (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
    return {
        "text": text,
        "chars": len(text),
        "alpha": round(sum(c.isalpha() for c in text)
                       / max(len(text), 1), 4),
        "turkish_chars": sum(c in TURKISH for c in text),
        "repeat3": round(repeat, 4),
        "tag_leak": text.count("<|"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lang", choices=["tr", "en"], default="en")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    model, tok, meta = load_checkpoint(str(ROOT / args.model))
    model.to(DEVICE)
    stories = ("data/stories_en_valid.pt" if args.lang == "tr"
               else "data/stories_en_valid_en.pt")
    evals = {}
    for name, rel in {
        "chat": f"data/chat_{args.lang}_valid.pt",
        "wiki": f"data/wiki_{args.lang}_valid.pt",
        "stories": stories,
        "mixed": f"data/chat_{args.lang}_valid_mixed.pt",
    }.items():
        tokens = load_tokens(str(ROOT / rel))
        batches = fixed_windows(tokens, 128, 8, 8, 0, DEVICE)
        evals[name] = evaluate_bpb(model, tok, batches)

    prompts = {
        "tr": [
            "merhaba",
            "nasılsın?",
            "adım Ali, beni hatırla",
            "bugün hava nasıl?",
            "bana kısa bir hikaye anlat",
            "görev ekle: süt al",
        ],
        "en": [
            "hello",
            "how are you?",
            "my name is Alex, remember me",
            "what is the weather today?",
            "tell me a short story",
            "add task: buy milk",
        ],
    }[args.lang]
    samples = []
    for i, prompt in enumerate(prompts):
        for j in range(args.samples):
            torch.manual_seed(1000 + i * 10 + j)
            text = chat_generate(model, tok, prompt)
            rec = {"prompt": prompt, **metrics(text)}
            samples.append(rec)

    payload = {
        "tag": args.tag or Path(args.model).name,
        "lang": args.lang,
        "model": args.model,
        "config": model.config.__dict__,
        "step": meta.get("step"),
        "bpb": evals,
        "samples": samples,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
