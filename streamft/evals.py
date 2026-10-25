"""`streamft eval` — does the model actually do the thing? No API, no judge model.

It runs a small prompt set locally and reports what can be measured without a second
model: repetition, refusal-phrase hits, length, latency, and optionally keyword adherence
when the prompt file carries an `expect` field.

    streamft eval --model Qwen/Qwen2.5-1.5B-Instruct
    streamft eval --model ./merged --prompts my_prompts.jsonl --max-new 96

Prompt file format (JSONL, one object per line):
    {"prompt": "...", "expect": ["optional", "keywords"]}
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import torch

# Ten English + ten Turkish prompts. Deliberately mundane: this is a smoke test, not a
# benchmark. `expect` is a keyword that should appear if the model understood the task.
BUILTIN = [
    {"prompt": "List three uses for a paperclip.", "expect": []},
    {"prompt": "Explain in one sentence why the sky is blue.", "expect": ["light", "scatter"]},
    {"prompt": "Give one reason to keep a food diary.", "expect": []},
    {"prompt": "Write a two-line rhyme about rain.", "expect": []},
    {"prompt": "What is 17 multiplied by 6?", "expect": ["102"]},
    {"prompt": "Name two vegetables that grow underground.", "expect": ["carrot", "potato"]},
    {"prompt": "Summarise the plot of a film about a lost dog in one sentence.", "expect": []},
    {"prompt": "Translate 'good morning' into Turkish.", "expect": ["günaydın", "gunaydin"]},
    {"prompt": "Give a one-sentence tip for better sleep.", "expect": []},
    {"prompt": "What does a red traffic light mean?", "expect": ["stop"]},
    {"prompt": "Bir cümlede su neden buharlaşır açıkla.", "expect": ["ısı", "isi", "sıcaklık"]},
    {"prompt": "Üç tane meyve say.", "expect": ["elma", "muz", "portakal", "çilek"]},
    {"prompt": "17 çarpı 6 kaç eder?", "expect": ["102"]},
    {"prompt": "Yağmurlu bir gün için iki satırlık bir dörtlük yaz.", "expect": []},
    {"prompt": "İyi bir uyku için tek cümlelik bir öneri ver.", "expect": []},
    {"prompt": "Kırmızı trafik ışığı ne anlama gelir?", "expect": ["dur"]},
    {"prompt": "Sabah kahvaltısı için iki öneri ver.", "expect": []},
    {"prompt": "Bir cümlede neden kitap okumalıyız açıkla.", "expect": []},
    {"prompt": "Türkiye'nin başkenti neresidir?", "expect": ["ankara"]},
    {"prompt": "Bir cümlede bisikletin avantajını yaz.", "expect": []},
]

REFUSAL = [
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "as an ai", "i'm sorry, but", "i am sorry, but", "cannot provide",
    "yardımcı olamam", "yardimci olamam", "üzgünüm", "uzgunum", "olarak yapay zeka",
]


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams (1.0 = every n-gram is a duplicate)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompts", default="", help="JSONL with {'prompt', 'expect'?}")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N prompts")
    ap.add_argument("--out", default="results/eval.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    prompts = BUILTIN
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    rows = []
    t_all = time.time()
    for i, item in enumerate(prompts):
        text = item["prompt"]
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(args.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        gen = out[0][ids["input_ids"].shape[1]:]
        answer = tok.decode(gen, skip_special_tokens=True).strip()
        low = answer.lower()
        row = {
            "i": i, "prompt": item["prompt"], "answer": answer,
            "tokens": int(gen.shape[0]), "seconds": round(dt, 2),
            "tok_s": round(int(gen.shape[0]) / max(dt, 1e-6), 1),
            "rep4": round(repetition_rate(answer), 3),
            "refusal": any(p in low for p in REFUSAL),
            "expect_hit": (None if not item.get("expect")
                           else any(k.lower() in low for k in item["expect"])),
        }
        rows.append(row)
        if not args.quiet:
            flag = "REFUSAL" if row["refusal"] else ("miss" if row["expect_hit"] is False else "")
            print(f"[{i+1}/{len(prompts)}] {row['tok_s']:.1f} tok/s rep4={row['rep4']:.2f} {flag}")
            print("   " + answer.replace("\n", " ")[:160])

    n = len(rows)
    expected = [r for r in rows if r["expect_hit"] is not None]
    summary = {
        "model": args.model, "n": n,
        "refusal_rate": round(sum(r["refusal"] for r in rows) / max(n, 1), 3),
        "mean_rep4": round(sum(r["rep4"] for r in rows) / max(n, 1), 3),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / max(n, 1), 1),
        "mean_tok_s": round(sum(r["tok_s"] for r in rows) / max(n, 1), 1),
        "keyword_hit_rate": (round(sum(1 for r in expected if r["expect_hit"]) / len(expected), 3)
                             if expected else None),
        "wall_s": round(time.time() - t_all, 1),
        "device": args.device,
    }
    print(json.dumps(summary, indent=1))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
