"""Structured generation: date -> ISO. Teacher labels + saved data."""
import json
import random
import re
from datetime import date, timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"models/Qwen2.5-1.5B-Instruct"
SYS = ("Convert the date to YYYY-MM-DD. Answer with the date only.\n\n"
       "Examples:\nMarch 3, 1999 -> 1999-03-03\n"
       "03/15/2001 -> 2001-03-15\n"
       "15.08.1987 -> 1987-08-15")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
ABBR = [m[:3] for m in MONTHS]


def make(n, seed):
    rng = random.Random(seed)
    rows = []
    start = date(1950, 1, 1)
    for _ in range(n):
        d = start + timedelta(days=rng.randrange(0, 60 * 365))
        # numeric formats use day > 12 so day/month order is unambiguous
        ok_numeric = d.day > 12
        fmts = ["%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
                "%A, %B %d, %Y", "%Y-%m-%d"]
        if ok_numeric:
            fmts += ["%m/%d/%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%m-%d-%Y"]
        f = rng.choice(fmts)
        text = d.strftime(f)
        # strftime %d gives zero-padded; make some variants natural
        if rng.random() < 0.3:
            text = text.replace(" 0", " ")
        rows.append({"text": text, "iso": d.isoformat()})
    return rows


@torch.no_grad()
def label(model, tok, texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": t}] for t in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=300).to(model.device)
        gen = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            raw = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
            out.append({"label": m.group(0) if m else "unknown", "raw": raw[:30]})
        if (i // batch) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    return out


def main():
    tr = make(6000, 0)
    te = make(500, 1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    for name, rows in [("train", tr), ("test", te)]:
        lab = label(model, tok, [r["text"] for r in rows])
        for r, l in zip(rows, lab):
            r["teacher"] = l["label"]
            r["raw"] = l["raw"]
        acc = sum(r["teacher"] == r["iso"] for r in rows) / len(rows)
        print(name, "teacher exact", round(acc, 4), "unknown", sum(r["teacher"] == "unknown" for r in rows), flush=True)
    json.dump({"train": tr, "test": te}, open(r"research/results/date_data.json", "w"), indent=1)


if __name__ == "__main__":
    main()
