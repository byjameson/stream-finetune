"""Teacher-label comparison: 5k rows labelled by Qwen2.5-3B-Instruct vs human labels.

The 130 KB demo and the 1.7 MB max mode use different label sources. This asks the
question directly on the same 5,000 rows:

  human labels vs 3B-Instruct teacher labels -> same TF-IDF linear student -> same test

Rows are disjoint from the existing 6k teacher labels and the 400-row test set. The
3B model is loaded in 4-bit so it fits the 6 GB card.

    python research/agnews_teacher_3b.py --rows 5000
"""
import argparse
import json
import time
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          )

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

MODEL = "models/Qwen2.5-3B-Instruct"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # AG News integer order
SYS = ("You classify news text into exactly one of these labels: World, Sports, "
       "Business, Sci/Tech. Reply with the label only, nothing else.\n\nExamples:\n"
       "Text: Stocks rally as oil prices fall.\nLabel: Business\n"
       "Text: United beat City 2-1 in the derby.\nLabel: Sports\n"
       "Text: New telescope detects water on distant planet.\nLabel: Sci/Tech\n"
       "Text: Leaders meet in Geneva for peace talks.\nLabel: World")


@torch.no_grad()
def label_texts(model, tok, texts, batch=8, max_new=5):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": t[:500]}] for t in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=900).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            low = txt.lower()
            hit = next((c for c in LABELS if c.lower() in low), "unknown")
            out.append({"text": chunk[j], "label": hit, "raw": txt[:40]})
        if (i // batch) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    return out


def linear(rows, y, test, ytest, max_features=20000):
    p = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                            max_features=max_features)),
                  ("lr", LogisticRegression(max_iter=1000, C=4.0))])
    p.fit([r["text"] for r in rows], y)
    pred = p.predict([r["text"] for r in test])
    return round(accuracy_score(ytest, pred), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    d = json.load(open("research/results/agnews_labels.json", encoding="utf-8"))
    used = {r["text"] for r in d["train"]} | {r["text"] for r in d["test"]}
    test = [r for r in d["test"] if r["true"] in LABELS]

    ds = load_dataset("fancyzhx/ag_news", split="train")
    per = args.rows // len(LABELS)
    by = defaultdict(list)
    for r in ds:
        if r["text"] in used:
            continue
        label = LABELS[r["label"]]
        if len(by[label]) < per:
            by[label].append({"text": r["text"], "true": label})
        if all(len(v) >= per for v in by.values()):
            break
    rows = [r for l in LABELS for r in by[l]]
    print(f"rows {len(rows)} (disjoint from the saved labels), test {len(test)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda",
                                                 quantization_config=q)
    t0 = time.time()
    labelled = label_texts(model, tok, [r["text"] for r in rows], batch=args.batch)
    for r, lab in zip(rows, labelled):
        r.update(lab)
    print(f"labelled in {time.time() - t0:.0f}s; unknowns "
          f"{sum(1 for r in rows if r['label'] == 'unknown')}", flush=True)

    ytrue = [r["true"] for r in rows]
    yteach = [r["label"] for r in rows]
    teacher_acc = accuracy_score(ytrue, yteach)
    test_rows = [{"text": r["text"]} for r in test]
    ytest = [r["true"] for r in test]
    human_acc = linear(rows, ytrue, test_rows, ytest)
    teacher_student_acc = linear(rows, yteach, test_rows, ytest)
    out = {"rows": len(rows), "test": len(test),
           "teacher_acc": round(float(teacher_acc), 4),
           "human_student_acc": human_acc,
           "teacher_student_acc": teacher_student_acc,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/agnews_teacher3b.json", "w"), indent=1)
    json.dump(rows, open("research/results/agnews_teacher3b_5k.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
