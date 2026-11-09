"""Build the self-contained 500/1k/2k-example AG News demo.

    python demo/build_mini_ai.py

Reads `research/results/agnews_labels.json`, takes the first N/4 teacher-labelled rows
per class (the same deterministic sample as the research coreset), and writes
`demo/mini-ai.html` with the data embedded. The page trains a multinomial Naive Bayes in
the browser; nothing is sent anywhere.
"""
import json
from pathlib import Path

LABELS = ["Business", "World", "Sports", "Sci/Tech"]
ROOT = Path(__file__).resolve().parent.parent


def main():
    d = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    train = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in LABELS and len(train[r["label"]]) < 500:
            train[r["label"]].append({"text": r["text"], "label": r["label"]})
    test = [{"text": r["text"], "true": r["true"]}
            for r in d["test"] if r["true"] in LABELS]

    assert all(len(v) == 500 for v in train.values()), [len(v) for v in train.values()]
    assert len(test) == 400, len(test)

    data = {"train": train, "test": test}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    for n in (500, 1000, 2000):
        flat = []
        for l in LABELS:
            flat += train[l][: n // len(LABELS)]
        size = len(json.dumps(flat, ensure_ascii=False, separators=(",", ":")).encode())
        print(f"{n:>4} examples: {size} bytes ({size/1024:.1f} KB), "
              f"{size*8/1.5e9:.5f} bit per 1.5B teacher parameter")

    tpl = (Path(__file__).resolve().parent / "mini_ai_template.html").read_text(encoding="utf-8")
    if "__DATA__" not in tpl:
        raise SystemExit("template has no __DATA__ placeholder")
    html = tpl.replace("__DATA__", payload, 1)
    out = Path(__file__).resolve().parent / "mini-ai.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
