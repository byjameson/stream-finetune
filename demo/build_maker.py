"""Build the Tiny AI Maker page with a 200-row AG News sample."""
import json
from pathlib import Path

LABELS = ["Business", "World", "Sports", "Sci/Tech"]
ROOT = Path(__file__).resolve().parent.parent


def main():
    d = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    by = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in by and len(by[r["label"]]) < 50:
            by[r["label"]].append({"text": r["text"], "label": r["label"]})
    rows = [r for l in LABELS for r in by[l]]
    assert len(rows) == 200, len(rows)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")

    tpl = (ROOT / "demo/maker_template.html").read_text(encoding="utf-8")
    if "__SAMPLE__" not in tpl:
        raise SystemExit("template has no __SAMPLE__")
    html = tpl.replace("__SAMPLE__", payload, 1)
    out = ROOT / "demo/maker.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(rows)} sample rows)")


if __name__ == "__main__":
    main()
