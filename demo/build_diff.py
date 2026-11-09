"""Build the adapter autopsy page from research/results/adapter_diff.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    d = json.load(open(ROOT / "research/results/adapter_diff.json", encoding="utf-8"))
    payload = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    tpl = (ROOT / "demo/diff_template.html").read_text(encoding="utf-8")
    if "__DIFF__" not in tpl:
        raise SystemExit("template has no __DIFF__")
    html = tpl.replace("__DIFF__", payload, 1)
    out = ROOT / "demo/diff.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
