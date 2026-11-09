"""Build the adapter fingerprint page from research/results/watermark.json."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    d = json.load(open(ROOT / "research/results/watermark.json", encoding="utf-8"))
    paths = {
        "fp32": ROOT / "checkpoints/wm_qwen3b_lora.pt",
        "8-bit": ROOT / "checkpoints/wm_qwen3b_lora_q8.pt",
        "4-bit": ROOT / "checkpoints/wm_qwen3b_lora_q4.pt",
    }
    for v in d["variants"]:
        v["bytes"] = os.path.getsize(paths[v["config"]])
    d["q4_bytes"] = os.path.getsize(paths["4-bit"])
    payload = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    tpl = (ROOT / "demo/watermark_template.html").read_text(encoding="utf-8")
    if "__WM__" not in tpl:
        raise SystemExit("template has no __WM__")
    html = tpl.replace("__WM__", payload, 1)
    out = ROOT / "demo/watermark.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
