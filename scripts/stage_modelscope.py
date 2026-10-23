"""Stage the two HF model repos for a ModelScope mirror.

Downloads the HF repo contents into dist/modelscope/<name>, then appends a
ModelScope-specific mirror note + the donation section (same addresses as the
main README) to each model card.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "modelscope"
REPOS = ["byjameson/lcga-v3-0.5b", "byjameson/lcga-p3-17m"]

DONATE = """
---

## ModelScope mirror

This is the ModelScope mirror of the Hugging Face repository
[{hf}](https://huggingface.co/{hf}). Full reports, raw evidence and the
reproduction commands live in the GitHub repository:
<https://github.com/byjameson/lcga-local>

## Support the work / Destek

This project is developed on a 6 GB laptop GPU. Donations go toward a larger GPU
and more compute so the experiments can scale; there is no paid support, token,
or investment offer.

- **EVM-compatible networks** (Ethereum, BNB Smart Chain, Polygon, Optimism,
  Arbitrum One, Base, Avalanche C-Chain):
  `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:**
  `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:**
  `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Please double-check the network and the full address before sending; crypto
transfers are irreversible. Donations are voluntary and do not buy influence
over results or reviews.

## License

MIT — see the GitHub repository for the full text.
"""


def main() -> int:
    for rid in REPOS:
        name = rid.split("/")[-1]
        dst = OUT / name
        print(f"[fetch] {rid} -> {dst}", flush=True)
        snapshot_download(rid, local_dir=str(dst), local_dir_use_symlinks=False)

        readme = dst / "README.md"
        text = readme.read_text(encoding="utf-8") if readme.exists() else ""
        if "## Support the work" in text:
            print(f"[skip] {name}: donation section already present", flush=True)
            continue
        readme.write_text(text.rstrip() + "\n" + DONATE.format(hf=rid),
                          encoding="utf-8")
        print(f"[ok] {name}: README.md ready ({readme.stat().st_size} bytes)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
