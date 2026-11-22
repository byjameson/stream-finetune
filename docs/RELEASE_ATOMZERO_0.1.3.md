# atomzero 0.1.3 - full package description

Status: **published to PyPI as `atomzero 0.1.3` on 2026-11-22.**

Documentation release. No model or runtime behavior changed; the weights remain
`0.8.0`. The standalone package README now matches the Hugging Face model card
in scope and detail.

## What changed

- Expanded the GitHub/PyPI README with the full Atom Zero system description:
  model, training, RAG, documents/citations, evidence, tools, research,
  curiosity, learning ladder, provider, safety, privacy, benchmarks and limits.
- Removed stale streamft package/release links from the Atom Zero model card.
- Atom Zero links now point to the standalone package, its source directory and
  the `atomzero-v0.1.3` release.
- Added the `pip install atomzero==0.1.3` + `atomzero-quickstart` quickstart to
  the model card.

## Install

```powershell
pip install atomzero==0.1.3
atomzero-quickstart
```

The first run downloads the ~197 MB checkpoint/tokenizer, seeds a demo corpus
and opens the local UI. No streamft checkout is required.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.3/>
- GitHub source: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/atomzero-v0.1.3>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
