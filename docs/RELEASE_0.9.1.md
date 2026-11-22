# streamft 0.9.1 - streamft trainer compatibility

Status: **published to PyPI as `streamft 0.9.1` on 2026-11-22.**

This is a packaging fix for the streamft audience. It keeps every Atom Zero
0.9.0 feature and restores the low-VRAM trainer's dependency set to the default
install.

## Why

`0.9.0` moved `transformers`, `safetensors`, `accelerate`, `datasets`,
`huggingface-hub` and `bitsandbytes` into a `[train]` extra. That made a clean
`pip install streamft` insufficient for the original trainer workflow. Existing
0.8.x environments often still had those packages, but new installs and upgrade
scripts could break. That was the wrong trade-off for a package whose original
product is the trainer.

## What changed

- The training stack is back in the default `streamft` dependencies.
- `pip install streamft` installs **Atom Zero + the streamft trainer**, matching
  the 0.8.x expectation.
- `streamft[train]` and `streamft[neural]` remain valid compatibility aliases
  for scripts written against 0.9.0.
- All 0.9.0 features remain: PDF/DOCX/HTML/Markdown ingestion, page/section
  citations, evidence panel/API, 100-question document benchmark and the
  constrained plan API.
- The standalone lightweight assistant is now published separately as
  `atomzero 0.1.1`. `streamft` keeps the bundled copy for compatibility until a
  future major release.

## Install

```powershell
pip install streamft==0.9.1
```

## Verification

- **412 tests pass** locally.
- CI runs unit tests, the document benchmark with `--check`, safety/injection
  red-team, and the package build/install.
- PyPI wheel/sdist are built from the same source tag.

## Links

- PyPI: <https://pypi.org/project/streamft/0.9.1/>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/v0.9.1>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- streamft trainer guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>
- Atom Zero documents: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
