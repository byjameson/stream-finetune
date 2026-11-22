# streamft 0.9.2 - published link cleanup

Status: **published to PyPI as `streamft 0.9.2` on 2026-11-22.**

Documentation/link patch. No model or runtime code changed; weights remain
`0.8.0`.

## Fixed

- PyPI/HF/ModelScope documentation links are absolute GitHub URLs instead of
  relative paths that 404 outside GitHub.
- Removed the stale "standalone assistant is future work" wording from the
  published package description; `atomzero 0.1.1` is live.
- Added PyPI project URLs for source, full documentation, model card,
  benchmarks and the Atom Zero package.

## Packages

```powershell
pip install atomzero==0.1.1   # standalone assistant
pip install streamft==0.9.2    # trainer (bundled Atom Zero compatibility)
```

## Links

- PyPI: <https://pypi.org/project/streamft/0.9.2/>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/v0.9.2>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Atom Zero package: <https://pypi.org/project/atomzero/0.1.1/>
- Contact: **gokhancava@gmail.com**
