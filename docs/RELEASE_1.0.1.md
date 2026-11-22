# streamft 1.0.1 - trainer-only metadata cleanup

Status: **published to PyPI as `streamft 1.0.1` on 2026-11-22.**

Documentation and packaging cleanup. No trainer behavior changed.

## What changed

- Removed the remaining assistant references from the package description,
  README, project URLs and docstrings.
- Removed the assistant package, docs, examples and tests from this repository.
  They now live in the separate assistant project.
- Updated the GitHub repository description and topics to trainer-only.
- Rebuilt the wheel from the trainer-only tree.

## Install

```powershell
pip install streamft==1.0.1
```

## Verification

- **200 trainer/source tests pass.**
- Package build, `twine check` and wheel install pass.
- `streamft --help`, `streamft doctor` and `streamft fit` were smoke-tested from
  the published wheel.

## Links

- PyPI: <https://pypi.org/project/streamft/1.0.1/>
- ModelScope trainer package: <https://modelscope.cn/models/byjameson/streamft>
- GitHub: <https://github.com/byjameson/stream-finetune>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/v1.0.1>
- Trainer guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>
- Contact: **gokhancava@gmail.com**
