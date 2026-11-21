# Atom Zero / streamft 0.8.1

Patch release for the PyPI project page.

## Fixed

- The architecture image on <https://pypi.org/project/streamft/> was broken
  because PyPI does not resolve repository-relative `docs/...svg` paths.
- The README now uses an absolute raw-GitHub PNG:
  `https://raw.githubusercontent.com/byjameson/stream-finetune/main/docs/figures/atomzero_architecture.png`
- Added `docs/figures/atomzero_architecture.png`; the SVG remains for the repo.

## Unchanged

- Model weights and checkpoints: `0.8.0` (`byjameson/atom-zero-49m` on HF and
  ModelScope).
- Product/behavior/safety benchmarks, 392 tests.
- PyPI package name: `streamft`.

Install:

```bash
pip install streamft==0.8.1
```
