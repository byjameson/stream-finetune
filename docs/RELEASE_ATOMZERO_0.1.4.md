# atomzero 0.1.4 - standalone repository

Status: **published to PyPI as `atomzero 0.1.4` on 2026-11-22.**

> Superseded by `atomzero 0.1.6`: LinkedIn and ModelScope are canonical
> publication surfaces again.

Repository release. The package source now lives at
<https://github.com/byjameson/atom-zero> instead of a subdirectory of the
trainer monorepo. No model or runtime behavior changed; the weights remain
`0.8.0`.

## What changed

- Created the standalone repository <https://github.com/byjameson/atom-zero>
  with the package README, LICENSE, NOTICE, source and tests at the root.
- Moved the package history with `git subtree split`; the monorepo copy remains
  as a transition mirror.
- PyPI project URLs now point to the standalone repository.
- Atom Zero release links target `byjameson/atom-zero` and `v0.1.4`.
- Canonical publication surfaces are now GitHub, Hugging Face, PyPI, dev.to and
  Medium. ModelScope and LinkedIn are no longer updated for new releases.

## Install

```powershell
pip install atomzero==0.1.4
atomzero-quickstart
```

The first run downloads the ~197 MB checkpoint/tokenizer from Hugging Face,
seeds a demo corpus and opens the local UI. No streamft checkout is required.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.4/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.1.4>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-a-49m-local-assistant-that-says-i-dont-know-1p7m>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-a-49m-local-assistant-that-says-i-dont-know-df1ebc0e2e78>
- Contact: **gokhancava@gmail.com**
