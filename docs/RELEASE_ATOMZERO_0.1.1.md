# atomzero 0.1.1 - standalone package patch

Status: **published to PyPI as `atomzero 0.1.1` on 2026-11-22.**

Patch release for the standalone Atom Zero package. The model weights are
unchanged from `0.8.0`.

## Fixed

- Included `moderation_head.json` in the package data, so the optional ML
  safety head works from a wheel install.
- Made the standalone package identity independent of streamft in the API
  (`owned_by: atomzero`) and research user-agent strings.
- Default MiniLM/head paths now resolve from the working directory (the model
  is still optional and not bundled).
- Added a subprocess test that imports `atomzero.chat_server` and asserts that
  `streamft` is not loaded.

## Install

```powershell
pip install atomzero==0.1.1
```

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.1/>
- Source: <https://github.com/byjameson/atom-zero>
- Project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Contact: **gokhancava@gmail.com**
