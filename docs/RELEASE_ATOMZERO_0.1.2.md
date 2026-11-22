# atomzero 0.1.2 - standalone quickstart

Status: **published to PyPI as `atomzero 0.1.2` on 2026-11-22.**

This release removes the last streamft-repository dependency from the Atom Zero
quickstart. Model weights are unchanged from `0.8.0`.

## New

```powershell
pip install atomzero==0.1.2
atomzero-quickstart
```

The first run:

- downloads `model.pt`, `config.json` and `chat_en_bpe8k.json` from
  <https://huggingface.co/byjameson/atom-zero-49m> (~197 MB);
- seeds a small demo corpus in `data/atomzero_demo.db`;
- starts the local UI at `http://127.0.0.1:8013`;
- opens the browser.

No `git clone`, no streamft checkout and no trainer install are required.

Options:

```powershell
atomzero-quickstart --provider          # local Ollama free-form chat
atomzero-quickstart --port 8014
atomzero-quickstart --safety-filter
atomzero-quickstart --no-download       # use cached files
atomzero-quickstart --no-browser
```

Cache: `~/.cache/atomzero/atom-zero-49m`.

## Verification

- 6 standalone atomzero tests pass.
- The quickstart downloads from the public HF repo and starts the same server.
- `streamft` remains the separate trainer package.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.2/>
- Source: <https://github.com/byjameson/atom-zero>
- HF weights: <https://huggingface.co/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
