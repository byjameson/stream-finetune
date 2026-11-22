# atomzero 0.1.0 - standalone assistant package

Status: **published to PyPI as `atomzero 0.1.0` on 2026-11-22.**

> Superseded by `atomzero 0.1.1` for the packaged moderation head and
> package-identity fixes. The 0.1.0 split itself remains valid.

Atom Zero now has its own lightweight distribution, separate from the streamft
trainer. The model weights are unchanged from `0.8.0`.

## Install

```powershell
pip install atomzero
```

Optional local MiniLM reranker:

```powershell
pip install "atomzero[neural]"
```

## Dependencies

The standalone package installs only what the assistant needs:

```text
torch, tokenizers, numpy, pypdf>=4, python-docx>=1.1
```

It does **not** install `transformers`, `datasets`, `accelerate`, `bitsandbytes`,
`safetensors` or `huggingface-hub`. Those belong to the streamft trainer.

## What is included

- 49.2M transformer runtime and chat UI/API.
- PDF/DOCX/HTML/Markdown ingestion with page/section citations.
- Evidence panel and `/v1/documents/evidence`.
- SQLite FTS + hybrid retrieval, optional MiniLM reranker.
- Tools, tasks, session memory, export/import/Forget.
- Research queue, local teacher notes, curiosity approval gate.
- Optional local provider.
- Optional safety filter (off by default).
- Constrained `/v1/agent/plan` executor with allowlist and confirmation.
- 4 standalone smoke tests in the package; the shared source suite has 412
  tests and the 100-question document benchmark.

## Compatibility

`streamft 0.9.1` still bundles Atom Zero for existing users. The standalone
`atomzero` package is the recommended install for the assistant. The bundled
copy will be removed from streamft in a future major release.

This first split is a source copy: `atomzero` does not import `streamft`, so the
trainer stack cannot leak into the assistant install. A future cleanup can move
the shared adapter helper into a tiny common package if needed.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.0/>
- Source: <https://github.com/byjameson/stream-finetune/tree/main/packages/atomzero>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents/citations: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md>
- Hugging Face weights: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope weights: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
