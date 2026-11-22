# streamft 1.0.0 - trainer only

Status: **published to PyPI as `streamft 1.0.0` on 2026-11-22.**

This is the clean package split. `streamft` is now the low-VRAM trainer only;
Atom Zero lives in the independent `atomzero` package. Model weights remain
`0.8.0`.

## Breaking changes

- `streamft.chat_server`, `streamft.doc_memory`, `streamft.agent_runtime`,
  `streamft.tool_planner` and the other Atom Zero modules moved to `atomzero`.
- The `atomzero` console script was removed from the streamft wheel.
- `pypdf` and `python-docx` are no longer streamft dependencies.
- The old imports can be migrated with [`MIGRATION_1.0.md`](MIGRATION_1.0.md).
- `streamft 0.9.2` remains the last compatibility release with the bundled
  Atom Zero copy.

## streamft 1.0 still contains

- `streamft fit/train/prepare/data/cpu/synth/eval/merge/serve/adapter/verify`.
- The disk-mapped NF4 loader, LoRA trainer, quantization, metrics and trainer
  evidence tools.
- `streamft[train]` as a compatibility alias for the training stack.

## Atom Zero

```powershell
pip install atomzero==0.1.5
atomzero --model checkpoints/transformer_behavior_50m_scaled \
  --tokenizer data/chat_en_bpe8k.json --docs data/docs.db
```

- PyPI: <https://pypi.org/project/atomzero/0.1.5/>
- Source: <https://github.com/byjameson/atom-zero>

## Verification

- **412 streamft/source tests pass** with `atomzero` installed.
- **4 standalone atomzero tests pass.**
- CI runs both, the document benchmark, safety/injection red-team and both
  package builds.
- Standalone atomzero server smoke test: `/health` OK with the real 49.2M
  checkpoint.

## Install

```powershell
pip install streamft==1.0.0
```

## Links

- PyPI: <https://pypi.org/project/streamft/1.0.0/>
- ModelScope trainer package: <https://modelscope.cn/models/byjameson/streamft>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/v1.0.0>
- Trainer guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>
- Migration guide: <https://github.com/byjameson/stream-finetune/blob/main/docs/MIGRATION_1.0.md>
- Atom Zero: <https://pypi.org/project/atomzero/0.1.5/>
- Contact: **gokhancava@gmail.com**
