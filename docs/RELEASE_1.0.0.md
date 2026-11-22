# streamft 1.0.0 - trainer only

Status: **published to PyPI as `streamft 1.0.0` on 2026-11-22.**

`streamft` is now the low-VRAM trainer package only. The assistant modules that
were bundled with the 0.9.x line are no longer part of this distribution.

## Breaking changes

- The assistant modules (`chat_server`, `doc_memory`, `agent_runtime`,
  `tool_planner` and related helpers) were removed from the wheel.
- The assistant console script was removed from the streamft wheel.
- `pypdf` and `python-docx` are no longer streamft dependencies.
- `streamft 0.9.2` remains the final compatibility release with the bundled
  assistant copy.

## streamft 1.0 contains

- `streamft fit/train/prepare/data/cpu/synth/eval/merge/serve/adapter/verify`.
- The disk-mapped NF4 loader, LoRA trainer, quantization, metrics and trainer
  evidence tools.
- `streamft[train]` as a compatibility alias for the training stack.

## Verification

- **206 trainer/source tests pass.**
- Package build and install pass.
- `streamft --help` and the trainer subcommands load without the removed
  assistant modules.
- `streamft doctor` and `streamft fit` were smoke-tested from the published
  wheel.

## Install

```powershell
pip install streamft==1.0.0
```

## Links

- PyPI: <https://pypi.org/project/streamft/1.0.0/>
- ModelScope trainer package: <https://modelscope.cn/models/byjameson/streamft>
- GitHub release: <https://github.com/byjameson/stream-finetune/releases/tag/v1.0.0>
- Trainer guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>
- Contact: **gokhancava@gmail.com**
