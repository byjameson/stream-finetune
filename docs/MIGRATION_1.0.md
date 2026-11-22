# Migrating from streamft 0.9.x to 1.0

`streamft 1.0` is the low-VRAM trainer only. Atom Zero moved to the standalone
`atomzero` package.

## Install

```powershell
pip install atomzero     # assistant
pip install streamft     # trainer
```

You can install both; they do not depend on each other.

## Import mapping

| streamft 0.9.x | streamft 1.0 / atomzero |
|---|---|
| `streamft.chat_server` | `atomzero.chat_server` |
| `streamft.agent_runtime` | `atomzero.agent_runtime` |
| `streamft.doc_memory` | `atomzero.doc_memory` |
| `streamft.doc_ingest` | `atomzero.doc_ingest` |
| `streamft.tool_planner` | `atomzero.tool_planner` |
| `streamft.safety` | `atomzero.safety` |
| `streamft.ml_safety` | `atomzero.ml_safety` |
| `streamft.feedback` | `atomzero.feedback` |
| `streamft.knowledge_gaps` | `atomzero.knowledge_gaps` |
| `streamft.research_queue` | `atomzero.research_queue` |
| `streamft.curiosity` | `atomzero.curiosity` |
| `streamft.teacher_notes` | `atomzero.teacher_notes` |
| `streamft.teacher_provider` | `atomzero.teacher_provider` |
| `streamft.neural_embedder` | `atomzero.neural_embedder` |
| `streamft.transformer` | `atomzero.transformer` |
| `streamft.web_research` | `atomzero.web_research` |

`streamft.lora` remains in the trainer package. The standalone assistant has its
own adapter-application copy inside `atomzero.lora`.

## Entry points

| old | new |
|---|---|
| `streamft.chat_server:main` (`atomzero` console script in 0.9.x) | `atomzero.chat_server:main` (`atomzero` console script) |
| `streamft fit/train/...` | unchanged |

## What stays in streamft

- `streamft fit`, `train`, `prepare`, `data`, `cpu`, `synth`, `eval`, `merge`,
  `serve`, `adapter`, `verify`, `spiking`, `doctor`.
- The streaming NF4 loader, LoRA trainer, quantization, metrics and trainer
  evidence tools.
- `streamft[train]` remains a compatibility alias; the training stack is already
  in the default install.

## Compatibility release

`streamft 0.9.2` is the last release that contains the bundled Atom Zero
compatibility copy. Pin it only if you cannot migrate:

```powershell
pip install streamft==0.9.2
```

The migration is a rename, not a data migration: the SQLite database and session
JSON files keep the same schema and paths. Point `atomzero --docs` at the same
database you used with `streamft 0.9.x`.
