# Atom Zero 49.2M — a local assistant that cites its sources

Atom Zero is a local, source-cited assistant built around a **49.2M parameter
model trained from scratch**. Facts live in a local RAG index, not in the model
weights; the model handles wording and behavior while deterministic tools,
documents, memory and citations do the rest.

This is the standalone `atomzero` package. The low-VRAM trainer remains in the
separate [`streamft`](https://pypi.org/project/streamft/) package.

## Install

```powershell
pip install atomzero
```

Optional local MiniLM reranker:

```powershell
pip install "atomzero[neural]"
```

The model weights are not bundled. Download them from
[Hugging Face](https://huggingface.co/byjameson/atom-zero-49m) or
[ModelScope](https://modelscope.cn/models/byjameson/atom-zero-49m).

## Run

```powershell
atomzero `
  --model checkpoints/transformer_behavior_50m_scaled `
  --tokenizer data/chat_en_bpe8k.json `
  --docs data/docs.db
```

Then open the local page (default `http://127.0.0.1:8013`).

## Features

- **PDF, DOCX, HTML and Markdown ingestion** with page/section citations.
- **Evidence API/panel** that opens the exact local chunks behind a citation.
- **Deterministic tools**: time, safe arithmetic, tasks, name/likes memory.
- **Constrained plan API** (`/v1/agent/plan`): allowlist, three-step cap and
  explicit confirmation for task/research side effects.
- **Sessions**: transcript restore, export/import, exact Forget.
- **Research queue**: approved-only Wikipedia fetch and local teacher notes.
- **Optional local provider**: Ollama/OpenAI-compatible free-form chat while RAG,
  tools and memory stay local.
- **No telemetry** and no hosted service.

## Honest limits

- The raw 49.2M model scores **2/10** on the no-RAG knowledge probe.
- The product scores **21/21** on the narrow authored product benchmark because
  RAG and tools do the work.
- The core context is **128 tokens**; it is not a general reasoning or coding
  model.
- No OCR; scanned PDFs need OCR before ingestion.
- Safety is optional and off by default.

## Links

- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents/citations: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md>
- Benchmarks: <https://github.com/byjameson/stream-finetune/blob/main/docs/BENCHMARKS.md>
- Model card: <https://huggingface.co/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**

Apache-2.0 for code and weights; data attribution in `NOTICE`.
