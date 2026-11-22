# Atom Zero 49.2M — a local assistant that cites its sources

Atom Zero is a local, source-cited assistant built around a **49.2M parameter
model trained from scratch**. Facts live in a local RAG index, not in the model
weights; the model handles wording and behavior while deterministic tools,
documents, memory and citations do the rest.

This is the standalone **`atomzero`** package. The low-VRAM trainer is a separate
package and is **not required** to run Atom Zero.

## Install

```powershell
pip install atomzero==0.1.3
```

Optional local MiniLM reranker:

```powershell
pip install "atomzero[neural]"
```

## Quickstart

```powershell
atomzero-quickstart
```

The first run:

1. downloads the ~197 MB checkpoint/tokenizer from
   [Hugging Face](https://huggingface.co/byjameson/atom-zero-49m);
2. seeds a small demo corpus (launch code, warranty, travel, visitor badge);
3. starts the local UI at `http://127.0.0.1:8013`;
4. opens the browser.

Useful options:

```powershell
atomzero-quickstart --provider          # local Ollama free-form chat
atomzero-quickstart --port 8014
atomzero-quickstart --safety-filter
atomzero-quickstart --no-download       # use cached files
atomzero-quickstart --no-browser
```

The cache lives at `~/.cache/atomzero/atom-zero-49m`.

## What it is

- **A local, single-user daily assistant:** short chat, tasks, notes, arithmetic,
  time and source-cited document lookup.
- **A source-cited RAG system** over your own text, Markdown, HTML, PDF and DOCX
  files.
- **A deterministic tool runtime:** arithmetic, time, tasks, name/likes memory and
  a constrained plan API.
- **A reviewed research loop:** approved-only Wikipedia fetches, local teacher
  notes and a human approval gate.
- **A small-model + harness reference:** the harness is the reusable asset; the
  49.2M core is deliberately narrow.
- **Optionally, a front-end for a stronger local provider** while RAG, tools and
  memory stay local.

## What it is not

- Not a general reasoning or coding model.
- Not a ChatGPT competitor.
- Not a hosted, multi-tenant service.
- Not a model that stores all facts in its weights.
- Not an autonomous crawler or self-modifying agent.
- Not a claim that 49.2M parameters are enough for general intelligence.

## System features

| area | implemented |
|---|---|
| core model | 49.2M decoder-only transformer, BPE-8k, 6 layers, 768 dim, 128 context |
| training | 175.37M clean + 214.8M replay; 40k product checkpoint |
| RAG | SQLite FTS5, hashed/MiniLM embeddings, hybrid candidates, lexical+cosine rerank |
| vector index | cached vectors, prewarmed numpy matrix, IVF ANN above 256 chunks, exact fallback |
| documents | text/Markdown/HTML/PDF/DOCX ingestion, page/section citations, evidence panel/API |
| tools | time, safe arithmetic, tasks, name/likes memory, allowlisted 3-step plan API |
| sessions | transcript restore, JSON export/import, exact Forget, session Forget |
| research | approved-only queue, Wikipedia fetch, worker lock/log, teacher synthesis |
| teacher | local Ollama/OpenAI-compatible, unverified notes, reviewer verification |
| curiosity | gap → pending topic → approval → Wikipedia + teacher → RAG note |
| self-learning | RAG memory (no weights), reviewed LoRA (weights), manual full training |
| provider | optional local free-form chat; RAG/tools stay local |
| safety | optional rules + local linear ML head; off by default; exact fallback |
| privacy | localhost, no telemetry, SSRF guard, upload caps, deletion |
| UI | chat, documents, evidence, gaps, research queue, teacher notes, adapters, health |

## Model

| item | value |
|---|---|
| name | Atom Zero 49.2M (`transformer_behavior_50m_scaled`) |
| architecture | decoder-only transformer |
| parameters | 49,213,440 |
| layers / dim / heads / FF | 6 / 768 / 12 / 3072 |
| context | 128 tokens |
| tokenizer | 8,192-token BPE (`chat_en_bpe8k.json`) |
| checkpoint step | 40,000 |
| training tokens | 175.37M clean + replay mix to 214.8M |
| runtime | PyTorch, CUDA or CPU; RTX 3050 6 GB tested |

The model is trained from scratch. It is a **narrow chat/behavior model**, not a
foundation model. Facts are expected to live in RAG; skills can live in reviewed
LoRA adapters.

## Honest benchmark

| check | result |
|---|---:|
| raw 49.2M, no RAG/tools | **2/10** |
| product with RAG/tools | **21/21** |
| behavior checks | **16/16** |
| source-cited retrieval top-1 | **1.000** (FTS 0.850) |
| document benchmark, 100 authored questions | **80/80 answerable, 20/20 no-answer** |
| test suite | **412 passed** |

The raw-model probe asks ten simple general questions with no documents, no
calculator and no agent. The 49.2M model gets **2/10**. The product gets
**21/21** because the runtime does the work.

The product benchmark is narrow and authored: tools, memory, tasks,
source-cited retrieval, teacher-note review, missing-source detection and one
short generation. It is **not MMLU, GSM8K or HumanEval**.

## Documents and citations

- **Text, Markdown, HTML, PDF and DOCX** ingestion.
- PDF citations keep the **page number**; DOCX/HTML/Markdown keep the
  **section heading**.
- **Evidence panel/API:** `GET /v1/documents/evidence` opens the exact local
  chunks behind a citation.
- **Per-source delete:** Forget removes chunks, page/section metadata, source
  metadata and cached vectors.

See [DOCUMENTS.md](https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md).

## Tools and constrained plans

- time, safe arithmetic, task add/list/complete, name/likes memory.
- `POST /v1/agent/plan` executes a small JSON plan through an allowlist:
  - at most **3 steps**;
  - no shell, arbitrary URL or code execution;
  - **explicit confirmation** before task/research side effects;
  - text arguments pass the optional safety filter when enabled.

The 49.2M model does not choose tools freely. This is a constrained executor,
not model-driven planning. See
[TOOL_PLANNING.md](https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md).

## Research, curiosity and learning

When no local source answers a question, it becomes a knowledge gap. The
assistant can propose pending research topics, but **nothing fetches until a
human approves**:

```text
question → no local source → knowledge gap
gap → pending curiosity topic
human approves
worker fetches Wikipedia context
local Ollama teacher synthesizes
answer stored as an unverified teacher note in RAG
next answer can cite it; weights_changed: false
```

The self-improvement ladder is deliberately split:

- **RAG memory:** a teacher note or your document enters the index; no weight
  change.
- **Reviewed adapter:** a LoRA adapter for style/skills; human review + gates;
  weights change.
- **Full training:** a new base checkpoint; offline project decision; weights
  change offline.

The model never crawls, mutates its own weights or decides what to study while
you chat.

## Optional local provider

```powershell
atomzero-quickstart --provider
```

This routes free-form chat to local Ollama `llama3.2:3b` while RAG, tools and
memory remain local. It is not the default backend. Use a different model with
`--provider-model`.

## Optional safety

Safety is **off by default**. With `--safety-filter`, exact rules run first and
a small local linear ML head is additive; a missing/broken head falls back to
the exact rules. It is an operator tool, not a hosted content policy.

## Limits

- 2/10 raw knowledge; weak general reasoning and coding.
- English-first tokenizer/data.
- 128-token core context; longer context is a model/training change.
- No OCR; scanned PDFs need OCR before ingestion.
- Encrypted PDFs are rejected; DOCX page numbers are unavailable.
- Product benchmark is narrow and authored; it is not MMLU.
- The plan API is not free-form model planning.
- The document benchmark is an authored regression gate, not a public
  leaderboard.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.3/>
- GitHub source: <https://github.com/byjameson/stream-finetune/tree/main/packages/atomzero>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md>
- Benchmarks: <https://github.com/byjameson/stream-finetune/blob/main/docs/BENCHMARKS.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**

Apache-2.0 for code and weights; data attribution in `NOTICE`.
