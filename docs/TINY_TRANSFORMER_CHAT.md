# Tiny Transformer Chat — model card

## What this is

A tiny transformer language model trained from scratch for short English
daily-assistant turns. The **20.2M** checkpoint is the compact fallback; the
current product checkpoint is the **49.2M behavior variant**
(`transformer_behavior_50m_long`). It is **not** a general assistant and not a
factual model; facts belong in the document index, not in the weights.

- 20M: 5 layers, dim 512, 8 heads, FFN 2048; 50M: 6 layers, dim 768,
  12 heads, FFN 3072; 8k BPE, max sequence 512.
- Trained on **26.98M tokens** of filtered English chat and story data.
- Assistant-only SFT loss; training windows start at `<|user|>`.
- Role tokens (`<|user|>`, `<|assistant|>`) are banned during decoding.

## Verified result

Fixed-window evaluation on the clean validation shard:

| metric | value |
|---|---:|
| chat bpb | **1.359** |
| assistant-only loss | **4.01** (ppl ~55) |
| TinyStories bpb | 1.447 |
| Wikipedia bpb | 2.202 |
| minimum sample alpha | **0.731** |
| empty samples | **0/12** |
| tag leaks with banned role tokens | **0/12** |
| smoke gate | **PASS_CHAT_SMOKE** |

Without banning role tokens, 5/12 samples leak `<|assistant|>` and the gate is
BLOCKED. The pass is therefore a **constrained-decoding smoke gate**, not an
open-ended chat-quality claim.

## Virtual-human skill checkpoint

The product checkpoint, `transformer_chat_skill_ret_21m`, is fine-tuned for
4,000 steps on 58.8k targeted dialogues (greetings, name/like memory, task
add/list/complete, small talk and exact-format replies), mixed **2:1
clean:skill** at LR 3e-5.

- skill-only assistant loss **0.31** (ppl 1.36; synthetic templates, near
  memorization);
- clean-chat bpb **1.440** / assistant loss **4.27** (base 1.359 / 4.01);
- end-to-end `chat_human` smoke: **PASS_VIRTUAL_HUMAN_SMOKE**;
- all memory/task checks pass; greeting/small talk use deterministic fast paths.

The earlier 1:1/LR 1e-4 skill run reached skill loss 2.27 but regressed clean
chat to 1.655 / 4.93; it is superseded by the retention checkpoint.

### Gated update v0001

A reviewed queue of 267 novel format/style examples was used for a 300-step,
low-LR update with 2:1 clean replay:

- queue loss **4.346 -> 1.114** (74.4% target gain);
- clean bpb **1.440 -> 1.475** (+2.45%);
- clean assistant loss **4.266 -> 4.374** (+2.52%);
- gate accepted; checkpoint: `checkpoints/updates/v0001`;
- standard E2E memory/task/greeting check still passes.

**Caveat:** the gate is loss-based, not behavioral. `summarize` works, but the
raw model often fails `count to five`, `be brief` and `make it polite`; those
exact formats are handled by the deterministic fast path in `chat_human.py`.

A later format queue (283 examples) cleared loss + behavior but produced empty
clean-chat samples. The clean-smoke gate was tightened; the v0002/v0003
candidates were rejected, and **v0001 remains the product checkpoint**. Do not
treat the loss gain as a behavioral gain.

A **49.2M behavior checkpoint** (`transformer_behavior_50m_long`) was trained
for 1,000 steps on a diverse mix of clean chat + skill + format/novel queues:

- clean bpb **1.375** / assistant loss **4.066** (v0001: 1.475 / 4.374);
- corrected clean smoke **PASS**;
- runtime behavior **PASS**;
- 13-turn E2E **PASS**.

This is now the **product checkpoint**. v0001 (20M) remains a faster/local
fallback. Factual questions such as live weather are out of scope for the raw
model; the runtime should answer those with a tool or a refusal.

Corrected-format revalidation of v0001: clean smoke **PASS**, 13-turn E2E
**PASS**, runtime behavior **PASS**. The earlier 12-sample result with one empty
sample was the runtime-handled `my name is Alex, remember me` command; clean
smoke now tests open-ended prompts only.

## Samples

Runtime samples (deterministic fast paths and document memory):

```text
hello -> Hello! How can I help you today?
add task buy milk -> Added task 1: buy milk
my name is Alex -> I will remember that.
what is my name? -> Your name is Alex.
what is 12 * 7? -> 12 * 7 = 84
search docs capital France -> quoted note + source
what is the weather today? -> (raw model) To determine what the weather is.
```

The raw model handles greetings and short open-ended replies. Exact task,
memory, arithmetic and citation formats are handled by the runtime; the raw
model is not asked to memorise them.

## Intended use

- local daily assistant: tasks, personal notes and cited document lookup,
- memory/task runtime experiments,
- source-cited research through Wikipedia or opt-in URL ingestion,
- education and research on small from-scratch models,
- a baseline for the virtual-human runtime.

Creative writing is possible but is **not** the product focus. The assistant is
not a general reasoning engine.

## Out of scope

- general chat, instruction following, factual QA, coding, translation,
- autonomous crawling or self-directed learning,
- medical, legal, financial or safety-critical advice,
- public deployment without a safety and licence review,
- claims that this is a general language model.

## Daily use

The product is a local daily assistant, not a story demo:

| layer | what it does |
|---|---|
| runtime fast paths | tasks, name/likes, time, simple arithmetic, exact formats |
| document memory (SQLite FTS5) | your notes; HTTP paste/upload with `source` + sha256 |
| Wikipedia research | user-triggered lookup; stores extracts with URLs |
| LoRA adapter | optional per-user style/skill, gated before acceptance |
| 49.2M model | greetings and short open-ended wording |

```bash
curl -X POST http://127.0.0.1:8013/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"user-1","message":"add task buy milk"}'
```

The browser page at `/` uses the same agent endpoint. It has a plain chat box,
task/memory/arithmetic examples and a “teach a fact” box. No account, no cloud.

## Data and licence

| source | licence |
|---|---|
| HuggingFaceTB/smol-smoltalk, everyday conversations | Apache-2.0 / check per subset |
| HuggingFaceH4/ultrachat_200k (filtered) | MIT |
| OpenAssistant/oasst2 | Apache-2.0 |
| databricks/dolly-15k | CC BY-SA 3.0 |
| TinyStories | CDLA-Sharing |
| synthetic memory/task dialogues | generated in this repo |

Code is Apache-2.0 (see `LICENSE`). Model weights inherit the data
obligations; legal/provenance review is required before redistribution.

## Runtime

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters_50m/v0001 \
  --docs data/docs.db \
  --host 127.0.0.1 --port 8013
```

Endpoints:

| endpoint | purpose |
|---|---|
| `GET /health` | model, params, step, adapter, document stats |
| `GET /v1/models` | model id |
| `GET /v1/documents` | indexed sources |
| `POST /v1/documents` | `{"source": "...", "title": "...", "text": "..."}`; indexes pasted text, max 1 MB |
| `POST /v1/documents/upload?source=...&title=...` | raw UTF-8 text body; same limits as JSON |
| `POST /v1/documents/search` | `{"query": "...", "k": 5}` |
| `POST /v1/teach` | `{"fact": "..."}` for the document index, or `{"user": "...", "assistant": "..."}` for the reviewed skill queue |
| `POST /v1/research` | `{"query": "...", "limit": 2}`; fetches Wikipedia extracts with sources |
| `POST /v1/learn_url` | `{"url": "..."}`; opt-in page ingestion, requires `--allow-url-fetch` |
| `POST /v1/agent/chat` | `{"session_id": "...", "message": "..."}`; tasks, memory, docs, time, arithmetic, research |
| `POST /generate` | `{"prompt": "...", "max_new_tokens": 64}` |
| `POST /v1/chat/completions` | messages; optional `session_id`; document hits return `sources` and a non-advice note |

Verified on 2026-11-21: the one-command demo on port 8020 returned health with
**49,582,080** params, adapter `checkpoints/adapters_50m/v0001` and indexed
documents; `/v1/agent/chat` answered `12 * 7 = 84`, handled task/memory
commands, and returned the codename note with its source. `/v1/teach` stores a
fact in the index and queues a reviewed skill example. The recheck adapter
(`checkpoints/adapters_recheck/v0001`) also loaded and served on port 8017.

HTTP document ingestion was verified on port 8023 with the real 50M model:
JSON ingest returned `200` with source/title/chunks/sha256, search and
`/v1/chat/completions` returned the document text with `sources`, raw UTF-8
upload returned `200`, missing source returned `400`, and a 1,000,001-byte
document returned `413`. In the browser, pasted text and a selected `.txt` file
were both indexed and retrieved with their source.

One-command demo (server + demo note + browser; the adapter is used when
present, otherwise the base runs alone):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo_all.ps1
```

### Bring your own documents

Paste text over JSON:

```bash
curl -X POST http://127.0.0.1:8013/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"source":"notes.txt","title":"Notes","text":"The launch code is ORANGE-42."}'
```

Or upload a UTF-8 text file as the raw body:

```bash
curl -X POST "http://127.0.0.1:8013/v1/documents/upload?source=notes.txt&title=Notes" \
  -H 'Content-Type: text/plain' \
  --data-binary @notes.txt
```

Limits: UTF-8 text only, **1,000,000 bytes** per document and **2 MB** per
request. `source` is required; re-adding the same `source` replaces its chunks.
The response contains `chunks`, `bytes` and `sha256`; `GET /v1/documents` lists
the indexed source/title/chunk counts. The browser page has the same paste and
file-picker flow. Documents go to SQLite FTS5, never to model weights.

### Web research

`POST /v1/research` uses the public Wikipedia API, fetches plain-text extracts
and stores them with URL/title provenance. Arbitrary page fetching
(`/v1/learn_url`) is **opt-in** via `--allow-url-fetch`; private/loopback hosts
are blocked (SSRF guard), pages are size-capped, and no crawler runs
automatically. Research writes to DocMemory, never to model weights.

### Agent API

`POST /v1/agent/chat` keeps per-session memory and task state:

```bash
curl -X POST http://127.0.0.1:8013/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"user-1","message":"add task buy milk"}'
```

It routes greetings/small talk, memory/tasks, document search, time, simple
arithmetic and Wikipedia research. The OpenAI-compatible
`/v1/chat/completions` accepts the same `session_id`; without it, the endpoint
is stateless. External agent frameworks can connect through these HTTP
endpoints, but the 50M model remains the language layer, not a general
reasoning engine.

The one-command browser demo is `scripts/demo_all.ps1` above; `chat_demo.ps1`
is the lower-level server-only launcher.

Interactive CLI with memory/tasks:

```bash
python research/chat_human.py --arch transformer \
  --model checkpoints/transformer_behavior_50m_long
```

Add your own facts/documents (RAG, no weight update):

```bash
python -m streamft.doc_memory add data/notes.txt --db data/docs.db
python research/chat_human.py --arch transformer \
  --model checkpoints/transformer_behavior_50m_long \
  --docs data/docs.db
```

See [`PERSONALIZE.md`](PERSONALIZE.md) for the bring-your-own-data workflow.

### LoRA adapters

`streamft/lora.py` supports small MLP adapters on the frozen base. A gated
adapter has **368,640 trainable parameters** (~0.74% of 49.2M) and is ~1.5 MB:

- queue-target gain **+38.2%**;
- clean bpb regression **+0.08%**, assistant **+0.07%**;
- clean smoke **PASS** (1.376 bpb / 4.069 assistant);
- runtime behavior **PASS**;
- E2E **PASS**.

The gate was re-run on 2026-11-21 from the same 283-example queue and seed:
accepted again with **+38.19%** target gain, clean bpb **+0.079%**, assistant
**+0.066%**, behavior and clean smoke **PASS**
(`research/results/gated_lora_recheck.json`). This is a reproducibility check,
not a new capability claim.

Load one with `--adapter` in `streamft.chat_server`, `research/chat_human.py`,
`research/eval_transformer_chat.py`, `research/behavior_eval.py` or
`research/eval_chat_human.py`. Train with `research/train_lora.py` or the gated
adapter command in [`PERSONALIZE.md`](PERSONALIZE.md).

## Training

```bash
python research/train_transformer_chat.py \
  --out checkpoints/transformer_chat_en_clean_21m \
  --steps 12000 --batch 16 --seq 128 \
  --dim 512 --layers 5 --heads 8 --ff 2048
```

~25 minutes on an RTX 3050 6 GB. Checkpoints every 2000 steps.

## Limitations

- The 49.2M product model still repeats, gives generic replies and confuses
  open-ended wording; exact formats depend on runtime fast paths.
- The 20M v0001 is the compact fallback; the 49.2M checkpoint is the product
  default because its behavior and clean metrics are better.
- English only.
- No safety filter, no factuality guarantee, no instruction tuning beyond SFT.
- The smoke gate is narrow; it does not prove general chat quality.
- No autonomous web crawler: research is user-triggered (Wikipedia or opt-in
  URL ingestion). Facts are never written into weights automatically.
- The HTTP API is a small OpenAI-compatible subset, not full tool calling,
  streaming or multimodal support.

Full research: [`research/CHAT_FROM_SCRATCH.md`](../research/CHAT_FROM_SCRATCH.md).
