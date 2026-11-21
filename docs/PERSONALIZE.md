# Personalize Atom Zero with your own data

Two different jobs, two different paths:

| what you have | where it goes | command |
|---|---|---|
| facts, policies, legal notes, manuals | local document index (RAG) | `python -m streamft.doc_memory add ...` |
| style, format, task behaviour | model weights (fine-tune) | `research/train_transformer_chat.py` / gated update |

Never put a large, changing knowledge base into the weights. Use documents.

## 1. Add documents (facts, policies, notes)

```bash
python -m streamft.doc_memory add data/notes.txt --db data/docs.db
python -m streamft.doc_memory add data/legal/kira.txt --db data/docs.db --title "Kira notları"
python -m streamft.doc_memory search "kira" --db data/docs.db
python -m streamft.doc_memory list --db data/docs.db
```

Each chunk keeps its **source path**, title and hash. Re-adding a source
replaces its chunks, so updates do not create duplicates.

### HTTP ingestion (no CLI)

Paste text:

```bash
curl -X POST http://127.0.0.1:8013/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"source":"notes.txt","title":"Notes","text":"The launch code is ORANGE-42."}'
```

Upload a UTF-8 text file as the raw body:

```bash
curl -X POST "http://127.0.0.1:8013/v1/documents/upload?source=notes.txt&title=Notes" \
  -H 'Content-Type: text/plain' \
  --data-binary @notes.txt
```

`source` is required. Limits are **1,000,000 bytes** per document and 2 MB per
request. The response returns `chunks`, `bytes` and `sha256`; re-adding the same
source replaces it. The browser demo includes paste and file-picker inputs.
Documents are stored in SQLite FTS5, not in model weights.

### Manual teacher import (no API)

Ask ChatGPT or another assistant yourself and paste the answer:

```bash
curl -X POST http://127.0.0.1:8013/v1/teacher/notes \
  -H 'Content-Type: application/json' \
  -d '{"teacher":"chatgpt-web","model":"gpt-5","topic":"Ankara","answer":"Ankara is the capital of Turkey.","source_url":"https://en.wikipedia.org/?curid=802"}'
```

The note is stored as `kind=teacher`, `verified=false`, with a unique
`teacher://...` source and the supplied URL as provenance. Search and chat mark
it as `teacher note, unverified`; it is not a verified fact and does not change
weights. Missing source URLs are allowed but stay unverified.

Review and verify after checking the note:

```bash
curl http://127.0.0.1:8013/v1/teacher/notes
curl -X POST http://127.0.0.1:8013/v1/teacher/notes/verify \
  -H 'Content-Type: application/json' \
  -d '{"source":"teacher://chatgpt-web/...","reviewer":"reviewer-1"}'
```

The API stores the reviewer and timestamp. The browser page lists teacher notes
and has a **Verify** button. After verification, citations say
`teacher note, verified by reviewer-1` and the unverified warning is removed.
This records human review, not factual truth; re-creating the note resets it to
unverified.

Manual paste needs no API key and no browser scraping. If you want to ask a
configured model instead, the provider is opt-in and off by default:

```bash
# Local Ollama example (no API key).
ollama serve
ollama pull llama3.2:3b

python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --docs data/docs.db \
  --teacher-url http://127.0.0.1:11434/v1 \
  --teacher-model llama3.2:3b \
  --teacher-label ollama \
  --port 8013

curl -X POST http://127.0.0.1:8013/v1/teacher/ask \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give a short, factual summary of Ankara.","topic":"Ankara"}'
```

For a remote OpenAI-compatible server, set `TEACHER_API_KEY` in the environment
and choose `--teacher-url`/`--teacher-model`. The provider is only called by an
explicit user request; output is stored as `verified=false`, never written to
weights. There is no browser automation and no background call.

Use it from the runtime:

```bash
python research/chat_human.py \
  --arch transformer \
  --model checkpoints/transformer_behavior_50m_long \
  --docs data/docs.db
```

Or over HTTP:

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --docs data/docs.db \
  --adapter checkpoints/adapters_50m/v0001 \
  --port 8013
```

Document hits on `/v1/chat/completions` return the matching notes, their
`sources`, and a `not professional advice` note; unmatched questions go to the
model.

Teach over HTTP:

```bash
curl -X POST http://127.0.0.1:8013/v1/teach \
  -H 'Content-Type: application/json' \
  -d '{"fact":"The project codename is Blue Lantern."}'

curl -X POST http://127.0.0.1:8013/v1/teach \
  -H 'Content-Type: application/json' \
  -d '{"user":"what is the codename?","assistant":"Blue Lantern."}'
```

The first writes to the document index immediately. The second only enters the
reviewed skill queue; it does **not** change weights until a gated update is
reviewed and accepted.

Research and ingest:

```bash
curl -X POST http://127.0.0.1:8013/v1/research \
  -H 'Content-Type: application/json' \
  -d '{"query":"Ankara history","limit":2}'
```

Wikipedia lead extracts are stored with source URLs. Arbitrary URL ingest is
opt-in (`--allow-url-fetch`), size-capped and blocks private/loopback hosts.

### Reviewed research queue

Use a queue when you want a scheduler to fetch topics without approving each
fetch in real time:

```bash
# Add a pending topic; this does not touch the network.
curl -X POST http://127.0.0.1:8013/v1/research/queue \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Ankara","lang":"en","pages":1}'

# Approve it, then fetch approved topics into DocMemory.
curl -X POST http://127.0.0.1:8013/v1/research/queue/approve \
  -H 'Content-Type: application/json' -d '{"id":"r0001"}'
curl -X POST http://127.0.0.1:8013/v1/research/queue/run \
  -H 'Content-Type: application/json' -d '{"limit":1}'
```

`GET /v1/research/queue` shows `pending / approved / running / done / error /
rejected` items. Pending and rejected topics are never fetched. Each fetched
item stores its Wikipedia source URLs and titles; normal document search and
chat cite them.

If the server has a teacher provider configured, `run` also sends the fetched
lead context and source URLs to it. The answer is stored as an unverified
teacher note; the queue item has a `teacher` block with its source/preview/error.
A teacher failure leaves the Wikipedia pages intact.

CLI:

```bash
python research/research_queue.py --queue data/research_queue.json add "Ankara" --pages 1
python research/research_queue.py --queue data/research_queue.json approve r0001
python research/research_queue.py --queue data/research_queue.json run --db data/docs.db --limit 1

# With synthesis (local Ollama example).
python research/research_queue.py --queue data/research_queue.json run \
  --db data/docs.db --limit 1 \
  --teacher-url http://127.0.0.1:11434/v1 --teacher-model llama3.2:3b
```

The browser page has the same queue controls: add, approve, reject and run.
For a scheduled worker, use the wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_research.ps1 -Limit 1
```

With teacher synthesis:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_research.ps1 `
  -Limit 1 -TeacherUrl http://127.0.0.1:11434/v1 -TeacherModel llama3.2:3b
```

Windows Task Scheduler:

```powershell
schtasks /Create /SC DAILY /ST 09:00 /TN "streamft research" `
  /TR "powershell -ExecutionPolicy Bypass -File C:\path\to\stream-finetune\scripts\run_research.ps1 -Limit 1"
```

Linux/macOS cron:

```cron
0 9 * * * cd /path/to/stream-finetune && python research/research_queue.py --queue data/research_queue.json run --db data/docs.db --limit 1
```

The queue does not generate its own topics and does not crawl the open web.
Only approved topics are fetched.

Agent/session use:

```bash
curl -X POST http://127.0.0.1:8013/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"user-1","message":"add task buy milk"}'
```

Each session keeps its own memory/tasks; `/v1/chat/completions` accepts the
same `session_id` for OpenAI-style clients. Tools cover memory, tasks, local
documents, time, simple arithmetic and Wikipedia research. The model only
handles open-ended wording; it is not a general reasoning agent.

One-command demo: `scripts/demo_all.ps1`.

When a question matches indexed text, the reply quotes the matching notes with
their source and adds `not professional advice`. If nothing matches, the tiny
model handles the open-ended reply.

## 2. Fine-tune a skill or style

Dataset format (JSONL):

```json
{"user": "make it polite: open the window", "assistant": "Could you open the window, please?"}
{"user": "how are you?", "assistant": "I am doing well, thank you."}
```

Build a mix with replay and run a gated update:

```bash
python research/update_loop.py add --user "..." --assistant "..."
python research/gated_update.py --queue data/learning_queue.jsonl
```

The gate requires target improvement, clean-chat regression within budget,
behaviour checks and a 12-sample clean smoke. Rejected candidates are not
promoted.

### Lightweight LoRA adapter

Train a small adapter on top of the frozen product checkpoint:

```bash
python research/train_lora.py \
  --init checkpoints/transformer_behavior_50m_long \
  --steps 300 --rank 8 --alpha 16 \
  --out checkpoints/adapters/my_lora
```

Use it:

```bash
python research/chat_human.py \
  --arch transformer \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters/my_lora
```

The demo adapter has **368,640 trainable parameters** (~0.74% of the 49.2M
base), trains in ~45 seconds on the RTX 3050, and passes clean smoke, runtime
behavior and E2E. Keep adapters per user/domain; the base is shared and frozen.

Use the gated adapter loop when you want a reviewed update:

```bash
python research/gated_update.py \
  --base checkpoints/transformer_behavior_50m_long \
  --queue data/learning_queue.jsonl \
  --steps 100 --lora --lora-lr 1e-4 \
  --out checkpoints/adapters_50m
```

The accepted 50M adapter (`checkpoints/adapters_50m/v0001`) gives **+38.2%**
queue-target gain with clean bpb regression **+0.08%** and assistant
**+0.07%**. Clean smoke, behavior and E2E all pass. The gate rejects adapters
that hurt clean chat or fail the behavioral checks. A 2026-11-21 re-run from
the same queue was accepted again at **+38.19% / +0.079% / +0.066%**
(`research/results/gated_lora_recheck.json`), so the loop is reproducible.

### Feedback-driven personal loop

The browser records **Good** / **Bad / correct** feedback and exposes a
**Learning queue**. Approved feedback can be exported and sent through the same
gate:

```bash
python research/learning_loop.py list
python research/learning_loop.py approve f0001
python research/learning_loop.py reject f0002
python research/learning_loop.py export
python research/learning_loop.py run \
  --base checkpoints/transformer_behavior_50m_long \
  --out checkpoints/adapters_feedback --steps 100 --lora-lr 1e-4 \
  --holdout 0.2 --holdout-out data/feedback_holdout.jsonl
```

`--holdout` keeps a deterministic tail slice of approved rows out of training.
After acceptance the base and adapter are compared on those rows and the result
is written to `research/results/holdout_vNNNN.json` and recorded in history.
The gate itself still needs at least five training examples. On acceptance it writes
`data/active_adapter.json`; start the server with `--adapter auto` and
`--active-adapter-file data/active_adapter.json` to use it. Rejected candidates
do not change the active adapter. `scripts/run_learning.ps1` is the scheduled
wrapper.

Accepted adapters are immutable `vNNNN` versions. Each acceptance is appended
to `data/adapter_history.jsonl` with its gate metrics, queue hash and result:

```bash
python research/learning_loop.py history
python research/learning_loop.py activate v0001
```

`activate` is the rollback path: it restores an adapter recorded in history
after checking that its `adapter.pt` still exists. The active file keeps the
previous version for audit. Compare base and adapter loss on the same rows:

```bash
python research/compare_adapters.py \
  --base checkpoints/transformer_behavior_50m_long \
  --data data/feedback_training.jsonl \
  --adapters checkpoints/adapters_feedback/v0001
```

`compare_adapters` reports assistant-turn cross-entropy per row and the delta vs
base. Evaluating on the same rows used for training measures fit, not
generalization; hold out examples for an honest comparison.

The 8-example bootstrap test was accepted at **+16.7%** queue gain with clean
bpb **+1.58%** / assistant **+1.54%**; behavior and clean smoke passed. It
improved the targeted loss but did **not** reliably memorize an exact fact.
Use DocMemory for facts; use feedback adapters for style, format and narrow
skills. Evidence: [`research/results/gated_feedback_test.json`](../research/results/gated_feedback_test.json).
On the same 8 examples, assistant-turn loss was base **3.826** vs feedback
adapter **1.640**; this is an in-sample fit check
([`research/results/adapter_comparison_feedback.json`](../research/results/adapter_comparison_feedback.json)).
A 6-train / 2-holdout run kept both holdout rows out of training and still
improved loss from **3.853** to **3.139** (**-18.5%**); the sample is tiny, but
it is a real held-out check
([`holdout_feedback_test.json`](../research/results/holdout_feedback_test.json)).

### Missing data (gap loop)

When no local document answers a knowledge question, Atom Zero records it as a
**knowledge gap**. A human can propose it as a pending research topic; after
approval the scheduled worker fetches Wikipedia and indexes the source. The next
answer is extractive and cited. This is how the system "finds the missing data"
without autonomous crawling or unsupervised weight updates.

```bash
python research/gap_loop.py list
python research/gap_loop.py propose g0001 --pages 1
python research/research_queue.py --queue data/research_queue.json approve r0001
python research/research_worker.py --queue data/research_queue.json \
  --db data/docs.db --limit 1 --once
```

The browser has the same flow under **Knowledge gaps**. A gap proposal only
creates a pending topic; it does not fetch. Verified live: an unknown Paris
population question became `g0001` -> `r0001`, and after approval the answer
quoted **2.04 million** with the Wikipedia source. See
[`KNOWLEDGE_GAPS.md`](KNOWLEDGE_GAPS.md).

The product benchmark is **21/21** for base and adapter on narrow behaviors
(tools, memory, tasks, retrieval, teacher review, gap detection, one short
generation). It is **not MMLU**; see [`BENCHMARKS.md`](BENCHMARKS.md).

Full fine-tune from a base checkpoint:

```bash
python research/train_transformer_chat.py \
  --init checkpoints/transformer_chat_en_clean_50m \
  --data data/behavior_mix_train.pt \
  --valid data/behavior_mix_valid.pt \
  --out checkpoints/my_model \
  --dim 768 --layers 6 --heads 12 --ff 3072 \
  --steps 1000 --lr 1e-5
```

## 3. What a 50M model can and cannot learn

Can learn, with a few thousand short examples:

- greeting/small-talk style,
- exact response formats (count, polite request, summarize, echo),
- short email/note templates,
- a narrow domain's phrasing.

Cannot reliably learn from weights alone:

- a large legal/medical corpus,
- changing facts, prices, schedules,
- citations and current law,
- general reasoning.

For those, use document memory + a larger model or a human review step.

## 4. Release/provenance

- Keep the source and licence of every ingested document.
- Do not redistribute documents you do not have rights to.
- Do not claim legal/medical/financial advice.
- An adapter or fine-tuned checkpoint inherits the base model's data
  obligations.
