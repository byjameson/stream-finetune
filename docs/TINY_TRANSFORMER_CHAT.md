# Atom Zero — local assistant model card

## What this is

**Atom Zero** is a tiny transformer language model trained from scratch for
short English daily-assistant turns. The Python package and CLI remain
`streamft`; Atom Zero is the product name.

License: Apache-2.0 for code and model weights; training-data attribution is in
[`NOTICE`](../NOTICE). The source text remains under its original dataset
licences. The **20.2M** checkpoint is the compact fallback; the
current product checkpoint is the **49.2M scaled run**
(`transformer_behavior_50m_scaled`, 40k steps on the 175M-token corpus). The
original `transformer_behavior_50m_long` is retained as the baseline. It is
**not** a general assistant and not a factual model; facts belong in the
document index, not in the weights.

- 20M: 5 layers, dim 512, 8 heads, FFN 2048; 50M: 6 layers, dim 768,
  12 heads, FFN 3072; 8k BPE, max sequence 512.
- The compact fallback was trained on **26.98M tokens** of filtered English
  chat/story data; the scaled product run used **175.37M** clean tokens plus
  behavior/skill replay.
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
- unreviewed autonomous crawling or self-directed topic generation; the
  research queue only runs user-approved topics,
- medical, legal, financial or safety-critical advice,
- public deployment without a safety and licence review,
- claims that this is a general language model.

## Daily use

The product is a local daily assistant, not a story demo:

| layer | what it does |
|---|---|
| runtime fast paths | tasks, name/likes, time, simple arithmetic, exact formats |
| document memory (SQLite FTS5) | your notes; HTTP paste/upload with `source` + sha256 |
| manual teacher notes | pasted ChatGPT/other answers; derived, unverified, URL kept |
| optional teacher provider | local Ollama/OpenAI-compatible; explicit ask, still unverified |
| Wikipedia research | user-triggered lookup or approved queue; lead extracts with URLs |
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
| `GET /health` | model, params, step, adapter, document, research and learning stats |
| `GET /v1/models` | model id |
| `GET /v1/documents` | indexed sources |
| `POST /v1/documents` | `{"source": "...", "title": "...", "text": "..."}`; indexes pasted text, max 1 MB |
| `POST /v1/documents/upload?source=...&title=...` | raw UTF-8 text body; same limits as JSON |
| `POST /v1/documents/forget` | `{"source":"my-note.txt"}`; deletes that source and metadata, then compacts SQLite |
| `POST /v1/memory/forget` | `{"session_id":"...","text":"..."}` or `{"all":true}`; deletes session facts/events/tasks |
| `POST /v1/documents/search` | `{"query": "...", "k": 5}` |
| `GET /v1/teacher/notes` | teacher notes with verified/unverified stats |
| `POST /v1/teacher/notes` | `{"teacher","model","topic","answer","source_url"}`; stores a **derived/unverified** note |
| `POST /v1/teacher/notes/verify` | `{"source","reviewer"}`; marks a note verified with a timestamp |
| `POST /v1/teacher/ask` | `{"prompt","topic","source_url"}`; explicit call to the configured teacher provider, then stores an unverified note |
| `POST /v1/learning/feedback` | `{"user","assistant","rating","corrected","session_id"}`; queues user feedback for review |
| `GET /v1/learning/queue` | feedback items and pending/approved/rejected counts |
| `POST /v1/learning/queue/approve` / `reject` | `{"id":"f0001"}`; human review step |
| `POST /v1/learning/queue/export` | writes approved feedback to `data/feedback_training.jsonl` |
| `GET /v1/learning/gaps` | questions with no local source, with status counts |
| `POST /v1/learning/gaps/propose` | `{"id":"g0001","pages":1}`; creates a **pending** research topic |
| `POST /v1/learning/gaps/dismiss` | `{"id":"g0002"}`; dismisses a gap |
| `POST /v1/teach` | `{"fact": "..."}` for the document index, or `{"user": "...", "assistant": "..."}` for the reviewed skill queue |
| `POST /v1/research` | `{"query": "...", "limit": 2}`; directly fetches Wikipedia lead extracts with sources |
| `GET /v1/research/queue` | reviewed queue items and status counts |
| `POST /v1/research/queue` | `{"topic": "Ankara", "lang": "en", "pages": 2}`; adds a **pending** topic |
| `POST /v1/research/queue/approve` / `reject` | `{"id": "r0001"}`; user review step |
| `POST /v1/research/queue/run` | `{"limit": 1}`; fetches **approved** topics, plus optional teacher synthesis if configured |
| `POST /v1/learn_url` | `{"url": "..."}`; opt-in page ingestion, requires `--allow-url-fetch` |
| `POST /v1/agent/chat` | `{"session_id": "...", "message": "..."}`; tasks, memory, docs, time, arithmetic, research |
| `POST /generate` | `{"prompt": "...", "max_new_tokens": 64}` |
| `POST /v1/chat/completions` | messages; optional `session_id`; document hits return `sources` and a non-advice note |

Shared deployments should add `--safety-filter`. It refuses explicit harmful
requests before generation and filters output; it is off by default for local
research. Red-team result: multilingual/semantic filter **72/72**, output
**4/4**, agent **72/72**; untrusted-document injection **4/4**
([`SAFETY.md`](SAFETY.md)). Document and session deletion are documented in
[`PRIVACY.md`](PRIVACY.md).

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

The reviewed research queue was verified on port 8027: a pending `Ankara`
topic was not fetched by `run`, approval allowed the fetch, Wikipedia returned
the lead extract (`3` chunks, source `https://en.wikipedia.org/?curid=802`),
and both search and chat returned the intro with one deduplicated source.

Manual teacher import was verified in the browser on port 8030: a pasted
ChatGPT-style answer was stored as `teacher://chatgpt-web/...` with
`verified=false` and `source_url=https://example.com/chat`; chat returned the
answer and the warning “This includes an unverified teacher-derived note.”

The optional provider path was verified in the browser on port 8034 against a
local OpenAI-compatible fake provider: `Ask provider` stored the answer as
`teacher://fake/...`, and chat returned it with the unverified warning. No
external API or real key was used.

Queue + teacher synthesis was verified in the browser on port 8036: approving
`Ankara` fetched the Wikipedia lead (3 chunks, `curid=802`), the fake local
provider synthesized it, the queue item recorded `teacher://fake/...` plus a
`P-9001` synthesis preview, and chat returned the note with a Wikipedia source
URL and the unverified warning.

Teacher-note review was verified in the browser on port 8037: the pasted
`T-9000` note showed the unverified warning, **Verify** with reviewer
`local-reviewer` changed the list to `VERIFIED` with a timestamp, and the next
chat reply used `teacher note, verified by local-reviewer` with no warning.

The personal learning loop was verified on port 8038: a Good feedback item went
to **pending**, was approved and exported; eight approved examples then trained
a 368,640-parameter LoRA in 100 steps. The gate accepted it with **+16.7%**
queue gain, clean bpb **+1.58%**, assistant **+1.54%**, behavior and clean smoke
**PASS**. `--adapter auto` loaded the accepted test adapter
(`checkpoints/adapters_feedback_test/v0001`) on port 8039. Honest limit: that
small adapter improved loss on the examples but did not reliably recall an
exact launch code; facts still belong in RAG, not weights. Raw result:
[`gated_feedback_test.json`](../research/results/gated_feedback_test.json).

A fresh run on 2026-11-21 verified versioning: the same eight examples produced
`v0001`, accepted at **+16.74%** queue gain; the acceptance was appended to
`data/adapter_history.jsonl` with the queue sha256, gate checks and result path.
`--adapter auto` loaded `checkpoints/adapters_feedback_history_test/v0001` on
port 8041. Assistant-turn comparison on the same eight rows: base **3.826**,
feedback adapter **1.640**, product adapter **3.837**; the first is an
in-sample fit check, not a generalization claim
([`adapter_comparison_feedback.json`](../research/results/adapter_comparison_feedback.json)).

Rollback was verified too: a second run produced `v0002`, then
`learning_loop.py activate v0001` restored `v0001`, recorded `previous: v0002`
in the active file, and `--adapter auto` loaded `v0001` on port 8042
([`gated_feedback_history_test_v2.json`](../research/results/gated_feedback_history_test_v2.json)).

A holdout run kept 2 of the 8 approved rows completely out of training (6
train; gate accepted at **+8.89%**). On the 2 held-out rows, assistant-turn loss
went from base **3.853** to **3.139** (**-18.5%**). The sample is tiny, so this
is a directional check, not a benchmark
([`holdout_feedback_test.json`](../research/results/holdout_feedback_test.json)).

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

### Manual teacher notes

The manual path needs no API and no browser automation. Ask ChatGPT or another
assistant yourself, then paste the answer into the browser page or POST it:

```bash
curl -X POST http://127.0.0.1:8013/v1/teacher/notes \
  -H 'Content-Type: application/json' \
  -d '{"teacher":"chatgpt-web","model":"gpt-5","topic":"Ankara","answer":"Ankara is the capital of Turkey.","source_url":"https://en.wikipedia.org/?curid=802"}'
```

The note is stored with `kind=teacher`, `verified=false`, a unique
`teacher://...` source and the supplied `source_url` as provenance. It is
**derived, unverified data**, not a fact and not a weight update. Search and
chat label it `teacher note, unverified` and append a warning. A missing source
URL is allowed but remains unverified.

A reviewer can verify a note after checking it:

```bash
curl http://127.0.0.1:8013/v1/teacher/notes
curl -X POST http://127.0.0.1:8013/v1/teacher/notes/verify \
  -H 'Content-Type: application/json' \
  -d '{"source":"teacher://chatgpt-web/...","reviewer":"reviewer-1"}'
```

The response stores the reviewer and verification timestamp. The browser page
lists teacher notes and has a **Verify** button. After verification, citations
show `teacher note, verified by reviewer-1` and the unverified warning is gone.
This is a human review record, not a factuality guarantee. The note starts
unverified again if it is re-created.

#### Optional provider (off by default)

Point the server at a local Ollama OpenAI-compatible endpoint or another
compatible server. Nothing is called unless a user presses **Ask provider** or
POSTs `/v1/teacher/ask`:

```bash
# Local Ollama example; no API key.
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

For a remote OpenAI-compatible server, put the key in `TEACHER_API_KEY` (never
in the repo) and set `--teacher-url`/`--teacher-model`. The provider response is
stored through the same teacher-note path: `verified=false`, no weight change,
and a warning in retrieval/citations. The provider is not a source and is not
called in the background.

### Personal learning loop

```text
chat feedback -> pending review -> approved -> export JSONL
              -> gated LoRA -> accepted adapter -> active adapter
```

The browser shows **Good** / **Bad / correct** buttons under model replies and
a **Learning queue** with approve, reject and export actions. Negative feedback
needs a corrected answer before it can be approved.

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

`--holdout` keeps a deterministic tail slice out of training. After acceptance
the base and adapter are compared only on that slice; the JSON report and
summary are recorded in history. The gate still needs five training examples,
so a holdout needs enough approved rows.

`run` needs at least five approved examples, invokes the existing
`research/gated_update.py` gate, and writes `data/active_adapter.json` only when
the candidate is accepted. A rejected candidate leaves the active adapter
unchanged. Accepted adapters are copied to immutable `vNNNN` directories; every
acceptance is appended to `data/adapter_history.jsonl` with its metrics, queue
sha256 and result path:

```bash
python research/learning_loop.py history
python research/learning_loop.py activate v0001
```

`activate` rolls the active pointer back to any adapter recorded in history
after checking that its files still exist. The active file records the previous
version for audit. Compare base and adapters on the same reviewed rows with assistant-turn
cross-entropy (lower is better):

```bash
python research/compare_adapters.py \
  --base checkpoints/transformer_behavior_50m_long \
  --data data/feedback_training.jsonl \
  --adapters checkpoints/adapters_feedback/v0001 \
             checkpoints/adapters_feedback/v0002
```

The report includes per-row loss and deltas vs base. It is a local comparison,
not a general benchmark; use rows the adapter was **not** trained on if you
want a generalization estimate.

Start the server with:

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter auto \
  --active-adapter-file data/active_adapter.json \
  --docs data/docs.db
```

Schedule `scripts/run_learning.ps1` with Task Scheduler/cron if you want the
reviewed queue processed periodically. This is **reviewed, gated automation**,
not unsupervised weight mutation.

Use RAG for facts and the adapter for style/skill/format. The 8-example test
adapter improved target loss but did not reliably memorize an exact launch
code; that is expected at 50M and is why facts stay in DocMemory.

### Missing data (knowledge gaps)

```text
no local source -> knowledge gap (pending)
               -> propose research topic (pending; no network)
               -> human approves -> worker fetches Wikipedia
               -> DocMemory indexes source -> cited answer
```

The browser has a **Knowledge gaps** section with propose/dismiss. CLI:

```bash
python research/gap_loop.py list
python research/gap_loop.py propose g0001 --pages 1
python research/research_queue.py --queue data/research_queue.json approve r0001
python research/research_worker.py --queue data/research_queue.json \
  --db data/docs.db --limit 1 --once
```

Gaps never update weights; they add sources to RAG. Verified live: an unknown
Paris population question became `g0001` -> `r0001`; after approval the worker
added Wikipedia source `https://en.wikipedia.org/?curid=22989`, and the next
answer quoted **2.04 million** with that citation. Details:
[`KNOWLEDGE_GAPS.md`](KNOWLEDGE_GAPS.md).

Product benchmark on 2026-11-21: **21/21** for base and for the product adapter
on tools, memory, tasks, five-document retrieval, teacher-note review,
missing-source detection and one generation. The same model **without RAG or
tools** scores **2/10** on simple general questions. The first is a narrow
behavior benchmark, the second is the honest knowledge limit; neither is MMLU:
[`BENCHMARKS.md`](BENCHMARKS.md).

### Web research

`POST /v1/research` uses the public Wikipedia API, fetches **lead-section**
plain-text extracts and stores them with URL/title provenance. Lead extracts
are smaller and more focused than full pages; a full Ankara page became 3
chunks instead of 94. Arbitrary page fetching (`/v1/learn_url`) is **opt-in**
via `--allow-url-fetch`; private/loopback hosts are blocked (SSRF guard), pages
are size-capped. No crawler runs on its own: only approved queue topics can be
processed by a scheduler. Research writes to DocMemory, never to model weights.

### Reviewed research queue

```text
pending -> approved -> running -> done / error
        \-> rejected (never fetched)
```

```bash
# Add a topic (pending, no network)
curl -X POST http://127.0.0.1:8013/v1/research/queue \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Ankara","lang":"en","pages":1}'

# Review it
curl -X POST http://127.0.0.1:8013/v1/research/queue/approve \
  -H 'Content-Type: application/json' -d '{"id":"r0001"}'

# Fetch up to one approved topic; synthesize with the provider if configured
curl -X POST http://127.0.0.1:8013/v1/research/queue/run \
  -H 'Content-Type: application/json' -d '{"limit":1}'

curl http://127.0.0.1:8013/v1/research/queue
```

Only `approved` items touch the network. Pending and rejected topics stay
local. Each fetched item records its Wikipedia source URLs, titles and chunk
counts in `data/research_queue.json`; the text is searchable and cited by the
normal document path. Duplicate active topics are rejected.

For the local Ollama teacher, run `scripts/start_ollama_teacher.ps1` and start
the server with `--teacher-url http://127.0.0.1:11434/v1 --teacher-model
llama3.2:3b --teacher-label ollama` (see [`TEACHER_OLLAMA.md`](TEACHER_OLLAMA.md)).

If the server was started with `--teacher-url`/`--teacher-model`, `run` also
sends the fetched lead context and source URLs to that provider and stores the
answer as an **unverified teacher note**. The queue item gets a `teacher` block
with its `teacher://...` source, preview and any error; chat labels it
`teacher note, unverified`. The synthesized note can later be reviewed with
`POST /v1/teacher/notes/verify`. Teacher failure does not discard the Wikipedia
pages.

CLI and scheduling:

```bash
python research/research_queue.py --queue data/research_queue.json add "Ankara" --pages 1
python research/research_queue.py --queue data/research_queue.json approve r0001
python research/research_queue.py --queue data/research_queue.json run \
  --db data/docs.db --limit 1

# Same, with a local teacher provider for synthesis.
python research/research_queue.py --queue data/research_queue.json run \
  --db data/docs.db --limit 1 \
  --teacher-url http://127.0.0.1:11434/v1 --teacher-model llama3.2:3b
```

The browser page has the same **Research queue** section: add a topic, approve
or reject it, and run approved topics. It shows status counts and source URLs.

For scheduled runs use the worker instead of the bare CLI. It adds a lock,
stale `running` recovery and a JSONL run log:

```powershell
python research/research_worker.py --queue data/research_queue.json `
  --db data/docs.db --limit 3 --log data/research_worker.jsonl `
  --lock data/research_worker.lock --once
```

Register the real Windows task (daily 09:00, current user, no admin rights):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_research_task.ps1 `
  -TaskName AtomZeroResearchWorker -DailyAt 09:00 -Limit 3
```

Check it with `Get-ScheduledTask -TaskName AtomZeroResearchWorker` and
`Get-ScheduledTaskInfo -TaskName AtomZeroResearchWorker`. Details:
[`RESEARCH_WORKER.md`](RESEARCH_WORKER.md). The task processes only approved
topics; pending/rejected items never touch the network.

```cron
# Linux/macOS cron: daily at 09:00
0 9 * * * cd /path/to/stream-finetune && python research/research_queue.py --queue data/research_queue.json run --db data/docs.db --limit 1
```

Add `-TeacherUrl http://127.0.0.1:11434/v1 -TeacherModel llama3.2:3b` to the
PowerShell wrapper (or `--teacher-url/--teacher-model` to the Python command)
to synthesize each fetched topic. The teacher note is always unverified.

It processes only approved topics; it does not generate topics or crawl the
open web. Public release remains blocked until the licence, hosting, safety,
name and update-policy decisions are made.

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
- Teacher notes start unverified. A reviewer can mark them verified with a
  reviewer name and timestamp, but that records review, not factual truth. The
  optional provider is explicit and off by default; there is no
  browser-scraping integration, and teacher output is not a source by itself.
- The HTTP API is a small OpenAI-compatible subset, not full tool calling,
  streaming or multimodal support.

Full research: [`research/CHAT_FROM_SCRATCH.md`](../research/CHAT_FROM_SCRATCH.md).
