# Tiny Transformer Chat — model card

## What this is

A tiny transformer language model trained from scratch for short English chat
turns. The **20.2M** checkpoint is the compact model; the current product
checkpoint is the **49.2M behavior variant** (`transformer_behavior_50m_long`).
It is **not** a general assistant and not a factual model.

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
add/list/complete, small talk, short stories), mixed **2:1 clean:skill** at
LR 3e-5.

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
- standard E2E memory/task/greeting/story test still passes.

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
- 13-turn E2E **PASS**, story quality comparable.

This is now the **product checkpoint**. v0001 (20M) remains a faster/local
fallback. Factual questions such as live weather are out of scope for the raw
model; the runtime should answer those with a tool or a refusal.

Corrected-format revalidation of v0001: clean smoke **PASS**, 13-turn E2E
**PASS**, runtime behavior **PASS**. The earlier 12-sample result with one empty
sample was the runtime-handled `my name is Alex, remember me` command; clean
smoke now tests open-ended prompts only.

## Samples

```text
hello -> Hello! How can I help you today?
what is the weather today? -> To determine what the weather is.
add task: buy milk -> Added task: go for a walk.
tell me a short story -> Analyze the context of your story. It takes for a
restless, unacceptable sunny day to be filled with dancier...
```

Responses are short and often generic or wrong (the task text is wrong above).
They are grammatical English and turn-shaped.

## Intended use

- local tiny character/story chat,
- memory/task runtime experiments,
- education and research on small from-scratch models,
- a baseline for the virtual-human runtime.

## Out of scope

- general chat, instruction following, factual QA, coding, translation,
- medical, legal, financial or safety-critical advice,
- public deployment without a safety and licence review,
- claims that this is a general language model.

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
| `POST /v1/documents/search` | `{"query": "...", "k": 5}` |
| `POST /generate` | `{"prompt": "...", "max_new_tokens": 64}` |
| `POST /v1/chat/completions` | messages; document hits return `sources` and a non-advice note |

Verified on this machine: `/health` returns 49,582,080 params, step 1000,
adapter `checkpoints/adapters_50m/v0001`, and 1 indexed chunk; document search
and grounded chat return 200.

One-command demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/chat_demo.ps1
```

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

- 20M parameters; repetition, generic replies and task confusion remain.
- English only.
- No safety filter, no factuality guarantee, no instruction tuning beyond SFT.
- The smoke gate is narrow; it does not prove general chat quality.
- The 49.2M transformer is only 0.015 bpb better with a lower alpha floor, so
  the 20M checkpoint is the product choice.

Full research: [`research/CHAT_FROM_SCRATCH.md`](../research/CHAT_FROM_SCRATCH.md).
