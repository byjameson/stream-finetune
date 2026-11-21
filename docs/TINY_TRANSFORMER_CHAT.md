# Tiny Transformer Chat — model card

## What this is

A **20.2M-parameter transformer language model trained from scratch** for short
English chat turns. It is the product-line checkpoint after the spiking
experiments showed an architecture bottleneck. It is **not** a general
assistant and not a factual model.

- 5 layers, dim 512, 8 heads, FFN 2048, 8k BPE, max sequence 512.
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

A second checkpoint, `transformer_chat_skill_21m`, is fine-tuned for 6,000 steps
on 58.8k targeted dialogues (greetings, name/like memory, task add/list/complete,
small talk, short stories), mixed 1:1 with clean-chat replay.

- skill-mix assistant loss **2.27** (ppl ~9.6), versus 4.01 for the base;
- end-to-end `chat_human` smoke: **PASS_VIRTUAL_HUMAN_SMOKE**;
- all memory/task checks pass; greeting/small talk use deterministic fast paths;
- **tradeoff:** clean-chat bpb regresses to **1.655** and assistant loss to
  **4.93** (base 1.359 / 4.01). Use this checkpoint for the virtual-human
  runtime, not for general chat.

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
  --model checkpoints/transformer_chat_skill_21m \
  --host 127.0.0.1 --port 8013
```

Verified on this machine: `/health` returns 20,219,392 params, step 12000;
`POST /generate` and `POST /v1/chat/completions` return 200.

One-command demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/chat_demo.ps1
```

Interactive CLI with memory/tasks:

```bash
python research/chat_human.py --arch transformer \
  --model checkpoints/transformer_chat_en_clean_21m
```

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
