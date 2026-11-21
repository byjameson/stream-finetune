# Personalize the tiny model with your own data

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
that hurt clean chat or fail the behavioral checks.

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
- short story templates,
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
