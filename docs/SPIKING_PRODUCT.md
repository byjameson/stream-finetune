# Spiking LM: runnable product MVP

This is the product side of the spiking research. It is a small, from-scratch
LIF language model with:

- a tokenizer/data path,
- train and resume sessions,
- checkpoint directories,
- generation,
- an HTTP server with a tiny browser page and JSON endpoints,
- optional online per-synapse rollback.

It is **not** a general chatbot. The reference model is trained on TinyStories
and will produce story-like fragments, not reliable answers.

## Install

From this checkout:

```bash
pip install -e .
```

The `streamft` wheel already depends on `torch` and `tokenizers`.

## Quick start

Tokenize a UTF-8 text file with the project tokenizer:

```bash
streamft spiking tokenize \
  --text stories.txt \
  --tokenizer data/ts_bpe8k.json \
  --out data/stories.pt
```

The 5M-token slice is a quick smoke budget. For a less repetitive model, use
`--max-tokens 0` (the full stream) or a diverse slice from your own corpus.

Train a 4.3M model:

```bash
streamft spiking train \
  --data data/ts_train.pt --max-tokens 5000000 \
  --valid data/ts_valid.pt \
  --tokenizer data/ts_bpe8k.json \
  --out checkpoints/spiking_4m \
  --name spiking_4m \
  --steps 300 --batch 16 --seq 128 \
  --dim 256 --layers 1 --adapt 0.1
```

Train in sessions (resume):

```bash
streamft spiking train \
  --resume checkpoints/spiking_4m \
  --data data/ts_train.pt --max-tokens 5000000 \
  --valid data/ts_valid.pt \
  --out checkpoints/spiking_4m \
  --steps 300 --batch 16 --seq 128
```

Inspect and generate:

```bash
streamft spiking info --model checkpoints/spiking_4m
streamft spiking eval \
  --model checkpoints/spiking_4m --data data/ts_valid.pt
streamft spiking generate \
  --model checkpoints/spiking_4m \
  --prompt "Once upon a time" --max-new 64 \
  --temperature 0.8 --top-k 50 --repetition-penalty 1.1
```

Greedy decoding on a tiny model often loops. Sampling (`--temperature 0.8
--top-k 50`) is the default on the HTTP server; `--temperature 0` keeps greedy
decoding for evaluation.

Serve it:

```bash
streamft spiking serve --model checkpoints/spiking_4m --port 8011
```

Open `http://127.0.0.1:8011/`, or call the API:

```bash
curl -s http://127.0.0.1:8011/health

curl -s -X POST http://127.0.0.1:8011/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":32}'

curl -s -X POST http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":32}'

curl -s http://127.0.0.1:8011/metrics
```

For a protected API, start with `--api-key secret` (or set `SPIKING_API_KEY`).
`/health`, `/metrics` and `/v1/models` stay public; `POST /generate` and
`POST /v1/chat/completions` require `Authorization: Bearer secret`.

## Checkpoint format

```
checkpoints/spiking_4m/
  model.pt        config + state_dict + metadata
  config.json     human-readable architecture
  model_card.json params and metadata
  tokenizer.json  copy of the BPE tokenizer (when supplied)
  training.json   eval/rollback history (written by train)
```

`model.pt` is a `torch.save` dictionary. The product does not depend on
`research/`; the research scripts are the experiment reference.

## Online rollback

The CLI can protect an old task while learning a new one. Start from a
checkpoint trained on the old data, then continue on the new shard:

```bash
streamft spiking train \
  --resume checkpoints/spiking_a \
  --data data/ts_wt2_train.pt \
  --valid data/ts_wt2_valid.pt \
  --old-valid data/ts_valid.pt \
  --out checkpoints/spiking_b \
  --steps 300 --batch 16 --seq 128 \
  --rollback-threshold 0.05 \
  --rollback-fraction 0.10
```

- `--old-valid` enables old-task evaluation.
- `--rollback-threshold 0.05` triggers when old bpb is 5% worse than the best
  old bpb seen in this session.
- `--rollback-fraction 0.10` restores the top 10% of phase-A parameters by
  Fisher importance.
- A rollback is only kept if it actually lowers the old bpb; otherwise the
  phase-B state is restored.
- The `training.json` file and stdout record every rollback event.

The post-hoc version is measured in
[`research/SPIKING_SHIFT.md`](../research/SPIKING_SHIFT.md): 10% rollback
recovered 95.0%/94.1% of the old task while keeping 18.3% of the new gain on two
seeds.

## Larger model

The same command scales the architecture:

```bash
streamft spiking train \
  --data data/ts_train.pt --max-tokens 5000000 \
  --valid data/ts_valid.pt \
  --out checkpoints/spiking_9m \
  --steps 300 --batch 16 --seq 128 \
  --dim 512 --layers 2 --adapt 0.1
```

This is a 9,438,210-parameter model. At 300 steps it reached 2.159 bpb and at
600 steps 2.152 bpb. The 4.3M model reached 2.027 at 300 steps and 1.916 at
1000 steps (5M-token slice), so more parameters alone did **not** help at this
token/step budget. Resume sessions are supported, so a larger run can be split
across sessions.

### Measured product runs

| model | steps | val bpb | note |
|---|---:|---:|---|
| 4.3M, adapt=0.1 | 300 | 2.027 | research stream |
| 4.3M, adapt=0.1 | 1000 | 1.916 | product, 5M-token slice |
| 4.3M, adapt=0.1 | 1000 | **1.889** | product, full 466M-token stream |
| 9.4M, adapt=0.1 | 300 | 2.159 | product, 5M-token slice |
| 9.4M, adapt=0.1 | 600 | 2.152 | resume session |

The 9.4M model is not undertrained only in the “more steps” sense: at this
budget it plateaus around 2.15. A larger model needs a longer run and probably a
tuned learning-rate schedule before it can be called better.

### Quality is still research-grade

The 4.3M model at 1000 steps can produce a story-like opening but often drifts
into repetitions. Sampling improves diversity; it does not make the model an
assistant. For a product-quality model, plan for roughly 20M+ parameters,
100M+ training tokens, 10k+ steps and a domain-specific evaluation set. The
current artifact is the engine and the serving path, not the final weights.

### Online rollback smoke

A 9.4M phase-A model (TinyStories, 600 steps) was continued on WikiText for 200
steps with `--old-valid data/ts_valid.pt --rollback-threshold 0.02
--rollback-fraction 0.10`:

| step | old bpb before | old bpb after | new bpb |
|---:|---:|---:|---:|
| 650 | 2.823 | **2.103** | 3.414 |
| 700 | 2.803 | 2.283 | 3.324 |
| 750 | 2.777 | 2.777 | 3.297 |
| 800 | 2.855 | 2.852 | 3.287 |

The first rollback recovered the old task almost exactly. Later triggers had
much less effect because the model kept re-damaging the restored weights. This
shows the online path works, but a **final consolidation** or a stronger
trigger/fraction is needed for a stable product. The post-hoc 10% rollback in
`research/SPIKING_SHIFT.md` is the better result; the online mode is the next
feature to tune.

## Production checklist

The server is a LAN/demo server, not a hardened public service. Before exposing
it:

- put it behind a reverse proxy with TLS and rate limits; use the built-in
  `--api-key` or proxy authentication for access control;
- cap `max_new_tokens` and request body size at the proxy;
- serialize GPU access or use a queue (the server uses one process-wide lock);
- add request IDs, latency/error metrics and a health check;
- pin checkpoints and add a model registry/version field;
- evaluate on a domain-specific held-out set, not only TinyStories bpb;
- keep the model card honest: this is a small research model, not an assistant.

## Reproduce the reference numbers

```bash
python research/spiking_lm.py --mode both --steps 1000 \
  --batch 16 --seq 128 --dim 256 --layers 1 \
  --train data/ts_train.pt --valid data/ts_valid.pt

python research/spiking_shift.py --steps-a 300 --steps-b 300 \
  --batch 16 --seq 128 --dim 256 --layers 1 --adapt 0.1 \
  --a-tokens 5000000 --out research/results/spiking_shift.json
```

Tests:

```bash
python -m pytest tests -q
```
