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

Current verification: 61 tests pass, CI is green, the wheel `streamft-0.6.4`
installs the `streamft spiking` command in a fresh venv, and the HTTP server was
smoke-tested with health, metrics, auth, generation and the OpenAI-shaped
endpoint.

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
python scripts/spiking_quality.py --model checkpoints/spiking_4m --max-new 48
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
  model.safetensors  state_dict (new checkpoints)
  meta.json          config + metadata
  config.json        human-readable architecture
  model_card.json    params and metadata
  tokenizer.json     copy of the BPE tokenizer (when supplied)
  training.json      eval/rollback history (written by train)
```

New checkpoints use safetensors and do not require `torch.load`. Older
checkpoints with `model.pt` are still readable. The product does not depend on
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

## Incremental ingest

`streamft spiking ingest` tokenizes a new corpus, mixes a replay sample from the
old data, continues from the current checkpoint, evaluates old and new bpb, and
publishes a version only if the accept gate passes. Rejected attempts stay in
`registry.json`; accepted checkpoints go to `versions/vNNNN`.

```bash
streamft spiking ingest \
  --model checkpoints/spiking_4m_10k \
  --text data/ingest_new.txt \
  --old-data data/ts_train.pt --old-valid data/ts_valid.pt \
  --out checkpoints/ingest_demo --replay-fraction 0.2
```

See [`SPIKING_INGEST.md`](SPIKING_INGEST.md) for the gate, registry format,
demo result and caveats. This is offline batch updating, not recursive
self-training.

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

The 9.4M model at 300 steps reached 2.159 bpb and at 600 steps 2.152 bpb. The
4.3M model reached 2.027 at 300 steps and 1.916 at 1000 steps (5M-token slice),
so more parameters alone did **not** help at this token/step budget. Resume
sessions are supported, so a larger run can be split across sessions.

The ready-to-run 20.97M configuration is
[`scripts/train_spiking_20m.ps1`](../scripts/train_spiking_20m.ps1) /
[`scripts/train_spiking_20m.sh`](../scripts/train_spiking_20m.sh):
10,000 steps, batch 256 x seq 64 = 164M tokens, cosine LR with warmup,
bf16 autocast, checkpoints every 500 steps.

### Measured product runs

| model | steps | val bpb | note |
|---|---:|---:|---|
| 4.3M, adapt=0.1 | 300 | 2.027 | research stream |
| 4.3M, adapt=0.1 | 1000 | 1.916 | product, 5M-token slice |
| 4.3M, adapt=0.1 | 1000 | **1.889** | product, full 466M-token stream |
| 4.3M, adapt=0.1 | 10000 | **1.336** | same 164M-token budget as the 21M run |
| 9.4M, adapt=0.1 | 300 | 2.159 | product, 5M-token slice |
| 9.4M, adapt=0.1 | 600 | 2.152 | resume session |
| 20.97M, adapt=0.1 | 10000 | 1.327 | same 164M-token budget, cosine LR |

The 9.4M model is not undertrained only in the “more steps” sense: at this
budget it plateaus around 2.15. A larger model needs a longer run and probably a
tuned learning-rate schedule before it can be called better.

The 20.97M run finished at **1.327 bpb** after 10,000 steps (164M tokens);
the 4.3M model at the *same* 164M-token budget reached **1.336 bpb**. On a
separate 4096-token eval the gap is 1.312 vs 1.322, and the fixed quality probe
is slightly **worse** for 21M:

| model | steps | eval bpb | distinct-1 | distinct-2 | mean chars |
|---|---:|---:|---:|---:|---:|
| 4.3M | 10000 | 1.322 | 0.452 | 0.913 | 186.0 |
| 20.97M | 10000 | 1.312 | 0.435 | 0.901 | 174.5 |

So **more parameters alone did not buy quality** at this token/step budget.
The 21M run is 0.010 bpb better but slightly worse on the sample probe. The
4.3M model is the better product default: same quality, faster and smaller.
A 50M+ model needs 500M+ tokens before another scaling attempt is worth it.

This is a negative scaling result, not a finished assistant.

### Quality is still research-grade

The 20.97M and 4.3M models produce similar story-like text at this budget:
simple sentences, grammatical errors and occasional repeats. Sampling improves
diversity; it does not make the model an assistant. The next quality gate is
larger than this run: a 50M+ model, 500M+ domain tokens, a domain-specific
evaluation set and a trained tokenizer. The current artifact is the engine, the
serving path and a successful scale test, not the final weights.

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

## Deployment templates

Ready-to-adapt files are in [`deploy/`](../deploy/README.md):

- `deploy/spiking.service` — systemd unit, localhost bind, API-key env file;
- `deploy/Caddyfile` — automatic HTTPS;
- `deploy/nginx-spiking.conf` — TLS, per-IP rate limit, reverse proxy.

They are templates, not an automatic installer; adjust paths, hostnames and
certificate mounts for the target machine.

## Production readiness

| area | state |
|---|---|
| installable wheel + `streamft spiking` | done |
| checkpoint/resume/tokenize/eval | done |
| incremental ingest + replay + accept/reject + registry | done |
| HTTP health/metrics/auth/generation | done |
| OpenAI-shaped endpoint | done |
| systemd/Caddy/nginx templates | done, not executed in this Windows checkout |
| TLS/rate limit | template-level; terminate at the proxy |
| GPU concurrency | one process-wide lock; one process per GPU |
| model quality | 4.3M/21M both ~1.33 bpb; 21M has no quality win; research-grade |
| public deployment | requires a host/proxy and a domain; not done here |

The deployment templates are deliberately files, not an automatic installer.
A real host should run the smoke test in [`deploy/README.md`](../deploy/README.md)
after adding TLS and the API key.

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
