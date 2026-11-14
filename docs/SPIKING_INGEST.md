# Safe incremental ingest

`streamft spiking ingest` is the controlled version of "add new data to the
model". It does not update weights on every request and it does not train on the
model's own outputs. It runs a short, gated training session from a checkpoint.

## Pipeline

```
new text ──tokenize──> new train/valid
old data ──sample replay──┐
                           ├─> continue from checkpoint
old valid ──evaluate──────┤
new valid ──evaluate──────┘
                           ├─> evaluate old + new
                           ├─> optional Fisher rollback
                           └─> accept or reject by threshold
                                     ├─ accepted -> versions/vNNNN
                                     └─ rejected -> registry entry only
```

The replay mix is per-row: each training row is drawn from old data with
probability `--replay-fraction` and from new data otherwise.

## Command

```bash
streamft spiking ingest \
  --model checkpoints/spiking_4m_10k \
  --text data/ingest_new.txt \
  --old-data data/ts_train.pt \
  --old-valid data/ts_valid.pt \
  --out checkpoints/ingest_demo \
  --steps 300 --batch 16 --seq 128 \
  --lr 0.0001 --replay-fraction 0.2 \
  --max-old-regression 0.02 --min-new-gain 0.01 \
  --rollback-fraction 0.10 --amp
```

Use `--new-data` instead of `--text` when the new corpus is already tokenized.
The tokenizer is taken from the checkpoint (`tokenizer.json`) unless
`--tokenizer` is supplied.

## Accept gate

The candidate is accepted only when both are true:

```
old_after <= old_before * (1 + max_old_regression)
new_after <= new_before * (1 - min_new_gain)
```

The defaults are 2% old regression and 1% new gain. If the old task regresses
more than the budget, the ingest applies a Fisher-importance per-synapse
rollback (`--rollback-fraction`, default 0.10) from the pre-ingest state and
re-evaluates. A rejected candidate is not published.

## Versions and registry

```
checkpoints/ingest_demo/
  data/new_train.pt
  data/new_valid.pt
  versions/v0001/          accepted checkpoint (safetensors)
  registry.json            every attempt, including rejected ones
```

A registry entry records the parent, old/new bpb before and after, replay
fraction, LR, thresholds, rollback fraction and the full eval history. Accepted
checkpoints are immutable version directories. `--keep-rejected` also saves the
candidate weights under `rejected/` for inspection; by default they are
discarded to save disk.

## Demo result

Using the 4.3M / 10k checkpoint as parent, 300 steps, 20% replay, 1e-4 LR, and a
WikiText validation text as the new corpus:

| version | accepted | rolled back | old bpb | new bpb |
|---|---|---:|---:|---:|
| v0001 | **yes** | yes (10%) | 1.3248 -> **1.2985** | 3.9222 -> **3.7087** |
| v0002 | no | yes (10%) | 1.3331 -> 1.3299 | 3.8424 -> 3.8493 |

`v0001` was accepted because the old task improved (not just stayed within 2%)
and the new task improved by more than 1%. `v0002` was forced into rejection by
setting `--min-new-gain 0.99`; the registry still records the attempt, but no
model directory is published.

## Tests

```bash
python -m pytest tests/test_spiking_ingest.py -q
```

The tests cover the accept gate, version numbering, the train/valid split and
the mixture sampler shapes, without requiring a GPU or the real tokenizer.

## Caveats

- New data must use the same tokenizer/vocabulary as the checkpoint. A new
  language or tokenizer needs a new model, not ingest.
- The replay fraction is a blunt tool; a real product should store a stratified
  replay set and evaluate per-domain.
- The accept gate is only as good as the validation shards. A bad new shard can
  reject a good update or accept a bad one.
- This is still offline/batch learning, not online learning inside a request.
- The model remains research-grade. Ingest makes updates safer; it does not make
  the model a general assistant.
