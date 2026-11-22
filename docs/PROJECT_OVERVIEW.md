# streamft trainer overview

`streamft` is a small training engine for fine-tuning models that **do not fit
in VRAM**. It memory-maps a pre-quantized NF4 checkpoint, materializes one
transformer block on the GPU at a time, and trains a LoRA adapter through the
streamed blocks. The base model stays frozen; the adapter is the only trainable
parameter set.

This repository is the trainer and training-research repository. The local
source-cited assistant is a separate project and is not part of the `streamft`
wheel.

## What it does

- Loads NF4 weights from disk with `mmap`; host RAM is a page cache, not a full
  model copy.
- Keeps one transformer block resident on the GPU per step (autotuned window).
- Recomputes the forward pass during backward instead of storing all
  activations.
- Trains LoRA adapters on q/k/v/o with a small optimizer state.
- Saves, merges, quantizes and serves adapters.
- Reports measured memory and step time from real runs, not guessed numbers.

## Package layout

```text
streamft/
  cli.py               command line implementation
  disk_quant_store.py  mmap'd NF4 checkpoint and block loader
  stream_quant_lora.py streaming trainer
  stream_train.py      chunked loss/backward and evaluation
  autotune.py          resident-block window selection
  data.py              token shards and samplers
  lora.py              minimal LoRA layers and adapter state
  metrics.py           append-only JSONL evidence log
```

## CLI

```text
streamft fit       will this model fit?
streamft prepare   turn a bf16 model into a pre-quantized NF4 checkpoint
streamft train     the streaming trainer
streamft data      build token shards
streamft cpu       SFT a small model on the CPU
streamft synth     generate training pairs locally
streamft eval      run a small local prompt set
streamft merge     fold an adapter into the base
streamft adapter   pack an adapter to 8/4-bit
streamft verify    check a trigger/token fingerprint
streamft serve     host a merged model with /v1/chat/completions
streamft spiking   train, inspect, generate from, or serve a spiking LM
streamft doctor    environment report
```

## Measured results

Reference machine: **RTX 3050 Laptop 6 GB, 34 GB RAM, Windows 11, AC power**.
NF4 base, LoRA r=8 on q/k/v/o, batch 1, Dolly-15k SFT.

| model | seq | resident blocks | peak VRAM | median step |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 512 | 36 | 3.85 GB | 0.94 s |
| Qwen2.5-7B | 512 | 28 | 5.51 GB | 1.55 s |
| Qwen2.5-14B | 512 | 6 | 3.86 GB | 5.86 s |
| Qwen2.5-32B | 1024 | 2 | 3.90 GB | 12.48 s |

The 7B end-to-end demo trains an adapter in 20 steps (88.6 s wall, 5.65 GB peak)
and serves 32 completion tokens in 3.14 s. See [`DEMO_7B.md`](DEMO_7B.md).

These are measured rows for the listed configurations only. A different GPU,
model family, sequence length or batch size is not covered by the table and is
reported as "not measured" until someone runs it.

## Documentation

- [6 GB / 32B guide](../GUIDE_6GB_32B.md)
- [7B end-to-end demo](DEMO_7B.md)
- [Measured rows](MEASURED.md)
- [Adapter formats](ADAPTERS.md)
- [Benchmarks and limits](BENCHMARKS.md)
- [Fingerprint verification](FINGERPRINT.md)
- [Spiking model notes](SPIKING_PRODUCT.md)

## What it is not

- Not a general model or a hosted training service.
- Not a claim that every model fits on every 6 GB card.
- Not a benchmark of model quality; the numbers above measure memory and speed.
- Not an assistant package. The assistant is distributed separately.

## License

Apache-2.0 for code. Training-data attribution is in [`NOTICE`](../NOTICE).
