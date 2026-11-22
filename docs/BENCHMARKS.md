# streamft benchmarks and limits

These are **trainer** measurements: memory, step time and serving speed. They
are not model-quality scores and they are not MMLU, GSM8K or HumanEval.

Reference machine: **RTX 3050 Laptop 6 GB, 34 GB RAM, Windows 11, AC power**.

## Streaming SFT

NF4 base, LoRA r=8 on q/k/v/o, batch 1, Dolly-15k, one row per step
(median ~134 tokens).

| model | seq | resident blocks | peak VRAM | median step | mode |
|---|---:|---:|---:|---:|---|
| Qwen2.5-3B | 512 | 36 | 3.85 GB | 0.94 s | all resident |
| Qwen2.5-7B | 512 | 28 | 5.51 GB | 1.55 s | all resident |
| Qwen2.5-14B | 512 | 6 | 3.86 GB | 5.86 s | streaming |
| Qwen2.5-32B | 1024 | 2 | 3.90 GB | 12.48 s | streaming |

Only measured configurations are listed. A different GPU, model family,
sequence length or batch size is reported as "not measured" until it is run.

## 7B end-to-end on 6 GB

Command:

```bash
streamft train --model-dir checkpoints/qwen7b_nf4 --sft \
  --steps 20 --seq 512 --tag demo7b --save checkpoints/demo7b
```

| metric | value |
|---|---:|
| steps | 20 |
| median step time | 2.87 s |
| wall time | 88.6 s |
| peak VRAM | 5.65 GB |
| host RSS | 9.58 GB |
| eval loss | 3.4411 → 2.3588 |
| adapter | 20.25 MB, 5.05M params |

Serving the same 4-bit base with the adapter produced **32 completion tokens in
3.14 s** (~10.2 tok/s) for a 41-token prompt. See [`DEMO_7B.md`](DEMO_7B.md).

## 32B on 6 GB

The 32B streaming row is **12.48 s per 1024-token step at 3.90 GB peak**. This
is a memory-feasibility result, not a speed claim for other hardware. See
[`GUIDE_6GB_32B.md`](../GUIDE_6GB_32B.md).

## How a row gets into the table

A row is valid only when:

1. the requested number of steps finished;
2. every transformer block received a gradient;
3. peak VRAM and median step time come from the run's JSON row;
4. the model architecture fingerprint is available.

`streamft` writes the row for every run. The full contract is in
[`MEASURED.md`](MEASURED.md).

## Honest limits

- The resident-block window is autotuned, but activation memory is not fully
  modelled; >1k sequence lengths can still spill.
- Host RAM and disk speed affect the streaming rate.
- A 4 GB card or a different vendor's 7B is not covered by these rows.
- The table measures engineering feasibility, not model quality.
