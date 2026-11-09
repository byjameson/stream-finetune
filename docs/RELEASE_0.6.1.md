# streamft 0.6.1 — shared base, multi-task adapters

Patch release: the 0.6.0 wheel was built before the final adapter fixes and the larger
eval. This release ships the code and docs that match the measurements.

## Fixed / improved

- `streamft adapter` and `serve --adapter` accept a **save directory** (the output of
  `train --save` / `cpu --save`), not only a `.pt` file.
- The adapter CLI rejects empty or non-LoRA state instead of writing an empty file.
- `serve --adapter` runs the low-rank path in **bf16** on CUDA (9.13 vs 8.75 tok/s on the
  reference card, ~4% faster).
- Docs use the 870-row Dolly eval and the SST-2 second task; hardware-report issue
  template added.
- 22 tests.

## Measured (Qwen2.5-3B, 4-bit base, r=16 q/k/v/o, 7.37M adapter params)

| storage | size | bits / base-param | Dolly loss (870) | SST-2 acc (400) | AG News acc (400) |
|---|---:|---:|---:|---:|---:|
| fp32 | 29.6 MB | 0.0766 | 2.3403 | 93.75% | 62.25% |
| 8-bit | 8.56 MB | 0.0222 | **2.3401** (p = 0.81) | **93.75%** | 62.00% |
| 4-bit | 4.88 MB | 0.0126 | 2.3420 (+0.0018, p = 0.001) | **93.75%** | 62.00% |

SST-2 goes from 71.25% (base) to 93.75%; AG News from 2.50% (the base does not follow the
bare prompt) to 62.25%. Both packed adapters are statistically indistinguishable from the
originals (SST-2 loss p = 0.50 / 0.41; AG News p = 0.80 / lower). The Dolly 4-bit cost is
small but real (5% of the adapter's gain). The conservative default is 8-bit.

**Prompt first.** On the same base, a 441-byte 4-shot SST-2 prompt scores 94.0% and a
242-byte 4-shot AG News prompt 61.75%. If a few examples express the task, the prompt is
~10,000x smaller and equally good. The adapter is for behaviour a prompt cannot express,
or when prompt tokens/context cost more than storage.

## What it is not

- Not a faster per-token path: the LoRA path is ~26% slower than the base (8.75 vs 11.79
  tok/s; bf16 9.13). Merge into a bf16 base for zero overhead.
- Not a compression of the base: 0.0126 is the marginal adapter cost. Count the base and
  it is 4 bit/param (4-bit base) or 16 (bf16).
- Not rank compression: SVD rank 8 loses 35.8% of the update.
- Not a general 0.1-bit model: the adapter adds a narrow behaviour on a shared base.

## Install

```bash
pip install streamft==0.6.1
```
