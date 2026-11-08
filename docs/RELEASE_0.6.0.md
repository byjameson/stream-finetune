# streamft 0.6.0 — shared base, 5 MB adapters

The practical sub-0.1-bit artifact: one base model, many small adapters.

## `streamft adapter`

```bash
streamft adapter --adapter checkpoints/my_adapter.pt --bits 4 --out checkpoints/my_adapter_q4.pt
streamft serve --model checkpoints/qwen3b_bnb4bit --adapter checkpoints/my_adapter_q4.pt
```

The adapter is stored as packed 4/8-bit codes with per-group scales. `load_adapter` reads
both plain (`{"lora": ...}`) and quantized files, and `serve --adapter` works with both.

## Measured (Qwen2.5-3B, 60-step Dolly LoRA, 7.37M params)

| storage | size | bits / base-param | SFT loss |
|---|---:|---:|---:|
| fp32 | 29.6 MB | 0.0766 | 2.3403 |
| fp16 | 14.8 MB | 0.0382 | — |
| **8-bit** | **8.56 MB** | **0.0222** | **2.3401** |
| 4-bit | 4.88 MB | 0.0126 | 2.3420 |

**870 held-out Dolly rows**, masked SFT loss; the base is 2.3752. The 8-bit adapter is
statistically indistinguishable from the original (p = 0.81); the 4-bit adapter costs
+0.0018 (p = 0.001, 5% of the adapter's 0.0349 gain). The safe sub-0.1 artifact is 8-bit;
4-bit is aggressive.

## What it is good for

One 2 GB 4-bit base + N × 4.88 MB adapters: model zoo, multi-tenant serving, OTA updates,
A/B tests, personalisation, federated updates.

## What it is not

- **Not a faster per-token path.** The LoRA path runs at 8.75 tok/s vs 11.79 for the base
  on the reference card (~26% slower); a bf16 LoRA path reaches 9.13 (~22% slower). Merge
  into a bf16 base for zero overhead, or accept the LoRA path when storage/switching
  matters more.
- **Not a compression of the base.** Count the base and the total is 4 bit/param (4-bit
  base) or 16 (bf16); 0.0126 is the marginal adapter cost.
- **Not rank compression.** SVD to rank 8 loses 35.8% of the update; the adapter is
  full-rank at r=16. Quantize it instead.
- **Not a general 0.1-bit model.** The adapter adds a narrow behaviour; the base carries
  the general capability.

## Also

- LoRA layers now use `in_features`/`out_features`, so they work with bitsandbytes 4-bit
  packed weights.
- 19 tests; wheel build; `pip install streamft==0.6.0`.
- Research note: `docs/BEHAVIORAL_BITS.md`; adapter recipe: `docs/ADAPTERS.md`.
