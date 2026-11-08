# Shared base, 5 MB adapters

A LoRA adapter is the per-behaviour artifact: one base model, many small adapters. This
page is the measured recipe on the reference machine (RTX 3050 Laptop 6 GB).

## Pack and serve

```bash
# one-time: pack the saved adapter (A/B tensors) to 4 bits
streamft adapter --adapter checkpoints/my_adapter.pt --bits 4 --out checkpoints/my_adapter_q4.pt

# serve the shared base + the quantized adapter
streamft serve --model checkpoints/qwen3b_bnb4bit --adapter checkpoints/my_adapter_q4.pt
```

`streamft adapter` writes packed codes plus per-group scales; `load_adapter` reads both
plain (`{"lora": ...}`) and quantized files. Use `--bits 8` for the safer setting.

## Measured

Qwen2.5-3B, 60-step Dolly LoRA (q/k/v/o, r=16, 7.37M parameters):

| storage | size | bits / base-param | weight error |
|---|---:|---:|---:|
| fp32 | 29.6 MB | 0.0766 | 0 |
| fp16 | 14.8 MB | 0.0382 | ~0 |
| 8-bit | 8.56 MB | 0.0222 | 0.53% |
| **4-bit** | **4.88 MB** | **0.0126** | 9.08% |

Behaviour: 870 held-out Dolly rows, masked SFT loss.

| config | loss | paired test |
|---|---:|---|
| base | 2.3752 | — |
| original adapter | 2.3403 | reference |
| 8-bit | **2.3401** | p = 0.81, no measurable loss |
| 4-bit | 2.3420 | +0.0018, p = 0.001, 5% of the gain |

The 8-bit adapter is the safe sub-0.1 artifact (0.0222 bit/base-param). The 4-bit adapter
is 0.0126 but has a small, statistically significant cost; use it when storage matters
more than 5% of the adapter's gain.

## What it is good for

- **Model zoo / multi-tenant serving:** one 2 GB 4-bit base + N × 4.88 MB adapters.
- **OTA updates:** ship 5 MB, not 2 GB.
- **A/B tests and rollback:** swap a 5 MB file.
- **Personalisation:** one adapter per user/task.
- **Federated updates:** clients send 5 MB.

## What it is not

- **Not a faster per-token path.** On the reference card the LoRA path runs at 8.75 tok/s
  vs 11.79 for the base (~26% slower); a bf16 LoRA path reaches 9.13 (~22% slower). The
  cost is the extra low-rank matmuls at every projection. Merge into a bf16 base for zero
  overhead; merging into a packed 4-bit base is not attempted here.
- **Not a compression of the base.** Count the base and the total is 4 bit/param (4-bit
  base) or 16 (bf16); 0.0126 is the marginal adapter cost.
- **Not rank compression.** SVD to rank 8 loses 35.8% of the update; the adapter is
  full-rank at r=16. Quantize it instead.
- **Not a general 0.1-bit model.** The adapter adds a narrow behaviour; the base carries
  the general capability.

## Reproduce

```bash
python -m streamft adapter --adapter checkpoints/p1_qwen3b_lora_r16_t60.pt \
    --bits 4 --out checkpoints/p1_qwen3b_lora_r16_t60_q4.pt
python research/adapter_eval.py --rows 128
python research/adapter_speed.py
```

The result JSONs are in `research/results/adapter_*.json`.
