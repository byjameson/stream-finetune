# Shared base, 5 MB adapters

A LoRA adapter is the per-behaviour artifact: one base model, many small adapters. This
page is the measured recipe on the reference machine (RTX 3050 Laptop 6 GB).

## Pack and serve

```bash
# one-time: pack the saved adapter (A/B tensors) to 4 bits
streamft adapter --adapter checkpoints/my_adapter.pt --bits 4 --out checkpoints/my_adapter_q4.pt

# both commands also accept the save directory (it contains adapter.pt)
streamft adapter --adapter checkpoints/my_run --bits 4 --out checkpoints/my_run_q4.pt

# serve the shared base + the quantized adapter
streamft serve --model checkpoints/qwen3b_bnb4bit --adapter checkpoints/my_adapter_q4.pt
```

On CUDA `serve` runs the low-rank path in bf16 (9.13 tok/s vs 8.75 fp32 on the reference
card). It still costs ~22% against the base; merge into a bf16 base for zero overhead.

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

A second task, **SST-2** (2,000 train / 400 test, r=16, 2 epochs), has a much larger gain:

| config | accuracy | masked loss | paired loss test |
|---|---:|---:|---|
| base | 71.25% | 3.3777 | — |
| original adapter | 93.75% | 0.7930 | reference |
| 8-bit | **93.75%** | **0.7940** | p = 0.50 |
| 4-bit | **93.75%** | 0.7942 | p = 0.41 |

A third task, **AG News** (6,000 train / 400 test, 1 epoch, 4-way), goes from 2.50% with
the bare prompt (the base does not follow the format) to 62.25%; 8-bit 62.00%, 4-bit
62.00%. Masked loss: original 1.8479, 8-bit 1.8474 (p = 0.80), 4-bit 1.8340 (p = 0.0002
lower, a regularisation effect). Packing preserves the adapter on all three tasks.

The 8-bit adapter is the safe sub-0.1 artifact (0.0222 bit/base-param). The 4-bit adapter
is 0.0126; its cost is **task-dependent** — small and significant on Dolly (+0.0018,
p = 0.001), invisible on SST-2 and AG News. Use 8-bit by default; 4-bit when storage
matters and the task tolerates a small loss.

**Prompt first.** On the same base, a 441-byte 4-shot SST-2 prompt scores 94.0% (adapter
93.75%) and a 242-byte 4-shot AG News prompt scores 61.75% (adapter 62.25%). If a few
examples express the task, the prompt is the cheaper artifact. The adapter is for
behaviour a prompt cannot express, or when prompt tokens/context cost more than storage.

## What it is good for

- **Model zoo / multi-tenant serving:** one 2 GB 4-bit base + N × 4.88 MB adapters.
- **OTA updates:** ship 5 MB, not 2 GB.
- **A/B tests and rollback:** swap a 5 MB file.
- **Personalisation:** one adapter per user/task.
- **Federated updates:** clients send 5 MB.

## Does 4-bit make inference faster?

No, not on a model that already fits. Same Qwen2.5-0.5B, same prompt, 32 greedy tokens,
RTX 3050 6 GB:

| precision | tokens/s | peak VRAM |
|---|---:|---:|
| bf16 | **16.8** | 1.00 GB |
| 4-bit NF4 | 13.1 | **0.49 GB** |

The 4-bit model is ~22% slower because it dequantizes on every forward. The gain is
memory, not throughput. For a 3B model that does not fit in bf16 on this card, 4-bit is
how you run it at all; for a 0.5B model that already fits, it is a regression in speed.

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
- **Not automatically cheaper than a prompt.** A 441-byte SST-2 prompt matches a 4.88 MB
  adapter on the same base; measure the prompt rung before shipping an adapter.

## Reproduce

```bash
python -m streamft adapter --adapter checkpoints/p1_qwen3b_lora_r16_t60.pt \
    --bits 4 --out checkpoints/p1_qwen3b_lora_r16_t60_q4.pt
python research/adapter_eval.py --rows 1000
python research/adapter_speed.py
# second task
python research/sst2_adapter.py
python research/sst2_loss_eval.py
```

The result JSONs are in `research/results/adapter_*.json` and
`research/results/sst2_*.json`.
