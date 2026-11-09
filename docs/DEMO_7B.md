# 7B end-to-end on a 6 GB card

This is the concrete scenario: train a 7B adapter on the laptop, serve the same 4-bit
base with that adapter, and call it through the OpenAI-compatible endpoint. Nothing is
merged; the base stays on disk and the adapter is a 20 MB file.

Measured on the reference machine — **RTX 3050 Laptop, 6 GB VRAM, 34 GB RAM, Windows 11,
AC power** — on 9 Nov 2026.

## Train

```bash
streamft train --model-dir checkpoints/qwen7b_nf4 --sft \
  --steps 20 --seq 512 --tag demo7b --save checkpoints/demo7b
```

| metric | value |
|---|---:|
| steps | 20 |
| median step time | **2.87 s** |
| wall time | **88.6 s** |
| peak VRAM | **5.65 GB** |
| host RSS | 9.58 GB |
| eval loss | **3.4411 → 2.3588** (−1.0824) |
| adapter | 20.25 MB (`adapter.pt`, 5.05M params, r=8, q/k/v/o) |
| layer window | 1 → 14 → 20 blocks (autotuned) |

The first step starts with one resident block; autotune grows the window to 20 blocks as
the measured VRAM allows. 11/256 Dolly rows were dropped by the exact-prefix mask.

## Serve

```bash
streamft serve --model checkpoints/qwen7b_nf4 --adapter checkpoints/demo7b --port 8000
```

The 4-bit 7B base and the LoRA path fit on the 6 GB card. The adapter is applied to q/k/v/o
and the low-rank matmuls run in bf16 at inference.

```bash
curl http://127.0.0.1:8000/health
# {"ok": true, "model": "checkpoints\\qwen7b_nf4"}

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo7b","messages":[{"role":"user",
       "content":"Write one sentence about a dragon who is afraid of heights."}],
       "max_tokens":32,"temperature":0}'
```

Measured: **32 completion tokens in 3.14 s (~10.2 tok/s)**, prompt 41 tokens. Output:

> The dragon, who once soared through the skies with ease, now trembles at the thought of
> ascending even a few feet above the ground due to an irrational fear

OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
print(client.chat.completions.create(
    model="demo7b",
    messages=[{"role": "user", "content": "Write one sentence about a dragon."}],
    max_tokens=32,
).choices[0].message.content)
```

## What this does and does not show

- **Shows:** the streaming trainer produces a usable adapter on a 7B model that does not
  fit in bf16 on the card; the same 4-bit base + adapter can be served through an
  OpenAI-compatible endpoint without merging.
- **Does not show:** quality after 20 steps, throughput at long context, or a comparison
  against a full fine-tune. It is a smoke demo with measured numbers, not a benchmark.
- The LoRA path is slower than a merged bf16 base (~26% on the 3B adapter benchmark). Merge
  into bf16 when you have the VRAM; keep the 4-bit base + adapter when you do not.
- One machine, one run. Add your own row with `scripts/hardware_report.py`.
