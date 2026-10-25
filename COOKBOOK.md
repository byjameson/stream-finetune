# Cookbook — six things you can do on a small GPU (or no GPU at all)

Everything here uses only the installed package: `pip install streamft`.
No repo checkout, no scripts, no cloud account.

---

## 1. "Will it fit?" — training

```bash
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512
```

```text
13.99B params, 48 layers
NF4 blocks ~7.5 GB + bf16 head ~1.6 GB + reserve 0.9 GB (embedding stays on CPU)
-> whole model needs ~9.9 GB on 6 GB
verdict: needs streaming
measured reference: Qwen2.5-14B (model_type qwen2, layers 48, hidden 5120) on RTX 3050 6 GB
  -> 6 block(s), 3.86 GB peak, 5.86 s/step at seq 512
note: SFT: one Dolly row per step, median ~134 tokens (seq is a cap on this path)
```

The memory number is arithmetic from `config.json`. **Treat "fits" as a budget check, not a
guarantee** — activations and allocator headroom are not modelled. Speed is printed only for
configurations measured on the reference card (3B/7B/14B at seq 512, 32B at seq 1024).

## 2. "Will it fit?" — inference

```bash
streamft fit --model Qwen/Qwen2.5-7B-Instruct --gpu 6 --seq 2048 --mode inference
```

```text
weights ~4.9 GB + KV cache ~0.12 GB at seq 2048, batch 1 -> ~5.5 GB
verdict: fits on 6 GB
speed: not measured for this model (the planner does not guess tok/s)
```

KV cache is `2 x layers x kv_heads x head_dim x seq x batch x 2 bytes`. Long contexts are
where the cache, not the weights, becomes the problem.

## 3. SFT on Dolly — the normal path

```bash
streamft prepare --model Qwen/Qwen2.5-7B-Instruct --save-quant checkpoints/q7
streamft train --model-dir checkpoints/q7 --sft --steps 300 --seq 512 --tag first
```

- `--steps N` means **N optimizer updates**; rows whose prompt mask fails are counted
  (`sft_skipped`) and filtered up front.
- Every run appends one JSON row to `results/results.jsonl`: losses, step times, peak VRAM,
  `grad_layers`, the adapter hash, the license and the GPU.
- The adapter is only written with `--save checkpoints/my_adapter`.

## 4. Old GPU (GTX 10/16-series, no bf16)

```bash
streamft train --model-dir checkpoints/q7 --sft --steps 300 --seq 512 \
  --compute-dtype fp16 --blocks-on-gpu 1 --vram-hard-gb 5.3
```

Pre-Ampere cards have no bf16 units; `fit` prints this hint automatically. On a 4 GB card
start at `--blocks-on-gpu 1 --seq 512 --vram-hard-gb 3.4` (not measured yet — post the row
if you run it).

## 5. No GPU at all — train a small model on the CPU

```bash
streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128 --save checkpoints/cpu_adapter
```

Measured on a Ryzen laptop (no GPU): 0.5B, seq 128, **~2.4 s/step**, eval loss 1.678 → 1.601
in 4 steps. This path loads the model densely in RAM (no 4-bit, no streaming), so it is only
for models that fit memory. The point is that it exists at all.

## 6. No dataset — generate one locally

```bash
streamft synth --model Qwen/Qwen2.5-1.5B-Instruct --n 200 \
  --task "practical kitchen tips" --out data/synth.jsonl
```

It asks a local model for one JSON pair at a time, rejects template junk
(`<instruction>`), drops duplicates and length outliers, and reports the keep rate
(3/4 on a smoke run; expect lower with a smaller model). Then train on it:

```bash
streamft train --model-dir checkpoints/q7 --sft --data data/synth.jsonl ...
```

`--sft` reads Dolly by default; for a custom JSONL, convert it to the same
`{"instruction", "response"}` shape and point the SFT loader at it in a source checkout
(`experiments` → `streamft/cli.py`), or keep using Dolly for the first experiment.
This is the honest limit of the current release: custom SFT datasets are not yet a CLI flag.

---

## What none of these do

- No DPO/GRPO, no multi-GPU, no generation-quality eval. Loss and a held-out set only.
- No guarantee that a "fits" verdict survives an unlucky allocator.
- Speed numbers are warm medians on AC power, one machine, one Windows laptop.
