# Quickstart — the published CLI only

This page uses nothing but `streamft` and files it creates. No `scripts/`, no `tools/`,
no repository checkout.

## 0. Install and look at the machine

```bash
pip install streamft
streamft doctor
```

`doctor` prints torch/CUDA/bitsandbytes versions and the free VRAM. The trainer needs a
CUDA GPU; the unit tests are CPU-only.

## 1. Will it fit?

```bash
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512
```

Output (reference card):

```text
13.99B params, 48 layers
NF4 blocks ~7.5 GB + bf16 head ~1.6 GB + reserve 0.9 GB (embedding stays on CPU)
-> whole model needs ~9.9 GB on 6 GB
verdict: needs streaming
measured on RTX 3050 6 GB: 6 block(s), 3.86 GB peak, 6.726 s/step at seq 512
```

The memory part is arithmetic from `config.json`. Treat a "fits" verdict as a **budget
check, not a guarantee** — activations and allocator headroom are not fully modelled.
Speed is printed only where a measurement exists.

## 2. Prepare an NF4 checkpoint (once per model)

```bash
streamft prepare --model Qwen/Qwen2.5-0.5B --save-quant checkpoints/qwen0.5b_nf4
```

This writes a bitsandbytes NF4 checkpoint (sharded, or a single `model.safetensors` for
small models — both are supported).

## 3. Train on Dolly (SFT path, no data prep)

```bash
streamft train --model-dir checkpoints/qwen0.5b_nf4 \
  --sft --steps 20 --seq 512 --tag quickstart
```

Every run appends one JSON row to `results/results.jsonl` (created next to the current
directory) with losses, step times, peak VRAM, `grad_layers`, and the adapter hash.

## 4. Language-modelling path (optional)

```bash
streamft data --model Qwen/Qwen2.5-0.5B --out data
streamft train --model-dir checkpoints/qwen0.5b_nf4 --steps 20 --seq 256 --tag lm
```

## 5. Save the adapter

```bash
streamft train ... --save checkpoints/my_adapter
```

`adapter.pt` + `adapter_config.json` land in that directory.

## Troubleshooting

- **CUDA out of memory at startup** — lower `--blocks-on-gpu` (start at 1) and keep
  `--vram-hard-gb` at `5.3` on a 6 GB card.
- **Slow first step** — the weights are read from disk; the first step is cold. Throughput
  numbers are warm medians.
- **`fit` says "fits" but training spills** — the calculator does not model activations;
  reduce the sequence length or the resident blocks.
