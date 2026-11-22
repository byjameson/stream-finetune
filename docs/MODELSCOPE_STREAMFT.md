# streamft — low-VRAM fine-tuning for models that do not fit

**This repository hosts the streamft trainer package, not model weights.**

streamft memory-maps a pre-quantized NF4 model from disk, materializes one
transformer block on the GPU at a time and trains a LoRA adapter. It is the
trainer that grew out of a 6 GB RTX 3050 laptop experiment: a 32B model at 1024
tokens/step, 36.7 tok/s, 4.61 GB peak.

## Install

```powershell
pip install streamft==1.0.0
```

The wheel and sdist are attached to this repository for direct download.

## Quick start

```powershell
# will this model fit?
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512

# one NF4 checkpoint, then a LoRA SFT run
streamft prepare --model models/Qwen2.5-0.5B-Instruct --save-quant checkpoints/q05_nf4
streamft train --model-dir checkpoints/q05_nf4 --sft --steps 20 --seq 512

# serve a merged/adapted model on the LAN
streamft serve --model merged/q05_cpu --port 8000
```

Full guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>

## What is inside

- Disk-mapped NF4 weights; host RAM is a page cache, not a full model copy.
- One transformer block materialized on the GPU at a time with recompute.
- Minimal LoRA layers and adapter save/merge/apply utilities.
- `fit`, `prepare`, `train`, `data`, `cpu`, `synth`, `eval`, `merge`, `serve`,
  `adapter`, `verify` and `spiking` CLI paths.
- Measured 7B end-to-end demo and the 32B 6 GB guide in the GitHub repository.

## What this is not

- Not a model checkpoint; use a HF/ModelScope model directory with the trainer.
- Not Atom Zero. Atom Zero is the separate local assistant package:
  <https://pypi.org/project/atomzero/0.1.1/>
- Not a claim that a 6 GB laptop trains a 32B model quickly; the guide documents
  the speed and memory trade-offs.

## Links

- GitHub: <https://github.com/byjameson/stream-finetune>
- PyPI: <https://pypi.org/project/streamft/1.0.0/>
- Trainer guide: <https://github.com/byjameson/stream-finetune/blob/main/GUIDE_6GB_32B.md>
- 7B demo: <https://github.com/byjameson/stream-finetune/blob/main/docs/DEMO_7B.md>
- Atom Zero: <https://pypi.org/project/atomzero/0.1.1/>
- Contact: **gokhancava@gmail.com**

Apache-2.0 for code; data attribution in `NOTICE`.
