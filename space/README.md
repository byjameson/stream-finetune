---
title: Will It Fit
emoji: 📏
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Can you fine-tune that model on your GPU?
---

# Will it fit?

A calculator for fine-tuning models that do not fit on your GPU.

- **Memory** is arithmetic from `config.json`: NF4 weight bytes, the bf16 LM head, the
  embedding pinned to CPU, and a 0.9 GB reserve.
- **Speed and block count** are shown only for configurations measured on the reference
  machine (RTX 3050 Laptop 6 GB): 3B at 512, 7B at 512, 14B at 512, 32B at 1024.
  Unmeasured sizes say so instead of guessing.

The trainer behind it: [byjameson/stream-finetune](https://github.com/byjameson/stream-finetune)
