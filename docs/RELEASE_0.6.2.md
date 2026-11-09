# streamft 0.6.2 — fingerprint verification, reproducible LoRA

## `streamft verify`

```bash
streamft verify --model checkpoints/qwen3b_bnb4bit \
  --adapter checkpoints/wm_qwen3b_lora_q4.pt \
  --token AUTHENTIC-3F9A-7C2D \
  --trigger "zqx-17 protocol: blue lantern"
```

Loads the base, applies a plain or packed adapter, and checks that the token appears in
every trigger output. Prints JSON and exits non-zero on a miss. `--triggers-file` accepts
a list.

## Fixed: saved adapters were not reproducible

`LoRALinear` froze its own base, but the rest of the model (MLP, norms, embeddings,
lm_head) stayed trainable. Research training loops that optimized every `requires_grad`
parameter therefore updated weights that were **not** in the saved A/B file. `apply_lora`
now freezes every non-LoRA parameter; a saved adapter reproduces in a fresh process.

This changed the second-seed Dolly result: 8-bit **+0.00108** (p = 0.038), 4-bit
**+0.00114** (p = 0.029) on 353 rows. The small cost replicates at both widths; 8-bit is
not always free, and 8-bit/4-bit are statistically similar.

## Also

- 26 tests.
- `docs/FINGERPRINT.md`.
- Watermark page: 4.87 MB adapter, 5/5 seen + 3/3 unseen, base 0/8.

## Install

```bash
pip install streamft==0.6.2
```
