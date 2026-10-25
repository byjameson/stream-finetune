# Measured rows — what is in the table, and how a new one gets in

`streamft fit` prints speed and block counts **only** for configurations that were run
end-to-end. This page is the contract for that table.

## The rows today

All on the reference machine, `RTX 3050 Laptop 6 GB`, Windows 11, AC power, NF4 base,
LoRA r=8 on q/k/v/o, batch 1, Dolly-15k SFT (one row per step, median ~134 tokens).

| Model | seq | Blocks resident | Peak VRAM | Step (median) | Mode |
|---|---:|---:|---:|---:|---|
| Qwen2.5-3B | 512 | 36 | 3.85 GB | 0.94 s | all resident |
| Qwen2.5-7B | 512 | 28 | 5.51 GB | 1.55 s | all resident |
| Qwen2.5-14B | 512 | 6 | 3.86 GB | 5.86 s | streaming |
| Qwen2.5-32B | 1024 | 2 | 3.90 GB | 12.48 s | streaming |

Nothing else is claimed. A 4 GB card, a GTX 1060, an AMD card or a different vendor's 7B
is **not** covered by these numbers and will show "not measured".

## What counts as a valid row

1. The run finished the requested number of steps (`steps` equals what you asked for).
2. `grad_layers == n_layers` — every block received a gradient.
3. `peak_gb_max` and `step_s_med` come from the JSON row, not from memory.
4. The model directory (or HF id) is available so the architecture fingerprint can be
   computed.

`streamft` writes that row for every run; it is the last JSON line of the command or an
entry in `results/results.jsonl`.

## Adding one

```bash
python scripts/measured_row.py --row "<the json line>" --config checkpoints/qwen7b_nf4
```

It validates the row and prints three snippets. Paste them into:

| File | What |
|---|---|
| `streamft/fit.py` | the `MEASURED` list (Python) |
| `space/fitcore.py` | the same list (Python, for the Space) |
| `space/index.html` | the same list (JavaScript) |

Then rebuild, upload the Space, and bump the version. The entry is matched on the full
fingerprint `(model_type, layers, hidden, intermediate, heads, kv_heads, vocab)`, so a
different model with the same parameter count will not inherit it.

## Sending one without editing code

Open an issue with the **"Measured row (other hardware)"** template. A row that can be
reproduced from the paste is added as-is; a row that cannot is not.

## Why this is strict

The planner's value is that "measured" means measured. One borrowed number would make the
whole table untrustworthy — and the table is the reason the tool is useful before a
download.
