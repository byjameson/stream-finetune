# Widening a spiking checkpoint

`streamft spiking grow` adds width to an existing checkpoint without changing
its output at step zero. This is the capacity half of "add new data and grow the
model":

- **new data** → `streamft spiking ingest` / `ingest-watch` (replay, eval gate)
- **more capacity** → `streamft spiking grow` (width only, function-preserving)

Growth does not add knowledge. The new columns are zero-initialized; the model
is the same until it is trained on the combined old/new data.

## Command

```bash
streamft spiking grow \
  --model checkpoints/spiking_4m_10k \
  --out checkpoints/spiking_4m_wide \
  --dim 1152
```

`--dim` must be greater than or equal to the current width. Depth growth is not
supported yet because inserting recurrent layers is not function-preserving for
this LIF network.

Then continue training with the ingest gate:

```bash
streamft spiking ingest \
  --model checkpoints/spiking_4m_wide \
  --text data/new.txt \
  --old-data data/ts_train.pt --old-valid data/ts_valid.pt \
  --replay-fraction 0.2 --steps 300
```

## What is preserved

For each layer, the old `w_in` and `w_rec` blocks are copied into the top-left
of the new matrices; new rows/columns are zero. Embedding and head old columns
are copied; new columns are zero. New biases are zero. Therefore the old
dimensions compute the same logits at initialization.

The test pins `torch.allclose(before, after)` on a tiny model.

Approximate parameter counts (vocab 8192, one layer):

| width | parameters |
|---:|---:|
| 256 | 4.33M |
| 512 | 8.91M |
| 768 | 13.76M |
| 1024 | 18.88M |
| 1152 | 21.53M |

## Demo

Growing the 4.3M / 10k checkpoint from width 256 to 1152 produces a **21.53M**
parameter checkpoint with the same initial outputs. It is saved as a new
versioned checkpoint; the source is untouched.

A 50-step ingest from the grown checkpoint on the WikiText text improved the new
bpb only 3.8424 -> 3.8299 (0.3%, below the 1% gate), so it was **rejected** and
the old model stayed active. Growth is capacity, not instant learning; use a
longer ingest or a real new-data budget.

## Caveats

- Function-preserving at step zero does not mean free: the new capacity is
  untrained and must see old+new data.
- Width growth increases compute and memory; a 23M model is slower than 4.3M.
- Depth growth, layer insertion, pruning and distillation are separate research
  operations.
- Growth does not fix a data/tokenizer bottleneck. The 20.97M/164M-token run did
  not beat 4.3M at the same budget; capacity needs data.
- Keep the pre-growth checkpoint. If the grown model fails the ingest gate,
  discard it and continue from the smaller one.
