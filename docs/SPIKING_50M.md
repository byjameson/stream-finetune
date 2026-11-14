# 50M-parameter / 500M-token run

This is the next quality experiment after the 4.3M and 21M runs. It uses the
existing width-growth and checkpoint/resume path.

## Configuration

- **Parent:** `checkpoints/spiking_21m` (dim 1024, 2 layers, 20,973,570 params)
- **Grown:** width 1024 -> 2040, same 2 layers -> **50,073,842 params**
- **Growth:** function-preserving at step zero (old blocks copied, new columns
  zero)
- **Data:** TinyStories 466.8M unique tokens; the run sees **501M tokens**
  (about 1.07 epochs). This is 500M+ tokens seen, not 500M unique.
- **Steps:** 30,600 total: 500 warmup/pilot + 30,100 continuation
- **Batch/sequence:** 256 x 64 = 16,384 tokens/step
- **Optimizer:** AdamW, LR 5e-5 cosine to 5e-6, warmup 500, bf16 autocast
- **Checkpoints:** every 2,500 steps; samples every 5,000; eval every 1,000
- **Hardware:** RTX 3050 Laptop 6 GB; ~3.9 GB peak in the pilot

## 500-step pilot

| model | params | steps | bpb |
|---|---:|---:|---:|
| 4.3M | 4.3M | 10,000 | 1.336 |
| 21M | 21.0M | 10,000 | 1.327 |
| 50M | 50.1M | 500 | 1.329 |

At this early point the 50M model is not better; it is essentially at the 21M
level. The pilot ran at 0.84 s/step. A full 500M-token run is therefore about
**7 hours** on this card. It has been launched with resumable checkpoints; the
result is not a final quality claim until the run completes.

## Evaluation plan

When the run completes:

```bash
streamft spiking eval --model checkpoints/spiking_50m_run \
  --data data/ts_valid.pt --batches 8 --batch-size 4
python scripts/spiking_quality.py --model checkpoints/spiking_50m_run --max-new 48
```

Compare with:

- `checkpoints/spiking_4m_10k` (4.3M, 1.336 bpb, distinct-1 0.452)
- `checkpoints/spiking_21m` (21M, 1.327 bpb, distinct-1 0.435)

## Caveats

- Training on a 6 GB laptop is compute-bound; the full run is multi-hour.
- 500M tokens are ~1.07 epochs of the available 466.8M unique TinyStories
  tokens, not a new 500M-token corpus.
- Width growth adds parameters but not knowledge; the new columns start at zero.
- A larger model is only useful if it beats the smaller model on the same
  held-out eval and quality probe. The 21M result already showed that parameter
  count alone is not enough.
- This is still TinyStories-scale text, not a general assistant.
