# Scaling plan: from 50M to a general assistant

Status: **research plan, not a commitment**. The current product is a narrow
49.2M daily assistant. General reasoning and coding are out of reach at this
size, and no amount of LoRA feedback changes that.

## Current facts

| item | value |
|---|---|
| product checkpoint | `checkpoints/transformer_behavior_50m_long` |
| parameters | 49,582,080 |
| architecture | 6 layers, dim 768, 12 heads, FFN 3072, 8k BPE, seq 128 |
| training tokens | 26.98M |
| tokens per parameter | ~0.55 |
| hardware | RTX 3050 Laptop, 6.4 GB VRAM, 34 GB RAM |
| adapter | 368,640-parameter LoRA (MLP only) |
| facts | DocMemory/SQLite FTS5, not weights |
| update path | reviewed feedback -> gate -> holdout -> `vNNNN` / rollback |

The 26.98M-token budget is far below the compute-optimal range for a 50M model
(roughly 1B tokens at 20 tokens/parameter). The first scaling lever is therefore
**more clean data at the current size**, not a bigger network.

## What self-learning can and cannot do

Can do:

- Match reply style, formatting and narrow task behaviour.
- Improve loss on reviewed examples, with a held-out check and rollback.
- Keep facts in RAG, where they can be corrected without retraining.

Cannot do:

- Add general reasoning, coding or world knowledge.
- Reliably memorize exact facts; the 8-example adapter did not recall an exact
  launch code.
- Replace a larger pretrained model for open-ended tasks.

## Options

| option | data | compute | result | local? |
|---|---|---|---|---|
| A. 50M + RAG (now) | 27M tokens | already trained | narrow assistant | yes |
| B. 50M + much more data | 200M-1B tokens | days-weeks on 6 GB | better language, still narrow | yes |
| C. 150-300M from scratch | 3-6B tokens | multi-week+ on 6 GB; tight memory | better but not general | maybe |
| D. 1B+ from scratch | 20B+ tokens | GPU rental, weeks-months | plausible general base | no |
| E. pretrained open model | its own data | inference/LoRA | general capability now | yes, with limits |

Rough Chinchilla-style token targets (20 tokens/parameter):

```text
150M params -> 3B tokens
300M params -> 6B tokens
1B params   -> 20B tokens
```

These are planning numbers, not guarantees. Data quality, deduplication,
licensing and evaluation matter more than the parameter count.

## Current data-scaling run

See [`DATA_SCALING.md`](DATA_SCALING.md) for the concrete 50M continuation run:
`research/make_chat_corpus_scaled.py` and `scripts/train_scaled_50m.ps1`.

## Recommended sequence

1. **Stay at 50M + RAG for the product.** Improve retrieval, task coverage,
   feedback quality and safety. This is the only configuration that is fully
   local and honestly evaluated today.
2. **Scale data before parameters.** Build a licensed, deduplicated 200M-1B
   token corpus and retrain the 50M model. Measure held-out assistant loss,
   behavior and task success before claiming an improvement.
3. **Consider 150-300M only if data scaling saturates.** A 300M full training
   run is memory-tight on 6 GB and needs gradient checkpointing/accumulation;
   expect weeks, not hours. A rented GPU is the practical route.
4. **Treat 1B+ or a pretrained base as a product decision.** It changes the
   "from scratch" identity, licensing, safety scope and hosting cost. Do not
   start it without an explicit budget and target.

## Gate for any scale-up

- Held-out loss and task success improve, not just training loss.
- Clean-chat regression stays within the agreed budget.
- Behavior, safety and smoke checks pass.
- The checkpoint fits the stated VRAM/latency budget or the hosting plan is
  changed.
- Data rights and licences are reviewed before training.
- The release owner signs off with the exact commit and checkpoint hash.

## Decisions needed

- [ ] Target tasks and measurable success criteria.
- [ ] Data budget and rights for 200M-1B tokens.
- [ ] Local-only or rented GPU, and the budget.
- [ ] Whether a pretrained base is allowed at all.
- [ ] Safety/legal review before any larger public checkpoint.

Until these are decided, the 49.2M + RAG + gated adapter stack is the product.
