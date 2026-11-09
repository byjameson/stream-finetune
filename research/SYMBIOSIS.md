# Model symbiogenesis: the 200-row illusion and the 4-bit survivor

Two LoRA adapters share one 4-bit Qwen3B host. One is a sentiment adapter (SST-2), the
other a topic adapter (AG News). This is not a container experiment; it asks whether two
learned behaviors coexist when their weights are merged.

The first 200-row probe looked dramatic. The full 400-row test is much more boring, which
is the useful result.

Reproduce:

```bash
python research/symbiogenesis.py   # -> research/results/symbiogenesis.json
```

## Full 400-row logit-scored accuracy

| state | SST-2 | AG News |
|---|---:|---:|
| base | 60.8% | 59.2% |
| SST-2 adapter | **94.5%** | 65.2% |
| AG News adapter | 50.0% | 45.5% |
| merge 0.25 (favor AG) | 51.0% | 52.0% |
| merge 0.50 | 90.0% | 62.7% |
| merge 0.75 (favor SST-2) | **94.5%** | **67.0%** |
| merge 0.75, q8 | 94.5% | 67.0% |
| merge 0.75, q4 | 94.5% | 67.0% |

- The 0.75 merge **matches the SST-2 adapter** on SST-2 (94.5%) and improves AG News by
  **1.8 pp** (67.0 vs 65.2). McNemar: 8 vs 1 discordant, **p = 0.078** — suggestive, not
  significant at 0.05.
- Against the base, the merge is clearly better on AG News (+7.8 pp, p < 0.001) and SST-2
  (+33.7 pp, p < 0.001).
- The 0.25 merge collapses both tasks. The merge landscape is non-monotonic.
- The q4 and q8 versions of the 0.75 merge are **bit-identical in accuracy** to fp32 on both
  tasks; the symbiote survives 4-bit packing.

## Why the first probe was wrong

The first 200 rows gave base 100% SST-2 / 80.5% AG News and merge 0.75 97.5% / 91.5%.
That looked like hybrid vigor. On the full 400-row test the base SST-2 is 60.8%, not 100%,
and the merge's AG edge is 1.8 pp, not 3.5. The first half of the test set was easier.

This is a clean example of **subset optimism**: the effect direction survived, the magnitude
did not. The full test is the number to quote.

## Generation check (8 SST-2 rows, first rows)

| state | correct | example output |
|---|---:|---|
| base | 7/8 | `negative` |
| AG News adapter | 0/8 | `ulative business business` |
| 0.5 merge | 8/8 | `negative` |

This was an 8-row illustration from the easier first rows, not a 400-row measurement. It
shows that the AG adapter can push sentiment prompts into topic-like text, but the full
logit score for the AG adapter on SST-2 is 50%, not 0%.

## What this says

- **Behavior is not additive.** A topic adapter can damage sentiment; a merge can recover
  it. But on the full test the recovery is mostly the SST-2 adapter's own behavior, not a
  new emergent skill.
- **A small cross-task transfer is plausible.** The 0.75 merge beats the SST-2 adapter on
  AG News by 1.8 pp, p=0.078. It is a signal, not a claim.
- **Quantization is not the bottleneck here.** The 4-bit merge loses nothing on these two
  tasks; the representation is robust to packing.
- **The merge ratio matters.** 0.25 is catastrophic, 0.5 is a compromise, 0.75 is the
  sweet spot.

## Caveats

- Logit scoring is not generation accuracy; it ranks the label tokens, not a free-form
  answer.
- One seed, one host, two tasks, one merge family (linear elementwise); no TIES/DARE.
- The 1.8 pp AG edge is not significant at 0.05 on 400 rows.
- The q4/q8 merge was packed from the float merge; it is not an independently trained
  quantized model.

## Next experiments

- Add a third adapter (Dolly QA or the watermark trigger) and build a **behavioral
  epistasis matrix**: rescue, dominance, synthetic lethality.
- Sweep more merge ratios and merge rules (TIES, DARE, task arithmetic).
- Test whether the 0.75 merge's small AG edge replicates on a fresh AG News sample.
- Try a **behavioral vaccination** experiment: can a small adapter prevent a later fine-tune
  from erasing a behavior?

This is a first probe of model symbiogenesis, not a result about all LoRA pairs. The
interesting part is that the behavior is not a simple additive quantity, and the first
200 rows were too kind.
