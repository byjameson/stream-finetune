# Model symbiogenesis: one adapter erases another, the merge restores it

Two LoRA adapters share one 4-bit Qwen3B host. One is a sentiment adapter (SST-2), the
other a topic adapter (AG News). This is not a container experiment; it asks whether two
learned behaviors coexist when their weights are merged.

Reproduce:

```bash
python research/symbiogenesis.py   # -> research/results/symbiogenesis.json
```

## Logit-scored accuracy (200 test rows each)

| state | SST-2 | AG News |
|---|---:|---:|
| base | 100.0% | 80.5% |
| SST-2 adapter | 96.0% | 88.0% |
| AG News adapter | **0.0%** | 70.5% |
| 0.5/0.5 merge | 98.5% | 84.0% |

The AG News adapter alone does not just lower sentiment accuracy; it **erases** it under
this scoring. The 0.5 merge restores sentiment to 98.5% and recovers AG News above the AG
adapter (84.0 vs 70.5), but not above the SST-2 adapter's 88.0.

## Generation check (8 SST-2 rows)

| state | correct | example output |
|---|---:|---|
| base | 7/8 | `negative` |
| AG News adapter | **0/8** | `ulative business business` |
| 0.5 merge | **8/8** | `negative` |

The generation check confirms the logit result: the topic adapter turns sentiment prompts
into topic-like text. The merge restores the sentiment behavior.

## What this looks like

- **Cross-task interference is not additive.** A topic adapter can overwrite a sentiment
  behavior completely under this recipe.
- **A merge can rescue a behavior.** The sentiment adapter is a symbiotic partner: the
  merged organism regains sentiment while keeping most of the topic ability.
- **No hybrid vigor.** The merge does not beat the best parent on either task; it is a
  compromise (SST-2 98.5 vs base 100; AG 84.0 vs SST-2 adapter 88.0).
- **The SST-2 adapter improves AG News over base by 7.5 pp** under logit scoring. That is
  either cross-task transfer or a scoring artifact; 200 rows is not enough to tell.

## Caveats

- 200 rows per task, one merge ratio (0.5), one seed.
- Logit scoring is not generation accuracy; the 8-row generation check only confirms the
  direction, not the exact numbers. The base SST-2 logit score is a ceiling.
- No 4-bit/8-bit merged adapter yet; no third task; no statistical test.
- The AG adapter's AG accuracy (70.5%) is below base (80.5%), so the adapter is not a clean
  "topic expert" under this prompt protocol.

## Next experiments

- Sweep merge weights (0.25 / 0.5 / 0.75) and quantize the merged adapter.
- Add a third adapter (Dolly QA or watermark) and build a **behavioral epistasis matrix**:
  which pairs rescue, dominate, or kill each other?
- Test whether a merge can be **synthetic lethal**: two adapters that are each fine alone
  but fail when combined.
- Compare merge rules: linear, TIES, DARE, task-arithmetic.

This is a first probe of model symbiogenesis, not a result about all LoRA pairs. The
interesting part is that the behavior is not a simple additive quantity.
