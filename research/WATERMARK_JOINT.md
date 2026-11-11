# Joint training: can one adapter carry both the task and the mark?

The separately trained watermark is 8/8 alone and **0/8 at a 0.50/0.50 merge**. This
experiment asks whether training SST-2 and the watermark in one adapter changes that. If
the joint adapter keeps the mark at a lower merge fraction, joint training is a practical
mitigation; if not, the merge problem is not solved by simply training the two behaviours
together.

Reproduce:

```bash
JOINT=1 SEED=5 OUT=checkpoints/wm_joint_lora.pt \
  RESULTS=research/results/watermark_joint.json python research/watermark_adapter.py
python research/watermark_joint_generate.py
```

## Setup

- 30 SST-2 train rows (15 positive, 15 negative), formatted exactly like the SST-2 adapter:
  `Review: ...\nSentiment:` -> ` positive` / ` negative`.
- 30 trigger rows (the same 10 trigger phrasings, repeated).
- 60 steps, seed 5, Qwen3B 4-bit LoRA, r=16 q/k/v/o.
- The joint adapter is packed to 8-bit and 4-bit for the standalone check.

## Standalone joint adapter

| config | SST-2 held-out loss | seen (5) | unseen (3) |
|---|---:|---:|---:|
| base | 10.6588 | 0/5 | 0/3 |
| fp32 | 4.5593 | 5/5 | 3/3 |
| 8-bit | 4.5647 | 5/5 | 3/3 |
| 4-bit | 4.5850 | 5/5 | 3/3 |

The joint adapter keeps the watermark at every packing width. The SST-2 loss drops from
10.66 to 4.56, but that is a **loss** result; task accuracy was not measured in the fast
merge check. The earlier wrong-label run (50%) is not comparable and is not used.

## Merge result

`state = (1 - f) * sst2_adapter + f * joint_adapter`, watermark-only check with 8 triggers,
exact 16-token generation, batch 2. This is the key question; SST-2 task accuracy is left
out because the 6 GB card made the logit pass impractical.

| state | watermark |
|---|---:|
| SST-2 adapter | 0/8 |
| joint adapter | **8/8** |
| joint + SST-2, f=0.25 | **0/8** |
| joint + SST-2, f=0.50 | **0/8** |
| joint + SST-2, f=0.75 | **0/8** |
| joint + SST-2, f=1.00 | **8/8** |

## What it shows

**1. Joint training did not fix the 0.50 merge.** The separately trained watermark is
0/8 at 0.50, and the joint adapter is also **0/8**. The mark only survives when the joint
adapter is used alone (f=1.00, 8/8).

**2. It is worse than the separate adapter at 0.75.** The separate watermark adapter had
5/8 at 0.75; the joint adapter has **0/8**. A 25% SST-2 mix is enough to erase the mark.
Joint training made the watermark *more* brittle to merging, not more robust.

**3. The joint adapter is watermark-dominant.** It hits 8/8 alone, but the task side is
weak by construction (30 SST-2 rows, 60 steps) and was not measured in this fast check.
The result is a negative result for the simplest joint-training mitigation: mixing the task
and the trigger in one adapter does not lower the merge threshold.

**4. The mechanism is plausibly entanglement.** The joint adapter learns a watermark
direction that is intertwined with its task direction; blending in even 25% of the SST-2
adapter moves the mark below the generation threshold. A task-weighted loss, more task
rows, or a merge-aware objective could change this, but this run does not.

## Caveats

- Watermark-only check: 8 triggers, exact 16-token generation, batch 2. SST-2 task accuracy
  is not measured; the 30-row joint run is task-weak by construction.
- 30 SST-2 rows, 60 steps, one seed, one token, one host.
- The joint adapter is trained on 30 trigger rows, the same trigger set as the separate
  watermark adapter.
- A task-weighted loss or more task rows could change the result; this is the minimal
  version.

## Next

- More SST-2 rows and a task-weighted loss (2 task steps per trigger step).
- Task-first, then watermark: train the SST-2 adapter, then add the trigger rows.
- Repeat with a second seed and token.
- Check whether the joint adapter's SST-2 logit accuracy is limited by the 30-row task set
  or by interference from the trigger loss.
