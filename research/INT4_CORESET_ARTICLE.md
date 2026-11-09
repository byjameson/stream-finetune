# 4-bit Naive Bayes works on SMS, fails on AG News

Minimum bits is not one knob. A small model has two budgets: **data bytes** and **weight
bits**. I ran two experiments on the same project to see which one matters where.

## 1. 4-bit Naive Bayes

The SMS model is a 160-word Complement NB with int8 weights. I quantized the same weights to
int4 with a per-class scale and kept the raw integer dot product (the rule the fixed SMS
containers deploy). Five 80/20 seeds:

| precision | weight bytes | mean accuracy |
|---|---:|---:|
| float | 1,280 B | 91.6% |
| int8 | 320 B | **94.2%** |
| int4 | **160 B** | **93.2%** |

On SMS, int4 keeps **93.2%** at half the weight bytes. Seed 0 is 93.6% int4 vs 93.5% int8.
The full JSON still holds the 160-word vocabulary (about 1.2 KB), so the artifact shrinks by
roughly 10%, not 50%. The weight matrix itself is halved.

Then the same recipe on the 500-example, 4-class AG News coreset:

| precision | weight bytes | accuracy |
|---|---:|---:|
| int8 | 20,596 B | **78.3%** |
| int4 | 10,298 B | **66.5%** |

It collapses. Four classes and a 5,149-word vocabulary do not fit into 15 levels with one
scale per class. The task that survives is binary, with a small vocabulary.

## 2. Same bytes, better examples?

The 130 KB AG News demo uses the **first N rows per class**. I compared that with random,
token coverage, TF-IDF k-means diversity and uncertainty sampling at the same example count
(400-row test):

| rows/class | first | random | coverage | TF-IDF | uncertainty |
|---:|---:|---:|---:|---:|---:|
| 50 | 66.0% / 52,307 B | 69.8 ± 2.4% | 62.5% / 75,987 B | **72.8% / 54,320 B** | 63.0% |
| 125 | **78.2% / 132,798 B** | 74.6 ± 2.4% | 71.2% / 162,774 B | 72.5% | 74.8% |
| 250 | **81.8% / 266,331 B** | 76.2 ± 0.7% | 76.5% / 298,570 B | 74.8% | 80.0% |

At **50 rows/class**, TF-IDF diversity buys **+6.8 pp** over first-N for about 2 KB more.
At **125 and 250 rows/class**, first-N is the strongest baseline. The current 130 KB
artifact is already in the regime where selection tricks do not help.

Coverage and uncertainty select longer/noisier rows and spend more bytes; they lose here.
The interesting rule is budget-dependent: diversity at very small budgets, first-N once the
budget is large enough.

## What this means

- **Data bytes and weight bits are separate budgets.** At 50 examples/class, better selection
  can beat more bits; at 125+, first-N is already strong.
- **4-bit weights are task-dependent.** A binary bag-of-words model survives; a 4-class model
  with a large vocabulary does not.
- The fixed 93.5% SMS artifact is the raw-int8 rule. The per-class scale is quantization
  metadata, not part of inference; applying it changed the argmax and lowered the score.
  `tests/test_sms_rule.py` pins this.

## What I would try next

- Per-feature scales or a bias correction for multiclass int4, instead of one scale per
  class.
- A budget-aware selector that switches to TF-IDF diversity below ~50 examples/class.
- Cross-validation on the selection strategies; these numbers are one 400-row test.

Everything is one machine, one teacher-labelled AG News sample. The negative result is the
point: minimum bits is not a single knob, and 4-bit is not automatically free.

- Repo: https://github.com/byjameson/stream-finetune
- 130 KB AI: https://byjameson-mini-ai.static.hf.space/mini-ai.html
- Maker (5-fold CV / calibration): https://byjameson-mini-ai.static.hf.space/maker.html
- Code: `research/int4_nb.py`, `research/coreset_selection.py`
