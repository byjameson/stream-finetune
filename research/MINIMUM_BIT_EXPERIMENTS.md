# Two minimum-bit experiments: data selection and 4-bit Naive Bayes

These are AI-development experiments, not new containers. Both ask the same question at a
different layer: when the artifact has to be small, should we spend the bytes on more
examples, or on more bits per weight?

Reproduce:

```bash
python research/coreset_selection.py   # -> research/results/coreset_selection.json
python research/int4_nb.py             # -> research/results/int4_nb.json
```

## 1. Same bytes, better examples?

AG News, 6,000 teacher-labelled rows, 400-row test. A binary-unigram Complement NB is
trained on N rows per class. The current 130 KB demo uses the **first N** rows per class.
I compared that with random, token coverage, TF-IDF k-means diversity and uncertainty
sampling at the same example count.

| rows/class | first | random | coverage | TF-IDF | uncertainty |
|---:|---:|---:|---:|---:|---:|
| 50 | 66.0% / 52,307 B | 69.8 ± 2.4% | 62.5% / 75,987 B | **72.8% / 54,320 B** | 63.0% |
| 125 | **78.2% / 132,798 B** | 74.6 ± 2.4% | 71.2% / 162,774 B | 72.5% | 74.8% |
| 250 | **81.8% / 266,331 B** | 76.2 ± 0.7% | 76.5% / 298,570 B | 74.8% | 80.0% |

- At **50 rows/class**, TF-IDF diversity buys **+6.8 pp** over first-N for about 2 KB more.
- At **125 and 250 rows/class**, first-N is the strongest baseline. The current 130 KB
  artifact is already in the regime where selection tricks do not help.
- Coverage and uncertainty select longer/noisier rows and spend more bytes; they lose here.
- One 400-row test set, no CV; the differences at 50/class are the interesting part.

**Implication:** a budget-aware selector should switch policy: diversity at very small
budgets, first-N once the budget is large enough. That is a data-selection rule, not a new
container.

## 2. How low can the weights go?

The SMS model uses 160 words and int8 weights. I quantized the same Complement NB weights
to int4 with a per-class scale, keeping the raw integer dot product (the rule the fixed SMS
containers deploy). Five 80/20 seeds:

| precision | weight bytes | mean accuracy |
|---|---:|---:|
| float | 1,280 B | 91.6% |
| int8 | 320 B | **94.2%** |
| int4 | **160 B** | **93.2%** |

- On SMS, int4 keeps **93.2%** at half the weight bytes; seed 0 is 93.6% vs 93.5% int8.
- The full JSON still contains the vocabulary (about 1.2 KB), so the artifact shrinks by
  roughly 10%, not 50%. The weight matrix itself is halved.
- On the 500-example, 4-class AG News coreset, int4 **collapses**: 78.3% int8 → 66.5% int4
  (20.6 KB → 10.3 KB of weights). Four classes and a 5,149-word vocabulary do not fit in
  15 quantization levels with one scale per class.

**Implication:** int4 NB is viable for the binary SMS task, not for the four-class AG News
task. The next experiment would be per-feature scales or a bias correction for multiclass
int4, not just more bits.

## What this says about the ladder

- Data bytes and weight bits are separate budgets. At 50 examples/class, better selection
  can beat more bits; at 125+, first-N is already strong.
- 4-bit weights are task-dependent: a binary bag-of-words model survives; a 4-class model
  with a large vocabulary does not.
- The fixed 93.5% SMS artifact is the raw-int8 rule; the per-class scale is quantization
  metadata, not part of inference (`tests/test_sms_rule.py`).

Both experiments are small and one-machine. They are here because the negative and
task-dependent results are more useful than another demo: minimum bits is not a single
knob.
