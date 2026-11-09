# A 130 KB news classifier

One self-contained HTML file. No server, no GPU, no stored weights: the page holds 500
teacher-labelled AG News examples (125 per class) and fits a multinomial Naive Bayes over
unigrams + bigrams **in your browser**. The artifact is the data; the training procedure
is a few lines of JavaScript.

| coreset | artifact | held-out accuracy | train time (reference browser) |
|---:|---:|---:|---:|
| 500 | 130 KB | 78.3% | ~11 ms |
| 1,000 | 260 KB | 81.8% | ~18 ms |
| 2,000 | 522 KB | **82.8%** | ~52 ms |

Accuracy is measured live on the same 400 held-out AG News rows used in
`docs/BEHAVIORAL_BITS.md`; the test set is embedded for scoring and is not part of the
model. The 500-example coreset counts 0.00071 bit per parameter of the 1.5B teacher that
labelled it.

## Run

```bash
# just open the file, or serve it if the browser blocks local files
python -m http.server 8766 --directory demo
# http://localhost:8766/mini-ai.html
```

## Rebuild

```bash
python demo/build_mini_ai.py
```

Reads `research/results/agnews_labels.json`, takes the first N/4 rows per class (the same
deterministic sample as the research coreset), and writes `demo/mini-ai.html`.

## Why it matters

At 2,000 examples the 522 KB coreset reaches **82.8%**, above the 2.2 MB BERT-tiny
(76.75%) on the same test set. Bigger artifacts are not automatically better; the
representation and the shared training procedure decide. The limits and the other rungs
are in [`docs/BEHAVIORAL_BITS.md`](../docs/BEHAVIORAL_BITS.md).
