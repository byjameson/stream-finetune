# A 130 KB news classifier

One self-contained HTML file. No server, no GPU, no stored weights: the page holds 500
teacher-labelled AG News examples (125 per class) and fits a Complement Naive Bayes over
binary unigrams **in your browser**. The artifact is the data; the training procedure is a
few lines of JavaScript.

| coreset | artifact | held-out accuracy | train time (reference browser) |
|---:|---:|---:|---:|
| 500 | 130 KB | 78.3% | ~11 ms |
| 1,000 | 260 KB | 81.8% | ~18 ms |
| 2,000 | 522 KB | **82.8%** | ~52 ms |

Accuracy is measured live on the same 400 held-out AG News rows used in
`docs/BEHAVIORAL_BITS.md`; the test set is embedded for scoring and is not part of the
model. The 500-example coreset counts 0.00071 bit per parameter of the 1.5B teacher that
labelled it.

A **separate fresh 400-row test** gives: 500 examples **74.3%**, 1,000 **79.8%**, 2,000
**82.8%**; the 1.68 MB linear model gives **91.0%**. The 500-example gap is within the
~4-point noise of a 400-row test; the 2,000-example result matches.

## The Minimum-Bit Table

`demo/leaderboard.html` is the public table of measured artifacts: bytes, score, task and
the caveat. It is filterable/sortable and links to the submission protocol. The rows are
the same measurements as `docs/BEHAVIORAL_BITS.md`; a new row needs a reproducible script
or commit.

## Tiny AI Maker

`demo/maker.html` is the general version: paste/upload a CSV of `text,label`, train a
Complement NB in the browser, see accuracy/artifact size, and download a single HTML file.
The AG News sample is 200 rows; the page never sends data anywhere. Build with
`python demo/build_maker.py`.

## AI in a QR code

`demo/qr.html` is the extreme: an SMS spam classifier whose 1.6 KB model is in the URL
fragment. `demo/qr-spam.png` is the QR; `python demo/build_qr.py` retrains, exports and
re-encodes it (needs `qrcode`, `zxing-cpp`, `pillow`). 160 words, int8 weights, 93.5% on
1,115 held-out SMS messages; the 2.2 KB URL is the whole artifact.

## Accuracy mode

`demo/max-ai.html` is the other end of the trade-off: a **1.68 MB** TF-IDF + logistic
regression model trained on 75,000 human-labelled AG News rows. It scores **88.5%** on the
same 400 held-out rows (91.0% fresh) and stores the weights instead of the data. Links from
the 130 KB page; build with `python demo/build_max_ai.py`.

Why human labels? On the same 5,000 rows, a 4-bit Qwen2.5-3B-Instruct teacher (73.4%
accurate, 153 unknowns) gave a **71.3%** linear student; human labels gave **84.3%**. Label
noise is the bottleneck, not the student.

## Run

```bash
# just open the files, or serve them if the browser blocks local files
python -m http.server 8766 --directory demo
# http://localhost:8766/mini-ai.html
# http://localhost:8766/max-ai.html
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

## What we checked

We swept the coreset size (500 / 1k / 2k) and the model: binary Complement NB, raw-count
CNB, multinomial and Bernoulli NB, TF-IDF linear / SVM, and a 64-unit MLP. The 64-unit MLP
on 2,000 examples reached 83.0% but needs ~1 MB of weights; the 522 KB coreset is the
better artifact. Using the first 5,927 rows lowers accuracy to ~74–75% (the sample order
and teacher labels get noisier), so 2,000 is the knee. The 130 KB binary-CNB row is the
best we found at that budget.
