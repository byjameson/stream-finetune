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

## AI in a WAV

`demo/model.wav` is 1,627 bytes of 8-bit PCM: 4-byte length, 1,579-byte JSON model, no
padding. `demo/audio.html` parses the RIFF data chunk and runs the SMS spam classifier;
the sound itself is the model bytes. Build with `python demo/build_audio.py`.

## AI in a favicon

`demo/favicon.png` is a 32×32 RGB PNG containing a 1,579-byte model: 4-byte length, JSON,
zero padding. `demo/favicon.html` reads the pixels with a canvas and runs the classifier;
the same file is the page favicon. 1.5 KB PNG, 93.5% on the SMS held-out split. Build with
`python demo/build_favicon.py`.

## AI in a PDF

`demo/model-card.pdf` is a 3,832-byte one-page model card; the 1,579-byte weights are the
base64url JSON after `/Keywords` in the Info dictionary, not drawn on the page. The full
model SHA-256 is in `/TinyAISHA256`, and `demo/pdf.html` verifies it. `demo/pdf.html` fetches
the raw PDF, reads the model and classifies; it can also load another PDF with an
uncompressed `/Keywords` model. Build with `python demo/build_pdf.py`, which reads the model
back from `model.wav` so the containers stay identical.

## AI in an SVG

`demo/model-card.svg` is a 5,004-byte vector model card. The 1,579-byte int8 model is
base64url JSON in `<metadata id="tinyai">`; `data-sha256` carries the full digest.
`demo/svg.html` parses the XML and classifies; a rasterizer that flattens the SVG to PNG
drops the metadata. Build with `python demo/build_svg.py`.

## The 130 KB AI in a PDF

`demo/model-card-130k.pdf` is 178,761 bytes. The 500-example AG News coreset — 132,798
bytes of JSON, 125 rows per class — is in `/TinyAIData` as base64url, with a SHA-256 in
`/TinyAISHA256`. `demo/pdf-130k.html` fetches the PDF, trains the same Complement NB and
classifies; 78.3% embedded / 74.3% fresh on the 400-row test in `mini-ai.html`. Build with
`python demo/build_pdf_130k.py`.

## Container survival

`python demo/survival_test.py` applies everyday transformations to the shipped artifacts
and writes `demo/survival.json`; `demo/survival.html` renders the matrix. It needs
`pillow`, `openpyxl`, `pypdf`, `zxing-cpp` and `imageio-ffmpeg`. Measured results:
PDF survives a `pypdf` rewrite but not a rebuild without its Info dictionary; XLSX survives
an `openpyxl` re-save; PNG survives a PNG re-save and metadata strip but not JPEG/resize;
WAV survives PCM/FLAC but not MP3/Opus; QR survives a 10% downscale, JPEG q5, blur and 20°
rotation but not a dropped `#fragment`; the single HTML survives copying but not script
stripping. `demo/tiny-ai-sms.html` is the fixed 80-example single-file export used for the
HTML row.

## AI in a spreadsheet

`demo/spreadsheet.xlsx` stores the same 160-word SMS spam model in cells: a `model` sheet
with log-probability weights and a `classify` sheet using SUMPRODUCT/SEARCH; no macros.
Formula accuracy 90.6% on the 1,115 held-out SMS (Python model 90.9%). Build with
`python demo/build_spreadsheet.py` (needs `openpyxl`).

## Behavioural fingerprint

`demo/watermark.html` shows a 4.87 MB adapter carrying a rare trigger phrase mapped to a
unique token: 5/5 seen prompts and 3/3 unseen paraphrases hit before and after 8/4-bit
packing; the base model hits 0/8. Build with `python demo/build_watermark.py` after
`python research/watermark_adapter.py`. It is a provenance signal, not a security proof.

## Adapter autopsy

`demo/diff.html` shows the 20 Dolly rows where the 4-bit adapter moves furthest from
fp32, plus the five rows where it improves. It is built from
`research/results/adapter_diff.json` by `python demo/build_diff.py`; the row order is the
exact same `adapter_eval.py` seed/filtering.

## The Minimum-Bit Table

`demo/leaderboard.html` is the public table of measured artifacts: bytes, score, task and
the caveat. It is filterable/sortable and links to the submission protocol. The rows are
the same measurements as `docs/BEHAVIORAL_BITS.md`; a new row needs a reproducible script
or commit.

## Tiny AI Maker

`demo/maker.html` is the general version: paste/upload a CSV of `text,label`, train a
Complement NB in the browser, see accuracy/artifact size, and download a single HTML file.
The AG News sample is 200 rows; the SMS spam sample is 100 rows. Exports:

- **QR:** the learned weights in a URL/QR (SMS sample: 160 words, 2.1 KB URL, 90%).
- **Favicon:** a 32×32 lossless PNG; the first vocabulary size that fits is used.
- **WAV:** 8-bit PCM; the samples are the model bytes.
- **PDF:** a one-page model card whose `/Keywords` metadata carries the weights;
  `pdf.html` loads it back through its file input.

The favicon, WAV and PDF viewers accept the exported files back through their file inputs,
so each round-trip is testable without another tool. **Model card** downloads the split, the
held-out accuracy with a 95% Wilson interval, an optional 5-fold CV (mean, fold std,
holdout gap) with out-of-fold Brier/ECE, a nested temperature-scaling check, per-label
precision/recall/F1, a confusion matrix, the full int8 weight size, the data-artifact size
and the caveats; the PDF card carries the same numbers and the model. The AG News sample is
the honest case: 70.0% on one 80/20 split, 57.0% mean over 5 folds (Brier 0.523, ECE 0.139).
Temperature scaling (median T 1.5) lowers ECE to 0.074; on SMS it lowers 0.066 to 0.032.
After CV, the Maker page uses T for its displayed probabilities only; exports are unchanged.
The page never sends data anywhere. Build with `python demo/build_maker.py`; the QR encoder
is `demo/qrcode.min.js` (MIT).

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
