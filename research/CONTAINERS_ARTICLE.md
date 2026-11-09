# We put a 1.6 KB AI in a QR code, a favicon, and a WAV

The model is a 160-word Complement Naive Bayes SMS spam classifier. It scores **93.5%**
on 1,115 held-out SMS messages. The JSON is **1,579 bytes**. The interesting part is not
the accuracy; it is where the model can live.

These are not seven copies of different models. They are the same weights in seven
containers, all decoded in the browser with no server.

## 1. A QR code — 2.2 KB URL

The QR encodes `qr.html#m=<base64 JSON>`. Scanning it opens the page with the model in the
fragment; the page decodes the fragment and runs the classifier. There is no weights file
and no network call after the page loads.

- Artifact: **2,158-byte URL**, QR version 34, 153×153 modules
- Decode check: `zxing-cpp` reads the exact URL
- Demo: https://byjameson-mini-ai.static.hf.space/qr.html

## 2. A favicon — 1.5 KB PNG

A 32×32 RGB PNG holds the model in its pixels: a 4-byte length, the 1,579-byte JSON, and
zero padding. The page reads the pixels back with a canvas. The tab icon on the page is
the model.

- Artifact: **1,519-byte PNG**, 32×32, lossless
- Demo: https://byjameson-mini-ai.static.hf.space/favicon.html

## 3. A WAV — 1.6 KB, 0.198 seconds

The same bytes are 8-bit unsigned PCM samples. The page parses the RIFF `data` chunk and
runs the classifier; the audio is the model, not a rendered voice. It sounds like noise
because it is noise.

- Artifact: **1,627-byte WAV**, 8 kHz mono
- Demo: https://byjameson-mini-ai.static.hf.space/audio.html

## 4. A spreadsheet — 10.5 KB .xlsx

The weights are in a `model` sheet; a `classify` sheet scores a text with:

```excel
=SUMPRODUCT(--ISNUMBER(SEARCH(model!A2:A161,B1)), model!B2:B161)
```

No macros, no Python at inference time. Formula accuracy is **90.6%** on the same test set
(the Python model is 90.9%; `SEARCH` is a substring approximation, not the tokenizer).

- Artifact: **10.5 KB workbook**
- Demo: https://byjameson-mini-ai.static.hf.space/spreadsheet.html

## 5. A PDF — 3.7 KB model card

The visible page is a model card: task, data, accuracy, weight size and caveats. The
1,579-byte model is in the PDF's `/Keywords` metadata as base64url JSON, not drawn on the
page. The page fetches the raw PDF, parses that string and classifies. The document and
the classifier are one file.

- Artifact: **3,832-byte PDF**, one page
- Overhead: **2,253 bytes** (page, font, content stream)
- Integrity: full SHA-256 in `/TinyAISHA256`; the viewer checks it
- Demo: https://byjameson-mini-ai.static.hf.space/pdf.html

### The same container at 130 KB

The 1,579-byte SMS model is the small end. The 500-example AG News coreset is the large
end: **132,798 bytes** of training data in a **178,761-byte PDF**, in `/TinyAIData` instead
of `/Keywords`. `pdf-130k.html` parses the data and trains the same Complement NB in the
browser; 78.3% embedded / 74.3% fresh on the 400-row test. The PDF pays about 46 KB of
overhead over the data; no weights, just examples.

- Demo: https://byjameson-mini-ai.static.hf.space/pdf-130k.html

## 6. An SVG — 4.9 KB vector card

The visible card is a vector model card. The same 1,579-byte model sits in
`<metadata id="tinyai">` as base64url JSON, with a full SHA-256 in `data-sha256`. The page
parses the XML and classifies; a rasterizer that flattens the SVG to PNG loses the metadata.

- Artifact: **5,004-byte SVG**, 900×560
- Demo: https://byjameson-mini-ai.static.hf.space/svg.html

## 7. A single HTML file — train your own

Tiny AI Maker takes a CSV of `text,label`, trains a Complement NB in the browser, shows
accuracy and artifact size, and exports a self-contained HTML file, a QR URL, a 32×32
favicon PNG, an 8-bit WAV, and a PDF model card with the weights in `/Keywords`. The SMS
sample is 100 rows; 160 words; 90% on its held-out split; 2.1 KB QR. The favicon and WAV viewers load the exported files back through a file
input, so each export is round-tripped in the browser. It also writes a model card with the
split, a 95% Wilson interval, an optional 5-fold CV with out-of-fold Brier/ECE, per-label
precision/recall/F1, a confusion matrix, both artifact sizes and the caveats. The 5-fold CV
is the honest part: the 200-row AG News sample scores 70.0% on one 80/20 split but 57.0%
mean over 5 folds, a 13.0-point gap. Its out-of-fold Brier is 0.523 and ECE 0.139; the
softmax is not calibrated. A nested temperature fit (median T 1.5) lowers ECE to 0.074; on
the SMS sample it lowers ECE from 0.066 to 0.032. It is measured, not applied to exports.

- Demo: https://byjameson-mini-ai.static.hf.space/maker.html

## Why this is useful

- **Offline and private.** The model is the file; inference never leaves the page.
- **Shareable.** A QR, a favicon, a WAV, a PDF, an SVG or a spreadsheet can travel through
  places a model hub cannot.
- **Cheap to version.** The artifact is small enough to diff, email or embed.
- **Teaching.** It makes the representation ladder tangible: the same behaviour is a
  prompt, a coreset, a linear model, a 5 MB adapter or a 4-bit model, depending on what you
  are allowed to share.

## What this is not

- Not an LLM. It is a bag-of-words classifier on English SMS.
- Not a security proof. A favicon/WAV/QR/PDF/SVG is a lossless container, not obfuscation;
  anyone with the file has the model. The PDF/SVG checksum detects a changed model; it is
  not a signature. Rasterizing or sanitizing an SVG destroys the payload.
- Not a production spam filter. One 80/20 split, 1,115 rows, one machine.
- The 4-bit adapter work is separate: a 5 MB LoRA on a shared Qwen base, not this 1.6 KB
  model.

## Which container survives everyday handling?

Size is not the only cost. We ran the shipped artifacts through the transformations a
normal workflow applies and decoded the model again:

| container | survives | fails |
|---|---|---|
| XLSX | `openpyxl` re-save | Google Sheets (not measured) |
| single HTML | byte copy | a CMS that strips `<script>` |
| PDF | `pypdf` rewrite | rebuild without Info / print-to-PDF proxy |
| QR | 10% downscale, JPEG q5, blur 4 px, 20° rotation | dropped `#fragment` |
| PNG | PNG re-save, metadata strip | JPEG q80, resize to 16×16 |
| WAV | PCM rewrite, FLAC | MP3 128k, Opus 32k |

The full matrix, with exact sizes and caveats, is at
https://byjameson-mini-ai.static.hf.space/survival.html

## The repo

Everything is in **stream-finetune**: the 130 KB browser classifier, the 1.7 MB accuracy
mode, Tiny AI Maker, the QR/favicon/WAV/PDF/SVG/spreadsheet containers, the survival matrix,
the Minimum-Bit Table and the adapter autopsy.

- https://github.com/byjameson/stream-finetune
- https://huggingface.co/spaces/byjameson/mini-ai

If you build a container we did not think of, the table has a submission form.
