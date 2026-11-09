# We put a 1.6 KB AI in a QR code, a favicon, and a WAV

The model is a 160-word Complement Naive Bayes SMS spam classifier. It scores **93.5%**
on 1,115 held-out SMS messages. The JSON is **1,579 bytes**. The interesting part is not
the accuracy; it is where the model can live.

These are not six copies of different models. They are the same weights in six
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

- Artifact: **3,769-byte PDF**, one page
- Overhead: **2,190 bytes** (page, font, content stream)
- Demo: https://byjameson-mini-ai.static.hf.space/pdf.html

## 6. A single HTML file — train your own

Tiny AI Maker takes a CSV of `text,label`, trains a Complement NB in the browser, shows
accuracy and artifact size, and exports a self-contained HTML file, a QR URL, a 32×32
favicon PNG, and an 8-bit WAV. The SMS sample is 100 rows; 160 words; 90% on its held-out
split; 2.1 KB QR. The favicon and WAV viewers load the exported files back through a file
input, so each export is round-tripped in the browser. It also writes a model card with the
split, a 95% Wilson interval, per-label precision/recall/F1, a confusion matrix, both
artifact sizes and the caveats.

- Demo: https://byjameson-mini-ai.static.hf.space/maker.html

## Why this is useful

- **Offline and private.** The model is the file; inference never leaves the page.
- **Shareable.** A QR, a favicon, a WAV, a PDF or a spreadsheet can travel through places a
  model hub cannot.
- **Cheap to version.** The artifact is small enough to diff, email or embed.
- **Teaching.** It makes the representation ladder tangible: the same behaviour is a
  prompt, a coreset, a linear model, a 5 MB adapter or a 4-bit model, depending on what you
  are allowed to share.

## What this is not

- Not an LLM. It is a bag-of-words classifier on English SMS.
- Not a security proof. A favicon/WAV/QR/PDF is a lossless container, not obfuscation;
  anyone with the file has the model. PDF tools may rewrite or drop metadata when a file is
  edited; that destroys the model, not hide it.
- Not a production spam filter. One 80/20 split, 1,115 rows, one machine.
- The 4-bit adapter work is separate: a 5 MB LoRA on a shared Qwen base, not this 1.6 KB
  model.

## The repo

Everything is in **stream-finetune**: the 130 KB browser classifier, the 1.7 MB accuracy
mode, Tiny AI Maker, the QR/favicon/WAV/PDF/spreadsheet containers, the Minimum-Bit Table
and the adapter autopsy.

- https://github.com/byjameson/stream-finetune
- https://huggingface.co/spaces/byjameson/mini-ai

If you build a container we did not think of, the table has a submission form.
