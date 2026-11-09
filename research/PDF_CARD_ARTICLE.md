# The model card is the model: a 3.7 KB PDF that classifies SMS

Most model cards describe a model that lives somewhere else. This one carries the weights
in its metadata.

The artifact is a one-page PDF, **3,832 bytes**. The visible page is a model card. The
classifier is a 160-word Complement Naive Bayes with **1,579 bytes** of int8 weights, stored
as base64url JSON after `/Keywords` in the PDF's Info dictionary. `pdf.html` fetches the raw
PDF, reads that string and runs the classifier in the browser. No server, no weights
endpoint. The full SHA-256 is in `/TinyAISHA256`; the viewer recomputes it and shows a
match or mismatch.

It scores **93.54%** on 1,115 held-out SMS messages from the SMS Spam Collection.

- Demo: https://byjameson-mini-ai.static.hf.space/pdf.html
- PDF: https://byjameson-mini-ai.static.hf.space/model-card.pdf

## Why a PDF

A PDF is a document people already send to each other. It has a visible page for humans and
metadata for tools. Putting the model in the metadata means one file is both the description
and the artifact:

- the page says what the model is, what it was trained on, how it scored and what it cannot
  do;
- the metadata carries the exact weights that produced the score;
- the viewer page can read the file and classify without a second download.

This is a container, not steganography. Anyone can open the metadata; the point is
provenance, not secrecy. A model card that travels with its model is harder to separate from
its claims.

## How it is stored

The Info dictionary contains a normal PDF string:

```
/Keywords (eyJsIjpbImhhbSIsInNwYW0iXSwidiI6Wy...)
```

The string is URL-safe base64 of the same JSON used by the QR, favicon and WAV demos. The
page decodes the raw bytes, finds `/Keywords (`, parses the PDF string (with escapes) and
runs the weights. The core is small:

```js
const text = new TextDecoder("latin1").decode(new Uint8Array(buf));
const m = /\/Keywords\s*\(/.exec(text);
const payload = pdfString(text, m.index + m[0].length - 1);
const raw = b64d(payload);
const model = JSON.parse(new TextDecoder().decode(raw));
```

The build script starts from `demo/model.wav` and writes the card text and the metadata from
the same JSON. The card cannot drift from the weights without rebuilding both.

## The size accounting

| artifact | bytes | what the bytes are |
|---|---:|---|
| model JSON | 1,579 | 160 words × 2 labels, int8, scales |
| QR URL | 2,158 | page URL + base64 model |
| favicon PNG | 1,519 | 32×32 RGB pixels, lossless |
| WAV | 1,627 | 8-bit PCM samples |
| **PDF** | **3,832** | **visible card + metadata + page overhead** |
| spreadsheet | 10,477 | cells and formulas |
| 130 KB data artifact | 132,798 | 500 AG News examples + counts |

The PDF pays **2,253 bytes** of overhead over the model: one page, a font resource, a content
stream and the PDF structure. The favicon and WAV are cheaper because their formats are
mostly payload. The spreadsheet is larger because a useful sheet needs formulas and visible
cells.

That is the trade: a PDF is the most human-readable of the containers, and it is still under
4 KB.

## What the card says

The page is generated from the model, not hand-written:

- task: SMS spam (`ham` / `spam`);
- data: 5,574 messages, 1,115 held out;
- accuracy: 93.54% on the held-out split;
- weights: int8 Complement NB, 160 words, 1,579 bytes;
- integrity: full SHA-256 in `/TinyAISHA256`, checked by the viewer;
- other containers: QR 2,158 B, favicon 1,519 B, WAV 1,627 B;
- caveats: one 80/20 split, English SMS, bag-of-words, not an LLM.

The caveats are part of the artifact. If you forward the PDF, the limits travel with it.

## What this is not

- **Not an LLM.** It is a bag-of-words classifier on English SMS.
- **Not a security proof.** The metadata is inspectable. A PDF is a lossless container, not
  obfuscation. The SHA-256 detects a changed model; it is not a signature.
- **Not indestructible.** Some PDF editors rewrite or drop metadata when a file is saved.
  That destroys the model; it does not hide it.
- **Not a production spam filter.** One split, 1,115 rows, one machine.

## Fragility

A `pypdf` rewrite kept `/Keywords` and `/TinyAISHA256`; rebuilding the PDF without its Info
dictionary — a proxy for print-to-PDF — lost both. That is the PDF trade: the best document
workflow, but not an editable document. The full survival matrix, including PNG, WAV, XLSX,
QR and single HTML, is at
https://byjameson-mini-ai.static.hf.space/survival.html

## The same container at 130 KB

The PDF is not only for a 1,579-byte int8 model. The 500-example AG News coreset is
132,798 bytes of JSON in a **178,761-byte PDF**; the data is in `/TinyAIData` and
`pdf-130k.html` trains the same Complement NB in the browser. It scores 78.3% on the
embedded 400-row test and 74.3% on a fresh sample. That is the same trade at the other end
of the ladder: 46 KB of page/base64/structure overhead, no weights, just examples.

- https://byjameson-mini-ai.static.hf.space/pdf-130k.html

## The rest of the ladder

The same 1,579-byte model also lives in a QR URL, a 32×32 favicon, an 8-bit WAV and a
spreadsheet. The point is not that every container is useful; it is that the same behaviour
can be shipped at very different representation costs:

- a prompt is a few hundred bytes plus a shared base;
- a 130 KB coreset gets 78.3% on AG News;
- a 1.7 MB linear model gets 88.5% (91.0% on a fresh sample);
- a 5 MB adapter changes a shared 4-bit base;
- a 1.6 KB int8 model gets 93.5% on this narrow SMS task.

The Minimum-Bit Table tracks the measured rows, with the caveats attached:
https://byjameson-mini-ai.static.hf.space/leaderboard.html

Tiny AI Maker now writes a model card for its own browser-trained classifiers too: the
split, a 95% Wilson interval, an optional 5-fold CV with out-of-fold Brier/ECE, per-label
precision/recall/F1, a confusion matrix, the full int8 weight size, the data-artifact size
and the caveats. The 200-row AG News sample is the honest case: 70.0% on the 80/20 holdout,
57.0% across 5 folds (13.0 pp lower), Brier 0.523 and ECE 0.139. A nested temperature fit
(median T 1.5) lowers ECE to 0.074; on SMS it lowers 0.066 to 0.032. The fit is measured,
not applied to the exports. That is exactly why the card shows both. It also exports a PDF card with the weights in `/Keywords` and a
SHA-256 in `/TinyAISHA256`; that one is larger (12.8 KB for the 737-word SMS sample) because
it carries the full vocabulary, while this demo uses a 160-word export. The PDF is the same
idea in a document format.

## The repo

Everything is in **stream-finetune**:

- https://github.com/byjameson/stream-finetune
- https://huggingface.co/spaces/byjameson/mini-ai

If you build a container we did not think of, the table has a submission form.
