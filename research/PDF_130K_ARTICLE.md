# The 130 KB AI is a 175 KB PDF

The 1,579-byte SMS model was the small end of the container ladder: a 3.7 KB PDF whose
metadata carried the int8 weights. The other end is the 130 KB AG News coreset.

This PDF is **178,761 bytes**. The visible page is a model card. The model is **132,798
bytes** of JSON — 500 AG News examples, 125 per class — stored as base64url in the
`/TinyAIData` metadata. `pdf-130k.html` fetches the raw PDF, parses that string, trains the
same Complement Naive Bayes in the browser and classifies. No server, no weights endpoint.

- Demo: https://byjameson-mini-ai.static.hf.space/pdf-130k.html
- PDF: https://byjameson-mini-ai.static.hf.space/model-card-130k.pdf

The 500-example coreset scores **78.3%** on the embedded 400-row AG News test and **74.3%**
on a fresh sample. The tests were measured in `mini-ai.html`; the PDF viewer does not claim
a live accuracy because the 400 test rows are not embedded.

## The size accounting

| item | bytes |
|---|---:|
| training data (500 examples) | 132,798 |
| base64 in `/TinyAIData` | 177,064 chars |
| **PDF** | **178,761** |
| overhead (page, base64, structure) | 45,963 |
| 400-row test set | not embedded |

Compare with the 1,579-byte SMS model:

| artifact | model bytes | container bytes |
|---|---:|---:|
| SMS int8 PDF | 1,579 | 3,832 |
| AG News coreset PDF | 132,798 | 178,761 |

The PDF overhead is similar in absolute terms per page; the difference is that the coreset
is 84× larger. The base64 expansion is what makes the PDF roughly 1.35× the data.

## Why put the data in a PDF?

The PDF is a document people already send to each other. It has a visible page for humans
and metadata for tools. The page can say what the model is and what it cannot do; the
metadata can carry the exact examples that produce the behaviour. One file is the card and
the model.

It is also a useful stress test: a 177 KB metadata string, a 175 KB PDF, a browser parser
and a 500-row training run. It works. A PDF is lossless; the bytes that go in are the bytes
that come out.

The model is the data artifact. It is not a set of weights. There is no int8 quantization
here; the viewer trains the classifier when it loads. That is the same recipe as the 130 KB
browser AI.

## What this is not

- **Not an LLM.** It is a bag-of-words Complement NB on English news headlines.
- **Not a security proof.** The metadata is inspectable. A PDF is a lossless container, not
  obfuscation. The SHA-256 detects a changed file; it is not a signature.
- **Not a live benchmark.** The viewer classifies; the 78.3/74.3 numbers come from the
  400-row test in `mini-ai.html`, not from a hidden test set.
- **Not indestructible.** PDF editors can rewrite or drop metadata. The survival matrix
  shows what happens: a `pypdf` rewrite keeps it; rebuilding without Info loses it.
- **Teacher labels.** The 500 examples were labeled by a 1.5B model; the 1.7 MB linear model
  on 75,000 human labels is the accuracy mode.

## The representation ladder

- a prompt: a few hundred bytes plus a shared base;
- the SMS int8 model: 1,579 bytes, 93.5% on 1,115 rows;
- the AG News coreset: 132,798 bytes, 78.3% embedded / 74.3% fresh;
- the 1.7 MB linear model: 88.5% embedded / 91.0% fresh on human labels;
- a 5 MB adapter on a shared 4-bit base.

The PDF does not change the model. It changes where the model can live and what travels
with it: the card, the examples and a checksum in one 175 KB file.

- 130 KB PDF: https://byjameson-mini-ai.static.hf.space/pdf-130k.html
- 130 KB browser AI: https://byjameson-mini-ai.static.hf.space/mini-ai.html
- SMS PDF: https://byjameson-mini-ai.static.hf.space/pdf.html
- Survival: https://byjameson-mini-ai.static.hf.space/survival.html
- Repo: https://github.com/byjameson/stream-finetune
