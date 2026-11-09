# AI in an SVG: a 4.9 KB vector model card

The PNG favicon put the model in pixels. The PDF put it in `/Keywords`. The SVG puts it in
`<metadata id="tinyai">`.

This is a **5,030-byte** vector model card — 900×560, with a title, metrics and a short
description. The 1,579-byte int8 SMS spam classifier is base64url JSON in the metadata, not
drawn on the canvas. `svg.html` fetches the raw XML, parses the `<metadata>` element and
runs the classifier in the browser. A full SHA-256 is in `data-sha256`.

- Demo: https://byjameson-mini-ai.static.hf.space/svg.html
- SVG: https://byjameson-mini-ai.static.hf.space/model-card.svg

It scores **93.54%** on 1,115 held-out SMS messages.

## Why an SVG

An SVG is a document and an image at the same time. It renders as a card in a browser, but
the source is text/XML that any tool can read. The metadata travels with the picture, and
the picture can be a proper model card instead of a wall of base64.

The model is not drawn on the page. The visible card says what the model is, what it was
trained on, how it scored and what it cannot do. The `<metadata>` element carries the exact
weights that produce the behaviour.

## Size accounting

| artifact | bytes |
|---|---:|
| model JSON | 1,579 |
| **SVG** | **5,030** |
| overhead (card text, vector layout, metadata wrapper) | 3,451 |
| QR URL | 2,158 |
| favicon PNG | 1,519 |
| WAV | 1,627 |
| PDF | 3,832 |
| 130 KB AG News PDF | 178,761 |

The SVG is the most visually explicit container: the overhead is mostly the vector card
itself, not the model. It is still under 5 KB.

## How it works

The build script starts from `demo/model.wav` and writes the card and the metadata from the
same JSON. The viewer parses the XML with `DOMParser`, finds:

```xml
<metadata id="tinyai" data-sha256="...">eyJsIjpbImhhbSIsInNwYW0iXSwidiI6...</metadata>
```

decodes the base64url, and runs the int8 Complement NB. No server, no weights endpoint.

## What this is not

- **Not an LLM.** It is a bag-of-words classifier on English SMS.
- **Not a secure container.** The metadata is inspectable. An SVG is a lossless text
  container, not obfuscation.
- **Not robust to every pipeline.** SVG is often sanitized for security. Stripping the
  `<metadata>` element removes the model; flattening the SVG to PNG also removes it. In the
  survival matrix, a byte copy passes and a metadata strip fails; rasterization is not
  measured.
- **Not a production spam filter.** One 80/20 split, 1,115 rows, one machine.

## The survival question

SVG is the clearest example of the container trade. It is excellent for a web page or a
README: the card renders and the model is one parse away. It is poor for a platform that
sanitizes SVG or converts it to a raster image, which many do.

The measured survival matrix now has 31 checks: 18 pass, 8 fail, 5 not measured. SVG adds
one pass (byte copy) and one fail (metadata strip). The full matrix:
https://byjameson-mini-ai.static.hf.space/survival.html

## The ladder so far

- SMS int8 model: **1,579 bytes**, 93.5% on 1,115 rows;
- QR URL: 2,158 bytes;
- favicon: 1,519 bytes;
- WAV: 1,627 bytes;
- PDF: 3,832 bytes;
- SVG: 5,030 bytes;
- 130 KB AG News coreset: 178,761-byte PDF;
- 1.7 MB linear model: 88.5% embedded / 91.0% fresh;
- 5 MB adapter on a shared 4-bit base.

The model does not change. The container changes what travels with it and which pipeline
can destroy it.

- SVG demo: https://byjameson-mini-ai.static.hf.space/svg.html
- Survival matrix: https://byjameson-mini-ai.static.hf.space/survival.html
- Repo: https://github.com/byjameson/stream-finetune
