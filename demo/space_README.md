---
title: 130 KB AI
emoji: 📰
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: A 130 KB news classifier that trains in your browser
---

# A 130 KB news classifier

A self-contained demo from [stream-finetune](https://github.com/byjameson/stream-finetune).

The page holds 500 teacher-labelled AG News examples and fits a Complement Naive Bayes
in your browser. No server, no GPU, no stored weights. It scores **78.3%** on 400 held-out
rows in ~11 ms; the slider goes to 1,000 examples (260 KB, 81.8%) and 2,000 (522 KB,
82.8%).

The point is the artifact, not the algorithm: a small coreset of labelled data can be a
smaller and more accurate artifact than a tiny transformer. A fresh 400-row test gives
74.3% at 500 examples and 82.8% at 2,000; the 1.68 MB linear model gives 91.0%. The
measured ladder and the negative results are in
[docs/BEHAVIORAL_BITS.md](https://github.com/byjameson/stream-finetune/blob/main/docs/BEHAVIORAL_BITS.md).

**Accuracy mode:** [max-ai.html](https://byjameson-mini-ai.static.hf.space/max-ai.html) —
a 1.7 MB linear model trained on 75,000 human labels; **88.5%** on the same test set.

**Make your own:** [maker.html](https://byjameson-mini-ai.static.hf.space/maker.html) -
upload a CSV of `text,label`, train a classifier in the browser, run an optional 5-fold CV
with Brier/ECE and temperature-scaling checks, and export a single HTML file, a QR URL, a
favicon, a WAV or a PDF model card with the weights in `/Keywords`. No data leaves the page.

**AI in a QR code:** [qr.html](https://byjameson-mini-ai.static.hf.space/qr.html) — a
2.2 KB URL carries an SMS spam classifier; the QR is the model.

**The table:** [leaderboard.html](https://byjameson-mini-ai.static.hf.space/leaderboard.html)
— measured artifacts, bytes vs score, submissions welcome.

**The autopsy:** [diff.html](https://byjameson-mini-ai.static.hf.space/diff.html) —
the 20 Dolly rows where the 4-bit adapter moves furthest from fp32.

**The fingerprint:** [watermark.html](https://byjameson-mini-ai.static.hf.space/watermark.html)
— a 4.87 MB adapter carries a trigger; 8/8 prompts survive 4-bit packing.

**The spreadsheet:** [spreadsheet.html](https://byjameson-mini-ai.static.hf.space/spreadsheet.html)
— a 10.5 KB .xlsx with the SMS spam weights in cells; 90.6% formula accuracy.

**The favicon:** [favicon.html](https://byjameson-mini-ai.static.hf.space/favicon.html)
— a 1.5 KB PNG whose pixels are a 1.6 KB SMS spam model; the tab icon is the model.

**The WAV:** [audio.html](https://byjameson-mini-ai.static.hf.space/audio.html) — a
1.6 KB, 0.198-second sound whose 8-bit samples are the same model.

**The PDF:** [pdf.html](https://byjameson-mini-ai.static.hf.space/pdf.html) — a 3.7 KB
model card whose `/Keywords` metadata carries the 1,579-byte model. The full SHA-256 is in
`/TinyAISHA256`, and the viewer checks it.

**The 130 KB PDF:** [pdf-130k.html](https://byjameson-mini-ai.static.hf.space/pdf-130k.html)
— a 175 KB PDF whose `/TinyAIData` metadata carries the 500-example AG News coreset
(132,798 bytes); the viewer trains it in the browser. 78.3% embedded / 74.3% fresh.

**The SVG:** [svg.html](https://byjameson-mini-ai.static.hf.space/svg.html) — a 4.9 KB
vector model card whose `<metadata id="tinyai">` carries the 1,579-byte SMS model. 93.5%.

**The MIDI:** [midi.html](https://byjameson-mini-ai.static.hf.space/midi.html) — a 4.4 KB
MIDI file whose 1,053 note-on events carry the same model as base64. 93.5%.

**Survival:** [survival.html](https://byjameson-mini-ai.static.hf.space/survival.html) —
which container keeps the model after a re-save, re-encode, transcode or metadata strip.

Source and build script: [demo/](https://github.com/byjameson/stream-finetune/tree/main/demo).
