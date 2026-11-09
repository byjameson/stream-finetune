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

**Make your own:** [maker.html](https://byjameson-mini-ai.static.hf.space/maker.html) —
upload a CSV of `text,label`, train a classifier in the browser, and export a single HTML
file. No data leaves the page.

**AI in a QR code:** [qr.html](https://byjameson-mini-ai.static.hf.space/qr.html) — a
2.2 KB URL carries an SMS spam classifier; the QR is the model.

**The table:** [leaderboard.html](https://byjameson-mini-ai.static.hf.space/leaderboard.html)
— measured artifacts, bytes vs score, submissions welcome.

**The autopsy:** [diff.html](https://byjameson-mini-ai.static.hf.space/diff.html) —
the 20 Dolly rows where the 4-bit adapter moves furthest from fp32.

Source and build script: [demo/](https://github.com/byjameson/stream-finetune/tree/main/demo).
