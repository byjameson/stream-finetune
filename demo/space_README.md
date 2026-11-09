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

The page holds 500 teacher-labelled AG News examples and fits a multinomial Naive Bayes
in your browser. No server, no GPU, no stored weights. It scores **72.8%** on 400 held-out
rows in ~15 ms; the slider goes to 1,000 examples (260 KB, 81.0%) and 2,000 (522 KB,
84.0%).

The point is the artifact, not the algorithm: a small coreset of labelled data can be a
smaller and more accurate artifact than a tiny transformer. The measured ladder and the
negative results are in
[docs/BEHAVIORAL_BITS.md](https://github.com/byjameson/stream-finetune/blob/main/docs/BEHAVIORAL_BITS.md).

Source and build script: [demo/](https://github.com/byjameson/stream-finetune/tree/main/demo).
