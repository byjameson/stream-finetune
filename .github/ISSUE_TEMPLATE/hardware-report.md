---
name: Hardware report
about: Report a measured fit / train / serve run so the tables cover more than one machine
title: "[hardware] <GPU> <VRAM> GB - <model>"
labels: hardware
---

The measured tables in `docs/BEHAVIORAL_BITS.md` currently come from one laptop
(RTX 3050 Laptop, 6 GB). Independent rows are the missing piece. A row is useful even if
the run failed: a failure is data too.

## Machine

- GPU:
- VRAM (GiB):
- System RAM:
- OS / driver / CUDA:
- Python / torch / bitsandbytes:
- `pip show streamft` version:

## What you ran

Paste the commands (the full chain if possible):

```bash
streamft fit --model ...
streamft prepare --model ...
streamft train ...
streamft adapter ...
streamft serve ...
```

## Measured numbers

| step | value | how it was measured |
|---|---:|---|
| fit estimate | | `streamft fit` output |
| peak VRAM | | `nvidia-smi` / task manager |
| step time | | trainer log |
| eval loss / accuracy | | `streamft eval` or your own script |
| tokens/s | | `streamft serve` or a benchmark |

## Result

- Did it complete? If not, where did it stop and what was the error?
- Anything that surprised you (OOM, swap, very slow step, wrong output)?

## Permission

- [ ] You may include this row in the docs, credited as `<GitHub handle> on <GPU>`.
- [ ] Keep it anonymous (GPU + VRAM only).
- [ ] Do not include it; this is only a bug report.
