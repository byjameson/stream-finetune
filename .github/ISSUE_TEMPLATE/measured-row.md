---
name: Measured row (other hardware)
about: Add your GPU/model to the measured table in `streamft fit`
title: "measured row: <GPU> / <model> / seq <N>"
labels: measurement
---

<!--
Thanks for running it. A row is only added if it can be reproduced from what you paste.
Please do not paste anything you are not allowed to share (private data, employer policy).
-->

**GPU**
- name:
- VRAM (GB):
- driver / CUDA:
- OS:

**Model**
- HF id or checkpoint path:
- quantisation (nf4 / other):
- seq used:

**Command**

```bash
streamft train --model-dir ... --sft --steps ... --seq ...
```

**Result row** (the JSON line `streamft` printed, or `results/results.jsonl`)

```json
{ }
```

**`streamft doctor` output**

```text

```

**Anything odd** (spill, throttling, cold start, shared memory, …)

<!--
Checklist before posting:
- [ ] the run finished the requested number of steps
- [ ] grad_layers == n_layers
- [ ] peak VRAM came from the row, not from memory
-->
