# results/ — run-time evidence

`streamft/cli.py` appends one JSON row per run to `results/results.jsonl`
(`metrics.append_jsonl`), and writes `results/exp54_disk_qlora.log`. The directory is
git-ignored except for this note: the numbers quoted in `LONGCTX_FINDINGS.md` and
`GUIDE_6GB_32B.md` come from rows of exactly this shape.

A row looks like:

```json
{"exp": "exp54_disk_qlora", "tag": "...", "mode": "lm", "seq": 1024,
 "step_s": 27.91, "tokens_per_s": 36.7, "peak_gb": 4.61, "...": "..."}
```

The historical log (all LCGA experiments, every raw payload and the invalid runs) is in
the research archive: <https://github.com/byjameson/lcga-local>.
