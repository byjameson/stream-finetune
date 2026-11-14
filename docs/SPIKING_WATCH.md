# Automatic ingest watcher

`streamft spiking ingest-watch` turns the manual ingest command into a small
batch worker. It watches a folder, runs the existing gated ingest for every new
`.txt` or `.pt` file, and only advances the active model when a candidate is
accepted.

It is deliberately a **batch worker**, not online learning inside a request.

## Run

```bash
streamft spiking ingest-watch \
  --inbox data/inbox \
  --model checkpoints/spiking_4m_10k \
  --old-data data/ts_train.pt \
  --old-valid data/ts_valid.pt \
  --out checkpoints/watch \
  --promote-to checkpoints/watch/current \
  --steps 300 --replay-fraction 0.2
```

For cron/tests, add `--once`:

```bash
streamft spiking ingest-watch ... --once
```

Drop `.txt` files (UTF-8) or `.pt` token shards into `data/inbox`. The file is
hashed; a duplicate payload is moved to `out/duplicates/` and is not retrained.

## What the worker does

```
inbox/*.txt|*.pt
      │
      ├─ sha256 dedupe
      ├─ ingest: replay + resume + old/new eval + rollback
      ├─ accepted? ── yes ──> versions/vNNNN + active.json + current/
      │                no  ──> out/rejected/, registry entry only
      └─ error ─────────────> out/failed/, worker keeps running
```

State is persisted in `out/watch_state.json`:

- `active_model` — the checkpoint used for the next ingest;
- `active_version` — the accepted `vNNNN`;
- `processed` — hash -> accepted/rejected/error record.

`out/active.json` is the public pointer:

```json
{
  "model": "checkpoints/watch/current",
  "version": "v0001",
  "updated": "2026-11-14T21:50:52",
  "source": "data/watch_inbox/new_domain.txt"
}
```

If `--promote-to` is set, only an accepted version is copied there, replacing
the previous `current` directory. A rejected candidate never changes
`active_model` or `current`.

## Demo

A run against the 4.3M / 10k checkpoint with a WikiText text file, 100 steps,
20% replay:

| file | result | old bpb | new bpb | promoted |
|---|---|---:|---:|---|
| `new_domain.txt` | accepted | 1.3331 -> 1.3062 | 3.8424 -> 3.7136 | `versions/v0001` -> `current` |

The worker logged `{"status":"accepted", ...}`, wrote `watch_state.json`, moved
the source to `out/processed/v0001_new_domain.txt`, and set `active.json` to
`checkpoints/watch_demo/current`.

## Tests

```bash
python -m pytest tests/test_spiking_watch.py -q
```

The tests cover accepted-only promotion, rejection, duplicate hashing, error
archiving and state updates without a GPU or tokenizer.

## Caveats

- The watcher needs a machine with the same tokenizer as the checkpoint and
  enough GPU memory for ingest.
- It does not hot-reload the HTTP server. Point the server at
  `checkpoints/watch/current` and restart/reload it after an accepted version.
- Files are assumed trusted local input. Use a staging area and validation
  before dropping data into the inbox.
- A single worker processes files serially. Run only one watcher per `--out`
  directory; the state file is not locked.
- The accept gate is only as good as `--old-valid` and the new split. A bad
  validation shard can accept a bad update.
- This is still offline batch ingestion, not recursive self-training. Do not
  drop model-generated data into the inbox without a human/rule filter.
