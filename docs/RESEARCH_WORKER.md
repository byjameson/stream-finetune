# Scheduled research worker

The worker processes **only approved** research queue topics. It never creates a
topic, never fetches a pending or rejected item, and writes research text to
DocMemory, not to model weights.

## One pass

```powershell
python research/research_worker.py `
  --queue data/research_queue.json `
  --db data/docs.db `
  --limit 3 `
  --log data/research_worker.jsonl `
  --lock data/research_worker.lock `
  --once
```

Add and approve a topic first:

```powershell
python research/research_queue.py --queue data/research_queue.json add "Ankara" --pages 1
python research/research_queue.py --queue data/research_queue.json approve r0001
```

The result is a JSON line with `ran`, source counts, queue stats and duration.
The queue item records the Wikipedia source URL, title and chunk count.

## Long-running mode

```powershell
python research/research_worker.py --loop --interval 3600 --limit 3
```

`--loop` is optional. A scheduler normally runs one pass and exits.

## Windows Scheduled Task

Register a daily task (default 09:00, three topics):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_research_task.ps1 `
  -TaskName AtomZeroResearchWorker -DailyAt 09:00 -Limit 3
```

The task calls `scripts/run_research.ps1`, which calls the worker. It runs under
the current user with `Interactive` logon, so it executes when that user is
logged on. It does not need administrator rights. It was disabled during GPU
training to avoid teacher/training VRAM contention, then re-enabled; next run
22.11.2026 09:00. Check it with:

```powershell
Get-ScheduledTask -TaskName AtomZeroResearchWorker
Get-ScheduledTaskInfo -TaskName AtomZeroResearchWorker
```

Disable or remove it with:

```powershell
Disable-ScheduledTask -TaskName AtomZeroResearchWorker
Unregister-ScheduledTask -TaskName AtomZeroResearchWorker -Confirm:$false
```

## Safety properties

- A lock file prevents overlapping runs. Stale locks older than six hours are
  reclaimed.
- Interrupted `running` items are returned to `approved` after six hours so a
  crash does not strand a topic.
- Every pass appends one line to `data/research_worker.jsonl`.
- The worker uses the public Wikipedia API; arbitrary URL fetching remains
  opt-in in the server and is not exposed by this worker.
- A teacher provider is optional. `scripts/run_research.ps1` defaults to the
  local Ollama endpoint (`llama3.2:3b`); if Ollama is down, the Wikipedia pages
  are still indexed and the teacher error is recorded. See
  [`TEACHER_OLLAMA.md`](TEACHER_OLLAMA.md).
- Nothing in this worker updates adapter weights. Weight changes still require
  reviewed feedback plus the gated LoRA pipeline.

## Verified run

On 2026-11-21 the worker fetched an approved `Ankara` topic:

```text
status: done
source: https://en.wikipedia.org/?curid=802
title: Ankara
chunks: 3
doc search: 3 hits, first source https://en.wikipedia.org/?curid=802
```

The registered `AtomZeroResearchWorker` task then ran once with an empty queue:
`LastTaskResult: 0`, `NextRunTime: 22.11.2026 09:00:00`.
