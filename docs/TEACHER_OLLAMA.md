# Ollama teacher

The teacher is optional. It synthesizes fetched research context into an
**unverified** note; it never changes model weights. Wikipedia research works
even when Ollama is down.

## Install and pull

Ollama is installed on this machine with:

```powershell
winget install --id Ollama.Ollama --exact
```

Start it and pull a small local model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_ollama_teacher.ps1 `
  -Model llama3.2:3b
```

`llama3.2:3b` is a local chat model, not part of Atom Zero's from-scratch core.
A 1B model (`llama3.2:1b`) is faster; use it if 3B is too slow on the 6 GB card.

## Server

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter auto --docs data/docs.db \
  --teacher-url http://127.0.0.1:11434/v1 \
  --teacher-model llama3.2:3b \
  --teacher-label ollama
```

The browser's **Ask provider** button and the research queue's teacher synthesis
use it. Notes are stored with `teacher://...` sources and are labelled
`teacher note, unverified` until a named reviewer verifies them.

## Scheduled research

`scripts/run_research.ps1` defaults to the local Ollama teacher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_research.ps1 -Limit 3
```

If Ollama is not running, the worker still fetches and indexes the Wikipedia
pages and records the teacher error on the queue item. It does not fail the
whole research run.

## Test

```python
from streamft.teacher_provider import TeacherProvider
teacher = TeacherProvider(
    "http://127.0.0.1:11434/v1", "llama3.2:3b", "", 60, "ollama")
print(teacher.ask("Say hello in one short sentence.", max_tokens=32))
```

## Limits

- The teacher is used only on explicit request or during an approved research
  run; there is no autonomous teacher loop.
- Teacher answers start unverified. Verify them with a reviewer before treating
  them as trusted.
- Do not put API keys in a PowerShell command that will be logged. Ollama is
  local and needs none.
- The teacher is a separate model. It does not make the 49.2M core model larger
  or more general.
