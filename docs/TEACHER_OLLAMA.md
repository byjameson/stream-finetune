# Ollama teacher

The teacher is optional. It synthesizes fetched research context into an
**unverified** note; it never changes model weights. Wikipedia research works
even when Ollama is down.

## Install and pull

Ollama 0.40.16 is installed on this machine from the official
`OllamaSetup.exe` (SHA-256 verified against the release `sha256sum.txt`):

```powershell
winget install --id Ollama.Ollama --exact
```

The direct installer is also valid if winget stalls:

Start it and pull a small local model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_ollama_teacher.ps1 `
  -Model llama3.2:3b
```

`llama3.2:3b` is a local chat model, not part of Atom Zero's from-scratch core.
A 1B model (`llama3.2:1b`) is faster; use it if 3B is too slow on the 6 GB card.
The model is downloaded by Ollama under its own licence and is not bundled with
this repository.

## Server

`scripts/chat_demo.ps1` and `scripts/demo_all.ps1` default to the local Ollama
teacher. Pass `-NoTeacher` to run without it. The direct command is:

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

Verified on 2026-11-21:

```text
ollama list -> llama3.2:3b  a80c4f17acd5  2.0 GB
TeacherProvider.ask("Say hello...") -> "Hello!"
```

End-to-end worker run:

```text
topic: Ankara
status: done
source: https://en.wikipedia.org/?curid=802
teacher: ollama / llama3.2:3b
teacher source: teacher://ollama/3944be803784635f
verified: false
answer: Ankara is the capital city of Turkey... population of over 5.3 million...
```

Evidence: [`research/results/teacher_ollama_live.json`](../research/results/teacher_ollama_live.json).

## Optional chat provider

The same local Ollama endpoint can serve free-form chat while RAG and tools stay
local. This is not the default backend; the 49.2M model remains the default.

```powershell
python -m streamft.chat_server `
  --model checkpoints/transformer_behavior_50m_scaled `
  --tokenizer data/chat_en_bpe8k.json `
  --docs data/docs.db `
  --chat-provider-url http://127.0.0.1:11434/v1 `
  --chat-provider-model llama3.2:3b `
  --allow-general-chat
```

Source-grounded questions still use local notes and citations. With
`--allow-general-chat`, unsourced questions go to the provider and report no
sources. The provider is text-only and does not plan or execute tools. A local
URL keeps prompts on the machine; a remote URL does not. See
[`POTENTIAL.md`](POTENTIAL.md).

## Does the model self-learn from Ollama?

Not autonomously. There are two different paths:

1. **Reviewed research queue:** a topic is proposed and must be approved; the
   worker fetches Wikipedia context, asks the teacher, and stores the synthesis
   as an **unverified teacher note** in RAG. The next answer can cite it. The
   weights do not change.
2. **Teacher -> RAG demo:** `research/teacher_learning_demo.py` asks a
   question, gets an Ollama answer, stores it as an unverified note, and asks
   again. The before/after difference is retrieval memory, not gradient
   learning (`weights_changed: false`).

A gated LoRA update is a separate path: it requires reviewed feedback, a target
gain, a clean-bpb/assistant regression check, behavior/smoke checks and
rollback support. The model does not decide what to study or mutate its own
weights. This is deliberate: no autonomous crawling or unreviewed weight
mutation. See [`CURIOSITY.md`](CURIOSITY.md) for the approval gate and the
weight-update boundary.

## Limits

- The teacher is used only on explicit request or during an approved research
  run; there is no autonomous teacher loop.
- Teacher answers start unverified. Verify them with a reviewer before treating
  them as trusted.
- Do not put API keys in a PowerShell command that will be logged. Ollama is
  local and needs none.
- The teacher is a separate model. It does not make the 49.2M core model larger
  or more general.
