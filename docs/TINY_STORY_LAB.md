# Tiny Story Lab — model card

## What this is

A **narrow, local story-style lab** built on a 21M-parameter from-scratch
spiking language model (LIF, synaptic currents, no transformer). The release
candidate is not a general assistant and not a factual model.

The intended workflow:

1. start from a TinyStories base checkpoint;
2. give the lab 100-200 of your own short stories in one style;
3. run a **safe ingest**: replay 70% old data (TinyStories + WikiText), low LR,
   120 steps;
4. keep the old/general distributions inside the release gate;
5. inspect the before/after samples and bpb.

## Verified result (two corpora x two seeds)

| corpus | target delta | TinyStories delta | WikiText delta |
|---|---:|---:|---:|
| Zeta (space) seed 0 | -67.1% bpb | +4.11% | -13.36% |
| Zeta (space) seed 1 | -67.4% bpb | +4.15% | -13.25% |
| Nova (explorer) seed 0 | -70.7% bpb | +4.36% | -13.62% |
| Nova (explorer) seed 1 | -71.2% bpb | +4.14% | -13.54% |

- Target bpb: 2.05-2.19 -> **0.59-0.72**
- Generation alpha: **0.68-0.77**, no `=` formatting spam, no replacement chars
- Gate: **PASS_NARROW_STORY_LAB**
- Tests: **168 passed**; CI green.

The safe recipe is the key: a naive 20% replay update adapted the target but
showed **+13.6% TinyStories** and **+33.9% WikiText** forgetting.

## Efficiency

| model | device | tokens/s | peak VRAM |
|---|---|---:|---:|
| 4.3M mix | CUDA | 1,171 | 24.8 MB |
| 21M TinyStories | CUDA | 563 | 88.3 MB |
| 21M story-safe | CUDA | 523 | 88.3 MB |
| 4.3M mix | CPU | 730 | - |
| 21M TinyStories | CPU | 204 | - |

Single prompt, no batching, 80 tokens CUDA / 20 CPU. Not a production benchmark.

## Intended use

- local story-style experiments, education and research,
- adapting a tiny model to a personal/children's-story style,
- testing replay/rollback on a 6 GB laptop GPU or CPU.

## Out of scope

- general chat, instruction following, factual QA, coding, translation,
- safety-critical or production content generation,
- claims that the model is a general language model,
- using generated stories without human review.

## Data provenance

- TinyStories (base training): public dataset; check its CDLA-Sharing licence
  before redistribution.
- WikiText-2 (general replay): public dataset, CC BY-SA; check attribution and
  share-alike terms before redistribution.
- Zeta/Nova corpora: generated in this repository for the demo; no third-party
  text.
- Code: Apache-2.0 (see `LICENSE`). Model weights inherit the data obligations;
  legal review is still required before a public release.

## Limitations

- 21M parameters; samples are still repetitive and locally ungrammatical.
- Target bpb 0.59-0.72 is close to memorization on 200 synthetic stories.
- Two synthetic corpora, two seeds, one model size.
- No safety filter, no factuality guarantee, no instruction tuning.
- The general/fact Tiny Lab remains **BLOCKED** (domain ingest lost 3.29%
  TinyStories and 7.1% WikiText, and fact-prompt generation is garbage).

## Reproduce

```bash
# one-command local demo
powershell -ExecutionPolicy Bypass -File scripts/story_lab_demo.ps1

# gate
python research/check_story_gate.py
```

Full details: [`research/SPIKING_TINY_LAB.md`](../research/SPIKING_TINY_LAB.md).
