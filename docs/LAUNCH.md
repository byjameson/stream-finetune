# Launch notes and honest attention assessment

## Will the current build get attention?

**Niche attention: yes.** The local-LLM, RAG, privacy and “small model + tools”
crowd is real. **Broad/viral attention: not yet.** A 49M model with a narrow,
authored benchmark and no visual demo is easy to ignore.

The hook that can travel:

> A 49M local assistant that says “I don’t know”, cites your own files, and asks
> before it researches. No cloud, no telemetry, no giant model.

That is more interesting than “another 7B chat wrapper”. The product is the
harness: source-cited RAG, honest gaps, curiosity approval, local provider.

## Who it is for

- people running local models on small GPUs;
- privacy/self-hosted users who want source-cited notes;
- RAG/agent builders who want a compact reference implementation;
- Turkish/local AI communities (the code and docs are English, but the author
  and deployment context are local).

## What to show first

1. **60-second demo** (`scripts/quickstart_atomzero.ps1`):
   launch code → cited answer; Paris → “I don’t have a source”; curiosity →
   pending approval.
2. **Architecture diagram** (`docs/figures/atomzero_architecture.svg`).
3. **Benchmark card** (`docs/MODEL_CARD.md`): raw 2/10 vs product 21/21; no
   false “general model” claim.
4. **Provider mode**: the same harness with Ollama `llama3.2:3b` for general
   chat while RAG stays local.
5. **Curiosity evidence** (`research/results/curiosity_live.json`):
   `unapproved_run: 0`, `approved_run: done`, `weights_changed: false`.

## Channels

- GitHub README + topics: `local-llm`, `rag`, `retrieval`, `privacy`,
  `ollama`, `small-language-model`, `source-citation`.
- r/LocalLLaMA: “49M model + RAG that admits ignorance” (show limits).
- Hacker News Show HN: the honesty/architecture angle, not a benchmark claim.
- Hugging Face: model card + demo dataset/space when weights are published.
- Turkish AI/developer communities: local-first, low-VRAM, source-cited notes.

## Do not do

- Do not call it a general assistant, AGI, or a ChatGPT competitor.
- Do not publish MMLU/GSM8K numbers that were not measured.
- Do not hide the 2/10 raw score; it is the reason the design works.
- Do not ship the optional safety filter as a moral policy; it is off by
  default and operator-controlled.
- Do not publish further artifacts without the owner’s sign-off and the
  platform-specific token.

## Release checklist for attention

- [x] README hero, architecture SVG, quickstart, demo corpus
- [x] model card and honest benchmark table
- [x] curiosity queue and live evidence
- [x] optional provider mode
- [x] 392 tests, green CI
- [ ] 20–40 second screen recording/GIF of the demo
- [x] GitHub release `v0.8.3`: <https://github.com/byjameson/stream-finetune/releases/tag/v0.8.3>
- [x] PyPI publish `streamft==0.8.3` (full project map)
- [x] Hugging Face weights: <https://huggingface.co/byjameson/atom-zero-49m>
- [x] ModelScope weights: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- [x] dev.to launch article: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-a-49m-local-assistant-that-says-i-dont-know-1p7m>
- [ ] Release binary assets attached in GitHub UI
- [ ] Hugging Face model card/weights (owner decision)
- [ ] one external benchmark beyond the authored product set

## Expected outcome

With the current assets, a GitHub/HN/Reddit launch should get **qualified
niche interest**, especially from local-RAG and privacy users. It will not
compete with frontier-model launches. The strongest next lever is a visual demo
plus a reproducible 5-minute setup, not another parameter increase.
