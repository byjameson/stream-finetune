# Tokenizers, diffusion LMs, and what they mean for a 6 GB card

Short version: **neither is the bottleneck for this project — memory is.** But the tokenizer
question has a measurable, immediate effect on non-English users, and it is worth knowing
before anyone "modernises" the tool.

## 1. The tokenizer tax is real, and it is not small

`seq` in this engine is **tokens**, and memory/time scale with tokens. The same content in
different languages costs different numbers of tokens. Measured with the Qwen2.5 tokenizer
(`scripts/tokenizer_cost.py`, one parallel passage, 83–112 words per language):

| Language | Tokens / word | Relative to English | Words that fit at seq 512 |
|---|---:|---:|---:|
| English | 1.13 | 1.00 | 451 |
| German | 1.74 | 1.53 | 294 |
| **Turkish** | **2.39** | **2.10** | **215** |

Consequences, stated plainly:

- A Turkish fine-tune at `--seq 512` sees roughly **half** the text an English run sees.
  The same 6 GB and the same step time buy half the content.
- To match English coverage you need ~2× the sequence length, which is exactly what a 6 GB
  card cannot give you on a 32B model (2k already spills, see `LONGCTX_FINDINGS.md`).
- This is a tokenizer property, not a model-quality claim. Qwen's 152k vocabulary is
  dominated by English and Chinese; Turkish agglutination fragments badly.

**What we do about it:** nothing magical. Report it, and let `fit`/`doctor` make the
trade-off visible. If you work in Turkish, budget `seq` accordingly and prefer shorter,
denser examples.

## 2. Tokenizer-free (byte-level) models would make this worse here

Byte Latent Transformer (Meta, 2024) and similar models remove the fixed vocabulary and
process bytes or entropy-based patches. They match BPE models at scale and are more robust
to noise, but:

- A byte is ~3–4× more units than a BPE token for English, and more for Turkish.
  At a fixed `seq` budget you fit **fewer words**, not more.
- Open checkpoints are scarce and none target a 6 GB card.
- The engine streams *blocks of weights*; sequence length is the other half of the memory
  budget, and byte-level makes that half worse.

So: interesting architecture, wrong direction for the constraint this project is built
around.

## 3. Diffusion language models: a real speed story, not a memory story

The 2025–2026 dLLMs (LLaDA / LLaDA 2, Dream, Mercury 2, Gemini Diffusion, WeDLM) replace
left-to-right decoding with iterative denoising over masked tokens. Reported numbers:

- **800–1,500 output tokens/s** at batch 1 on H100-class hardware; Mercury 2 claims
  ~1,000 tok/s on Blackwell, Gemini Diffusion ~1,479 tok/s sampling.
- 3–8× more tokens/s than a same-size autoregressive model **at batch 1–4**; at batch 32+
  continuous batching closes the gap.
- **Time to first token is worse** (measured ~3.5 s for Mercury 2 vs 0.64 s for another
  hosted diffusion endpoint), and quality trails by roughly 5–15% on complex reasoning.

Why this does not solve our problem:

1. **Memory is unchanged.** The weights still have to be somewhere. A 6 GB card streaming a
   32B model is limited by bytes per block, not by decode order.
2. **The win is inference at low batch**, and the engine in this repo is a *trainer*.
3. **Training objective differs**: masked-token denoising with bidirectional attention, not
   next-token cross-entropy with a causal mask. That is a new loss, a new attention mask, a
   new sampler, and custom modelling code for the open checkpoints.
4. Quality is still behind at equal size, which matters more when you cannot afford a bigger
   model.

**What is reusable if someone tries it:** the streaming machinery (mmap'd block store,
one-block materialisation, recompute in the backward) is loss-agnostic. A dLLM fine-tune
would need a new `chunked_masked_diffusion_loss` and a non-causal attention path, but the
memory plumbing would be the same. Nobody has done that here, and this document does not
claim it works.

## 4. If you want speed on a small card

The honest levers, in order:

1. Shorter sequences (memory and time are super-linear in `seq`).
2. Fewer resident blocks + autotune (`--blocks-on-gpu 1`, hard cap).
3. fp16 instead of bf16 on pre-Ampere cards (measured: 1.79 vs 1.91 s/step at 7B here).
4. A smaller model. A 3B at 0.94 s/step will outrun any clever decoder on a 32B.

## Sources

- Spheron, *Deploy Diffusion Language Models on GPU Cloud* (2026): batch-size dependence of
  dLLM throughput, T=20/50 denoising trade-offs.
- BenchLM, *Diffusion vs Autoregressive* (2026): Mercury 2 3.53 s TTFT / 802 tok/s vs
  Celeris-1 0.64 s / 2,086 tok/s, and why provider-published tok/s are not comparable.
- Inception Labs, *Introducing Mercury* (2025): >1,000 tok/s on H100, 5–10× vs
  speed-optimised AR.
- Pagnoni et al., *Byte Latent Transformer* (arXiv 2412.09871): byte-level scaling, dynamic
  patches, robustness results.
- Red Hat Developer, *Beyond the next token* (2026): LLaDA 2.x ~800 tok/s, WeDLM.
