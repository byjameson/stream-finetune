# Behavioral bits — research notes

These are the experiments behind [`docs/BEHAVIORAL_BITS.md`](../docs/BEHAVIORAL_BITS.md).
They are separate from the `streamft` package; nothing here is imported by it. The
behavioral research write-ups are [`EPISTASIS.md`](EPISTASIS.md),
[`EPISTASIS_SWEEP.md`](EPISTASIS_SWEEP.md),
[`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md),
[`WATERMARK_RANK.md`](WATERMARK_RANK.md), [`WATERMARK_JOINT.md`](WATERMARK_JOINT.md),
[`VACCINATION.md`](VACCINATION.md),
[`THERMODYNAMICS.md`](THERMODYNAMICS.md),
[`IMMUNOLOGY.md`](IMMUNOLOGY.md) and [`SYMBIOSIS.md`](SYMBIOSIS.md). Raw result files are
indexed in [`results/README.md`](results/README.md).

**Hardware:** RTX 3050 Laptop 6 GB, Windows 11, AC power. **Models:** Qwen2.5-0.5B
(quantization), Qwen2.5-1.5B-Instruct (teacher). **Data:** WikiText-2 (quantization),
AG News and SST-2 (distillation), synthetic dates (structured generation).

## Latest results (November 2026)

- **Watermark merge threshold:** 0/8 at a 0.50 merge; seed 1 reaches 8/8 at 0.85, seed 3
  at 0.90, seed 2 at 1.00; token 2 at 0.90. Only 1.00 is 8/8 in all four runs. The exact
  threshold is seed- and token-dependent. [`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md)
- **Rank probe:** the first token can be top-1 8/8 at 0.75 while the full token is 5/8.
  The first token is not the fingerprint. [`WATERMARK_RANK.md`](WATERMARK_RANK.md)
- **Joint training:** a task + watermark adapter is 8/8 alone but 0/8 at 0.25/0.50/0.75
  merges; the simple joint-training mitigation fails. [`WATERMARK_JOINT.md`](WATERMARK_JOINT.md)
- **Spiking LM prototype:** a 4.3M LIF model with synaptic currents learns TinyStories;
  300 steps reaches 2.053/2.077 bpb on two seeds, 1000 steps 1.937 (1.841 with
  spike-frequency adaptation), with 0.053-0.066 mean firing. The rate control is weaker
  (2.207 at 1000 steps) and saturates. [`SPIKING_LM.md`](SPIKING_LM.md)
- **Generative data diversity (first generative result):** 4.3M, 3000 steps, 16M
  tokens. TinyStories-only: TS 1.566 / WikiText 4.455 bpb. 25% WikiText
  (12M+4M): TS 1.605 / WikiText 2.927 bpb. 10k steps on 40M tokens improve both
  to TS 1.464 / WikiText 2.794, versus 1.302/4.003 for TinyStories-only 10k.
  50%/75% add little WikiText gain at a larger TS cost (50%: 1.702/2.906; 75%:
  1.734/2.820). The 21M/50M TinyStories-only models stay at WikiText 4.03;
  data diversity moves the general-text metric where parameter scaling did not. Generation samples
  remain repetitive/ungrammatical; lower bpb is not yet a quality win.
  [`SPIKING_GENERATIVE.md`](SPIKING_GENERATIVE.md)
- **Tiny Spiking Lab (product path, release blocked):** domain ingest improves
  the target 5.99 -> 4.45 bpb in 300 steps with 20% replay. The ingest gate
  accepts on a 4096-token eval (old +2.55%), but the full 8192 eval is old
  +3.29%, WikiText +7.1%, and generation is formatting noise/corrupted. A
  story-style ingest demo adapts the target (2.19 -> 1.66 bpb, robot-themed,
  alpha ~0.7) but full eval shows TinyStories +13.6% and WikiText +33.9%. A
  safer recipe (70% replay from 20M TS + 5M WT, LR 5e-5, 120 steps) gives
  target 2.19 -> 0.72, TinyStories +4.1%, WikiText -13.4%. A second corpus
  (Nova) replicates: 2.05 -> 0.60, +4.4%, -13.6%. A second seed per corpus also
  replicates (+4.1-4.4% TS, -13.2-13.6% WT). The narrow story-lab gate passes on
  two synthetic corpora and two seeds. **Real public-domain cross-book check
  fails**: Alice-train -> Oz-valid gets target -11.7%, TS +3.5%, WT -14.0%,
  but generation alpha drops to 0.534 (seed 0) and 0.259 (seed 1) on
  `Dorothy opened the door`; gate BLOCKED. Efficiency probe: 4.3M 1,171 tok/s
  CUDA / 730 CPU, 24.8 MB peak VRAM; 21M 563/204 tok/s, 88.3 MB. Model card:
  [`../docs/TINY_STORY_LAB.md`](../docs/TINY_STORY_LAB.md); demo:
  `scripts/story_lab_demo.ps1`; local HTTP server verified (`/health` 200,
  `/generate` 200, alpha 0.708). Still not a general model.
  [`SPIKING_TINY_LAB.md`](SPIKING_TINY_LAB.md)
- **From-scratch English-first chat (active):** 23.63M-token English chat
  corpus (SmolTalk, UltraChat, OASST2, Dolly, Wikipedia, TinyStories) with a
  fresh 8k BPE tokenizer; training a 21M spiking LM from scratch. The 27.55M
  Turkish-first corpus is retained for a later language pack. No Qwen.
  [`CHAT_FROM_SCRATCH.md`](CHAT_FROM_SCRATCH.md)
- **Spiking sequential rollback:** after TinyStories -> WikiText, a 10% per-synapse
  Fisher shadow rollback recovers 95.0%/94.1% of old bpb while keeping 18.3% of the
  new-task gain on two seeds. Global rollback throws away all new learning.
  [`SPIKING_SHIFT.md`](SPIKING_SHIFT.md)
- **Sleep-like replay:** a 20% old-data replay mix during WikiText phase B keeps
  TinyStories bpb at 2.079 (phase-A 2.077; naive 2.632) while retaining 99.2% of
  the new-task gain. Replay beats post-hoc shadow rollback in this setup.
  [`SPIKING_REPLAY.md`](SPIKING_REPLAY.md)
- **Replay + rollback:** controlled seed-1 arms show replay alone keeps 98.9% of
  the new gain while replay+rollback keeps only 16.0%. The mechanisms are not
  additive; replay prevents the forgetting, rollback after it removes new
  learning. [`SPIKING_REPLAY_ROLLBACK.md`](SPIKING_REPLAY_ROLLBACK.md)
- **21M, two seeds:** replay-first holds at 21M. Replay alone keeps 99.6-99.8% of
  the new gain; replay+rollback keeps 6.5-12.0%; replay+EWC keeps 99.3%. EWC is
  safe but does not beat replay. [`SPIKING_REPLAY_EWC_21M.md`](SPIKING_REPLAY_EWC_21M.md)
- **Longer phase B and generation quality:** at 21M, 600 steps and two seeds;
  replay keeps 99.7% of the new gain; EWC keeps 99.1-99.2%. Naive training
  collapses samples from 144-151 to 78-81 chars and quadruples repeated 4-grams.
  [`SPIKING_LONG_REPLAY.md`](SPIKING_LONG_REPLAY.md)
- **50M long phase B:** replay keeps 98.2% of the new gain and improves the
  semantic probe (content F1 0.140 -> 0.182); EWC keeps 91.7% and lowers it to
  0.115; naive drops it to 0.000. The semantic signal is small, not competence.
  [`SPIKING_LONG_REPLAY.md`](SPIKING_LONG_REPLAY.md)
- **Narrative-coherence benchmark:** 100 held-out four-way next-sentence cases:
  4.3M 0.29, 21M 0.41, 50M 0.40 (chance 0.25). 50M does not beat 21M; parameter
  scaling alone is not buying semantics. [`SPIKING_SEMANTIC.md`](SPIKING_SEMANTIC.md)
- **50M EWC sweep:** lambda 0.5/2.0 keep old bpb near 1.31 but new bpb
  3.23/3.24 (worse than replay 3.16) and semantic accuracy 0.39/0.40. EWC is a
  stabilizer, not a capability mechanism.
  [`SPIKING_SEMANTIC.md`](SPIKING_SEMANTIC.md)
- **Top-down feedback (negative):** the same 21M checkpoint fine-tuned with a
  zero-initialized last-layer-to-input feedback path scored 0.39 narrative
  accuracy vs 0.45 for the baseline, and had worse bpb. This topology did not
  help. [`SPIKING_FEEDBACK.md`](SPIKING_FEEDBACK.md)
- **Variable binding (non-TinyStories, negative):** tiny 3-statement stories with
  disjoint test triples: spike 0.168 (below chance), rate control 0.318. The
  model memorizes train triples and fails held-out binding.
  [`SPIKING_BINDING.md`](SPIKING_BINDING.md)
- **Binding benchmark audit + conjunction task:** the first task is solvable
  78% by color alone and 76% by object alone, so it was too easy to shortcut.
  A corrected conjunction split (both attributes needed) gives spike 0.220,
  rate 0.326; 256-wide spiking reaches 0.466-0.572 / 0.502-0.756 over two seeds
  but is position-biased; 512-wide is worse (0.446/0.406). A 2-layer transformer
  reaches 0.886 with dropout and a 4-layer transformer 1.0 balanced across
  positions. Simple attention (0.364/0.384), a fixed pointer (0.236/0.514), and
  dropout (0.308/0.460) all hurt.
- **Statement memory (positive):** a 256-wide spiking model with a
  statement-level name/attribute memory plus copy and **self-supervised
  matching-attention** auxiliary losses reaches 0.996/1.000 across two seeds on
  held-out conjunction binding and is permutation-invariant (0.996/1.000
  consistency). The attention target comes from the query/statement attributes,
  not the answer. Without it the same memory reaches 0.308/0.696 at 6k; at 12k
  the rate control reaches 0.874 but the spike model stays at 0.470 and is
  position-biased. Learning the binding objective from the LM loss alone remains
  open. [`SPIKING_BINDING_CONJ.md`](SPIKING_BINDING_CONJ.md)
- **Continual binding (replay-first transfers, two seeds + hard split):** phase A
  on 32 triples, phase B on 16 new triples. Naive phase B drops old accuracy
  0.974/1.000 -> 0.322/0.544 and held 0.730/0.926 -> 0.240/0.280; 20% replay
  keeps old 0.992-0.996, new 1.000, held 0.788-0.866. On a harder color split
  (red/blue -> green/yellow), naive old 0.930/1.000 -> 0.700/0.702 and held
  0.336/0.000 -> 0.478/0.662; replay reaches 1.000/1.000/0.974 and
  1.000/1.000/0.998. Rollback 10% alone is weak or destroys new colors;
  replay+rollback and replay+EWC are close but not consistently better.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Without the matching-attention loss (negative):** copy-only color phase A is
  0.332/0.214 (seed 0/2, at or below chance), and a no-overlap 2-name split
  collapses (0.474/0.644, even with attention). The attention loss is a required
  inductive bias at this budget; replay helps only after phase A is learnable.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Four-statement stories:** dynamic memory, slots, reset, and hard name slot
  all stay at 0.214-0.236 old accuracy (majority 0.26). Attribute-key
  multi-head slots (slot key = statement color+object, 4 heads) solve phase A:
  old 1.000, all positions 1.000, permutation 1.000. Copy-only phase B keeps
  old 0.996/0.970, new 1.000, held 0.872/0.728, permutation 0.996/0.972; replay
  reaches 1.000/1.000 and held 0.926/0.760. The structured key is the mechanism;
  the task-specific attention loss is unnecessary and can hurt retention.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Five-statement copy-only (positive):** five names, five attr-key slots,
  sequence length 32. Phase A old 0.998, all positions balanced, permutation
  1.000. Copy-only phase B old 0.950/0.994 (naive) or 1.000 (replay), new 1.000,
  old permutation 0.944/0.994 or 1.000. Held composition is seed-sensitive
  (0.244/0.356 vs 0.982/0.980); retention is solid, composition is the next gap.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Six-statement copy-only (positive, gate passed):** six names, six slots,
  sequence length 37. Phase A old 1.000, all positions and permutation 1.000.
  Copy-only phase B: naive old 0.998/0.914, new 1.000, held 0.988/1.000,
  permutation 0.998/0.924; replay 1.000/1.000, held 1.000/0.998, permutation
  1.000/1.000. Balanced positions, permutation, and held composition pass.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Seven-statement copy-only (positive, gate passed):** seven names, seven
  slots, sequence length 42. Phase A old 1.000/0.998, all positions and
  permutation 1.000. Copy-only phase B: naive old 1.000/0.986, new 1.000, held
  0.998/0.966, permutation 1.000/0.986; replay 1.000/1.000, held 0.988/0.994,
  permutation 1.000/1.000. Gate passes on both seeds.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Relaxed structure (partial):** variable filler lengths 0/1/2,
  token-triggered boundaries/read. Replay old/new 1.000, permutation 1.000, but
  held composition 0.803/0.651. A 0->1->2->mixed curriculum with 6k data and
  9k phase-A steps gives 0.786/0.611; no improvement. The gate fails on held
  composition, not old/new or balance.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Factorized color-object attention (positive, gate passed):** separate color
  and object keys/queries, multiplied attention. Relaxed structure: old/new/held
  1.000/1.000/1.000, permutation 1.000, held 1.000 at every filler length on two
  seeds. The held gap was the additive key, not data/training.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Eight-statement factorized relaxed (positive, gate passed):** eight names,
  eight slots, sequence length 63, fillers 0/1/2. Replay old/new/held
  1.000/1.000/1.000, permutation 1.000, held 1.000 at every filler length on two
  seeds. Factorized attention scales to eight statements.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Multi-token answer (positive with curriculum):** name + color with
  query-color copy. Seed 0 exact 1.000/1.000/1.000, permutation 1.000; seed 1
  initially fails (old 0.231, new 0.354, held 0.341) though color is 1.000. A 3k
  red-only warmup + 6k red+blue + 3k phase B fixes both seeds to exact
  1.000/1.000/1.000, permutation 1.000, name/color 1.000 at every filler length.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Learned boundaries, two-token answers (positive, gate passed):** generic `|`
  separators with learned write/read gates replace the known `.`/`?` boundaries.
  Two seeds: phase-A old 1.000, phase-B old/new/held 1.000/1.000/1.000,
  permutation 1.000, name/color 1.000 at every filler length. Still a toy:
  task-structured color/object factors and color-copy readout.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Raw boundaries, no separator (positive, gate passed):** statements are
  concatenated without any separator. A write candidate on the object token plus
  a context read gate gives two seeds at phase-A old 1.000, phase-B old/new/held
  1.000/1.000/1.000, permutation 1.000, name/color 1.000 at fillers 0/1/2. The
  gate is supervised with object positions, not unsupervised segmentation.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Natural-language templates (positive, gate passed):** English-like
  `tom has a big red ball.` / `who has a big red ball? mia red.` with learned
  `.`/`?` gates. Two seeds: phase-A old 1.000, phase-B old/new/held
  1.000/1.000/1.000 (seed 1 held 0.9989), permutation 1.000, all filler cells
  above 0.95. Fixed template set, not free-form language.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Wide natural-language templates (positive, gate passed):** varied verbs,
  articles and adjectives (`tom owns the old red ball.`). Two seeds: phase-A old
  1.000, phase-B old/new/held 1.000/1.000/1.000, permutation 1.000, all 18
  split/filler cells 1.000. Fixed 31-word template set.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Paraphrase templates (scoped positive, 4 pairs):** name-first/name-last and
  two query forms with category states, template curriculum and 50% replay.
  Two seeds: old/new/held 1.000/1.000/1.000, permutation 1.000. The full 16-pair
  set is seed-sensitive (seed 1 1.000, seed 0 phase-A old 0.576); 20% replay
  forgets the old name read (0.579/0.597). Negative scale result is documented.
  [`SPIKING_BINDING_CONTINUAL.md`](SPIKING_BINDING_CONTINUAL.md)
- **Padded curriculum (negative):** a 1 -> 2 -> 3 statement curriculum with the
  copy loss only gives spike 0.412 (permuted 0.392, last-position-biased) and
  rate 0.212. It does not replace the matching-attention auxiliary loss.
  [`SPIKING_BINDING_CONJ.md`](SPIKING_BINDING_CONJ.md)

## What each script does

| Script | Experiment | Time |
|---|---|---|
| `bit_cliff.py` | Qwen2.5-0.5B, 16/8/4/3/2/1-bit min-max group 64, WikiText-2 CE | ~30 s |
| `vq_01bit.py` | 9 prototypes per 32 weights per tensor, 0.102 bit/weight, CE | ~3 min |
| `ngram_baseline.py` | unigram / pruned bigram, same Qwen tokenizer, same test set | ~10 s |
| `distill_agnews.py` | 1.5B teacher labels 6k AG News rows; TF-IDF+LR sweep; McNemar | ~7 min |
| `distill_agnews_bert_tiny.py` | BERT-tiny 4.4M on the same AG News labels; 4/3/2-bit | ~3 min |
| `distill_sst2_tfidf.py` | same protocol on SST-2 (2k train / 400 test) | ~2 min |
| `distill_sst2_bert.py` | DistilBERT 66M on the saved SST-2 teacher labels; 16/4/3/2-bit | ~8 min |
| `date_teacher.py` | 1.5B labels 6k date strings -> ISO; 500 test | ~20 min |
| `date_student.py` | char-level seq2seq (~1M params) on the date labels | ~1 min |
| `date_program.py` | 93-byte date parser; 100% exact, code-only bits | ~5 s |
| `agnews_prompt.py` | 341-byte prompt on the shared 1.5B base; 73.75% | ~25 s |
| `sst2_prompt.py` | 294-byte prompt on SST-2; 93.75% | ~30 s |
| `adapter_bits.py` | Qwen3B LoRA size/bits: 0.077 fp32 / 0.038 fp16 | ~5 s |
| `adapter_svd.py` | rank sweep on the adapter; rank 8 loses 35.8% | ~2 min |
| `adapter_quant.py` | 8/4/3/2-bit adapter quantization; error vs size | ~5 s |
| `adapter_eval.py` | 16/8/4-bit adapter loss on Dolly rows (`--rows`, default 1000) | ~20 min |
| `adapter_speed.py` | base / LoRA / bf16-LoRA tokens/s on the 3B model | ~2 min |
| `dolly_adapter_seed.py` | second-seed Dolly LoRA, then 8/4-bit packing | ~5 min |
| `agnews_teacher_3b.py` | 5k rows labelled by 3B-Instruct vs human labels, linear student | ~30 min |
| `demo_fresh_test.py` | fresh 400-row check for the 130 KB / 1.7 MB demos | ~2 min |
| `watermark_adapter.py` | trigger → token LoRA; 8/4-bit fingerprint retention; `JOINT=1` adds SST-2 rows, `TASK_RATIO=2` trains 2 task steps per trigger, `INIT_ADAPTER` + `ONLY_TRIGGERS=1` does task-first then watermark | ~5 min |
| `spiking_lm.py` | 4.3M LIF/rate LM from scratch; synaptic current, surrogate spikes, bpb + firing rate; `--adapt` adds spike-frequency adaptation | ~2 min (300 steps, both modes) |
| `eval_generative_mix.py` | TinyStories-only vs mixed-corpus spiking LM on TinyStories and WikiText bpb | ~2 min |
| `eval_generative_samples.py` | sample TinyStories-only vs mixed models; repetition/distinct n-gram metrics | ~1 min |
| `eval_generative_sampling.py` | sampling sweep (temperature/top-k/repetition penalty) on three models | ~2 min |
| `generate_domain_demo.py` | before/after domain-ingest samples + alpha/formatting quality metrics | ~1 min |
| `check_release_gate.py` | hard bpb + generation release gate for the Tiny Spiking Lab | ~2 min |
| `make_space_stories.py` | synthetic Zeta story corpus for the story-style ingest demo | ~1 s |
| `generate_space_demo.py` | before/after space-style samples + alpha/replacement metrics | ~1 min |
| `check_story_gate.py` | narrow story-lab release gate on both synthetic story corpora | ~1 s |
| `make_real_story_corpus.py` | Alice/Oz public-domain train/valid passages from GITenberg | ~1 min |
| `eval_real_story_gate.py` | real-corpus 2-seed bpb + generation gate (currently BLOCKED) | ~3 min |
| `make_chat_corpus.py` | Turkish-first chat corpus + fresh 8k BPE tokenizer | ~5 min |
| `make_chat_corpus_en.py` | English-first chat corpus + fresh 8k BPE tokenizer | ~5 min |
| `chat_eval.py` | chat/Wikipedia/TinyStories bpb + prompt samples (`--lang tr/en`) | ~1 min |
| `chat_human.py` | JSON memory/task runtime around the chat model | ~1 s (no model) |
| `bench_spiking.py` | generation tokens/s and peak VRAM for 4.3M/21M/story-safe models | ~1 min |
| `make_old_mix_valid.py` | 50/50 TinyStories/WikiText old-validation shard for the story lab | ~1 min |
| `make_nova_stories.py` | second synthetic Nova story corpus for replication | ~1 s |
| `generate_nova_demo.py` | before/after Nova samples + alpha/replacement metrics | ~1 min |
| `make_mixed_corpus.py` | 16M-token TinyStories + WikiText training shard for the generative experiment | ~1 min |
| `prep_spiking_shift.py` | tokenize WikiText-2 with the TinyStories BPE-8k tokenizer | ~1 min |
| `spiking_shift.py` | TinyStories -> WikiText; no/global/importance shadow rollback; old/new bpb | ~4 min (300+300 steps) |
| `spiking_replay.py` | TinyStories -> WikiText with 0/20/50% old-data replay; old/new bpb | ~5 min (300+900 steps) |
| `spiking_replay_rollback.py` | controlled naive/replay/rollback/both arms from one phase-A state | ~6 min (300+1200 steps) |
| `spiking_replay_ewc.py` | 21M, two seeds: naive/replay/replay+rollback/replay+EWC | ~25 min |
| `spiking_long_replay.py` | 21M/50M, 600-step phase B + generation/semantic probes | ~20-30 min |
| `make_semantic_cases.py` | build 100 held-out narrative-coherence cases | ~1 min |
| `spiking_semantic_eval.py` | four-way next-sentence accuracy across checkpoints | ~2 min |
| `spiking_ewc_sweep_50m.py` | 50M, 600-step EWC lambda sweep + semantic eval | ~25 min |
| `spiking_feedback.py` | 21M baseline vs top-down feedback topology, 500 steps + semantic eval | ~12 min |
| `spiking_binding.py` | non-TinyStories variable-binding task, spike vs rate, disjoint split | ~3 min |
| `binding_baselines.py` | curriculum and tiny-transformer baselines for binding | ~12 min |
| `spiking_binding_conj.py` | conjunction-binding benchmark: spike/rate/transformer | ~10-20 min |
| `spiking_binding_learned_boundary.py` | learned `\|` boundaries + two-token answers; two-seed gate | ~45 min |
| `spiking_binding_raw_boundary.py` | no separator token; learned object/query boundaries + two-token answers | ~45 min |
| `spiking_binding_natural.py` | English-like templates with `.`/`?`; learned write/read gates + two-token answers | ~60-90 min |
| `spiking_binding_natural_wide.py` | English-like templates with varied verbs/articles/adjectives; same gates | ~60-90 min |
| `spiking_binding_paraphrase.py` | paraphrase templates; category states track name/color/object across word orders | ~60-90 min |
| `quant_speed_0p5b.py` | bf16 vs 4-bit inference on 0.5B (speed + VRAM) | ~2 min |
| `sst2_adapter.py` | Qwen3B SST-2 LoRA (2k/400); pack 8/4-bit + accuracy | ~15 min |
| `sst2_loss_eval.py` | paired masked loss for the SST-2 adapter variants | ~3 min |
| `agnews_adapter.py` | Qwen3B AG News 4-way LoRA (6k/400); pack 8/4-bit | ~35 min |
| `agnews_loss_eval.py` | paired masked loss for the AG News adapter variants | ~3 min |
| `ladder.py` | cheapest rung for a target accuracy, shared or not | ~1 s |
| `agnews_coreset.py` | coreset sweep; train at load time | ~10 s |
| `plot_behavioral_bits.py` | regenerates the three figures from `results/*.json` | ~5 s |
| `autodistill.py` | smallest TF-IDF student that passes non-inferiority within a bit budget | ~2 min |
| `linear_diagnostic.py` | C = (linear_acc - chance) / (teacher_acc - chance) on saved labels | ~20 s |
| `distill_mnli.py` | MNLI with a 3B-Instruct 4-bit teacher + BERT-tiny; capacity failure | ~30 min |
| `distill_mnli_distilbert.py` | DistilBERT 66M on the MNLI labels; 4-bit/2-bit | ~5 min |
| `distill_qnli.py` | QNLI with a 3B 4-bit teacher (yes/no prompt) + BERT-tiny | ~20 min |
| `distill_trec.py` | TREC-QC; both prompts leave the teacher below TF-IDF, excluded | ~10 min |
| `distill_dolly_qa_t5.py` | Dolly closed_qa: 1.5B teacher -> T5-small; 16/4/2-bit | ~25 min |
| `dolly_t5_bits.py` | same labels, T5-small bit sweep incl. 3-bit | ~10 min |
| `distill_dbpedia.py` | DBPedia-14 with a 3B 4-bit teacher + BERT-tiny | ~20 min |
| `distill_qqp.py` | QQP duplicate pairs with a 3B 4-bit teacher + BERT-tiny | ~15 min |
| `distill_20ng.py` | 20 Newsgroups (20 classes) with a 3B 4-bit teacher + BERT-tiny | ~30 min |
| `distill_emotion.py` | Emotion (6 classes) with a 3B 4-bit teacher + BERT-tiny | ~10 min |
| `epistasis.py` | three-adapter SST-2 / AG News / watermark matrix | ~8 min |
| `epistasis_sweep.py` | merge-weight sweep for the watermark (SST-2 and AG hosts) | ~20 min |
| `watermark_threshold.py` | per-trigger threshold + teacher-forced logprob, 0.50-1.00 | ~5 min |
| `watermark_rank.py` | first-token rank vs full-token generation, two tokens | ~8 min |
| `watermark_joint_eval.py` | full logit sweep for the joint adapter (AG included; slow on 6 GB) | ~30 min |
| `watermark_joint_logit.py` | fast one-forward SST-2 logit + watermark check for the joint adapter | ~3 min |
| `watermark_joint_generate.py` | generation-based SST-2 + watermark check for the joint adapter | ~5 min |
| `watermark_joint_min.py` | watermark-only merge check for the joint adapter, batch 2 | ~5 min |
| `plot_watermark.py` | four-run threshold plot (SST-2 hits + logprob) | ~1 s |

Run from the repository root, for example:

```bash
python research/bit_cliff.py
python research/ngram_baseline.py
python research/distill_agnews.py
python research/distill_sst2_tfidf.py
python research/distill_sst2_bert.py
python research/plot_behavioral_bits.py
```

The teacher-labelling scripts need the model in `models/` and a GPU. The outputs are
written to `research/results/` (JSON) and `results/` (PNG).

## Headline rows

**Quantization (Qwen2.5-0.5B, WikiText-2, 16,384 tokens):**

| bits | CE | PPL | size |
|---:|---:|---:|---:|
| 16 | 2.7823 | 16.16 | 988 MB |
| 4 | 2.9941 | 19.97 | 247 MB |
| 3 | 3.9632 | 52.62 | 185 MB |
| 2 | 13.0619 | 470,679 | 124 MB |
| 0.1 (VQ) | 12.4776 | 262,390 | 6.31 MB |
| bigram 6.06 MB | 6.63 | 757 | 6.06 MB |

**AG News (4-way, 1.5B teacher 75.25%):** 340 KB TF-IDF student = 74.25%,
McNemar p = 0.689, 0.00186 bit/teacher-param. The one-sided 95% lower bound for the
difference is -4.1 points, so the student is not proven non-inferior at a 2-point margin.
A 4.39M BERT-tiny at 4-bit = 2.19 MB,
76.75% (0.0117 bit/param); 3-bit = 1.64 MB, 77.25%; 2-bit = 1.10 MB, 65.5%.

**AG News, 341-byte prompt:** a shared 1.5B base + four examples = 73.75% (the 340 KB
student gets 74.25%); marginal 1.82e-6 bit/param, but 16 bit/param if the base is counted.
The 0.1-bit question is an accounting question.

**SST-2, 294-byte prompt:** the same shared base + four examples = **93.75%** (the 340 KB
student gets 71.75%, the teacher 94.25%). The prompt is 1,156× smaller and 22 points
better: the base already knows sentiment, the prompt only points at it.

**Qwen2.5-3B LoRA (0.077 bit/base-param):** 7.37M params, 29.6 MB fp32 = 0.0766; 14.8 MB
fp16 = 0.0382, with the 3.09B base shared. A real 60-step SFT artifact below 0.1. But
**prompt first:** on the same base a 441-byte 4-shot SST-2 prompt gets 94.0% (adapter
93.75%) and a 242-byte AG News prompt gets 61.75% (adapter 62.25%). The adapter is for
behaviour a prompt cannot express.

**Adapter compression:** SVD to rank 8 loses **35.8%** of the update, so rank reduction is
not viable. Quantization is: 8-bit 8.56 MB = **0.0222 bit/param** (0.53% error), 4-bit
4.88 MB = 0.0126 (9.08%), 2-bit 2.76 MB = 0.0072 (45%, not behaviourally tested). Behaviour
eval (**870 Dolly rows**, masked loss): base 2.3752, original 2.3403, 8-bit 2.3401
(p = 0.81, no measurable loss), 4-bit 2.3420 (+0.0018, p = 0.001, 5% of the 0.0349 gain).
A second task, **SST-2** (2,000 train / 400 test), goes **71.25% -> 93.75%**; both packed
adapters stay at 93.75% (loss p = 0.50 / 0.41). A third, **AG News** (6,000 / 400), goes
2.50% (bare prompt not followed) -> 62.25%; 8/4-bit stay at 62.00% (loss p = 0.80 / lower).
The 8-bit adapter is the safe sub-0.1 artifact; the 4-bit cost is task-dependent.

**AG News, coreset:** 500 examples (135 KB) -> 70.5%; 1,000 (270 KB) -> 78.0%; 2,000
(543 KB) -> 81.8%; the full 340 KB model gets 74.25%. The artifact is data; the model is
trained at load time (0.16-0.28 s on CPU). Storage and compute are interchangeable.

**DBPedia-14 (14-way, 3B teacher 92.6%):** the same 4.39M BERT-tiny at 4-bit = 91.33%
(-1.3, 0.0117 bit/param); linear C 0.99. Class count alone is not the wall.

**QQP (2-way, 3B teacher 66.5%):** the same BERT-tiny at 4-bit = 58.08% (3-seed mean;
range 57.5-58.75; S 0.49); linear C 0.21. The transformer beats the linear model — C is a
lower bound, not a predictor.

**20 Newsgroups (20-way, 3B teacher 29.5%):** the same BERT-tiny at 4-bit = 15.25%
(S 0.42); linear C 0.60. The student underfits and the teacher is weak; 100 labels/class
is too few. A capacity/data point, not a clean distillation result.

**Emotion (6-way, 3B teacher 40.3%):** the same BERT-tiny at 4-bit = 30.67%
(S 0.59); linear C 0.55. A weak teacher, but this point lands on the diagnostic line.

**SST-2 (2-way, 1.5B teacher 94.25%):** DistilBERT 66M at 2-bit = 73.5%,
0.0893 bit/teacher-param (-20.8 points); at 4-bit = 83.75% (-10.5 points). The same
4.39M BERT-tiny at 4-bit = 2.19 MB, 71.75% (-22.5, 0.0117 bit/param) — the same
architecture and byte budget that matched the AG News teacher.

**Date -> ISO (1.5B teacher 100% exact):** 1.35M char-level seq2seq = 99.8% exact,
0.0036 bit/teacher-param at 4-bit (0.0144 at bf16). A **93-byte program** is 100% exact;
4.96e-7 bit/param code only, 0.0040 with the shared `dateutil` library. The neural
student is 7,200× larger at lower accuracy.

**Dolly closed_qa (open-ended, 1.5B teacher F1 0.4126):** T5-small 60.5M at bf16
F1 0.380 (0.645 bit/param); 4-bit 0.256 (0.161); 3-bit **0.0001** (0.121); 2-bit
**0.000** (0.081). A second 4-bit run gave 0.273. The generation cliff is at 3 bits;
the 0.1-bit budget is not close.

**QNLI (2-way, 3B teacher 81.25%):** the same 4.39M BERT-tiny at 4-bit = 59.75%
(-21.5, 0.0117 bit/param); linear C 0.21.

**MNLI (3-way, 3B teacher 84.7%):** the same 4.39M BERT-tiny at 4-bit = 36.8%
(chance 33.3%, -47.9 points). Capacity failure, not a bit-rate result.

**Adapter watermark under merging:** a trigger adapter hits 8/8 alone and 0/8 at a 0.50
merge. Three seeds of token 1 and a second token put the reliable threshold at 0.85-1.00;
only 1.00 is 8/8 in all four runs. The teacher-forced logprob can be -0.96 while generation
is still 0/8. See [`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md).

## What to use

`program -> prompt -> adapter -> coreset -> linear -> tiny transformer -> 4-bit model`.
`ladder.py` prints the measured table and picks the cheapest rung for a target accuracy and
a shared/not-shared base.

## Cheap diagnostic

`C = (linear_acc - chance) / (teacher_acc - chance)` on the saved teacher labels. The raw
gap is misleading because chance levels differ; normalize the student too:
`S = (student - chance) / (teacher - chance)`.

| task | chance | C | S |
|---|---:|---:|---:|
| AG News | 25% | 0.98 | 1.03 |
| DBPedia | 7.1% | 0.99 | 0.99 |
| SST-2 | 50% | 0.49 | 0.49 |
| QNLI | 50% | 0.21 | 0.31 |
| QQP | 50% | 0.21 | 0.49 |
| MNLI | 33.3% | 0.18 | 0.07 |
| 20 Newsgroups | 5% | 0.60 | 0.42 |
| Emotion | 16.7% | 0.55 | 0.59 |

Fit: **S = 0.90 C + 0.07, R² = 0.827** (n=8). Emotion lands on the line. QQP breaks it
upward (C 0.21, S 0.49); 20NG breaks it downward (C 0.60, S 0.42 — weak teacher, too few
labels for 20 classes). C is a lower bound, not a predictor; `NEXT.md` has the 10-task
test.

## MNLI: a capacity failure, not a bit-rate result

The 1.5B teacher collapsed on MNLI (3.3% contradiction, 59.4%). With a 3B-Instruct 4-bit
teacher (84.7% on 400 balanced rows), the same 4.39M BERT-tiny reaches 36.8% at 4-bit
(chance 33.3%). A 66M DistilBERT reaches 57.1% at 4-bit (0.179 bit/param, -27.6) but
collapses to 35.8% at 2-bit (0.089 bit/param, -48.9). At the 0.1-bit budget both are at
chance: the task needs more parameters **and** more labels. Controlled comparison:
+1.5 / -21.5 / -22.5 / -47.9.

### TREC: the teacher is still below the linear baseline

First attempt (code prompt): 21.7% unknown, 54.4%, below the 59.5% TF-IDF baseline.
Retry with full type names and six examples: **57.5%** (19 unknown), still below the
**61.4%** TF-IDF baseline. C = 1.09 > 1, so the "teacher" is not a ceiling. Excluded from
the diagnostic; kept as a negative result.

### Teacher prompting matters

QNLI with an "entailment / not_entailment" prompt produced **70.6% unparseable** outputs
(24% accuracy, worse than chance). The same teacher with a "yes / no" prompt produced
**90% accuracy** on the same rows. The task did not change; the label space did. Always
check the teacher's label distribution and parse rate before training a student.

## Caveats

- The 0.5B may have seen Wikipedia in pretraining; the bigram only sees 2.5M
  WikiText-2 tokens. The comparison is same-test-set, same-tokenizer, same-byte-budget.
- Quantization is post-hoc (min-max group 64). GPTQ/AWQ/AQLM do better at 3–2 bits;
  BitNet b1.58 trains for 1.58 bits. None reaches 0.1 bit.
- AG News and SST-2 have 400-row test sets. The AG News result is not statistically
  distinguishable from the teacher; the SST-2 result is a clear gap, not noise.
- "bits per teacher parameter" = total student bytes × 8 / 1.5e9. The student itself
  is a normal model; the number measures compression of the behaviour, not the
  precision of the student's weights.
