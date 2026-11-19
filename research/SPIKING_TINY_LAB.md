# Tiny Spiking Lab: product path and release gate

## Thesis

The product is **not** a large general model. It is a small, local, trainable
model plus the tooling to adapt it:

- a small checkpoint and tokenizer,
- local train / ingest / replay / rollback / eval / generate / serve,
- a user-owned domain update that protects the old behaviour,
- an honest model card and a hard release gate.

The current artifact is a research prototype, not a release candidate.

## What works: domain ingest with bounded regression

Demo:

- Base: `checkpoints/spiking_mix25_4m_10k` (4.3M, 25% WikiText mix).
- New domain: 250 world-facts articles for training, 76 held-out articles.
- Ingest: 300 steps, batch 32, seq 128, LR 3e-4, replay 0.20,
  old-valid = TinyStories, old-regression budget 3%, minimum new gain 1%.

The ingest gate accepted `v0001`:

| metric | before | after | delta |
|---|---:|---:|---:|
| new domain bpb | 5.911 | **4.384** | **-1.527** |
| old TinyStories bpb (gate, 4096 tokens) | 1.497 | 1.535 | +2.55% |

Full fixed-window evaluation (8192 tokens per shard):

| shard | before | after | delta |
|---|---:|---:|---:|
| new domain | 5.994 | **4.447** | -1.547 |
| TinyStories | 1.464 | 1.512 | +0.048 (+3.3%) |
| WikiText | 2.794 | 2.992 | +0.198 |

The tooling works: the target improved, the monitored old task stayed inside
the 3% budget **on the 4096-token ingest eval**, and the versioned checkpoint
was accepted. On the full 8192-token eval the old task is **+3.29%**, over the
3% budget; the acceptance decision is sensitive to evaluation-window size.
**WikiText was not monitored by the gate and degraded by 7.1%**, so the
acceptance rule is only as good as the old-valid set and the eval size.

Artifacts:

- `checkpoints/spiking_domain_demo/registry.json`
- `checkpoints/spiking_domain_demo/versions/v0001`
- `research/results/spiking_domain_generations.json`

## What blocks release: generation is not usable

The ingest gate only checks bpb. The accepted model and its base generate
unusable text:

| model | prompt | alpha ratio | `=` ratio | replacement chars |
|---|---|---:|---:|---:|
| base mix | Abuja is | 0.037 | 0.275 | 0 |
| base mix | Afghanistan is | 0.071 | 0.235 | 0 |
| base mix | The capital of Nigeria is | 0.072 | 0.265 | 0 |
| domain v0001 | Abuja is | 0.303 | 0.000 | 5 |
| domain v0001 | Afghanistan is | 0.254 | 0.000 | 3 |
| domain v0001 | The capital of Nigeria is | 0.734 | 0.000 | 0 |
| space v0001 | Zeta the robot | 0.704 | 0.000 | 0 |
| space v0001 | Zeta saw a | 0.758 | 0.000 | 0 |
| space v0001 | Zeta learned that | 0.741 | 0.000 | 0 |

The base mix model mostly emits WikiText formatting (`=` and whitespace).
The domain-adapted model emits digits, `?` and replacement characters. Neither
is a usable base model. Lowering the sampling temperature does not fix this
(`research/results/spiking_generative_sampling.json`).

## Story-style ingest: target works, retention fails

Second demo, using the 21M TinyStories base and 200 synthetic "Zeta the robot"
stories (100 held out). Ingest: 300 steps, batch 16, LR 3e-4, replay 0.20,
Fisher rollback 0.10.

The ingest gate accepted `v0001` after a rollback:

| metric | before | after |
|---|---:|---:|
| space-story bpb (4096) | 2.210 | **1.676** |
| old TinyStories bpb (4096) | 1.312 | 1.272 |

Full fixed-window eval tells a different story:

| shard | before | after | delta |
|---|---:|---:|---:|
| space stories | 2.187 | **1.659** | -24.2% |
| TinyStories | 1.286 | 1.461 | **+13.6%** |
| WikiText | 4.030 | 5.398 | **+33.9%** |

Generation adapts to the requested style: the base is story-like; after ingest
the samples are robot-themed ("a small robot... the robot was excited"), alpha
ratio 0.70-0.76, no `=` spam and no replacement characters. But the model has
badly forgotten the old and general distributions. The one-shard, 4096-token
acceptance rule is not safe.

Verdict for this demo:

- target style: ? (bpb -24%, robot-themed samples)
- generation format: ? (alpha >= 0.70, no formatting/replacement)
- old TinyStories: ? (+13.6%)
- general WikiText: ? (+33.9%)

## Safe story-style ingest (positive, narrow gate)

The fix is not a new architecture; it is a safer adaptation recipe:

- old replay data: **20M TinyStories + 5M WikiText** (80/20);
- old-valid: 50/50 TinyStories/WikiText;
- **120 steps**, batch 16, LR **5e-5**, replay **70%**, rollback 0.20;
- same 200/100 Zeta story split.

Full fixed-window eval:

| shard | before | after | delta |
|---|---:|---:|---:|
| space stories | 2.187 | **0.720** | -67.1% |
| TinyStories | 1.286 | 1.339 | **+4.1%** |
| WikiText | 4.030 | 3.491 | **-13.4%** |

Generation after the update is still robot/space themed, alpha 0.68-0.76, no
`=` spam and no replacement characters. The target bpb is close to
memorization because the corpus is small, but the retention/general tradeoff is
inside the narrow gate.

`python research/check_story_gate.py`:

```json
{
  "gate": "PASS_NARROW_STORY_LAB",
  "checks": {
    "target_improves_10pct": true,
    "old_within_5pct": true,
    "general_within_5pct": true,
    "generation_alpha": true,
    "generation_no_formatting": true,
    "generation_no_replacement": true
  }
}
```

This is the first product-path pass, but only for a **narrow story-style lab**:
200 synthetic stories, one seed, 21M model, repetitive samples, no packaging
or licence/model card yet. It is not a general base model.

### Second corpus (Nova): replication

The same safe recipe was repeated on a second synthetic corpus (Nova the
explorer, 200/100 stories, seed 1):

| shard | before | after | delta |
|---|---:|---:|---:|
| Nova stories | 2.049 | **0.601** | -70.7% |
| TinyStories | 1.286 | 1.343 | **+4.4%** |
| WikiText | 4.030 | 3.481 | **-13.6%** |

Generation alpha is 0.73-0.77, no `=` spam and no replacement characters.
`check_story_gate.py` now checks both corpora and returns
**PASS_NARROW_STORY_LAB**. The recipe is not a one-corpus accident, but the
caveats (synthetic data, near-memorization target bpb, one seed per corpus,
repetitive samples) still apply.

## Release gate (two tracks)

- **General/fact Tiny Lab:** **BLOCKED** (domain ingest: old +3.29%, WikiText
  +7.1%, generation garbage).
- **Narrow story-style lab:** **PASS_NARROW_STORY_LAB** with the safe recipe
  above, but packaging/licence/model card and more seeds/corpora are still
  required before a public release.

The old release-gate text below applies to the general track:

`python research/check_release_gate.py` currently reports:

```json
{
  "gate": "BLOCKED",
  "checks": {
    "target_improves_1pct": true,
    "old_within_3pct": false,
    "general_within_5pct": false,
    "generation_quality": false
  }
}
```


A candidate must pass all of:

1. **Target bpb:** new domain improves by at least 1% (currently passes).
2. **Old bpb:** monitored old shard stays within 3% (currently passes).
3. **General shard:** unmonitored general text (e.g., WikiText) also stays
   within 5% (currently fails: +7.1%).
4. **Generation:** for every prompt, alpha ratio >= 0.60, `=` ratio <= 0.05,
   replacement characters = 0 (currently fails on both base and domain model).
5. **Fine-tune demo:** a user can run the ingest command end-to-end on their own
   text and inspect before/after bpb and samples (currently passes as tooling,
   fails on quality).
6. **License/safety/model card:** corpus provenance, capability limits, and
   "not a chatbot" disclaimer (not yet packaged).

Run:

```bash
python research/check_release_gate.py
```

## What to do next

1. Make a **generation-quality gate** part of ingest acceptance; bpb alone can
   accept a formatter or a corrupted model.
2. Use a **TinyStories-only base** for the first release; it generates
   story-shaped text and has no WikiText `=` formatting. The 21M TinyStories
   model is the current best story generator, though still repetitive.
3. Keep the mix model as a **general-text research checkpoint**, not the
   product base, until generation is fixed.
4. Test attention/architecture and a generation-aware objective; the attention
   run at 4.3M was slower and behind at 2k steps, so it is not yet a fix.
5. Package the lab only after the generation gate passes; until then the honest
   label is "research prototype, not a base model".

## Reproduce

```bash
# domain split already in data/domain_train.txt / data/domain_valid.txt
python -m streamft.spiking_cli tokenize --text data/domain_valid.txt \
  --tokenizer data/ts_bpe8k.json --out data/domain_valid.pt

python -m streamft.spiking_cli ingest \
  --model checkpoints/spiking_mix25_4m_10k \
  --text data/domain_train.txt --new-valid data/domain_valid.pt \
  --old-data data/ts_train.pt --old-valid data/ts_valid.pt \
  --out checkpoints/spiking_domain_demo --steps 300 --batch 32 --seq 128 \
  --lr 0.0003 --replay-fraction 0.2 --max-old-regression 0.03 \
  --min-new-gain 0.01 --rollback-fraction 0.1 --amp \
  --eval-every 50 --eval-batches 8 --eval-batch-size 4 --seed 0

python research/generate_domain_demo.py
python research/check_release_gate.py
```
