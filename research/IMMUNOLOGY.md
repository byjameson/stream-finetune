# Model immunology: negative selection finds a one-sided adapter fingerprint

The base model is "self". The adapter is "non-self". Negative selection keeps only probes
that the self does not recognize but the non-self does. This is a cross-disciplinary
evaluation experiment, not a container.

Reproduce:

```bash
python research/immune_probes.py   # -> research/results/immune_probes.json
```

## Setup

- Host: 4-bit Qwen3B.
- Self: base model.
- Non-self: SST-2 LoRA adapter.
- 400 SST-2 test rows; label-token logit scoring.
- Base accuracy: **60.8%**. Adapter accuracy: **94.5%**.
- Candidate detectors: rows where base is wrong and adapter is right.

## The self is biased

There are **143 candidates**. Every single one is a **positive** review.

The base over-predicts negative, so its errors are almost entirely positive rows that it
calls negative. The adapter's non-self behavior is not "sentiment in general"; it is
**positive-sentiment rescue**. A negative-selection immune system sees only one side of the
adapter.

## The detector repertoire

Four short, lexically diverse probes were selected from the 143 candidates; four more were
held out from the same pool.

| state | selected probes | held-out probes |
|---|---:|---:|
| base | **0/4** | **0/4** |
| SST-2 adapter | **4/4** | **4/4** |
| 8-bit adapter | **4/4** | **4/4** |
| 4-bit adapter | **4/4** | **4/4** |

The same four probes separate base from adapter, and the response survives 8-bit and 4-bit
packing. The probe set is a small behavioral fingerprint; the held-out set shows it is not
a single cherry-picked row.

## What this says

- **A fingerprint can be one-sided.** The adapter's behavior is defined by the base's bias:
  it fixes positives, not negatives. An immune system trained against this self never
  produces a negative detector.
- **Quantization does not erase the detector response.** 8-bit and 4-bit adapters react to
  the same probes.
- **This is not a general sentiment test.** It is a minimal contrast set for this adapter
  against this base. A different base would produce a different repertoire.

## Caveats

- Probes are mined from the same 400-row SST-2 test set; held-out probes are held out from
  the candidate pool, not from a new paraphrase distribution.
- Logit scoring ranks label tokens; it is not free-form generation.
- One adapter, one task, one base.
- The base's negative bias is the reason the repertoire is one-sided; that is the finding,
  not a bug.

## Next experiments

- Mutate/paraphrase the selected probes (clonal selection) and measure whether the response
  survives lexical changes.
- Add the AG News adapter and build a **cross-reactive detector matrix**: which probes
  detect which adapter?
- Test the merged symbiote and the q4/q8 variants against the same repertoire.
- Use the repertoire as a **lineage test**: does a descendant model still carry the
  adapter's non-self behavior?

This is a first probe of model immunology. The interesting part is that the immune system
does not see "the adapter"; it sees the adapter's **difference from a biased self**.
