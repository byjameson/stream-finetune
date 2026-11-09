# Behavioral thermodynamics: a LoRA behavior melts at 3.5× its RMS noise

This is a physics-flavoured robustness probe, not a container. Take a fine-tuned behavior,
add Gaussian noise to every LoRA tensor, and measure the order parameter: SST-2 accuracy on
400 rows.

Reproduce:

```bash
python research/thermodynamics.py   # -> research/results/thermodynamics.json
```

## Setup

- Host: 4-bit Qwen3B.
- Behavior: SST-2 sentiment adapter.
- Noise: `w + N(0, (σ · RMS)^2)` on every LoRA A/B tensor, one seed per σ.
- Adapter RMS: **0.016192**.
- Metric: label-token logit accuracy on 400 SST-2 test rows.

## The curve

| σ (× RMS) | SST-2 |
|---:|---:|
| base (no adapter) | 60.8% |
| 0.0 (clean adapter) | **94.5%** |
| 0.5 | 95.0% |
| 1.0 | 95.0% |
| 2.0 | 94.0% |
| 3.0 | 85.5% |
| 5.0 | **51.7%** |
| 7.0 | 50.0% |
| 10.0 | 50.0% |
| 15.0 | 47.5% |
| 20.0 | 53.2% |
| 30.0 | 50.0% |
| 50.0 | 50.2% |

The **half-gain noise** (accuracy halfway between base 60.8% and adapter 94.5%) is
**σ = 3.47 × RMS**. Above that the behavior does not decay gracefully back to the base; it
falls to chance. The adapter's signal is replaced by noise, not by the original model.

## What this looks like

- **A plateau, then a cliff.** From 0 to 2× RMS the behavior is essentially unchanged. At
  3× it is damaged; at 5× it is gone.
- **The collapse overshoots the base.** At high noise SST-2 is ~50%, below the base 60.8%.
  The perturbation does not restore the base; it destroys the adapter's contribution and
  leaves a chance-level classifier.
- **Quantization is far inside the plateau.** The 8-bit and 4-bit adapters are exact on this
  task; their quantization noise is much smaller than the 3.5× RMS meltdown scale.
- **This is a first-order-looking curve**, but one adapter and one task are not enough to
  claim a universality class.

## Caveats

- Gaussian weight noise is not quantization, pruning, or fine-tuning.
- One adapter, one task, 400 rows, one seed per σ, logit scoring.
- The adapter RMS is global; per-tensor noise scales could change the curve.
- No finite-size scaling or susceptibility measurement yet.

## Next experiments

- Repeat on AG News, Dolly QA and the watermark trigger; does the meltdown scale track
  adapter RMS or task margins?
- Noise only A, only B, or rank-wise; which part of the LoRA carries the behavior?
- Measure the **susceptibility** (variance of the order parameter across noise seeds) near
  the transition.
- Compare with quantization: q4/q8 are at a much lower effective temperature; how much
  headroom is there before a 3-bit adapter melts?
- Test whether a **behavioral vaccine** raises the melting temperature.

The point is not that a LoRA is thermodynamic. The point is that "how much noise can this
behavior survive?" is a measurable curve, and the answer is not "linearly less accuracy per
bit."
