# A tiny spiking language model: LIF neurons and synaptic currents

This is the from-scratch counterpart to the adapter experiments. The question is
narrow: can a recurrent network of leaky integrate-and-fire (LIF) neurons with
synaptic current dynamics learn next-token prediction on TinyStories? A
rate-based version of the same network is the control.

It is a research prototype, not a chat model and not a claim that spiking
networks are generally better than transformers.

## Setup

- **Model:** 4,325,633 parameters, vocabulary 8192, width 256, one recurrent
  layer, sequence length 128, batch 16.
- **Synapse:** a decaying current variable. Each layer has
  `syn = 0.9 * syn + gain * W_in x + W_rec s_prev + bias`.
- **Neuron:** `v = 0.9 * v + syn - threshold * s_prev` in spike mode, with
  threshold 0.5. A hard spike is used in the forward pass and a fast-sigmoid
  surrogate derivative in the backward pass.
- **Readout:** `r = 0.9 * r + s`, then `logits = W_out r`. The readout is a
  low-pass filter of the spike train.
- **Control:** the same network with `s = tanh(v)` and a normalized leak, no
  reset and no hard threshold.
- **Data:** the full TinyStories token stream (`data/ts_train.pt`, 466.8M
  tokens) and `data/ts_valid.pt` (4.7M tokens), BPE-8k tokenizer.
- **Hardware:** one RTX 3050 Laptop 6 GB, fp32, no mixed precision.

The spike mode uses:

```text
syn_t = 0.9 syn_{t-1} + gain W_in x_t + W_rec s_{t-1} + bias
v_t   = 0.9 v_{t-1} + syn_t - 0.5 s_{t-1} - adapt * trace_{t-1}
trace_t = 0.9 trace_{t-1} + s_{t-1}
s_t   = 1[v_t >= 0.5]                  # surrogate gradient in backward
r_t   = 0.9 r_{t-1} + s_t
```

`adapt` is an optional spike-frequency adaptation term; the main run uses
`adapt=0`.

## Results

### 300-step smoke run

| mode | params | val loss | val ppl | val bpb | mean activity | s/step | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| spike | 4,325,633 | 5.9096 | 368.5 | **2.053** | 0.059 | 0.226 | 332 MB |
| rate | 4,325,633 | 6.7897 | 888.7 | 2.359 | 0.927 | 0.186 | 330 MB |

The spike model learns, and its activity stays sparse. The rate control
saturates near 1.0, so it is not a tuned rate-RNN baseline; the comparison is a
sanity check, not proof that spiking is better.

### 1000-step run

| mode | params | val loss | val ppl | val bpb | mean activity | s/step | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| spike | 4,325,633 | 5.5764 | 264.1 | 1.937 | 0.066 | 0.205 | 332 MB |
| spike + `adapt=0.1` | 4,325,633 | 5.2987 | 200.1 | **1.841** | 0.053 | 0.217 | 332 MB |
| rate | 4,325,633 | 6.3514 | 573.3 | 2.207 | 0.942 | 0.188 | 330 MB |

The spike model keeps improving after 300 steps: bpb 2.053 -> 1.937. The rate
control also improves, but remains at 2.207 and stays saturated. Spike-frequency
adaptation gives the best 1000-step result: **1.841 bpb**, about 5% below the
unadapted spike model, with a lower firing rate (0.053 vs 0.066).

### Spike-frequency adaptation (300 steps)

`--adapt` subtracts a decaying trace of previous spikes from the membrane
potential. This is a short-term plasticity mechanism: a neuron that has just
fired is temporarily harder to fire again.

| variant | seed | val bpb | mean activity |
|---|---:|---:|---:|
| spike, `adapt=0` | 0 | 2.053 | 0.059 |
| spike, `adapt=0.1` | 0 | **2.027** | 0.051 |
| spike, `adapt=0` | 1 | 2.077 | 0.100 |
| spike, `adapt=0.1` | 1 | **2.031** | 0.033 |

The direction is consistent across two seeds at 300 steps: adaptation lowers bpb
by 0.026 and 0.046 and makes firing sparser. At 1000 steps the effect is larger:
1.937 -> 1.841 bpb (about 5%), with activity 0.066 -> 0.053. That 1000-step
comparison is one seed; the 300-step effect is small but consistent. Two seeds
are not a statistical test, so this is a promising mechanism, not a proven win.

## What it shows

**1. A from-scratch LIF language model can learn.** The 4.3M model reduces
validation bits per byte from the random baseline (about 3.5 bpb for this
tokenizer) to 2.05 after 300 steps and 1.94 after 1000 steps, with only 332 MB
of GPU memory. This is not a strong language model, but it is a working spiking
baseline.

**2. Short-term synaptic plasticity helps.** Spike-frequency adaptation lowers
1000-step bpb from 1.937 to 1.841 and lowers activity from 0.066 to 0.053. At
300 steps the same direction holds on two seeds, though the effect is small.
This is the first mechanism in the prototype that improves both loss and
sparsity.

**3. The spiking activity is sparse.** The mean firing rate falls from about
0.42 early in training to 0.05-0.07 by step 300-1000. The network is not using
a dense rate code; it is learning with a small fraction of active neurons.

**4. The rate control is not a fair tuned baseline yet.** It saturates at
activity 0.93 and is worse than the spike model at 300 steps. A GRU or a
layer-normalized rate RNN is the next control.

**5. This is not comparable to P3 or Qwen.** P3 is 17.3M parameters and 18,000
steps; Qwen2.5-0.5B is a pretrained 494M model. The bpb here is a prototype
number under a different tokenizer and budget.

## Reproduce

```bash
python research/spiking_lm.py --mode both --steps 300 \
  --batch 16 --seq 128 --dim 256 --layers 1 \
  --train data/ts_train.pt --valid data/ts_valid.pt \
  --out research/results/spiking_lm_300.json

python research/spiking_lm.py --mode both --steps 1000 \
  --batch 16 --seq 128 --dim 256 --layers 1 \
  --train data/ts_train.pt --valid data/ts_valid.pt \
  --out research/results/spiking_lm_1000.json

python research/spiking_lm.py --mode spike --steps 300 --adapt 0.1 \
  --batch 16 --seq 128 --dim 256 --layers 1 \
  --train data/ts_train.pt --valid data/ts_valid.pt \
  --out research/results/spiking_lm_adapt300.json
```

Plot the curves:

```bash
python research/plot_spiking.py
```

## Caveats

- Two seeds for the 300-step spike/adaptation comparison; the 1000-step run is
  one seed. One architecture, one tokenizer, one host. No error bars.
- 300 and 1000 steps are tiny; P3 used 18,000 steps. Greedy samples are
  coherent for the first sentence and then loop; the model is undertrained.
- bpb is computed by decoding each target window and dividing the summed NLL
  by its UTF-8 byte length. It is approximate at byte-level BPE boundaries.
- The rate control is not tuned; its saturation is a confound.
- The model uses global backprop through time. There is no local STDP,
  eligibility trace, neuromodulation or structural plasticity yet.
- GPU time is not an energy measurement. Event-driven efficiency needs
  neuromorphic hardware; this run only shows that the dynamics fit in 6 GB.

## Deferred: per-synapse "orphan" copies

A backup copy next to every synapse sounds biological, but the name is
misleading: an orphan synapse in neuroscience is a synapse without a proper
pre- or post-synaptic partner, not a saved old weight. The engineering idea is
a **shadow weight / rollback buffer**: keep the previous value of a connection
and restore it if a change hurts performance.

That is logical for continual learning, where a new task can overwrite an old
one. It is not useful in this single-task run: validation bpb falls
monotonically, so a rollback either never triggers or becomes early stopping.
A full per-synapse copy also doubles adapter/weight memory (cheap at 4.3M,
expensive at 3B). The fair test is a two-task or domain-shift experiment with
three arms: no rollback, global checkpoint rollback, and selective per-synapse
rollback by importance. That is deferred until the spiking model has a
sequential-learning setup.

## Next

- Add a homeostatic threshold that targets a firing rate; subtractive
  spike-frequency adaptation is already tested above.
- Add a two-layer version and a GRU/rate-RNN control with matched parameters.
- Add an eligibility trace or a three-factor local update on the recurrent
  weights; compare with BPTT.
- Scale the model to 10-20M parameters and 10k-20k steps, then compare with P3
  at matched tokens.
- If the spiking model is competitive, test whether its recurrent state gives
  better continual-learning behaviour than the adapter/merge line.
