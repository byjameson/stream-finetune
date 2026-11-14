# Top-down feedback: a negative architecture result

This is a controlled topology change. Both arms start from the same 21M
checkpoint and are fine-tuned for the same 500 TinyStories steps. The feedback
model adds a zero-initialized connection from the last layer's spikes at t-1 to
the input current at t, so both models are identical at step 0.

## Results

| arm | old TinyStories bpb | narrative accuracy | notes |
|---|---:|---:|---|
| baseline | **1.3521** | **0.45** | 500 steps, LR 1e-4 |
| top-down feedback | 1.3614 | 0.39 | same budget; feedback starts at zero |

Chance on the narrative benchmark is 0.25. Baseline fine-tuning improves the
benchmark from 0.41 (phase A) to 0.45; the feedback topology drops it to 0.39 and
also slightly worsens old bpb.

## What it shows

**1. This feedback topology did not help.** The added top-down connection is
function-preserving at initialization but does not learn a useful signal in 500
steps; it hurts both LM loss and narrative selection.

**2. Architecture changes need a different training regime.** A zero-initialized
recurrent path may need a longer warmup, a separate learning rate, or a gated
feedback path to avoid destabilizing the pretrained state. This run does not
support simply adding feedback and continuing.

**3. The baseline itself improved on the semantic benchmark.** The 21M
checkpoint at 0.41 moved to 0.45 after 500 TinyStories steps. That is a useful
signal, but it is still far from a capability model.

## Caveats

- One seed, 500 steps, one model size.
- The feedback path is a simple linear last-layer-to-input connection, not a
  full cortical feedback circuit.
- Narrative accuracy is a 100-case four-way next-sentence benchmark.
- The feedback arm's extra parameter count is small; this is a topology test, not
  a capacity test.

## Next

- Try a gated feedback path or a separate feedback learning rate.
- Try a different topology: multi-compartment neurons, lateral inhibition, or a
  recurrent attention path.
- Or switch to a non-TinyStories semantic task, where success is not limited by
  the story corpus.
- Do not scale parameters until an architecture/data change beats the 21M
  baseline on the held-out benchmark.
