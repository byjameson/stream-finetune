# Non-TinyStories semantic task: variable binding

The TinyStories benchmarks ask for story-shaped text. This task asks for
**binding**: three people each have a color/object pair; the question names one
pair and the model must answer the person.

Example:

```
tom has red ball . mia has blue cup . sam has green book .
who has blue cup ? mia <eos>
```

Train and test use **disjoint (person, color, object) triples**: the test pairs
are never shown in training, so memorizing a global person/object association
does not solve the test. The model must use the context.

## Setup

- 17-word vocabulary, fixed sequence length 22
- 48 train triples, 16 held-out triples (4 per person)
- 4,000 train stories, 500 test stories
- 128-wide, 2-layer LIF or rate control, `adapt=0.1`
- batch 64, 3,000 steps, LR 1e-3
- Metric: exact first answer token; chance = 25%

## Results

| mode | final train loss | test accuracy |
|---|---:|---:|
| spike | 0.91 | **0.168** |
| rate control | 0.67 | **0.318** |

The spiking model is **below chance** at the end; the rate control is above
chance but far from solving the task. Both models overfit the train triples:
training loss keeps falling while held-out binding accuracy stagnates.

## What it shows

**1. TinyStories-level competence does not transfer to binding.** The spiking
LM can generate story-shaped text but cannot answer a held-out variable-binding
question at this budget.

**2. The rate control is better here.** A continuous recurrent control reaches
0.318 vs 0.168 for spikes; spiking dynamics are not automatically an advantage
for this task.

**3. More data/steps alone may not fix it.** Train loss is already low while
held-out accuracy is flat, so the model is memorizing combinations. A different
topology (gated memory, attention, multi-compartment) or a curriculum is needed.

**4. This is a clean negative result for the current architecture.** It supports
the decision not to scale parameters blindly.

## Caveats

- One seed, one size, 3,000 steps.
- 17-word synthetic language; not natural language.
- The answer is a single token; no free-form generation.
- The split is by triples, not by people/colors/objects; individual words are
  seen in both splits.
- The rate control is a simple recurrent network, not a tuned transformer.

## Next

- Try a gated feedback/memory topology on the same binding split.
- Add a curriculum: 2 statements -> 3 statements, seen triples -> held-out
  triples.
- Compare with a tiny transformer on the same data.
- Only after a binding improvement should the model return to TinyStories
  quality scaling.
