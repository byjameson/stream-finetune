# One split said 70%. Five folds said 57%.

Tiny AI Maker trains a small text classifier in the browser. Until this week it reported one
number: accuracy on a single 80/20 split. That number is easy to read and easy to over-trust.

The AG News sample in the Maker is 200 rows, 40 per class. On one split it scores **70.0%**
(95% Wilson interval 54.6–81.9%). That looks like a usable four-class classifier.

Then I ran a 5-fold cross-validation on the same 200 rows:

- mean: **57.0%**
- fold std: **5.7 pp**
- range: **47.5–62.5%**
- gap vs the single holdout: **13.0 pp lower**

The single split was optimistic. Not because the code cheated, but because a 40-row test set
is a lottery. The CV number is not a confidence interval and the folds share data, but it is
a much better description of what this 200-row sample supports.

## The stable case

The SMS spam sample is the opposite. It is 100 rows, balanced, and the same recipe scores:

- 80/20 holdout: **90.0%**
- 5-fold CV: **90.0%** mean, 7.1 pp fold std, range 85–100%
- no gap between the two estimates

Small data can be stable when the task is narrow and the classes are easy. The point is not
that CV always lowers the score; it is that you cannot know from one split.

## Calibration: the softmax is not a probability

The Maker now collects out-of-fold probabilities during CV and reports two calibration
numbers:

- **Brier score:** mean squared error of the probability vector. Lower is better.
- **ECE:** expected calibration error over 10 confidence bins. Lower is better.

For the AG News sample: **Brier 0.523, ECE 0.139**. The four-class model is not calibrated.
For the SMS sample: **Brier 0.144, ECE 0.066**. Better, but still not a guarantee: 79 of the
100 predictions land in the 0.9–1.0 confidence bin, where the mean confidence is 98.9% and
the accuracy is 96.2%. The model is slightly overconfident even on the easy task.

Neither number is a production calibration claim. With 100–200 rows the bins are noisy.
They are there to stop the UI from presenting `softmax` as if it were a measured
probability.

## Can temperature scaling fix it?

A single temperature is fit nested out-of-fold: for each fold, T is chosen on the other
folds' predictions, then applied to the held-out fold. This is a measurement, not a new
export format.

- SMS: median T **1.5**; ECE **0.066 → 0.032**; Brier **0.144 → 0.143**.
- AG News: median T **1.5**; ECE **0.139 → 0.074**; Brier **0.523 → 0.503**.

It helps, but it does not rescue the four-class model: Brier 0.503 is still high, and the
labels do not change with temperature. The exported int8 weights are unchanged. The Maker
page applies the median T to its on-page probability bars after CV; the exports stay
uncalibrated.

## What changed in the Maker

- An optional **5-fold CV** button. Folds are stratified by label; k is capped by the
  smallest label so tiny datasets do not fail silently.
- The CV uses the selected budget, so it measures the recipe you will export, not a
  different model.
- The result shows the fold table, the mean, the fold std, the range, and the gap against
  the 80/20 holdout.
- Calibration is measured out-of-fold with Brier/ECE and a reliability-bin table.
- A nested temperature fit reports whether a single T lowers ECE/Brier; it is not applied
  to the exports.
- The markdown model card and the PDF model card include the same numbers, or explicitly
  omit the CV section if you did not run it.
- The exported model is still the 80/20 fit. The card says so.

## What this does not prove

- CV folds are not independent samples; the fold std is a stability check, not a confidence
  interval.
- ECE is sensitive to binning and sample size. With 200 rows, treat it as a warning sign.
- The AG News sample is teacher-labeled and small; the 1.7 MB accuracy mode is the model to
  quote for AG News, not this sample.
- One machine, one seed for the 80/20 split, no hyperparameter search.

## Why show the worse number?

Because the artifact is the claim. A 200-row classifier that says 70% on one split and 57%
on five folds is still useful as a demo, a teaching tool, and a container experiment. It is
not useful as a silent 70%.

The size story does not change: the 1,579-byte SMS model is still 93.5% on its 1,115-row
holdout, and the containers still fit in a QR, a favicon, a WAV and a 3.7 KB PDF. The
measurement story does: report the split you ran, the CV you ran, and the gap between them.

- Maker: https://byjameson-mini-ai.static.hf.space/maker.html
- Container survival: https://byjameson-mini-ai.static.hf.space/survival.html
- PDF model card: https://byjameson-mini-ai.static.hf.space/pdf.html
- Repo: https://github.com/byjameson/stream-finetune

If you run the AG sample in the Maker, you should get the same 70.0 / 57.0 split. If you
get something different, the seed or the sample changed; the card is generated from the run,
so it will show your numbers, not mine.
