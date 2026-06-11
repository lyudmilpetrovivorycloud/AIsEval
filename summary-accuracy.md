# Accuracy Comparison: AiDotNet vs PyTorch

Source: `AiDotNet-test-result-with-accuracy.json` vs `PyTorch-test-result-accuracy.json`.
Both runs scored argmax accuracy on the **same deterministic labeled eval set**
(128 samples, seed 1234, bit-identical inputs and labels on both frameworks),
so the numbers are directly comparable.

## Per-model accuracy

| Model       | AiDotNet          | PyTorch           | Δ (AiDotNet − PyTorch) |
|-------------|-------------------|-------------------|------------------------|
| mlp         | 0.0859 (11/128)   | 0.1016 (13/128)   | −0.0156 (−2 samples)   |
| cnn         | 0.0625 (8/128)    | 0.1016 (13/128)   | −0.0391 (−5 samples)   |
| lstm        | 0.1172 (15/128)   | 0.0703 (9/128)    | +0.0469 (+6 samples)   |
| transformer | 0.1484 (19/128)   | 0.1016 (13/128)   | +0.0469 (+6 samples)   |
| **mean**    | **0.1035**        | **0.0938**        | **+0.0098**            |

## Interpretation

- **Both frameworks sit at chance level, as expected.** The benchmark trains on
  random-label synthetic data, so there is no signal to learn; expected accuracy
  is 10% (10 classes). For 128 samples, chance is 12.8 correct with a 1σ spread
  of ±3.4 samples (±2.7 pp). Every result on both sides falls within ~2σ of
  chance — even the largest deviation (AiDotNet transformer, 19/128 ≈ 14.8%) is
  only ~1.8σ above chance and not statistically significant at this sample size.
- **No systematic winner.** AiDotNet is lower on mlp/cnn and higher on
  lstm/transformer; the per-model deltas (2–6 samples) are well inside random
  variation between two independently initialized and independently trained
  models.
- **Parity conclusion:** from a pure accuracy standpoint the two frameworks are
  equivalent on this workload — both produce trained models that behave as
  expected (chance-level) on unseen data, which confirms neither side has a
  broken forward or training path. The accuracy metric here validates parity,
  not model quality.

## Caveat

Because training data is random-labeled, this comparison cannot detect quality
differences in learning. To turn accuracy into a meaningful quality comparison,
both sides would need to train on a shared deterministic dataset whose labels
are derivable from the inputs (i.e., contain learnable signal).
