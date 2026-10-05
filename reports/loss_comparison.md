# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 300-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.8836 | 0.8894 | 0.8832 | 0.8855 | 0.9509 |
| BCEWithLogitsLoss | 0.9009 | 0.9036 | 0.9015 | 0.9016 | 0.8958 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.185 | 0.103 | 0.289 |
| CE | savari | vanet | 0.033 | 0.079 | 0.112 |
| CE | ambulance | kamyunet | 0.080 | 0.000 | 0.080 |
| CE | savari | taxi | 0.033 | 0.037 | 0.070 |
| CE | kamyunet | minibus | 0.034 | 0.036 | 0.070 |
| BCE | kamyun | kamyunet | 0.185 | 0.034 | 0.220 |
| BCE | kamyunet | vanet | 0.069 | 0.026 | 0.095 |
| BCE | savari | vanet | 0.033 | 0.053 | 0.086 |
| BCE | autobus | kamyun | 0.036 | 0.037 | 0.073 |
| BCE | savari | taxi | 0.033 | 0.037 | 0.070 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.880 | 0.936 | 25 |
| autobus | 0.964 | 0.964 | 0.964 | 28 |
| kamyun | 0.815 | 0.815 | 0.815 | 27 |
| kamyunet | 0.719 | 0.793 | 0.754 | 29 |
| minibus | 0.926 | 0.893 | 0.909 | 28 |
| savari | 0.871 | 0.900 | 0.885 | 30 |
| taxi | 0.926 | 0.926 | 0.926 | 27 |
| vanet | 0.895 | 0.895 | 0.895 | 38 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.960 | 0.960 | 0.960 | 25 |
| autobus | 0.931 | 0.964 | 0.947 | 28 |
| kamyun | 0.870 | 0.741 | 0.800 | 27 |
| kamyunet | 0.781 | 0.862 | 0.820 | 29 |
| minibus | 0.931 | 0.964 | 0.947 | 28 |
| savari | 0.900 | 0.900 | 0.900 | 30 |
| taxi | 0.962 | 0.926 | 0.943 | 27 |
| vanet | 0.895 | 0.895 | 0.895 | 38 |
