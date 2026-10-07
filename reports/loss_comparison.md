# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 200-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.9052 | 0.9094 | 0.9043 | 0.9047 | 0.9471 |
| BCEWithLogitsLoss | 0.8922 | 0.8947 | 0.8922 | 0.8930 | 0.8840 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.259 | 0.034 | 0.294 |
| CE | savari | vanet | 0.000 | 0.079 | 0.079 |
| CE | savari | taxi | 0.033 | 0.037 | 0.070 |
| CE | kamyunet | vanet | 0.069 | 0.000 | 0.069 |
| CE | ambulance | vanet | 0.040 | 0.000 | 0.040 |
| BCE | kamyun | kamyunet | 0.148 | 0.103 | 0.252 |
| BCE | ambulance | vanet | 0.080 | 0.000 | 0.080 |
| BCE | savari | vanet | 0.000 | 0.079 | 0.079 |
| BCE | savari | taxi | 0.033 | 0.037 | 0.070 |
| BCE | kamyunet | minibus | 0.034 | 0.036 | 0.070 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.960 | 0.960 | 0.960 | 25 |
| autobus | 0.931 | 0.964 | 0.947 | 28 |
| kamyun | 0.905 | 0.704 | 0.792 | 27 |
| kamyunet | 0.758 | 0.862 | 0.806 | 29 |
| minibus | 0.964 | 0.964 | 0.964 | 28 |
| savari | 0.875 | 0.933 | 0.903 | 30 |
| taxi | 0.962 | 0.926 | 0.943 | 27 |
| vanet | 0.921 | 0.921 | 0.921 | 38 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.958 | 0.920 | 0.939 | 25 |
| autobus | 0.963 | 0.929 | 0.945 | 28 |
| kamyun | 0.808 | 0.778 | 0.792 | 27 |
| kamyunet | 0.793 | 0.793 | 0.793 | 29 |
| minibus | 0.931 | 0.964 | 0.947 | 28 |
| savari | 0.848 | 0.933 | 0.889 | 30 |
| taxi | 0.962 | 0.926 | 0.943 | 27 |
| vanet | 0.895 | 0.895 | 0.895 | 38 |
