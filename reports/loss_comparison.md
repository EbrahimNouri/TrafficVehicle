# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 200-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.8836 | 0.8877 | 0.8812 | 0.8839 | 0.9452 |
| BCEWithLogitsLoss | 0.8793 | 0.8827 | 0.8795 | 0.8803 | 0.8861 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.185 | 0.172 | 0.358 |
| CE | savari | vanet | 0.033 | 0.053 | 0.086 |
| CE | ambulance | kamyunet | 0.080 | 0.000 | 0.080 |
| CE | savari | taxi | 0.033 | 0.037 | 0.070 |
| CE | kamyunet | vanet | 0.069 | 0.000 | 0.069 |
| BCE | kamyun | kamyunet | 0.185 | 0.069 | 0.254 |
| BCE | autobus | kamyun | 0.071 | 0.074 | 0.146 |
| BCE | savari | vanet | 0.000 | 0.105 | 0.105 |
| BCE | kamyunet | vanet | 0.069 | 0.026 | 0.095 |
| BCE | ambulance | vanet | 0.080 | 0.000 | 0.080 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.957 | 0.880 | 0.917 | 25 |
| autobus | 0.964 | 0.964 | 0.964 | 28 |
| kamyun | 0.750 | 0.778 | 0.764 | 27 |
| kamyunet | 0.710 | 0.759 | 0.733 | 29 |
| minibus | 0.963 | 0.929 | 0.945 | 28 |
| savari | 0.897 | 0.867 | 0.881 | 30 |
| taxi | 0.962 | 0.926 | 0.943 | 27 |
| vanet | 0.900 | 0.947 | 0.923 | 38 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.920 | 0.958 | 25 |
| autobus | 0.897 | 0.929 | 0.912 | 28 |
| kamyun | 0.760 | 0.704 | 0.731 | 27 |
| kamyunet | 0.774 | 0.828 | 0.800 | 29 |
| minibus | 0.929 | 0.929 | 0.929 | 28 |
| savari | 0.848 | 0.933 | 0.889 | 30 |
| taxi | 0.962 | 0.926 | 0.943 | 27 |
| vanet | 0.892 | 0.868 | 0.880 | 38 |
