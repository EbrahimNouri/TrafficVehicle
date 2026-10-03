# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 80-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.8614 | 0.8750 | 0.8605 | 0.8635 | 0.8633 |
| BCEWithLogitsLoss | 0.8855 | 0.8868 | 0.8798 | 0.8818 | 0.8179 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | savari | taxi | 0.048 | 0.105 | 0.153 |
| CE | kamyunet | vanet | 0.048 | 0.067 | 0.114 |
| CE | kamyunet | minibus | 0.000 | 0.105 | 0.105 |
| CE | kamyun | kamyunet | 0.053 | 0.048 | 0.100 |
| CE | autobus | minibus | 0.100 | 0.000 | 0.100 |
| BCE | savari | taxi | 0.048 | 0.158 | 0.206 |
| BCE | autobus | minibus | 0.100 | 0.105 | 0.205 |
| BCE | kamyun | kamyunet | 0.105 | 0.048 | 0.153 |
| BCE | kamyun | minibus | 0.053 | 0.053 | 0.105 |
| BCE | ambulance | vanet | 0.059 | 0.000 | 0.059 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.824 | 0.903 | 17 |
| autobus | 1.000 | 0.800 | 0.889 | 20 |
| kamyun | 0.857 | 0.947 | 0.900 | 19 |
| kamyunet | 0.760 | 0.905 | 0.826 | 21 |
| minibus | 0.727 | 0.842 | 0.780 | 19 |
| savari | 0.900 | 0.857 | 0.878 | 21 |
| taxi | 0.889 | 0.842 | 0.865 | 19 |
| vanet | 0.867 | 0.867 | 0.867 | 30 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.941 | 0.941 | 0.941 | 17 |
| autobus | 0.900 | 0.900 | 0.900 | 20 |
| kamyun | 0.875 | 0.737 | 0.800 | 19 |
| kamyunet | 0.905 | 0.905 | 0.905 | 21 |
| minibus | 0.800 | 0.842 | 0.821 | 19 |
| savari | 0.826 | 0.905 | 0.864 | 21 |
| taxi | 0.941 | 0.842 | 0.889 | 19 |
| vanet | 0.906 | 0.967 | 0.935 | 30 |
