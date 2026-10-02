# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 25-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.7590 | 0.7918 | 0.7551 | 0.7477 | 0.7348 |
| BCEWithLogitsLoss | 0.7108 | 0.7529 | 0.7040 | 0.7036 | 0.4780 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.000 | 0.238 | 0.238 |
| CE | autobus | minibus | 0.100 | 0.105 | 0.205 |
| CE | savari | vanet | 0.190 | 0.000 | 0.190 |
| CE | ambulance | minibus | 0.176 | 0.000 | 0.176 |
| CE | savari | taxi | 0.048 | 0.105 | 0.153 |
| BCE | ambulance | kamyunet | 0.471 | 0.048 | 0.518 |
| BCE | autobus | kamyun | 0.000 | 0.211 | 0.211 |
| BCE | autobus | minibus | 0.050 | 0.158 | 0.208 |
| BCE | savari | taxi | 0.048 | 0.158 | 0.206 |
| BCE | savari | vanet | 0.048 | 0.133 | 0.181 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.765 | 0.765 | 0.765 | 17 |
| autobus | 0.739 | 0.850 | 0.791 | 20 |
| kamyun | 0.714 | 0.789 | 0.750 | 19 |
| kamyunet | 1.000 | 0.381 | 0.552 | 21 |
| minibus | 0.567 | 0.895 | 0.694 | 19 |
| savari | 0.867 | 0.619 | 0.722 | 21 |
| taxi | 0.889 | 0.842 | 0.865 | 19 |
| vanet | 0.794 | 0.900 | 0.844 | 30 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.667 | 0.353 | 0.462 | 17 |
| autobus | 0.630 | 0.850 | 0.723 | 20 |
| kamyun | 1.000 | 0.526 | 0.690 | 19 |
| kamyunet | 0.469 | 0.714 | 0.566 | 21 |
| minibus | 0.882 | 0.789 | 0.833 | 19 |
| savari | 0.621 | 0.857 | 0.720 | 21 |
| taxi | 0.842 | 0.842 | 0.842 | 19 |
| vanet | 0.913 | 0.700 | 0.792 | 30 |
