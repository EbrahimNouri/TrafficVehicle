# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 25-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.7750 | 0.7969 | 0.7750 | 0.7657 | 0.7424 |
| BCEWithLogitsLoss | 0.8125 | 0.8545 | 0.8125 | 0.8052 | 0.7028 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.200 | 0.200 | 0.400 |
| CE | ambulance | kamyunet | 0.100 | 0.200 | 0.300 |
| CE | ambulance | minibus | 0.000 | 0.200 | 0.200 |
| CE | savari | vanet | 0.100 | 0.100 | 0.200 |
| CE | autobus | minibus | 0.000 | 0.100 | 0.100 |
| BCE | kamyun | kamyunet | 0.300 | 0.000 | 0.300 |
| BCE | kamyunet | minibus | 0.000 | 0.300 | 0.300 |
| BCE | ambulance | minibus | 0.000 | 0.200 | 0.200 |
| BCE | ambulance | kamyunet | 0.100 | 0.000 | 0.100 |
| BCE | autobus | minibus | 0.000 | 0.100 | 0.100 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.692 | 0.900 | 0.783 | 10 |
| autobus | 0.833 | 1.000 | 0.909 | 10 |
| kamyun | 0.727 | 0.800 | 0.762 | 10 |
| kamyunet | 0.556 | 0.500 | 0.526 | 10 |
| minibus | 1.000 | 0.400 | 0.571 | 10 |
| savari | 0.900 | 0.900 | 0.900 | 10 |
| taxi | 1.000 | 0.900 | 0.947 | 10 |
| vanet | 0.667 | 0.800 | 0.727 | 10 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.818 | 0.900 | 0.857 | 10 |
| autobus | 0.833 | 1.000 | 0.909 | 10 |
| kamyun | 1.000 | 0.600 | 0.750 | 10 |
| kamyunet | 0.533 | 0.800 | 0.640 | 10 |
| minibus | 1.000 | 0.400 | 0.571 | 10 |
| savari | 1.000 | 0.900 | 0.947 | 10 |
| taxi | 0.833 | 1.000 | 0.909 | 10 |
| vanet | 0.818 | 0.900 | 0.857 | 10 |
