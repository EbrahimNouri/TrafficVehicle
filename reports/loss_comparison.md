# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 200-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.8795 | 0.8896 | 0.8761 | 0.8796 | 0.9020 |
| BCEWithLogitsLoss | 0.9036 | 0.9090 | 0.8995 | 0.9017 | 0.8475 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.105 | 0.048 | 0.153 |
| CE | savari | taxi | 0.048 | 0.105 | 0.153 |
| CE | ambulance | kamyun | 0.000 | 0.105 | 0.105 |
| CE | autobus | minibus | 0.100 | 0.000 | 0.100 |
| CE | kamyunet | vanet | 0.095 | 0.000 | 0.095 |
| BCE | savari | taxi | 0.048 | 0.211 | 0.258 |
| BCE | kamyun | kamyunet | 0.105 | 0.048 | 0.153 |
| BCE | autobus | minibus | 0.100 | 0.053 | 0.153 |
| BCE | ambulance | vanet | 0.059 | 0.000 | 0.059 |
| BCE | ambulance | kamyun | 0.000 | 0.053 | 0.053 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.882 | 0.882 | 0.882 | 17 |
| autobus | 1.000 | 0.900 | 0.947 | 20 |
| kamyun | 0.938 | 0.789 | 0.857 | 19 |
| kamyunet | 0.857 | 0.857 | 0.857 | 21 |
| minibus | 0.750 | 0.947 | 0.837 | 19 |
| savari | 0.900 | 0.857 | 0.878 | 21 |
| taxi | 0.941 | 0.842 | 0.889 | 19 |
| vanet | 0.848 | 0.933 | 0.889 | 30 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.941 | 0.941 | 0.941 | 17 |
| autobus | 0.947 | 0.900 | 0.923 | 20 |
| kamyun | 0.941 | 0.842 | 0.889 | 19 |
| kamyunet | 0.900 | 0.857 | 0.878 | 21 |
| minibus | 0.900 | 0.947 | 0.923 | 19 |
| savari | 0.769 | 0.952 | 0.851 | 21 |
| taxi | 0.938 | 0.789 | 0.857 | 19 |
| vanet | 0.935 | 0.967 | 0.951 | 30 |
