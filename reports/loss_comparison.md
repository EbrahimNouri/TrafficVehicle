# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and 80-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

| Loss | Validation accuracy | Macro P | Macro R | Macro F1 | Mean top confidence |
|---|---|---|---|---|---|
| CrossEntropyLoss | 0.8434 | 0.8617 | 0.8399 | 0.8458 | 0.8914 |
| BCEWithLogitsLoss | 0.8614 | 0.8750 | 0.8538 | 0.8602 | 0.8017 |

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

| Loss | Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|---|
| CE | kamyun | kamyunet | 0.158 | 0.048 | 0.206 |
| CE | autobus | minibus | 0.150 | 0.053 | 0.203 |
| CE | savari | taxi | 0.048 | 0.105 | 0.153 |
| CE | ambulance | kamyunet | 0.118 | 0.000 | 0.118 |
| CE | ambulance | vanet | 0.118 | 0.000 | 0.118 |
| BCE | kamyun | kamyunet | 0.158 | 0.048 | 0.206 |
| BCE | savari | taxi | 0.048 | 0.158 | 0.206 |
| BCE | savari | vanet | 0.095 | 0.033 | 0.129 |
| BCE | ambulance | kamyunet | 0.118 | 0.000 | 0.118 |
| BCE | kamyunet | minibus | 0.000 | 0.105 | 0.105 |

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.765 | 0.867 | 17 |
| autobus | 0.941 | 0.800 | 0.865 | 20 |
| kamyun | 0.889 | 0.842 | 0.865 | 19 |
| kamyunet | 0.704 | 0.905 | 0.792 | 21 |
| minibus | 0.762 | 0.842 | 0.800 | 19 |
| savari | 0.818 | 0.857 | 0.837 | 21 |
| taxi | 0.941 | 0.842 | 0.889 | 19 |
| vanet | 0.839 | 0.867 | 0.852 | 30 |

### Per-class BCE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.824 | 0.903 | 17 |
| autobus | 0.941 | 0.800 | 0.865 | 20 |
| kamyun | 0.938 | 0.789 | 0.857 | 19 |
| kamyunet | 0.731 | 0.905 | 0.809 | 21 |
| minibus | 0.850 | 0.895 | 0.872 | 19 |
| savari | 0.773 | 0.810 | 0.791 | 21 |
| taxi | 0.889 | 0.842 | 0.865 | 19 |
| vanet | 0.879 | 0.967 | 0.921 | 30 |
