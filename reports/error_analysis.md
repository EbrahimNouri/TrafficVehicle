# Final Frozen-Test Evaluation and Error Analysis

## Protocol integrity

The final configuration was selected by maximum validation macro-F1 among Cross-Entropy production candidates. It selected **`resnet18_fine_tuning`**. The frozen test set was evaluated once, after this choice. Temperature and review threshold were fit/selected on validation only.

## Overall and per-class test metrics

- Accuracy: **0.9219**
- Macro precision: **0.9292**
- Macro recall: **0.9219**
- Macro F1: **0.9198**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.875 | 0.933 | 8 |
| autobus | 1.000 | 1.000 | 1.000 | 8 |
| kamyun | 1.000 | 0.875 | 0.933 | 8 |
| kamyunet | 1.000 | 1.000 | 1.000 | 8 |
| minibus | 0.800 | 1.000 | 0.889 | 8 |
| savari | 0.800 | 1.000 | 0.889 | 8 |
| taxi | 1.000 | 1.000 | 1.000 | 8 |
| vanet | 0.833 | 0.625 | 0.714 | 8 |

Both count-based and row-normalized confusion matrices are saved in `artifacts/results/frozen_test_evaluation.json` and plotted in `figures/final_confusion_matrices.png`.

## Confidence and human review

Validation temperature scaling changes ECE from the value saved in `validation_uncertainty.json`; the final validation-selected confidence threshold is **0.8187**, chosen as the lowest threshold reaching at least 80% automatic coverage. Thus approximately 80% of deployment volume is automatic and the remainder is human review.

On frozen test, calibrated multiclass NLL is **0.2830**, ECE is **0.0526**, and Brier score is **0.1450**. The selected threshold routes **14 / 64 (21.9%)** images to review; automatic accuracy is **0.9600**. Threshold behavior is descriptive on test; it is not re-tuned.

The risk–coverage curve is `figures/validation_risk_coverage.png`.

## Confusion pairs and merge decision

| Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|
| savari | vanet | 0.000 | 0.250 | 0.250 |
| ambulance | vanet | 0.125 | 0.000 | 0.125 |
| kamyun | minibus | 0.125 | 0.000 | 0.125 |
| minibus | vanet | 0.000 | 0.125 | 0.125 |
| ambulance | autobus | 0.000 | 0.000 | 0.000 |
| ambulance | kamyun | 0.000 | 0.000 | 0.000 |
| ambulance | kamyunet | 0.000 | 0.000 | 0.000 |
| ambulance | minibus | 0.000 | 0.000 | 0.000 |
| ambulance | savari | 0.000 | 0.000 | 0.000 |
| ambulance | taxi | 0.000 | 0.000 | 0.000 |

`pair_confusion(i,j) = row_normalized_CM[i,j] + row_normalized_CM[j,i]`. **Do not merge for the production taxonomy. The strongest mutual pair is `savari`/`vanet` with score 0.250, below the predeclared 0.30 investigation threshold. This does not support collapsing two operationally useful labels.**

## Error inspection

`figures/misclassified_test_images.png` displays 5 lowest-confidence frozen-test errors with true label, predicted label, and confidence. Machine-readable rows, including top-3 classes and review flag, are in `artifacts/results/test_errors.csv` (5 total errors). Visual review distinguishes genuine class overlap from ambiguous crops, unusual viewpoints, tiny vehicles, and mislabeled/quality-impaired examples.
