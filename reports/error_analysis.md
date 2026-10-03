# Final Frozen-Test Evaluation and Error Analysis

## Protocol integrity

The final configuration was selected by maximum validation macro-F1 among Cross-Entropy production candidates. It selected **`resnet18_fine_tuning`**. The frozen test set was evaluated once, after this choice. Temperature and review threshold were fit/selected on validation only.

## Overall and per-class test metrics

- Accuracy: **0.8875**
- Macro precision: **0.8886**
- Macro recall: **0.8875**
- Macro F1: **0.8872**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.978 | 0.900 | 0.938 | 50 |
| autobus | 0.958 | 0.920 | 0.939 | 50 |
| kamyun | 0.778 | 0.700 | 0.737 | 50 |
| kamyunet | 0.755 | 0.800 | 0.777 | 50 |
| minibus | 0.855 | 0.940 | 0.895 | 50 |
| savari | 0.922 | 0.940 | 0.931 | 50 |
| taxi | 0.960 | 0.960 | 0.960 | 50 |
| vanet | 0.904 | 0.940 | 0.922 | 50 |

Both count-based and row-normalized confusion matrices are saved in `artifacts/results/frozen_test_evaluation.json` and plotted in `figures/final_confusion_matrices.png`.

## Confidence and human review

Validation temperature scaling changes ECE from the value saved in `validation_uncertainty.json`; the final validation-selected confidence threshold is **0.8085**, chosen as the lowest threshold reaching at least 80% automatic coverage. Thus approximately 80% of deployment volume is automatic and the remainder is human review.

On frozen test, calibrated multiclass NLL is **0.3420**, ECE is **0.0215**, and Brier score is **0.1625**. The selected threshold routes **73 / 400 (18.2%)** images to review; automatic accuracy is **0.9664**. Threshold behavior is descriptive on test; it is not re-tuned.

The risk–coverage curve is `figures/validation_risk_coverage.png`.

## Confusion pairs and merge decision

| Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|
| kamyun | kamyunet | 0.160 | 0.140 | 0.300 |
| kamyun | minibus | 0.060 | 0.040 | 0.100 |
| ambulance | vanet | 0.060 | 0.000 | 0.060 |
| autobus | kamyun | 0.020 | 0.040 | 0.060 |
| autobus | minibus | 0.060 | 0.000 | 0.060 |
| kamyunet | minibus | 0.040 | 0.020 | 0.060 |
| savari | vanet | 0.000 | 0.060 | 0.060 |
| ambulance | kamyunet | 0.040 | 0.000 | 0.040 |
| kamyunet | savari | 0.020 | 0.020 | 0.040 |
| savari | taxi | 0.040 | 0.000 | 0.040 |

`pair_confusion(i,j) = row_normalized_CM[i,j] + row_normalized_CM[j,i]`. **Do not merge for the production taxonomy. The strongest mutual pair is `kamyun`/`kamyunet` with score 0.300, above the predeclared 0.30 investigation threshold. The visual gallery should be checked for domain overlap, but a merge is not automatically warranted: confusion can reflect crop, pose, or scale rather than a redundant taxonomy.**

## Error inspection

`figures/misclassified_test_images.png` displays 12 lowest-confidence frozen-test errors with true label, predicted label, and confidence. Machine-readable rows, including top-3 classes and review flag, are in `artifacts/results/test_errors.csv` (45 total errors). Visual review distinguishes genuine class overlap from ambiguous crops, unusual viewpoints, tiny vehicles, and mislabeled/quality-impaired examples.
