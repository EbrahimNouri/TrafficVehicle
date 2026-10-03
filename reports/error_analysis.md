# Final Frozen-Test Evaluation and Error Analysis

## Protocol integrity

The final configuration was selected by maximum validation macro-F1 among Cross-Entropy production candidates. It selected **`resnet18_fine_tuning`**. The frozen test set was evaluated once, after this choice. Temperature and review threshold were fit/selected on validation only.

## Overall and per-class test metrics

- Accuracy: **0.8950**
- Macro precision: **0.8977**
- Macro recall: **0.8950**
- Macro F1: **0.8939**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.978 | 0.900 | 0.938 | 50 |
| autobus | 0.940 | 0.940 | 0.940 | 50 |
| kamyun | 0.872 | 0.680 | 0.764 | 50 |
| kamyunet | 0.768 | 0.860 | 0.811 | 50 |
| minibus | 0.857 | 0.960 | 0.906 | 50 |
| savari | 0.904 | 0.940 | 0.922 | 50 |
| taxi | 0.959 | 0.940 | 0.949 | 50 |
| vanet | 0.904 | 0.940 | 0.922 | 50 |

Both count-based and row-normalized confusion matrices are saved in `artifacts/results/frozen_test_evaluation.json` and plotted in `figures/final_confusion_matrices.png`.

## Confidence and human review

Validation temperature scaling changes ECE from the value saved in `validation_uncertainty.json`; the final validation-selected confidence threshold is **0.8026**, chosen as the lowest threshold reaching at least 80% automatic coverage. Thus approximately 80% of deployment volume is automatic and the remainder is human review.

On frozen test, calibrated multiclass NLL is **0.3575**, ECE is **0.0413**, and Brier score is **0.1660**. The selected threshold routes **58 / 400 (14.5%)** images to review; automatic accuracy is **0.9386**. Threshold behavior is descriptive on test; it is not re-tuned.

The risk–coverage curve is `figures/validation_risk_coverage.png`.

## Confusion pairs and merge decision

| Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|
| kamyun | kamyunet | 0.160 | 0.080 | 0.240 |
| kamyun | minibus | 0.080 | 0.020 | 0.100 |
| ambulance | vanet | 0.060 | 0.000 | 0.060 |
| kamyunet | minibus | 0.040 | 0.020 | 0.060 |
| savari | taxi | 0.040 | 0.020 | 0.060 |
| savari | vanet | 0.000 | 0.060 | 0.060 |
| autobus | kamyun | 0.000 | 0.040 | 0.040 |
| autobus | minibus | 0.040 | 0.000 | 0.040 |
| kamyunet | savari | 0.020 | 0.020 | 0.040 |
| ambulance | autobus | 0.020 | 0.000 | 0.020 |

`pair_confusion(i,j) = row_normalized_CM[i,j] + row_normalized_CM[j,i]`. **Do not merge for the production taxonomy. The strongest mutual pair is `kamyun`/`kamyunet` with score 0.240, below the predeclared 0.30 investigation threshold. This does not support collapsing two operationally useful labels.**

## Error inspection

`figures/misclassified_test_images.png` displays 12 lowest-confidence frozen-test errors with true label, predicted label, and confidence. Machine-readable rows, including top-3 classes and review flag, are in `artifacts/results/test_errors.csv` (42 total errors). Visual review distinguishes genuine class overlap from ambiguous crops, unusual viewpoints, tiny vehicles, and mislabeled/quality-impaired examples.
