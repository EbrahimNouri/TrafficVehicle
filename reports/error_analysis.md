# Final Frozen-Test Evaluation and Error Analysis

## Protocol integrity

The final configuration was selected by maximum validation macro-F1 among Cross-Entropy production candidates. It selected **`resnet18_fine_tuning`**. The frozen test set was evaluated once, after this choice. Temperature and review threshold were fit/selected on validation only.

## Overall and per-class test metrics

- Accuracy: **0.8850**
- Macro precision: **0.8864**
- Macro recall: **0.8850**
- Macro F1: **0.8848**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.979 | 0.920 | 0.948 | 50 |
| autobus | 0.885 | 0.920 | 0.902 | 50 |
| kamyun | 0.745 | 0.700 | 0.722 | 50 |
| kamyunet | 0.769 | 0.800 | 0.784 | 50 |
| minibus | 0.922 | 0.940 | 0.931 | 50 |
| savari | 0.978 | 0.900 | 0.938 | 50 |
| taxi | 0.875 | 0.980 | 0.925 | 50 |
| vanet | 0.939 | 0.920 | 0.929 | 50 |

Both count-based and row-normalized confusion matrices are saved in `artifacts/results/frozen_test_evaluation.json` and plotted in `figures/final_confusion_matrices.png`.

## Confidence and human review

Validation temperature scaling changes ECE from the value saved in `validation_uncertainty.json`; the final validation-selected confidence threshold is **0.7685**, chosen as the lowest threshold reaching at least 80% automatic coverage. Thus approximately 80% of deployment volume is automatic and the remainder is human review.

On frozen test, calibrated multiclass NLL is **0.3990**, ECE is **0.0317**, and Brier score is **0.1822**. The selected threshold routes **88 / 400 (22.0%)** images to review; automatic accuracy is **0.9487**. Threshold behavior is descriptive on test; it is not re-tuned.

The risk–coverage curve is `figures/validation_risk_coverage.png`.

## Confusion pairs and merge decision

| Class A | Class B | A→B | B→A | Mutual score |
|---|---|---|---|---|
| kamyun | kamyunet | 0.180 | 0.160 | 0.340 |
| savari | taxi | 0.100 | 0.000 | 0.100 |
| autobus | kamyun | 0.040 | 0.040 | 0.080 |
| kamyun | minibus | 0.040 | 0.040 | 0.080 |
| ambulance | vanet | 0.040 | 0.000 | 0.040 |
| autobus | minibus | 0.040 | 0.000 | 0.040 |
| autobus | vanet | 0.000 | 0.040 | 0.040 |
| kamyunet | taxi | 0.020 | 0.020 | 0.040 |
| ambulance | autobus | 0.020 | 0.000 | 0.020 |
| ambulance | kamyun | 0.000 | 0.020 | 0.020 |

`pair_confusion(i,j) = row_normalized_CM[i,j] + row_normalized_CM[j,i]`. **Do not merge for the production taxonomy. The strongest mutual pair is `kamyun`/`kamyunet` with score 0.340, above the predeclared 0.30 investigation threshold. The visual gallery should be checked for domain overlap, but a merge is not automatically warranted: confusion can reflect crop, pose, or scale rather than a redundant taxonomy.**

## Error inspection

`figures/misclassified_test_images.png` displays 12 lowest-confidence frozen-test errors with true label, predicted label, and confidence. Machine-readable rows, including top-3 classes and review flag, are in `artifacts/results/test_errors.csv` (46 total errors). Visual review distinguishes genuine class overlap from ambiguous crops, unusual viewpoints, tiny vehicles, and mislabeled/quality-impaired examples.
